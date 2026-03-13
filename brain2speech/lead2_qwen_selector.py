#!/usr/bin/env python3
"""Use base Qwen to select best LIFT candidate from multi-seed outputs."""
import json, sys, time, re
import numpy as np
import h5py
import torch
from Levenshtein import distance as lev_distance

sys.path.insert(0, 'brain2speech')
from config import CLASS_TO_ARPABET, N_CLASSES
from train_beyond_paper import compute_per, ctc_greedy_decode, DATA_DIR, RESULTS_DIR, CTC_BLANK
from lead2_train_dcond import PaperExactDCoND, build_marginalization_matrix_ext

device = torch.device('cuda:0')

# Load HDF5
h5_path = DATA_DIR / 'sentences_paper_256d.h5'
print("Loading data...")
trials = []
with h5py.File(h5_path, 'r') as f:
    n_trials = f.attrs['n_trials']
    for i in range(n_trials):
        grp = f[f'trial_{i:05d}']
        trials.append({
            'features_raw': grp['features'][:],
            'phoneme_indices': grp['phoneme_indices'][:],
            'session': grp.attrs['session'],
            'text': grp.attrs.get('text', ''),
        })

session_names = sorted(set(t['session'] for t in trials))
session_to_idx = {s: i for i, s in enumerate(session_names)}
for t in trials:
    t['session_idx'] = session_to_idx[t['session']]
sessions = {}
for t in trials:
    sessions.setdefault(t['session'], []).append(t)

val_sessions = session_names[-6:-4]
train_sessions = session_names[:-6]
train_sids = {session_to_idx[s] for s in train_sessions}
val_trials = [t for s in val_sessions for t in sessions[s]]

M_ext = build_marginalization_matrix_ext()

# Decode with all 4 seeds
ckpts = {
    's42': 'L2_L2_pe_dcond_adam_s42_long_best.pt',
    's43': 'L2_L2_pe_dcond_adam_s43_best.pt',
    's44': 'L2_L2_pe_dcond_adam_s44_best.pt',
    's45': 'L2_L2_pe_dcond_adam_s45_best.pt',
}

N_EVAL = 200
all_seed_phones = {seed: [] for seed in ckpts}

for seed, name in ckpts.items():
    path = str(RESULTS_DIR / name)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = PaperExactDCoND(**ckpt['model_kwargs'])
    state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    model.set_train_sessions(train_sids)
    
    for i in range(0, N_EVAL, 32):
        batch = val_trials[i:i+32]
        features = [torch.FloatTensor(t['features_raw']) for t in batch]
        feat_lens = [f.shape[0] for f in features]
        max_T = max(feat_lens)
        C = features[0].shape[1]
        padded = torch.zeros(len(batch), max_T, C).to(device)
        for j, f in enumerate(features):
            padded[j, :f.shape[0], :] = f
        session_ids = torch.LongTensor([t.get('session_idx', 0) for t in batch]).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits = model(padded, session_ids)
        probs = logits.float().softmax(dim=-1).cpu()
        mono_probs = probs @ M_ext
        mono_lp = (mono_probs + 1e-10).log().numpy()
        for j, t in enumerate(batch):
            T_raw = feat_lens[j]
            T_out = max(1, (T_raw - model.kernel_size) // model.stride + 1)
            T_out = min(T_out, mono_lp.shape[1])
            decoded = ctc_greedy_decode(mono_lp[j, :T_out], blank=CTC_BLANK)
            all_seed_phones[seed].append(' '.join([CLASS_TO_ARPABET[p] for p in decoded]))
    
    del model; torch.cuda.empty_cache()
    print(f"  Decoded {seed}")

gt_texts = [val_trials[i]['text'] for i in range(N_EVAL)]

# Load LIFT model
print("Loading LIFT...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_path = '/mnt/home/vincent.wilmet/brain2speech/models/phoneme_lift_qwen/final'
tok = AutoTokenizer.from_pretrained(adapter_path)
lift_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-2B', torch_dtype=torch.bfloat16, device_map='cuda:0')
lift_model = PeftModel.from_pretrained(lift_model, adapter_path)
lift_model.eval()
print("LIFT loaded", flush=True)

PROMPT = "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:"

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def lift_single(phones):
    prompt = PROMPT.format(phones=phones)
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(lift_model.device)
    with torch.no_grad():
        out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def compute_text_perplexity(text):
    """Compute perplexity of text under the BASE model (without LIFT adapter).
    Lower = more fluent English."""
    # Disable adapter for scoring
    lift_model.disable_adapter_layers()
    inputs = tok(text, return_tensors='pt').to(lift_model.device)
    with torch.no_grad():
        outputs = lift_model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = inputs.input_ids[:, 1:]
        log_probs = shift_logits.log_softmax(dim=-1)
        token_lps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        avg_lp = token_lps.mean().item()
    lift_model.enable_adapter_layers()
    return -avg_lp  # Return negative = perplexity (lower is better)

# Run comparison
print("\nComparing selection strategies...", flush=True)
results = {'single_s45': [], 'majority': [], 'base_perplexity': [], 'oracle': []}
t0 = time.time()

for idx in range(N_EVAL):
    gt = gt_texts[idx]
    if not gt:
        continue
    gt_words = normalize_text(gt)
    if not gt_words:
        continue
    
    # Run LIFT on each seed
    candidates = {}
    for seed in ckpts:
        resp = lift_single(all_seed_phones[seed][idx])
        candidates[seed] = resp
    
    # Single s45
    w45 = lev_distance(normalize_text(candidates['s45']), gt_words) / len(gt_words)
    results['single_s45'].append(w45)
    
    # Oracle
    wers = {seed: lev_distance(normalize_text(r), gt_words) / len(gt_words) 
            for seed, r in candidates.items()}
    results['oracle'].append(min(wers.values()))
    
    # Majority vote
    from collections import Counter
    normalized = [' '.join(normalize_text(r)) for r in candidates.values()]
    counter = Counter(normalized)
    majority = counter.most_common(1)[0][0]
    wmaj = lev_distance(majority.split(), gt_words) / len(gt_words)
    results['majority'].append(wmaj)
    
    # Base model perplexity selection
    unique_candidates = list(set(candidates.values()))
    if len(unique_candidates) > 1:
        perplexities = []
        for c in unique_candidates:
            ppl = compute_text_perplexity(c)
            perplexities.append((c, ppl))
        # Select lowest perplexity (most fluent)
        best = min(perplexities, key=lambda x: x[1])
        best_text = best[0]
    else:
        best_text = unique_candidates[0]
    
    wbp = lev_distance(normalize_text(best_text), gt_words) / len(gt_words)
    results['base_perplexity'].append(wbp)
    
    n = len(results['single_s45'])
    if n % 50 == 0:
        elapsed = time.time() - t0
        for k, v in results.items():
            print(f"  [{k}] {n} | WER: {np.mean(v):.1%}", flush=True)
        print(f"  ({elapsed:.0f}s)", flush=True)

# Report
print(f"\n{'='*60}")
print(f"CANDIDATE SELECTION ({len(results['single_s45'])} trials)")
print(f"{'='*60}")
for k, v in results.items():
    print(f"{k:20s}: Mean WER: {np.mean(v):.1%}, Median: {np.median(v):.1%}")

save_data = {k: {'mean_wer': float(np.mean(v)), 'median_wer': float(np.median(v))} for k, v in results.items()}
save_path = str(RESULTS_DIR / 'L2_qwen_selector.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {save_path}")
