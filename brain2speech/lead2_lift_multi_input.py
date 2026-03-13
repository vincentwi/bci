#!/usr/bin/env python3
"""Feed multiple DCoND seeds' phonemes in one prompt to LIFT."""
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

# Load LIFT
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

PROMPT_SINGLE = "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:"

PROMPT_MULTI = """Convert these decoded brain-computer interface phonemes to English text. Multiple decoder models produced the following phoneme sequences for the same utterance. Use all of them to determine the correct English sentence.

Decoder 1: {p1}
Decoder 2: {p2}
Decoder 3: {p3}
Decoder 4: {p4}

English:"""

PROMPT_MULTI_V2 = """These phoneme sequences were decoded from the same brain signal by different models. Convert to the most likely English sentence.

{p1}
{p2}
{p3}
{p4}

English:"""

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def lift_gen(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(lift_model.device)
    with torch.no_grad():
        out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# Compare
print("\nComparing single vs multi-input LIFT...", flush=True)
results = {'single': [], 'multi_v1': [], 'multi_v2': []}
t0 = time.time()

seeds = list(ckpts.keys())
for idx in range(N_EVAL):
    gt = gt_texts[idx]
    if not gt:
        continue
    gt_words = normalize_text(gt)
    if not gt_words:
        continue
    
    # Single (s45)
    r1 = lift_gen(PROMPT_SINGLE.format(phones=all_seed_phones['s45'][idx]))
    w1 = lev_distance(normalize_text(r1), gt_words) / len(gt_words)
    results['single'].append(w1)
    
    # Multi v1
    p = {f'p{i+1}': all_seed_phones[s][idx] for i, s in enumerate(seeds)}
    r2 = lift_gen(PROMPT_MULTI.format(**p))
    w2 = lev_distance(normalize_text(r2), gt_words) / len(gt_words)
    results['multi_v1'].append(w2)
    
    # Multi v2
    r3 = lift_gen(PROMPT_MULTI_V2.format(**p))
    w3 = lev_distance(normalize_text(r3), gt_words) / len(gt_words)
    results['multi_v2'].append(w3)
    
    n = len(results['single'])
    if n % 50 == 0:
        elapsed = time.time() - t0
        for k, v in results.items():
            print(f"  [{k}] {n} | WER: {np.mean(v):.1%}", flush=True)
        print(f"  ({elapsed:.0f}s)", flush=True)

# Report
print(f"\n{'='*60}")
print(f"MULTI-INPUT LIFT ({len(results['single'])} trials)")
print(f"{'='*60}")
for k, v in results.items():
    print(f"{k:20s}: Mean WER: {np.mean(v):.1%}, Median: {np.median(v):.1%}")

save_data = {k: {'mean_wer': float(np.mean(v)), 'median_wer': float(np.median(v))} for k, v in results.items()}
save_path = str(RESULTS_DIR / 'L2_multi_input_lift.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {save_path}")
