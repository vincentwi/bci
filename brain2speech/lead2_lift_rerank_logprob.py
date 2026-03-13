#!/usr/bin/env python3
"""LIFT reranking using model log-probability to select best candidate."""
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

# Decode with best model
N_EVAL = 300
print(f"Decoding {N_EVAL} val trials...")
ckpt_path = str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s45_best.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model = PaperExactDCoND(**ckpt['model_kwargs'])
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state)
model.to(device).eval()
model.set_train_sessions(train_sids)
M_ext = build_marginalization_matrix_ext()

val_data = []
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
        target = t['phoneme_indices'].tolist()
        val_data.append({
            'decoded_phones': ' '.join([CLASS_TO_ARPABET[p] for p in decoded]),
            'gt_text': t['text'],
            'per': compute_per(decoded, target),
        })

del model; torch.cuda.empty_cache()

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

PROMPT = "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:"

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def compute_response_logprob(prompt_text, response_text):
    """Compute log-probability of response given prompt."""
    full_text = prompt_text + response_text
    full_ids = tok(full_text, return_tensors='pt').input_ids.to(lift_model.device)
    prompt_ids = tok(prompt_text, return_tensors='pt').input_ids
    prompt_len = prompt_ids.shape[1]
    
    with torch.no_grad():
        outputs = lift_model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab)
    
    # Only score response tokens
    shift_logits = logits[:, prompt_len-1:-1, :]  # logits predicting response tokens
    shift_labels = full_ids[:, prompt_len:]  # actual response tokens
    
    log_probs = shift_logits.float().log_softmax(dim=-1)
    token_lps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Return mean log-prob per token (length-normalized)
    return token_lps.mean().item()

def lift_generate_with_rerank(phones, n_samples=8, temperature=0.7):
    """Generate multiple candidates and rerank by model log-probability."""
    prompt = PROMPT.format(phones=phones)
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(lift_model.device)
    input_len = inputs.input_ids.shape[1]
    
    candidates = []
    
    # Greedy decode
    with torch.no_grad():
        out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
    greedy = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
    candidates.append(greedy)
    
    # Sample with temperature
    for _ in range(n_samples):
        with torch.no_grad():
            out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=True,
                                       temperature=temperature, top_p=0.9,
                                       pad_token_id=tok.eos_token_id)
        sampled = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        if sampled not in candidates:
            candidates.append(sampled)
    
    # Also try lower temperature for more focused samples
    for _ in range(3):
        with torch.no_grad():
            out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=True,
                                       temperature=0.3, top_p=0.95,
                                       pad_token_id=tok.eos_token_id)
        sampled = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        if sampled not in candidates:
            candidates.append(sampled)
    
    # Score each candidate by log-probability
    scored = []
    for c in candidates:
        lp = compute_response_logprob(text, c)
        scored.append((c, lp))
    
    # Sort by log-prob (higher = better)
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'greedy': greedy,
        'best_logprob': scored[0][0],
        'all_candidates': [(c, float(lp)) for c, lp in scored],
    }

# Run comparison
print("\nComparing greedy vs log-prob reranking...", flush=True)
results_greedy = []
results_reranked = []
results_oracle = []
t0 = time.time()

for idx, d in enumerate(val_data):
    if not d['gt_text']:
        continue
    gt_words = normalize_text(d['gt_text'])
    if not gt_words:
        continue

    result = lift_generate_with_rerank(d['decoded_phones'], n_samples=8, temperature=0.7)
    
    # Greedy WER
    greedy_words = normalize_text(result['greedy'])
    wer_greedy = lev_distance(greedy_words, gt_words) / len(gt_words)
    results_greedy.append({'wer': wer_greedy, 'per': d['per']})
    
    # Log-prob reranked WER
    reranked_words = normalize_text(result['best_logprob'])
    wer_reranked = lev_distance(reranked_words, gt_words) / len(gt_words)
    results_reranked.append({'wer': wer_reranked, 'per': d['per']})
    
    # Oracle (best of all)
    best_wer = float('inf')
    for c, _ in result['all_candidates']:
        c_words = normalize_text(c)
        w = lev_distance(c_words, gt_words) / len(gt_words)
        if w < best_wer:
            best_wer = w
    results_oracle.append({'wer': best_wer, 'per': d['per']})
    
    n = len(results_greedy)
    if n % 50 == 0:
        elapsed = time.time() - t0
        wg = np.mean([r['wer'] for r in results_greedy])
        wr = np.mean([r['wer'] for r in results_reranked])
        wo = np.mean([r['wer'] for r in results_oracle])
        print(f"  {n} | Greedy: {wg:.1%} | LogProb: {wr:.1%} | Oracle: {wo:.1%} | {elapsed:.0f}s", flush=True)

# Final report
wg = np.mean([r['wer'] for r in results_greedy])
wr = np.mean([r['wer'] for r in results_reranked])
wo = np.mean([r['wer'] for r in results_oracle])
mg = np.median([r['wer'] for r in results_greedy])
mr = np.median([r['wer'] for r in results_reranked])
mo = np.median([r['wer'] for r in results_oracle])

print(f"\n{'='*60}")
print(f"LOG-PROB RERANKING LIFT ({len(results_greedy)} trials)")
print(f"{'='*60}")
print(f"Greedy:     Mean WER: {wg:.1%}, Median: {mg:.1%}")
print(f"LogProb:    Mean WER: {wr:.1%}, Median: {mr:.1%}")
print(f"Oracle:     Mean WER: {wo:.1%}, Median: {mo:.1%}")
print(f"\nLogProb reranking: {(wg-wr)/wg:.0%} relative improvement")
print(f"Oracle upper bound: {(wg-wo)/wg:.0%} relative improvement potential")

save_data = {
    'greedy': {'mean_wer': float(wg), 'median_wer': float(mg)},
    'logprob_reranked': {'mean_wer': float(wr), 'median_wer': float(mr)},
    'oracle': {'mean_wer': float(wo), 'median_wer': float(mo)},
    'n_trials': len(results_greedy),
    'n_samples': 8,
}
save_path = str(RESULTS_DIR / 'L2_lift_logprob_rerank.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {save_path}")
