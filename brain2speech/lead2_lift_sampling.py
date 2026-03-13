#!/usr/bin/env python3
"""Try temperature sampling with LIFT to generate multiple candidates,
then pick the best one based on model log-probability."""
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
print(f"Loading data...")
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

# Decode subset
N_EVAL = 200
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

def compute_seq_logprob(model, tok, text):
    """Compute log-probability of text under the model."""
    inputs = tok(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()
        log_probs = shift_logits.log_softmax(dim=-1)
        token_lps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_lps.sum().item() / token_lps.shape[1]

def lift_generate(phones, n_samples=5, temperature=0.7, greedy_first=True):
    """Generate multiple candidates with sampling, plus one greedy."""
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

    return candidates

# Compare strategies
print("\nComparing greedy vs sampling LIFT...", flush=True)
results_greedy = []
results_best_of_n = []
results_majority = []
t0 = time.time()

for idx, d in enumerate(val_data):
    if not d['gt_text']:
        continue
    gt_words = normalize_text(d['gt_text'])
    if not gt_words:
        continue

    candidates = lift_generate(d['decoded_phones'], n_samples=5, temperature=0.7)

    # Greedy = first candidate
    greedy_words = normalize_text(candidates[0])
    wer_greedy = lev_distance(greedy_words, gt_words) / len(gt_words)
    results_greedy.append({'wer': wer_greedy, 'per': d['per']})

    # Best of N (oracle - pick closest to GT, upper bound)
    wers = []
    for c in candidates:
        c_words = normalize_text(c)
        w = lev_distance(c_words, gt_words) / len(gt_words)
        wers.append(w)
    best_idx = np.argmin(wers)
    results_best_of_n.append({'wer': wers[best_idx], 'per': d['per']})

    # Majority vote: pick most common candidate
    from collections import Counter
    normalized_cands = [' '.join(normalize_text(c)) for c in candidates]
    counter = Counter(normalized_cands)
    majority = counter.most_common(1)[0][0]
    majority_words = majority.split()
    wer_majority = lev_distance(majority_words, gt_words) / len(gt_words)
    results_majority.append({'wer': wer_majority, 'per': d['per']})

    n = len(results_greedy)
    if n % 50 == 0:
        elapsed = time.time() - t0
        wg = np.mean([r['wer'] for r in results_greedy])
        wb = np.mean([r['wer'] for r in results_best_of_n])
        wm = np.mean([r['wer'] for r in results_majority])
        print(f"  {n} | Greedy: {wg:.1%} | BestOfN: {wb:.1%} | Majority: {wm:.1%} | {elapsed:.0f}s", flush=True)

# Report
wg = np.mean([r['wer'] for r in results_greedy])
wb = np.mean([r['wer'] for r in results_best_of_n])
wm = np.mean([r['wer'] for r in results_majority])
mg = np.median([r['wer'] for r in results_greedy])
mb = np.median([r['wer'] for r in results_best_of_n])
mm = np.median([r['wer'] for r in results_majority])

print(f"\n{'='*60}")
print(f"SAMPLING LIFT COMPARISON ({len(results_greedy)} trials)")
print(f"{'='*60}")
print(f"Greedy:     Mean WER: {wg:.1%}, Median: {mg:.1%}")
print(f"Best-of-6:  Mean WER: {wb:.1%}, Median: {mb:.1%}  (oracle upper bound)")
print(f"Majority:   Mean WER: {wm:.1%}, Median: {mm:.1%}")
print(f"\nBest-of-N shows {(wg-wb)/wg:.0%} relative improvement potential from reranking")

save_data = {
    'greedy': {'mean_wer': float(wg), 'median_wer': float(mg)},
    'best_of_6': {'mean_wer': float(wb), 'median_wer': float(mb)},
    'majority': {'mean_wer': float(wm), 'median_wer': float(mm)},
    'n_trials': len(results_greedy),
}
save_path = str(RESULTS_DIR / 'L2_sampling_lift_comparison.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {save_path}")
