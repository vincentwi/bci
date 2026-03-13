#!/usr/bin/env python3
"""Custom CTC beam search with KenLM phoneme scoring + LIFT.

pyctcdecode doesn't work well with phoneme-level labels (no word boundaries).
Instead, use our own prefix beam search with KenLM scoring phoneme n-grams.
"""
import json, sys, time, re, warnings
warnings.filterwarnings('ignore')
import numpy as np
import h5py
import torch
from Levenshtein import distance as lev_distance
import kenlm

sys.path.insert(0, 'brain2speech')
from config import CLASS_TO_ARPABET, N_CLASSES, ARPABET_TO_CLASS
from train_beyond_paper import compute_per, ctc_greedy_decode, DATA_DIR, RESULTS_DIR, CTC_BLANK
from lead2_train_dcond import PaperExactDCoND, build_marginalization_matrix_ext

device = torch.device('cuda:0')

# Load KenLM
kenlm_path = '/mnt/home/vincent.wilmet/brain2speech/data/phoneme_5gram.arpa'
print(f"Loading KenLM: {kenlm_path}")
lm = kenlm.Model(kenlm_path)
print(f"  Order: {lm.order}")

def ctc_beam_search_kenlm(log_probs, blank=CTC_BLANK, beam_width=50, lm_weight=1.5):
    """CTC prefix beam search with KenLM phoneme scoring.

    log_probs: (T, C) numpy array
    Returns: list of (phoneme_indices, score) sorted by score desc
    """
    T, C = log_probs.shape
    # Each beam: (prefix_tuple, score)
    # Score = log_ctc + lm_weight * log_lm
    beams = {(): 0.0}  # empty prefix -> score 0

    for t in range(T):
        new_beams = {}
        lp = log_probs[t]  # (C,)

        for prefix, score in beams.items():
            # Option 1: emit blank (extend beam without changing prefix)
            blank_score = score + lp[blank]
            if prefix not in new_beams or new_beams[prefix] < blank_score:
                new_beams[prefix] = blank_score

            # Option 2: emit each non-blank class
            top_k = np.argsort(lp)[-20:]  # only consider top-20 classes
            for c in top_k:
                if c == blank:
                    continue
                # CTC: if same as last, it's a repeat (collapsed)
                if prefix and prefix[-1] == c:
                    new_prefix = prefix  # repeat same char = no change
                else:
                    new_prefix = prefix + (c,)

                # CTC score
                new_score = score + lp[c]

                # Add LM score for new phone
                if new_prefix != prefix:
                    phone_str = ' '.join([CLASS_TO_ARPABET[p] for p in new_prefix])
                    # Score just the last n-gram
                    lm_score = lm.score(phone_str, bos=True, eos=False)
                    if len(prefix) > 0:
                        old_phone_str = ' '.join([CLASS_TO_ARPABET[p] for p in prefix])
                        old_lm_score = lm.score(old_phone_str, bos=True, eos=False)
                        lm_delta = lm_score - old_lm_score
                    else:
                        lm_delta = lm_score
                    new_score += lm_weight * lm_delta

                if new_prefix not in new_beams or new_beams[new_prefix] < new_score:
                    new_beams[new_prefix] = new_score

        # Prune to beam_width
        sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
        beams = dict(sorted_beams[:beam_width])

    # Add EOS LM score
    final = []
    for prefix, score in beams.items():
        phone_str = ' '.join([CLASS_TO_ARPABET[p] for p in prefix])
        eos_score = lm.score(phone_str, bos=True, eos=True)
        bos_score = lm.score(phone_str, bos=True, eos=False)
        eos_bonus = eos_score - bos_score
        final.append((list(prefix), score + lm_weight * eos_bonus))

    final.sort(key=lambda x: x[1], reverse=True)
    return final

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
print(f"Val: {len(val_trials)} trials")

# Load DCoND
ckpt_path = str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s45_best.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model = PaperExactDCoND(**ckpt['model_kwargs'])
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state)
model.to(device).eval()
model.set_train_sessions(train_sids)
M_ext = build_marginalization_matrix_ext()

# Decode with both greedy and beam
print("\nDecoding with greedy + KenLM beam search...")
val_data = []
t0 = time.time()

# Use subset for faster experimentation
N_TRIALS = min(200, len(val_trials))

for i in range(0, N_TRIALS, 16):
    batch = val_trials[i:i+16]
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
        lp = mono_lp[j, :T_out]

        # Greedy
        greedy = ctc_greedy_decode(lp, blank=CTC_BLANK)
        target = t['phoneme_indices'].tolist()
        per_greedy = compute_per(greedy, target)

        # Beam search with KenLM
        beam_results = ctc_beam_search_kenlm(lp, beam_width=30, lm_weight=1.5)
        if beam_results:
            beam_top = beam_results[0][0]
            per_beam = compute_per(beam_top, target)
            # Get N-best
            nbest_phones = [' '.join([CLASS_TO_ARPABET[p] for p in b[0]]) for b in beam_results[:5]]
        else:
            beam_top = greedy
            per_beam = per_greedy
            nbest_phones = [' '.join([CLASS_TO_ARPABET[p] for p in greedy])]

        greedy_phones = ' '.join([CLASS_TO_ARPABET[p] for p in greedy])
        beam_phones = ' '.join([CLASS_TO_ARPABET[p] for p in beam_top])

        val_data.append({
            'greedy_phones': greedy_phones,
            'beam_phones': beam_phones,
            'nbest': nbest_phones,
            'gt_text': t['text'],
            'per_greedy': per_greedy,
            'per_beam': per_beam,
        })

    n = len(val_data)
    pg = np.mean([d['per_greedy'] for d in val_data])
    pb = np.mean([d['per_beam'] for d in val_data])
    elapsed = time.time() - t0
    print(f"  {n}/{N_TRIALS} | Greedy PER: {pg:.1%} | Beam PER: {pb:.1%} | {elapsed:.0f}s", flush=True)

pg = np.mean([d['per_greedy'] for d in val_data])
pb = np.mean([d['per_beam'] for d in val_data])
print(f"\nGreedy PER: {pg:.1%} | KenLM Beam PER: {pb:.1%}")

del model; torch.cuda.empty_cache()

# If beam helps PER, run LIFT on both
print("\nLoading LIFT...")
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

print("\nRunning LIFT on greedy + beam...", flush=True)
results_greedy = []
results_beam = []
t0 = time.time()

for idx, d in enumerate(val_data):
    if not d['gt_text']:
        continue
    gt_words = normalize_text(d['gt_text'])
    if not gt_words:
        continue

    r1 = lift_single(d['greedy_phones'])
    w1 = lev_distance(normalize_text(r1), gt_words) / len(gt_words)
    results_greedy.append({'wer': w1, 'per': d['per_greedy'], 'lift': r1, 'gt': d['gt_text']})

    r2 = lift_single(d['beam_phones'])
    w2 = lev_distance(normalize_text(r2), gt_words) / len(gt_words)
    results_beam.append({'wer': w2, 'per': d['per_beam'], 'lift': r2, 'gt': d['gt_text']})

    n = len(results_greedy)
    if n % 50 == 0:
        elapsed = time.time() - t0
        wg = np.mean([r['wer'] for r in results_greedy])
        wb = np.mean([r['wer'] for r in results_beam])
        print(f"  {n} | Greedy+LIFT WER: {wg:.1%} | Beam+LIFT WER: {wb:.1%} | {elapsed:.0f}s", flush=True)

wg = np.mean([r['wer'] for r in results_greedy])
wb = np.mean([r['wer'] for r in results_beam])
mg = np.median([r['wer'] for r in results_greedy])
mb = np.median([r['wer'] for r in results_beam])

print(f"\n{'='*60}")
print(f"KENLM BEAM + LIFT COMPARISON ({len(results_greedy)} trials)")
print(f"{'='*60}")
print(f"Greedy + LIFT:     Mean WER: {wg:.1%}, Median: {mg:.1%}")
print(f"KenLM Beam + LIFT: Mean WER: {wb:.1%}, Median: {mb:.1%}")
print(f"\nGreedy PER: {pg:.1%} | Beam PER: {pb:.1%}")

# Examples
print("\nExamples where beam improves:")
n_shown = 0
for i in range(len(results_greedy)):
    g, b = results_greedy[i], results_beam[i]
    if b['wer'] < g['wer'] - 0.1 and n_shown < 10:
        print(f"  GT:     {g['gt']}")
        print(f"  Greedy: {g['lift']} (WER={g['wer']:.0%})")
        print(f"  Beam:   {b['lift']} (WER={b['wer']:.0%})")
        print()
        n_shown += 1

print("\nExamples where beam hurts:")
n_shown = 0
for i in range(len(results_greedy)):
    g, b = results_greedy[i], results_beam[i]
    if b['wer'] > g['wer'] + 0.1 and n_shown < 5:
        print(f"  GT:     {g['gt']}")
        print(f"  Greedy: {g['lift']} (WER={g['wer']:.0%})")
        print(f"  Beam:   {b['lift']} (WER={b['wer']:.0%})")
        print()
        n_shown += 1

save_data = {
    'greedy_lift': {'mean_wer': float(wg), 'median_wer': float(mg)},
    'beam_lift': {'mean_wer': float(wb), 'median_wer': float(mb)},
    'greedy_per': float(pg), 'beam_per': float(pb),
    'n_trials': len(results_greedy),
    'beam_width': 30, 'lm_weight': 1.5,
}
save_path = str(RESULTS_DIR / 'L2_kenlm_beam_lift_comparison.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {save_path}")
