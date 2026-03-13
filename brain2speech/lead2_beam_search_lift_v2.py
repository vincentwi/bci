#!/usr/bin/env python3
"""CTC beam search with KenLM + LIFT comparison."""
import json, sys, time, re, warnings
warnings.filterwarnings('ignore')
import numpy as np
import h5py
import torch
from Levenshtein import distance as lev_distance

sys.path.insert(0, 'brain2speech')
from config import CLASS_TO_ARPABET, N_CLASSES
from train_beyond_paper import compute_per, ctc_greedy_decode, DATA_DIR, RESULTS_DIR, CTC_BLANK
from lead2_train_dcond import PaperExactDCoND, build_marginalization_matrix_ext

device = torch.device('cuda:0')

# Build pyctcdecode decoder
from pyctcdecode import build_ctcdecoder

labels = [CLASS_TO_ARPABET[i] for i in range(N_CLASSES)] + ['']
kenlm_path = '/mnt/home/vincent.wilmet/brain2speech/data/phoneme_5gram.arpa'
print(f"Building decoder with KenLM...")

# Try different alpha/beta combos
decoder = build_ctcdecoder(labels=labels, kenlm_model_path=kenlm_path, alpha=2.0, beta=1.5)
print("Decoder ready")

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
print("\nDecoding val set with greedy + beam search...")
val_data = []
t0 = time.time()

for i in range(0, len(val_trials), 16):
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
    mono_probs = (probs @ M_ext).numpy()  # (B, T, 41)

    for j, t in enumerate(batch):
        T_raw = feat_lens[j]
        T_out = max(1, (T_raw - model.kernel_size) // model.stride + 1)
        T_out = min(T_out, mono_probs.shape[1])
        lp = mono_probs[j, :T_out]  # (T, 41) probs
        log_probs = np.log(lp + 1e-10)

        # Greedy
        greedy = ctc_greedy_decode(log_probs, blank=CTC_BLANK)
        target = t['phoneme_indices'].tolist()
        per_greedy = compute_per(greedy, target)
        greedy_phones = ' '.join([CLASS_TO_ARPABET[p] for p in greedy])

        # Beam search
        try:
            beam_text = decoder.decode(lp.astype(np.float32), beam_width=150)
            # beam_text is space-separated phonemes
            beam_phones_list = beam_text.strip().split() if beam_text.strip() else []
            beam_indices = []
            from config import ARPABET_TO_CLASS
            for p in beam_phones_list:
                if p in ARPABET_TO_CLASS:
                    beam_indices.append(ARPABET_TO_CLASS[p])
            per_beam = compute_per(beam_indices, target)
            beam_phones = beam_text.strip()
        except Exception as e:
            per_beam = per_greedy
            beam_phones = greedy_phones

        # Also get N-best beams
        try:
            beams = decoder.decode_beams(lp.astype(np.float32), beam_width=150)
            nbest = [b[0].strip() for b in beams[:5]]
        except:
            nbest = [greedy_phones]

        val_data.append({
            'greedy_phones': greedy_phones,
            'beam_phones': beam_phones,
            'nbest': nbest,
            'gt_text': t['text'],
            'per_greedy': per_greedy,
            'per_beam': per_beam,
        })

    if (i // 16) % 10 == 0:
        n = len(val_data)
        pg = np.mean([d['per_greedy'] for d in val_data])
        pb = np.mean([d['per_beam'] for d in val_data])
        print(f"  {n}/{len(val_trials)} | Greedy PER: {pg:.1%} | Beam PER: {pb:.1%} | {time.time()-t0:.0f}s", flush=True)

pg = np.mean([d['per_greedy'] for d in val_data])
pb = np.mean([d['per_beam'] for d in val_data])
print(f"\nDone! Greedy PER: {pg:.1%} | Beam PER: {pb:.1%}")

del model; torch.cuda.empty_cache()

# Load LIFT
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

PROMPT_NBEST = """Convert these decoded brain-computer interface phonemes to English text. Multiple candidate decodings are provided, ranked by likelihood. Use all candidates to determine the best English sentence.

{candidates}

English:"""

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

def lift_nbest(candidates):
    cand_block = "\n".join([f"Candidate {i+1}: {c}" for i, c in enumerate(candidates)])
    prompt = PROMPT_NBEST.format(candidates=cand_block)
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(lift_model.device)
    with torch.no_grad():
        out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# Compare strategies
print("\nComparing LIFT strategies...", flush=True)
results = {'greedy_lift': [], 'beam_lift': [], 'nbest_lift': []}
t0 = time.time()

for idx, d in enumerate(val_data):
    if not d['gt_text']:
        continue
    gt_words = normalize_text(d['gt_text'])
    if not gt_words:
        continue

    # Greedy + LIFT
    r1 = lift_single(d['greedy_phones'])
    w1 = lev_distance(normalize_text(r1), gt_words) / len(gt_words)
    results['greedy_lift'].append({'wer': w1, 'per': d['per_greedy'], 'response': r1, 'gt': d['gt_text']})

    # Beam top-1 + LIFT
    r2 = lift_single(d['beam_phones'])
    w2 = lev_distance(normalize_text(r2), gt_words) / len(gt_words)
    results['beam_lift'].append({'wer': w2, 'per': d['per_beam'], 'response': r2})

    # N-best + LIFT
    if len(d['nbest']) >= 2:
        r3 = lift_nbest(d['nbest'][:3])
    else:
        r3 = r2
    w3 = lev_distance(normalize_text(r3), gt_words) / len(gt_words)
    results['nbest_lift'].append({'wer': w3, 'per': d['per_greedy'], 'response': r3})

    n = len(results['greedy_lift'])
    if n % 100 == 0:
        elapsed = time.time() - t0
        for name, res in results.items():
            print(f"  [{name}] {n} | WER: {np.mean([r['wer'] for r in res]):.1%}", flush=True)
        print(f"  ({elapsed:.0f}s)", flush=True)

# Final report
print(f"\n{'='*70}")
print(f"BEAM SEARCH + LIFT COMPARISON (Val Set)")
print(f"{'='*70}")

for name, res in results.items():
    avg_wer = np.mean([r['wer'] for r in res])
    med_wer = np.median([r['wer'] for r in res])
    print(f"\n{name}: {len(res)} trials")
    print(f"  Mean WER:   {avg_wer:.1%}")
    print(f"  Median WER: {med_wer:.1%}")

# Examples where beam differs
print("\nExamples where beam_lift differs from greedy_lift:")
n_shown = 0
for i in range(len(results['greedy_lift'])):
    g = results['greedy_lift'][i]
    b = results['beam_lift'][i]
    if abs(g['wer'] - b['wer']) > 0.15 and n_shown < 10:
        print(f"  GT:     {g['gt']}")
        print(f"  Greedy: {g['response']} (WER={g['wer']:.0%})")
        print(f"  Beam:   {b['response']} (WER={b['wer']:.0%})")
        print()
        n_shown += 1

# Save
save_data = {}
for name, res in results.items():
    save_data[name] = {
        'mean_wer': float(np.mean([r['wer'] for r in res])),
        'median_wer': float(np.median([r['wer'] for r in res])),
        'n_trials': len(res),
    }
save_data['greedy_per'] = float(np.mean([d['per_greedy'] for d in val_data]))
save_data['beam_per'] = float(np.mean([d['per_beam'] for d in val_data]))

save_path = str(RESULTS_DIR / 'L2_beam_lift_comparison.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {save_path}")
