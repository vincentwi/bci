#!/usr/bin/env python3
"""Evaluate LIFT v2 on test+val sets and compare with v1."""
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

test_sessions = session_names[-4:]
val_sessions = session_names[-6:-4]
train_sessions = session_names[:-6]
train_sids = {session_to_idx[s] for s in train_sessions}
test_trials = [t for s in test_sessions for t in sessions[s]]
val_trials = [t for s in val_sessions for t in sessions[s]]

# Decode
ckpt_path = str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s45_best.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model = PaperExactDCoND(**ckpt['model_kwargs'])
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state)
model.to(device).eval()
model.set_train_sessions(train_sids)
M_ext = build_marginalization_matrix_ext()

def decode_trials(trial_list):
    data = []
    for i in range(0, len(trial_list), 32):
        batch = trial_list[i:i+32]
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
            data.append({
                'decoded_phones': ' '.join([CLASS_TO_ARPABET[p] for p in decoded]),
                'gt_text': t['text'],
                'per': compute_per(decoded, target),
            })
    return data

print("Decoding test...")
test_data = decode_trials(test_trials)
print(f"  PER: {np.mean([d['per'] for d in test_data]):.1%}")
print("Decoding val...")
val_data = decode_trials(val_trials)
print(f"  PER: {np.mean([d['per'] for d in val_data]):.1%}")

del model; torch.cuda.empty_cache()

# Try loading v2 model
import os
v2_path = '/mnt/home/vincent.wilmet/brain2speech/models/phoneme_lift_qwen_v2'
# Check if checkpoints exist
if os.path.exists(os.path.join(v2_path, 'final')):
    adapter_path = os.path.join(v2_path, 'final')
elif os.path.exists(v2_path):
    # Find latest checkpoint
    ckpts = [d for d in os.listdir(v2_path) if d.startswith('checkpoint-')]
    if ckpts:
        latest = sorted(ckpts, key=lambda x: int(x.split('-')[1]))[-1]
        adapter_path = os.path.join(v2_path, latest)
        print(f"Using latest checkpoint: {adapter_path}")
    else:
        print("No checkpoints found yet!")
        sys.exit(1)
else:
    print(f"v2 path not found: {v2_path}")
    sys.exit(1)

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

# Evaluate both v1 and v2
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROMPT = "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:"

for version, apath in [('v1', '/mnt/home/vincent.wilmet/brain2speech/models/phoneme_lift_qwen/final'),
                        ('v2', adapter_path)]:
    print(f"\n{'='*60}")
    print(f"Loading LIFT {version} from {apath}...")
    tok = AutoTokenizer.from_pretrained(apath)
    lift_model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3.5-2B', torch_dtype=torch.bfloat16, device_map='cuda:0')
    lift_model = PeftModel.from_pretrained(lift_model, apath)
    lift_model.eval()
    print(f"LIFT {version} loaded", flush=True)
    
    for split_name, data_list in [('TEST', test_data), ('VAL', val_data)]:
        results = []
        t0 = time.time()
        for i, v in enumerate(data_list):
            if not v['gt_text']:
                continue
            prompt = PROMPT.format(phones=v['decoded_phones'])
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors='pt').to(lift_model.device)
            with torch.no_grad():
                out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
            response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            gt_words = normalize_text(v['gt_text'])
            lift_words = normalize_text(response)
            wer = lev_distance(lift_words, gt_words) / len(gt_words) if gt_words else 0.0
            results.append({'gt': v['gt_text'], 'lift': response, 'wer': wer, 'per': v['per']})
            if len(results) % 300 == 0:
                elapsed = time.time() - t0
                avg = np.mean([r['wer'] for r in results])
                print(f"  [{split_name}] {len(results)} | WER: {avg:.1%} | {elapsed:.0f}s", flush=True)
        
        avg_wer = np.mean([r['wer'] for r in results])
        med_wer = np.median([r['wer'] for r in results])
        avg_per = np.mean([r['per'] for r in results])
        print(f"\n  LIFT {version} {split_name}: {len(results)} trials | PER: {avg_per:.1%}")
        print(f"  Mean WER:   {avg_wer:.4f} ({avg_wer:.1%})")
        print(f"  Median WER: {med_wer:.4f} ({med_wer:.1%})")
        
        # PER-bucketed WER
        for per_lo, per_hi in [(0, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 1.0)]:
            bucket = [r for r in results if per_lo <= r['per'] < per_hi]
            if bucket:
                print(f"    PER [{per_lo:.0%}-{per_hi:.0%}): {len(bucket)} trials, WER={np.mean([r['wer'] for r in bucket]):.1%}")
    
    del lift_model, tok
    torch.cuda.empty_cache()

print("\nDone!")
