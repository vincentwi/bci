#!/usr/bin/env python3
"""Evaluate the DCoND-LIFT corrector on val set."""
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

test_sessions = session_names[-4:]
val_sessions = session_names[-6:-4]
train_sessions = session_names[:-6]
train_sids = {session_to_idx[s] for s in train_sessions}
test_trials = [t for s in test_sessions for t in sessions[s]]
val_trials = [t for s in val_sessions for t in sessions[s]]
M_ext = build_marginalization_matrix_ext()

# Decode with seed45
print("Decoding with seed45...")
ckpt_path = str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s45_best.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model = PaperExactDCoND(**ckpt['model_kwargs'])
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state)
model.to(device).eval()
model.set_train_sessions(train_sids)

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

val_data = decode_trials(val_trials)
test_data = decode_trials(test_trials)
del model; torch.cuda.empty_cache()

# Load LIFT v1 
print("Loading LIFT v1...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_path = '/mnt/home/vincent.wilmet/brain2speech/models/phoneme_lift_qwen/final'
tok = AutoTokenizer.from_pretrained(adapter_path)
lift_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-2B', torch_dtype=torch.bfloat16, device_map='cuda:0')
lift_model = PeftModel.from_pretrained(lift_model, adapter_path)
lift_model.eval()

LIFT_PROMPT = "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:"

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

# Get LIFT translations for val + test
print("Getting LIFT v1 translations...")
for split_name, data_list in [('val', val_data), ('test', test_data)]:
    for d in data_list:
        if not d['gt_text']:
            d['lift_text'] = ''
            continue
        prompt = LIFT_PROMPT.format(phones=d['decoded_phones'])
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(lift_model.device)
        with torch.no_grad():
            out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
        d['lift_text'] = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"  {split_name}: {len(data_list)} LIFT translations done")

del lift_model; torch.cuda.empty_cache()

# Load corrector
print("\nLoading DCoND-LIFT corrector...")
corrector_path = '/mnt/home/vincent.wilmet/brain2speech/models/dcond_lift_corrector/final'
corrector_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-2B', torch_dtype=torch.bfloat16, device_map='cuda:0')
corrector_model = PeftModel.from_pretrained(corrector_model, corrector_path)
corrector_model.eval()
tok2 = AutoTokenizer.from_pretrained(corrector_path)
print("Corrector loaded")

CORRECTOR_PROMPT = """Correct this brain-computer interface transcription using both the decoded text and raw phonemes.

Decoded text: {text}
Phonemes: {phones}

Corrected text:"""

# Evaluate on val and test
for split_name, data_list in [('VAL', val_data), ('TEST', test_data)]:
    results_lift = []
    results_corrected = []
    t0 = time.time()
    
    for d in data_list:
        if not d['gt_text'] or not d['lift_text']:
            continue
        gt_words = normalize_text(d['gt_text'])
        if not gt_words:
            continue
        
        # LIFT v1
        lift_words = normalize_text(d['lift_text'])
        wer_lift = lev_distance(lift_words, gt_words) / len(gt_words)
        results_lift.append(wer_lift)
        
        # Corrector
        prompt = CORRECTOR_PROMPT.format(text=d['lift_text'], phones=d['decoded_phones'])
        messages = [{"role": "user", "content": prompt}]
        text = tok2.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        inputs = tok2(text, return_tensors='pt').to(corrector_model.device)
        with torch.no_grad():
            out = corrector_model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok2.eos_token_id)
        corrected = tok2.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        corrected_words = normalize_text(corrected)
        wer_corrected = lev_distance(corrected_words, gt_words) / len(gt_words)
        results_corrected.append(wer_corrected)
    
    elapsed = time.time() - t0
    print(f"\n{split_name}: {len(results_lift)} trials ({elapsed:.0f}s)")
    print(f"  LIFT v1:    Mean WER: {np.mean(results_lift):.1%}, Median: {np.median(results_lift):.1%}")
    print(f"  Corrected:  Mean WER: {np.mean(results_corrected):.1%}, Median: {np.median(results_corrected):.1%}")
    
    # PER-bucketed
    for per_lo, per_hi in [(0, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 1.0)]:
        idx_bucket = [i for i, d in enumerate(data_list) if d['gt_text'] and per_lo <= d['per'] < per_hi]
        if idx_bucket:
            bucket_lift = [results_lift[i] for i in range(len(results_lift)) if i < len(idx_bucket)]
            bucket_corr = [results_corrected[i] for i in range(len(results_corrected)) if i < len(idx_bucket)]

save_data = {
    'val_lift': float(np.mean([w for w in results_lift])) if split_name == 'VAL' else None,
    'val_corrected': float(np.mean([w for w in results_corrected])) if split_name == 'VAL' else None,
}
save_path = str(RESULTS_DIR / 'L2_dcond_lift_corrector_eval.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {save_path}")
