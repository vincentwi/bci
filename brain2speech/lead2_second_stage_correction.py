#!/usr/bin/env python3
"""Second-stage LLM correction: use a larger model to clean up LIFT output.
The DCoND paper's key insight: feed BOTH text AND phonemes to the correction model."""
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

# Decode with seed45
N_EVAL = 200
print(f"Decoding {N_EVAL} val trials...")
ckpt_path = str(RESULTS_DIR / 'L2_L2_pe_dcond_adam_s45_best.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model = PaperExactDCoND(**ckpt['model_kwargs'])
state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state)
model.to(device).eval()
model.set_train_sessions(train_sids)

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

# Load LIFT v1 and get initial translations
print("Loading LIFT v1...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_path = '/mnt/home/vincent.wilmet/brain2speech/models/phoneme_lift_qwen/final'
tok = AutoTokenizer.from_pretrained(adapter_path)
lift_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-2B', torch_dtype=torch.bfloat16, device_map='cuda:0')
lift_model = PeftModel.from_pretrained(lift_model, adapter_path)
lift_model.eval()

PROMPT = "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:"

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

print("Getting LIFT v1 translations...")
lift_translations = []
for d in val_data:
    if not d['gt_text']:
        lift_translations.append('')
        continue
    prompt = PROMPT.format(phones=d['decoded_phones'])
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(lift_model.device)
    with torch.no_grad():
        out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
    response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    lift_translations.append(response)

del lift_model; torch.cuda.empty_cache()

# Now load a larger model for correction
print("\nLoading Qwen3.5-2B (base, no adapter) for second-stage correction...")
# Try the base model without adapter first
base_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-2B', torch_dtype=torch.bfloat16, device_map='cuda:0')
base_model.eval()
tok2 = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B')
print("Base model loaded", flush=True)

# Correction prompts
CORRECTION_PROMPT_V1 = """The following sentence was decoded from a brain-computer interface and may contain errors. Correct any obvious mistakes while preserving the original meaning.

Decoded: {text}
Phonemes: {phones}

Corrected:"""

CORRECTION_PROMPT_V2 = """Fix any errors in this brain-computer interface transcription. Only fix clear mistakes, don't change correct words.

Input: {text}
Corrected:"""

CORRECTION_PROMPT_V3 = """A brain-computer interface produced this text and these phonemes. The phonemes are the raw BCI output. Produce the correct English sentence.

Text: {text}
Phonemes: {phones}

Correct sentence:"""

def correct_text(prompt_text, model, tokenizer):
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# Compare strategies
print("\nComparing correction strategies...", flush=True)
results = {
    'lift_v1': [], 
    'correct_v1_text_phones': [],
    'correct_v2_text_only': [],
    'correct_v3_text_phones': [],
}
t0 = time.time()

for idx in range(N_EVAL):
    d = val_data[idx]
    if not d['gt_text']:
        continue
    gt_words = normalize_text(d['gt_text'])
    if not gt_words:
        continue
    
    lift_text = lift_translations[idx]
    
    # LIFT v1 baseline
    w1 = lev_distance(normalize_text(lift_text), gt_words) / len(gt_words)
    results['lift_v1'].append(w1)
    
    # Correction v1: text + phones
    corrected = correct_text(
        CORRECTION_PROMPT_V1.format(text=lift_text, phones=d['decoded_phones']),
        base_model, tok2)
    w2 = lev_distance(normalize_text(corrected), gt_words) / len(gt_words)
    results['correct_v1_text_phones'].append(w2)
    
    # Correction v2: text only
    corrected = correct_text(
        CORRECTION_PROMPT_V2.format(text=lift_text),
        base_model, tok2)
    w3 = lev_distance(normalize_text(corrected), gt_words) / len(gt_words)
    results['correct_v2_text_only'].append(w3)
    
    # Correction v3: text + phones (different format)
    corrected = correct_text(
        CORRECTION_PROMPT_V3.format(text=lift_text, phones=d['decoded_phones']),
        base_model, tok2)
    w4 = lev_distance(normalize_text(corrected), gt_words) / len(gt_words)
    results['correct_v3_text_phones'].append(w4)
    
    n = len(results['lift_v1'])
    if n % 50 == 0:
        elapsed = time.time() - t0
        for k, v in results.items():
            print(f"  [{k}] {n} | WER: {np.mean(v):.1%}", flush=True)
        print(f"  ({elapsed:.0f}s)", flush=True)

# Report
print(f"\n{'='*60}")
print(f"SECOND-STAGE CORRECTION ({len(results['lift_v1'])} trials)")
print(f"{'='*60}")
for k, v in results.items():
    print(f"{k:30s}: Mean WER: {np.mean(v):.1%}, Median: {np.median(v):.1%}")

save_data = {k: {'mean_wer': float(np.mean(v)), 'median_wer': float(np.median(v))} for k, v in results.items()}
save_path = str(RESULTS_DIR / 'L2_second_stage_correction.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {save_path}")
