#!/usr/bin/env python3
"""Test different LIFT prompts to find the best one."""
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

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

# Define prompt variants
PROMPTS = {
    'v1_original': "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:",
    
    'v2_simple': "Phonemes: {phones}\n\nText:",
    
    'v3_context': "These phonemes were decoded from a brain-computer interface. The speaker was reading common English sentences. Some phonemes may be incorrect. Convert to the most likely English sentence.\n\nPhonemes: {phones}\n\nEnglish:",
    
    'v4_examples': """Convert decoded BCI phonemes to English text. Examples:
Phonemes: DH AH SIL K AE T SIL S AE T SIL AA N SIL DH AH SIL M AE T
English: The cat sat on the mat

Phonemes: {phones}
English:""",
    
    'v5_error_aware': "These phonemes were decoded from a brain-computer interface and may contain errors (insertions, deletions, substitutions). Convert to the most likely English sentence, correcting any obvious phoneme errors.\n\nPhonemes: {phones}\n\nEnglish:",
}

# Test each prompt
for pname, prompt_template in PROMPTS.items():
    print(f"\n--- Testing prompt: {pname} ---", flush=True)
    results = []
    t0 = time.time()
    
    for idx, d in enumerate(val_data):
        if not d['gt_text']:
            continue
        gt_words = normalize_text(d['gt_text'])
        if not gt_words:
            continue
        
        prompt = prompt_template.format(phones=d['decoded_phones'])
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(lift_model.device)
        with torch.no_grad():
            out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
        response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        lift_words = normalize_text(response)
        wer = lev_distance(lift_words, gt_words) / len(gt_words)
        results.append({'wer': wer, 'per': d['per']})
    
    elapsed = time.time() - t0
    avg_wer = np.mean([r['wer'] for r in results])
    med_wer = np.median([r['wer'] for r in results])
    print(f"  {pname}: {len(results)} trials | Mean WER: {avg_wer:.1%} | Median: {med_wer:.1%} | {elapsed:.0f}s")

print("\nDone!")
