#!/usr/bin/env python3
"""Ensemble DCoND decode: average log-probs from 4 seeds, try smarter decoding."""
import json, sys, time
import numpy as np
import h5py
import torch

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

M_ext = build_marginalization_matrix_ext()

# Load all 4 seed models
ckpts = [
    'L2_L2_pe_dcond_adam_s42_long_best.pt',
    'L2_L2_pe_dcond_adam_s43_best.pt',
    'L2_L2_pe_dcond_adam_s44_best.pt',
    'L2_L2_pe_dcond_adam_s45_best.pt',
]

models = []
for name in ckpts:
    path = str(RESULTS_DIR / name)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = PaperExactDCoND(**ckpt['model_kwargs'])
    state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    model.set_train_sessions(train_sids)
    models.append(model)
    print(f"  Loaded {name}")

def decode_trials_ensemble(trial_list, models, M_ext, method='avg_prob'):
    """Decode using ensemble of models.
    
    Methods:
    - avg_prob: average softmax probabilities (then log + greedy)
    - avg_logprob: average log-softmax (then greedy) 
    - max_prob: take max softmax probability per frame
    - vote: each model decodes independently, majority vote on phonemes
    """
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
        
        # Get predictions from all models
        all_probs = []
        all_logprobs = []
        for model in models:
            with torch.no_grad(), torch.amp.autocast('cuda'):
                logits = model(padded, session_ids)
            probs = logits.float().softmax(dim=-1).cpu()
            mono_probs = probs @ M_ext  # (B, T, 41)
            all_probs.append(mono_probs)
            all_logprobs.append((mono_probs + 1e-10).log())
        
        # Ensemble
        if method == 'avg_prob':
            ens_probs = torch.stack(all_probs).mean(dim=0)
            ens_lp = (ens_probs + 1e-10).log().numpy()
        elif method == 'avg_logprob':
            ens_lp = torch.stack(all_logprobs).mean(dim=0).numpy()
        elif method == 'max_prob':
            ens_probs = torch.stack(all_probs).max(dim=0)[0]
            ens_lp = (ens_probs + 1e-10).log().numpy()
        elif method == 'geometric':
            # Geometric mean of probabilities = exp(mean(log_probs))
            ens_lp = torch.stack(all_logprobs).mean(dim=0).numpy()
        
        for j, t in enumerate(batch):
            T_raw = feat_lens[j]
            T_out = max(1, (T_raw - models[0].kernel_size) // models[0].stride + 1)
            T_out = min(T_out, ens_lp.shape[1])
            decoded = ctc_greedy_decode(ens_lp[j, :T_out], blank=CTC_BLANK)
            target = t['phoneme_indices'].tolist()
            data.append({
                'decoded_phones': ' '.join([CLASS_TO_ARPABET[p] for p in decoded]),
                'gt_text': t['text'],
                'per': compute_per(decoded, target),
            })
    return data

# Also decode with single best model for comparison
print("\nDecoding with single best (seed45)...")
single_data_val = decode_trials_ensemble(val_trials, [models[3]], M_ext, 'avg_prob')
single_data_test = decode_trials_ensemble(test_trials, [models[3]], M_ext, 'avg_prob')
print(f"  Val PER (single): {np.mean([d['per'] for d in single_data_val]):.3f}")
print(f"  Test PER (single): {np.mean([d['per'] for d in single_data_test]):.3f}")

# Try different ensemble methods
results = {}
for method in ['avg_prob', 'avg_logprob', 'geometric']:
    print(f"\nDecoding with ensemble ({method})...")
    t0 = time.time()
    ens_val = decode_trials_ensemble(val_trials, models, M_ext, method)
    ens_test = decode_trials_ensemble(test_trials, models, M_ext, method)
    val_per = np.mean([d['per'] for d in ens_val])
    test_per = np.mean([d['per'] for d in ens_test])
    elapsed = time.time() - t0
    print(f"  Val PER: {val_per:.3f} | Test PER: {test_per:.3f} | {elapsed:.0f}s")
    results[method] = {
        'val_per': float(val_per),
        'test_per': float(test_per),
        'val_data': ens_val,
        'test_data': ens_test,
    }

# Also try weighted ensemble (give more weight to best seed)
print(f"\nDecoding with weighted ensemble (0.1, 0.15, 0.2, 0.55 for seeds 42-45)...")
# Weight by inverse val PER
weights = [0.1, 0.15, 0.2, 0.55]  # seed45 gets most weight
for i in range(0, len(val_trials), 32):
    batch = val_trials[i:i+32]
    features = [torch.FloatTensor(t['features_raw']) for t in batch]
    feat_lens = [f.shape[0] for f in features]
    max_T = max(feat_lens)
    C = features[0].shape[1]
    padded = torch.zeros(len(batch), max_T, C).to(device)
    for j, f in enumerate(features):
        padded[j, :f.shape[0], :] = f
    session_ids = torch.LongTensor([t.get('session_idx', 0) for t in batch]).to(device)
    
    weighted_probs = None
    for k, model in enumerate(models):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits = model(padded, session_ids)
        probs = logits.float().softmax(dim=-1).cpu()
        mono_probs = probs @ M_ext
        if weighted_probs is None:
            weighted_probs = mono_probs * weights[k]
        else:
            weighted_probs += mono_probs * weights[k]
    
    ens_lp = (weighted_probs + 1e-10).log().numpy()
    for j, t in enumerate(batch):
        T_raw = feat_lens[j]
        T_out = max(1, (T_raw - models[0].kernel_size) // models[0].stride + 1)
        T_out = min(T_out, ens_lp.shape[1])
        # Just compute PER for val
        decoded = ctc_greedy_decode(ens_lp[j, :T_out], blank=CTC_BLANK)
        target = t['phoneme_indices'].tolist()
        if i == 0 and j == 0:
            weighted_pers = []
        weighted_pers.append(compute_per(decoded, target))

weighted_val_per = np.mean(weighted_pers) if weighted_pers else 0
print(f"  Val PER (weighted): {weighted_val_per:.3f}")

# Report
print(f"\n{'='*60}")
print(f"ENSEMBLE DECODE COMPARISON")
print(f"{'='*60}")
print(f"Single (seed45):  Val PER: {np.mean([d['per'] for d in single_data_val]):.3f} | Test PER: {np.mean([d['per'] for d in single_data_test]):.3f}")
for method, r in results.items():
    print(f"Ensemble ({method:12s}): Val PER: {r['val_per']:.3f} | Test PER: {r['test_per']:.3f}")
print(f"Weighted ensemble: Val PER: {weighted_val_per:.3f}")

# Save best ensemble phones for LIFT
best_method = min(results, key=lambda k: results[k]['val_per'])
print(f"\nBest ensemble method: {best_method}")

save_data = {
    'single_seed45': {
        'val_per': float(np.mean([d['per'] for d in single_data_val])),
        'test_per': float(np.mean([d['per'] for d in single_data_test])),
    },
}
for method, r in results.items():
    save_data[f'ensemble_{method}'] = {'val_per': r['val_per'], 'test_per': r['test_per']}
save_data['weighted_ensemble'] = {'val_per': float(weighted_val_per)}

save_path = str(RESULTS_DIR / 'L2_ensemble_decode_comparison.json')
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"Saved: {save_path}")
