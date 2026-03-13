#!/usr/bin/env python3
"""
Paper-exact preprocessing for Willett et al. 2023 replication.

Key differences from preprocess_sentences.py:
1. 256D features: spikePow[:,:128] + tx1[:,:128] (area 6v only)
2. Inter-word SIL tokens in phoneme targets (paper: "silence phoneme added to end of each word")
3. Rolling z-score per paper's equation 2
4. Also produces 1280D version for Track B experiments

Usage:
    python preprocess_paper_exact.py                    # produce both 256D and 1280D
    python preprocess_paper_exact.py --features 256     # 256D only
    python preprocess_paper_exact.py --features 1280    # 1280D only
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = Path("/mnt/home/vincent.wilmet/docs/data/dryad/competitionData")
SENTENCES_DIR = Path("/mnt/home/vincent.wilmet/docs/data/dryad/sentences")
OUTPUT_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")

# ── Phoneme mapping (must match config.py) ────────────────────────────
ARPABET_CLASSES = [
    'B', 'CH', 'SIL', 'D', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'ER', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'DH', 'T', 'TH',
    'V', 'W', 'Y', 'Z', 'ZH',
    'OY', 'EH', 'EY', 'UH', 'IY', 'OW', 'UW', 'IH', 'AA', 'AW',
    'AY', 'AH', 'AO', 'AE',
]
PHONE_TO_IDX = {p: i for i, p in enumerate(ARPABET_CLASSES)}
SIL_IDX = PHONE_TO_IDX['SIL']  # 2
N_CLASSES = len(ARPABET_CLASSES)  # 40
CTC_BLANK = N_CLASSES  # 40


def strip_stress(phone):
    """Remove stress markers from ARPABET vowels: 'AH0' → 'AH'."""
    return re.sub(r'[012]$', '', phone)


def sentence_to_phonemes_with_sil(text, g2p_model):
    """Convert sentence text to phoneme sequence WITH inter-word SIL tokens.

    Paper: "We automatically added a 'silence' phoneme to the end of each word
    in order to denote the separation between words."

    Returns:
        phoneme_indices: list of int
        phoneme_labels: list of str
    """
    words = text.strip().split()
    indices = []
    labels = []

    for word_idx, word in enumerate(words):
        # g2p each word individually to get clean phoneme sequence
        raw_phones = g2p_model(word)
        for p in raw_phones:
            if not p.strip() or p in ' .,!?;:\'"()-':
                continue
            p_clean = strip_stress(p)
            if p_clean in PHONE_TO_IDX:
                indices.append(PHONE_TO_IDX[p_clean])
                labels.append(p_clean)

        # Add SIL after every word (including last word)
        indices.append(SIL_IDX)
        labels.append('SIL')

    return indices, labels


def rolling_zscore_block(trials_in_block):
    """Rolling z-score within a block, matching paper's equation 2.

    Paper: For the first 10 sentences of a new block, use weighted average:
        u_i = (11-i)/10 * u_prev + (i-1)/10 * u_curr
    After 10 sentences, use mean of last min(20, N) sentences.

    Since we don't have the previous block's stats during preprocessing,
    we use the first sentence's stats as u_prev (warm start).

    Args:
        trials_in_block: list of (T, C) feature arrays for each trial in the block

    Returns:
        list of (T, C) z-scored feature arrays
    """
    if not trials_in_block:
        return []

    C = trials_in_block[0].shape[1]

    # Compute per-trial means and stds
    trial_means = [t.mean(axis=0) for t in trials_in_block]
    trial_stds = [t.std(axis=0) for t in trials_in_block]

    # First trial uses its own stats (as u_prev reference)
    u_prev = trial_means[0].copy()
    s_prev = trial_stds[0].copy()
    s_prev[s_prev < 1e-8] = 1.0

    results = []
    for i, trial in enumerate(trials_in_block):
        sentence_num = i + 1  # 1-indexed

        if sentence_num <= 10:
            # Warmup: weighted average of prior estimate and current block mean
            if i == 0:
                u_curr = trial_means[0]
                s_curr = trial_stds[0]
            else:
                # Mean of all sentences so far in this block
                u_curr = np.mean(trial_means[:i+1], axis=0)
                s_curr = np.mean(trial_stds[:i+1], axis=0)

            w_prev = (11 - sentence_num) / 10.0
            w_curr = (sentence_num - 1) / 10.0
            u = w_prev * u_prev + w_curr * u_curr
            s = w_prev * s_prev + w_curr * s_curr
        else:
            # After warmup: mean of last min(20, N) sentences
            window = min(20, i + 1)
            u = np.mean(trial_means[i+1-window:i+1], axis=0)
            s = np.mean(trial_stds[i+1-window:i+1], axis=0)

        s[s < 1e-8] = 1.0
        results.append(((trial - u) / s).astype(np.float32))

    return results


def extract_features_256d(mat, trial_idx):
    """Extract 256D features: spikePow[:,:128] + tx1[:,:128] (area 6v only).

    Paper: "threshold crossing counts and spike band power from the 128
    electrodes in area 6v were concatenated to yield a 256 x 1 feature vector"

    Arrays 1-2 (channels 0-127) are in area 6v.
    """
    sp = mat['spikePow'][0, trial_idx][:, :128]   # (T, 128) area 6v spike power
    tx = mat['tx1'][0, trial_idx][:, :128]          # (T, 128) area 6v threshold crossings
    T = sp.shape[0]
    return np.concatenate([tx[:T], sp[:T]], axis=1).astype(np.float32)  # (T, 256)


def extract_features_1280d(mat, trial_idx):
    """Extract 1280D features: all channels, all feature types."""
    sp = mat['spikePow'][0, trial_idx]
    tx1 = mat['tx1'][0, trial_idx]
    tx2 = mat['tx2'][0, trial_idx]
    tx3 = mat['tx3'][0, trial_idx]
    tx4 = mat['tx4'][0, trial_idx]
    T = sp.shape[0]
    return np.concatenate([sp[:T], tx1[:T], tx2[:T], tx3[:T], tx4[:T]], axis=1).astype(np.float32)


def process_session(mat_path, g2p_model, feature_dim=256):
    """Process one .mat session file."""
    mat = sio.loadmat(str(mat_path))
    n_trials = mat['spikePow'].shape[1]
    session_name = mat_path.stem

    extract_fn = extract_features_256d if feature_dim == 256 else extract_features_1280d

    trials = []
    for i in range(n_trials):
        features = extract_fn(mat, i)
        text = mat['sentenceText'][i].strip()
        block_id = int(mat['blockIdx'][i, 0])

        phone_idx, phone_labels = sentence_to_phonemes_with_sil(text, g2p_model)
        if len(phone_idx) == 0:
            continue

        trials.append({
            'features': features,
            'phoneme_indices': np.array(phone_idx, dtype=np.int32),
            'phoneme_labels': phone_labels,
            'text': text,
            'block_id': block_id,
            'session': session_name,
            'n_frames': features.shape[0],
            'n_phonemes': len(phone_idx),
        })

    return trials


def apply_rolling_zscore(all_trials):
    """Apply rolling z-score per block within each session."""
    # Group by session and block
    session_blocks = {}
    for trial in all_trials:
        key = (trial['session'], trial['block_id'])
        session_blocks.setdefault(key, []).append(trial)

    # Apply rolling z-score within each block
    for key, block_trials in session_blocks.items():
        features_list = [t['features'] for t in block_trials]
        zscored = rolling_zscore_block(features_list)
        for trial, feat in zip(block_trials, zscored):
            trial['features'] = feat


def save_to_hdf5(trials, output_path, feature_dim):
    """Save variable-length trials to HDF5."""
    with h5py.File(output_path, 'w') as f:
        f.attrs['n_trials'] = len(trials)
        f.attrs['n_classes'] = N_CLASSES
        f.attrs['ctc_blank'] = CTC_BLANK
        f.attrs['feature_dim'] = feature_dim
        f.attrs['has_interword_sil'] = True

        for i, trial in enumerate(trials):
            grp = f.create_group(f'trial_{i:05d}')
            grp.create_dataset('features', data=trial['features'], compression='gzip')
            grp.create_dataset('phoneme_indices', data=trial['phoneme_indices'])
            grp.attrs['text'] = trial['text']
            grp.attrs['session'] = trial['session']
            grp.attrs['block_id'] = trial['block_id']
            grp.attrs['n_frames'] = trial['n_frames']
            grp.attrs['n_phonemes'] = trial['n_phonemes']
            grp.attrs['phoneme_labels'] = ' '.join(trial['phoneme_labels'])

    print(f"  Saved {len(trials)} trials to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=int, choices=[256, 1280], default=0,
                        help='Feature dim (0=both 256 and 1280)')
    parser.add_argument('--no-rolling-zscore', action='store_true',
                        help='Use block-level z-score instead of rolling')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_dims = [256, 1280] if args.features == 0 else [args.features]

    print("Initializing g2p model...")
    from g2p_en import G2p
    g2p = G2p()

    # Verify SIL token handling
    test_idx, test_labels = sentence_to_phonemes_with_sil("hello world", g2p)
    print(f"  Test: 'hello world' → {test_labels}")
    assert 'SIL' in test_labels, "SIL tokens not inserted!"
    sil_count = sum(1 for l in test_labels if l == 'SIL')
    print(f"  SIL tokens: {sil_count} (expected: 2, one per word)")

    for feature_dim in feature_dims:
        print(f"\n{'='*70}")
        print(f"Processing with {feature_dim}D features (area 6v only)" if feature_dim == 256
              else f"Processing with {feature_dim}D features (all electrodes)")
        print(f"{'='*70}")

        # Process train split
        split_dir = DATA_DIR / 'train'
        mat_files = sorted(split_dir.glob('*.mat'))
        print(f"\nTrain: {len(mat_files)} sessions")

        all_trials = []
        for mat_path in mat_files:
            print(f"  {mat_path.name}...", end=' ', flush=True)
            trials = process_session(mat_path, g2p, feature_dim)
            print(f"{len(trials)} trials, "
                  f"avg {np.mean([t['n_frames'] for t in trials]):.0f} frames, "
                  f"avg {np.mean([t['n_phonemes'] for t in trials]):.0f} phonemes")
            all_trials.extend(trials)

        # Apply z-scoring
        if args.no_rolling_zscore:
            print("\nApplying per-block z-score normalization...")
            # Group by session for block-level z-scoring
            sessions = {}
            for trial in all_trials:
                sessions.setdefault(trial['session'], []).append(trial)
            for session_trials in sessions.values():
                all_features = np.concatenate([t['features'] for t in session_trials])
                all_block_ids = np.concatenate([
                    np.full(t['n_frames'], t['block_id']) for t in session_trials
                ])
                # Z-score per block
                result = all_features.copy().astype(np.float32)
                for bid in np.unique(all_block_ids):
                    mask = all_block_ids == bid
                    block_data = result[mask]
                    mu = block_data.mean(axis=0)
                    sd = block_data.std(axis=0)
                    sd[sd < 1e-8] = 1.0
                    result[mask] = (block_data - mu) / sd
                offset = 0
                for trial in session_trials:
                    n = trial['n_frames']
                    trial['features'] = result[offset:offset + n]
                    offset += n
        else:
            print("\nApplying rolling z-score normalization (paper equation 2)...")
            apply_rolling_zscore(all_trials)

        # Save
        suffix = f"paper_{feature_dim}d"
        output_path = OUTPUT_DIR / f"sentences_{suffix}.h5"
        save_to_hdf5(all_trials, output_path, feature_dim)

        # Stats
        total_frames = sum(t['n_frames'] for t in all_trials)
        total_phonemes = sum(t['n_phonemes'] for t in all_trials)
        n_sil = sum(sum(1 for p in t['phoneme_indices'] if p == SIL_IDX) for t in all_trials)
        print(f"\nSummary ({feature_dim}D):")
        print(f"  Sessions: {len(mat_files)}")
        print(f"  Total trials: {len(all_trials)}")
        print(f"  Total frames: {total_frames} ({total_frames * 20 / 1000 / 3600:.1f} hours)")
        print(f"  Total phonemes: {total_phonemes}")
        print(f"  Total SIL tokens: {n_sil} ({n_sil/total_phonemes*100:.1f}% of phonemes)")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Avg frames/trial: {total_frames / len(all_trials):.0f}")
        print(f"  Avg phonemes/trial: {total_phonemes / len(all_trials):.1f}")

        # Save stats
        stats_path = OUTPUT_DIR / f"sentences_{suffix}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'feature_dim': feature_dim,
                'has_interword_sil': True,
                'rolling_zscore': not args.no_rolling_zscore,
                'n_sessions': len(mat_files),
                'n_trials': len(all_trials),
                'total_frames': total_frames,
                'total_phonemes': total_phonemes,
                'total_sil_tokens': n_sil,
            }, f, indent=2)


if __name__ == '__main__':
    main()
