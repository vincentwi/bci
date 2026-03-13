#!/usr/bin/env python3
"""
Preprocess competitionData for CTC-based phoneme decoding.

Converts sentence-level neural data into a format suitable for CTC training:
1. Load .mat files → extract spikePow + tx1-tx4 features
2. Z-score features per block
3. Convert sentence text → ARPABET phoneme sequences via g2p
4. Save as HDF5 for efficient variable-length loading

Output: brain2speech/data/sentences_{split}.h5
  - Each trial: neural features (T, 1280) + phoneme labels (P,)

Usage:
    python preprocess_sentences.py              # process train + test
    python preprocess_sentences.py --split train
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from g2p_en import G2p

DATA_DIR = Path("/mnt/home/vincent.wilmet/docs/data/dryad/competitionData")
OUTPUT_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")

# Map g2p output (with stress markers) to our 39+1 ARPABET classes
# Strip stress numbers from vowels to match our class set
ARPABET_CLASSES = [
    'B', 'CH', 'SIL', 'D', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'ER', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'DH', 'T', 'TH',
    'V', 'W', 'Y', 'Z', 'ZH',
    'OY', 'EH', 'EY', 'UH', 'IY', 'OW', 'UW', 'IH', 'AA', 'AW',
    'AY', 'AH', 'AO', 'AE',
]
PHONE_TO_IDX = {p: i for i, p in enumerate(ARPABET_CLASSES)}
N_CLASSES = len(ARPABET_CLASSES)  # 40

# CTC blank token
CTC_BLANK = N_CLASSES  # index 40


def strip_stress(phone):
    """Remove stress markers from ARPABET vowels: 'AH0' → 'AH'."""
    return re.sub(r'[012]$', '', phone)


def sentence_to_phonemes(text, g2p_model):
    """Convert sentence text to ARPABET phoneme index sequence.

    Returns:
        phoneme_indices: list of int (class indices, no blanks)
        phoneme_labels: list of str (ARPABET labels)
    """
    raw_phones = g2p_model(text.strip())

    # Filter: keep only actual phonemes (no spaces, punctuation)
    phones = [p for p in raw_phones if p.strip() and p not in ' .,!?;:\'"()-']

    indices = []
    labels = []
    for p in phones:
        p_clean = strip_stress(p)
        if p_clean in PHONE_TO_IDX:
            indices.append(PHONE_TO_IDX[p_clean])
            labels.append(p_clean)
        # Skip unknown phonemes (rare)

    return indices, labels


def zscore_by_block(features, block_ids):
    """Z-score features per block. Same as preprocess.py."""
    result = features.copy().astype(np.float32)
    for bid in np.unique(block_ids):
        mask = block_ids == bid
        block_data = result[mask]
        mu = block_data.mean(axis=0)
        sd = block_data.std(axis=0)
        sd[sd < 1e-8] = 1.0
        result[mask] = (block_data - mu) / sd
    return result


def process_session(mat_path, g2p_model):
    """Process one .mat session file.

    Returns list of dicts: {features, phoneme_indices, phoneme_labels, text, session}
    """
    mat = sio.loadmat(str(mat_path))
    n_trials = mat['spikePow'].shape[1]
    session_name = mat_path.stem

    trials = []
    for i in range(n_trials):
        sp = mat['spikePow'][0, i]  # (T, 256)
        tx1 = mat['tx1'][0, i]       # (T, 256)
        tx2 = mat['tx2'][0, i]
        tx3 = mat['tx3'][0, i]
        tx4 = mat['tx4'][0, i]

        T = sp.shape[0]
        # Concatenate: spikePow + tx1-4 = 1280 features
        features = np.concatenate([sp, tx1[:T], tx2[:T], tx3[:T], tx4[:T]], axis=1)

        text = mat['sentenceText'][i].strip()
        block_id = int(mat['blockIdx'][i, 0])

        phone_idx, phone_labels = sentence_to_phonemes(text, g2p_model)

        if len(phone_idx) == 0:
            continue

        trials.append({
            'features': features.astype(np.float32),
            'phoneme_indices': np.array(phone_idx, dtype=np.int32),
            'phoneme_labels': phone_labels,
            'text': text,
            'block_id': block_id,
            'session': session_name,
            'n_frames': T,
            'n_phonemes': len(phone_idx),
        })

    return trials


def save_to_hdf5(trials, output_path):
    """Save variable-length trials to HDF5."""
    with h5py.File(output_path, 'w') as f:
        f.attrs['n_trials'] = len(trials)
        f.attrs['n_classes'] = N_CLASSES
        f.attrs['ctc_blank'] = CTC_BLANK

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
    parser.add_argument('--split', choices=['train', 'test', 'both'], default='both')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing g2p model...")
    g2p = G2p()

    splits = ['train', 'test'] if args.split == 'both' else [args.split]

    for split in splits:
        split_dir = DATA_DIR / split
        mat_files = sorted(split_dir.glob('*.mat'))
        print(f"\n{'='*60}")
        print(f"Processing {split} split: {len(mat_files)} sessions")
        print(f"{'='*60}")

        all_trials = []
        session_stats = []

        for mat_path in mat_files:
            print(f"  {mat_path.name}...", end=' ')
            trials = process_session(mat_path, g2p)
            print(f"{len(trials)} trials, "
                  f"avg {np.mean([t['n_frames'] for t in trials]):.0f} frames, "
                  f"avg {np.mean([t['n_phonemes'] for t in trials]):.0f} phonemes")

            session_stats.append({
                'session': mat_path.stem,
                'n_trials': len(trials),
                'avg_frames': float(np.mean([t['n_frames'] for t in trials])),
                'avg_phonemes': float(np.mean([t['n_phonemes'] for t in trials])),
            })
            all_trials.extend(trials)

        # Z-score by block within each session
        print("\nApplying per-block z-score normalization...")
        # Group trials by session for normalization
        sessions = {}
        for trial in all_trials:
            s = trial['session']
            if s not in sessions:
                sessions[s] = []
            sessions[s].append(trial)

        for session_name, session_trials in sessions.items():
            # Concatenate all frames in session
            all_features = np.concatenate([t['features'] for t in session_trials])
            all_block_ids = np.concatenate([
                np.full(t['n_frames'], t['block_id']) for t in session_trials
            ])

            # Z-score per block
            normed = zscore_by_block(all_features, all_block_ids)

            # Split back into trials
            offset = 0
            for trial in session_trials:
                n = trial['n_frames']
                trial['features'] = normed[offset:offset + n]
                offset += n

        # Save
        output_path = OUTPUT_DIR / f"sentences_{split}.h5"
        save_to_hdf5(all_trials, output_path)

        # Stats
        total_frames = sum(t['n_frames'] for t in all_trials)
        total_phonemes = sum(t['n_phonemes'] for t in all_trials)
        print(f"\n{split} summary:")
        print(f"  Sessions: {len(mat_files)}")
        print(f"  Total trials: {len(all_trials)}")
        print(f"  Total frames: {total_frames} ({total_frames * 20 / 1000:.0f}s)")
        print(f"  Total phonemes: {total_phonemes}")
        print(f"  Avg phonemes/trial: {total_phonemes / len(all_trials):.1f}")
        print(f"  Avg frames/trial: {total_frames / len(all_trials):.0f}")

        # Save stats
        stats_path = OUTPUT_DIR / f"sentences_{split}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'split': split,
                'n_sessions': len(mat_files),
                'n_trials': len(all_trials),
                'total_frames': total_frames,
                'total_phonemes': total_phonemes,
                'session_stats': session_stats,
            }, f, indent=2)


if __name__ == '__main__':
    main()
