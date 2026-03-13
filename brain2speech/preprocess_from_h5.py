#!/usr/bin/env python3
"""
Quick preprocessing: Create 256D and 1280D+SIL HDF5 from existing sentences_train.h5.

This is faster than preprocess_paper_exact.py (reads HDF5 instead of .mat files)
but uses existing block-level z-scoring rather than rolling z-score.

Usage:
    python preprocess_from_h5.py           # both 256D and 1280D
    python preprocess_from_h5.py --only 256
"""
import argparse
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
INPUT_H5 = DATA_DIR / "sentences_train.h5"

ARPABET_CLASSES = [
    'B', 'CH', 'SIL', 'D', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'ER', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'DH', 'T', 'TH',
    'V', 'W', 'Y', 'Z', 'ZH',
    'OY', 'EH', 'EY', 'UH', 'IY', 'OW', 'UW', 'IH', 'AA', 'AW',
    'AY', 'AH', 'AO', 'AE',
]
PHONE_TO_IDX = {p: i for i, p in enumerate(ARPABET_CLASSES)}
SIL_IDX = PHONE_TO_IDX['SIL']  # 2
N_CLASSES = len(ARPABET_CLASSES)
CTC_BLANK = N_CLASSES


def strip_stress(phone):
    return re.sub(r'[012]$', '', phone)


def sentence_to_phonemes_with_sil(text, g2p_model):
    """Convert sentence to phonemes WITH inter-word SIL tokens."""
    words = text.strip().split()
    indices = []
    labels = []
    for w_idx, word in enumerate(words):
        raw_phones = g2p_model(word)
        for p in raw_phones:
            if not p.strip() or p in ' .,!?;:\'"()-':
                continue
            p_clean = strip_stress(p)
            if p_clean in PHONE_TO_IDX:
                indices.append(PHONE_TO_IDX[p_clean])
                labels.append(p_clean)
        # SIL after every word (including last)
        indices.append(SIL_IDX)
        labels.append('SIL')
    return indices, labels


def select_256d(features_1280d):
    """spikePow[:,:128] + tx1[:,:128] = area 6v 256D."""
    sp_6v = features_1280d[:, 0:128]
    tx1_6v = features_1280d[:, 256:384]
    return np.concatenate([sp_6v, tx1_6v], axis=1)


def create_h5(input_path, output_path, feature_fn, g2p_model):
    """Create new HDF5 with selected features and SIL-augmented targets."""
    with h5py.File(input_path, 'r') as fin, \
         h5py.File(output_path, 'w') as fout:

        n_trials = fin.attrs['n_trials']
        fout.attrs['n_trials'] = n_trials
        fout.attrs['n_classes'] = N_CLASSES
        fout.attrs['ctc_blank'] = CTC_BLANK
        fout.attrs['has_interword_sil'] = True

        t0 = time.time()
        for i in range(n_trials):
            if i % 1000 == 0:
                print(f"  {i}/{n_trials} ({time.time()-t0:.1f}s)...", flush=True)

            grp_in = fin[f'trial_{i:05d}']
            features = grp_in['features'][:].astype(np.float32)
            features_out = feature_fn(features)

            text = grp_in.attrs['text']
            phone_idx, phone_labels = sentence_to_phonemes_with_sil(text, g2p_model)
            if len(phone_idx) == 0:
                phone_idx = grp_in['phoneme_indices'][:].tolist()
                phone_labels = grp_in.attrs.get('phoneme_labels', '').split()

            grp_out = fout.create_group(f'trial_{i:05d}')
            grp_out.create_dataset('features', data=features_out, compression='gzip',
                                   compression_opts=4)
            grp_out.create_dataset('phoneme_indices',
                                   data=np.array(phone_idx, dtype=np.int32))
            grp_out.attrs['text'] = text
            grp_out.attrs['session'] = grp_in.attrs['session']
            grp_out.attrs['block_id'] = grp_in.attrs['block_id']
            grp_out.attrs['n_frames'] = features_out.shape[0]
            grp_out.attrs['n_phonemes'] = len(phone_idx)
            grp_out.attrs['phoneme_labels'] = ' '.join(phone_labels)

        print(f"  Done: {n_trials} trials in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=int, choices=[256, 1280], default=None)
    args = parser.parse_args()

    if not INPUT_H5.exists():
        print(f"ERROR: {INPUT_H5} not found")
        sys.exit(1)

    print("Loading g2p model...")
    from g2p_en import G2p
    g2p = G2p()

    # Verify SIL insertion
    test_idx, test_lbl = sentence_to_phonemes_with_sil("hello world", g2p)
    sil_count = sum(1 for l in test_lbl if l == 'SIL')
    print(f"  Test: 'hello world' → {test_lbl} ({sil_count} SIL tokens)")

    if args.only is None or args.only == 256:
        out = DATA_DIR / "sentences_paper_256d.h5"
        print(f"\nCreating 256D + SIL → {out}")
        create_h5(INPUT_H5, out, select_256d, g2p)
        with h5py.File(out, 'r') as f:
            s = f['trial_00000']['features'].shape
            print(f"  Feature shape: {s}")

    if args.only is None or args.only == 1280:
        out = DATA_DIR / "sentences_paper_1280d.h5"
        print(f"\nCreating 1280D + SIL → {out}")
        create_h5(INPUT_H5, out, lambda x: x, g2p)
        with h5py.File(out, 'r') as f:
            s = f['trial_00000']['features'].shape
            print(f"  Feature shape: {s}")

    print("\nDone!")


if __name__ == '__main__':
    main()
