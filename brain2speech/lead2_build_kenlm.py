#!/usr/bin/env python3
"""Build KenLM 5-gram phoneme LM from training data.

Usage:
    python brain2speech/lead2_build_kenlm.py \
        --data brain2speech/data/sentences_paper_256d.h5 \
        --output brain2speech/data/phoneme_5gram.arpa --order 5

Produces a phoneme-level n-gram language model for use in CTC beam search decoding.
The ARPA file can be used with pyctcdecode or our custom beam search in lead2_decode_kenlm.py.
"""
import subprocess
import argparse
import h5py
import sys

sys.path.insert(0, 'brain2speech')
from config import CLASS_TO_ARPABET


def build_kenlm(h5_path, output_arpa, order=5):
    """Build KenLM model from phoneme sequences in HDF5 data."""
    corpus = output_arpa.replace('.arpa', '.txt')

    # Extract phoneme sequences from training data
    print(f"Extracting phoneme sequences from {h5_path}...")
    session_names = []
    with h5py.File(h5_path, 'r') as f:
        n_trials = f.attrs['n_trials']
        # Get session names for train/val/test split
        sessions = set()
        for i in range(n_trials):
            sessions.add(f[f'trial_{i:05d}'].attrs['session'])
        session_names = sorted(sessions)

    # Train sessions = all except last 6 (last 4=test, -6:-4=val)
    train_sessions = set(session_names[:-6])

    n_written = 0
    with h5py.File(h5_path, 'r') as f:
        with open(corpus, 'w') as out:
            for i in range(f.attrs['n_trials']):
                grp = f[f'trial_{i:05d}']
                if grp.attrs['session'] not in train_sessions:
                    continue
                phones = grp['phoneme_indices'][:]
                names = [CLASS_TO_ARPABET[int(p)] for p in phones]
                out.write(' '.join(names) + '\n')
                n_written += 1

    print(f"Wrote {n_written} training sentences to {corpus}")

    # Build KenLM ARPA model
    print(f"Building {order}-gram model...")
    subprocess.run(
        ['lmplz', '-o', str(order), '--text', corpus,
         '--arpa', output_arpa, '--discount_fallback'],
        check=True
    )
    print(f"ARPA model: {output_arpa}")

    # Build binary for faster loading
    bin_path = output_arpa.replace('.arpa', '.bin')
    try:
        subprocess.run(['build_binary', output_arpa, bin_path], check=True)
        print(f"Binary model: {bin_path}")
    except FileNotFoundError:
        print("build_binary not found, skipping binary model")

    return output_arpa


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='brain2speech/data/sentences_paper_256d.h5')
    parser.add_argument('--output', default='brain2speech/data/phoneme_5gram.arpa')
    parser.add_argument('--order', type=int, default=5)
    args = parser.parse_args()
    build_kenlm(args.data, args.output, args.order)
