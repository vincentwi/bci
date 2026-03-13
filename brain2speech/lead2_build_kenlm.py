#!/usr/bin/env python3
"""
Lead 2: Build KenLM phoneme n-gram language model from training data.

Builds a phoneme-level n-gram model (default: 5-gram) from training transcripts
in HDF5 format. Outputs ARPA and binary formats for use with pyctcdecode.

Sources:
  - tbenst/silent_speech: KenLM beam search for BCI decoding
  - arxiv 2412.17227: 5-gram phoneme LM used by top teams

Usage:
    python brain2speech/lead2_build_kenlm.py \
        --data sentences_paper_256d.h5 \
        --output brain2speech/data/phoneme_5gram.arpa --order 5

    # Verify:
    python brain2speech/lead2_build_kenlm.py --verify \
        --lm brain2speech/data/phoneme_5gram.bin
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CLASS_TO_ARPABET, N_CLASSES

DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")


def extract_phoneme_corpus(h5_path, output_txt, split='train'):
    """Extract phoneme sequences from HDF5 training data into text corpus.

    Each line is a space-separated sequence of ARPABET phoneme names.
    Only uses training sessions (excluding last 6 sessions: 4 test + 2 val).
    """
    sequences = []
    with h5py.File(h5_path, 'r') as f:
        n_trials = f.attrs['n_trials']

        # Collect session info to split train/val/test
        sessions_per_trial = []
        for i in range(n_trials):
            grp = f[f'trial_{i:05d}']
            sessions_per_trial.append(grp.attrs['session'])

        session_names = sorted(set(sessions_per_trial))
        n_sessions = len(session_names)

        if split == 'train':
            # Match train_beyond_paper.py: last 4 = test, next 2 = val
            train_sessions = set(session_names[:-6])
        elif split == 'all':
            train_sessions = set(session_names)
        else:
            raise ValueError(f"Unknown split: {split}")

        for i in range(n_trials):
            grp = f[f'trial_{i:05d}']
            if grp.attrs['session'] not in train_sessions:
                continue
            phones = grp['phoneme_indices'][:]
            names = [CLASS_TO_ARPABET[int(p)] for p in phones]
            sequences.append(' '.join(names))

    with open(output_txt, 'w') as f:
        for seq in sequences:
            f.write(seq + '\n')

    print(f"Extracted {len(sequences)} phoneme sequences to {output_txt}")
    print(f"  Sessions used: {len(train_sessions)} ({split})")
    print(f"  Vocabulary: {len(CLASS_TO_ARPABET)} phonemes")
    return len(sequences)


def build_kenlm(corpus_txt, output_arpa, order=5):
    """Build KenLM n-gram model from phoneme corpus.

    Requires kenlm tools (lmplz, build_binary) to be installed.
    """
    # Check if lmplz is available
    lmplz = shutil.which('lmplz')
    build_binary = shutil.which('build_binary')

    if not lmplz or not build_binary:
        # Try common install locations
        for prefix in ['/usr/local/bin', os.path.expanduser('~/.local/bin'),
                       '/opt/kenlm/bin']:
            if os.path.exists(os.path.join(prefix, 'lmplz')):
                lmplz = os.path.join(prefix, 'lmplz')
                build_binary = os.path.join(prefix, 'build_binary')
                break

    if not lmplz:
        print("ERROR: lmplz not found. Install KenLM:")
        print("  pip install kenlm")
        print("  # Or build from source:")
        print("  git clone https://github.com/kpu/kenlm.git")
        print("  cd kenlm && mkdir build && cd build")
        print("  cmake .. && make -j4")
        print("  export PATH=$PWD/bin:$PATH")
        sys.exit(1)

    print(f"\nBuilding {order}-gram ARPA model...")
    print(f"  lmplz: {lmplz}")
    print(f"  Input: {corpus_txt}")
    print(f"  Output: {output_arpa}")

    cmd = [lmplz, '-o', str(order),
           '--text', corpus_txt,
           '--arpa', output_arpa,
           '--discount_fallback']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"lmplz STDERR:\n{result.stderr}")
        raise RuntimeError(f"lmplz failed with exit code {result.returncode}")

    print(f"  ARPA model: {output_arpa} ({os.path.getsize(output_arpa) / 1e6:.1f} MB)")

    # Build binary for faster loading
    bin_path = output_arpa.replace('.arpa', '.bin')
    print(f"\nBuilding binary model...")
    cmd = [build_binary, output_arpa, bin_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"build_binary STDERR:\n{result.stderr}")
        raise RuntimeError(f"build_binary failed with exit code {result.returncode}")

    print(f"  Binary model: {bin_path} ({os.path.getsize(bin_path) / 1e6:.1f} MB)")
    return bin_path


def verify_kenlm(bin_path):
    """Verify KenLM model by scoring sample sequences."""
    try:
        import kenlm
    except ImportError:
        print("pip install kenlm to verify the model")
        return

    model = kenlm.Model(bin_path)
    print(f"\nKenLM model loaded: {bin_path}")
    print(f"  Order: {model.order}")

    # Score some test sequences
    test_sequences = [
        "DH AH SIL K AE T SIL S AE T SIL AA N SIL DH AH SIL M AE T",
        "AY SIL W AA N T SIL T UW SIL G OW SIL HH OW M",
        "Z Z Z Z Z Z Z Z",  # Should score poorly
        "SIL SIL SIL",  # Mostly silence
    ]

    print("\nSample scores (higher = more likely):")
    for seq in test_sequences:
        score = model.score(seq, bos=True, eos=True)
        print(f"  {score:8.2f}  {seq[:60]}")


def build_pyctcdecode_vocab():
    """Build vocabulary list for pyctcdecode, matching CTC output order.

    Returns list of phoneme labels where index i corresponds to CTC class i.
    The blank token is represented as '' (empty string) at the end.
    """
    vocab = []
    for i in range(N_CLASSES):
        vocab.append(CLASS_TO_ARPABET[i])
    vocab.append('')  # CTC blank at index N_CLASSES (40)
    return vocab


def main():
    parser = argparse.ArgumentParser(
        description='Build KenLM phoneme n-gram for DCoND decoding')
    parser.add_argument('--data', type=str, default='sentences_paper_256d.h5',
                        help='HDF5 data file name')
    parser.add_argument('--output', type=str,
                        default=str(DATA_DIR / 'phoneme_5gram.arpa'),
                        help='Output ARPA file path')
    parser.add_argument('--order', type=int, default=5,
                        help='N-gram order (default: 5)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'all'],
                        help='Which data split to use for LM training')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing KenLM model')
    parser.add_argument('--lm', type=str, default=None,
                        help='Path to .bin model for verification')
    args = parser.parse_args()

    if args.verify:
        lm_path = args.lm or str(DATA_DIR / 'phoneme_5gram.bin')
        verify_kenlm(lm_path)
        return

    h5_path = DATA_DIR / args.data
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract phoneme corpus
    corpus_txt = str(output_path).replace('.arpa', '_corpus.txt')
    n_seqs = extract_phoneme_corpus(h5_path, corpus_txt, split=args.split)

    # Step 2: Build KenLM model
    bin_path = build_kenlm(corpus_txt, str(output_path), order=args.order)

    # Step 3: Verify
    verify_kenlm(bin_path)

    # Step 4: Save vocab for pyctcdecode
    vocab = build_pyctcdecode_vocab()
    vocab_path = str(output_path).replace('.arpa', '_vocab.txt')
    with open(vocab_path, 'w') as f:
        for v in vocab:
            f.write((v or '<blank>') + '\n')
    print(f"\nVocab saved: {vocab_path} ({len(vocab)} tokens)")

    print(f"\n{'=' * 60}")
    print(f"KenLM {args.order}-gram phoneme model ready!")
    print(f"  ARPA: {args.output}")
    print(f"  Binary: {bin_path}")
    print(f"  Vocab: {vocab_path}")
    print(f"  Corpus: {corpus_txt} ({n_seqs} sequences)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
