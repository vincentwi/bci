#!/usr/bin/env python3
"""Build KenLM n-gram phoneme LM from training transcripts.

Builds ARPA-format n-gram language models over phoneme sequences extracted
from the H5 dataset. Supports both with-SIL and without-SIL variants.

Since lmplz binary may not be available, also includes a fallback that
builds the LM using kenlm's Python API via a temporary ARPA file generated
by our own n-gram counter.

Usage:
    # Build 4-gram (no SIL)
    python brain2speech/lead4_build_phoneme_lm.py \
        --data /mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5 \
        --output brain2speech/data/phoneme_4gram.arpa --order 4

    # Build 5-gram with SIL tokens
    python brain2speech/lead4_build_phoneme_lm.py \
        --data /mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5 \
        --output brain2speech/data/phoneme_5gram_sil.arpa --order 5 --keep-sil

References:
    - Source 9 (tbenst): 4-gram phoneme LM with lm_weight=2.0
    - Source 1 (Willett): 5-gram WFST on Dryad
    - Source 5 (paper): 5-gram, beam=18, acoustic_scale=0.8
"""
import argparse
import collections
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_TO_ARPABET


def extract_phoneme_corpus(h5_path, keep_sil=False, split='train'):
    """Extract phoneme sequences from H5 dataset.

    Args:
        h5_path: Path to sentences_paper_256d.h5
        keep_sil: If True, keep SIL tokens (index 2) in sequences
        split: 'train', 'val', 'test', or 'all'

    Returns:
        List of phoneme name sequences (list of list of str)
    """
    sequences = []
    with h5py.File(h5_path, 'r') as f:
        n_trials = f.attrs['n_trials']
        # Use 80/10/10 split
        n_train = int(n_trials * 0.8)
        n_val = int(n_trials * 0.1)

        if split == 'train':
            trial_range = range(n_train)
        elif split == 'val':
            trial_range = range(n_train, n_train + n_val)
        elif split == 'test':
            trial_range = range(n_train + n_val, n_trials)
        else:  # 'all'
            trial_range = range(n_trials)

        for i in trial_range:
            key = f'trial_{i:05d}'
            if key not in f:
                continue
            phones = f[key]['phoneme_indices'][:]
            names = []
            for p in phones:
                p_int = int(p)
                if p_int == 2 and not keep_sil:  # SIL
                    continue
                if p_int in CLASS_TO_ARPABET:
                    names.append(CLASS_TO_ARPABET[p_int])
            if names:
                sequences.append(names)

    print(f"Extracted {len(sequences)} sequences from {split} split")
    return sequences


def write_corpus(sequences, output_path):
    """Write phoneme sequences to text file (one sentence per line)."""
    with open(output_path, 'w') as f:
        for seq in sequences:
            f.write(' '.join(seq) + '\n')
    print(f"Wrote corpus: {output_path} ({len(sequences)} lines)")
    return output_path


def build_arpa_native(sequences, order=4):
    """Build ARPA-format n-gram LM natively in Python.

    Uses modified Kneser-Ney smoothing approximation (absolute discounting
    with interpolation backoff).

    Args:
        sequences: List of phoneme name sequences
        order: n-gram order (3, 4, or 5)

    Returns:
        ARPA-format string
    """
    # Count n-grams for all orders
    ngram_counts = {}  # {order: {tuple: count}}
    for n in range(1, order + 1):
        ngram_counts[n] = collections.Counter()

    for seq in sequences:
        # Add BOS/EOS markers
        padded = ['<s>'] * (order - 1) + seq + ['</s>']
        for n in range(1, order + 1):
            for i in range(n - 1, len(padded)):
                ngram = tuple(padded[i - n + 1:i + 1])
                ngram_counts[n][ngram] += 1

    # Compute vocabulary
    vocab = set()
    for seq in sequences:
        vocab.update(seq)
    vocab.add('<s>')
    vocab.add('</s>')
    vocab = sorted(vocab)
    V = len(vocab)

    # Absolute discounting parameter
    D = 0.75

    # Build ARPA format
    lines = ['\\data\\']
    for n in range(1, order + 1):
        lines.append(f'ngram {n}={len(ngram_counts[n])}')
    lines.append('')

    for n in range(1, order + 1):
        lines.append(f'\\{n}-grams:')

        # Get context counts for this order
        if n == 1:
            total = sum(ngram_counts[1].values())
            for ngram, count in sorted(ngram_counts[n].items()):
                prob = count / total
                log_prob = math.log10(max(prob, 1e-10))
                # Backoff weight (only if not highest order)
                if n < order:
                    lines.append(f'{log_prob:.6f}\t{" ".join(ngram)}\t0.000000')
                else:
                    lines.append(f'{log_prob:.6f}\t{" ".join(ngram)}')
        else:
            # Group by context (prefix)
            context_counts = collections.Counter()
            for ngram, count in ngram_counts[n].items():
                context = ngram[:-1]
                context_counts[context] += count

            for ngram, count in sorted(ngram_counts[n].items()):
                context = ngram[:-1]
                ctx_total = context_counts[context]
                # Discounted probability
                prob = max(count - D, 0) / ctx_total
                # Add backoff mass for lower order
                n_unique = len([ng for ng in ngram_counts[n] if ng[:-1] == context])
                backoff_mass = D * n_unique / ctx_total
                # Interpolate with uniform
                prob += backoff_mass / V
                log_prob = math.log10(max(prob, 1e-10))

                if n < order:
                    lines.append(f'{log_prob:.6f}\t{" ".join(ngram)}\t0.000000')
                else:
                    lines.append(f'{log_prob:.6f}\t{" ".join(ngram)}')

        lines.append('')

    lines.append('\\end\\')
    return '\n'.join(lines)


def build_with_lmplz(corpus_path, output_arpa, order=4):
    """Build n-gram LM using KenLM's lmplz tool (if available).

    lmplz options:
        -o <order>: n-gram order
        -S <mem>: memory allocation (e.g. '80%' or '2G')
        --discount_fallback: use 0.75 discount if count < 4
    """
    lmplz = shutil.which('lmplz')
    if not lmplz:
        return False

    try:
        subprocess.run([
            lmplz, '-o', str(order),
            '-S', '2G',
            '--discount_fallback',
            '--text', str(corpus_path),
            '--arpa', str(output_arpa),
        ], check=True, capture_output=True, text=True)
        print(f"Built {order}-gram with lmplz: {output_arpa}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"lmplz failed: {e}, falling back to native builder")
        return False


def build_binary(arpa_path):
    """Convert ARPA to binary format using KenLM build_binary (if available)."""
    build_bin = shutil.which('build_binary')
    bin_path = str(arpa_path).replace('.arpa', '.bin')

    if build_bin:
        try:
            subprocess.run([
                build_bin, '-a', '22', str(arpa_path), bin_path
            ], check=True, capture_output=True, text=True)
            print(f"Built binary: {bin_path}")
            return bin_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Fallback: kenlm Python API can read ARPA directly
    print(f"build_binary not available; use ARPA file directly with kenlm")
    return str(arpa_path)


def verify_lm(model_path):
    """Verify the built LM works with kenlm Python API."""
    import kenlm
    model = kenlm.Model(model_path)
    # Test with a sample phoneme sequence
    test_seqs = [
        'AH B AW T DH AH',
        'HH EH L OW',
        'DH AH K AE T',
    ]
    print(f"\nVerification (order={model.order}):")
    for seq in test_seqs:
        score = model.score(seq, bos=True, eos=True)
        print(f"  '{seq}' → log10_prob = {score:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Build KenLM phoneme LM')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to sentences H5 file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output ARPA file path')
    parser.add_argument('--order', type=int, default=4,
                        help='N-gram order (default: 4)')
    parser.add_argument('--keep-sil', action='store_true',
                        help='Keep SIL tokens in sequences')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'all'],
                        help='Which data split to use for LM training')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify built LM')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract corpus
    sequences = extract_phoneme_corpus(args.data, keep_sil=args.keep_sil,
                                        split=args.split)

    # Write corpus text file
    corpus_path = str(output_path).replace('.arpa', '.txt')
    write_corpus(sequences, corpus_path)

    # Try lmplz first, fall back to native
    success = build_with_lmplz(corpus_path, str(output_path), order=args.order)

    if not success:
        print(f"Building {args.order}-gram natively...")
        arpa_text = build_arpa_native(sequences, order=args.order)
        with open(str(output_path), 'w') as f:
            f.write(arpa_text)
        print(f"Wrote ARPA: {output_path}")

    # Try to build binary
    model_path = build_binary(str(output_path))

    # Verify
    if args.verify:
        verify_lm(model_path)

    print(f"\nDone! LM available at: {model_path}")
    return model_path


if __name__ == '__main__':
    main()
