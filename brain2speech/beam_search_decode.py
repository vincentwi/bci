#!/usr/bin/env python3
"""
CTC Prefix Beam Search with Phoneme Language Model.

Implements CTC prefix beam search decoding with an optional n-gram phoneme
language model to improve PER over greedy decoding.

Usage:
    python brain2speech/beam_search_decode.py \
        --model brain2speech/results/paper_replica_best.pt \
        --beam-width 20 --lm-weight 0.5

    # Full sweep over beam widths and LM weights:
    python brain2speech/beam_search_decode.py \
        --model brain2speech/results/paper_replica_best.pt --sweep
"""
import argparse
import collections
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")

# ── Phoneme vocabulary (must match preprocess_sentences.py) ───────────
ARPABET_CLASSES = [
    'B', 'CH', 'SIL', 'D', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'ER', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'DH', 'T', 'TH',
    'V', 'W', 'Y', 'Z', 'ZH',
    'OY', 'EH', 'EY', 'UH', 'IY', 'OW', 'UW', 'IH', 'AA', 'AW',
    'AY', 'AH', 'AO', 'AE',
]
N_CLASSES = 40   # 39 phonemes + SIL
CTC_BLANK = 40   # blank index


# =====================================================================
# N-gram Phoneme Language Model
# =====================================================================

class PhonemeNgramLM:
    """Simple n-gram language model over phoneme sequences.

    Supports unigram, bigram, and trigram with Kneser-Ney-style
    backoff (simplified: absolute discounting + uniform backoff).
    """

    def __init__(self, order=3, discount=0.5):
        self.order = order
        self.discount = discount
        self.vocab_size = N_CLASSES  # 40 phonemes
        # Counts: key = tuple of context, value = Counter of next phoneme
        self.counts = {}       # {context_tuple: Counter}
        self.context_totals = {}  # {context_tuple: total_count}
        self.unigram_probs = None

    def train(self, sequences):
        """Train on a list of phoneme index sequences."""
        # Collect n-gram counts for orders 1..self.order
        unigram_counts = collections.Counter()

        for seq in sequences:
            # Prepend BOS tokens (use -1 as sentinel, won't appear in vocab)
            padded = [-1] * (self.order - 1) + list(seq)
            for i in range(self.order - 1, len(padded)):
                phoneme = padded[i]
                if phoneme < 0 or phoneme >= self.vocab_size:
                    continue
                unigram_counts[phoneme] += 1
                # Collect counts for all context lengths
                for n in range(1, self.order + 1):
                    context = tuple(padded[i - n + 1:i])
                    if context not in self.counts:
                        self.counts[context] = collections.Counter()
                    self.counts[context][phoneme] += 1

        # Compute context totals
        for ctx, counter in self.counts.items():
            self.context_totals[ctx] = sum(counter.values())

        # Unigram probabilities (with add-1 smoothing)
        total = sum(unigram_counts.values()) + self.vocab_size
        self.unigram_probs = np.zeros(self.vocab_size)
        for p in range(self.vocab_size):
            self.unigram_probs[p] = (unigram_counts.get(p, 0) + 1) / total

        n_sequences = len(sequences)
        n_ngrams = sum(self.context_totals.values())
        print(f"  LM trained: order={self.order}, "
              f"{n_sequences} sequences, {n_ngrams} n-grams, "
              f"vocab={self.vocab_size}")

    def score(self, phoneme, context):
        """Return log probability of phoneme given context (tuple of ints).

        Uses backoff: try longest context first, fall back to shorter.
        """
        for n in range(min(len(context), self.order - 1), 0, -1):
            ctx = tuple(context[-n:])
            if ctx in self.counts and self.counts[ctx].get(phoneme, 0) > 0:
                count = self.counts[ctx][phoneme]
                total = self.context_totals[ctx]
                n_types = len(self.counts[ctx])
                # Absolute discounting
                prob = max(count - self.discount, 0) / total
                # Backoff mass
                backoff_mass = (self.discount * n_types) / total
                # Interpolate with lower order
                lower = self._lower_order_prob(phoneme, ctx[1:])
                prob += backoff_mass * lower
                return math.log(prob + 1e-20)
        # Unigram fallback
        return math.log(self.unigram_probs[phoneme] + 1e-20)

    def _lower_order_prob(self, phoneme, context):
        """Recursive lower-order probability for backoff."""
        if len(context) == 0:
            return self.unigram_probs[phoneme]
        ctx = tuple(context)
        if ctx in self.counts and self.counts[ctx].get(phoneme, 0) > 0:
            count = self.counts[ctx][phoneme]
            total = self.context_totals[ctx]
            return count / total
        return self.unigram_probs[phoneme]


# =====================================================================
# CTC Prefix Beam Search
# =====================================================================

NEG_INF = float('-inf')


def logsumexp(a, b):
    """Numerically stable log(exp(a) + exp(b))."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    mx = max(a, b)
    return mx + math.log(math.exp(a - mx) + math.exp(b - mx))


def ctc_prefix_beam_search(log_probs, beam_width=10, blank=CTC_BLANK,
                           lm=None, alpha=0.0, beta=0.0):
    """CTC prefix beam search decoder.

    Follows the algorithm from Hannun (2017) "Sequence Modeling with CTC".

    Args:
        log_probs: (T, V) numpy array of log probabilities (already log_softmax'd)
        beam_width: number of prefixes to keep at each step
        blank: index of the CTC blank token
        lm: optional PhonemeNgramLM instance
        alpha: LM weight (0 = no LM)
        beta: length bonus per phoneme emitted

    Returns:
        best_seq: list of phoneme indices (decoded sequence)
    """
    T, V = log_probs.shape

    # Each beam entry: prefix (tuple) -> (p_blank, p_nonblank)
    # p_blank = log prob of all alignments ending in blank
    # p_nonblank = log prob of all alignments ending in a non-blank
    beams = {(): (0.0, NEG_INF)}  # empty prefix starts with blank prob=1

    for t in range(T):
        new_beams = collections.defaultdict(lambda: (NEG_INF, NEG_INF))

        # Prune to top beam_width candidates by total prob
        scored = []
        for prefix, (pb, pnb) in beams.items():
            total = logsumexp(pb, pnb)
            scored.append((total, prefix, pb, pnb))
        scored.sort(reverse=True)
        scored = scored[:beam_width]

        for _, prefix, p_blank, p_nonblank in scored:
            last = prefix[-1] if prefix else None

            # ── Extension with blank ──
            # Blank doesn't change the prefix, just transitions to blank state
            p_b_new = log_probs[t, blank] + logsumexp(p_blank, p_nonblank)
            old_pb, old_pnb = new_beams[prefix]
            new_beams[prefix] = (logsumexp(old_pb, p_b_new), old_pnb)

            # ── Extension with each non-blank character ──
            for c in range(V):
                if c == blank:
                    continue

                # New prefix when extending
                if c == last:
                    # Same char as last: only from blank state creates repeat
                    # From non-blank state: stays in same prefix (no new char)
                    # Case 1: repeat from blank → new char appended
                    p_nb_repeat = log_probs[t, c] + p_blank
                    new_prefix = prefix + (c,)
                    old_pb2, old_pnb2 = new_beams[new_prefix]
                    # Apply LM score for the repeated character
                    lm_score = 0.0
                    if lm is not None and alpha > 0:
                        ctx = list(prefix)
                        lm_score = alpha * lm.score(c, ctx) + beta
                    new_beams[new_prefix] = (old_pb2,
                                             logsumexp(old_pnb2,
                                                       p_nb_repeat + lm_score))

                    # Case 2: continuation (same char, no blank between)
                    # stays in same prefix
                    p_nb_continue = log_probs[t, c] + p_nonblank
                    old_pb3, old_pnb3 = new_beams[prefix]
                    new_beams[prefix] = (old_pb3,
                                         logsumexp(old_pnb3, p_nb_continue))
                else:
                    # Different char: extend prefix from either state
                    new_prefix = prefix + (c,)
                    p_nb_new = log_probs[t, c] + logsumexp(p_blank, p_nonblank)
                    # LM score
                    lm_score = 0.0
                    if lm is not None and alpha > 0:
                        ctx = list(prefix)
                        lm_score = alpha * lm.score(c, ctx) + beta
                    old_pb2, old_pnb2 = new_beams[new_prefix]
                    new_beams[new_prefix] = (old_pb2,
                                             logsumexp(old_pnb2,
                                                       p_nb_new + lm_score))

        beams = dict(new_beams)

    # Select best prefix by total log probability (with LM + length bonus)
    best_prefix = ()
    best_score = NEG_INF
    for prefix, (pb, pnb) in beams.items():
        score = logsumexp(pb, pnb)
        if score > best_score:
            best_score = score
            best_prefix = prefix

    return list(best_prefix)


# =====================================================================
# Greedy CTC decode (baseline)
# =====================================================================

def ctc_greedy_decode(log_probs, blank=CTC_BLANK):
    """Standard CTC greedy (argmax) decoding."""
    best_path = log_probs.argmax(axis=1)
    decoded = []
    prev = -1
    for t in best_path:
        if t != prev:
            if t != blank:
                decoded.append(int(t))
        prev = t
    return decoded


# =====================================================================
# PER computation
# =====================================================================

def compute_per(predicted, target):
    """Phoneme Error Rate via edit distance."""
    import editdistance
    if len(target) == 0:
        return 1.0 if len(predicted) > 0 else 0.0
    return editdistance.eval(predicted, target) / len(target)


# =====================================================================
# Data loading (mirrors train_paper_replica.py)
# =====================================================================

def causal_gaussian_smooth(features, sd_bins=2, delay_bins=8):
    """Causal Gaussian smoothing matching the paper."""
    from scipy.ndimage import convolve1d
    kernel_len = delay_bins + 4 * sd_bins + 1
    t = np.arange(kernel_len)
    center = delay_bins
    kernel = np.exp(-0.5 * ((t - center) / sd_bins) ** 2)
    kernel = kernel / kernel.sum()
    result = convolve1d(features, kernel[::-1], axis=0, mode='constant', cval=0.0)
    return result.astype(np.float32)


def stack_and_stride(features, kernel_size=1, stride=1):
    """Stack consecutive bins and stride."""
    if kernel_size == 1 and stride == 1:
        return features
    T, C = features.shape
    T_new = (T - kernel_size) // stride + 1
    if T_new <= 0:
        pad_len = kernel_size - T + stride
        features = np.pad(features, ((0, pad_len), (0, 0)), mode='edge')
        T = features.shape[0]
        T_new = (T - kernel_size) // stride + 1
    from numpy.lib.stride_tricks import as_strided
    byte_stride = features.strides
    stacked = as_strided(
        features,
        shape=(T_new, kernel_size, C),
        strides=(byte_stride[0] * stride, byte_stride[0], byte_stride[1])
    ).copy().reshape(T_new, C * kernel_size)
    return stacked


def load_trials(h5_path, smooth_sigma=2, kernel_size=1, stride=1,
                causal_smooth=True, max_trials=None):
    """Load trials from HDF5 and preprocess."""
    trials = []
    with h5py.File(h5_path, 'r') as f:
        n_trials = f.attrs['n_trials']
        if max_trials:
            n_trials = min(n_trials, max_trials)
        for i in range(n_trials):
            grp = f[f'trial_{i:05d}']
            features = grp['features'][:].astype(np.float32)

            if causal_smooth and smooth_sigma > 0:
                features = causal_gaussian_smooth(features,
                                                  sd_bins=int(smooth_sigma),
                                                  delay_bins=8)

            features_stacked = stack_and_stride(features, kernel_size, stride)

            trials.append({
                'features_stacked': features_stacked,
                'phoneme_indices': grp['phoneme_indices'][:],
                'session': grp.attrs['session'],
                'n_frames': features_stacked.shape[0],
            })
    return trials


def split_trials(trials):
    """Split into train/val/test by session (matches train_paper_replica.py)."""
    session_names = sorted(set(t['session'] for t in trials))
    session_to_idx = {s: i for i, s in enumerate(session_names)}
    n_sessions = len(session_names)

    for t in trials:
        t['session_idx'] = session_to_idx[t['session']]

    sessions = {}
    for t in trials:
        sessions.setdefault(t['session'], []).append(t)

    test_sessions = session_names[-4:]
    remaining = session_names[:-4]
    val_sessions = remaining[-2:]
    train_sessions = remaining[:-2]

    train_trials = [t for s in train_sessions for t in sessions[s]]
    val_trials = [t for s in val_sessions for t in sessions[s]]
    test_trials = [t for s in test_sessions for t in sessions[s]]

    train_session_idxs = {session_to_idx[s] for s in train_sessions}

    print(f"Sessions: {n_sessions} "
          f"({len(train_sessions)} train, {len(val_sessions)} val, "
          f"{len(test_sessions)} test)")
    print(f"Trials: {len(train_trials)} train, {len(val_trials)} val, "
          f"{len(test_trials)} test")

    return (train_trials, val_trials, test_trials,
            n_sessions, train_session_idxs)


# =====================================================================
# Model loading
# =====================================================================

def load_model(model_path, n_features_stacked=1280, n_classes=41,
               hidden=512, n_layers=5, dropout=0.4, n_sessions=24,
               bidirectional=True, activation='tanh',
               train_session_idxs=None, device='cpu'):
    """Load trained PaperGRUSimple model from checkpoint."""
    import torch
    # Import model class from train script
    sys.path.insert(0, str(Path(__file__).parent))
    from train_paper_replica import PaperGRUSimple

    model = PaperGRUSimple(
        n_features_stacked=n_features_stacked,
        n_classes=n_classes,
        hidden=hidden,
        n_layers=n_layers,
        dropout=dropout,
        n_sessions=n_sessions,
        bidirectional=bidirectional,
        activation=activation,
    )

    if train_session_idxs is not None:
        model.set_train_sessions(train_session_idxs)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    direction = 'bidirectional' if bidirectional else 'unidirectional'
    print(f"Model loaded: PaperGRUSimple ({direction}), {n_params/1e6:.1f}M params")
    return model


# =====================================================================
# Inference: extract log_probs for all trials
# =====================================================================

def get_log_probs(model, trials, device='cpu', batch_size=32):
    """Run model inference and return list of (log_probs, target) per trial."""
    import torch
    from torch.amp import autocast

    results = []
    n = len(trials)

    for start in range(0, n, batch_size):
        batch = trials[start:start + batch_size]
        features_list = [torch.FloatTensor(t['features_stacked']) for t in batch]
        lengths = [f.shape[0] for f in features_list]

        max_T = max(lengths)
        C = features_list[0].shape[1]
        padded = torch.zeros(len(batch), max_T, C)
        for i, f in enumerate(features_list):
            padded[i, :f.shape[0], :] = f

        session_ids = torch.LongTensor([t['session_idx'] for t in batch])

        padded = padded.to(device)
        session_ids = session_ids.to(device)

        with torch.no_grad():
            use_cuda = device != 'cpu' and str(device) != 'cpu'
            if use_cuda:
                with autocast("cuda"):
                    logits = model(padded, session_ids=session_ids)
            else:
                logits = model(padded, session_ids=session_ids)
            log_probs = logits.log_softmax(dim=2).cpu().numpy()

        for i, t in enumerate(batch):
            T = lengths[i]
            target = t['phoneme_indices'].tolist()
            results.append((log_probs[i, :T, :], target))

    return results


# =====================================================================
# Decode all trials with given method
# =====================================================================

def decode_all(trial_log_probs, method='greedy', beam_width=10,
               lm=None, alpha=0.0, beta=0.0, verbose=True):
    """Decode all trials and compute PER.

    Args:
        trial_log_probs: list of (log_probs_array, target_list)
        method: 'greedy' or 'beam'
        beam_width: beam width for beam search
        lm: PhonemeNgramLM instance (or None)
        alpha: LM weight
        beta: length bonus

    Returns:
        mean_per: mean phoneme error rate
        all_per: list of per-trial PER values
    """
    all_per = []
    t0 = time.time()

    for i, (log_probs, target) in enumerate(trial_log_probs):
        if method == 'greedy':
            decoded = ctc_greedy_decode(log_probs)
        else:
            decoded = ctc_prefix_beam_search(
                log_probs, beam_width=beam_width, blank=CTC_BLANK,
                lm=lm, alpha=alpha, beta=beta
            )
        per = compute_per(decoded, target)
        all_per.append(per)

        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            mean_so_far = np.mean(all_per)
            print(f"    {i+1}/{len(trial_log_probs)} trials, "
                  f"PER so far: {mean_so_far:.3f}, "
                  f"elapsed: {elapsed:.1f}s")

    mean_per = float(np.mean(all_per))
    elapsed = time.time() - t0
    return mean_per, all_per, elapsed


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CTC Prefix Beam Search with Phoneme LM')
    parser.add_argument('--model', type=str,
                        default=str(RESULTS_DIR / 'paper_replica_best.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--beam-width', type=int, default=20,
                        help='Beam width for beam search')
    parser.add_argument('--lm-weight', type=float, default=0.5,
                        help='LM weight (alpha)')
    parser.add_argument('--length-bonus', type=float, default=0.0,
                        help='Length bonus per phoneme (beta)')
    parser.add_argument('--lm-order', type=int, default=3,
                        help='N-gram order for phoneme LM (default: 3 = trigram)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run sweep over beam widths and LM weights')
    parser.add_argument('--eval-set', type=str, default='both',
                        choices=['val', 'test', 'both'],
                        help='Which set(s) to evaluate')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (-1 for CPU)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for model inference')
    parser.add_argument('--max-trials', type=int, default=None,
                        help='Limit number of trials (for debugging)')
    # Model hyperparams (must match the saved checkpoint)
    parser.add_argument('--n-features', type=int, default=1280)
    parser.add_argument('--n-classes', type=int, default=41)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--n-sessions', type=int, default=24)
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--no-bidirectional', dest='bidirectional',
                        action='store_false')
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--kernel-size', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--smooth', type=float, default=2)
    args = parser.parse_args()

    import torch

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    h5_path = DATA_DIR / 'sentences_train.h5'
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found")
        sys.exit(1)

    trials = load_trials(
        h5_path, smooth_sigma=args.smooth,
        kernel_size=args.kernel_size, stride=args.stride,
        max_trials=args.max_trials,
    )
    (train_trials, val_trials, test_trials,
     n_sessions, train_session_idxs) = split_trials(trials)

    # ── Train phoneme LM on training set ──────────────────────────────
    print(f"\n[2/4] Training {args.lm_order}-gram phoneme LM on training data...")
    train_sequences = [t['phoneme_indices'].tolist() for t in train_trials]
    lm = PhonemeNgramLM(order=args.lm_order)
    lm.train(train_sequences)

    # ── Load model ────────────────────────────────────────────────────
    print(f"\n[3/4] Loading model from {args.model}...")
    model = load_model(
        args.model,
        n_features_stacked=args.n_features,
        n_classes=args.n_classes,
        hidden=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_sessions=args.n_sessions,
        bidirectional=args.bidirectional,
        activation=args.activation,
        train_session_idxs=train_session_idxs,
        device=device,
    )

    # ── Run inference (get log_probs) ─────────────────────────────────
    print(f"\n[4/4] Running inference and decoding...")

    eval_sets = {}
    if args.eval_set in ('val', 'both'):
        print("  Computing log_probs on val set...")
        eval_sets['val'] = get_log_probs(model, val_trials, device=device,
                                         batch_size=args.batch_size)
        print(f"    {len(eval_sets['val'])} trials")

    if args.eval_set in ('test', 'both'):
        print("  Computing log_probs on test set...")
        eval_sets['test'] = get_log_probs(model, test_trials, device=device,
                                          batch_size=args.batch_size)
        print(f"    {len(eval_sets['test'])} trials")

    # ── Sweep or single run ───────────────────────────────────────────
    if args.sweep:
        beam_widths = [5, 10, 20, 50]
        lm_weights = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        length_bonuses = [0.0]

        print("\n" + "=" * 80)
        print("BEAM SEARCH SWEEP")
        print("=" * 80)

        # First: greedy baseline
        for split_name, data in eval_sets.items():
            per, _, elapsed = decode_all(data, method='greedy', verbose=False)
            print(f"\n  Greedy  | {split_name:5s} PER = {per:.4f} "
                  f"({per:.1%}) | {elapsed:.1f}s")

        # Header for results table
        results_table = []

        for split_name, data in eval_sets.items():
            print(f"\n{'─' * 70}")
            print(f"  {split_name.upper()} SET — Beam Search Results")
            print(f"{'─' * 70}")
            print(f"  {'Beam':>6s}  {'Alpha':>6s}  {'Beta':>6s}  "
                  f"{'PER':>8s}  {'Time':>8s}")
            print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}")

            for bw in beam_widths:
                for alpha in lm_weights:
                    for beta in length_bonuses:
                        per, _, elapsed = decode_all(
                            data, method='beam', beam_width=bw,
                            lm=lm if alpha > 0 else None,
                            alpha=alpha, beta=beta,
                            verbose=False)
                        print(f"  {bw:6d}  {alpha:6.2f}  {beta:6.2f}  "
                              f"{per:8.4f}  {elapsed:7.1f}s")
                        results_table.append({
                            'split': split_name, 'beam_width': bw,
                            'alpha': alpha, 'beta': beta,
                            'per': per, 'time': elapsed
                        })

        # Summary: best config per split
        print(f"\n{'=' * 70}")
        print("SUMMARY — Best configurations")
        print(f"{'=' * 70}")
        for split_name in eval_sets.keys():
            split_results = [r for r in results_table if r['split'] == split_name]
            if split_results:
                best = min(split_results, key=lambda r: r['per'])
                print(f"  {split_name:5s}: PER={best['per']:.4f} ({best['per']:.1%}) "
                      f"| beam={best['beam_width']}, "
                      f"alpha={best['alpha']:.2f}, beta={best['beta']:.2f}")

    else:
        # Single configuration run
        print("\n" + "=" * 70)
        print("DECODING RESULTS")
        print("=" * 70)

        for split_name, data in eval_sets.items():
            # Greedy
            per_greedy, _, t_greedy = decode_all(
                data, method='greedy', verbose=False)

            # Beam search without LM
            per_beam_nolm, _, t_beam_nolm = decode_all(
                data, method='beam', beam_width=args.beam_width,
                lm=None, alpha=0.0, beta=0.0, verbose=True)

            # Beam search with LM
            per_beam_lm, _, t_beam_lm = decode_all(
                data, method='beam', beam_width=args.beam_width,
                lm=lm, alpha=args.lm_weight, beta=args.length_bonus,
                verbose=True)

            print(f"\n  {split_name.upper()} SET:")
            print(f"  {'─' * 55}")
            print(f"  {'Method':<30s}  {'PER':>8s}  {'Time':>8s}")
            print(f"  {'─' * 55}")
            print(f"  {'Greedy':<30s}  {per_greedy:8.4f}  {t_greedy:7.1f}s")
            print(f"  {'Beam (w=' + str(args.beam_width) + ', no LM)':<30s}  "
                  f"{per_beam_nolm:8.4f}  {t_beam_nolm:7.1f}s")
            print(f"  {'Beam (w=' + str(args.beam_width) + ', α=' + str(args.lm_weight) + ')':<30s}  "
                  f"{per_beam_lm:8.4f}  {t_beam_lm:7.1f}s")
            print(f"  {'─' * 55}")

            # Improvement
            if per_greedy > 0:
                improv_nolm = (per_greedy - per_beam_nolm) / per_greedy * 100
                improv_lm = (per_greedy - per_beam_lm) / per_greedy * 100
                print(f"  Improvement (beam no LM): {improv_nolm:+.1f}% relative")
                print(f"  Improvement (beam + LM):  {improv_lm:+.1f}% relative")

    print("\nDone.")


if __name__ == '__main__':
    main()
