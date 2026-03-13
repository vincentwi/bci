#!/usr/bin/env python3
"""KenLM-backed CTC beam search with N-best output.

Uses kenlm Python API directly with CTC prefix beam search for phoneme-level
decoding. pyctcdecode is designed for character/BPE-level CTC and concatenates
multi-character tokens, which breaks phoneme-level decoding.

Instead, we integrate KenLM scoring into the existing CTC prefix beam search
from beam_search_decode.py, providing proper phoneme-level n-gram scoring.

Usage:
    # Single decode with specific params
    python brain2speech/lead4_decode_kenlm.py \
        --model brain2speech/results/paper_replica_best.pt \
        --lm brain2speech/data/phoneme_4gram.arpa \
        --beam-width 50 --alpha 0.5 --beta 0.0

    # Sweep over hyperparameters
    python brain2speech/lead4_decode_kenlm.py \
        --model brain2speech/results/paper_replica_best.pt \
        --lm brain2speech/data/phoneme_4gram.arpa --sweep

    # Generate N-best lists for OPT rescoring
    python brain2speech/lead4_decode_kenlm.py \
        --model brain2speech/results/paper_replica_best.pt \
        --lm brain2speech/data/phoneme_4gram.arpa \
        --nbest 100 --output brain2speech/results/L4_nbest.json

References:
    - Source 9 (tbenst): beam=150 train, beam=5000 inference, lm_weight=2.0, 4-gram
    - Source 5 (Willett): beam=18, acoustic_scale=0.8, blank_penalty=log(7)
    - Source 1 (Willett repo): lmDecoderUtils.build_lm_decoder(beam=18, nbest=1)
"""
import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_TO_ARPABET, N_CLASSES

# CTC blank at index 40
CTC_BLANK = 40
NEG_INF = float('-inf')


def logsumexp(a, b):
    """Numerically stable log(exp(a) + exp(b))."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    mx = max(a, b)
    return mx + math.log(math.exp(a - mx) + math.exp(b - mx))


class PureARPAModel:
    """Pure Python ARPA n-gram reader (fallback when kenlm C++ not available).

    Reads ARPA format files and provides n-gram lookup with backoff.
    """

    def __init__(self, arpa_path):
        self.ngrams = {}  # order -> {tuple_of_words: (log10_prob, backoff)}
        self.order = 0
        self._load_arpa(arpa_path)

    def _load_arpa(self, path):
        """Parse ARPA file format."""
        with open(path, 'r') as f:
            current_order = 0
            in_data = False
            for line in f:
                line = line.strip()
                if not line or line.startswith('\\data\\'):
                    in_data = True
                    continue
                if line.startswith('\\end\\'):
                    break
                if line.startswith('\\') and line.endswith(':'):
                    # e.g. \1-grams:
                    current_order = int(line.split('-')[0][1:])
                    self.order = max(self.order, current_order)
                    if current_order not in self.ngrams:
                        self.ngrams[current_order] = {}
                    continue
                if current_order > 0 and line and not line.startswith('ngram'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        log10_prob = float(parts[0])
                        words = tuple(parts[1].split())
                        backoff = float(parts[2]) if len(parts) > 2 else 0.0
                        self.ngrams[current_order][words] = (log10_prob, backoff)

        total = sum(len(v) for v in self.ngrams.values())
        print(f"ARPA loaded: order={self.order}, {total} n-grams, path={path}")

    def score_word(self, word, context):
        """Get conditional log10 probability P(word | context) with backoff.

        Args:
            word: Single word string
            context: Tuple of preceding words

        Returns:
            log10 probability
        """
        # Try longest matching context first
        for ctx_len in range(min(len(context), self.order - 1), -1, -1):
            ctx = context[-ctx_len:] if ctx_len > 0 else ()
            ngram = ctx + (word,)
            order = len(ngram)
            if order in self.ngrams and ngram in self.ngrams[order]:
                return self.ngrams[order][ngram][0]

            # Apply backoff
            if ctx_len > 0 and len(ctx) in self.ngrams:
                ctx_entry = self.ngrams[len(ctx)].get(ctx)
                if ctx_entry is not None:
                    backoff = ctx_entry[1]
                    # Recurse with shorter context
                    shorter_ctx = ctx[1:] if len(ctx) > 1 else ()
                    return backoff + self.score_word(word, shorter_ctx)

        # Uniform fallback
        return -5.0

    def score(self, sentence, bos=True, eos=False):
        """Score a full sentence (log10 probability).

        Compatible with kenlm.Model.score() interface.
        """
        words = sentence.split()
        if not words:
            return 0.0
        total = 0.0
        for i, word in enumerate(words):
            ctx = tuple(words[:i])
            total += self.score_word(word, ctx)
        return total


class KenLMPhonemeLM:
    """KenLM-backed phoneme language model.

    Wraps kenlm Python API to provide phoneme-level scoring compatible
    with CTC prefix beam search. Falls back to pure Python ARPA reader
    if kenlm C++ library is not available.
    """

    def __init__(self, model_path):
        try:
            import kenlm
            self.model = kenlm.Model(model_path)
            self.order = self.model.order
            self._use_kenlm = True
            print(f"KenLM (C++) loaded: order={self.order}, path={model_path}")
        except ImportError:
            print("kenlm C++ not available, using pure Python ARPA reader")
            self.model = PureARPAModel(model_path)
            self.order = self.model.order
            self._use_kenlm = False

    def score(self, phoneme_idx, context_indices):
        """Score phoneme given context (for CTC beam search compatibility).

        Args:
            phoneme_idx: Integer phoneme index (0-39)
            context_indices: List of preceding phoneme indices

        Returns:
            Log probability (natural log)
        """
        # Convert indices to phoneme names
        phoneme = CLASS_TO_ARPABET.get(phoneme_idx, '')
        if not phoneme:
            return -10.0

        # Build context string from recent phonemes
        context_names = [CLASS_TO_ARPABET.get(int(c), '') for c in context_indices]
        context_names = [c for c in context_names if c]

        if self._use_kenlm:
            # KenLM: conditional via full - context scoring
            full_seq = ' '.join(context_names + [phoneme])
            ctx_seq = ' '.join(context_names) if context_names else ''

            full_score = self.model.score(full_seq, bos=(len(context_names) == 0),
                                           eos=False)
            if context_names:
                ctx_score = self.model.score(ctx_seq, bos=True, eos=False)
            else:
                ctx_score = 0.0

            cond_log10 = full_score - ctx_score
        else:
            # Pure Python ARPA: direct conditional lookup
            ctx_tuple = tuple(context_names)
            cond_log10 = self.model.score_word(phoneme, ctx_tuple)

        # Convert log10 to natural log
        return cond_log10 * math.log(10)

    def score_sequence(self, phoneme_indices):
        """Score a full phoneme sequence.

        Args:
            phoneme_indices: List of phoneme indices

        Returns:
            Total log probability (natural log)
        """
        names = [CLASS_TO_ARPABET.get(int(p), '') for p in phoneme_indices]
        names = [n for n in names if n]
        if not names:
            return -100.0
        seq = ' '.join(names)
        log10_score = self.model.score(seq, bos=True, eos=False)
        return log10_score * math.log(10)


def ctc_prefix_beam_search_kenlm(log_probs, beam_width=50, blank=CTC_BLANK,
                                   lm=None, alpha=0.5, beta=0.0, nbest=1):
    """CTC prefix beam search with KenLM language model.

    Extended version of beam_search_decode.ctc_prefix_beam_search that
    returns N-best results instead of just the top-1.

    Args:
        log_probs: (T, V) numpy array of log probabilities
        beam_width: Number of prefixes to keep
        blank: CTC blank token index
        lm: KenLMPhonemeLM instance (or None)
        alpha: LM weight
        beta: Length bonus per phoneme
        nbest: Number of results to return

    Returns:
        List of (phoneme_indices, score) tuples, sorted by score
    """
    T, V = log_probs.shape

    # Each beam: prefix (tuple) -> (p_blank, p_nonblank)
    beams = {(): (0.0, NEG_INF)}

    for t in range(T):
        new_beams = collections.defaultdict(lambda: (NEG_INF, NEG_INF))

        # Prune to top beam_width
        scored = []
        for prefix, (pb, pnb) in beams.items():
            total = logsumexp(pb, pnb)
            scored.append((total, prefix, pb, pnb))
        scored.sort(reverse=True)
        scored = scored[:beam_width]

        for _, prefix, p_blank, p_nonblank in scored:
            last = prefix[-1] if prefix else None

            # Extension with blank
            p_b_new = log_probs[t, blank] + logsumexp(p_blank, p_nonblank)
            old_pb, old_pnb = new_beams[prefix]
            new_beams[prefix] = (logsumexp(old_pb, p_b_new), old_pnb)

            # Extension with each non-blank
            for c in range(V):
                if c == blank:
                    continue

                if c == last:
                    # Same as last: repeat from blank creates new char
                    p_nb_repeat = log_probs[t, c] + p_blank
                    new_prefix = prefix + (c,)
                    old_pb2, old_pnb2 = new_beams[new_prefix]
                    lm_score = 0.0
                    if lm is not None and alpha > 0:
                        lm_score = alpha * lm.score(c, list(prefix)) + beta
                    new_beams[new_prefix] = (old_pb2,
                                             logsumexp(old_pnb2,
                                                       p_nb_repeat + lm_score))

                    # Continuation: same char, no blank between
                    p_nb_continue = log_probs[t, c] + p_nonblank
                    old_pb3, old_pnb3 = new_beams[prefix]
                    new_beams[prefix] = (old_pb3,
                                         logsumexp(old_pnb3, p_nb_continue))
                else:
                    # Different char: extend from either state
                    new_prefix = prefix + (c,)
                    p_nb_new = log_probs[t, c] + logsumexp(p_blank, p_nonblank)
                    lm_score = 0.0
                    if lm is not None and alpha > 0:
                        lm_score = alpha * lm.score(c, list(prefix)) + beta
                    old_pb2, old_pnb2 = new_beams[new_prefix]
                    new_beams[new_prefix] = (old_pb2,
                                             logsumexp(old_pnb2,
                                                       p_nb_new + lm_score))

        beams = dict(new_beams)

    # Collect N-best
    results = []
    for prefix, (pb, pnb) in beams.items():
        score = logsumexp(pb, pnb)
        results.append((list(prefix), score))
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:nbest]


def decode_single(log_probs_np, lm, beam_width=50, alpha=0.5, beta=0.0,
                   nbest=1):
    """Decode single utterance with KenLM beam search.

    Args:
        log_probs_np: (T, 41) log-probabilities
        lm: KenLMPhonemeLM or None
        beam_width: Beam width
        alpha: LM weight
        beta: Length bonus
        nbest: Number of results

    Returns:
        List of (phoneme_names_str, phoneme_indices, score) tuples
    """
    results = ctc_prefix_beam_search_kenlm(
        log_probs_np, beam_width=beam_width, lm=lm,
        alpha=alpha, beta=beta, nbest=nbest)

    output = []
    for indices, score in results:
        names = [CLASS_TO_ARPABET[i] for i in indices if i in CLASS_TO_ARPABET]
        output.append((' '.join(names), indices, score))

    return output


def create_decoder(kenlm_path, alpha=0.5, beta=0.0):
    """Create KenLM-backed decoder (returns LM object for use with beam search).

    Note: This returns a KenLMPhonemeLM, not a pyctcdecode decoder.
    Use with ctc_prefix_beam_search_kenlm() or decode_single().
    """
    if kenlm_path is None:
        return None
    return KenLMPhonemeLM(str(kenlm_path))


def get_model_log_probs(model, trials, device='cuda', batch_size=16):
    """Get CTC log-probabilities from a trained model.

    Args:
        model: Trained CTC model (BeyondPaperGRU or similar)
        trials: List of trial dicts with 'features' or 'features_raw' and 'session_idx'

    Returns:
        List of (log_probs_np, target_indices) tuples
    """
    model.eval()
    results = []

    for trial in trials:
        # Handle different feature key names
        if 'features_raw' in trial:
            features = torch.FloatTensor(trial['features_raw']).unsqueeze(0).to(device)
        elif 'features' in trial:
            features = torch.FloatTensor(trial['features']).unsqueeze(0).to(device)
        else:
            raise KeyError(f"Trial has keys: {list(trial.keys())}, need 'features' or 'features_raw'")

        session_id = torch.LongTensor([trial.get('session_idx', 0)]).to(device)

        with torch.no_grad():
            logits = model(features, session_id)  # (1, T, 41)
            log_probs = F.log_softmax(logits, dim=-1)

        log_probs_np = log_probs[0].cpu().numpy()
        target = trial.get('phoneme_indices', np.array([]))
        if hasattr(target, 'tolist'):
            target = target.tolist()

        results.append((log_probs_np, target))

    return results


def evaluate_decoder(trial_log_probs, lm, beam_width=50, alpha=0.5, beta=0.0):
    """Evaluate decoder on a set of trials.

    Args:
        trial_log_probs: List of (log_probs_np, target_indices)
        lm: KenLMPhonemeLM or None
        beam_width: Beam width
        alpha: LM weight
        beta: Length bonus

    Returns:
        Mean PER
    """
    import editdistance

    total_edits, total_len = 0, 0
    for log_probs_np, target in trial_log_probs:
        results = decode_single(log_probs_np, lm, beam_width=beam_width,
                                 alpha=alpha, beta=beta, nbest=1)
        if results:
            # results[0] = (phoneme_names_str, phoneme_indices, score)
            decoded_phones = results[0][0].split()
        else:
            decoded_phones = []

        target_phones = [CLASS_TO_ARPABET[int(t)] for t in target if int(t) in CLASS_TO_ARPABET]

        edits = editdistance.eval(decoded_phones, target_phones)
        total_edits += edits
        total_len += len(target_phones)

    per = total_edits / max(total_len, 1)
    return per


def greedy_baseline(trial_log_probs):
    """Greedy CTC decode baseline for comparison."""
    import editdistance

    total_edits, total_len = 0, 0
    for log_probs_np, target in trial_log_probs:
        best_path = log_probs_np.argmax(axis=1)
        decoded = []
        prev = -1
        for t in best_path:
            if t != prev:
                if t != CTC_BLANK:
                    decoded.append(int(t))
            prev = t

        decoded_phones = [CLASS_TO_ARPABET[d] for d in decoded if d in CLASS_TO_ARPABET]
        target_phones = [CLASS_TO_ARPABET[int(t)] for t in target if int(t) in CLASS_TO_ARPABET]

        edits = editdistance.eval(decoded_phones, target_phones)
        total_edits += edits
        total_len += len(target_phones)

    per = total_edits / max(total_len, 1)
    return per


def comprehensive_sweep(trial_log_probs, kenlm_path, output_path=None):
    """Comprehensive hyperparameter sweep.

    Sweep grid (Source 9 + Source 5 + exploration):
    - beam_width: [50, 100, 200, 500, 1000]
    - alpha (LM weight): [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    - beta (insertion bonus): [0.0, 0.5, 1.0]
    """
    results = {}
    best_per = float('inf')
    best_params = None

    # Greedy baseline
    greedy_per = greedy_baseline(trial_log_probs)
    print(f"Greedy baseline PER: {greedy_per:.4f}")
    results['greedy'] = greedy_per

    # No-LM beam search baseline
    for beam in [10, 20, 50]:
        per = evaluate_decoder(trial_log_probs, None, beam_width=beam, alpha=0.0)
        key = f'no_lm_beam{beam}'
        results[key] = per
        print(f"No LM beam={beam}: PER={per:.4f}")

    # KenLM sweep
    lm = create_decoder(kenlm_path)
    beam_widths = [10, 20, 50, 100]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    betas = [0.0, 0.5, 1.0]

    total = len(beam_widths) * len(alphas) * len(betas)
    done = 0

    for beam in beam_widths:
        for alpha in alphas:
            for beta in betas:
                per = evaluate_decoder(trial_log_probs, lm, beam_width=beam,
                                       alpha=alpha, beta=beta)
                key = f'beam{beam}_a{alpha:.1f}_b{beta:.1f}'
                results[key] = per
                done += 1

                if per < best_per:
                    best_per = per
                    best_params = (beam, alpha, beta)

                if done % 10 == 0 or per < best_per + 0.001:
                    print(f"[{done}/{total}] beam={beam:5d} α={alpha:.1f} β={beta:.1f} "
                          f"PER={per:.4f} {'*** BEST' if per == best_per else ''}")

    print(f"\n{'='*60}")
    print(f"Best: beam={best_params[0]}, α={best_params[1]:.1f}, β={best_params[2]:.1f}")
    print(f"Best PER: {best_per:.4f} (greedy: {greedy_per:.4f}, "
          f"improvement: {greedy_per - best_per:.4f})")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'greedy_per': greedy_per,
                'best_per': best_per,
                'best_params': {
                    'beam_width': best_params[0],
                    'alpha': best_params[1],
                    'beta': best_params[2],
                },
                'all_results': results,
            }, f, indent=2)
        print(f"Results saved: {output_path}")

    return results, best_params, best_per


def generate_nbest(trial_log_probs, lm, beam_width=50, alpha=0.5, beta=0.0,
                    top_n=100, output_path=None):
    """Generate N-best lists for all trials (for OPT rescoring).

    Args:
        trial_log_probs: List of (log_probs_np, target_indices)
        lm: KenLMPhonemeLM
        beam_width: Beam width
        alpha: LM weight
        beta: Length bonus
        top_n: Number of candidates per trial
        output_path: Output JSON path

    Returns:
        List of dicts with 'trial_idx', 'target', 'candidates'
    """
    all_nbest = []

    for trial_idx, (log_probs_np, target) in enumerate(trial_log_probs):
        results = decode_single(log_probs_np, lm, beam_width=beam_width,
                                 alpha=alpha, beta=beta, nbest=top_n)

        target_phones = [CLASS_TO_ARPABET[int(t)] for t in target
                         if int(t) in CLASS_TO_ARPABET]

        candidates = []
        for rank, (phoneme_str, indices, score) in enumerate(results):
            candidates.append({
                'rank': rank,
                'phonemes': phoneme_str,
                'score': float(score),
            })

        all_nbest.append({
            'trial_idx': trial_idx,
            'target_phonemes': ' '.join(target_phones),
            'n_candidates': len(candidates),
            'candidates': candidates,
        })

        if (trial_idx + 1) % 50 == 0:
            print(f"  Generated N-best for {trial_idx + 1}/{len(trial_log_probs)} trials")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(all_nbest, f, indent=2)
        print(f"N-best lists saved: {output_path} ({len(all_nbest)} trials)")

    return all_nbest


def load_model_and_data(model_path, data_path=None, eval_set='val',
                        device='cuda', max_trials=None):
    """Load model and prepare trial data.

    Tries to load using beam_search_decode.py's load_trials/load_model first,
    then falls back to train_beyond_paper.py's load_h5_dataset.
    """
    from beam_search_decode import load_trials, split_trials, load_model, get_log_probs

    if data_path is None:
        data_path = '/mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5'

    # Load trials
    trials = load_trials(data_path, max_trials=max_trials)
    train_trials, val_trials, test_trials, n_sessions, train_sids = split_trials(trials)

    if eval_set == 'val':
        eval_trials = val_trials
    elif eval_set == 'test':
        eval_trials = test_trials
    else:
        eval_trials = val_trials + test_trials

    # Load model
    model = load_model(model_path, device=device, n_sessions=n_sessions,
                       train_session_idxs=train_sids)

    # Get log probs
    trial_log_probs = get_log_probs(model, eval_trials, device=device)

    print(f"Loaded {len(eval_trials)} {eval_set} trials, model from {model_path}")
    return trial_log_probs, model, eval_trials


def main():
    parser = argparse.ArgumentParser(description='KenLM CTC Beam Search Decoder')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained CTC model checkpoint')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to H5 dataset')
    parser.add_argument('--lm', type=str, required=True,
                        help='Path to KenLM .arpa or .bin file')
    parser.add_argument('--beam-width', type=int, default=50,
                        help='Beam width (default: 50)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='LM weight (default: 0.5)')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='Insertion bonus (default: 0.0)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run comprehensive hyperparameter sweep')
    parser.add_argument('--nbest', type=int, default=0,
                        help='Generate N-best lists (0 = disabled)')
    parser.add_argument('--eval-set', type=str, default='val',
                        choices=['val', 'test', 'both'],
                        help='Evaluation set')
    parser.add_argument('--max-trials', type=int, default=None,
                        help='Max trials to load')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Load model and data
    trial_log_probs, model, eval_trials = load_model_and_data(
        args.model, args.data, args.eval_set, device, args.max_trials)

    if args.sweep:
        # Comprehensive sweep
        output = args.output or 'brain2speech/results/L4_kenlm_sweep.json'
        comprehensive_sweep(trial_log_probs, args.lm, output_path=output)

    elif args.nbest > 0:
        # Generate N-best lists
        lm = create_decoder(args.lm)
        output = args.output or 'brain2speech/results/L4_nbest.json'
        generate_nbest(trial_log_probs, lm, beam_width=args.beam_width,
                       alpha=args.alpha, beta=args.beta,
                       top_n=args.nbest, output_path=output)

    else:
        # Single evaluation
        greedy_per = greedy_baseline(trial_log_probs)
        print(f"Greedy PER: {greedy_per:.4f}")

        lm = create_decoder(args.lm)
        per = evaluate_decoder(trial_log_probs, lm, beam_width=args.beam_width,
                               alpha=args.alpha, beta=args.beta)
        print(f"KenLM beam PER: {per:.4f} (beam={args.beam_width}, "
              f"α={args.alpha}, β={args.beta})")
        print(f"Improvement: {greedy_per - per:.4f} ({(greedy_per - per)/greedy_per*100:.1f}% relative)")


if __name__ == '__main__':
    main()
