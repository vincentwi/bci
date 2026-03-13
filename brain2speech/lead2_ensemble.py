#!/usr/bin/env python3
"""
Lead 2: Multi-seed DCoND ensemble with log-probability averaging.

Combines multiple DCoND models (different seeds) by averaging their
diphone log-probabilities before marginalizing to monophone.

Sources:
  - arxiv 2412.17227: 10-model ensemble + fine-tuned LLM = winning formula
  - CIBR-Okubo: 2-model ensemble 8.89%+9.16% → 8.26% WER
  - DCoND paper: ensemble + OPT + GPT-3.5 = 5.77% WER

Architecture:
  Model 1 (seed 42) → logits_1 (B,T,1601) → log_softmax
  Model 2 (seed 43) → logits_2 (B,T,1601) → log_softmax
  Model 3 (seed 44) → logits_3 (B,T,1601) → log_softmax
  Model 4 (seed 45) → logits_4 (B,T,1601) → log_softmax
            ↓
  avg_lp = mean(log_softmax(logits_i))
            ↓
  mono_probs = exp(avg_lp) @ M_ext  (1601→41)
            ↓
  mono_log_probs → KenLM beam search

Usage:
    # Evaluate ensemble
    CUDA_VISIBLE_DEVICES=4,5,6,7 python brain2speech/lead2_ensemble.py \
        --checkpoints results/L2_dcond_seed42.pt,results/L2_dcond_seed43.pt,\
results/L2_dcond_seed44.pt,results/L2_dcond_seed45.pt \
        --eval-set val

    # Export ensemble log probs for downstream decode
    CUDA_VISIBLE_DEVICES=4 python brain2speech/lead2_ensemble.py \
        --checkpoints ... --export results/L2_ensemble_logprobs.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CLASS_TO_ARPABET, N_CLASSES
from train_beyond_paper import (
    load_h5_dataset, compute_per, ctc_greedy_decode,
    DATA_DIR, RESULTS_DIR, CTC_BLANK,
)
from lead2_train_dcond import (
    DCoNDDecoder, PaperExactDCoND, build_marginalization_matrix_ext,
    N_DIPHONE_CLASSES, collate_dcond,
)


class DCoNDEnsemble:
    """Ensemble of multiple DCoND models with weighted log-prob averaging.

    Key insight: DCoND models output 1601-class diphone probabilities.
    Averaging in diphone space preserves transition information that would
    be lost if averaging after marginalization.

    Steps:
      1. Each model outputs (B, T, 1601) diphone logits
      2. log_softmax each model's logits
      3. Weighted average of log-softmax'd logits
      4. Marginalize averaged probs to monophone: (B, T, 41)
      5. Feed to downstream decoder (greedy, beam search, etc.)
    """

    def __init__(self, checkpoint_paths, device='cuda', weights=None):
        self.models = []
        self.device = device

        for path in checkpoint_paths:
            print(f"  Loading: {path}")
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            model_kwargs = ckpt['model_kwargs']
            model_class = ckpt.get('model_class', 'DCoNDDecoder')

            # Create appropriate model class
            if model_class == 'PaperExactDCoND':
                model = PaperExactDCoND(**model_kwargs)
            else:
                model = DCoNDDecoder(**model_kwargs)

            # Handle DataParallel prefix
            state = ckpt['model_state_dict']
            new_state = {}
            for k, v in state.items():
                new_state[k.replace('module.', '')] = v
            model.load_state_dict(new_state)

            model.to(device).eval()
            self.models.append(model)

            val_per = ckpt.get('val_per', '?')
            seed = ckpt.get('seed', '?')
            print(f"    Seed={seed}, Val PER={val_per}")

        n = len(self.models)
        if weights is None:
            self.weights = [1.0 / n] * n
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

        print(f"Ensemble: {n} models, weights={self.weights}")

    def get_diphone_log_probs(self, features, session_ids):
        """Get weighted-average diphone log probs from ensemble.

        Args:
            features: (B, T, D) tensor on device
            session_ids: (B,) tensor on device

        Returns: (B, T_out, 1601) numpy array of averaged log probs
        """
        all_lp = []
        with torch.no_grad():
            for model in self.models:
                with autocast("cuda"):
                    logits = model(features, session_ids)
                lp = logits.log_softmax(dim=-1)
                all_lp.append(lp)

        # Weighted average of log probs
        avg_lp = torch.zeros_like(all_lp[0])
        for lp, w in zip(all_lp, self.weights):
            avg_lp += w * lp

        return avg_lp

    def get_monophone_log_probs(self, features, session_ids, M_ext):
        """Get ensemble monophone log probs by averaging at monophone level.

        For each model: softmax(diphone) → marginalize → monophone_probs
        Then average monophone_probs across models → log

        Returns: (B, T_out, 41) numpy array
        """
        avg_mono = None
        with torch.no_grad():
            for model, w in zip(self.models, self.weights):
                with autocast("cuda"):
                    logits = model(features, session_ids)
                # Marginalize each model independently
                probs = logits.float().softmax(dim=-1).cpu()  # (B, T_out, 1601)
                mono_probs = probs @ M_ext  # (B, T_out, 41)
                if avg_mono is None:
                    avg_mono = w * mono_probs
                else:
                    avg_mono += w * mono_probs
        return (avg_mono + 1e-10).log().numpy()

    def get_all_monophone_log_probs(self, trials, M_ext, device,
                                     batch_size=16):
        """Extract monophone log probs for all trials.

        Returns: list of (log_probs_np, target_indices_list)
        """
        results = []

        # Get kernel/stride from first model
        model0 = self.models[0]
        kernel_size = model0.kernel_size
        stride = model0.stride

        for i in range(0, len(trials), batch_size):
            batch = trials[i:i + batch_size]
            features = [torch.FloatTensor(t['features_raw']) for t in batch]
            feat_lens = [f.shape[0] for f in features]

            max_T = max(feat_lens)
            C = features[0].shape[1]
            padded = torch.zeros(len(batch), max_T, C)
            for j, f in enumerate(features):
                padded[j, :f.shape[0], :] = f

            session_ids = torch.LongTensor(
                [t.get('session_idx', 0) for t in batch])

            padded = padded.to(device)
            session_ids = session_ids.to(device)

            mono_lp = self.get_monophone_log_probs(padded, session_ids, M_ext)

            for j, t in enumerate(batch):
                T_raw = feat_lens[j]
                T_out = max(1, (T_raw - kernel_size) // stride + 1)
                T_out = min(T_out, mono_lp.shape[1])
                target = t['phoneme_indices'].tolist()
                results.append((mono_lp[j, :T_out, :], target))

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Lead 2: DCoND Ensemble')
    parser.add_argument('--checkpoints', type=str, required=True,
                        help='Comma-separated paths to DCoND checkpoints')
    parser.add_argument('--weights', type=str, default=None,
                        help='Comma-separated weights (default: equal)')
    parser.add_argument('--data', type=str, default='sentences_paper_256d.h5')
    parser.add_argument('--eval-set', choices=['val', 'test', 'both'],
                        default='val')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--export', type=str, default=None,
                        help='Export ensemble log probs to .npz')
    parser.add_argument('--max-trials', type=int, default=None)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')

    # Parse checkpoints
    checkpoint_paths = [p.strip() for p in args.checkpoints.split(',')]
    weights = None
    if args.weights:
        weights = [float(w) for w in args.weights.split(',')]

    # Load ensemble
    print(f"Loading {len(checkpoint_paths)}-model ensemble...")
    ensemble = DCoNDEnsemble(checkpoint_paths, device=device, weights=weights)

    M_ext = build_marginalization_matrix_ext()  # (1601, 41)

    # Load data
    h5_path = DATA_DIR / args.data
    print(f"\nLoading data: {h5_path}")
    trials = load_h5_dataset(h5_path, max_trials=args.max_trials, load_raw=True)

    # Session split
    session_names = sorted(set(t['session'] for t in trials))
    session_to_idx = {s: i for i, s in enumerate(session_names)}
    for t in trials:
        t['session_idx'] = session_to_idx[t['session']]

    sessions = {}
    for t in trials:
        sessions.setdefault(t['session'], []).append(t)

    test_sessions = session_names[-4:]
    remaining = session_names[:-4]
    val_sessions = remaining[-2:]
    train_sessions = remaining[:-2]

    # Set train sessions so val/test use shared weights
    train_sids = {session_to_idx[s] for s in train_sessions}
    for model in ensemble.models:
        model.set_train_sessions(train_sids)

    eval_sets = {}
    if args.eval_set in ('val', 'both'):
        eval_sets['val'] = [t for s in val_sessions for t in sessions[s]]
    if args.eval_set in ('test', 'both'):
        eval_sets['test'] = [t for s in test_sessions for t in sessions[s]]

    # Also evaluate individual models for comparison
    print(f"\n{'=' * 70}")
    print("ENSEMBLE vs INDIVIDUAL MODELS")
    print(f"{'=' * 70}")

    for split_name, trials_list in eval_sets.items():
        print(f"\n{split_name.upper()} SET ({len(trials_list)} trials)")

        # Individual model PERs
        for model_idx, model in enumerate(ensemble.models):
            pers = []
            for i in range(0, len(trials_list), args.batch_size):
                batch = trials_list[i:i + args.batch_size]
                features = [torch.FloatTensor(t['features_raw']) for t in batch]
                feat_lens = [f.shape[0] for f in features]

                max_T = max(feat_lens)
                C = features[0].shape[1]
                padded = torch.zeros(len(batch), max_T, C).to(device)
                for j, f in enumerate(features):
                    padded[j, :f.shape[0], :] = f

                session_ids = torch.LongTensor(
                    [t.get('session_idx', 0) for t in batch]).to(device)

                with torch.no_grad():
                    with autocast("cuda"):
                        logits = model(padded, session_ids)
                    probs = logits.float().softmax(dim=-1).cpu()
                    mono_probs = probs @ M_ext
                    mono_lp = (mono_probs + 1e-10).log().numpy()

                    for j, t in enumerate(batch):
                        T_raw = feat_lens[j]
                        T_out = max(1, (T_raw - model.kernel_size) //
                                    model.stride + 1)
                        T_out = min(T_out, mono_lp.shape[1])
                        decoded = ctc_greedy_decode(mono_lp[j, :T_out],
                                                    blank=CTC_BLANK)
                        target = t['phoneme_indices'].tolist()
                        pers.append(compute_per(decoded, target))

            print(f"  Model {model_idx}: PER = {np.mean(pers):.4f} "
                  f"({np.mean(pers):.1%})")

        # Ensemble PER
        t0 = time.time()
        all_results = ensemble.get_all_monophone_log_probs(
            trials_list, M_ext, device, batch_size=args.batch_size)

        pers = []
        for lp, target in all_results:
            decoded = ctc_greedy_decode(lp, blank=CTC_BLANK)
            pers.append(compute_per(decoded, target))

        elapsed = time.time() - t0
        print(f"  ENSEMBLE: PER = {np.mean(pers):.4f} "
              f"({np.mean(pers):.1%}) [{elapsed:.1f}s]")

        # Export if requested
        if args.export:
            export_data = {
                'log_probs': [lp for lp, _ in all_results],
                'targets': [t for _, t in all_results],
            }
            np.savez_compressed(args.export, **{
                f'lp_{i}': lp for i, (lp, _) in enumerate(all_results)
            }, **{
                f'tgt_{i}': np.array(t) for i, (_, t) in enumerate(all_results)
            })
            print(f"\nExported: {args.export}")

    # Save ensemble results
    results = {
        'n_models': len(checkpoint_paths),
        'checkpoints': checkpoint_paths,
        'weights': ensemble.weights,
    }
    results_path = RESULTS_DIR / 'L2_ensemble_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {results_path}")


if __name__ == '__main__':
    main()
