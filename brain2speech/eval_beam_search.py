#!/usr/bin/env python3
"""
Beam search decoding evaluation for EnhancedGRU (lead1_train_gru_v2) checkpoints.

Loads a trained checkpoint, runs inference on val/test sets, then applies:
1. Greedy decoding (baseline)
2. Beam search without LM
3. Beam search with trigram phoneme LM

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u brain2speech/eval_beam_search.py \
      --checkpoint brain2speech/results/lead1/L1.3_cibr_withinday_best.pt \
      --sweep --gpu 0

    # Ensemble beam search (average log probs from multiple checkpoints):
    CUDA_VISIBLE_DEVICES=2 python -u brain2speech/eval_beam_search.py \
      --checkpoint brain2speech/results/lead1/L1.3_cibr_withinday_best.pt \
                   brain2speech/results/lead1/L1.7b_seed123_best.pt \
                   brain2speech/results/lead1/L1.7b_seed456_best.pt \
                   brain2speech/results/lead1/L1.7b_seed789_best.pt \
      --sweep --gpu 0
"""
import argparse
import collections
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.amp import autocast

# Import from lead1 training script
sys.path.insert(0, str(Path(__file__).parent))
from lead1_train_gru_v2 import (
    EnhancedGRU, causal_gaussian_smooth, load_h5_raw, collate_raw,
    N_CLASSES, CTC_BLANK, N_OUTPUT,
)
from beam_search_decode import (
    PhonemeNgramLM, ctc_prefix_beam_search, ctc_greedy_decode, compute_per,
)

DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")


def load_checkpoint_and_model(ckpt_path, device):
    """Load EnhancedGRU from lead1 checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp = ckpt['hyperparams']

    model = EnhancedGRU(
        input_dim=hp.get('input_dim', 256),
        n_classes=N_OUTPUT,
        hidden=hp['hidden'],
        n_layers=hp['n_layers'],
        dropout=hp['dropout'],
        n_sessions=ckpt['n_sessions'],
        kernel_size=hp['kernel_size'],
        stride=hp['stride'],
        bidirectional=hp['bidirectional'],
        day_hidden=hp.get('day_hidden', 256),
        use_post_rnn_head=hp.get('use_post_rnn_head', False),
        head_dim=hp.get('head_dim', 256),
        use_speckled_mask=hp.get('use_speckled_mask', False),
        speckled_p=hp.get('speckled_p', 0.3),
        ortho_init=False,  # Already initialized from checkpoint
        return_hidden=False,
    )

    train_sids = set(ckpt.get('train_session_idxs', range(ckpt['n_sessions'])))
    model.set_train_sessions(train_sids)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    bidir = 'BiGRU' if hp['bidirectional'] else 'UniGRU'
    print(f"  Loaded {ckpt_path}: {bidir} h={hp['hidden']} k={hp['kernel_size']} "
          f"({n_params/1e6:.1f}M), val={ckpt.get('best_val_per', '?'):.1%}, "
          f"test={ckpt.get('test_per', '?'):.1%}")

    return model, hp, ckpt


def get_log_probs_enhanced(model, trials, device, hp, batch_size=64):
    """Run inference with EnhancedGRU and return per-trial log_probs."""
    kernel_size = hp['kernel_size']
    stride = hp['stride']
    results = []

    for start in range(0, len(trials), batch_size):
        batch = trials[start:start + batch_size]
        features_list = [torch.FloatTensor(t['features_raw']) for t in batch]
        raw_lengths = [f.shape[0] for f in features_list]

        max_T = max(raw_lengths)
        C = features_list[0].shape[1]
        padded = torch.zeros(len(batch), max_T, C)
        for i, f in enumerate(features_list):
            padded[i, :f.shape[0], :] = f

        session_ids = torch.LongTensor([t['session_idx'] for t in batch])
        padded = padded.to(device)
        session_ids = session_ids.to(device)

        with torch.no_grad():
            with autocast("cuda"):
                logits = model(padded, session_ids)
            log_probs = logits.log_softmax(dim=2).float().cpu().numpy()

        for i, t in enumerate(batch):
            T_raw = raw_lengths[i]
            T_out = max(1, (T_raw - kernel_size) // stride + 1)
            T_out = min(T_out, log_probs.shape[1])
            target = t['phoneme_indices'].tolist()
            results.append((log_probs[i, :T_out, :], target))

    return results


def decode_all(trial_log_probs, method='greedy', beam_width=10,
               lm=None, alpha=0.0, beta=0.0, verbose=False):
    """Decode all trials and compute PER."""
    all_per = []
    t0 = time.time()

    for i, (log_probs, target) in enumerate(trial_log_probs):
        if method == 'greedy':
            decoded = ctc_greedy_decode(log_probs)
        else:
            decoded = ctc_prefix_beam_search(
                log_probs, beam_width=beam_width, blank=CTC_BLANK,
                lm=lm, alpha=alpha, beta=beta)
        per = compute_per(decoded, target)
        all_per.append(per)

        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(trial_log_probs)}, PER={np.mean(all_per):.4f}, "
                  f"{elapsed:.1f}s")

    mean_per = float(np.mean(all_per))
    elapsed = time.time() - t0
    return mean_per, all_per, elapsed


def main():
    parser = argparse.ArgumentParser(description='Beam search decoding for EnhancedGRU')
    parser.add_argument('--checkpoint', nargs='+', required=True,
                        help='Path(s) to model checkpoint(s). Multiple = ensemble.')
    parser.add_argument('--data', type=str,
                        default=str(DATA_DIR / 'sentences_paper_256d.h5'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-set', type=str, default='both',
                        choices=['val', 'test', 'both'])
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep beam widths and LM weights')
    parser.add_argument('--beam-width', type=int, default=25)
    parser.add_argument('--lm-weight', type=float, default=0.5)
    parser.add_argument('--lm-order', type=int, default=3)
    parser.add_argument('--length-bonus', type=float, default=0.0)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    ensemble_mode = len(args.checkpoint) > 1

    # ── Load data ──
    print("\n[1] Loading data...")
    # Read first checkpoint to get hyperparams for data loading
    first_ckpt = torch.load(args.checkpoint[0], map_location='cpu', weights_only=False)
    first_hp = first_ckpt['hyperparams']
    kernel_size = first_hp['kernel_size']
    stride = first_hp['stride']
    min_frames = kernel_size + stride

    trials = load_h5_raw(
        args.data, smooth_sigma=2, causal_smooth=True, min_frames=min_frames)

    # ── Within-day split (same as training, fixed seed=42) ──
    session_names = sorted(set(t['session'] for t in trials))
    session_to_idx = {s: i for i, s in enumerate(session_names)}
    n_sessions = len(session_names)
    for t in trials:
        t['session_idx'] = session_to_idx[t['session']]

    sessions_dict = {}
    for t in trials:
        sessions_dict.setdefault(t['session'], []).append(t)

    rng = np.random.RandomState(42)
    train_trials, val_trials, test_trials = [], [], []
    for s_name in session_names:
        s_trials = sessions_dict[s_name]
        rng.shuffle(s_trials)
        n_test = 40
        n_val = max(1, int(len(s_trials) * args.val_ratio))
        test_trials.extend(s_trials[:n_test])
        val_trials.extend(s_trials[n_test:n_test + n_val])
        train_trials.extend(s_trials[n_test + n_val:])

    print(f"  Trials: {len(train_trials)} train, {len(val_trials)} val, {len(test_trials)} test")

    # ── Train phoneme LM ──
    print(f"\n[2] Training {args.lm_order}-gram phoneme LM...")
    train_sequences = [t['phoneme_indices'].tolist() for t in train_trials]
    lm = PhonemeNgramLM(order=args.lm_order)
    lm.train(train_sequences)

    # ── Load model(s) ──
    print(f"\n[3] Loading {len(args.checkpoint)} model(s)...")
    models = []
    hps = []
    for ckpt_path in args.checkpoint:
        model, hp, ckpt = load_checkpoint_and_model(ckpt_path, device)
        models.append(model)
        hps.append(hp)

    # ── Get log probs ──
    print(f"\n[4] Running inference...")
    eval_sets = {}

    for split_name, split_trials in [('val', val_trials), ('test', test_trials)]:
        if args.eval_set not in (split_name, 'both'):
            continue

        print(f"  {split_name}: {len(split_trials)} trials")

        if ensemble_mode:
            # Average log probs across models
            all_model_probs = []
            for i, (model, hp) in enumerate(zip(models, hps)):
                print(f"    Model {i+1}/{len(models)}...")
                probs = get_log_probs_enhanced(model, split_trials, device, hp,
                                                batch_size=args.batch_size)
                all_model_probs.append(probs)

            # Average log probs
            ensemble_probs = []
            for trial_idx in range(len(split_trials)):
                target = all_model_probs[0][trial_idx][1]
                # Average in log space (geometric mean of probabilities)
                log_probs_list = [all_model_probs[m][trial_idx][0]
                                  for m in range(len(models))]
                # Align lengths (take min)
                min_T = min(lp.shape[0] for lp in log_probs_list)
                avg_lp = np.mean([lp[:min_T] for lp in log_probs_list], axis=0)
                ensemble_probs.append((avg_lp, target))
            eval_sets[split_name] = ensemble_probs
        else:
            eval_sets[split_name] = get_log_probs_enhanced(
                models[0], split_trials, device, hps[0],
                batch_size=args.batch_size)

    # ── Decode ──
    print(f"\n{'='*75}")
    if ensemble_mode:
        print(f"BEAM SEARCH DECODING — {len(models)}-MODEL ENSEMBLE")
    else:
        ckpt_name = Path(args.checkpoint[0]).stem
        print(f"BEAM SEARCH DECODING — {ckpt_name}")
    print(f"{'='*75}")

    if args.sweep:
        beam_widths = [5, 10, 25, 50, 100]
        lm_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        length_bonuses = [0.0, 0.3, 0.5]

        for split_name, data in eval_sets.items():
            # Greedy baseline
            per_greedy, _, t_greedy = decode_all(data, method='greedy')
            print(f"\n  {split_name.upper()}: Greedy PER = {per_greedy:.4f} ({per_greedy:.1%}) [{t_greedy:.1f}s]")

            print(f"\n  {'Beam':>6} {'Alpha':>6} {'Beta':>5} {'PER':>8} {'%':>6} {'Improv':>8} {'Time':>7}")
            print(f"  {'─'*6} {'─'*6} {'─'*5} {'─'*8} {'─'*6} {'─'*8} {'─'*7}")

            best_per = per_greedy
            best_cfg = None

            for bw in beam_widths:
                for alpha in lm_weights:
                    for beta in length_bonuses:
                        if alpha == 0 and beta > 0:
                            continue  # Skip meaningless combos
                        per, _, elapsed = decode_all(
                            data, method='beam', beam_width=bw,
                            lm=lm if alpha > 0 else None,
                            alpha=alpha, beta=beta)
                        improv = (per_greedy - per) / per_greedy * 100
                        marker = ' ***' if per < best_per else ''
                        print(f"  {bw:6d} {alpha:6.2f} {beta:5.2f} "
                              f"{per:8.4f} {per:5.1%} {improv:+7.1f}% "
                              f"{elapsed:6.1f}s{marker}")
                        if per < best_per:
                            best_per = per
                            best_cfg = (bw, alpha, beta)

            print(f"\n  BEST {split_name}: {best_per:.4f} ({best_per:.1%})")
            if best_cfg:
                print(f"    beam={best_cfg[0]}, alpha={best_cfg[1]}, beta={best_cfg[2]}")
                print(f"    Improvement: {per_greedy:.1%} → {best_per:.1%} "
                      f"({(per_greedy - best_per)/per_greedy*100:+.1f}% relative)")
    else:
        for split_name, data in eval_sets.items():
            per_greedy, _, t_greedy = decode_all(data, method='greedy')
            per_beam, _, t_beam = decode_all(
                data, method='beam', beam_width=args.beam_width,
                lm=lm, alpha=args.lm_weight, beta=args.length_bonus,
                verbose=True)

            print(f"\n  {split_name.upper()}:")
            print(f"    Greedy:                    {per_greedy:.4f} ({per_greedy:.1%}) [{t_greedy:.1f}s]")
            print(f"    Beam(w={args.beam_width}, α={args.lm_weight}): "
                  f"{per_beam:.4f} ({per_beam:.1%}) [{t_beam:.1f}s]")
            improv = (per_greedy - per_beam) / per_greedy * 100
            print(f"    Improvement: {improv:+.1f}% relative")

    print("\nDone.")


if __name__ == '__main__':
    main()
