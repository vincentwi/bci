#!/usr/bin/env python3
"""
Lead 2: KenLM-backed CTC beam search decoder for DCoND models.

Decodes DCoND model output using pyctcdecode with KenLM phoneme LM.
Supports N-best output for downstream OPT/LLM rescoring.

Key parameters (from tbenst/silent_speech):
  - beam_width: 150 (train) / 5000 (inference)
  - lm_weight: 2.0
  Source: https://github.com/tbenst/silent_speech

Sweep axes:
  - beam_width: [50, 100, 150, 200, 500, 1000, 5000]
  - lm_weight: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

Usage:
    # Single model decode with sweep
    CUDA_VISIBLE_DEVICES=6 python brain2speech/lead2_decode_kenlm.py \
        --model brain2speech/results/L2_dcond_best.pt \
        --lm brain2speech/data/phoneme_5gram.bin \
        --sweep

    # N-best output for rescoring
    CUDA_VISIBLE_DEVICES=6 python brain2speech/lead2_decode_kenlm.py \
        --model brain2speech/results/L2_dcond_best.pt \
        --lm brain2speech/data/phoneme_5gram.bin \
        --beam-width 5000 --lm-weight 2.0 --top-n 100 \
        --output brain2speech/results/L2_kenlm_nbest.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CLASS_TO_ARPABET, N_CLASSES
from train_beyond_paper import (
    load_h5_dataset, compute_per, ctc_greedy_decode,
    DATA_DIR, RESULTS_DIR, CTC_BLANK,
)
from lead2_train_dcond import (
    DCoNDDecoder, PaperExactDCoND, build_marginalization_matrix_ext,
    N_DIPHONE_CLASSES, N_MONO, collate_dcond,
)


def build_vocab():
    """Build vocabulary list for pyctcdecode.

    Returns list where index i is the label for CTC class i.
    Blank is '' at index N_CLASSES (40).
    """
    vocab = []
    for i in range(N_CLASSES):
        vocab.append(CLASS_TO_ARPABET[i])
    vocab.append('')  # blank
    return vocab


def load_dcond_model(checkpoint_path, device='cuda'):
    """Load DCoND model from cross-lead compatible checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model_kwargs = ckpt['model_kwargs']
    model_class = ckpt.get('model_class', 'DCoNDDecoder')

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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_class}: {n_params / 1e6:.1f}M params")
    print(f"  Hidden: {model_kwargs['hidden']}, Layers: {model_kwargs['n_layers']}")
    print(f"  Kernel: {model_kwargs['kernel_size']}, Stride: {model_kwargs['stride']}")
    return model, model_kwargs


def get_monophone_log_probs(model, features, session_ids, M_ext, device):
    """Run DCoND model, marginalize diphone→monophone, return log probs.

    Args:
        model: DCoNDDecoder
        features: (B, T, D) tensor
        session_ids: (B,) tensor
        M_ext: (1601, 41) marginalization matrix

    Returns: (B, T_out, 41) numpy array of monophone log probs
    """
    from torch.amp import autocast

    with torch.no_grad():
        features = features.to(device)
        session_ids = session_ids.to(device)
        with autocast("cuda"):
            logits = model(features, session_ids)  # (B, T_out, 1601)

        # Marginalize to monophone
        probs = logits.float().softmax(dim=-1).cpu()  # (B, T_out, 1601)
        mono_probs = probs @ M_ext  # (B, T_out, 41)
        mono_log_probs = (mono_probs + 1e-10).log().numpy()

    return mono_log_probs


def decode_with_kenlm(log_probs, decoder, beam_width=150):
    """Decode single trial with pyctcdecode KenLM beam search.

    Args:
        log_probs: (T, 41) numpy array of monophone log probs
        decoder: pyctcdecode BeamSearchDecoderCTC
        beam_width: beam width

    Returns: decoded phoneme string
    """
    # pyctcdecode expects (T, V) probs (NOT log probs)
    probs = np.exp(log_probs)
    result = decoder.decode(probs, beam_width=beam_width)
    return result


def decode_nbest_kenlm(log_probs, decoder, beam_width=5000, top_n=100):
    """Decode single trial returning N-best list.

    Returns: list of (phoneme_string, score) tuples
    """
    probs = np.exp(log_probs)
    beams = decoder.decode_beams(probs, beam_width=beam_width)
    results = []
    for beam in beams[:top_n]:
        text = beam[0]  # decoded string
        # beam format: (text, frames, logit_score, lm_score)
        logit_score = beam[2] if len(beam) > 2 else 0.0
        lm_score = beam[3] if len(beam) > 3 else 0.0
        results.append({
            'text': text,
            'logit_score': float(logit_score),
            'lm_score': float(lm_score),
            'combined_score': float(logit_score + lm_score),
        })
    return results


def phoneme_string_to_indices(phoneme_str):
    """Convert space-separated phoneme string back to class indices."""
    from config import ARPABET_TO_CLASS
    if not phoneme_str.strip():
        return []
    tokens = phoneme_str.strip().split()
    indices = []
    for tok in tokens:
        if tok in ARPABET_TO_CLASS:
            indices.append(ARPABET_TO_CLASS[tok])
    return indices


def main():
    parser = argparse.ArgumentParser(
        description='Lead 2: KenLM CTC beam search for DCoND')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to DCoND checkpoint (.pt)')
    parser.add_argument('--lm', type=str,
                        default=str(DATA_DIR / 'phoneme_5gram.bin'),
                        help='Path to KenLM binary model')
    parser.add_argument('--data', type=str, default='sentences_paper_256d.h5',
                        help='HDF5 data file')

    # Decode settings
    parser.add_argument('--beam-width', type=int, default=150)
    parser.add_argument('--lm-weight', type=float, default=2.0,
                        help='LM weight (alpha in pyctcdecode)')
    parser.add_argument('--word-score', type=float, default=0.0,
                        help='Word insertion bonus (beta in pyctcdecode)')
    parser.add_argument('--top-n', type=int, default=1,
                        help='N-best output (1 = single best)')

    # Evaluation
    parser.add_argument('--eval-set', choices=['val', 'test', 'both'],
                        default='val')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep beam width and LM weight')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON for N-best results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-trials', type=int, default=None)
    args = parser.parse_args()

    # Import pyctcdecode
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        print("ERROR: pip install pyctcdecode kenlm")
        sys.exit(1)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')

    # Load model
    print(f"Loading model: {args.model}")
    model, model_kwargs = load_dcond_model(args.model, device)
    M_ext = build_marginalization_matrix_ext()  # (1601, 41)

    kernel_size = model_kwargs['kernel_size']
    stride = model_kwargs['stride']

    # Load data
    h5_path = DATA_DIR / args.data
    print(f"\nLoading data: {h5_path}")
    trials = load_h5_dataset(h5_path, max_trials=args.max_trials,
                             load_raw=True)

    # Split
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

    # Set train sessions for day-specific weights
    train_sids = {session_to_idx[s] for s in train_sessions}
    model.set_train_sessions(train_sids)

    eval_sets = {}
    if args.eval_set in ('val', 'both'):
        eval_sets['val'] = [t for s in val_sessions for t in sessions[s]]
    if args.eval_set in ('test', 'both'):
        eval_sets['test'] = [t for s in test_sessions for t in sessions[s]]

    for name, trials_list in eval_sets.items():
        print(f"  {name}: {len(trials_list)} trials")

    # Build pyctcdecode decoder
    vocab = build_vocab()
    print(f"\nBuilding pyctcdecode decoder...")
    print(f"  Vocab: {len(vocab)} tokens (40 phonemes + blank)")
    print(f"  KenLM: {args.lm}")

    # Get monophone log probs for all eval trials
    print(f"\nExtracting monophone log probs...")
    all_log_probs = {}  # {split_name: [(log_probs, target_indices), ...]}

    for split_name, trials_list in eval_sets.items():
        split_results = []
        for i in range(0, len(trials_list), args.batch_size):
            batch = trials_list[i:i + args.batch_size]
            features = [torch.FloatTensor(t['features_raw']) for t in batch]
            feat_lens = [f.shape[0] for f in features]

            max_T = max(feat_lens)
            C = features[0].shape[1]
            padded = torch.zeros(len(batch), max_T, C)
            for j, f in enumerate(features):
                padded[j, :f.shape[0], :] = f

            session_ids = torch.LongTensor(
                [t.get('session_idx', 0) for t in batch])

            mono_lp = get_monophone_log_probs(
                model, padded, session_ids, M_ext, device)

            for j, t in enumerate(batch):
                T_raw = feat_lens[j]
                T_out = max(1, (T_raw - kernel_size) // stride + 1)
                T_out = min(T_out, mono_lp.shape[1])
                target = t['phoneme_indices'].tolist()
                split_results.append((mono_lp[j, :T_out, :], target))

        all_log_probs[split_name] = split_results
        print(f"  {split_name}: {len(split_results)} trials extracted")

    if args.sweep:
        beam_widths = [50, 100, 150, 200, 500]
        lm_weights = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

        print(f"\n{'=' * 80}")
        print("KENLM BEAM SEARCH SWEEP")
        print(f"{'=' * 80}")

        # Greedy baseline first
        for split_name, data in all_log_probs.items():
            pers = []
            for lp, target in data:
                decoded = ctc_greedy_decode(lp, blank=CTC_BLANK)
                pers.append(compute_per(decoded, target))
            print(f"\n  Greedy | {split_name} PER = {np.mean(pers):.4f} "
                  f"({np.mean(pers):.1%})")

        results_table = []

        for split_name, data in all_log_probs.items():
            print(f"\n{'─' * 70}")
            print(f"  {split_name.upper()} — KenLM Beam Search")
            print(f"  {'Beam':>6}  {'LM_wt':>6}  {'PER':>8}  {'Time':>8}")
            print(f"  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}")

            for lm_weight in lm_weights:
                # Build decoder with this LM weight
                if lm_weight > 0 and os.path.exists(args.lm):
                    decoder = build_ctcdecoder(
                        labels=vocab,
                        kenlm_model_path=args.lm,
                        alpha=lm_weight,
                        beta=args.word_score,
                    )
                else:
                    decoder = build_ctcdecoder(labels=vocab)

                for bw in beam_widths:
                    if lm_weight == 0 and bw > 100:
                        continue  # Skip large beams without LM

                    t0 = time.time()
                    pers = []
                    for lp, target in data:
                        result = decoder.decode(np.exp(lp), beam_width=bw)
                        decoded = phoneme_string_to_indices(result)
                        pers.append(compute_per(decoded, target))
                    elapsed = time.time() - t0
                    mean_per = np.mean(pers)
                    print(f"  {bw:6d}  {lm_weight:6.1f}  "
                          f"{mean_per:8.4f}  {elapsed:7.1f}s")

                    results_table.append({
                        'split': split_name, 'beam_width': bw,
                        'lm_weight': lm_weight, 'per': float(mean_per),
                        'time': elapsed,
                    })

        # Summary
        print(f"\n{'=' * 70}")
        print("BEST CONFIGURATIONS")
        for split_name in all_log_probs.keys():
            split_r = [r for r in results_table if r['split'] == split_name]
            if split_r:
                best = min(split_r, key=lambda x: x['per'])
                print(f"  {split_name}: PER={best['per']:.4f} ({best['per']:.1%}) "
                      f"| beam={best['beam_width']}, lm_wt={best['lm_weight']:.1f}")

        # Save sweep results
        sweep_path = RESULTS_DIR / 'L2_kenlm_sweep.json'
        with open(sweep_path, 'w') as f:
            json.dump(results_table, f, indent=2)
        print(f"\nSweep results: {sweep_path}")

    else:
        # Single decode run
        print(f"\nDecoding: beam={args.beam_width}, lm_weight={args.lm_weight}")

        if os.path.exists(args.lm):
            decoder = build_ctcdecoder(
                labels=vocab,
                kenlm_model_path=args.lm,
                alpha=args.lm_weight,
                beta=args.word_score,
            )
        else:
            print(f"WARNING: LM not found at {args.lm}, using no LM")
            decoder = build_ctcdecoder(labels=vocab)

        all_nbest = {}

        for split_name, data in all_log_probs.items():
            t0 = time.time()
            pers = []
            nbest_list = []

            for idx, (lp, target) in enumerate(data):
                if args.top_n > 1:
                    # N-best decode
                    beams = decode_nbest_kenlm(
                        lp, decoder,
                        beam_width=args.beam_width,
                        top_n=args.top_n)
                    best_text = beams[0]['text'] if beams else ''
                    decoded = phoneme_string_to_indices(best_text)
                    nbest_list.append({
                        'trial_idx': idx,
                        'target': target,
                        'nbest': beams,
                    })
                else:
                    result = decoder.decode(np.exp(lp),
                                            beam_width=args.beam_width)
                    decoded = phoneme_string_to_indices(result)

                pers.append(compute_per(decoded, target))

                if (idx + 1) % 200 == 0:
                    print(f"    {idx+1}/{len(data)}, "
                          f"PER so far: {np.mean(pers):.4f}")

            elapsed = time.time() - t0
            mean_per = np.mean(pers)

            print(f"\n  {split_name}: PER = {mean_per:.4f} ({mean_per:.1%}) "
                  f"({elapsed:.1f}s)")

            if nbest_list:
                all_nbest[split_name] = nbest_list

        # Save N-best output
        if args.output and all_nbest:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(all_nbest, f, indent=2)
            print(f"\nN-best output: {output_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
