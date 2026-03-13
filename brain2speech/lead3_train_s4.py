#!/usr/bin/env python3
"""
Lead 3: S4 Decoder Training — Structured State Space Model.

S4D (diagonal variant) with HiPPO initialization, bidirectional, FFT convolution.
Ref: Gu et al. ICLR 2022, tbenst/silent_speech s4.py

Usage:
    # Basic S4
    CUDA_VISIBLE_DEVICES=9 python brain2speech/lead3_train_s4.py \
        --experiment L3_s4_base --n-layers 6 --d-state 64 --d-model 512 \
        --lr 0.001 --scheduler cosine --max-minibatches 10000 --seed 42

    # S4 with layer-wise downsampling (tbenst pattern)
    CUDA_VISIBLE_DEVICES=9 python brain2speech/lead3_train_s4.py \
        --experiment L3_s4_downsample --n-layers 6 --d-state 64 \
        --downsample-layers 0,1,2 --downsample-factor 2 \
        --lr 0.001 --scheduler cosine --max-minibatches 10000 --seed 42
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# ── Paths ──
DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")

N_CLASSES = 40
CTC_BLANK = N_CLASSES

sys.path.insert(0, str(Path(__file__).parent))
from train_beyond_paper import (
    load_h5_dataset, collate_raw, ctc_greedy_decode, compute_per,
)
from lead3_models import S4Decoder


def evaluate_model(model, dataloader, device, kernel_size=14, stride=4,
                   downsample_layers=None, downsample_factor=2):
    """Evaluate S4 model."""
    model.eval()
    all_per = []
    total_loss = 0
    n_batches = 0
    criterion = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)

    with torch.no_grad():
        for features, targets, feat_lens, tgt_lens, session_ids in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            session_ids = session_ids.to(device)

            with autocast("cuda"):
                logits = model(features, session_ids=session_ids)
                log_probs = logits.log_softmax(dim=2)
                log_probs_t = log_probs.transpose(0, 1)

                # Output length after stacking
                out_lens = ((feat_lens - kernel_size) // stride + 1).clamp(min=1)
                # Account for downsampling layers
                if downsample_layers:
                    for _ in downsample_layers:
                        out_lens = out_lens // downsample_factor
                out_lens = out_lens.clamp(min=1)

                input_lengths = out_lens.clamp(max=log_probs.shape[1]).to(device)
                target_lengths = tgt_lens.to(device)

                loss = criterion(log_probs_t, targets, input_lengths, target_lengths)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1

            log_probs_np = log_probs.cpu().numpy()
            T_out = log_probs_np.shape[1]
            offset = 0
            for i in range(len(feat_lens)):
                T = min(out_lens[i].item(), T_out)
                P = tgt_lens[i].item()
                decoded = ctc_greedy_decode(log_probs_np[i, :T, :])
                trial_targets = targets.cpu()[offset:offset + P].numpy().tolist()
                offset += P
                all_per.append(compute_per(decoded, trial_targets))

    return {
        'per': float(np.mean(all_per)) if all_per else 1.0,
        'ctc_loss': total_loss / max(n_batches, 1),
    }


def train_epoch(model, dataloader, optimizer, criterion, scaler, device,
                white_noise_sd=1.0, offset_noise_sd=0.2,
                kernel_size=14, stride=4,
                downsample_layers=None, downsample_factor=2):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for features, targets, feat_lens, tgt_lens, session_ids in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        session_ids = session_ids.to(device)

        if white_noise_sd > 0:
            features = features + torch.randn_like(features) * white_noise_sd
        if offset_noise_sd > 0:
            B, T, C = features.shape
            offset = torch.randn(B, 1, C, device=device) * offset_noise_sd
            features = features + offset

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            logits = model(features, session_ids=session_ids)
            log_probs = logits.log_softmax(dim=2)
            log_probs_t = log_probs.transpose(0, 1)

            out_lens = ((feat_lens - kernel_size) // stride + 1).clamp(min=1)
            if downsample_layers:
                for _ in downsample_layers:
                    out_lens = out_lens // downsample_factor
                out_lens = out_lens.clamp(min=1)

            input_lengths = out_lens.clamp(max=log_probs.shape[1]).to(device)
            target_lengths = tgt_lens.to(device)

            loss = criterion(log_probs_t, targets, input_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Lead 3: S4 Decoder')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--data', type=str, default='sentences_paper_256d.h5')
    parser.add_argument('--input-dim', type=int, default=256)

    # Model
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--d-state', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--day-hidden', type=int, default=256)
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--no-bidirectional', dest='bidirectional',
                        action='store_false')
    parser.add_argument('--downsample-layers', type=str, default='',
                        help='Comma-separated layer indices for 2x downsampling')
    parser.add_argument('--downsample-factor', type=int, default=2)

    # Training
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-minibatches', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--adam-eps', type=float, default=0.1)
    parser.add_argument('--l2-reg', type=float, default=1e-5)
    parser.add_argument('--white-noise', type=float, default=1.0)
    parser.add_argument('--offset-noise', type=float, default=0.2)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine', 'step'])
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)

    # Stacking
    parser.add_argument('--kernel-size', type=int, default=14)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--smooth', type=float, default=2)
    parser.add_argument('--max-trials', type=int, default=None)

    args = parser.parse_args()

    # Parse downsample layers
    downsample_layers = []
    if args.downsample_layers:
        downsample_layers = [int(x) for x in args.downsample_layers.split(',')]

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'LEAD 3 — S4: {args.experiment}')
    print(f'  d_model={args.d_model}, layers={args.n_layers}, d_state={args.d_state}')
    print(f'  Bidirectional: {args.bidirectional}')
    print(f'  Downsample layers: {downsample_layers or "none"}')
    print(f'  LR: {args.lr}, Scheduler: {args.scheduler}')
    print(f'  Seed: {args.seed}')
    print('=' * 70)

    # Load data
    h5_path = DATA_DIR / args.data
    if not h5_path.exists():
        print(f'ERROR: {h5_path} not found')
        sys.exit(1)

    trials = load_h5_dataset(
        h5_path, max_trials=args.max_trials,
        smooth_sigma=args.smooth,
        kernel_size=args.kernel_size, stride=args.stride,
        causal_smooth=True, load_raw=True,
    )

    # Session index
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
    train_sessions_list = remaining[:-2]

    train_trials = [t for s in train_sessions_list for t in sessions[s]]
    val_trials = [t for s in val_sessions for t in sessions[s]]
    test_trials = [t for s in test_sessions for t in sessions[s]]
    train_session_idxs = {session_to_idx[s] for s in train_sessions_list}

    print(f'\nSessions: {n_sessions} ({len(train_sessions_list)} train, '
          f'{len(val_sessions)} val, {len(test_sessions)} test)')
    print(f'Trials: {len(train_trials)} train, {len(val_trials)} val, '
          f'{len(test_trials)} test')

    train_loader = DataLoader(train_trials, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_raw,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_trials, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_raw, num_workers=2)
    test_loader = DataLoader(test_trials, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_raw, num_workers=2)

    # Build model
    n_output = N_CLASSES + 1  # 41
    model = S4Decoder(
        n_features_per_frame=args.input_dim,
        n_classes=n_output,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        dropout=args.dropout,
        n_sessions=n_sessions,
        kernel_size=args.kernel_size,
        stride=args.stride,
        day_hidden=args.day_hidden,
        bidirectional=args.bidirectional,
        downsample_layers=downsample_layers,
        downsample_factor=args.downsample_factor,
    )
    model.set_train_sessions(train_session_idxs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nS4Decoder: {n_params/1e6:.1f}M params')

    model = model.to(device)

    # S4 benefits from separate LR for SSM params (typically lower)
    ssm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 's4_fwd' in name or 's4_bwd' in name:
            ssm_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': args.lr},
        {'params': ssm_params, 'lr': min(args.lr, 1e-3)},  # Cap SSM LR
    ], betas=(0.9, 0.999), eps=args.adam_eps, weight_decay=args.l2_reg)

    batches_per_epoch = len(train_loader)
    total_epochs = max(1, args.max_minibatches // batches_per_epoch)

    if args.scheduler == 'cosine':
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6)
    elif args.scheduler == 'step':
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total_epochs // 3), gamma=0.5)
    else:
        base_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.001,
            total_iters=total_epochs)

    if args.warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=args.warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, base_scheduler],
            milestones=[args.warmup_steps])
    else:
        scheduler = base_scheduler

    print(f'Training: {total_epochs} epochs, {batches_per_epoch} batches/epoch')

    criterion = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
    scaler = GradScaler()

    best_val_per = float('inf')
    best_state = None
    wait = 0
    history = []

    for epoch in range(total_epochs):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            white_noise_sd=args.white_noise, offset_noise_sd=args.offset_noise,
            kernel_size=args.kernel_size, stride=args.stride,
            downsample_layers=downsample_layers,
            downsample_factor=args.downsample_factor,
        )

        val_metrics = evaluate_model(
            model, val_loader, device,
            kernel_size=args.kernel_size, stride=args.stride,
            downsample_layers=downsample_layers,
            downsample_factor=args.downsample_factor,
        )

        scheduler.step()

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1:3d}/{total_epochs} | '
              f'loss={train_loss:.4f} | '
              f'val_PER={val_metrics["per"]:.3f} '
              f'val_CTC={val_metrics["ctc_loss"]:.3f} | '
              f'lr={cur_lr:.2e} | {elapsed:.1f}s')

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_per': val_metrics['per'],
            'val_ctc_loss': val_metrics['ctc_loss'],
            'lr': cur_lr,
            'elapsed': elapsed,
        })

        if val_metrics['per'] < best_val_per:
            best_val_per = val_metrics['per']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            print(f'  *** New best val PER: {best_val_per:.3f} ***')
        else:
            wait += 1

        if wait >= args.patience:
            print(f'Early stop at epoch {epoch+1}')
            break

    # Test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    test_metrics = evaluate_model(
        model, test_loader, device,
        kernel_size=args.kernel_size, stride=args.stride,
        downsample_layers=downsample_layers,
        downsample_factor=args.downsample_factor,
    )

    print(f'\n{"="*70}')
    print(f'RESULTS — {args.experiment}')
    print(f'  Val PER:      {best_val_per:.1%}')
    print(f'  Test PER:     {test_metrics["per"]:.1%}')
    print(f'  Test CTC:     {test_metrics["ctc_loss"]:.4f}')
    print(f'{"="*70}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'experiment': args.experiment,
        'lead': 'L3',
        'model': 's4',
        'test_per': test_metrics['per'],
        'test_ctc_loss': test_metrics['ctc_loss'],
        'best_val_per': float(best_val_per),
        'n_train': len(train_trials),
        'n_val': len(val_trials),
        'n_test': len(test_trials),
        'epochs_trained': epoch + 1,
        'n_params': n_params,
        'hyperparams': vars(args),
        'history': history,
    }

    save_path = RESULTS_DIR / f'L3_s4_{args.experiment}_{timestamp}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results: {save_path}')

    model_path = RESULTS_DIR / f'L3_s4_{args.experiment}_best.pt'
    torch.save({
        'state_dict': best_state,
        'model_config': {
            'model_type': 's4',
            'n_features_per_frame': args.input_dim,
            'n_classes': n_output,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'd_state': args.d_state,
            'dropout': args.dropout,
            'n_sessions': n_sessions,
            'kernel_size': args.kernel_size,
            'stride': args.stride,
            'day_hidden': args.day_hidden,
            'bidirectional': args.bidirectional,
            'downsample_layers': downsample_layers,
            'downsample_factor': args.downsample_factor,
        },
        'test_per': test_metrics['per'],
        'best_val_per': float(best_val_per),
    }, model_path)
    print(f'Model: {model_path}')


if __name__ == '__main__':
    main()
