#!/usr/bin/env python3
"""
Lead 3: Masked Autoencoder (MAE) Pretraining + CTC Fine-tuning.

Two-phase training:
  Phase A: Self-supervised MAE pretraining (reconstruct masked patches)
  Phase B: CTC fine-tuning with pretrained encoder

Ref: arxiv 2511.21740 (BIT / Inner Speech)

Usage:
    # Phase A: MAE Pretraining
    CUDA_VISIBLE_DEVICES=8 python brain2speech/lead3_train_mae.py \
        --mode pretrain \
        --data sentences_paper_256d.h5 \
        --epochs 50 --lr 1e-4 --batch-size 64 \
        --d-model 256 --n-layers 4 --n-heads 8 \
        --mask-ratio 0.5 --patch-size 5 \
        --output brain2speech/results/mae_pretrained.pt

    # Phase B: CTC Fine-tuning
    CUDA_VISIBLE_DEVICES=10 python brain2speech/lead3_train_mae.py \
        --mode finetune \
        --pretrained brain2speech/results/mae_pretrained.pt \
        --data sentences_paper_256d.h5 \
        --finetune-strategy progressive \
        --lr-encoder 1e-5 --lr-head 1e-3 \
        --max-minibatches 10000 --seed 42

    # No-pretrain ablation (same arch, random init)
    CUDA_VISIBLE_DEVICES=10 python brain2speech/lead3_train_mae.py \
        --mode finetune --no-pretrain \
        --data sentences_paper_256d.h5 \
        --lr 1e-3 --max-minibatches 10000 --seed 42
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
from lead3_models import NeuralMAE, PatchCTCDecoder


# ═══════════════════════════════════════════════════════════════════════
# MAE Pretraining
# ═══════════════════════════════════════════════════════════════════════

def collate_raw_mae(batch):
    """Collate for MAE pretraining — no targets needed, just features."""
    features = [torch.FloatTensor(t['features_raw']) for t in batch]
    feature_lengths = torch.IntTensor([f.shape[0] for f in features])

    max_T = max(f.shape[0] for f in features)
    C = features[0].shape[1]
    padded = torch.zeros(len(features), max_T, C)
    for i, f in enumerate(features):
        padded[i, :f.shape[0], :] = f

    return padded, feature_lengths


def pretrain_mae(args):
    """Phase A: Self-supervised MAE pretraining."""
    device = torch.device('cuda:0')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'LEAD 3 — MAE Pretraining: {args.experiment}')
    print(f'  d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}')
    print(f'  mask_ratio={args.mask_ratio}, patch_size={args.patch_size}')
    print(f'  epochs={args.epochs}, lr={args.lr}')
    print('=' * 70)

    # Load all data (no labels needed for self-supervised)
    h5_path = DATA_DIR / args.data
    if not h5_path.exists():
        print(f'ERROR: {h5_path} not found')
        sys.exit(1)

    trials = load_h5_dataset(
        h5_path, max_trials=args.max_trials,
        smooth_sigma=args.smooth, causal_smooth=True, load_raw=True,
    )

    print(f'\nLoaded {len(trials)} trials for pretraining')

    # Use all trials for pretraining (no split needed — self-supervised)
    # But hold out last 10% for validation of reconstruction loss
    n_val = max(100, len(trials) // 10)
    np.random.seed(args.seed)
    indices = np.random.permutation(len(trials))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_trials = [trials[i] for i in train_indices]
    val_trials = [trials[i] for i in val_indices]

    print(f'  Pretrain: {len(train_trials)} train, {len(val_trials)} val')

    train_loader = DataLoader(train_trials, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_raw_mae,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_trials, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_raw_mae,
                            num_workers=2)

    # Build MAE model
    model = NeuralMAE(
        n_features=args.input_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        dropout=args.mae_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nNeuralMAE: {n_params/1e6:.1f}M params')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.05,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Warmup
    warmup_epochs = min(5, args.epochs // 10)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, scheduler], milestones=[warmup_epochs])

    scaler = GradScaler()
    best_val_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0
        total_masked = 0
        n_batches = 0
        for features, feat_lens in train_loader:
            features = features.to(device)

            # Add noise augmentation
            if args.white_noise > 0:
                features = features + torch.randn_like(features) * args.white_noise

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss, n_masked = model(features)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_masked += n_masked
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss_sum = 0
        val_batches = 0
        with torch.no_grad():
            for features, feat_lens in val_loader:
                features = features.to(device)
                with autocast("cuda"):
                    loss, _ = model(features)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss_sum += loss.item()
                    val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step()

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1:3d}/{args.epochs} | '
              f'train_MSE={train_loss:.6f} | val_MSE={val_loss:.6f} | '
              f'lr={cur_lr:.2e} | {elapsed:.1f}s')

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': cur_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'  *** New best val MSE: {best_val_loss:.6f} ***')

    # Save pretrained model
    output_path = args.output or str(RESULTS_DIR / f'mae_pretrained_{args.experiment}.pt')
    torch.save({
        'state_dict': best_state,
        'mae_config': {
            'n_features': args.input_dim,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'mask_ratio': args.mask_ratio,
            'patch_size': args.patch_size,
            'dropout': args.mae_dropout,
        },
        'best_val_loss': best_val_loss,
        'history': history,
    }, output_path)
    print(f'\nPretrained MAE saved: {output_path}')
    print(f'Best val MSE: {best_val_loss:.6f}')

    return output_path


# ═══════════════════════════════════════════════════════════════════════
# CTC Fine-tuning
# ═══════════════════════════════════════════════════════════════════════

def finetune_ctc(args):
    """Phase B: CTC fine-tuning with (optionally) pretrained encoder."""
    device = torch.device('cuda:0')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'LEAD 3 — MAE Fine-tune: {args.experiment}')
    print(f'  Pretrained: {args.pretrained or "NONE (from scratch)"}')
    print(f'  Strategy: {args.finetune_strategy}')
    print(f'  Seed: {args.seed}')
    print('=' * 70)

    # Load data
    h5_path = DATA_DIR / args.data
    if not h5_path.exists():
        print(f'ERROR: {h5_path} not found')
        sys.exit(1)

    trials = load_h5_dataset(
        h5_path, max_trials=args.max_trials,
        smooth_sigma=args.smooth, causal_smooth=True, load_raw=True,
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

    print(f'\nTrials: {len(train_trials)} train, {len(val_trials)} val, '
          f'{len(test_trials)} test')

    train_loader = DataLoader(train_trials, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_raw,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_trials, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_raw, num_workers=2)
    test_loader = DataLoader(test_trials, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_raw, num_workers=2)

    # Determine config from pretrained or defaults
    if args.pretrained and not args.no_pretrain:
        ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=True)
        mae_config = ckpt['mae_config']
        d_model = mae_config['d_model']
        n_layers = mae_config['n_layers']
        n_heads = mae_config['n_heads']
        patch_size = mae_config['patch_size']
        n_features = mae_config['n_features']
        mae_dropout = mae_config.get('dropout', 0.1)
        print(f'  Using pretrained config: d_model={d_model}, layers={n_layers}, '
              f'heads={n_heads}, patch={patch_size}')
    else:
        d_model = args.d_model
        n_layers = args.n_layers
        n_heads = args.n_heads
        patch_size = args.patch_size
        n_features = args.input_dim
        mae_dropout = args.mae_dropout

    # Build CTC decoder model
    n_output = N_CLASSES + 1  # 41
    model = PatchCTCDecoder(
        n_features=n_features,
        n_classes=n_output,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        patch_size=patch_size,
        dropout=mae_dropout,
        n_sessions=n_sessions,
        day_hidden=args.day_hidden,
        use_day_specific=True,
    )
    model.set_train_sessions(train_session_idxs)

    # Load pretrained weights
    if args.pretrained and not args.no_pretrain:
        mae_model = NeuralMAE(
            n_features=n_features, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads,
            patch_size=patch_size, dropout=mae_dropout,
        )
        mae_model.load_state_dict(ckpt['state_dict'])
        print('\nLoading pretrained MAE weights into PatchCTCDecoder...')
        model.load_mae_weights(mae_model)
        del mae_model
    else:
        print('\nTraining from scratch (no pretraining)')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'PatchCTCDecoder: {n_params/1e6:.1f}M params')

    model = model.to(device)

    # Optimizer setup depends on fine-tuning strategy
    batches_per_epoch = len(train_loader)
    total_epochs = max(1, args.max_minibatches // batches_per_epoch)

    if args.finetune_strategy == 'frozen':
        # Freeze encoder, only train output head + day-specific
        for name, param in model.named_parameters():
            if 'encoder' in name or 'patch_proj' in name or 'pos_embed' in name:
                param.requires_grad = False
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_head, eps=0.1, weight_decay=1e-5,
        )
    elif args.finetune_strategy == 'progressive':
        # Start frozen, progressively unfreeze
        for name, param in model.named_parameters():
            if 'encoder' in name or 'patch_proj' in name or 'pos_embed' in name:
                param.requires_grad = False
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_head, eps=0.1, weight_decay=1e-5,
        )
    else:  # 'full'
        # Different LR for encoder vs head
        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'encoder' in name or 'patch_proj' in name or 'pos_embed' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': args.lr_encoder},
            {'params': head_params, 'lr': args.lr_head},
        ], eps=0.1, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6)

    print(f'Training: {total_epochs} epochs, strategy={args.finetune_strategy}')

    criterion = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
    scaler = GradScaler()

    best_val_per = float('inf')
    best_state = None
    wait = 0
    history = []

    # Progressive unfreeze schedule
    unfreeze_epoch_1 = total_epochs // 3  # Unfreeze top layers
    unfreeze_epoch_2 = 2 * total_epochs // 3  # Unfreeze all

    for epoch in range(total_epochs):
        t0 = time.time()

        # Progressive unfreezing
        if args.finetune_strategy == 'progressive':
            if epoch == unfreeze_epoch_1:
                print(f'\n  >>> Unfreezing top encoder layers (epoch {epoch+1})')
                # Unfreeze last 2 encoder layers
                encoder_layers = list(model.encoder.layers)
                for layer in encoder_layers[-2:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                optimizer.add_param_group({
                    'params': [p for l in encoder_layers[-2:]
                               for p in l.parameters()],
                    'lr': args.lr_encoder,
                })

            elif epoch == unfreeze_epoch_2:
                print(f'\n  >>> Unfreezing all encoder layers (epoch {epoch+1})')
                for param in model.parameters():
                    param.requires_grad = True
                # Rebuild optimizer with all params
                encoder_params = []
                head_params = []
                for name, param in model.named_parameters():
                    if 'encoder' in name or 'patch_proj' in name or 'pos_embed' in name:
                        encoder_params.append(param)
                    else:
                        head_params.append(param)
                optimizer = torch.optim.Adam([
                    {'params': encoder_params, 'lr': args.lr_encoder},
                    {'params': head_params, 'lr': args.lr_head * 0.1},
                ], eps=0.1, weight_decay=1e-5)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_epochs - epoch, eta_min=1e-6)

        # Train
        model.train()
        total_loss = 0
        n_batches = 0

        for features, targets, feat_lens, tgt_lens, session_ids in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            session_ids = session_ids.to(device)

            # Noise augmentation
            if args.white_noise > 0:
                features = features + torch.randn_like(features) * args.white_noise
            if args.offset_noise > 0:
                B, T, C = features.shape
                offset = torch.randn(B, 1, C, device=device) * args.offset_noise
                features = features + offset

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                logits = model(features, session_ids=session_ids)
                log_probs = logits.log_softmax(dim=2)
                log_probs_t = log_probs.transpose(0, 1)

                # Output lengths after patchification
                out_lens = (feat_lens // patch_size).clamp(min=1)
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

        train_loss = total_loss / max(n_batches, 1)

        # Validate
        val_metrics = _evaluate_patch_model(model, val_loader, device, patch_size)

        scheduler.step()
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1:3d}/{total_epochs} | '
              f'loss={train_loss:.4f} | '
              f'val_PER={val_metrics["per"]:.3f} | '
              f'lr={cur_lr:.2e} | {elapsed:.1f}s')

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_per': val_metrics['per'],
            'lr': cur_lr,
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

    test_metrics = _evaluate_patch_model(model, test_loader, device, patch_size)

    print(f'\n{"="*70}')
    print(f'RESULTS — {args.experiment}')
    print(f'  Val PER:  {best_val_per:.1%}')
    print(f'  Test PER: {test_metrics["per"]:.1%}')
    print(f'  Pretrained: {bool(args.pretrained and not args.no_pretrain)}')
    print(f'{"="*70}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'experiment': args.experiment,
        'lead': 'L3',
        'model': 'mae_finetune',
        'test_per': test_metrics['per'],
        'best_val_per': float(best_val_per),
        'n_train': len(train_trials),
        'n_val': len(val_trials),
        'n_test': len(test_trials),
        'epochs_trained': epoch + 1,
        'n_params': n_params,
        'pretrained': bool(args.pretrained and not args.no_pretrain),
        'finetune_strategy': args.finetune_strategy,
        'hyperparams': vars(args),
        'history': history,
    }

    save_path = RESULTS_DIR / f'L3_mae_{args.experiment}_{timestamp}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results: {save_path}')

    model_path = RESULTS_DIR / f'L3_mae_{args.experiment}_best.pt'
    torch.save({
        'state_dict': best_state,
        'model_config': {
            'model_type': 'mae_finetune',
            'n_features': n_features,
            'n_classes': n_output,
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'patch_size': patch_size,
            'n_sessions': n_sessions,
            'day_hidden': args.day_hidden,
        },
        'test_per': test_metrics['per'],
        'best_val_per': float(best_val_per),
    }, model_path)
    print(f'Model: {model_path}')


def _evaluate_patch_model(model, dataloader, device, patch_size):
    """Evaluate patch-based CTC model."""
    model.eval()
    all_per = []

    with torch.no_grad():
        for features, targets, feat_lens, tgt_lens, session_ids in dataloader:
            features = features.to(device)
            session_ids = session_ids.to(device)

            with autocast("cuda"):
                logits = model(features, session_ids=session_ids)
                log_probs = logits.log_softmax(dim=2).cpu().numpy()

            out_lens = (feat_lens // patch_size).clamp(min=1)
            T_out = log_probs.shape[1]
            offset = 0
            for i in range(len(feat_lens)):
                T = min(out_lens[i].item(), T_out)
                P = tgt_lens[i].item()
                decoded = ctc_greedy_decode(log_probs[i, :T, :])
                trial_targets = targets[offset:offset + P].numpy().tolist()
                offset += P
                all_per.append(compute_per(decoded, trial_targets))

    return {'per': float(np.mean(all_per)) if all_per else 1.0}


def main():
    parser = argparse.ArgumentParser(description='Lead 3: MAE Pretrain + Fine-tune')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['pretrain', 'finetune'],
                        help='pretrain: MAE self-supervised, finetune: CTC')
    parser.add_argument('--experiment', type=str, default='mae_default')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--data', type=str, default='sentences_paper_256d.h5')
    parser.add_argument('--input-dim', type=int, default=256)

    # MAE architecture
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--mask-ratio', type=float, default=0.5)
    parser.add_argument('--patch-size', type=int, default=5)
    parser.add_argument('--mae-dropout', type=float, default=0.1)
    parser.add_argument('--day-hidden', type=int, default=256)

    # Pretraining
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for pretrained model')
    parser.add_argument('--white-noise', type=float, default=0.5)
    parser.add_argument('--smooth', type=float, default=2)
    parser.add_argument('--max-trials', type=int, default=None)

    # Fine-tuning
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained MAE checkpoint')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Train from scratch (ablation)')
    parser.add_argument('--finetune-strategy', type=str, default='full',
                        choices=['frozen', 'full', 'progressive'])
    parser.add_argument('--lr-encoder', type=float, default=1e-5)
    parser.add_argument('--lr-head', type=float, default=1e-3)
    parser.add_argument('--max-minibatches', type=int, default=10000)
    parser.add_argument('--offset-noise', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == 'pretrain':
        pretrain_mae(args)
    elif args.mode == 'finetune':
        finetune_ctc(args)


if __name__ == '__main__':
    main()
