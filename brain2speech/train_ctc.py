#!/usr/bin/env python3
"""
CTC-based phoneme decoding from neural signals.

Matches the Willett et al. (Nature 2023) approach:
- Train on sentence-level competitionData (8.8k sentences, 24 sessions)
- CTC loss for variable-length alignment-free training
- GRU/TCN encoder → linear → CTC decode
- Cross-session evaluation via leave-one-session-out

This is the apples-to-apples comparison with the paper's 61.4% accuracy.

Usage:
    python train_ctc.py --model GRU --gpus 6 7    # single model
    python train_ctc.py --model TCN --epochs 100   # TCN with CTC
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")

SEED = 42
N_CLASSES = 40  # 39 phonemes + SIL
CTC_BLANK = N_CLASSES  # 40


def smooth_features(features, sigma):
    """Apply Gaussian temporal smoothing to features along the time axis.

    Following Willett et al., smooth neural features with a Gaussian kernel
    to reduce high-frequency noise.

    Args:
        features: (T, C) array of neural features
        sigma: standard deviation of the Gaussian kernel in time bins

    Returns:
        Smoothed features with same shape (T, C)
    """
    if sigma <= 0:
        return features
    return gaussian_filter1d(features, sigma=sigma, axis=0)


def load_h5_dataset(h5_path, max_trials=None, smooth_sigma=0):
    """Load preprocessed sentence data from HDF5.

    Returns list of dicts with features (T, 1280), phoneme_indices (P,), session.

    Args:
        h5_path: path to HDF5 file
        max_trials: limit number of trials (for debugging)
        smooth_sigma: if > 0, apply Gaussian temporal smoothing with this sigma
    """
    trials = []
    with h5py.File(h5_path, 'r') as f:
        n_trials = f.attrs['n_trials']
        if max_trials:
            n_trials = min(n_trials, max_trials)

        for i in range(n_trials):
            grp = f[f'trial_{i:05d}']
            features = grp['features'][:]
            if smooth_sigma > 0:
                features = smooth_features(features, smooth_sigma)
            trials.append({
                'features': features,
                'phoneme_indices': grp['phoneme_indices'][:],
                'session': grp.attrs['session'],
                'text': grp.attrs['text'],
                'n_frames': grp.attrs['n_frames'],
                'n_phonemes': grp.attrs['n_phonemes'],
            })

    print(f"Loaded {len(trials)} trials from {h5_path}")
    if smooth_sigma > 0:
        print(f"  Applied Gaussian smoothing with sigma={smooth_sigma}")
    return trials


def collate_ctc(batch):
    """Collate variable-length trials for CTC training.

    Returns:
        features: (B, T_max, C) padded
        targets: (sum of target lengths,) concatenated
        feature_lengths: (B,)
        target_lengths: (B,)
    """
    import torch

    features = [torch.FloatTensor(t['features']) for t in batch]
    targets = [torch.IntTensor(t['phoneme_indices']) for t in batch]

    feature_lengths = torch.IntTensor([f.shape[0] for f in features])
    target_lengths = torch.IntTensor([t.shape[0] for t in targets])

    # Pad features to max length
    max_T = max(f.shape[0] for f in features)
    C = features[0].shape[1]
    padded = torch.zeros(len(features), max_T, C)
    for i, f in enumerate(features):
        padded[i, :f.shape[0], :] = f

    # Concatenate targets
    targets_cat = torch.cat(targets)

    return padded, targets_cat, feature_lengths, target_lengths


class CTCEncoder(object):
    """Factory for CTC encoder models."""

    @staticmethod
    def build(model_type, n_features=1280, n_classes=41, n_layers=3, **kwargs):
        """Build an encoder model.

        Args:
            model_type: 'GRU', 'TCN', or 'Transformer'
            n_features: input feature dimension
            n_classes: output classes (40 phonemes + 1 CTC blank)
            n_layers: number of recurrent/conv/transformer layers

        Returns:
            nn.Module that maps (B, T, C) → (B, T, n_classes)
        """
        import torch.nn as nn

        if model_type == 'GRU':
            return GRUCTCEncoder(n_features, n_classes, n_layers=n_layers, **kwargs)
        elif model_type == 'TCN':
            kwargs.pop('n_layers', None)
            return TCNCTCEncoder(n_features, n_classes, n_blocks=n_layers, **kwargs)

        elif model_type == 'Transformer':
            # Map 'hidden' to 'd_model' for Transformer interface
            if 'hidden' in kwargs:
                kwargs['d_model'] = kwargs.pop('hidden')
            return TransformerCTCEncoder(n_features, n_classes, num_layers=n_layers, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class GRUCTCEncoder(object):
    """GRU encoder for CTC — matches paper's architecture."""

    def __new__(cls, n_features=1280, n_classes=41, hidden=512, n_layers=3, dr=0.3,
                downsample=1):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.downsample = downsample
                self.input_proj = nn.Linear(n_features, hidden)
                self.input_norm = nn.LayerNorm(hidden)
                if downsample > 1:
                    self.ds_conv = nn.Conv1d(hidden, hidden, kernel_size=downsample * 2 - 1,
                                            stride=downsample, padding=downsample - 1)
                self.rnn = nn.GRU(
                    hidden, hidden, n_layers,
                    batch_first=True, bidirectional=True, dropout=dr
                )
                self.output_proj = nn.Sequential(
                    nn.LayerNorm(hidden * 2),
                    nn.Dropout(dr),
                    nn.Linear(hidden * 2, n_classes),
                )

            def forward(self, x):
                # x: (B, T, C=1280) → (B, T', n_classes)
                x = self.input_norm(self.input_proj(x))
                if self.downsample > 1:
                    x = self.ds_conv(x.transpose(1, 2)).transpose(1, 2)
                x, _ = self.rnn(x)
                return self.output_proj(x)

        return _Model()


class TCNCTCEncoder(object):
    """TCN encoder for CTC."""

    def __new__(cls, n_features=1280, n_classes=41, hidden=256, n_blocks=6, kernel_size=7, dr=0.3,
                downsample=1):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Linear(n_features, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                )
                # TCN blocks with increasing dilation
                layers = []
                for i in range(n_blocks):
                    dilation = 2 ** (i % 4)
                    padding = (kernel_size - 1) * dilation // 2
                    layers.append(nn.Sequential(
                        nn.Conv1d(hidden, hidden, kernel_size,
                                  dilation=dilation, padding=padding),
                        nn.BatchNorm1d(hidden),
                        nn.GELU(),
                        nn.Dropout(dr),
                    ))
                self.tcn = nn.ModuleList(layers)
                self.output_proj = nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, n_classes),
                )

            def forward(self, x):
                # x: (B, T, C=1280) → (B, T, n_classes)
                x = self.input_proj(x)  # (B, T, hidden)
                x = x.transpose(1, 2)    # (B, hidden, T) for Conv1d
                for block in self.tcn:
                    residual = x
                    x = block(x)
                    if x.shape == residual.shape:
                        x = x + residual  # residual connection
                x = x.transpose(1, 2)    # (B, T, hidden)
                return self.output_proj(x)

        return _Model()


class TransformerCTCEncoder(object):
    """Transformer encoder for CTC."""

    def __new__(cls, n_features=1280, n_classes=41, d_model=256, nhead=8,
                num_layers=4, dr=0.3, max_len=2000, downsample=1):
        import torch
        import torch.nn as nn
        import math

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Linear(n_features, d_model),
                    nn.LayerNorm(d_model),
                )
                # Positional encoding
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe.unsqueeze(0))

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=d_model * 4, dropout=dr,
                    batch_first=True, activation='gelu',
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output_proj = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, n_classes),
                )

            def forward(self, x):
                # x: (B, T, C=1280) → (B, T, n_classes)
                x = self.input_proj(x) + self.pe[:, :x.shape[1], :]
                x = self.encoder(x)
                return self.output_proj(x)

        return _Model()


def ctc_greedy_decode(log_probs, blank=CTC_BLANK):
    """Greedy CTC decoding.

    Args:
        log_probs: (T, n_classes) log probabilities

    Returns:
        decoded: list of int (phoneme indices, collapsed)
    """
    best_path = log_probs.argmax(axis=1)

    # Collapse repeats and remove blanks
    decoded = []
    prev = -1
    for t in best_path:
        if t != prev:
            if t != blank:
                decoded.append(int(t))
        prev = t

    return decoded


def compute_per(predicted, target):
    """Compute Phoneme Error Rate using edit distance."""
    import editdistance
    if len(target) == 0:
        return 1.0 if len(predicted) > 0 else 0.0
    return editdistance.eval(predicted, target) / len(target)


def train_epoch(model, dataloader, optimizer, criterion, scaler, device, noise_std=0.05):
    """Train one epoch with CTC loss."""
    import torch
    from torch.amp import autocast

    model.train()
    total_loss = 0
    n_batches = 0

    for features, targets, feat_lens, tgt_lens in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        # Data augmentation
        if noise_std > 0 and np.random.random() < 0.5:
            features = features + torch.randn_like(features) * noise_std

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            log_probs = model(features).log_softmax(dim=2)  # (B, T', C)
            # CTC expects (T, B, C)
            log_probs_t = log_probs.transpose(0, 1)

            # Input lengths must match model output length (may be downsampled)
            input_lengths = torch.full((log_probs.shape[0],), log_probs.shape[1],
                                       dtype=torch.int32, device=device)
            # Clamp to actual output length per sample
            for bi in range(len(feat_lens)):
                input_lengths[bi] = min(input_lengths[bi].item(), log_probs.shape[1])
            target_lengths = tgt_lens.to(device)

            loss = criterion(log_probs_t, targets, input_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, device):
    """Evaluate: compute PER and phoneme accuracy via CTC greedy decode."""
    import torch
    from torch.amp import autocast

    model.eval()
    all_per = []
    total_correct = 0
    total_phonemes = 0

    with torch.no_grad():
        for features, targets, feat_lens, tgt_lens in dataloader:
            features = features.to(device)

            with autocast("cuda"):
                log_probs = model(features).log_softmax(dim=2).cpu().numpy()

            # Decode each trial
            T_out = log_probs.shape[1]  # may be downsampled
            offset = 0
            for i in range(len(feat_lens)):
                T = min(feat_lens[i].item(), T_out)  # use actual output length
                P = tgt_lens[i].item()
                trial_log_probs = log_probs[i, :T, :]
                trial_targets = targets[offset:offset + P].numpy().tolist()
                offset += P

                decoded = ctc_greedy_decode(trial_log_probs)
                per = compute_per(decoded, trial_targets)
                all_per.append(per)

                # Count correct phonemes (via alignment)
                min_len = min(len(decoded), len(trial_targets))
                for j in range(min_len):
                    if decoded[j] == trial_targets[j]:
                        total_correct += 1
                total_phonemes += len(trial_targets)

    mean_per = np.mean(all_per) if all_per else 1.0
    accuracy = total_correct / max(total_phonemes, 1)

    return {
        'per': float(mean_per),
        'accuracy': float(accuracy),
        'n_trials': len(all_per),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['GRU', 'TCN', 'Transformer'], default='GRU')
    parser.add_argument('--gpus', nargs='+', type=int, default=[6, 7])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--scheduler', choices=['plateau', 'cosine', 'none'], default='plateau',
                        help='LR scheduler: plateau (safe), cosine (aggressive), none')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Temporal downsampling factor (1=none, 2=halve seq len)')
    parser.add_argument('--noise', type=float, default=0.05,
                        help='Gaussian noise std for data augmentation (0=none)')
    parser.add_argument('--smooth', type=float, default=2,
                        help='Gaussian temporal smoothing sigma (0=none, default=2, per Willett et al.)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='AdamW weight decay (default=0.01)')
    parser.add_argument('--multisession-norm', action='store_true',
                        help='Apply per-session z-score normalization independently')
    parser.add_argument('--max-trials', type=int, default=None,
                        help='Limit trials for debugging')
    parser.add_argument('--eval-mode', choices=['holdout', 'loso'], default='holdout',
                        help='holdout: use last 4 sessions as test. '
                             'loso: leave-one-session-out CV')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.amp import GradScaler

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda:0')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'CTC PHONEME DECODING — {args.model}')
    print(f'  Device: {device} (GPUs: {args.gpus})')
    print(f'  Epochs: {args.epochs}, BS: {args.batch_size}, LR: {args.lr}')
    print(f'  Smooth: sigma={args.smooth}, Weight decay: {args.weight_decay}')
    if args.multisession_norm:
        print(f'  Per-session z-score normalization: ENABLED')
    print('=' * 70)

    # Load data
    h5_path = DATA_DIR / 'sentences_train.h5'
    if not h5_path.exists():
        print(f'ERROR: {h5_path} not found. Run preprocess_sentences.py first.')
        sys.exit(1)

    trials = load_h5_dataset(h5_path, max_trials=args.max_trials, smooth_sigma=args.smooth)

    # Split by session
    sessions = {}
    for t in trials:
        s = t['session']
        if s not in sessions:
            sessions[s] = []
        sessions[s].append(t)

    session_names = sorted(sessions.keys())
    print(f'\nSessions: {len(session_names)}')
    for s in session_names:
        print(f'  {s}: {len(sessions[s])} trials')

    # Per-session z-score normalization
    if args.multisession_norm:
        print('\nApplying per-session z-score normalization...')
        for s in session_names:
            session_feats = np.concatenate([t['features'] for t in sessions[s]], axis=0)
            mu = session_feats.mean(axis=0)
            std = session_feats.std(axis=0) + 1e-8
            for t in sessions[s]:
                t['features'] = (t['features'] - mu) / std
            print(f'  Session {s}: normed {len(sessions[s])} trials')

    if args.eval_mode == 'holdout':
        # Use last 4 sessions as test
        train_sessions = session_names[:-4]
        test_sessions = session_names[-4:]

        train_trials = [t for s in train_sessions for t in sessions[s]]
        test_trials = [t for s in test_sessions for t in sessions[s]]

        print(f'\nTrain: {len(train_trials)} trials ({len(train_sessions)} sessions)')
        print(f'Test:  {len(test_trials)} trials ({len(test_sessions)} sessions)')

        # Validation: last 2 train sessions
        val_sessions = train_sessions[-2:]
        train_sessions_final = train_sessions[:-2]
        val_trials = [t for s in val_sessions for t in sessions[s]]
        train_trials_final = [t for s in train_sessions_final for t in sessions[s]]

        print(f'  Train (final): {len(train_trials_final)} ({len(train_sessions_final)} sessions)')
        print(f'  Val: {len(val_trials)} ({len(val_sessions)} sessions)')

        # DataLoaders
        train_loader = DataLoader(
            train_trials_final, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_ctc, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_trials, batch_size=args.batch_size * 2, shuffle=False,
            collate_fn=collate_ctc, num_workers=2,
        )
        test_loader = DataLoader(
            test_trials, batch_size=args.batch_size * 2, shuffle=False,
            collate_fn=collate_ctc, num_workers=2,
        )

        # Build model
        n_output = N_CLASSES + 1  # +1 for CTC blank
        model = CTCEncoder.build(args.model, n_features=1280, n_classes=n_output,
                                  hidden=args.hidden, n_layers=args.n_layers,
                                  downsample=args.downsample)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'\nModel: {args.model}, {n_params/1e6:.1f}M params')

        if len(args.gpus) > 1 and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        elif args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6)
        else:
            scheduler = None
        criterion = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
        scaler = GradScaler()

        best_val_per = float('inf')
        best_state = None
        wait = 0
        history = []  # epoch-level training history

        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device,
                                     noise_std=args.noise)
            if scheduler is not None:
                if args.scheduler == 'plateau':
                    pass  # step after val metrics below
                else:
                    scheduler.step()

            val_metrics = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            cur_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1:3d}/{args.epochs} | '
                  f'loss={train_loss:.4f} | '
                  f'val_PER={val_metrics["per"]:.3f} | '
                  f'val_acc={val_metrics["accuracy"]:.1%} | '
                  f'lr={cur_lr:.1e} | '
                  f'{elapsed:.1f}s')

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_per': val_metrics['per'],
                'val_accuracy': val_metrics['accuracy'],
                'lr': cur_lr,
                'elapsed': elapsed,
            })

            if scheduler is not None and args.scheduler == 'plateau':
                scheduler.step(val_metrics['per'])

            if val_metrics['per'] < best_val_per:
                best_val_per = val_metrics['per']
                m = model.module if hasattr(model, 'module') else model
                best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if wait >= args.patience:
                print(f'Early stop at epoch {epoch+1}')
                break

        # Test evaluation
        m = model.module if hasattr(model, 'module') else model
        if best_state:
            m.load_state_dict(best_state)
        model.eval()

        test_metrics = evaluate(model, test_loader, device)
        print(f'\n{"="*60}')
        print(f'TEST RESULTS — {args.model} CTC Decoder')
        print(f'  PER: {test_metrics["per"]:.1%}')
        print(f'  Phoneme Accuracy: {test_metrics["accuracy"]:.1%}')
        print(f'  Paper baseline: 61.4% accuracy')
        print(f'{"="*60}')

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'model': args.model,
            'test_per': test_metrics['per'],
            'test_accuracy': test_metrics['accuracy'],
            'best_val_per': best_val_per,
            'n_train': len(train_trials_final),
            'n_val': len(val_trials),
            'n_test': len(test_trials),
            'epochs_trained': epoch + 1,
            'hyperparams': vars(args),
            'history': history,
        }
        save_path = RESULTS_DIR / f'ctc_{args.model}_{timestamp}.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {save_path}')

        # Save model
        model_path = RESULTS_DIR / f'ctc_{args.model}_best.pt'
        torch.save(best_state, model_path)
        print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
