#!/usr/bin/env python3
"""
Track B: Beyond Paper — Novel methods to beat 19.7% PER.

Builds on the paper-exact architecture but removes the real-time constraint:
- Bidirectional GRU (paper uses unidirectional for real-time)
- 1280D features (paper uses 256D area 6v only)
- Conformer encoder
- Larger models, curriculum learning, data augmentation

Usage:
    # B1: Bidirectional GRU with 256D features
    CUDA_VISIBLE_DEVICES=4,5 python train_beyond_paper.py \
        --experiment B1_bidir_256d --gpus 0 1 \
        --data sentences_paper_256d.h5 --bidirectional

    # B2: 1280D features with correct architecture
    CUDA_VISIBLE_DEVICES=6,7 python train_beyond_paper.py \
        --experiment B2_1280d_bidir --gpus 0 1 \
        --data sentences_paper_1280d.h5 --bidirectional --input-dim 1280

    # B3a: Larger GRU
    CUDA_VISIBLE_DEVICES=4 python train_beyond_paper.py \
        --experiment B3a_large_gru --gpus 0 \
        --data sentences_paper_256d.h5 --bidirectional \
        --hidden 768 --n-layers 6

    # B3c: Conformer
    CUDA_VISIBLE_DEVICES=6 python train_beyond_paper.py \
        --experiment B3c_conformer --gpus 0 \
        --data sentences_paper_256d.h5 --model conformer
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

DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")

SEED = 42
N_CLASSES = 40
CTC_BLANK = N_CLASSES  # 40


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def causal_gaussian_smooth(features, sd_bins=2, delay_bins=8):
    """Causal Gaussian smoothing with delay (paper: SD=40ms, delay=160ms)."""
    from scipy.ndimage import convolve1d
    kernel_len = delay_bins + 4 * sd_bins + 1
    t = np.arange(kernel_len)
    center = delay_bins
    kernel = np.exp(-0.5 * ((t - center) / sd_bins) ** 2)
    kernel = kernel / kernel.sum()
    result = convolve1d(features, kernel[::-1], axis=0, mode='constant', cval=0.0)
    return result.astype(np.float32)


def stack_and_stride(features, kernel_size=14, stride=4):
    """Stack consecutive bins and stride (paper: kernel=14, stride=4)."""
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


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_h5_dataset(h5_path, max_trials=None, smooth_sigma=2,
                    kernel_size=14, stride=4, causal_smooth=True,
                    load_raw=False):
    """Load data. If load_raw=True, skip stacking (for PaperGRUDecoder)."""
    trials = []
    with h5py.File(h5_path, 'r') as f:
        n_trials = f.attrs['n_trials']
        if max_trials:
            n_trials = min(n_trials, max_trials)
        t_start = time.time()
        for i in range(n_trials):
            if i % 2000 == 0 and i > 0:
                print(f"  Loading {i}/{n_trials} ({time.time()-t_start:.1f}s)...",
                      flush=True)
            grp = f[f'trial_{i:05d}']
            features = grp['features'][:].astype(np.float32)

            # Causal Gaussian smoothing
            if causal_smooth and smooth_sigma > 0:
                features = causal_gaussian_smooth(features, sd_bins=int(smooth_sigma),
                                                  delay_bins=8)

            if load_raw:
                # Keep raw features for models that stack internally
                trial = {
                    'features_raw': features,
                    'phoneme_indices': grp['phoneme_indices'][:],
                    'session': grp.attrs['session'],
                    'text': grp.attrs['text'],
                    'n_frames_raw': features.shape[0],
                    'n_phonemes': grp.attrs['n_phonemes'],
                }
            else:
                features_stacked = stack_and_stride(features, kernel_size, stride)
                trial = {
                    'features_stacked': features_stacked,
                    'phoneme_indices': grp['phoneme_indices'][:],
                    'session': grp.attrs['session'],
                    'text': grp.attrs['text'],
                    'n_frames': features_stacked.shape[0],
                    'n_frames_raw': features.shape[0],
                    'n_phonemes': grp.attrs['n_phonemes'],
                }
            trials.append(trial)

    elapsed = time.time() - t_start
    if load_raw:
        dim = trials[0]['features_raw'].shape[1]
        avg_len = np.mean([t['features_raw'].shape[0] for t in trials])
        print(f"Loaded {len(trials)} trials ({elapsed:.1f}s) — raw {dim}D, avg {avg_len:.0f} frames")
    else:
        dim = trials[0]['features_stacked'].shape[1]
        avg_len = np.mean([t['features_stacked'].shape[0] for t in trials])
        print(f"Loaded {len(trials)} trials ({elapsed:.1f}s) — stacked {dim}D, avg {avg_len:.0f} steps")
    return trials


# ═══════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F


class DaySpecificInputLayer(nn.Module):
    """Per-session affine + softsign, applied per-frame BEFORE stacking."""
    def __init__(self, n_features, hidden, n_sessions, dropout=0.4):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.dropout_pre = nn.Dropout(dropout)
        self.dropout_post = nn.Dropout(dropout)
        self.day_weights = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, hidden))
            for _ in range(n_sessions)
        ])
        self.day_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden))
            for _ in range(n_sessions)
        ])
        self.shared_weight = nn.Parameter(torch.empty(n_features, hidden))
        self.shared_bias = nn.Parameter(torch.zeros(hidden))
        self._train_sids = set(range(n_sessions))

        for w in list(self.day_weights) + [self.shared_weight]:
            nn.init.xavier_uniform_(w)

    def set_train_sessions(self, sids):
        self._train_sids = set(sids)

    def forward(self, x, session_ids):
        """x: (B, T, C_in), session_ids: (B,)"""
        B, T, C = x.shape
        out = torch.zeros(B, T, self.hidden, device=x.device, dtype=x.dtype)
        for i in range(B):
            sid = session_ids[i].item()
            use_shared = (sid not in self._train_sids)
            if self.training and not use_shared and torch.rand(1).item() < 0.2:
                use_shared = True
            if use_shared or sid < 0 or sid >= len(self.day_weights):
                w, b = self.shared_weight, self.shared_bias
            else:
                w, b = self.day_weights[sid], self.day_biases[sid]
            xi = self.dropout_pre(x[i])
            projected = F.linear(xi, w.T, b)
            activated = projected / (projected.abs() + 1)  # softsign
            out[i] = self.dropout_post(activated)
        return out


class BeyondPaperGRU(nn.Module):
    """Paper architecture with bidirectional option.

    Architecture:
    Raw features → Day-specific(input_dim→256, softsign) per-frame →
    Stack 14 frames → 3584D → Linear(3584→hidden) → LayerNorm →
    N-layer GRU (uni/bidir) → Dropout → Linear → CTC
    """
    def __init__(self, n_features_per_frame, n_classes=41,
                 hidden=512, n_layers=5, dropout=0.4,
                 n_sessions=24, kernel_size=14, stride=4,
                 bidirectional=True, day_hidden=256):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.bidirectional = bidirectional

        # Day-specific: input_dim → day_hidden per frame
        self.day_input = DaySpecificInputLayer(
            n_features_per_frame, day_hidden, n_sessions, dropout=dropout
        )

        # After stacking: kernel_size * day_hidden → GRU
        gru_input_dim = kernel_size * day_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(gru_input_dim, hidden),
            nn.LayerNorm(hidden),
        )

        self.rnn = nn.GRU(
            hidden, hidden, n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
        )

        rnn_out = hidden * (2 if bidirectional else 1)
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out, n_classes),
        )

    def set_train_sessions(self, sids):
        self.day_input.set_train_sessions(sids)

    def forward(self, x, session_ids):
        """x: (B, T_raw, input_dim) raw features."""
        # Day-specific per frame
        x = self.day_input(x, session_ids)  # (B, T_raw, day_hidden)

        # Stack frames
        B, T, H = x.shape
        T_new = (T - self.kernel_size) // self.stride + 1
        if T_new <= 0:
            T_new = 1
        indices = torch.arange(T_new, device=x.device) * self.stride
        stacked_list = []
        for k in range(self.kernel_size):
            idx = (indices + k).clamp(max=T - 1)
            stacked_list.append(x[:, idx, :])
        x = torch.cat(stacked_list, dim=-1)  # (B, T_new, kernel_size * day_hidden)

        x = self.input_proj(x)
        x, _ = self.rnn(x)
        return self.output_proj(x)


class ConformerBlock(nn.Module):
    """Conformer block: FFN → MHSA → Conv → FFN → LayerNorm."""
    def __init__(self, d_model, n_heads=8, conv_kernel=31, dropout=0.1,
                 ff_expansion=4):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        # Depthwise conv
        self.conv_ln = nn.LayerNorm(d_model)
        padding = (conv_kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),  # pointwise expansion
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, conv_kernel, padding=padding,
                      groups=d_model),  # depthwise
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),  # pointwise
            nn.Dropout(dropout),
        )

        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # Half-step FFN
        x = x + 0.5 * self.ff1(x)
        # MHSA
        residual = x
        x_ln = self.attn_ln(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = residual + self.attn_drop(attn_out)
        # Conv
        residual = x
        x_ln = self.conv_ln(x).transpose(1, 2)  # (B, C, T)
        x = residual + self.conv(x_ln).transpose(1, 2)
        # Half-step FFN
        x = x + 0.5 * self.ff2(x)
        return self.final_ln(x)


class ConformerDecoder(nn.Module):
    """Conformer encoder with day-specific input + stacking."""
    def __init__(self, n_features_per_frame, n_classes=41,
                 d_model=256, n_layers=6, n_heads=8, dropout=0.1,
                 n_sessions=24, kernel_size=14, stride=4,
                 day_hidden=256, conv_kernel=31):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.day_input = DaySpecificInputLayer(
            n_features_per_frame, day_hidden, n_sessions, dropout=0.4
        )

        stacked_dim = kernel_size * day_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(stacked_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, conv_kernel, dropout)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def set_train_sessions(self, sids):
        self.day_input.set_train_sessions(sids)

    def forward(self, x, session_ids):
        """x: (B, T_raw, input_dim)."""
        x = self.day_input(x, session_ids)

        # Stack
        B, T, H = x.shape
        T_new = max(1, (T - self.kernel_size) // self.stride + 1)
        indices = torch.arange(T_new, device=x.device) * self.stride
        stacked = []
        for k in range(self.kernel_size):
            idx = (indices + k).clamp(max=T - 1)
            stacked.append(x[:, idx, :])
        x = torch.cat(stacked, dim=-1)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


# ═══════════════════════════════════════════════════════════════════════
# COLLATION
# ═══════════════════════════════════════════════════════════════════════

def collate_raw(batch):
    """Collate raw (un-stacked) features."""
    features = [torch.FloatTensor(t['features_raw']) for t in batch]
    targets = [torch.IntTensor(t['phoneme_indices']) for t in batch]
    sessions = [t.get('session_idx', 0) for t in batch]

    feature_lengths = torch.IntTensor([f.shape[0] for f in features])
    target_lengths = torch.IntTensor([t.shape[0] for t in targets])

    max_T = max(f.shape[0] for f in features)
    C = features[0].shape[1]
    padded = torch.zeros(len(features), max_T, C)
    for i, f in enumerate(features):
        padded[i, :f.shape[0], :] = f

    targets_cat = torch.cat(targets)
    session_ids = torch.LongTensor(sessions)

    return padded, targets_cat, feature_lengths, target_lengths, session_ids


def collate_stacked(batch):
    """Collate pre-stacked features."""
    features = [torch.FloatTensor(t['features_stacked']) for t in batch]
    targets = [torch.IntTensor(t['phoneme_indices']) for t in batch]
    sessions = [t.get('session_idx', 0) for t in batch]

    feature_lengths = torch.IntTensor([f.shape[0] for f in features])
    target_lengths = torch.IntTensor([t.shape[0] for t in targets])

    max_T = max(f.shape[0] for f in features)
    C = features[0].shape[1]
    padded = torch.zeros(len(features), max_T, C)
    for i, f in enumerate(features):
        padded[i, :f.shape[0], :] = f

    targets_cat = torch.cat(targets)
    session_ids = torch.LongTensor(sessions)

    return padded, targets_cat, feature_lengths, target_lengths, session_ids


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

def ctc_greedy_decode(log_probs, blank=CTC_BLANK):
    best_path = log_probs.argmax(axis=1)
    decoded = []
    prev = -1
    for t in best_path:
        if t != prev:
            if t != blank:
                decoded.append(int(t))
        prev = t
    return decoded


def compute_per(predicted, target):
    import editdistance
    if len(target) == 0:
        return 1.0 if len(predicted) > 0 else 0.0
    return editdistance.eval(predicted, target) / len(target)


def train_epoch(model, dataloader, optimizer, criterion, scaler, device,
                white_noise_sd=1.0, offset_noise_sd=0.2, kernel_size=14,
                stride=4, use_raw=True):
    """Train one epoch with paper's noise augmentation."""
    from torch.amp import autocast
    model.train()
    total_loss = 0
    n_batches = 0

    for features, targets, feat_lens, tgt_lens, session_ids in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        session_ids = session_ids.to(device)

        # Paper noise augmentation
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

            # Compute output lengths based on model type
            if use_raw:
                # Model stacks internally: output T = (input_T - kernel) // stride + 1
                out_lens = ((feat_lens - kernel_size) // stride + 1).clamp(min=1)
                input_lengths = out_lens.clamp(max=log_probs.shape[1]).to(device)
            else:
                input_lengths = feat_lens.clamp(max=log_probs.shape[1]).to(device)

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


def evaluate_model(model, dataloader, device):
    from torch.amp import autocast
    model.eval()
    all_per = []

    with torch.no_grad():
        for features, targets, feat_lens, tgt_lens, session_ids in dataloader:
            features = features.to(device)
            session_ids = session_ids.to(device)

            with autocast("cuda"):
                logits = model(features, session_ids=session_ids)
                log_probs = logits.log_softmax(dim=2).cpu().numpy()

            T_out = log_probs.shape[1]
            offset = 0
            for i in range(len(feat_lens)):
                T = min(feat_lens[i].item(), T_out)
                P = tgt_lens[i].item()
                decoded = ctc_greedy_decode(log_probs[i, :T, :])
                trial_targets = targets[offset:offset + P].numpy().tolist()
                offset += P
                all_per.append(compute_per(decoded, trial_targets))

    return {'per': float(np.mean(all_per)) if all_per else 1.0}


def main():
    parser = argparse.ArgumentParser(description='Track B: Beyond Paper')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (e.g., B1_bidir_256d)')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--data', type=str, default='sentences_paper_256d.h5',
                        help='HDF5 data file in data/')
    parser.add_argument('--input-dim', type=int, default=256,
                        help='Feature dimension per frame (256 or 1280)')

    # Model
    parser.add_argument('--model', type=str, default='gru',
                        choices=['gru', 'conformer'],
                        help='Model architecture')
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--day-hidden', type=int, default=256,
                        help='Day-specific layer output dim')
    parser.add_argument('--n-heads', type=int, default=8,
                        help='Attention heads (conformer)')
    parser.add_argument('--conv-kernel', type=int, default=31,
                        help='Conv kernel size (conformer)')

    # Training
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-minibatches', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--adam-eps', type=float, default=0.1)
    parser.add_argument('--l2-reg', type=float, default=1e-5)
    parser.add_argument('--white-noise', type=float, default=1.0)
    parser.add_argument('--offset-noise', type=float, default=0.2)
    parser.add_argument('--scheduler', type=str, default='linear',
                        choices=['linear', 'plateau', 'cosine'])
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='LR warmup steps')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)

    # Stacking
    parser.add_argument('--kernel-size', type=int, default=14)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--smooth', type=float, default=2)

    parser.add_argument('--max-trials', type=int, default=None)

    # Data augmentation
    parser.add_argument('--specaugment', action='store_true',
                        help='Apply SpecAugment-style masking')
    parser.add_argument('--time-stretch', action='store_true',
                        help='Random time stretching ±10%')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))

    from torch.utils.data import DataLoader
    from torch.amp import GradScaler

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'TRACK B: {args.experiment}')
    print(f'  Model: {args.model} ({"bidir" if args.bidirectional else "unidir"})')
    print(f'  Data: {args.data} ({args.input_dim}D)')
    print(f'  Hidden: {args.hidden}, Layers: {args.n_layers}')
    print(f'  LR: {args.lr}, Eps: {args.adam_eps}, Dropout: {args.dropout}')
    print(f'  Noise: white={args.white_noise}, offset={args.offset_noise}')
    print(f'  Seed: {args.seed}')
    print('=' * 70)

    # Load data (raw, model handles stacking)
    h5_path = DATA_DIR / args.data
    if not h5_path.exists():
        print(f'ERROR: {h5_path} not found')
        print(f'  Run: python preprocess_from_h5.py')
        sys.exit(1)

    trials = load_h5_dataset(
        h5_path, max_trials=args.max_trials,
        smooth_sigma=args.smooth,
        kernel_size=args.kernel_size, stride=args.stride,
        causal_smooth=True, load_raw=True,
    )

    # Build session index
    session_names = sorted(set(t['session'] for t in trials))
    session_to_idx = {s: i for i, s in enumerate(session_names)}
    n_sessions = len(session_names)
    for t in trials:
        t['session_idx'] = session_to_idx[t['session']]

    # Split by session (same split as train_paper_replica.py)
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
    print(f'\nSessions: {n_sessions} ({len(train_sessions)} train, '
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

    if args.model == 'gru':
        model = BeyondPaperGRU(
            n_features_per_frame=args.input_dim,
            n_classes=n_output,
            hidden=args.hidden,
            n_layers=args.n_layers,
            dropout=args.dropout,
            n_sessions=n_sessions,
            kernel_size=args.kernel_size,
            stride=args.stride,
            bidirectional=args.bidirectional,
            day_hidden=args.day_hidden,
        )
    elif args.model == 'conformer':
        model = ConformerDecoder(
            n_features_per_frame=args.input_dim,
            n_classes=n_output,
            d_model=args.hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            n_sessions=n_sessions,
            kernel_size=args.kernel_size,
            stride=args.stride,
            day_hidden=args.day_hidden,
            conv_kernel=args.conv_kernel,
        )

    model.set_train_sessions(train_session_idxs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {args.model} ({n_params/1e6:.1f}M params)')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), eps=args.adam_eps,
        weight_decay=args.l2_reg,
    )

    batches_per_epoch = len(train_loader)
    total_epochs = max(1, args.max_minibatches // batches_per_epoch)

    # Scheduler with warmup
    if args.scheduler == 'plateau':
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)
    elif args.scheduler == 'cosine':
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6)
    else:  # linear
        base_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.001,
            total_iters=total_epochs)

    # Warmup
    if args.warmup_steps > 0 and args.scheduler != 'plateau':
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=args.warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, base_scheduler],
            milestones=[args.warmup_steps])
        use_plateau = False
    elif args.scheduler == 'plateau':
        scheduler = base_scheduler
        use_plateau = True
    else:
        scheduler = base_scheduler
        use_plateau = False

    print(f'Training: {total_epochs} epochs, {batches_per_epoch} batches/epoch')
    print(f'Scheduler: {args.scheduler}, warmup: {args.warmup_steps} steps')

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
            kernel_size=args.kernel_size, stride=args.stride, use_raw=True,
        )
        val_metrics = evaluate_model(model, val_loader, device)

        if use_plateau:
            scheduler.step(val_metrics['per'])
        else:
            scheduler.step()

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1:3d}/{total_epochs} | '
              f'loss={train_loss:.4f} | '
              f'val_PER={val_metrics["per"]:.3f} | '
              f'lr={cur_lr:.2e} | '
              f'{elapsed:.1f}s')

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_per': val_metrics['per'],
            'lr': cur_lr,
            'elapsed': elapsed,
        })

        if val_metrics['per'] < best_val_per:
            best_val_per = val_metrics['per']
            m = model.module if hasattr(model, 'module') else model
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            wait = 0
            print(f'  *** New best val PER: {best_val_per:.3f} ***')
        else:
            wait += 1

        if wait >= args.patience:
            print(f'Early stop at epoch {epoch+1}')
            break

    # Test
    m = model.module if hasattr(model, 'module') else model
    if best_state:
        m.load_state_dict(best_state)
    model.eval()

    test_metrics = evaluate_model(model, test_loader, device)

    print(f'\n{"="*70}')
    print(f'RESULTS — {args.experiment}')
    print(f'  Val PER:  {best_val_per:.1%}')
    print(f'  Test PER: {test_metrics["per"]:.1%}')
    print(f'  Paper:    19.7% PER (unidirectional, 256D)')
    print(f'  Previous: 51.9% PER (bidir GRU, 1280D, wrong arch)')
    print(f'{"="*70}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'experiment': args.experiment,
        'track': 'B',
        'model': args.model,
        'test_per': test_metrics['per'],
        'best_val_per': float(best_val_per),
        'n_train': len(train_trials),
        'n_val': len(val_trials),
        'n_test': len(test_trials),
        'epochs_trained': epoch + 1,
        'n_params': n_params,
        'hyperparams': vars(args),
        'history': history,
    }

    save_path = RESULTS_DIR / f'trackB_{args.experiment}_{timestamp}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results: {save_path}')

    model_path = RESULTS_DIR / f'trackB_{args.experiment}_best.pt'
    torch.save(best_state, model_path)
    print(f'Model: {model_path}')


if __name__ == '__main__':
    main()
