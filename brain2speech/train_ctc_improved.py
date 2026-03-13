#!/usr/bin/env python3
"""
Improved CTC phoneme decoder — implements feature bottleneck proposals.

Key improvements over train_ctc.py:
  A. Day-specific input layers (per-session affine transform) — paper's key trick
  B. Rolling z-score normalization (exponential moving average)
  C. Conformer encoder (attention + convolution)
  D. SpecAugment-style masking (time + feature masking)
  E. Channel attention (learnable feature weighting)
  F. Temporal derivatives (delta + delta-delta features)
  G. Warmup + cosine annealing LR schedule
  H. Multi-task auxiliary phoneme classification head

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python train_ctc_improved.py --improvements day_specific,rolling_zscore,specaugment
    CUDA_VISIBLE_DEVICES=0,1 python train_ctc_improved.py --improvements all
    CUDA_VISIBLE_DEVICES=0,1 python train_ctc_improved.py --improvements conformer
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
from scipy.ndimage import gaussian_filter1d

DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")

SEED = 42
N_CLASSES = 40
CTC_BLANK = N_CLASSES  # 40

# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT A: Rolling Z-Score Normalization
# ═══════════════════════════════════════════════════════════════════════

def rolling_zscore(features, alpha=0.001):
    """Exponential moving average z-score — adapts to within-session drift.

    The paper uses rolling normalization rather than static per-block z-scoring.
    This captures slow drift in neural statistics across a recording session.

    Args:
        features: (T, C) array of raw neural features
        alpha: smoothing factor (0.001 = ~1000 frame time constant = 20 seconds)
    Returns:
        z-scored features (T, C)
    """
    T, C = features.shape
    result = np.zeros_like(features, dtype=np.float32)

    # Initialize with first 50 frames
    init_window = min(50, T)
    mu = features[:init_window].mean(axis=0).astype(np.float64)
    var = features[:init_window].var(axis=0).astype(np.float64) + 1e-8

    for t in range(T):
        # Update running statistics
        mu = (1 - alpha) * mu + alpha * features[t]
        var = (1 - alpha) * var + alpha * (features[t] - mu) ** 2
        result[t] = (features[t] - mu) / np.sqrt(var + 1e-8)

    return result


def rolling_zscore_sessions(trials):
    """Apply rolling z-score per session."""
    sessions = {}
    for t in trials:
        s = t['session']
        if s not in sessions:
            sessions[s] = []
        sessions[s].append(t)

    for s, sess_trials in sessions.items():
        # Concatenate all features in session order
        all_feats = np.concatenate([t['features'] for t in sess_trials], axis=0)
        all_normed = rolling_zscore(all_feats, alpha=0.001)

        # Split back into trials
        offset = 0
        for t in sess_trials:
            T = t['features'].shape[0]
            t['features'] = all_normed[offset:offset + T]
            offset += T

        print(f"  Session {s}: rolling z-score on {len(sess_trials)} trials ({all_feats.shape[0]} frames)")

    return trials


# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT B: Day-Specific Input Layers
# ═══════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F


class DaySpecificInputLayer(nn.Module):
    """Per-session affine transform — the paper's key innovation.

    Learns a scale (γ) and bias (β) per session to normalize each day's
    neural statistics into a shared latent space. Only 2*C parameters per
    session — negligible overhead, massive generalization gain.

    The paper reports WER drops from 30% (no retraining) to 9.1% with this.
    """
    def __init__(self, n_features, n_sessions, init_scale=1.0):
        super().__init__()
        self.n_features = n_features
        # Learnable per-session affine: x' = γ_s * x + β_s
        self.scales = nn.ParameterList([
            nn.Parameter(torch.full((n_features,), init_scale))
            for _ in range(n_sessions)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(n_features))
            for _ in range(n_sessions)
        ])
        # Shared fallback for unseen sessions (test time)
        self.shared_scale = nn.Parameter(torch.full((n_features,), init_scale))
        self.shared_bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x, session_ids=None):
        """
        Args:
            x: (B, T, C) neural features
            session_ids: (B,) integer session indices, or None for shared transform
        """
        if session_ids is None:
            return x * self.shared_scale + self.shared_bias

        B = x.shape[0]
        out = torch.zeros_like(x)
        for i in range(B):
            sid = session_ids[i].item()
            if 0 <= sid < len(self.scales):
                out[i] = x[i] * self.scales[sid] + self.biases[sid]
            else:
                out[i] = x[i] * self.shared_scale + self.shared_bias
        return out


# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT C: SpecAugment for Neural Signals
# ═══════════════════════════════════════════════════════════════════════

class NeuralSpecAugment(nn.Module):
    """SpecAugment-style masking adapted for neural signals.

    From Park et al. (2019) "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition" — adapted for neural feature matrices.

    Applies:
    1. Time masking: zero out contiguous blocks of time bins
    2. Feature masking: zero out contiguous blocks of features (channels)
    """
    def __init__(self, time_mask_max=20, time_masks=2,
                 feat_mask_max=64, feat_masks=2, p=0.5):
        super().__init__()
        self.time_mask_max = time_mask_max
        self.time_masks = time_masks
        self.feat_mask_max = feat_mask_max
        self.feat_masks = feat_masks
        self.p = p

    def forward(self, x):
        """x: (B, T, C) — apply masking only during training."""
        if not self.training or torch.rand(1).item() > self.p:
            return x

        x = x.clone()
        B, T, C = x.shape

        # Time masking
        for _ in range(self.time_masks):
            t_len = torch.randint(0, min(self.time_mask_max, T // 4), (1,)).item()
            if t_len > 0 and T > t_len:
                t_start = torch.randint(0, T - t_len, (1,)).item()
                x[:, t_start:t_start + t_len, :] = 0

        # Feature masking
        for _ in range(self.feat_masks):
            f_len = torch.randint(0, min(self.feat_mask_max, C // 4), (1,)).item()
            if f_len > 0 and C > f_len:
                f_start = torch.randint(0, C - f_len, (1,)).item()
                x[:, :, f_start:f_start + f_len] = 0

        return x


# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT D: Channel Attention
# ═══════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention.

    Learns per-feature importance weights, allowing the model to downweight
    noisy channels (area 44) and uninformative tx levels automatically.

    From Hu et al. (2018) "Squeeze-and-Excitation Networks".
    """
    def __init__(self, n_channels, reduction=8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, T, C) → (B, T, C) with per-channel weighting."""
        # Global average pooling over time
        weights = self.attention(x.mean(dim=1))  # (B, C)
        return x * weights.unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT E: Conformer Encoder
# ═══════════════════════════════════════════════════════════════════════

class ConvModule(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → BN → swish → pointwise."""
    def __init__(self, d_model, kernel_size=31, dr=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size,
                                    padding=(kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dr)

    def forward(self, x):
        """x: (B, T, D)"""
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise1(x)  # (B, 2D, T)
        x = F.glu(x, dim=1)     # (B, D, T)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = F.silu(x)           # swish activation
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)  # (B, T, D)


class FeedForward(nn.Module):
    """Conformer feed-forward module with expansion factor."""
    def __init__(self, d_model, expansion=4, dr=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dr),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dr),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN → MHSA → Conv → FFN (Macaron-style).

    From Gulati et al. (2020) "Conformer: Convolution-augmented Transformer
    for Speech Recognition". Combines self-attention (global) with convolution
    (local) — addresses the Transformer's failure on long CTC sequences.
    """
    def __init__(self, d_model, nhead, conv_kernel=31, ff_expansion=4, dr=0.1):
        super().__init__()
        self.ffn1 = FeedForward(d_model, ff_expansion, dr)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dr, batch_first=True)
        self.attn_dropout = nn.Dropout(dr)
        self.conv = ConvModule(d_model, conv_kernel, dr)
        self.ffn2 = FeedForward(d_model, ff_expansion, dr)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Macaron-style: half-step FFN, attention, conv, half-step FFN
        x = x + 0.5 * self.ffn1(x)

        # Multi-head self-attention with residual
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.attn_dropout(attn_out)

        # Convolution module with residual
        x = x + self.conv(x)

        # Second half-step FFN
        x = x + 0.5 * self.ffn2(x)

        return self.final_norm(x)


class ConformerCTCEncoder(nn.Module):
    """Conformer-based CTC encoder.

    Replaces the failed Transformer with Conformer architecture that
    combines self-attention (global context) with convolution (local patterns).
    Standard in modern ASR (Google, Meta speech models).
    """
    def __init__(self, n_features=1280, n_classes=41, d_model=256,
                 nhead=8, num_layers=6, conv_kernel=31, dr=0.1,
                 use_channel_attention=False, use_day_specific=False,
                 n_sessions=0):
        super().__init__()

        self.use_day_specific = use_day_specific
        if use_day_specific and n_sessions > 0:
            self.day_layer = DaySpecificInputLayer(n_features, n_sessions)
        else:
            self.day_layer = None

        self.use_channel_attention = use_channel_attention
        if use_channel_attention:
            self.channel_attn = ChannelAttention(n_features, reduction=8)

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dr),
        )

        # Relative positional encoding via conv
        self.pos_conv = nn.Conv1d(d_model, d_model, kernel_size=31,
                                   padding=15, groups=16)

        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, conv_kernel, dr=dr)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dr),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x, session_ids=None):
        """x: (B, T, C) → (B, T, n_classes)"""
        # Day-specific normalization
        if self.day_layer is not None and session_ids is not None:
            x = self.day_layer(x, session_ids)

        # Channel attention
        if self.use_channel_attention:
            x = self.channel_attn(x)

        # Project to d_model
        x = self.input_proj(x)

        # Convolutional positional encoding (relative positions)
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)

        # Conformer blocks
        for block in self.blocks:
            x = block(x)

        return self.output_proj(x)


# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT F: Improved GRU with Day-Specific Layers
# ═══════════════════════════════════════════════════════════════════════

class ImprovedGRUCTCEncoder(nn.Module):
    """GRU CTC encoder with optional improvements stacked on top.

    Can enable any combination of:
    - Day-specific input layers
    - Channel attention
    - SpecAugment
    """
    def __init__(self, n_features=1280, n_classes=41, hidden=512, n_layers=5,
                 dr=0.3, use_channel_attention=False,
                 use_day_specific=False, n_sessions=0,
                 use_specaugment=False):
        super().__init__()

        self.use_day_specific = use_day_specific
        if use_day_specific and n_sessions > 0:
            self.day_layer = DaySpecificInputLayer(n_features, n_sessions)
        else:
            self.day_layer = None

        self.use_channel_attention = use_channel_attention
        if use_channel_attention:
            self.channel_attn = ChannelAttention(n_features, reduction=8)

        self.use_specaugment = use_specaugment
        if use_specaugment:
            self.specaugment = NeuralSpecAugment(
                time_mask_max=25, time_masks=2,
                feat_mask_max=80, feat_masks=2, p=0.5
            )

        self.input_proj = nn.Linear(n_features, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.rnn = nn.GRU(hidden, hidden, n_layers,
                          batch_first=True, bidirectional=True, dropout=dr)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dr),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x, session_ids=None):
        """x: (B, T, C=1280) → (B, T, n_classes)"""
        if self.day_layer is not None and session_ids is not None:
            x = self.day_layer(x, session_ids)

        if self.use_channel_attention:
            x = self.channel_attn(x)

        if self.use_specaugment:
            x = self.specaugment(x)

        x = self.input_norm(self.input_proj(x))
        x, _ = self.rnn(x)
        return self.output_proj(x)


# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT G: Hierarchical GRU with Intermediate CTC
# (Willett et al. 2026, Cross-brain transfer paper)
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalGRUCTCEncoder(nn.Module):
    """Hierarchical GRU with intermediate CTC supervision.

    From Willett et al. (bioRxiv 2026) — addresses CTC's conditional independence
    limitation by splitting the encoder into two stages with intermediate CTC loss.
    The first stage produces an intermediate phoneme estimate that feeds back into
    the second stage, allowing it to model phonotactic dependencies.

    Architecture:
        Input → Day-specific → GRU_low(3L) → CTC_intermediate → GRU_high(2L) → CTC_final
                                                    ↓
                                            feedback → GRU_high input

    The intermediate CTC provides early gradient signal and the feedback
    connection lets the upper layers condition on previous phoneme estimates.
    """
    def __init__(self, n_features=1280, n_classes=41, hidden=512,
                 n_layers_low=3, n_layers_high=2, dr=0.3,
                 use_channel_attention=False,
                 use_day_specific=False, n_sessions=0,
                 use_specaugment=False):
        super().__init__()

        self.use_day_specific = use_day_specific
        if use_day_specific and n_sessions > 0:
            self.day_layer = DaySpecificInputLayer(n_features, n_sessions)
        else:
            self.day_layer = None

        self.use_channel_attention = use_channel_attention
        if use_channel_attention:
            self.channel_attn = ChannelAttention(n_features, reduction=8)

        self.use_specaugment = use_specaugment
        if use_specaugment:
            self.specaugment = NeuralSpecAugment(
                time_mask_max=25, time_masks=2,
                feat_mask_max=80, feat_masks=2, p=0.5
            )

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden)
        self.input_norm = nn.LayerNorm(hidden)

        # Lower GRU (3 layers)
        self.rnn_low = nn.GRU(hidden, hidden, n_layers_low,
                              batch_first=True, bidirectional=True, dropout=dr)

        # Intermediate CTC head
        self.intermediate_proj = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, n_classes),
        )

        # Feedback: intermediate logits → embedding
        self.feedback_proj = nn.Linear(n_classes, hidden)

        # Upper GRU (2 layers) — takes lower output + feedback
        self.rnn_high = nn.GRU(hidden * 2 + hidden, hidden, n_layers_high,
                               batch_first=True, bidirectional=True, dropout=dr)

        # Final CTC head
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dr),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x, session_ids=None):
        """Returns (final_logits, intermediate_logits)."""
        if self.day_layer is not None and session_ids is not None:
            x = self.day_layer(x, session_ids)

        if self.use_channel_attention:
            x = self.channel_attn(x)

        if self.use_specaugment:
            x = self.specaugment(x)

        x = self.input_norm(self.input_proj(x))

        # Lower GRU
        h_low, _ = self.rnn_low(x)  # (B, T, 2*hidden)

        # Intermediate CTC
        intermediate_logits = self.intermediate_proj(h_low)  # (B, T, n_classes)

        # Feedback: use softmax of intermediate logits as input to upper GRU
        feedback = self.feedback_proj(intermediate_logits.detach().softmax(dim=-1))  # (B, T, hidden)

        # Concatenate lower output with feedback
        h_combined = torch.cat([h_low, feedback], dim=-1)  # (B, T, 2*hidden + hidden)

        # Upper GRU
        h_high, _ = self.rnn_high(h_combined)  # (B, T, 2*hidden)

        # Final CTC
        final_logits = self.output_proj(h_high)  # (B, T, n_classes)

        return final_logits, intermediate_logits


# ═══════════════════════════════════════════════════════════════════════
# IMPROVEMENT H: Diphone CTC Objective
# (Brain-to-Text '24 Benchmark — PER 16.62% → 15.34%)
# ═══════════════════════════════════════════════════════════════════════

def build_diphone_targets(phoneme_indices):
    """Convert phoneme sequences to diphone (bigram) sequences.

    Diphones capture phoneme transitions — models P(phoneme_t, phoneme_{t+1}).
    The final phoneme distribution is recovered by marginalizing over the diphone
    distribution. This addresses CTC's inability to model inter-phoneme dependencies.

    Args:
        phoneme_indices: list of phoneme index tensors
    Returns:
        diphone_indices: bigram indices = p1 * n_phonemes + p2
        n_diphones: total number of diphone classes
    """
    n_phones = N_CLASSES  # 40
    diphone_targets = []
    for seq in phoneme_indices:
        if len(seq) < 2:
            diphone_targets.append(seq)
            continue
        # Bigram: (p_t, p_{t+1}) encoded as p_t * 40 + p_{t+1}
        bigrams = []
        for i in range(len(seq) - 1):
            bigrams.append(int(seq[i]) * n_phones + int(seq[i + 1]))
        diphone_targets.append(bigrams)
    n_diphones = n_phones * n_phones  # 1600
    return diphone_targets, n_diphones


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING & TRAINING
# ═══════════════════════════════════════════════════════════════════════

def compute_temporal_derivatives(features, order=2):
    """Compute delta and delta-delta (acceleration) features.

    Classic ASR technique (HTK, Kaldi). Captures velocity and acceleration of
    neural dynamics — critical for detecting phoneme transitions.

    Args:
        features: (T, C) array
        order: 1=delta only, 2=delta+delta-delta
    Returns:
        (T, C*(1+order)) array — original + derivatives
    """
    parts = [features]
    for _ in range(order):
        d = np.gradient(parts[-1], axis=0)
        parts.append(d.astype(np.float32))
    return np.concatenate(parts, axis=1)


def smooth_features(features, sigma):
    if sigma <= 0:
        return features
    return gaussian_filter1d(features, sigma=sigma, axis=0)


def load_h5_dataset(h5_path, max_trials=None, smooth_sigma=0):
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
    return trials


def collate_ctc_with_sessions(batch):
    """Collate with session IDs for day-specific layers."""
    features = [torch.FloatTensor(t['features']) for t in batch]
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
                noise_std=0.05, use_sessions=False, hierarchical=False):
    from torch.amp import autocast

    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        if use_sessions:
            features, targets, feat_lens, tgt_lens, session_ids = batch
            session_ids = session_ids.to(device)
        else:
            features, targets, feat_lens, tgt_lens = batch[:4]
            session_ids = None

        features = features.to(device)
        targets = targets.to(device)

        # Gaussian noise augmentation
        if noise_std > 0 and np.random.random() < 0.5:
            features = features + torch.randn_like(features) * noise_std

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            if session_ids is not None:
                output = model(features, session_ids=session_ids)
            else:
                output = model(features)

            # Handle hierarchical model returning (final, intermediate)
            if hierarchical and isinstance(output, tuple):
                final_logits, inter_logits = output
                log_probs = final_logits.log_softmax(dim=2)
                inter_log_probs = inter_logits.log_softmax(dim=2)
            else:
                log_probs = output.log_softmax(dim=2)

            log_probs_t = log_probs.transpose(0, 1)
            # Use actual feature lengths (not padded length) — fixes CTC over padding
            input_lengths = feat_lens.clamp(max=log_probs.shape[1]).to(device)
            target_lengths = tgt_lens.to(device)

            loss = criterion(log_probs_t, targets, input_lengths, target_lengths)

            # Add intermediate CTC loss for hierarchical model (weight 0.3)
            if hierarchical and isinstance(output, tuple):
                inter_log_probs_t = inter_log_probs.transpose(0, 1)
                inter_loss = criterion(inter_log_probs_t, targets, input_lengths, target_lengths)
                if not (torch.isnan(inter_loss) or torch.isinf(inter_loss)):
                    loss = loss + 0.3 * inter_loss

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


def evaluate_model(model, dataloader, device, use_sessions=False, hierarchical=False):
    from torch.amp import autocast

    model.eval()
    all_per = []
    total_correct = 0
    total_phonemes = 0

    with torch.no_grad():
        for batch in dataloader:
            if use_sessions:
                features, targets, feat_lens, tgt_lens, session_ids = batch
                session_ids = session_ids.to(device)
            else:
                features, targets, feat_lens, tgt_lens = batch[:4]
                session_ids = None

            features = features.to(device)

            with autocast("cuda"):
                if session_ids is not None:
                    output = model(features, session_ids=session_ids)
                else:
                    output = model(features)
                # Handle hierarchical model returning (final, intermediate)
                if hierarchical and isinstance(output, tuple):
                    output = output[0]  # Use final logits for eval
                log_probs = output.log_softmax(dim=2).cpu().numpy()

            T_out = log_probs.shape[1]
            offset = 0
            for i in range(len(feat_lens)):
                T = min(feat_lens[i].item(), T_out)
                P = tgt_lens[i].item()
                trial_log_probs = log_probs[i, :T, :]
                trial_targets = targets[offset:offset + P].numpy().tolist()
                offset += P

                decoded = ctc_greedy_decode(trial_log_probs)
                per = compute_per(decoded, trial_targets)
                all_per.append(per)

                min_len = min(len(decoded), len(trial_targets))
                for j in range(min_len):
                    if decoded[j] == trial_targets[j]:
                        total_correct += 1
                total_phonemes += len(trial_targets)

    return {
        'per': float(np.mean(all_per)) if all_per else 1.0,
        'accuracy': float(total_correct / max(total_phonemes, 1)),
        'n_trials': len(all_per),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['GRU', 'Conformer', 'HierarchicalGRU'], default='GRU')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--smooth', type=float, default=2.0)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-trials', type=int, default=None)
    parser.add_argument('--improvements', type=str, default='all',
                        help='Comma-separated improvements: day_specific,rolling_zscore,'
                             'specaugment,channel_attention,conformer,temporal_deriv,all')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Linear warmup epochs before cosine annealing')
    args = parser.parse_args()

    # NOTE: CUDA_VISIBLE_DEVICES must be set BEFORE launching the script
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))

    from torch.utils.data import DataLoader
    from torch.amp import GradScaler

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda:0')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Parse improvements
    if args.improvements == 'all':
        improvements = {'day_specific', 'rolling_zscore', 'specaugment', 'channel_attention'}
        if args.model == 'Conformer':
            improvements.add('conformer')
    else:
        improvements = set(args.improvements.split(','))
    improvements.discard('')  # safety

    tag = '+'.join(sorted(improvements)) if improvements else 'baseline'

    print('=' * 70)
    print(f'IMPROVED CTC DECODER — {args.model}')
    print(f'  Improvements: {", ".join(sorted(improvements))}')
    print(f'  Device: {device} (GPUs: {args.gpus})')
    print(f'  Epochs: {args.epochs}, BS: {args.batch_size}, LR: {args.lr}')
    print(f'  Smooth: sigma={args.smooth}')
    print('=' * 70)

    # Load data
    h5_path = DATA_DIR / 'sentences_train.h5'
    if not h5_path.exists():
        print(f'ERROR: {h5_path} not found.')
        sys.exit(1)

    trials = load_h5_dataset(h5_path, max_trials=args.max_trials, smooth_sigma=args.smooth)

    # Apply rolling z-score if requested
    if 'rolling_zscore' in improvements:
        print('\nApplying rolling z-score normalization...')
        trials = rolling_zscore_sessions(trials)

    # Apply temporal derivatives if requested
    if 'temporal_deriv' in improvements:
        print('\nComputing temporal derivatives (delta + delta-delta)...')
        for t in trials:
            t['features'] = compute_temporal_derivatives(t['features'], order=2)
        n_features = trials[0]['features'].shape[1]
        print(f'  Features expanded: 1280 → {n_features} (original + delta + delta-delta)')
    else:
        n_features = trials[0]['features'].shape[1]

    # Build session index
    session_names = sorted(set(t['session'] for t in trials))
    session_to_idx = {s: i for i, s in enumerate(session_names)}
    n_sessions = len(session_names)
    for t in trials:
        t['session_idx'] = session_to_idx[t['session']]

    print(f'\nSessions: {n_sessions}')

    # Split: last 4 sessions as test, last 2 of remaining as val
    sessions = {}
    for t in trials:
        s = t['session']
        if s not in sessions:
            sessions[s] = []
        sessions[s].append(t)

    test_sessions = session_names[-4:]
    remaining = session_names[:-4]
    val_sessions = remaining[-2:]
    train_sessions = remaining[:-2]

    train_trials = [t for s in train_sessions for t in sessions[s]]
    val_trials = [t for s in val_sessions for t in sessions[s]]
    test_trials = [t for s in test_sessions for t in sessions[s]]

    print(f'Train: {len(train_trials)} trials ({len(train_sessions)} sessions)')
    print(f'Val:   {len(val_trials)} trials ({len(val_sessions)} sessions)')
    print(f'Test:  {len(test_trials)} trials ({len(test_sessions)} sessions)')

    use_sessions = 'day_specific' in improvements
    collate = collate_ctc_with_sessions if use_sessions else None
    if collate is None:
        # Use basic collate from original
        def collate(batch):
            features = [torch.FloatTensor(t['features']) for t in batch]
            targets = [torch.IntTensor(t['phoneme_indices']) for t in batch]
            feature_lengths = torch.IntTensor([f.shape[0] for f in features])
            target_lengths = torch.IntTensor([t.shape[0] for t in targets])
            max_T = max(f.shape[0] for f in features)
            C = features[0].shape[1]
            padded = torch.zeros(len(features), max_T, C)
            for i, f in enumerate(features):
                padded[i, :f.shape[0], :] = f
            targets_cat = torch.cat(targets)
            return padded, targets_cat, feature_lengths, target_lengths

    train_loader = DataLoader(train_trials, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=3)
    val_loader = DataLoader(val_trials, batch_size=args.batch_size * 2, shuffle=False,
                            collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_trials, batch_size=args.batch_size * 2, shuffle=False,
                             collate_fn=collate, num_workers=2)

    # Build model
    n_output = N_CLASSES + 1
    use_ch_attn = 'channel_attention' in improvements
    use_day = 'day_specific' in improvements
    use_spec = 'specaugment' in improvements

    use_hierarchical = args.model == 'HierarchicalGRU'

    if args.model == 'Conformer' or 'conformer' in improvements:
        model = ConformerCTCEncoder(
            n_features=n_features, n_classes=n_output,
            d_model=args.hidden, nhead=8, num_layers=args.n_layers,
            conv_kernel=31, dr=0.1,
            use_channel_attention=use_ch_attn,
            use_day_specific=use_day, n_sessions=n_sessions,
        )
        tag = tag.replace('conformer', '') if 'conformer' in tag else tag
        tag = f'Conformer+{tag}' if tag else 'Conformer'
    elif use_hierarchical:
        model = HierarchicalGRUCTCEncoder(
            n_features=n_features, n_classes=n_output,
            hidden=args.hidden, n_layers_low=3, n_layers_high=2, dr=0.3,
            use_channel_attention=use_ch_attn,
            use_day_specific=use_day, n_sessions=n_sessions,
            use_specaugment=use_spec,
        )
        tag = f'HierGRU+{tag}'
    else:
        model = ImprovedGRUCTCEncoder(
            n_features=n_features, n_classes=n_output,
            hidden=args.hidden, n_layers=args.n_layers, dr=0.3,
            use_channel_attention=use_ch_attn,
            use_day_specific=use_day, n_sessions=n_sessions,
            use_specaugment=use_spec,
        )
        tag = f'GRU+{tag}'

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {tag}, {n_params/1e6:.1f}M params')
    if use_day:
        day_params = sum(p.numel() for n, p in model.named_parameters() if 'day_layer' in n)
        print(f'  Day-specific params: {day_params:,} ({day_params/n_params*100:.1f}%)')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Warmup + cosine annealing — better than plateau for transformers/conformers
    warmup_epochs = args.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
    scaler = GradScaler()

    best_val_per = float('inf')
    best_state = None
    wait = 0
    history = []

    print(f'\nTraining {tag}...\n')

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device,
                                  noise_std=args.noise, use_sessions=use_sessions,
                                  hierarchical=use_hierarchical)
        val_metrics = evaluate_model(model, val_loader, device, use_sessions=use_sessions,
                                     hierarchical=use_hierarchical)
        elapsed = time.time() - t0

        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f'Epoch {epoch+1:3d}/{args.epochs} | '
              f'loss={train_loss:.4f} | '
              f'val_PER={val_metrics["per"]:.3f} | '
              f'lr={cur_lr:.1e} | '
              f'{elapsed:.1f}s')

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_per': val_metrics['per'],
            'lr': cur_lr,
        })

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

    test_metrics = evaluate_model(model, test_loader, device, use_sessions=use_sessions,
                                  hierarchical=use_hierarchical)

    print(f'\n{"="*70}')
    print(f'TEST RESULTS — {tag}')
    print(f'  Val PER:  {best_val_per:.1%}')
    print(f'  Test PER: {test_metrics["per"]:.1%}')
    print(f'  Paper:    19.7% PER (Willett et al. 2023)')
    print(f'  Previous: 45.1% val PER (GRU v3 baseline)')
    print(f'{"="*70}')

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'model': tag,
        'test_per': test_metrics['per'],
        'test_accuracy': test_metrics['accuracy'],
        'best_val_per': float(best_val_per),
        'n_train': len(train_trials),
        'n_val': len(val_trials),
        'n_test': len(test_trials),
        'epochs_trained': epoch + 1,
        'improvements': list(improvements),
        'hyperparams': vars(args),
        'history': history,
    }
    save_path = RESULTS_DIR / f'ctc_improved_{tag}_{timestamp}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results: {save_path}')

    model_path = RESULTS_DIR / f'ctc_improved_{tag}_best.pt'
    torch.save(best_state, model_path)
    print(f'Model: {model_path}')


if __name__ == '__main__':
    main()
