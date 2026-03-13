#!/usr/bin/env python3
"""
Lead 1: GRU Baselines & Literature Reproductions — Unified Training Script.

Supports ALL GRU variants via CLI presets:
  - paper:     Willett et al. 2023 (unidirectional, hidden=512, kernel=14)
  - cffan:     cffan/neural_seq_decoder (BiGRU, hidden=1024, kernel=32)
  - cibr:      CIBR-Okubo 2nd place (SGD, ortho init, batch=128)
  - linderman: Benchmark lessons (post-RNN head, speckled mask)
  - enhanced:  All techniques combined

Usage:
    # Sanity check
    CUDA_VISIBLE_DEVICES=0 python -u brain2speech/lead1_train_gru_v2.py \
      --preset paper --max-trials 50 --max-epochs 2 --experiment L1.0_sanity

    # Reproduce Willett paper
    CUDA_VISIBLE_DEVICES=0 python -u brain2speech/lead1_train_gru_v2.py \
      --preset paper --experiment L1.1_paper_exact

    # Reproduce cffan (BiGRU + kernel=32)
    CUDA_VISIBLE_DEVICES=1 python -u brain2speech/lead1_train_gru_v2.py \
      --preset cffan --experiment L1.2_cffan
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# ── Constants ──
N_CLASSES = 40   # 39 phonemes + SIL
CTC_BLANK = N_CLASSES  # 40
N_OUTPUT = N_CLASSES + 1  # 41 (including CTC blank)

# Data lives outside the git repo
DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "lead1"


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def causal_gaussian_smooth(features, sd_bins=2, delay_bins=8):
    """Causal Gaussian smoothing with delay (paper: SD=40ms, delay=160ms).

    Args:
        features: (T, C) neural features
        sd_bins: SD in 20ms bins (40ms = 2 bins)
        delay_bins: delay in 20ms bins (160ms = 8 bins)
    """
    from scipy.ndimage import convolve1d
    kernel_len = delay_bins + 4 * sd_bins + 1
    t = np.arange(kernel_len)
    center = delay_bins
    kernel = np.exp(-0.5 * ((t - center) / sd_bins) ** 2)
    kernel = kernel / kernel.sum()
    result = convolve1d(features, kernel[::-1], axis=0, mode='constant', cval=0.0)
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_h5_raw(h5_path, max_trials=None, smooth_sigma=2, causal_smooth=True,
                min_frames=0):
    """Load raw (un-stacked) features from H5. Stacking done in model forward."""
    trials = []
    with h5py.File(h5_path, 'r') as f:
        n_trials = f.attrs['n_trials']
        if max_trials:
            n_trials = min(n_trials, max_trials)
        t_start = time.time()
        skipped = 0
        for i in range(n_trials):
            if i % 2000 == 0 and i > 0:
                print(f"  Loading {i}/{n_trials} ({time.time()-t_start:.1f}s)...",
                      flush=True)
            grp = f[f'trial_{i:05d}']
            features = grp['features'][:].astype(np.float32)

            # Skip very short trials
            if min_frames > 0 and features.shape[0] < min_frames:
                skipped += 1
                continue

            # Causal Gaussian smoothing
            if causal_smooth and smooth_sigma > 0:
                features = causal_gaussian_smooth(features, sd_bins=int(smooth_sigma),
                                                  delay_bins=8)

            trials.append({
                'features_raw': features,
                'phoneme_indices': grp['phoneme_indices'][:],
                'session': grp.attrs['session'],
                'text': grp.attrs['text'],
                'n_frames_raw': features.shape[0],
                'n_phonemes': grp.attrs['n_phonemes'],
            })

    elapsed = time.time() - t_start
    dim = trials[0]['features_raw'].shape[1]
    avg_len = np.mean([t['features_raw'].shape[0] for t in trials])
    print(f"Loaded {len(trials)} trials ({elapsed:.1f}s) — raw {dim}D, "
          f"avg {avg_len:.0f} frames" +
          (f", skipped {skipped} short" if skipped else ""))
    return trials


def collate_raw(batch):
    """Collate raw (un-stacked) features with session IDs."""
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


# ═══════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════

class EfficientDaySpecificLayer(nn.Module):
    """Einsum-based day-specific input layer (Source: cffan/neural_seq_decoder).

    Per-session affine + softsign, applied per-frame BEFORE stacking.
    Uses batched einsum instead of Python loop over batch elements.

    Forward: einsum('btd,bdk->btk', x, W[session_ids]) + b[session_ids]
    Then softsign activation: x / (|x| + 1)
    """
    def __init__(self, in_dim, out_dim, n_sessions, dropout=0.4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_sessions = n_sessions
        self.dropout_pre = nn.Dropout(dropout)
        self.dropout_post = nn.Dropout(dropout)

        # Per-session weights and biases as single tensors for einsum
        self.weights = nn.Parameter(torch.empty(n_sessions, in_dim, out_dim))
        self.biases = nn.Parameter(torch.zeros(n_sessions, out_dim))

        # Shared fallback for unseen sessions
        self.shared_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.shared_bias = nn.Parameter(torch.zeros(out_dim))

        # Track train sessions
        self._train_sids = set(range(n_sessions))

        # Xavier uniform init
        for i in range(n_sessions):
            nn.init.xavier_uniform_(self.weights.data[i])
        nn.init.xavier_uniform_(self.shared_weight.data)

    def set_train_sessions(self, sids):
        self._train_sids = set(sids)

    def forward(self, x, session_ids):
        """
        Args:
            x: (B, T, in_dim) raw features
            session_ids: (B,) integer session indices
        Returns:
            (B, T, out_dim) after softsign + dropout
        """
        B, T, D = x.shape
        x = self.dropout_pre(x)

        # Check if all sessions are valid and in training set
        sids = session_ids.cpu().numpy()
        all_valid = all(0 <= s < self.n_sessions for s in sids)

        if all_valid and not self.training:
            # Fast path: batched einsum
            W = self.weights[session_ids]  # (B, in_dim, out_dim)
            b = self.biases[session_ids]   # (B, out_dim)
            projected = torch.einsum('btd,bdk->btk', x, W) + b.unsqueeze(1)
        elif all_valid and self.training:
            # Training: occasionally use shared weights (20% of time per sample)
            use_shared = torch.rand(B, device=x.device) < 0.2
            # Also use shared for non-train sessions
            for i, s in enumerate(sids):
                if s not in self._train_sids:
                    use_shared[i] = True

            # Build per-sample weight/bias
            W = self.weights[session_ids].clone()  # (B, in_dim, out_dim)
            b = self.biases[session_ids].clone()   # (B, out_dim)
            shared_mask = use_shared.nonzero(as_tuple=True)[0]
            if len(shared_mask) > 0:
                W[shared_mask] = self.shared_weight.unsqueeze(0).expand(len(shared_mask), -1, -1)
                b[shared_mask] = self.shared_bias.unsqueeze(0).expand(len(shared_mask), -1)
            projected = torch.einsum('btd,bdk->btk', x, W) + b.unsqueeze(1)
        else:
            # Fallback: loop for out-of-range session IDs
            projected = torch.zeros(B, T, self.out_dim, device=x.device, dtype=x.dtype)
            for i in range(B):
                sid = sids[i]
                if 0 <= sid < self.n_sessions and sid in self._train_sids:
                    w, b_ = self.weights[sid], self.biases[sid]
                else:
                    w, b_ = self.shared_weight, self.shared_bias
                projected[i] = x[i] @ w + b_

        # Softsign activation
        activated = projected / (projected.abs() + 1)
        return self.dropout_post(activated)


class SpeckledMask(nn.Module):
    """Speckled masking (Source: Benchmark Lessons Learned).

    Randomly zeros out individual elements of the feature map with probability p.
    More granular than SpecAugment — coordinated dropout at element level.
    """
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(torch.full_like(x, 1.0 - self.p))
        return x * mask / (1.0 - self.p)  # Scale to maintain expected value


class PostRNNHead(nn.Module):
    """Post-RNN processing head (Source: Benchmark Lessons Learned).

    LayerNorm → Dropout → Linear → GELU

    Applied between RNN output and final classification layer.
    Reported to give 9.22% → 8.00% WER improvement.
    """
    def __init__(self, rnn_dim, head_dim, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(rnn_dim),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, head_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.head(x)


class EnhancedGRU(nn.Module):
    """Unified GRU model supporting all literature enhancements.

    Architecture pipeline:
        Raw 256D (20ms bins)
        → EfficientDaySpecificLayer(256 → day_hidden, softsign)
        → Stack & stride (kernel=14/32, stride=4) inside forward()
        → Linear(day_hidden * kernel → hidden) + LayerNorm
        → Optional SpeckledMask(p=0.3)
        → N-layer GRU (uni/bidirectional)
        → Optional PostRNNHead: [LayerNorm → Dropout → Linear → GELU]
        → Output Linear(rnn_out → 41)
        → CTC loss + optional supTCon loss + optional FastEmit
    """
    def __init__(self, input_dim, n_classes=41, hidden=512, n_layers=5,
                 dropout=0.4, n_sessions=24, kernel_size=14, stride=4,
                 bidirectional=False, day_hidden=256,
                 use_post_rnn_head=False, head_dim=256,
                 use_speckled_mask=False, speckled_p=0.3,
                 ortho_init=False, return_hidden=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.bidirectional = bidirectional
        self.return_hidden = return_hidden
        self.hidden_size = hidden

        # Day-specific: input_dim → day_hidden per frame
        self.day_input = EfficientDaySpecificLayer(
            input_dim, day_hidden, n_sessions, dropout=dropout
        )

        # After stacking: kernel_size * day_hidden → hidden
        stacked_dim = kernel_size * day_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(stacked_dim, hidden),
            nn.LayerNorm(hidden),
        )

        # Optional speckled mask
        self.speckled_mask = SpeckledMask(speckled_p) if use_speckled_mask else None

        # N-layer GRU
        self.rnn = nn.GRU(
            hidden, hidden, n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
        )

        rnn_out_dim = hidden * (2 if bidirectional else 1)

        # Optional post-RNN head
        if use_post_rnn_head:
            self.post_rnn_head = PostRNNHead(rnn_out_dim, head_dim, dropout)
            output_input_dim = head_dim
        else:
            self.post_rnn_head = None
            output_input_dim = rnn_out_dim

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_input_dim, n_classes),
        )

        # Store rnn_out_dim for supTCon
        self.rnn_out_dim = rnn_out_dim

        # Optional orthogonal init for GRU (Source: CIBR-Okubo)
        if ortho_init:
            self._apply_ortho_init()

    def _apply_ortho_init(self):
        """Orthogonal init for weight_hh, Xavier for weight_ih (Source: CIBR-Okubo)."""
        for name, param in self.rnn.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def set_train_sessions(self, sids):
        self.day_input.set_train_sessions(sids)

    def compute_output_length(self, input_length):
        """Compute output sequence length after stacking."""
        return ((input_length - self.kernel_size) // self.stride + 1).clamp(min=1)

    def forward(self, x, session_ids):
        """
        Args:
            x: (B, T_raw, input_dim) raw features
            session_ids: (B,) integer session indices
        Returns:
            logits: (B, T_stacked, n_classes)
            hidden: (B, T_stacked, rnn_out_dim) if return_hidden=True
        """
        # Day-specific per frame
        x = self.day_input(x, session_ids)  # (B, T_raw, day_hidden)

        # Stack frames inside forward (so different kernel sizes share data loader)
        B, T, H = x.shape
        T_new = max(1, (T - self.kernel_size) // self.stride + 1)
        indices = torch.arange(T_new, device=x.device) * self.stride
        stacked_list = []
        for k in range(self.kernel_size):
            idx = (indices + k).clamp(max=T - 1)
            stacked_list.append(x[:, idx, :])
        x = torch.cat(stacked_list, dim=-1)  # (B, T_new, kernel_size * day_hidden)

        # Project to hidden dim
        x = self.input_proj(x)  # (B, T_new, hidden)

        # Optional speckled mask
        if self.speckled_mask is not None:
            x = self.speckled_mask(x)

        # GRU
        rnn_out, _ = self.rnn(x)  # (B, T_new, hidden * dirs)

        # Optional post-RNN head
        if self.post_rnn_head is not None:
            head_out = self.post_rnn_head(rnn_out)
            logits = self.output_proj(head_out)
        else:
            logits = self.output_proj(rnn_out)

        if self.return_hidden:
            return logits, rnn_out
        return logits


# ═══════════════════════════════════════════════════════════════════════
# AUXILIARY LOSSES
# ═══════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss on hidden states (Sources: tbenst, MONA LISA).

    Positive pairs = frames with the same phoneme label.
    Temperature τ = 0.1, weight = 0.1 in total loss.
    Subsamples to max_frames to avoid OOM on NxN similarity matrix.
    """
    def __init__(self, temperature=0.1, max_frames=2048):
        super().__init__()
        self.temperature = temperature
        self.max_frames = max_frames

    def forward(self, hidden, phoneme_labels):
        """
        Args:
            hidden: (N, D) hidden states for valid frames
            phoneme_labels: (N,) phoneme class indices for each frame
        Returns:
            scalar loss
        """
        N = hidden.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=hidden.device)

        # Subsample if too many frames
        if N > self.max_frames:
            perm = torch.randperm(N, device=hidden.device)[:self.max_frames]
            hidden = hidden[perm]
            phoneme_labels = phoneme_labels[perm]
            N = self.max_frames

        # L2 normalize
        hidden = F.normalize(hidden, dim=1)

        # Similarity matrix
        sim = torch.mm(hidden, hidden.t()) / self.temperature  # (N, N)

        # Mask: positive pairs have same label
        labels = phoneme_labels.unsqueeze(0)  # (1, N)
        pos_mask = (labels == labels.t()).float()  # (N, N)
        # Remove self-similarity
        pos_mask.fill_diagonal_(0)

        # Check we have positive pairs
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=hidden.device)

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Log-sum-exp over all negatives + positives (exclude self)
        self_mask = torch.eye(N, device=hidden.device)
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log(exp(sim_pos) / sum(exp(sim_all))) over positive pairs
        log_prob = sim - log_sum_exp
        loss = -(log_prob * pos_mask).sum() / pos_mask.sum()

        return loss


def fastemit_regularizer(log_probs, blank_idx=CTC_BLANK, lam=0.01):
    """FastEmit regularizer — penalizes blank probability (Source: Benchmark).

    L_fe = λ * mean(log_probs[:, :, blank])
    """
    return lam * log_probs[:, :, blank_idx].mean()


# ═══════════════════════════════════════════════════════════════════════
# CTC DECODE & METRICS
# ═══════════════════════════════════════════════════════════════════════

def ctc_greedy_decode(log_probs, blank=CTC_BLANK):
    """Greedy CTC decoding."""
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
    """Phoneme Error Rate via edit distance."""
    import editdistance
    if len(target) == 0:
        return 1.0 if len(predicted) > 0 else 0.0
    return editdistance.eval(predicted, target) / len(target)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def train_epoch_v2(model, dataloader, optimizer, criterion, scaler, device, args,
                   suptcon_loss_fn=None):
    """Train one epoch with all enhancements.

    Noise augmentation (paper style):
      x' = x + ε_t + φ
      where ε_t ~ N(0, white_noise_sd) per time step
      and φ ~ N(0, offset_noise_sd) constant per minibatch per feature

    Optional: supTCon loss, FastEmit regularizer.
    """
    model.train()
    total_loss = 0
    total_ctc = 0
    total_suptcon = 0
    total_fastemit = 0
    n_batches = 0

    for features, targets, feat_lens, tgt_lens, session_ids in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        session_ids = session_ids.to(device)

        # Paper-style noise augmentation
        if args.white_noise > 0:
            features = features + torch.randn_like(features) * args.white_noise
        if args.offset_noise > 0:
            B, T, C = features.shape
            offset = torch.randn(B, 1, C, device=device) * args.offset_noise
            features = features + offset

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            output = model(features, session_ids)
            if isinstance(output, tuple):
                logits, hidden = output
            else:
                logits = output
                hidden = None

            log_probs = logits.log_softmax(dim=2)
            log_probs_t = log_probs.transpose(0, 1)  # (T, B, C) for CTC

            # Compute output lengths (stacking happens inside model)
            out_lens = model.compute_output_length(feat_lens) if not hasattr(model, 'module') \
                else model.module.compute_output_length(feat_lens)
            input_lengths = out_lens.clamp(max=log_probs.shape[1]).to(device)
            target_lengths = tgt_lens.to(device)

            ctc_loss = criterion(log_probs_t, targets, input_lengths, target_lengths)

            total_batch_loss = ctc_loss

            # Optional supTCon loss
            suptcon_val = 0.0
            if suptcon_loss_fn is not None and hidden is not None and args.use_suptcon:
                # Flatten hidden states and create phoneme labels via CTC forced alignment
                # Simple approach: expand target labels to match output length
                # For now, use a rough alignment based on uniform distribution
                B_actual = logits.shape[0]
                all_hidden = []
                all_labels = []
                offset = 0
                for i in range(B_actual):
                    T_out = input_lengths[i].item()
                    n_ph = target_lengths[i].item()
                    if n_ph == 0 or T_out == 0:
                        offset += n_ph
                        continue
                    h_i = hidden[i, :T_out]  # (T_out, D)
                    ph_i = targets[offset:offset + n_ph]  # (n_ph,)
                    # Uniform alignment: repeat each phoneme T_out/n_ph times
                    indices = torch.arange(T_out, device=device) * n_ph // T_out
                    indices = indices.clamp(max=n_ph - 1)
                    labels_i = ph_i[indices]
                    all_hidden.append(h_i)
                    all_labels.append(labels_i)
                    offset += n_ph

                if all_hidden:
                    cat_hidden = torch.cat(all_hidden, dim=0)
                    cat_labels = torch.cat(all_labels, dim=0)
                    suptcon_val = suptcon_loss_fn(cat_hidden, cat_labels)
                    total_batch_loss = total_batch_loss + args.suptcon_weight * suptcon_val

            # Optional FastEmit
            fastemit_val = 0.0
            if args.use_fastemit:
                fastemit_val = fastemit_regularizer(log_probs, CTC_BLANK, args.fastemit_lambda)
                total_batch_loss = total_batch_loss + fastemit_val

        if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
            continue

        scaler.scale(total_batch_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += total_batch_loss.item()
        total_ctc += ctc_loss.item()
        if isinstance(suptcon_val, torch.Tensor):
            total_suptcon += suptcon_val.item()
        if isinstance(fastemit_val, torch.Tensor):
            total_fastemit += fastemit_val.item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        'loss': total_loss / n,
        'ctc_loss': total_ctc / n,
        'suptcon_loss': total_suptcon / n,
        'fastemit_loss': total_fastemit / n,
    }


def evaluate_model(model, dataloader, device, kernel_size, stride):
    """Evaluate with greedy CTC decoding."""
    model.eval()
    all_per = []

    with torch.no_grad():
        for features, targets, feat_lens, tgt_lens, session_ids in dataloader:
            features = features.to(device)
            session_ids = session_ids.to(device)

            with autocast("cuda"):
                output = model(features, session_ids)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                log_probs = logits.log_softmax(dim=2).float().cpu().numpy()

            # Compute output lengths
            out_lens = ((feat_lens - kernel_size) // stride + 1).clamp(min=1)

            offset = 0
            for i in range(len(feat_lens)):
                T = min(out_lens[i].item(), log_probs.shape[1])
                P = tgt_lens[i].item()
                decoded = ctc_greedy_decode(log_probs[i, :T, :])
                trial_targets = targets[offset:offset + P].numpy().tolist()
                offset += P
                all_per.append(compute_per(decoded, trial_targets))

    return {'per': float(np.mean(all_per)) if all_per else 1.0}


# ═══════════════════════════════════════════════════════════════════════
# PRESETS
# ═══════════════════════════════════════════════════════════════════════

PRESETS = {
    'paper': {
        'hidden': 512, 'n_layers': 5, 'bidirectional': False,
        'kernel_size': 14, 'stride': 4, 'day_hidden': 256,
        'optimizer': 'adam', 'lr': 0.02, 'adam_eps': 0.1,
        'weight_decay': 1e-5, 'batch_size': 64,
        'white_noise': 1.0, 'offset_noise': 0.2,
        'scheduler': 'linear', 'max_minibatches': 100000,
        'use_post_rnn_head': False, 'use_speckled_mask': False,
        'ortho_init': False, 'dropout': 0.4,
        'split': 'within-day', 'patience': 50,
    },
    'cffan': {
        'hidden': 1024, 'n_layers': 5, 'bidirectional': True,
        'kernel_size': 32, 'stride': 4, 'day_hidden': 256,
        'optimizer': 'adam', 'lr': 0.02, 'adam_eps': 0.1,
        'weight_decay': 1e-5, 'batch_size': 64,
        'white_noise': 1.0, 'offset_noise': 0.2,
        'scheduler': 'linear', 'max_minibatches': 100000,
        'use_post_rnn_head': False, 'use_speckled_mask': False,
        'ortho_init': False, 'dropout': 0.4,
        'split': 'within-day', 'patience': 50,
    },
    'cibr': {
        'hidden': 1024, 'n_layers': 5, 'bidirectional': True,
        'kernel_size': 32, 'stride': 4, 'day_hidden': 256,
        'optimizer': 'sgd', 'lr': 0.1, 'momentum': 0.9,
        'weight_decay': 1e-5, 'batch_size': 128,
        'white_noise': 0.8, 'offset_noise': 0.2,
        'scheduler': 'step', 'step_size': 4000, 'step_gamma': 0.1,
        'max_minibatches': 100000,
        'use_post_rnn_head': False, 'use_speckled_mask': False,
        'ortho_init': True, 'dropout': 0.4,
        'split': 'within-day', 'patience': 50,
    },
    'linderman': {
        'hidden': 512, 'n_layers': 5, 'bidirectional': True,
        'kernel_size': 14, 'stride': 4, 'day_hidden': 256,
        'optimizer': 'sgd', 'lr': 0.025, 'momentum': 0.9,
        'weight_decay': 1e-5, 'batch_size': 64,
        'white_noise': 1.0, 'offset_noise': 0.2,
        'scheduler': 'linear', 'max_minibatches': 100000,
        'use_post_rnn_head': True, 'head_dim': 256,
        'use_speckled_mask': True, 'speckled_p': 0.3,
        'ortho_init': False, 'dropout': 0.4,
        'split': 'within-day', 'patience': 50,
    },
    'enhanced': {
        'hidden': 1024, 'n_layers': 5, 'bidirectional': True,
        'kernel_size': 32, 'stride': 4, 'day_hidden': 256,
        'optimizer': 'sgd', 'lr': 0.1, 'momentum': 0.9,
        'weight_decay': 1e-5, 'batch_size': 128,
        'white_noise': 0.8, 'offset_noise': 0.2,
        'scheduler': 'step', 'step_size': 4000, 'step_gamma': 0.1,
        'max_minibatches': 100000,
        'use_post_rnn_head': True, 'head_dim': 256,
        'use_speckled_mask': True, 'speckled_p': 0.3,
        'ortho_init': True, 'dropout': 0.4,
        'split': 'within-day', 'patience': 50,
    },
}


def apply_preset(args, preset_name):
    """Apply preset defaults, but CLI args take precedence."""
    if preset_name not in PRESETS:
        return
    preset = PRESETS[preset_name]
    # Only set values that weren't explicitly provided on CLI
    for key, value in preset.items():
        cli_key = key.replace('-', '_')
        if not hasattr(args, f'_explicit_{cli_key}'):
            setattr(args, cli_key, value)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Lead 1: GRU Baselines & Literature Reproductions')

    # Experiment
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (e.g., L1.1_paper_exact)')
    parser.add_argument('--preset', type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help='Preset configuration')

    # Data
    parser.add_argument('--data', type=str,
                        default=str(DATA_DIR / 'sentences_paper_256d.h5'),
                        help='Path to HDF5 data file')
    parser.add_argument('--input-dim', type=int, default=256,
                        help='Feature dimension per frame (256 or 1280)')
    parser.add_argument('--max-trials', type=int, default=None)

    # Model architecture
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--bidirectional', type=str, default='false',
                        help='true/false for bidirectional GRU')
    parser.add_argument('--day-hidden', type=int, default=256)
    parser.add_argument('--kernel-size', type=int, default=14)
    parser.add_argument('--stride', type=int, default=4)

    # Post-RNN head
    parser.add_argument('--use-post-rnn-head', action='store_true', default=False)
    parser.add_argument('--head-dim', type=int, default=256)

    # Speckled mask
    parser.add_argument('--use-speckled-mask', action='store_true', default=False)
    parser.add_argument('--speckled-p', type=float, default=0.3)

    # Init
    parser.add_argument('--ortho-init', action='store_true', default=False)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--adam-eps', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--weight-decay', type=float, default=1e-5)

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='linear',
                        choices=['linear', 'step', 'cosine', 'plateau'])
    parser.add_argument('--step-size', type=int, default=4000,
                        help='StepLR: step size in minibatches')
    parser.add_argument('--step-gamma', type=float, default=0.1,
                        help='StepLR: decay factor')
    parser.add_argument('--lr-end', type=float, default=0.0,
                        help='End LR for linear decay')
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='LR warmup steps (epochs)')

    # Training
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-minibatches', type=int, default=10000)
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override max epochs (computed from max-minibatches)')
    parser.add_argument('--white-noise', type=float, default=1.0)
    parser.add_argument('--offset-noise', type=float, default=0.2)
    parser.add_argument('--smooth', type=float, default=2,
                        help='Causal Gaussian smoothing SD in bins')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default='within-day',
                        choices=['within-day', 'cross-session'],
                        help='within-day: paper protocol (40/day held out, all sessions trained). '
                             'cross-session: last 4 sessions = test (harder)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Fraction of per-day trials for validation (within-day split)')

    # Auxiliary losses
    parser.add_argument('--use-suptcon', action='store_true', default=False)
    parser.add_argument('--suptcon-weight', type=float, default=0.1)
    parser.add_argument('--suptcon-temp', type=float, default=0.1)
    parser.add_argument('--return-hidden', action='store_true', default=False,
                        help='Return hidden states for supTCon')
    parser.add_argument('--use-fastemit', action='store_true', default=False)
    parser.add_argument('--fastemit-lambda', type=float, default=0.01)

    # Track which args were explicitly set
    args, _ = parser.parse_known_args()

    # Apply preset before final parse
    if args.preset:
        # Parse again to get defaults vs explicit
        namespace = parser.parse_args()

        # Determine which args were explicitly provided
        import sys as _sys
        explicit = set()
        for arg in _sys.argv[1:]:
            if arg.startswith('--'):
                key = arg.lstrip('-').replace('-', '_')
                explicit.add(key)

        # Apply preset, overridden by explicit CLI args
        preset = PRESETS[args.preset]
        for key, value in preset.items():
            cli_key = key.replace('-', '_')
            if cli_key not in explicit:
                setattr(namespace, cli_key, value)

        args = namespace
    else:
        args = parser.parse_args()

    # Handle bidirectional as string or bool
    if isinstance(args.bidirectional, str):
        args.bidirectional = args.bidirectional.lower() in ('true', '1', 'yes')

    # If supTCon enabled, force return_hidden
    if args.use_suptcon:
        args.return_hidden = True

    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Print config ──
    print('=' * 70)
    print(f'LEAD 1: {args.experiment}')
    if args.preset:
        print(f'  Preset: {args.preset}')
    print(f'  Data: {args.data} ({args.input_dim}D)')
    print(f'  Model: hidden={args.hidden}, layers={args.n_layers}, '
          f'{"BiGRU" if args.bidirectional else "UniGRU"}')
    print(f'  Stacking: kernel={args.kernel_size} ({args.kernel_size*20}ms), '
          f'stride={args.stride} ({args.stride*20}ms)')
    print(f'  Day-specific: {args.input_dim}→{args.day_hidden} softsign')
    print(f'  Optimizer: {args.optimizer} lr={args.lr}' +
          (f' eps={args.adam_eps}' if args.optimizer == 'adam' else
           f' momentum={args.momentum}'))
    print(f'  Noise: white={args.white_noise}, offset={args.offset_noise}')
    print(f'  Batch: {args.batch_size}, Dropout: {args.dropout}')
    if args.use_post_rnn_head:
        print(f'  Post-RNN head: dim={args.head_dim}')
    if args.use_speckled_mask:
        print(f'  Speckled mask: p={args.speckled_p}')
    if args.ortho_init:
        print(f'  Orthogonal init: enabled')
    if args.use_suptcon:
        print(f'  SupTCon: weight={args.suptcon_weight}, temp={args.suptcon_temp}')
    if args.use_fastemit:
        print(f'  FastEmit: lambda={args.fastemit_lambda}')
    print(f'  Split: {args.split}')
    print(f'  Seed: {args.seed}')
    print('=' * 70)

    # ── Load data ──
    h5_path = Path(args.data)
    if not h5_path.exists():
        print(f'ERROR: {h5_path} not found')
        sys.exit(1)

    min_frames = args.kernel_size + args.stride  # Skip trials shorter than 1 output step
    trials = load_h5_raw(
        h5_path, max_trials=args.max_trials,
        smooth_sigma=args.smooth, causal_smooth=True,
        min_frames=min_frames,
    )

    # ── Build session index ──
    session_names = sorted(set(t['session'] for t in trials))
    session_to_idx = {s: i for i, s in enumerate(session_names)}
    n_sessions = len(session_names)
    for t in trials:
        t['session_idx'] = session_to_idx[t['session']]

    # ── Split data ──
    sessions_dict = {}
    for t in trials:
        sessions_dict.setdefault(t['session'], []).append(t)

    if args.split == 'within-day':
        # Paper protocol: hold out ~40 sentences per day, all sessions have trained layers
        # IMPORTANT: Use fixed seed=42 for split so all models share the same test set
        rng = np.random.RandomState(42)
        train_trials, val_trials, test_trials = [], [], []
        for s_name in session_names:
            s_trials = sessions_dict[s_name]
            rng.shuffle(s_trials)
            n_test = 40  # Paper: 40 sentences per day held out for test
            n_val = max(1, int(len(s_trials) * args.val_ratio))
            test_trials.extend(s_trials[:n_test])
            val_trials.extend(s_trials[n_test:n_test + n_val])
            train_trials.extend(s_trials[n_test + n_val:])
        # All sessions are "train" sessions (day-specific layers trained for all)
        train_session_idxs = set(range(n_sessions))
        split_desc = f'within-day (40/day test, {args.val_ratio:.0%} val)'
    else:
        # Cross-session: last 4 = test, prev 2 = val
        test_sessions = session_names[-4:]
        remaining = session_names[:-4]
        val_sessions = remaining[-2:]
        train_sessions_list = remaining[:-2]

        train_trials = [t for s in train_sessions_list for t in sessions_dict[s]]
        val_trials = [t for s in val_sessions for t in sessions_dict[s]]
        test_trials = [t for s in test_sessions for t in sessions_dict[s]]
        train_session_idxs = {session_to_idx[s] for s in train_sessions_list}
        split_desc = f'cross-session (last 4 test, prev 2 val)'

    print(f'\nSplit: {split_desc}')
    print(f'Trials: {len(train_trials)} train, {len(val_trials)} val, '
          f'{len(test_trials)} test')
    print(f'Feature dim: {trials[0]["features_raw"].shape[1]}')
    print(f'Avg raw frames: {np.mean([t["n_frames_raw"] for t in trials]):.0f}')

    # ── Data loaders ──
    train_loader = DataLoader(train_trials, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_raw,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_trials, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_raw, num_workers=2)
    test_loader = DataLoader(test_trials, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_raw, num_workers=2)

    # ── Build model ──
    model = EnhancedGRU(
        input_dim=args.input_dim,
        n_classes=N_OUTPUT,
        hidden=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_sessions=n_sessions,
        kernel_size=args.kernel_size,
        stride=args.stride,
        bidirectional=args.bidirectional,
        day_hidden=args.day_hidden,
        use_post_rnn_head=args.use_post_rnn_head,
        head_dim=getattr(args, 'head_dim', 256),
        use_speckled_mask=args.use_speckled_mask,
        speckled_p=getattr(args, 'speckled_p', 0.3),
        ortho_init=args.ortho_init,
        return_hidden=getattr(args, 'return_hidden', False),
    )
    model.set_train_sessions(train_session_idxs)

    n_params = sum(p.numel() for p in model.parameters())
    direction = 'BiGRU' if args.bidirectional else 'UniGRU'
    print(f'\nModel: EnhancedGRU ({direction}), {n_params/1e6:.1f}M params')
    print(f'  Stacked input: {args.kernel_size * args.day_hidden}D → {args.hidden}D')
    print(f'  RNN output: {model.rnn_out_dim}D')
    print(f'  Day-specific: {n_sessions} sessions '
          f'({len(train_session_idxs)} train, '
          f'{n_sessions - len(train_session_idxs)} shared)')

    model = model.to(device)

    # ── Optimizer ──
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=getattr(args, 'momentum', 0.9),
            weight_decay=args.weight_decay,
            nesterov=getattr(args, 'nesterov', False),
        )
    else:  # adam
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr,
            betas=(0.9, 0.999), eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )

    # ── Scheduler ──
    batches_per_epoch = len(train_loader)
    if args.max_epochs:
        total_epochs = args.max_epochs
    else:
        total_epochs = max(1, args.max_minibatches // batches_per_epoch)

    if args.scheduler == 'step':
        # Step size is in minibatches, convert to epochs
        step_epochs = max(1, getattr(args, 'step_size', 4000) // batches_per_epoch)
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_epochs,
            gamma=getattr(args, 'step_gamma', 0.1))
        use_plateau = False
    elif args.scheduler == 'cosine':
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6)
        use_plateau = False
    elif args.scheduler == 'plateau':
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)
        use_plateau = True
    else:  # linear
        end_factor = max(args.lr_end / args.lr, 0.001) if args.lr_end > 0 else 0.001
        base_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=end_factor,
            total_iters=total_epochs)
        use_plateau = False

    # Warmup
    if args.warmup_steps > 0 and not use_plateau:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=args.warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, base_scheduler],
            milestones=[args.warmup_steps])
    else:
        scheduler = base_scheduler

    print(f'\nTraining: {total_epochs} epochs, {batches_per_epoch} batches/epoch')
    print(f'Scheduler: {args.scheduler}' +
          (f', warmup={args.warmup_steps}' if args.warmup_steps > 0 else ''))

    # ── Loss ──
    criterion = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
    scaler = GradScaler()

    suptcon_loss_fn = None
    if args.use_suptcon:
        suptcon_loss_fn = SupConLoss(
            temperature=args.suptcon_temp, max_frames=2048)

    # ── Training loop ──
    best_val_per = float('inf')
    best_state = None
    wait = 0
    history = []

    print(f'\n{"Epoch":>5} {"Loss":>8} {"CTC":>8} {"ValPER":>8} {"LR":>10} {"Time":>6}')
    print('-' * 55)

    for epoch in range(total_epochs):
        t0 = time.time()

        train_metrics = train_epoch_v2(
            model, train_loader, optimizer, criterion, scaler, device, args,
            suptcon_loss_fn=suptcon_loss_fn,
        )

        val_metrics = evaluate_model(
            model, val_loader, device, args.kernel_size, args.stride)

        if use_plateau:
            scheduler.step(val_metrics['per'])
        else:
            scheduler.step()

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

        # Print progress
        extra = ''
        if args.use_suptcon and train_metrics['suptcon_loss'] > 0:
            extra += f' sc={train_metrics["suptcon_loss"]:.4f}'
        if args.use_fastemit and train_metrics['fastemit_loss'] != 0:
            extra += f' fe={train_metrics["fastemit_loss"]:.4f}'

        marker = ''
        if val_metrics['per'] < best_val_per:
            marker = ' ***'

        print(f'{epoch+1:5d} {train_metrics["loss"]:8.4f} '
              f'{train_metrics["ctc_loss"]:8.4f} '
              f'{val_metrics["per"]:8.3f} {cur_lr:10.2e} {elapsed:5.1f}s'
              f'{extra}{marker}')

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'ctc_loss': train_metrics['ctc_loss'],
            'suptcon_loss': train_metrics['suptcon_loss'],
            'fastemit_loss': train_metrics['fastemit_loss'],
            'val_per': val_metrics['per'],
            'lr': cur_lr,
            'elapsed': elapsed,
        })

        if val_metrics['per'] < best_val_per:
            best_val_per = val_metrics['per']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= args.patience:
            print(f'\nEarly stop at epoch {epoch+1} (patience={args.patience})')
            break

    # ── Test ──
    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(
        model, test_loader, device, args.kernel_size, args.stride)

    print(f'\n{"="*70}')
    print(f'RESULTS — {args.experiment}')
    print(f'  Best Val PER: {best_val_per:.1%}')
    print(f'  Test PER:     {test_metrics["per"]:.1%}')
    print(f'  Paper ref:    19.7% PER (Willett et al. 2023)')
    print(f'  Params:       {n_params/1e6:.1f}M')
    print(f'{"="*70}')

    # ── Save results ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'experiment': args.experiment,
        'preset': args.preset,
        'test_per': test_metrics['per'],
        'best_val_per': float(best_val_per),
        'n_train': len(train_trials),
        'n_val': len(val_trials),
        'n_test': len(test_trials),
        'epochs_trained': epoch + 1,
        'n_params': n_params,
        'hyperparams': {k: v for k, v in vars(args).items()
                        if not k.startswith('_')},
        'history': history,
    }

    results_path = RESULTS_DIR / f'{args.experiment}_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved: {results_path}')

    # Save model checkpoint
    ckpt_path = RESULTS_DIR / f'{args.experiment}_best.pt'
    checkpoint = {
        'model_state_dict': best_state,
        'hyperparams': {k: v for k, v in vars(args).items()
                        if not k.startswith('_')},
        'n_sessions': n_sessions,
        'session_to_idx': session_to_idx,
        'train_session_idxs': list(train_session_idxs),
        'best_val_per': float(best_val_per),
        'test_per': test_metrics['per'],
        'n_params': n_params,
    }
    torch.save(checkpoint, ckpt_path)
    print(f'Checkpoint saved: {ckpt_path}')


if __name__ == '__main__':
    main()
