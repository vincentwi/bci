#!/usr/bin/env python3
"""
Lead 2: DCoND (Divide-and-Conquer Neural Decoding) — 1st Place Reproduction.

Implements diphone-based CTC decoding with progressive alpha schedule.
The key innovation: model 1600 diphone classes instead of 40 monophones,
then marginalize back to monophone probabilities for decoding.

Sources:
  - arxiv 2411.10657 (DCoND paper): diphone loss, progressive alpha
  - arxiv 2412.17227 (Benchmark '24): SGD>Adam, post-RNN norm, ensemble
  - CIBR-Okubo-Lab/speechBCI_2024: kernel=32, hidden=1024, SGD config

Architecture:
  Raw 256D → Day-specific(256→256, softsign) per-frame
    → Stack kernel frames, stride 4 → kernel*256 D
    → Linear(kernel*256 → hidden) + LayerNorm
    → N-layer BiGRU(hidden)
    → Post-RNN: LayerNorm + Dropout + Linear + GELU
    → Linear(rnn_out → 1601) → DiphoneCTC loss

Usage:
    # DCoND-512 with progressive alpha, SGD
    CUDA_VISIBLE_DEVICES=4 python brain2speech/lead2_train_dcond.py \\
        --experiment L2_dcond_512_sgd \\
        --data sentences_paper_256d.h5 --input-dim 256 \\
        --bidirectional --hidden 512 --n-layers 5 \\
        --optimizer sgd --lr 0.1 --momentum 0.9 \\
        --alpha 0.6 --progressive-alpha \\
        --post-rnn-norm --max-epochs 120 --seed 42

    # Alpha ablation
    for ALPHA in 0.2 0.4 0.6 0.8 1.0; do
        CUDA_VISIBLE_DEVICES=7 python brain2speech/lead2_train_dcond.py \\
            --experiment L2_dcond_alpha_${ALPHA} \\
            --alpha $ALPHA --max-epochs 120 --seed 42
    done
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

# Add brain2speech to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_beyond_paper import (
    DaySpecificInputLayer, load_h5_dataset, collate_raw,
    ctc_greedy_decode, compute_per, causal_gaussian_smooth, stack_and_stride,
    DATA_DIR, RESULTS_DIR, N_CLASSES, CTC_BLANK,
)

# ═══════════════════════════════════════════════════════════════════════
# DIPHONE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

N_MONO = N_CLASSES  # 40 (39 phonemes + SIL)
N_DIPHONE = N_MONO * N_MONO  # 1600
CTC_BLANK_DIPHONE = N_DIPHONE  # 1600
N_DIPHONE_CLASSES = N_DIPHONE + 1  # 1601 (1600 diphones + blank)


# ═══════════════════════════════════════════════════════════════════════
# DIPHONE UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def mono_to_diphone_targets(mono_seq):
    """Convert monophone target sequence to diphone target sequence.

    [p0, p1, p2, ..., pN] → [p0*40+p1, p1*40+p2, ..., pN-1*40+pN]

    The diphone sequence is 1 element shorter than the monophone sequence.
    Each diphone index encodes (preceding_phoneme, current_phoneme).

    Source: arxiv 2411.10657, Section 2.1
    """
    if len(mono_seq) < 2:
        return mono_seq[:0] if hasattr(mono_seq, '__getitem__') else []
    return [int(mono_seq[i]) * N_MONO + int(mono_seq[i + 1])
            for i in range(len(mono_seq) - 1)]


def build_marginalization_matrix():
    """Build (1600, 40) matrix: M[d, m] = 1 if diphone d has current_phoneme == m.

    Diphone index = prev * N_MONO + curr, so current_phoneme = d % N_MONO.

    Used to recover monophone probabilities from diphone probabilities:
        mono_probs = diphone_probs @ M  # (B, T, 1600) @ (1600, 40) → (B, T, 40)

    Source: arxiv 2411.10657, Section 2.2
    """
    M = torch.zeros(N_DIPHONE, N_MONO)
    for prev in range(N_MONO):
        for curr in range(N_MONO):
            M[prev * N_MONO + curr, curr] = 1.0
    return M


def build_marginalization_matrix_ext():
    """Build extended (1601, 41) marginalization matrix including blank mapping.

    Maps diphone+blank logits → monophone+blank logits:
    - First 1600 rows: diphone → current phoneme (via build_marginalization_matrix)
    - Row 1600 (diphone blank) → column 40 (monophone blank)
    """
    M_ext = torch.zeros(N_DIPHONE + 1, N_MONO + 1)
    M_ext[:N_DIPHONE, :N_MONO] = build_marginalization_matrix()
    M_ext[N_DIPHONE, N_MONO] = 1.0  # blank → blank
    return M_ext


# ═══════════════════════════════════════════════════════════════════════
# DCOND LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

class DiphoneCTCLoss(nn.Module):
    """Combined diphone + monophone CTC loss.

    L = alpha * L_c(diphone) + (1-alpha) * L_s(monophone)

    Source: arxiv 2411.10657, Equation 1
    Optimal alpha=0.6 (ablation: 0.2→8.47%, 0.6→8.06%, 1.0→9.46% WER)

    The monophone loss L_s is computed by MARGINALIZING diphone probabilities:
    1. Softmax over 1601 diphone+blank logits → probabilities
    2. Matrix multiply by M_ext (1601→41) → monophone probabilities
    3. Log → CTC loss on monophone targets

    This marginalization acts as regularization, preventing overfitting
    to rare diphone combinations.
    """

    def __init__(self, n_mono=N_MONO, alpha=0.6):
        super().__init__()
        self.alpha = alpha
        n_di = n_mono * n_mono  # 1600
        self.ctc_di = nn.CTCLoss(blank=n_di, zero_infinity=True)
        self.ctc_mono = nn.CTCLoss(blank=n_mono, zero_infinity=True)

        M_ext = build_marginalization_matrix_ext()
        self.register_buffer('M_ext', M_ext)

    def forward(self, logits, mono_targets, diphone_targets,
                input_lengths, mono_target_lengths, diphone_target_lengths):
        """
        logits: (B, T, 1601) raw model output (diphone + blank)
        mono_targets: concatenated monophone target indices
        diphone_targets: concatenated diphone target indices
        input_lengths: (B,) output sequence lengths
        mono_target_lengths: (B,) monophone target lengths
        diphone_target_lengths: (B,) diphone target lengths
        """
        # Diphone CTC loss (L_c)
        di_lp = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, 1601)
        L_c = self.ctc_di(di_lp, diphone_targets,
                          input_lengths, diphone_target_lengths)

        # Marginalize to monophone probabilities, then compute mono CTC (L_s)
        di_probs = logits.softmax(dim=-1)  # (B, T, 1601)
        mono_probs = di_probs @ self.M_ext  # (B, T, 41)
        mono_lp = (mono_probs + 1e-10).log().transpose(0, 1)  # (T, B, 41)
        L_s = self.ctc_mono(mono_lp, mono_targets,
                            input_lengths, mono_target_lengths)

        combined = self.alpha * L_c + (1 - self.alpha) * L_s
        # Return combined loss + individual components for logging
        return combined, L_c.detach(), L_s.detach()


def get_progressive_alpha(epoch, warmup=10, target=0.6, reverse=False):
    """Progressive alpha schedule.

    Original (DCoND paper, arxiv 2411.10657):
      Epochs 0-10: α=1.0 → gradually decrease to target α=0.6
      (Works when encoder is pre-trained; fails from scratch with 1601-class CTC)

    Reverse (for training from scratch):
      Epochs 0-10: α=0.0 (pure monophone → learn encoder features first)
      Epochs 11-20: α=0.1 (introduce diphone gradually)
      ...increase by 0.1 every 10 epochs until target α
      This lets the encoder learn useful features via easy 41-class CTC
      before adding the harder 1601-class diphone objective.
    """
    if reverse:
        if epoch <= warmup:
            return 0.0
        step = (epoch - 1) // 10
        alpha = step * 0.1
        return min(alpha, target)
    else:
        if epoch <= warmup:
            return 1.0
        step = (epoch - 1) // 10
        alpha = 1.0 - step * 0.1
        return max(alpha, target)


# ═══════════════════════════════════════════════════════════════════════
# DCOND DECODER MODEL
# ═══════════════════════════════════════════════════════════════════════

class DCoNDDecoder(nn.Module):
    """BiGRU with 1601-class diphone output head.

    Architecture identical to BeyondPaperGRU except:
    - Output: Linear(rnn_out, 1601) instead of Linear(rnn_out, 41)
    - Post-RNN norm (Source: Linderman, arxiv 2412.17227, 9.22%→8.00%)
    - SGD optimizer recommended (Source: CIBR, Linderman)

    At inference: marginalize 1601→41 for monophone decoding.
    """

    def __init__(self, n_features_per_frame=256, n_diphones=N_DIPHONE_CLASSES,
                 hidden=512, n_layers=5, dropout=0.4,
                 n_sessions=24, kernel_size=14, stride=4,
                 bidirectional=True, day_hidden=256,
                 post_rnn_norm=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.bidirectional = bidirectional
        self.n_diphones = n_diphones

        # Reuse existing DaySpecificInputLayer
        self.day_input = DaySpecificInputLayer(
            n_features_per_frame, day_hidden, n_sessions, dropout=dropout)

        gru_input_dim = kernel_size * day_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(gru_input_dim, hidden),
            nn.LayerNorm(hidden))

        self.rnn = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if n_layers > 1 else 0)

        rnn_out = hidden * (2 if bidirectional else 1)

        # Post-RNN norm (Source: Linderman 9.22%→8.00% WER)
        if post_rnn_norm:
            self.post_rnn = nn.Sequential(
                nn.LayerNorm(rnn_out),
                nn.Dropout(dropout),
                nn.Linear(rnn_out, rnn_out),
                nn.GELU())
        else:
            self.post_rnn = nn.Identity()

        # 1601 outputs: 1600 diphones + 1 CTC blank
        self.output_proj = nn.Linear(rnn_out, n_diphones)

        # Separate monophone head for pre-training (bypasses 1601 softmax)
        self.mono_head = nn.Linear(rnn_out, N_MONO + 1)  # 41 classes

    def set_train_sessions(self, sids):
        self.day_input.set_train_sessions(sids)

    def _encode(self, x, session_ids):
        """Shared encoder: day-specific → stack → proj → GRU → post-RNN."""
        x = self.day_input(x, session_ids)
        B, T, H = x.shape
        T_new = max(1, (T - self.kernel_size) // self.stride + 1)
        indices = torch.arange(T_new, device=x.device) * self.stride
        stacked = []
        for k in range(self.kernel_size):
            idx = (indices + k).clamp(max=T - 1)
            stacked.append(x[:, idx, :])
        x = torch.cat(stacked, dim=-1)
        x = self.input_proj(x)
        x, _ = self.rnn(x)
        x = self.post_rnn(x)
        return x

    def forward(self, x, session_ids):
        """x: (B, T_raw, input_dim) raw features. Returns (B, T_out, 1601)."""
        return self.output_proj(self._encode(x, session_ids))

    def forward_mono(self, x, session_ids):
        """Direct monophone logits (B, T_out, 41) for pre-training."""
        return self.mono_head(self._encode(x, session_ids))


class PaperExactDCoND(nn.Module):
    """Paper-exact architecture + diphone head for DCoND training.

    Matches the paper_exact_100k model (18.4% PER) architecture exactly:
    - DaySpecificInputLayer: 256D → 256D per frame (softsign)
    - Stack 14 frames → 3584D
    - GRU takes 3584D directly (NO input_proj!)
    - Unidirectional 5-layer GRU-512
    - Dropout + Linear → 41 classes (mono head)

    Added for DCoND:
    - Dropout + Linear → 1601 classes (diphone head)
    - Both heads share the same encoder
    """

    def __init__(self, n_features_per_frame=256, n_diphones=N_DIPHONE_CLASSES,
                 hidden=512, n_layers=5, dropout=0.4,
                 n_sessions=24, kernel_size=14, stride=4,
                 day_hidden=256, bidirectional=False,
                 post_rnn_norm=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.bidirectional = bidirectional
        self.n_diphones = n_diphones

        # Day-specific: 256→256 per frame (matches paper_exact)
        self.day_input = DaySpecificInputLayer(
            n_features_per_frame, day_hidden, n_sessions, dropout=dropout)

        # NO input_proj — GRU takes full stacked features
        gru_input_dim = kernel_size * day_hidden  # 14*256 = 3584

        self.rnn = nn.GRU(gru_input_dim, hidden, n_layers, batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if n_layers > 1 else 0)

        rnn_out = hidden * (2 if bidirectional else 1)

        # Mono head: matches paper_exact output_proj (Sequential[Dropout, Linear])
        self.output_proj_mono = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out, N_MONO + 1),  # 41 classes
        )

        # Diphone head: same structure but 1601 classes
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out, n_diphones),  # 1601 classes
        )

        # Alias for dual-head compatibility
        self.mono_head = self.output_proj_mono[1]  # The Linear layer

    def set_train_sessions(self, sids):
        self.day_input.set_train_sessions(sids)

    def _encode(self, x, session_ids):
        """Shared encoder: day-specific → stack → GRU (no projection)."""
        x = self.day_input(x, session_ids)
        B, T, H = x.shape
        T_new = max(1, (T - self.kernel_size) // self.stride + 1)
        indices = torch.arange(T_new, device=x.device) * self.stride
        stacked = []
        for k in range(self.kernel_size):
            idx = (indices + k).clamp(max=T - 1)
            stacked.append(x[:, idx, :])
        x = torch.cat(stacked, dim=-1)  # (B, T_new, 3584)
        x, _ = self.rnn(x)
        return x

    def forward(self, x, session_ids):
        """Returns diphone logits (B, T_out, 1601)."""
        enc = self._encode(x, session_ids)
        return self.output_proj(enc)

    def forward_mono(self, x, session_ids):
        """Returns monophone logits (B, T_out, 41)."""
        enc = self._encode(x, session_ids)
        return self.output_proj_mono(enc)

    def load_paper_exact_weights(self, ckpt_path):
        """Load weights from paper_exact checkpoint into matching layers.

        Maps paper_exact state dict keys to PaperExactDCoND keys:
        - day_input.* → day_input.* (direct match)
        - rnn.* → rnn.* (direct match)
        - output_proj.1.weight/bias → output_proj_mono.1.weight/bias
        """
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        # paper_exact saves state dict directly (no 'model_state_dict' wrapper)
        if 'model_state_dict' in state:
            state = state['model_state_dict']

        my_state = self.state_dict()
        loaded, skipped = 0, 0
        for k, v in state.items():
            # Map paper_exact output_proj → our output_proj_mono
            new_k = k.replace('output_proj.', 'output_proj_mono.')
            if new_k in my_state and my_state[new_k].shape == v.shape:
                my_state[new_k] = v
                loaded += 1
            elif k in my_state and my_state[k].shape == v.shape:
                my_state[k] = v
                loaded += 1
            else:
                skipped += 1
                print(f'  Skip: {k} ({v.shape})')

        self.load_state_dict(my_state)
        print(f'  Loaded {loaded} params from paper_exact, skipped {skipped}')
        return loaded


# ═══════════════════════════════════════════════════════════════════════
# COLLATION FOR DIPHONE TRAINING
# ═══════════════════════════════════════════════════════════════════════

def collate_dcond(batch):
    """Collate that produces BOTH monophone AND diphone targets per trial.

    Returns: (features_padded, mono_targets_cat, diphone_targets_cat,
              feat_lens, mono_tgt_lens, diphone_tgt_lens, session_ids)
    """
    features = [torch.FloatTensor(t['features_raw']) for t in batch]
    mono_targets = [torch.IntTensor(t['phoneme_indices']) for t in batch]

    # Convert mono targets to diphone targets
    diphone_targets = []
    for t in batch:
        mono = t['phoneme_indices'].tolist()
        di = mono_to_diphone_targets(mono)
        diphone_targets.append(torch.IntTensor(di))

    feat_lens = torch.IntTensor([f.shape[0] for f in features])
    mono_tgt_lens = torch.IntTensor([t.shape[0] for t in mono_targets])
    diphone_tgt_lens = torch.IntTensor([t.shape[0] for t in diphone_targets])
    session_ids = torch.LongTensor([t.get('session_idx', 0) for t in batch])

    # Pad features
    max_T = max(f.shape[0] for f in features)
    C = features[0].shape[1]
    features_padded = torch.zeros(len(features), max_T, C)
    for i, f in enumerate(features):
        features_padded[i, :f.shape[0], :] = f

    mono_targets_cat = torch.cat(mono_targets)
    diphone_targets_cat = torch.cat(diphone_targets)

    return (features_padded, mono_targets_cat, diphone_targets_cat,
            feat_lens, mono_tgt_lens, diphone_tgt_lens, session_ids)


# ═══════════════════════════════════════════════════════════════════════
# DCOND INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def dcond_greedy_decode(logits, M_ext):
    """Decode diphone logits by marginalizing to monophone probs.

    logits: (T, 1601) raw model output (single trial)
    M_ext: (1601, 41) marginalization matrix

    Returns: list of monophone indices
    """
    probs = torch.softmax(logits, dim=-1)  # (T, 1601)
    mono_probs = probs @ M_ext  # (T, 41)
    mono_log_probs = (mono_probs + 1e-10).log()
    return ctc_greedy_decode(mono_log_probs.cpu().numpy(), blank=CTC_BLANK)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION LOOPS
# ═══════════════════════════════════════════════════════════════════════

def train_epoch_dcond(model, dataloader, optimizer, criterion, scaler, device,
                      white_noise_sd=0.8, offset_noise_sd=0.2,
                      kernel_size=14, stride=4,
                      speckled_mask=0.0, fastemit_lambda=0.0,
                      grad_clip=5.0, use_dual_head=False, di_weight=0.3):
    """Train one epoch with DCoND diphone loss.

    If use_dual_head=True:
      L = (1-di_weight)*L_mono_direct + di_weight*L_di_ctc
      Uses direct mono_head (41-class) as primary gradient + diphone as auxiliary.
      This prevents CTC collapse from the 1601-class softmax.
    """
    from torch.amp import autocast
    model.train()
    total_loss = 0
    total_lc = 0
    total_ls = 0
    n_batches = 0

    mono_ctc_fn = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True) if use_dual_head else None
    di_ctc_fn = nn.CTCLoss(blank=CTC_BLANK_DIPHONE, zero_infinity=True) if use_dual_head else None

    for batch in dataloader:
        (features, mono_targets, diphone_targets,
         feat_lens, mono_tgt_lens, diphone_tgt_lens, session_ids) = batch

        features = features.to(device)
        mono_targets = mono_targets.to(device)
        diphone_targets = diphone_targets.to(device)
        session_ids = session_ids.to(device)

        # Noise augmentation (paper: white=1.0, CIBR: white=0.8)
        if white_noise_sd > 0:
            features = features + torch.randn_like(features) * white_noise_sd
        if offset_noise_sd > 0:
            B, T, C = features.shape
            offset = torch.randn(B, 1, C, device=device) * offset_noise_sd
            features = features + offset

        # Speckled masking (Source: Linderman, arxiv 2412.17227)
        if speckled_mask > 0:
            mask = torch.bernoulli(
                torch.full_like(features, 1.0 - speckled_mask))
            features = features * mask / (1.0 - speckled_mask)

        optimizer.zero_grad(set_to_none=True)
        m = model.module if hasattr(model, 'module') else model

        with autocast("cuda"):
            # Compute output lengths
            out_lens = ((feat_lens - kernel_size) // stride + 1).clamp(min=1)

            if use_dual_head:
                # Dual-head: direct mono (stable) + diphone (auxiliary)
                enc = m._encode(features, session_ids)
                # Use full output paths (including dropout) for both heads
                if hasattr(m, 'output_proj_mono'):
                    # PaperExactDCoND: Sequential(Dropout, Linear)
                    mono_logits = m.output_proj_mono(enc)  # (B, T_out, 41)
                else:
                    # DCoNDDecoder: bare Linear
                    mono_logits = m.mono_head(enc)  # (B, T_out, 41)
                di_logits = m.output_proj(enc)  # (B, T_out, 1601)

                input_lengths = out_lens.clamp(max=mono_logits.shape[1]).to(device)

                # Direct monophone CTC (stable, primary gradient)
                mono_lp = mono_logits.log_softmax(-1).transpose(0, 1)
                L_mono = mono_ctc_fn(mono_lp, mono_targets,
                                     input_lengths, mono_tgt_lens.to(device))

                # Diphone CTC (auxiliary)
                di_lp = di_logits.log_softmax(-1).transpose(0, 1)
                L_di = di_ctc_fn(di_lp, diphone_targets,
                                  input_lengths, diphone_tgt_lens.to(device))

                # Clamp diphone loss to prevent explosion
                if torch.isnan(L_di) or torch.isinf(L_di):
                    L_di = torch.zeros_like(L_mono)

                loss = (1 - di_weight) * L_mono + di_weight * L_di
                lc = L_di.detach()
                ls = L_mono.detach()
            else:
                logits = model(features, session_ids=session_ids)  # (B, T_out, 1601)
                input_lengths = out_lens.clamp(max=logits.shape[1]).to(device)

                mono_tgt_lens_d = mono_tgt_lens.to(device)
                diphone_tgt_lens_d = diphone_tgt_lens.to(device)

                loss, lc, ls = criterion(logits, mono_targets, diphone_targets,
                                 input_lengths, mono_tgt_lens_d, diphone_tgt_lens_d)

            # FastEmit regularization (Source: Linderman)
            if fastemit_lambda > 0:
                if use_dual_head:
                    blank_lp = di_logits.log_softmax(dim=-1)[:, :, CTC_BLANK_DIPHONE]
                else:
                    log_probs = logits.log_softmax(dim=-1)
                    blank_lp = log_probs[:, :, CTC_BLANK_DIPHONE]
                fastemit_loss = -fastemit_lambda * blank_lp.mean()
                loss = loss + fastemit_loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_lc += lc.item()
        total_ls += ls.item()
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_lc / n, total_ls / n


def evaluate_dcond(model, dataloader, device, M_ext,
                   kernel_size=14, stride=4, use_dual_head=False):
    """Evaluate DCoND model by marginalizing to monophone for PER computation.

    If use_dual_head: also evaluate direct mono_head PER for comparison.
    """
    from torch.amp import autocast
    model.eval()
    all_per = []
    all_per_direct = []
    m = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        for batch in dataloader:
            (features, mono_targets, diphone_targets,
             feat_lens, mono_tgt_lens, diphone_tgt_lens, session_ids) = batch

            features = features.to(device)
            session_ids = session_ids.to(device)

            with autocast("cuda"):
                logits = model(features, session_ids=session_ids)
                if use_dual_head:
                    mono_logits_direct = m.forward_mono(features, session_ids)

            # Marginalize diphone → monophone for decoding
            probs = logits.float().softmax(dim=-1).cpu()  # (B, T_out, 1601)
            mono_probs = probs @ M_ext  # (B, T_out, 41)
            mono_log_probs = (mono_probs + 1e-10).log().numpy()

            if use_dual_head:
                direct_lp = mono_logits_direct.float().log_softmax(-1).cpu().numpy()

            T_max = mono_log_probs.shape[1]
            # Compute per-trial output lengths accounting for stacking
            out_lens = ((feat_lens - kernel_size) // stride + 1).clamp(min=1)

            offset = 0
            for i in range(len(feat_lens)):
                T = min(out_lens[i].item(), T_max)
                P = mono_tgt_lens[i].item()
                decoded = ctc_greedy_decode(mono_log_probs[i, :T, :],
                                           blank=CTC_BLANK)
                trial_targets = mono_targets[offset:offset + P].numpy().tolist()
                offset += P
                all_per.append(compute_per(decoded, trial_targets))
                if use_dual_head:
                    decoded_d = ctc_greedy_decode(direct_lp[i, :T, :],
                                                 blank=CTC_BLANK)
                    all_per_direct.append(compute_per(decoded_d, trial_targets))

    result = {'per': float(np.mean(all_per)) if all_per else 1.0}
    if use_dual_head and all_per_direct:
        result['per_direct'] = float(np.mean(all_per_direct))
    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Lead 2: DCoND Diphone System (1st Place Reproduction)')

    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--data', type=str, default='sentences_paper_256d.h5')
    parser.add_argument('--input-dim', type=int, default=256)

    # DCoND-specific
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='DCoND alpha: weight of diphone loss (0.6 optimal)')
    parser.add_argument('--progressive-alpha', action='store_true',
                        help='Use progressive alpha schedule (1.0→0.6 over 50 epochs)')
    parser.add_argument('--reverse-alpha', action='store_true',
                        help='Reverse schedule: 0.0→target over 50 epochs (better from scratch)')
    parser.add_argument('--max-epochs', type=int, default=120,
                        help='DCoND needs 120 epochs (vs 100 for monophone)')
    parser.add_argument('--pretrain-mono-epochs', type=int, default=0,
                        help='Phase 1: train with monophone-only loss (alpha=0) for N epochs first')
    parser.add_argument('--init-from', type=str, default=None,
                        help='Initialize encoder from a checkpoint (skips mismatched output layer)')
    parser.add_argument('--phase2-lr', type=float, default=None,
                        help='Learning rate for Phase 2 diphone training (default: lr/5)')
    parser.add_argument('--dual-head', action='store_true',
                        help='Use dual-head training: direct mono CTC + diphone CTC auxiliary')
    parser.add_argument('--di-weight', type=float, default=0.3,
                        help='Weight of diphone loss in dual-head mode (default: 0.3)')

    # Model architecture
    parser.add_argument('--arch', choices=['dcond', 'paper_exact'],
                        default='dcond',
                        help='Model architecture: dcond (BiGRU+proj+postRNN) or '
                             'paper_exact (unidir GRU, no proj, matches 18%% PER model)')
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--no-bidirectional', dest='bidirectional',
                        action='store_false')
    parser.add_argument('--day-hidden', type=int, default=256)
    parser.add_argument('--kernel-size', type=int, default=14)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--post-rnn-norm', action='store_true', default=True)
    parser.add_argument('--no-post-rnn-norm', dest='post_rnn_norm',
                        action='store_false')

    # Optimizer (Sources: CIBR, Linderman — SGD >> Adam)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--adam-eps', type=float, default=0.1)
    parser.add_argument('--l2-reg', type=float, default=1e-5)
    parser.add_argument('--step-size', type=int, default=None,
                        help='StepLR step_size (4000 for CIBR m1, 5000 for m2)')
    parser.add_argument('--step-gamma', type=float, default=0.1)
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'linear', 'cosine'])

    # Training
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-minibatches', type=int, default=None,
                        help='If set, overrides max-epochs')
    parser.add_argument('--white-noise', type=float, default=0.8)
    parser.add_argument('--offset-noise', type=float, default=0.2)
    parser.add_argument('--smooth', type=float, default=2)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-trials', type=int, default=None)

    # Linderman enhancements
    parser.add_argument('--speckled-mask', type=float, default=0.0,
                        help='Speckled masking prob (0.3 from Linderman)')
    parser.add_argument('--fastemit', type=float, default=0.0,
                        help='FastEmit lambda (0.01 from Linderman)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Linear LR warmup epochs (critical for 1601-class CTC)')
    parser.add_argument('--grad-clip', type=float, default=5.0,
                        help='Gradient clipping max norm')

    args = parser.parse_args()

    # Only set CUDA_VISIBLE_DEVICES if not already set externally
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))

    from torch.utils.data import DataLoader
    from torch.amp import GradScaler, autocast

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'LEAD 2: DCoND DIPHONE SYSTEM — {args.experiment}')
    print(f'  Architecture: BiGRU-{args.hidden} × {args.n_layers}L, '
          f'kernel={args.kernel_size}')
    print(f'  Output: {N_DIPHONE_CLASSES} classes (1600 diphones + blank)')
    print(f'  Alpha: {args.alpha} '
          f'{"(progressive)" if args.progressive_alpha else "(fixed)"}')
    print(f'  Optimizer: {args.optimizer.upper()}'
          f'{" (Nesterov)" if args.nesterov else ""}, lr={args.lr}')
    print(f'  Post-RNN norm: {args.post_rnn_norm}')
    print(f'  Noise: white={args.white_noise}, offset={args.offset_noise}')
    print(f'  Epochs: {args.max_epochs}, Seed: {args.seed}')
    print('=' * 70)

    # Load data (raw, model handles stacking)
    h5_path = DATA_DIR / args.data
    if not h5_path.exists():
        print(f'ERROR: {h5_path} not found')
        print(f'  Run preprocess_paper_exact.py first')
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

    # Split by session (same as train_beyond_paper.py)
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

    # Validate diphone targets
    sample_mono = train_trials[0]['phoneme_indices'].tolist()
    sample_di = mono_to_diphone_targets(sample_mono)
    print(f'\nDiphone validation:')
    print(f'  Sample monophone seq length: {len(sample_mono)}')
    print(f'  Sample diphone seq length: {len(sample_di)} '
          f'(should be {len(sample_mono) - 1})')
    print(f'  Diphone range: [{min(sample_di)}, {max(sample_di)}] '
          f'(should be in [0, {N_DIPHONE - 1}])')

    train_loader = DataLoader(train_trials, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_dcond,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_trials, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_dcond,
                            num_workers=2)
    test_loader = DataLoader(test_trials, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_dcond,
                             num_workers=2)

    # Build model
    if args.arch == 'paper_exact':
        model = PaperExactDCoND(
            n_features_per_frame=args.input_dim,
            n_diphones=N_DIPHONE_CLASSES,
            hidden=args.hidden,
            n_layers=args.n_layers,
            dropout=args.dropout,
            n_sessions=n_sessions,
            kernel_size=args.kernel_size,
            stride=args.stride,
            day_hidden=args.day_hidden,
            bidirectional=False,  # paper_exact is always unidirectional
        )
    else:
        model = DCoNDDecoder(
            n_features_per_frame=args.input_dim,
            n_diphones=N_DIPHONE_CLASSES,
            hidden=args.hidden,
            n_layers=args.n_layers,
            dropout=args.dropout,
            n_sessions=n_sessions,
            kernel_size=args.kernel_size,
            stride=args.stride,
            bidirectional=args.bidirectional,
            day_hidden=args.day_hidden,
            post_rnn_norm=args.post_rnn_norm,
        )
    model.set_train_sessions(train_session_idxs)

    # Initialize from checkpoint (transfer learning)
    if args.init_from:
        print(f'\nLoading weights from: {args.init_from}')
        if args.arch == 'paper_exact':
            model.load_paper_exact_weights(args.init_from)
        else:
            ckpt = torch.load(args.init_from, map_location='cpu', weights_only=True)
            src_state = ckpt.get('model_state_dict', ckpt)
            src_state = {k.replace('module.', ''): v for k, v in src_state.items()}
            tgt_state = model.state_dict()
            loaded, skipped = 0, 0
            for k, v in src_state.items():
                if k in tgt_state and tgt_state[k].shape == v.shape:
                    tgt_state[k] = v
                    loaded += 1
                else:
                    skipped += 1
            model.load_state_dict(tgt_state)
            print(f'  Loaded {loaded} params, skipped {skipped} (shape mismatch)')

    n_params = sum(p.numel() for p in model.parameters())
    actual_class = type(model.module if hasattr(model, 'module') else model).__name__
    print(f'\nModel: {actual_class} ({n_params / 1e6:.1f}M params)')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Marginalization matrix for evaluation
    M_ext = build_marginalization_matrix_ext()  # (1601, 41), kept on CPU

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=args.momentum, nesterov=args.nesterov,
            weight_decay=args.l2_reg)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr,
            betas=(0.9, 0.999), eps=args.adam_eps,
            weight_decay=args.l2_reg)

    # Determine total epochs
    batches_per_epoch = len(train_loader)
    if args.max_minibatches:
        total_epochs = max(1, args.max_minibatches // batches_per_epoch)
    else:
        total_epochs = args.max_epochs

    # Scheduler
    if args.scheduler == 'step' and args.step_size:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma)
        step_per_batch = True
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.001,
            total_iters=total_epochs)
        step_per_batch = False
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6)
        step_per_batch = False
    else:
        scheduler = None
        step_per_batch = False

    print(f'Training: {total_epochs} epochs, {batches_per_epoch} batches/epoch')
    print(f'Optimizer: {args.optimizer.upper()}, Scheduler: {args.scheduler}')
    if args.progressive_alpha:
        print(f'Progressive alpha: 1.0\u2192{args.alpha} over 50 epochs')
    if args.reverse_alpha:
        print(f'Reverse alpha: 0.0\u2192{args.alpha} over 50 epochs (monophone first)')
    if args.warmup_epochs > 0:
        print(f'LR warmup: {args.warmup_epochs} epochs (0\u2192{args.lr})')

    # DCoND loss
    criterion = DiphoneCTCLoss(n_mono=N_MONO, alpha=args.alpha)
    criterion = criterion.to(device)
    scaler = GradScaler()

    # ── Phase 1: Monophone pre-training (direct 41-class CTC head) ──
    if args.pretrain_mono_epochs > 0:
        print(f'\n{"=" * 70}')
        print(f'PHASE 1: Monophone pre-training ({args.pretrain_mono_epochs} epochs)')
        print(f'  Using direct 41-class mono_head (bypasses 1601-class softmax)')
        print(f'{"=" * 70}')
        mono_ctc = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
        for epoch in range(args.pretrain_mono_epochs):
            t0 = time.time()
            # LR warmup for pre-training phase
            if epoch < min(3, args.pretrain_mono_epochs):
                warmup_lr = args.lr * (epoch + 1) / min(3, args.pretrain_mono_epochs)
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr

            # Train one epoch with direct monophone head
            m = model.module if hasattr(model, 'module') else model
            model.train()
            total_loss = 0
            n_batches = 0
            for batch in train_loader:
                (features, mono_targets, diphone_targets,
                 feat_lens, mono_tgt_lens, diphone_tgt_lens, session_ids) = batch
                features = features.to(device)
                mono_targets = mono_targets.to(device)
                session_ids = session_ids.to(device)
                if args.white_noise > 0:
                    features = features + torch.randn_like(features) * args.white_noise
                if args.offset_noise > 0:
                    B, T, C = features.shape
                    features = features + torch.randn(B, 1, C, device=device) * args.offset_noise
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    mono_logits = m.forward_mono(features, session_ids)
                    out_lens = ((feat_lens - args.kernel_size) // args.stride + 1).clamp(min=1)
                    input_lengths = out_lens.clamp(max=mono_logits.shape[1]).to(device)
                    mono_lp = mono_logits.log_softmax(-1).transpose(0, 1)
                    loss = mono_ctc(mono_lp, mono_targets,
                                    input_lengths, mono_tgt_lens.to(device))
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Evaluate with direct monophone head
            model.eval()
            all_per = []
            with torch.no_grad():
                for batch in val_loader:
                    (features, mono_targets_v, diphone_targets_v,
                     feat_lens, mono_tgt_lens_v, diphone_tgt_lens_v, session_ids) = batch
                    features = features.to(device)
                    session_ids = session_ids.to(device)
                    with autocast("cuda"):
                        mono_logits = m.forward_mono(features, session_ids)
                    mono_lp = mono_logits.float().log_softmax(-1).cpu().numpy()
                    out_lens = ((feat_lens - args.kernel_size) // args.stride + 1).clamp(min=1)
                    offset = 0
                    for i in range(len(feat_lens)):
                        T_out = min(out_lens[i].item(), mono_lp.shape[1])
                        P = mono_tgt_lens_v[i].item()
                        decoded = ctc_greedy_decode(mono_lp[i, :T_out, :], blank=CTC_BLANK)
                        target = mono_targets_v[offset:offset + P].numpy().tolist()
                        offset += P
                        all_per.append(compute_per(decoded, target))
            val_per = float(np.mean(all_per)) if all_per else 1.0

            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]['lr']
            print(f'Pre-train {epoch + 1:3d}/{args.pretrain_mono_epochs} | '
                  f'loss={avg_loss:.4f} | '
                  f'val_PER={val_per:.3f} | '
                  f'lr={cur_lr:.2e} | {elapsed:.1f}s')

        # Phase 2 lr: much lower to avoid CTC collapse with 1601-class softmax
        phase2_lr = args.phase2_lr if args.phase2_lr else args.lr / 5
        print(f'Phase 1 done (val_PER={val_per:.3f}). '
              f'Switching to Phase 2 (DCoND alpha={args.alpha}, lr={phase2_lr:.1e}).\n')
        for pg in optimizer.param_groups:
            pg['lr'] = phase2_lr
        # Update args.lr for warmup calculation in Phase 2
        args.lr = phase2_lr

    best_val_per = float('inf')
    best_state = None
    wait = 0
    history = []

    for epoch in range(total_epochs):
        t0 = time.time()

        # LR warmup: linearly increase from lr/20 to target lr
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # Progressive alpha schedule
        if args.progressive_alpha or args.reverse_alpha:
            current_alpha = get_progressive_alpha(
                epoch + 1, target=args.alpha, reverse=args.reverse_alpha)
            criterion.alpha = current_alpha
        else:
            current_alpha = args.alpha

        train_loss, lc_avg, ls_avg = train_epoch_dcond(
            model, train_loader, optimizer, criterion, scaler, device,
            white_noise_sd=args.white_noise,
            offset_noise_sd=args.offset_noise,
            kernel_size=args.kernel_size, stride=args.stride,
            speckled_mask=args.speckled_mask,
            fastemit_lambda=args.fastemit,
            grad_clip=args.grad_clip,
            use_dual_head=args.dual_head,
            di_weight=args.di_weight,
        )

        val_metrics = evaluate_dcond(model, val_loader, device, M_ext,
                                     kernel_size=args.kernel_size,
                                     stride=args.stride,
                                     use_dual_head=args.dual_head)

        # Step scheduler
        if scheduler and not step_per_batch:
            scheduler.step()
        elif scheduler and step_per_batch:
            # StepLR already stepped per batch in training loop? No, step per epoch
            scheduler.step()

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

        per_str = f'val_PER={val_metrics["per"]:.3f}'
        if 'per_direct' in val_metrics:
            per_str += f' (mono={val_metrics["per_direct"]:.3f})'
        print(f'Epoch {epoch + 1:3d}/{total_epochs} | '
              f'loss={train_loss:.4f} (Lc={lc_avg:.4f} Ls={ls_avg:.4f}) | '
              f'{per_str} | '
              f'α={current_alpha:.2f} | '
              f'lr={cur_lr:.2e} | '
              f'{elapsed:.1f}s')

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'loss_diphone': lc_avg,
            'loss_monophone': ls_avg,
            'val_per': val_metrics['per'],
            'alpha': current_alpha,
            'lr': cur_lr,
            'elapsed': elapsed,
        })

        # Track best: always use marginalized diphone PER for early stopping
        # (this is the DCoND metric we care about; mono PER is just for reference)
        track_per = val_metrics['per']
        if track_per < best_val_per:
            best_val_per = track_per
            m = model.module if hasattr(model, 'module') else model
            best_state = {k: v.cpu().clone()
                          for k, v in m.state_dict().items()}
            wait = 0
            print(f'  *** New best val PER: {best_val_per:.3f} ***')
        else:
            wait += 1

        if wait >= args.patience:
            print(f'Early stop at epoch {epoch + 1}')
            break

    # Test
    m = model.module if hasattr(model, 'module') else model
    if best_state:
        m.load_state_dict(best_state)
    model.eval()

    test_metrics = evaluate_dcond(model, test_loader, device, M_ext,
                                  kernel_size=args.kernel_size,
                                  stride=args.stride,
                                  use_dual_head=args.dual_head)

    print(f'\n{"=" * 70}')
    print(f'RESULTS — {args.experiment}')
    print(f'  Val PER (diphone):  {best_val_per:.1%}')
    print(f'  Test PER (diphone): {test_metrics["per"]:.1%}')
    if 'per_direct' in test_metrics:
        print(f'  Test PER (mono):    {test_metrics["per_direct"]:.1%}')
    print(f'  Alpha:    {args.alpha} '
          f'{"(progressive)" if args.progressive_alpha else "(fixed)"}')
    print(f'  DCoND paper target: ~15% PER (diphone, pre-LM)')
    print(f'{"=" * 70}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save results JSON
    results = {
        'experiment': args.experiment,
        'lead': 'L2_DCoND',
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

    save_path = RESULTS_DIR / f'L2_{args.experiment}_{timestamp}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results: {save_path}')

    # Save checkpoint in cross-lead compatible format
    model_path = RESULTS_DIR / f'L2_{args.experiment}_best.pt'
    model_class = 'PaperExactDCoND' if args.arch == 'paper_exact' else 'DCoNDDecoder'
    model_kwargs = {
        'n_features_per_frame': args.input_dim,
        'n_diphones': N_DIPHONE_CLASSES,
        'hidden': args.hidden,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'n_sessions': n_sessions,
        'kernel_size': args.kernel_size,
        'stride': args.stride,
        'day_hidden': args.day_hidden,
    }
    if args.arch == 'paper_exact':
        model_kwargs['bidirectional'] = False
        model_kwargs['post_rnn_norm'] = False
    else:
        model_kwargs['bidirectional'] = args.bidirectional
        model_kwargs['post_rnn_norm'] = args.post_rnn_norm
    torch.save({
        'model_state_dict': best_state,
        'model_class': model_class,
        'model_kwargs': model_kwargs,
        'n_classes': N_CLASSES + 1,  # 41 monophones (after marginalization)
        'is_diphone': True,
        'val_per': float(best_val_per),
        'test_per': test_metrics['per'],
        'experiment': args.experiment,
        'seed': args.seed,
    }, model_path)
    print(f'Checkpoint: {model_path}')


if __name__ == '__main__':
    main()
