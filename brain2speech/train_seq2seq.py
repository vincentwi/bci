#!/usr/bin/env python3
"""
Autoregressive and transducer models for phoneme decoding from neural signals.

Goes beyond CTC (which has conditionally independent outputs) by adding:
- RNN-T (Transducer): encoder + prediction network → joint network → transducer loss
- Seq2Seq + Attention: encoder-decoder with location-sensitive attention
- LSTM-CTC: LSTM encoder with CTC loss (drop-in replacement for GRU)
- Causal GRU/LSTM: unidirectional encoders for streaming BCI

The key insight: CTC can't model P(phoneme_t | phoneme_{t-1}) because outputs are
conditionally independent. Autoregressive models learn phonotactic constraints implicitly.

Usage:
    python train_seq2seq.py --model RNN-T --gpus 0 1 --patience 10
    python train_seq2seq.py --model Seq2Seq --gpus 2 3 --patience 10
    python train_seq2seq.py --model LSTM-CTC --gpus 4 5 --patience 10
    python train_seq2seq.py --model CausalGRU-CTC --gpus 6 7 --patience 10
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")

SEED = 42
N_CLASSES = 40   # 39 phonemes + SIL
CTC_BLANK = N_CLASSES  # 40
SOS_TOKEN = N_CLASSES + 1  # 41 — start of sequence for seq2seq
EOS_TOKEN = N_CLASSES + 2  # 42 — end of sequence for seq2seq
PAD_TOKEN = -1


def smooth_features(features, sigma):
    if sigma <= 0:
        return features
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(features, sigma=sigma, axis=0)


def load_h5_dataset(h5_path, max_trials=None, smooth_sigma=0):
    import h5py
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


# ─── Collate functions ───────────────────────────────────────────────────

def collate_ctc(batch):
    """CTC collate: pad features, concatenate targets."""
    import torch
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


def collate_seq2seq(batch):
    """Seq2Seq collate: pad features AND targets separately."""
    import torch
    features = [torch.FloatTensor(t['features']) for t in batch]
    # Add SOS/EOS tokens to targets
    targets = [torch.LongTensor(
        [SOS_TOKEN] + t['phoneme_indices'].tolist() + [EOS_TOKEN]
    ) for t in batch]

    feature_lengths = torch.LongTensor([f.shape[0] for f in features])
    target_lengths = torch.LongTensor([t.shape[0] for t in targets])

    max_T = max(f.shape[0] for f in features)
    max_U = max(t.shape[0] for t in targets)
    C = features[0].shape[1]

    feat_padded = torch.zeros(len(features), max_T, C)
    tgt_padded = torch.full((len(targets), max_U), PAD_TOKEN, dtype=torch.long)

    for i in range(len(features)):
        feat_padded[i, :features[i].shape[0], :] = features[i]
        tgt_padded[i, :targets[i].shape[0]] = targets[i]

    return feat_padded, tgt_padded, feature_lengths, target_lengths


def collate_rnnt(batch):
    """RNN-T collate: pad features and targets separately (no SOS/EOS)."""
    import torch
    features = [torch.FloatTensor(t['features']) for t in batch]
    targets = [torch.IntTensor(t['phoneme_indices']) for t in batch]
    feature_lengths = torch.IntTensor([f.shape[0] for f in features])
    target_lengths = torch.IntTensor([t.shape[0] for t in targets])
    max_T = max(f.shape[0] for f in features)
    max_U = max(t.shape[0] for t in targets)
    C = features[0].shape[1]
    feat_padded = torch.zeros(len(features), max_T, C)
    tgt_padded = torch.zeros(len(targets), max_U, dtype=torch.int32)
    for i in range(len(features)):
        feat_padded[i, :features[i].shape[0], :] = features[i]
        tgt_padded[i, :targets[i].shape[0]] = targets[i]
    return feat_padded, tgt_padded, feature_lengths, target_lengths


# ─── Models ──────────────────────────────────────────────────────────────

def build_model(model_type, n_features=1280, hidden=512, n_layers=5, dr=0.3):
    """Build encoder-decoder model."""
    import torch
    import torch.nn as nn

    if model_type == 'LSTM-CTC':
        return _build_lstm_ctc(n_features, hidden, n_layers, dr)
    elif model_type == 'CausalGRU-CTC':
        return _build_causal_gru_ctc(n_features, hidden, n_layers, dr)
    elif model_type == 'CausalLSTM-CTC':
        return _build_causal_lstm_ctc(n_features, hidden, n_layers, dr)
    elif model_type == 'RNN-T':
        return _build_rnnt(n_features, hidden, n_layers, dr)
    elif model_type == 'Seq2Seq':
        return _build_seq2seq(n_features, hidden, n_layers, dr)
    else:
        raise ValueError(f"Unknown model: {model_type}")


def _build_lstm_ctc(n_features, hidden, n_layers, dr):
    """LSTM encoder + CTC — bidirectional, drop-in replacement for GRU."""
    import torch.nn as nn
    n_classes = N_CLASSES + 1  # +blank

    class LSTMCTCEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_features, hidden)
            self.input_norm = nn.LayerNorm(hidden)
            self.rnn = nn.LSTM(
                hidden, hidden, n_layers,
                batch_first=True, bidirectional=True, dropout=dr
            )
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Dropout(dr),
                nn.Linear(hidden * 2, n_classes),
            )

        def forward(self, x, lengths=None):
            x = self.input_norm(self.input_proj(x))
            x, _ = self.rnn(x)
            return self.output_proj(x)

    return LSTMCTCEncoder()


def _build_causal_gru_ctc(n_features, hidden, n_layers, dr):
    """Unidirectional GRU + CTC — causal, for streaming BCI."""
    import torch.nn as nn
    n_classes = N_CLASSES + 1

    class CausalGRUCTCEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_features, hidden)
            self.input_norm = nn.LayerNorm(hidden)
            self.rnn = nn.GRU(
                hidden, hidden, n_layers,
                batch_first=True, bidirectional=False, dropout=dr
            )
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(dr),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x, lengths=None):
            x = self.input_norm(self.input_proj(x))
            x, _ = self.rnn(x)
            return self.output_proj(x)

    return CausalGRUCTCEncoder()


def _build_causal_lstm_ctc(n_features, hidden, n_layers, dr):
    """Unidirectional LSTM + CTC — causal, for streaming BCI."""
    import torch.nn as nn
    n_classes = N_CLASSES + 1

    class CausalLSTMCTCEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_features, hidden)
            self.input_norm = nn.LayerNorm(hidden)
            self.rnn = nn.LSTM(
                hidden, hidden, n_layers,
                batch_first=True, bidirectional=False, dropout=dr
            )
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(dr),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x, lengths=None):
            x = self.input_norm(self.input_proj(x))
            x, _ = self.rnn(x)
            return self.output_proj(x)

    return CausalLSTMCTCEncoder()


def _build_rnnt(n_features, hidden, n_layers, dr):
    """RNN-Transducer: encoder + prediction network + joint network.

    The prediction network is autoregressive over previous phoneme outputs,
    giving the model P(y_t | x_{1:T}, y_{1:t-1}) instead of CTC's P(y_t | x_{1:T}).
    """
    import torch
    import torch.nn as nn

    class RNNTransducer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_classes = N_CLASSES + 1  # +blank
            self.hidden = hidden

            # Encoder: bidirectional LSTM over neural features
            self.enc_proj = nn.Linear(n_features, hidden)
            self.enc_norm = nn.LayerNorm(hidden)
            self.encoder = nn.LSTM(
                hidden, hidden, n_layers,
                batch_first=True, bidirectional=True, dropout=dr
            )
            self.enc_out = nn.Linear(hidden * 2, hidden)

            # Prediction network: unidirectional LSTM over previous outputs
            self.pred_embed = nn.Embedding(self.n_classes, 256)
            self.pred_rnn = nn.LSTM(
                256, hidden, 2,
                batch_first=True, dropout=dr
            )
            self.pred_out = nn.Linear(hidden, hidden)

            # Joint network
            self.joint = nn.Sequential(
                nn.Tanh(),
                nn.Linear(hidden, self.n_classes),
            )

        def encode(self, x):
            """Encode neural features: (B, T, 1280) → (B, T, hidden)."""
            x = self.enc_norm(self.enc_proj(x))
            x, _ = self.encoder(x)
            return self.enc_out(x)

        def predict(self, y_prev):
            """Prediction network: (B, U) → (B, U, hidden)."""
            emb = self.pred_embed(y_prev)
            out, _ = self.pred_rnn(emb)
            return self.pred_out(out)

        def joint_forward(self, enc_out, pred_out):
            """Joint network: (B, T, 1, H) + (B, 1, U+1, H) → (B, T, U+1, classes)."""
            # Broadcast: enc (B,T,1,H) + pred (B,1,U+1,H) → (B,T,U+1,H)
            return self.joint(enc_out.unsqueeze(2) + pred_out.unsqueeze(1))

        def forward(self, x, targets, target_lengths=None):
            """Full forward for training.

            Args:
                x: (B, T, 1280) neural features
                targets: (B, U_max) phoneme indices (padded)
                target_lengths: (B,) actual target lengths

            Returns:
                logits: (B, T, U+1, n_classes) for transducer loss
            """
            enc_out = self.encode(x)  # (B, T, H)

            # Prepend blank/SOS token to targets for prediction network
            B = targets.shape[0]
            blank_start = torch.zeros(B, 1, dtype=targets.dtype, device=targets.device)
            y_prev = torch.cat([blank_start, targets], dim=1)  # (B, U+1)
            pred_out = self.predict(y_prev)  # (B, U+1, H)

            logits = self.joint_forward(enc_out, pred_out)  # (B, T, U+1, C)
            return logits

        def greedy_decode(self, x):
            """Greedy RNN-T decode for a single sample."""
            enc_out = self.encode(x.unsqueeze(0))  # (1, T, H)
            T = enc_out.shape[1]

            device = x.device
            blank = N_CLASSES

            decoded = []
            # Init prediction network state
            y_prev = torch.zeros(1, 1, dtype=torch.long, device=device)  # blank
            pred_state = None

            for t in range(T):
                enc_t = enc_out[:, t:t+1, :]  # (1, 1, H)

                for _ in range(50):  # max emissions per frame
                    emb = self.pred_embed(y_prev)
                    pred_out, pred_state_new = self.pred_rnn(emb, pred_state)
                    pred_out = self.pred_out(pred_out)

                    logits = self.joint(enc_t + pred_out)  # (1, 1, C)
                    token = logits.argmax(dim=-1).item()

                    if token == blank:
                        break
                    decoded.append(token)
                    y_prev = torch.tensor([[token]], dtype=torch.long, device=device)
                    pred_state = pred_state_new

            return decoded

    return RNNTransducer()


def _build_seq2seq(n_features, hidden, n_layers, dr):
    """Seq2Seq with location-sensitive attention.

    Encoder: bidirectional LSTM over neural features
    Decoder: unidirectional LSTM, autoregressive over phoneme outputs
    Attention: additive (Bahdanau) with location features
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Seq2SeqAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_classes = N_CLASSES + 3  # phonemes + SOS + EOS
            self.hidden = hidden

            # Encoder
            self.enc_proj = nn.Linear(n_features, hidden)
            self.enc_norm = nn.LayerNorm(hidden)
            self.encoder = nn.LSTM(
                hidden, hidden, n_layers,
                batch_first=True, bidirectional=True, dropout=dr
            )
            self.enc_out_proj = nn.Linear(hidden * 2, hidden)

            # Decoder
            self.dec_embed = nn.Embedding(self.n_classes, 256)
            self.dec_rnn = nn.LSTMCell(256 + hidden, hidden)
            self.dec_rnn2 = nn.LSTMCell(hidden, hidden)

            # Attention (Bahdanau with location)
            self.attn_W = nn.Linear(hidden, 128, bias=False)
            self.attn_V = nn.Linear(hidden, 128, bias=False)
            self.attn_loc = nn.Conv1d(1, 32, kernel_size=31, padding=15)
            self.attn_loc_proj = nn.Linear(32, 128, bias=False)
            self.attn_score = nn.Linear(128, 1, bias=False)

            # Output
            self.output_proj = nn.Linear(hidden + hidden, self.n_classes)
            self.dropout = nn.Dropout(dr)

        def encode(self, x):
            x = self.enc_norm(self.enc_proj(x))
            x, _ = self.encoder(x)
            return self.enc_out_proj(x)  # (B, T, H)

        def attention(self, dec_hidden, enc_out, prev_attn_weights):
            """Compute attention context.

            Args:
                dec_hidden: (B, H) decoder hidden state
                enc_out: (B, T, H) encoder outputs
                prev_attn_weights: (B, T) previous attention weights

            Returns:
                context: (B, H)
                attn_weights: (B, T)
            """
            # Location features
            loc_feat = self.attn_loc(prev_attn_weights.unsqueeze(1))  # (B, 32, T)
            loc_feat = self.attn_loc_proj(loc_feat.transpose(1, 2))   # (B, T, 128)

            # Score
            energy = self.attn_score(torch.tanh(
                self.attn_W(enc_out) +           # (B, T, 128)
                self.attn_V(dec_hidden).unsqueeze(1) +  # (B, 1, 128)
                loc_feat                          # (B, T, 128)
            )).squeeze(-1)                        # (B, T)

            attn_weights = F.softmax(energy, dim=-1)  # (B, T)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)  # (B, H)
            return context, attn_weights

        def forward(self, x, targets, feature_lengths=None, target_lengths=None):
            """Teacher-forced forward pass.

            Args:
                x: (B, T, 1280)
                targets: (B, U_max) with SOS prepended, EOS appended
            """
            enc_out = self.encode(x)  # (B, T, H)
            B, T, H = enc_out.shape
            U = targets.shape[1]

            device = x.device

            # Init decoder state
            h1 = torch.zeros(B, H, device=device)
            c1 = torch.zeros(B, H, device=device)
            h2 = torch.zeros(B, H, device=device)
            c2 = torch.zeros(B, H, device=device)
            attn_weights = torch.zeros(B, T, device=device)
            context = torch.zeros(B, H, device=device)

            outputs = []
            # Teacher forcing: feed ground truth tokens
            for u in range(U - 1):  # -1 because last token is EOS (no prediction after)
                token = targets[:, u].clamp(min=0)  # (B,) clamp PAD to 0
                emb = self.dec_embed(token)  # (B, 256)

                rnn_input = torch.cat([emb, context], dim=-1)  # (B, 256+H)
                h1, c1 = self.dec_rnn(rnn_input, (h1, c1))
                h2, c2 = self.dec_rnn2(self.dropout(h1), (h2, c2))

                context, attn_weights = self.attention(h2, enc_out, attn_weights)

                output = self.output_proj(torch.cat([h2, context], dim=-1))  # (B, C)
                outputs.append(output)

            return torch.stack(outputs, dim=1)  # (B, U-1, C)

        def greedy_decode(self, x, max_len=200):
            """Greedy autoregressive decode for a single sample."""
            enc_out = self.encode(x.unsqueeze(0))  # (1, T, H)
            B, T, H = enc_out.shape
            device = x.device

            h1 = torch.zeros(1, H, device=device)
            c1 = torch.zeros(1, H, device=device)
            h2 = torch.zeros(1, H, device=device)
            c2 = torch.zeros(1, H, device=device)
            attn_weights = torch.zeros(1, T, device=device)
            context = torch.zeros(1, H, device=device)

            token = torch.tensor([SOS_TOKEN], device=device)
            decoded = []

            for _ in range(max_len):
                emb = self.dec_embed(token)
                rnn_input = torch.cat([emb, context], dim=-1)
                h1, c1 = self.dec_rnn(rnn_input, (h1, c1))
                h2, c2 = self.dec_rnn2(self.dropout(h1), (h2, c2))
                context, attn_weights = self.attention(h2, enc_out, attn_weights)
                output = self.output_proj(torch.cat([h2, context], dim=-1))
                token = output.argmax(dim=-1)

                if token.item() == EOS_TOKEN:
                    break
                if token.item() < N_CLASSES:
                    decoded.append(token.item())

            return decoded

    return Seq2SeqAttention()


# ─── Training functions ──────────────────────────────────────────────────

def compute_per(predicted, target):
    import editdistance
    if len(target) == 0:
        return 1.0 if len(predicted) > 0 else 0.0
    return editdistance.eval(predicted, target) / len(target)


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


def train_ctc_epoch(model, dataloader, optimizer, criterion, scaler, device, noise_std=0.05):
    import torch
    from torch.amp import autocast
    model.train()
    total_loss = 0
    n_batches = 0
    for features, targets, feat_lens, tgt_lens in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        if noise_std > 0 and np.random.random() < 0.5:
            features = features + torch.randn_like(features) * noise_std
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            log_probs = model(features).log_softmax(dim=2)
            log_probs_t = log_probs.transpose(0, 1)
            input_lengths = torch.full((log_probs.shape[0],), log_probs.shape[1],
                                       dtype=torch.int32, device=device)
            loss = criterion(log_probs_t, targets, input_lengths, tgt_lens.to(device))
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


def eval_ctc(model, dataloader, device):
    import torch
    from torch.amp import autocast
    model.eval()
    all_per = []
    with torch.no_grad():
        for features, targets, feat_lens, tgt_lens in dataloader:
            features = features.to(device)
            with autocast("cuda"):
                log_probs = model(features).log_softmax(dim=2).cpu().numpy()
            offset = 0
            for i in range(len(feat_lens)):
                T = min(feat_lens[i].item(), log_probs.shape[1])
                P = tgt_lens[i].item()
                decoded = ctc_greedy_decode(log_probs[i, :T, :])
                trial_targets = targets[offset:offset + P].numpy().tolist()
                offset += P
                all_per.append(compute_per(decoded, trial_targets))
    return {'per': float(np.mean(all_per)) if all_per else 1.0}


def train_seq2seq_epoch(model, dataloader, optimizer, criterion, scaler, device, noise_std=0.05):
    import torch
    from torch.amp import autocast
    model.train()
    total_loss = 0
    n_batches = 0
    for features, targets, feat_lens, tgt_lens in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        if noise_std > 0 and np.random.random() < 0.5:
            features = features + torch.randn_like(features) * noise_std
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            # targets has SOS prepended and EOS appended
            logits = model(features, targets, feat_lens, tgt_lens)  # (B, U-1, C)
            # Target for loss: shift targets by 1 (predict next token)
            target_shifted = targets[:, 1:]  # remove SOS, keep EOS
            # Mask out padding positions
            B, U_minus1, C = logits.shape
            mask = target_shifted != PAD_TOKEN
            logits_flat = logits[mask]  # (valid_tokens, C)
            targets_flat = target_shifted[mask]  # (valid_tokens,)
            if logits_flat.shape[0] == 0:
                continue
            loss = criterion(logits_flat, targets_flat)
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


def eval_seq2seq(model, dataloader, device):
    import torch
    from torch.amp import autocast
    model.eval()
    all_per = []
    with torch.no_grad():
        for features, targets, feat_lens, tgt_lens in dataloader:
            features = features.to(device)
            for i in range(features.shape[0]):
                T = feat_lens[i].item()
                sample = features[i, :T, :]
                decoded = model.module.greedy_decode(sample) if hasattr(model, 'module') else model.greedy_decode(sample)
                # Extract ground truth (remove SOS and EOS)
                tgt = targets[i, 1:tgt_lens[i].item()-1].numpy().tolist()
                all_per.append(compute_per(decoded, tgt))
    return {'per': float(np.mean(all_per)) if all_per else 1.0}


def train_rnnt_epoch(model, dataloader, optimizer, scaler, device, noise_std=0.05):
    """Train one epoch with RNN-T loss."""
    import torch
    from torch.amp import autocast
    try:
        import torchaudio
        rnnt_loss = torchaudio.transforms.RNNTLoss(blank=N_CLASSES)
    except (ImportError, AttributeError):
        # Fallback: use simple frame-level approximation
        print("WARNING: torchaudio RNNTLoss not available, using approximation")
        return _train_rnnt_approx_epoch(model, dataloader, optimizer, scaler, device, noise_std)

    model.train()
    total_loss = 0
    n_batches = 0
    for features, targets, feat_lens, tgt_lens in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        feat_lens = feat_lens.to(device)
        tgt_lens = tgt_lens.to(device)

        if noise_std > 0 and np.random.random() < 0.5:
            features = features + torch.randn_like(features) * noise_std

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            logits = model(features, targets, tgt_lens)  # (B, T, U+1, C)
            log_probs = logits.log_softmax(dim=-1)
            loss = rnnt_loss(log_probs, targets.int(), feat_lens.int(), tgt_lens.int())

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


def _train_rnnt_approx_epoch(model, dataloader, optimizer, scaler, device, noise_std):
    """Approximate RNN-T training using CTC loss on encoder output.

    This is a fallback when torchaudio.transforms.RNNTLoss is unavailable.
    We train the encoder with CTC and the prediction network separately.
    """
    import torch
    import torch.nn as nn
    from torch.amp import autocast
    criterion = nn.CTCLoss(blank=N_CLASSES, zero_infinity=True)
    model.train()
    total_loss = 0
    n_batches = 0
    for features, targets, feat_lens, tgt_lens in dataloader:
        features = features.to(device)
        # Concatenate targets for CTC
        targets_cat = torch.cat([targets[i, :tgt_lens[i]] for i in range(targets.shape[0])]).to(device)
        if noise_std > 0 and np.random.random() < 0.5:
            features = features + torch.randn_like(features) * noise_std
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            enc_out = model.encode(features)  # (B, T, H)
            # Simple CTC head from encoder
            logits = model.joint.forward(enc_out)  # (B, T, C)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            input_lengths = torch.full((features.shape[0],), enc_out.shape[1],
                                       dtype=torch.int32, device=device)
            loss = criterion(log_probs, targets_cat, input_lengths, tgt_lens.to(device))
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


def eval_rnnt(model, dataloader, device):
    """Evaluate RNN-T with greedy decode."""
    import torch
    from torch.amp import autocast
    model.eval()
    all_per = []
    m = model.module if hasattr(model, 'module') else model
    with torch.no_grad():
        for features, targets, feat_lens, tgt_lens in dataloader:
            features = features.to(device)
            for i in range(features.shape[0]):
                T = feat_lens[i].item()
                sample = features[i, :T, :]
                decoded = m.greedy_decode(sample)
                tgt = targets[i, :tgt_lens[i].item()].numpy().tolist()
                all_per.append(compute_per(decoded, tgt))
    return {'per': float(np.mean(all_per)) if all_per else 1.0}


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=[
        'LSTM-CTC', 'CausalGRU-CTC', 'CausalLSTM-CTC',
        'RNN-T', 'Seq2Seq'
    ], required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--smooth', type=float, default=2)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-trials', type=int, default=None)
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

    is_ctc = args.model.endswith('-CTC')
    is_rnnt = args.model == 'RNN-T'
    is_seq2seq = args.model == 'Seq2Seq'

    print('=' * 70)
    print(f'SEQ2SEQ PHONEME DECODING — {args.model}')
    print(f'  Device: {device} (GPUs: {args.gpus})')
    print(f'  Type: {"CTC" if is_ctc else "RNN-T" if is_rnnt else "Seq2Seq"}')
    print(f'  Epochs: {args.epochs}, BS: {args.batch_size}, LR: {args.lr}')
    print(f'  Patience: {args.patience} (fast iteration)')
    print('=' * 70)

    # Load data
    h5_path = DATA_DIR / 'sentences_train.h5'
    if not h5_path.exists():
        print(f'ERROR: {h5_path} not found.')
        sys.exit(1)

    trials = load_h5_dataset(h5_path, max_trials=args.max_trials, smooth_sigma=args.smooth)

    # Split by session
    sessions = {}
    for t in trials:
        s = t['session']
        sessions.setdefault(s, []).append(t)

    session_names = sorted(sessions.keys())
    print(f'\nSessions: {len(session_names)}')

    # Holdout split: last 4 test, last 2 of train = val
    train_sessions = session_names[:-4]
    test_sessions = session_names[-4:]
    val_sessions = train_sessions[-2:]
    train_sessions_final = train_sessions[:-2]

    train_trials = [t for s in train_sessions_final for t in sessions[s]]
    val_trials = [t for s in val_sessions for t in sessions[s]]
    test_trials = [t for s in test_sessions for t in sessions[s]]

    print(f'Train: {len(train_trials)}, Val: {len(val_trials)}, Test: {len(test_trials)}')

    # Collate function depends on model type
    if is_ctc:
        collate_fn = collate_ctc
    elif is_rnnt:
        collate_fn = collate_rnnt
    else:
        collate_fn = collate_seq2seq

    train_loader = DataLoader(train_trials, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_trials, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_trials, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)

    # Build model
    model = build_model(args.model, n_features=1280, hidden=args.hidden,
                        n_layers=args.n_layers, dr=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.model}, {n_params/1e6:.1f}M params')

    if len(args.gpus) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)

    if is_ctc:
        criterion = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
    elif is_seq2seq:
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    else:
        criterion = None  # RNN-T uses its own loss

    scaler = GradScaler()

    best_val_per = float('inf')
    best_state = None
    wait = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        if is_ctc:
            train_loss = train_ctc_epoch(model, train_loader, optimizer, criterion,
                                          scaler, device, args.noise)
        elif is_seq2seq:
            train_loss = train_seq2seq_epoch(model, train_loader, optimizer, criterion,
                                              scaler, device, args.noise)
        else:  # RNN-T
            train_loss = train_rnnt_epoch(model, train_loader, optimizer,
                                           scaler, device, args.noise)

        # Eval
        if is_ctc:
            val_metrics = eval_ctc(model, val_loader, device)
        elif is_rnnt:
            val_metrics = eval_rnnt(model, val_loader, device)
        else:
            val_metrics = eval_seq2seq(model, val_loader, device)

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

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
            'elapsed': elapsed,
        })

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

    # Test
    m = model.module if hasattr(model, 'module') else model
    if best_state:
        m.load_state_dict(best_state)
    model.eval()

    if is_ctc:
        test_metrics = eval_ctc(model, test_loader, device)
    elif is_rnnt:
        test_metrics = eval_rnnt(model, test_loader, device)
    else:
        test_metrics = eval_seq2seq(model, test_loader, device)

    print(f'\n{"="*60}')
    print(f'TEST RESULTS — {args.model}')
    print(f'  PER: {test_metrics["per"]:.1%}')
    print(f'  Paper baseline PER: 19.7%')
    print(f'{"="*60}')

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'model': args.model,
        'test_per': test_metrics['per'],
        'best_val_per': best_val_per,
        'n_train': len(train_trials),
        'n_val': len(val_trials),
        'n_test': len(test_trials),
        'epochs_trained': epoch + 1,
        'n_params': n_params,
        'hyperparams': vars(args),
        'history': history,
    }
    save_path = RESULTS_DIR / f'seq2seq_{args.model}_{timestamp}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {save_path}')

    model_path = RESULTS_DIR / f'seq2seq_{args.model}_best.pt'
    torch.save(best_state, model_path)
    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
