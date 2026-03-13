"""
Speech BCI Model Architectures
===============================
BiLSTM, TCN, and Transformer models for:
  - Word classification (ECoG high-gamma → word label)
  - Acoustic decoding (ECoG high-gamma → mel spectrogram)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BiLSTM Models (baseline)
# ============================================================

class WordLSTM(nn.Module):
    """BiLSTM word classifier: HG features → word label.

    Architecture: BiLSTM(d_in, d_hid) → concat last hidden states →
                  Linear(2*d_hid, d_hid) → ReLU → Dropout → Linear(d_hid, n_cls)
    """
    def __init__(self, d_in, d_hid, n_cls, n_layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hid, n_layers, batch_first=True,
                            dropout=drop, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(d_hid * 2, d_hid), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d_hid, n_cls))

    def forward(self, x, lens):
        pk = nn.utils.rnn.pack_padded_sequence(
            x, lens, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(pk)
        return self.head(torch.cat([h[-2], h[-1]], dim=1))


class AcLSTM(nn.Module):
    """BiLSTM acoustic decoder: HG features → mel spectrogram per frame.

    Architecture: BiLSTM(d_in, d_hid) → per-frame projection →
                  Linear(2*d_hid, d_hid) → ReLU → Dropout → Linear(d_hid, d_out)
    """
    def __init__(self, d_in, d_hid, d_out, n_layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hid, n_layers, batch_first=True,
                            dropout=drop, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(d_hid * 2, d_hid), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d_hid, d_out))

    def forward(self, x, lens=None):
        if lens is not None:
            pk = nn.utils.rnn.pack_padded_sequence(
                x, lens, batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(pk)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        return self.proj(out)


# ============================================================
# TCN Models
# ============================================================

class CausalConv1d(nn.Module):
    """1D convolution with causal (left) padding for temporal data."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)

    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        return out


class TCNBlock(nn.Module):
    """Residual block: two causal dilated convolutions + skip connection.

    CausalConv → BN → ReLU → Dropout → CausalConv → BN → ReLU → Dropout + residual
    """
    def __init__(self, n_ch, kernel_size, dilation, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(n_ch, n_ch, kernel_size, dilation),
            nn.BatchNorm1d(n_ch), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(n_ch, n_ch, kernel_size, dilation),
            nn.BatchNorm1d(n_ch), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return F.relu(x + self.net(x))


class TCNEncoder(nn.Module):
    """Stack of TCN blocks with increasing dilation.

    Receptive field with dilations=[1,2,4,8], kernel=3:
        2 * sum(dilations) * (kernel-1) + 1 = 61 frames = 610ms at 100Hz
    """
    def __init__(self, d_in, n_filters=64, kernel_size=3,
                 dilations=(1, 2, 4, 8), dropout=0.3):
        super().__init__()
        self.input_proj = nn.Conv1d(d_in, n_filters, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(n_filters, kernel_size, d, dropout) for d in dilations])

    def forward(self, x):
        # x: (B, T, C) → (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)  # back to (B, T, n_filters)


class TCNClassifier(nn.Module):
    """TCN encoder → masked mean pooling → linear head → word classes."""
    def __init__(self, d_in=128, n_filters=64, n_cls=6,
                 kernel_size=3, dilations=(1, 2, 4, 8), dropout=0.3):
        super().__init__()
        self.encoder = TCNEncoder(d_in, n_filters, kernel_size,
                                  dilations, dropout)
        self.head = nn.Sequential(
            nn.Linear(n_filters, n_filters), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_filters, n_cls))

    def forward(self, x, lens):
        enc = self.encoder(x)
        B, T, D = enc.shape
        lens_t = torch.as_tensor(lens, device=x.device).float()
        mask = (torch.arange(T, device=x.device).unsqueeze(0)
                < lens_t.unsqueeze(1)).unsqueeze(-1).float()
        pooled = (enc * mask).sum(dim=1) / lens_t.unsqueeze(-1)
        return self.head(pooled)


class TCNDecoder(nn.Module):
    """TCN encoder → per-frame linear projection → mel bins."""
    def __init__(self, d_in=128, n_filters=64, d_out=40,
                 kernel_size=3, dilations=(1, 2, 4, 8), dropout=0.3):
        super().__init__()
        self.encoder = TCNEncoder(d_in, n_filters, kernel_size,
                                  dilations, dropout)
        self.proj = nn.Sequential(
            nn.Linear(n_filters, n_filters), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_filters, d_out))

    def forward(self, x, lens=None):
        return self.proj(self.encoder(x))


# ============================================================
# Transformer Models
# ============================================================

class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TFClassifier(nn.Module):
    """Transformer encoder → masked mean pool → word classes.

    4 layers, 4 heads, d_ff=256 (conservative for small datasets).
    """
    def __init__(self, d_in=128, d_model=128, n_heads=4, n_layers=4,
                 d_ff=256, n_cls=6, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pe = SinusoidalPE(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='relu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_cls))

    def forward(self, x, lens):
        B, T, _ = x.shape
        lens_t = torch.as_tensor(lens, device=x.device)
        pad_mask = (torch.arange(T, device=x.device).unsqueeze(0)
                    >= lens_t.unsqueeze(1))
        h = self.pe(self.input_proj(x))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / lens_t.float().unsqueeze(-1)
        return self.head(pooled)


class TFDecoder(nn.Module):
    """Transformer encoder → per-frame projection → mel bins."""
    def __init__(self, d_in=128, d_model=128, n_heads=4, n_layers=4,
                 d_ff=256, d_out=40, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pe = SinusoidalPE(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='relu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_out))

    def forward(self, x, lens=None):
        B, T, _ = x.shape
        h = self.pe(self.input_proj(x))
        if lens is not None:
            lens_t = torch.as_tensor(lens, device=x.device)
            pad_mask = (torch.arange(T, device=x.device).unsqueeze(0)
                        >= lens_t.unsqueeze(1))
            h = self.encoder(h, src_key_padding_mask=pad_mask)
        else:
            h = self.encoder(h)
        return self.proj(h)


# ============================================================
# Model Registry
# ============================================================

CLASSIFIER_REGISTRY = {
    'BiLSTM': lambda n_cls, d_in=128: WordLSTM(d_in, 128, n_cls),
    'TCN': lambda n_cls, d_in=128: TCNClassifier(d_in, 64, n_cls),
    'Transformer': lambda n_cls, d_in=128: TFClassifier(d_in, 128, 4, 4, 256, n_cls),
}

DECODER_REGISTRY = {
    'BiLSTM': lambda d_out=40, d_in=128: AcLSTM(d_in, 150, d_out),
    'TCN': lambda d_out=40, d_in=128: TCNDecoder(d_in, 64, d_out),
    'Transformer': lambda d_out=40, d_in=128: TFDecoder(d_in, 128, 4, 4, 256, d_out),
}


def get_classifier(arch, n_cls, d_in=128):
    """Create a classifier by architecture name."""
    if arch not in CLASSIFIER_REGISTRY:
        raise ValueError(f'Unknown arch: {arch}. Options: {list(CLASSIFIER_REGISTRY)}')
    return CLASSIFIER_REGISTRY[arch](n_cls, d_in)


def get_decoder(arch, d_out=40, d_in=128):
    """Create a decoder by architecture name."""
    if arch not in DECODER_REGISTRY:
        raise ValueError(f'Unknown arch: {arch}. Options: {list(DECODER_REGISTRY)}')
    return DECODER_REGISTRY[arch](d_out, d_in)


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
