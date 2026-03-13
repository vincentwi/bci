#!/usr/bin/env python3
"""
Lead 3: SSM/Mamba + Advanced Architectures — Shared Model Components.

Classes:
    BiMambaBlock, BiMambaBlockSeparate, BiMambaDecoder
    S4Block, S4Decoder
    ResBlock, ResBlockEncoder, ResBlockGRUDecoder
    NeuralMAE, PatchCTCDecoder

References:
    [1] arxiv 2412.17227 — Linderman BiMamba (Caduceus weight-tying)
    [2] arxiv 2403.05583 — MONA LISA S4/ResBlock
    [3] arxiv 2511.21740 — BIT MAE pretraining
    [4] tbenst/silent_speech — S4, Hyena, ResBlock implementations
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Day-Specific Input Layer (imported from train_beyond_paper.py at runtime)
# We re-export it here so all lead3 scripts can import from one place.
# ═══════════════════════════════════════════════════════════════════════

def _import_day_specific():
    """Lazy import to avoid circular deps."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from train_beyond_paper import DaySpecificInputLayer
    return DaySpecificInputLayer


# ═══════════════════════════════════════════════════════════════════════
# Frame Stacking (shared utility)
# ═══════════════════════════════════════════════════════════════════════

def stack_frames(x, kernel_size, stride):
    """Stack consecutive frames in a differentiable way.

    Args:
        x: (B, T, C) tensor
        kernel_size: number of frames to stack
        stride: step between stacked windows

    Returns:
        (B, T_new, C * kernel_size) where T_new = (T - kernel_size) // stride + 1
    """
    B, T, C = x.shape
    T_new = max(1, (T - kernel_size) // stride + 1)
    indices = torch.arange(T_new, device=x.device) * stride
    stacked = []
    for k in range(kernel_size):
        idx = (indices + k).clamp(max=T - 1)
        stacked.append(x[:, idx, :])
    return torch.cat(stacked, dim=-1)  # (B, T_new, C * kernel_size)


# ═══════════════════════════════════════════════════════════════════════
# BiMamba Components (Source: arxiv 2412.17227, kuleshov-group/caduceus)
# ═══════════════════════════════════════════════════════════════════════

class BiMambaBlock(nn.Module):
    """Weight-tied bidirectional Mamba block (Caduceus pattern).

    Forward: process left→right with Mamba
    Reverse: flip input, process with SAME Mamba, flip output back
    Combine: element-wise addition
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.4):
        super().__init__()
        from mamba_ssm.modules.mamba_simple import Mamba
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """(B, L, D) → (B, L, D)"""
        fwd = self.mamba(x)
        rev = self.mamba(torch.flip(x, dims=[1]))
        rev = torch.flip(rev, dims=[1])
        return self.dropout(self.norm(fwd + rev))


class BiMambaBlockSeparate(nn.Module):
    """Bidirectional Mamba with separate forward/backward modules.

    More parameters (2x) but potentially better bidirectional modeling.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.4):
        super().__init__()
        from mamba_ssm.modules.mamba_simple import Mamba
        self.mamba_fwd = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """(B, L, D) → (B, L, D)"""
        fwd = self.mamba_fwd(x)
        rev = self.mamba_bwd(torch.flip(x, dims=[1]))
        rev = torch.flip(rev, dims=[1])
        return self.dropout(self.norm(fwd + rev))


class BiMambaDecoder(nn.Module):
    """Full BiMamba decoder — drop-in GRU replacement.

    Architecture:
        Raw 256D → DaySpecific(256→day_hidden, softsign) per-frame
        → Stack kernel_size frames, stride 4 → (kernel_size * day_hidden)D
        → Linear(stacked_dim → d_model) + LayerNorm
        → N-layer BiMamba stack (each with residual)
        → Post-backbone: LayerNorm + Dropout + Linear + GELU
        → Linear(d_model → n_classes) → CTC loss
    """
    def __init__(self, n_features_per_frame=256, n_classes=41,
                 d_model=512, n_layers=5, d_state=16, d_conv=4, expand=2,
                 dropout=0.4, n_sessions=24, kernel_size=14, stride=4,
                 day_hidden=256, weight_tie=True, post_backbone_norm=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.post_backbone_norm = post_backbone_norm

        DaySpecificInputLayer = _import_day_specific()
        self.day_input = DaySpecificInputLayer(
            n_features_per_frame, day_hidden, n_sessions, dropout=dropout
        )

        stacked_dim = kernel_size * day_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(stacked_dim, d_model),
            nn.LayerNorm(d_model),
        )

        BlockClass = BiMambaBlock if weight_tie else BiMambaBlockSeparate
        self.layers = nn.ModuleList([
            BlockClass(d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        if post_backbone_norm:
            self.post_backbone = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
        else:
            self.post_backbone = nn.Identity()

        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def set_train_sessions(self, sids):
        self.day_input.set_train_sessions(sids)

    def forward(self, x, session_ids):
        """x: (B, T_raw, input_dim) raw features."""
        # Day-specific per frame
        x = self.day_input(x, session_ids)  # (B, T_raw, day_hidden)

        # Stack frames
        x = stack_frames(x, self.kernel_size, self.stride)

        # Project to d_model
        x = self.input_proj(x)

        # BiMamba layers with residual
        for layer in self.layers:
            x = layer(x) + x

        # Post-backbone processing
        x = self.post_backbone(x)

        return self.output_proj(x)

    def get_hidden(self, x, session_ids):
        """Get hidden representations before output projection (for supTCon)."""
        x = self.day_input(x, session_ids)
        x = stack_frames(x, self.kernel_size, self.stride)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x) + x
        x = self.post_backbone(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
# S4 Components (Source: tbenst/silent_speech, Gu et al. ICLR 2022)
# ═══════════════════════════════════════════════════════════════════════

class S4DKernel(nn.Module):
    """S4D (diagonal) kernel with HiPPO-LegS initialization.

    Implements the diagonal approximation of S4:
        A = diag(a_1, ..., a_N)  (complex diagonal)
        B, C are complex vectors
        Kernel K = C @ exp(A * dt) @ B computed via FFT

    Reference: Gu et al. "On the Parameterization and Initialization of
    Diagonal State Space Models" (2022)
    """
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # HiPPO-LegS initialization for diagonal A
        # A_n = -1/2 + n*i (simplified diagonal HiPPO)
        A_real = -0.5 * torch.ones(d_model, d_state)
        A_imag = torch.arange(d_state).float().unsqueeze(0).expand(d_model, -1)
        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)

        # B initialization (complex)
        self.B_real = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.B_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.5)

        # C initialization (complex)
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.5)

        # Discretization step size (log-parameterized)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # D skip connection
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, L):
        """Compute S4D convolution kernel of length L.

        Returns: (d_model, L) real-valued kernel
        """
        dt = self.log_dt.exp()  # (d_model,)

        # Complex A, B, C
        A = torch.complex(self.A_real, self.A_imag)  # (d_model, d_state)
        B = torch.complex(self.B_real, self.B_imag)
        C = torch.complex(self.C_real, self.C_imag)

        # Discretize: A_bar = exp(A * dt)
        dtA = A * dt.unsqueeze(-1)  # (d_model, d_state)

        # Compute kernel via Vandermonde-like product
        # K[l] = sum_n C_n * A_bar_n^l * B_n * dt
        A_bar = torch.exp(dtA)  # (d_model, d_state)

        # Powers: A_bar^0, A_bar^1, ..., A_bar^(L-1)
        powers = torch.arange(L, device=A.device).float()  # (L,)
        # A_bar^l = exp(dtA * l), shape: (d_model, d_state, L)
        vandermonde = torch.exp(dtA.unsqueeze(-1) * powers.unsqueeze(0).unsqueeze(0))

        # K = real(C * B * dt * vandermonde summed over state dim)
        CB = (C * B * dt.unsqueeze(-1))  # (d_model, d_state)
        K = torch.einsum('dn,dnl->dl', CB, vandermonde).real  # (d_model, L)

        return K


class S4Block(nn.Module):
    """S4 block with bidirectional option.

    Pre-norm → S4 convolution → dropout → residual connection.
    Bidirectional: forward kernel + reverse kernel combined.
    """
    def __init__(self, d_model, d_state=64, bidirectional=True, dropout=0.4):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model)
        self.s4_fwd = S4DKernel(d_model, d_state)
        if bidirectional:
            self.s4_bwd = S4DKernel(d_model, d_state)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # Output projection for mixing after S4
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """(B, L, D) → (B, L, D)"""
        residual = x
        x = self.norm(x)

        B, L, D = x.shape
        # S4 convolution via FFT
        # x: (B, L, D) → (B, D, L) for conv
        x_t = x.transpose(1, 2)  # (B, D, L)

        # Compute kernel and convolve via FFT
        K_fwd = self.s4_fwd(L)  # (D, L)
        y_fwd = self._fft_conv(x_t, K_fwd)

        if self.bidirectional:
            K_bwd = self.s4_bwd(L)
            x_rev = torch.flip(x_t, dims=[2])
            y_bwd = torch.flip(self._fft_conv(x_rev, K_bwd), dims=[2])
            y = y_fwd + y_bwd
        else:
            y = y_fwd

        # Add skip connection (D parameter from S4)
        y = y + x_t * self.s4_fwd.D.unsqueeze(0).unsqueeze(-1)

        y = y.transpose(1, 2)  # (B, L, D)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.out_proj(y)

        return y + residual

    def _fft_conv(self, x, K):
        """FFT-based convolution: x (B, D, L), K (D, L) → (B, D, L)"""
        L = x.shape[2]
        # Pad for linear (non-circular) convolution
        fft_len = 2 * L
        X = torch.fft.rfft(x, n=fft_len, dim=2)
        K_f = torch.fft.rfft(K, n=fft_len, dim=1).unsqueeze(0)
        Y = X * K_f
        y = torch.fft.irfft(Y, n=fft_len, dim=2)[..., :L]
        return y


class S4Decoder(nn.Module):
    """S4-based decoder for BCI phoneme decoding.

    Architecture:
        Raw 256D → DaySpecific → Stack → Linear → LayerNorm
        → N × S4Block (bidirectional, d_state=64)
        → Post-backbone: LayerNorm + Dropout + Linear + GELU
        → Linear(d_model → n_classes) → CTC
    """
    def __init__(self, n_features_per_frame=256, n_classes=41,
                 d_model=512, n_layers=6, d_state=64,
                 dropout=0.4, n_sessions=24, kernel_size=14, stride=4,
                 day_hidden=256, bidirectional=True,
                 downsample_layers=None, downsample_factor=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.downsample_layers = set(downsample_layers or [])
        self.downsample_factor = downsample_factor

        DaySpecificInputLayer = _import_day_specific()
        self.day_input = DaySpecificInputLayer(
            n_features_per_frame, day_hidden, n_sessions, dropout=dropout
        )

        stacked_dim = kernel_size * day_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(stacked_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.layers = nn.ModuleList()
        self.downsamplers = nn.ModuleDict()
        for i in range(n_layers):
            self.layers.append(
                S4Block(d_model, d_state=d_state,
                        bidirectional=bidirectional, dropout=dropout)
            )
            if i in self.downsample_layers:
                # Learned linear downsampling
                self.downsamplers[str(i)] = nn.Sequential(
                    nn.Linear(d_model * downsample_factor, d_model),
                    nn.LayerNorm(d_model),
                )

        self.post_backbone = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def set_train_sessions(self, sids):
        self.day_input.set_train_sessions(sids)

    def forward(self, x, session_ids):
        """x: (B, T_raw, input_dim)."""
        x = self.day_input(x, session_ids)
        x = stack_frames(x, self.kernel_size, self.stride)
        x = self.input_proj(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.downsample_layers and str(i) in self.downsamplers:
                B, T, D = x.shape
                f = self.downsample_factor
                T_new = T // f
                if T_new > 0:
                    x = x[:, :T_new * f, :].reshape(B, T_new, D * f)
                    x = self.downsamplers[str(i)](x)

        x = self.post_backbone(x)
        return self.output_proj(x)


# ═══════════════════════════════════════════════════════════════════════
# ResBlock Components (Source: tbenst/silent_speech, arxiv 2403.05583)
# ═══════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Stride-2 conv downsampling + residual scaling beta=1/sqrt(2).

    Source: tbenst/silent_speech architecture.py
    Beta scaling from "Fixup Initialization" / moment control.
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=2):
        super().__init__()
        self.beta = 1.0 / math.sqrt(2)
        padding = (kernel - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """(B, C_in, T) → (B, C_out, T//stride)"""
        res = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(self.beta * x + self.beta * res)


class ResBlockEncoder(nn.Module):
    """3-block ResBlock encoder → 8x temporal downsampling.

    Raw (B, T, C) → permute(B, C, T) → 3x ResBlock(stride=2) → permute(B, T, C_out)
    Output: (B, T//8, d_model)
    """
    def __init__(self, n_features=256, d_model=512):
        super().__init__()
        self.blocks = nn.Sequential(
            ResBlock(n_features, 128, stride=2),   # T → T/2
            ResBlock(128, 256, stride=2),           # T/2 → T/4
            ResBlock(256, d_model, stride=2),       # T/4 → T/8
        )

    def forward(self, x):
        """(B, T, C) → (B, T//8, d_model)"""
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.blocks(x)     # (B, d_model, T//8)
        return x.transpose(1, 2)  # (B, T//8, d_model)


class ResBlockGRUDecoder(nn.Module):
    """Hybrid: ResBlock encoder → GRU sequence processor → CTC.

    Architecture:
        Raw 256D → DaySpecific(256→day_hidden) per-frame
        → ResBlockEncoder (8x downsample)
        → BiGRU(n_layers, d_model)
        → Post-RNN: LayerNorm + Dropout + Linear + GELU
        → Linear → CTC
    """
    def __init__(self, n_features_per_frame=256, n_classes=41,
                 d_model=512, n_gru_layers=3, dropout=0.4,
                 n_sessions=24, day_hidden=256,
                 post_backbone_norm=True):
        super().__init__()
        self.post_backbone_norm_flag = post_backbone_norm

        DaySpecificInputLayer = _import_day_specific()
        self.day_input = DaySpecificInputLayer(
            n_features_per_frame, day_hidden, n_sessions, dropout=dropout
        )

        self.resblock_encoder = ResBlockEncoder(day_hidden, d_model)

        self.rnn = nn.GRU(
            d_model, d_model, n_gru_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_gru_layers > 1 else 0,
        )

        rnn_out = d_model * 2  # bidirectional

        if post_backbone_norm:
            self.post_backbone = nn.Sequential(
                nn.LayerNorm(rnn_out),
                nn.Dropout(dropout),
                nn.Linear(rnn_out, d_model),
                nn.GELU(),
            )
            self.output_proj = nn.Linear(d_model, n_classes)
        else:
            self.post_backbone = nn.Identity()
            self.output_proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(rnn_out, n_classes),
            )

    def set_train_sessions(self, sids):
        self.day_input.set_train_sessions(sids)

    def forward(self, x, session_ids):
        """x: (B, T_raw, input_dim)."""
        x = self.day_input(x, session_ids)  # (B, T, day_hidden)
        x = self.resblock_encoder(x)         # (B, T//8, d_model)
        x, _ = self.rnn(x)                   # (B, T//8, 2*d_model)
        x = self.post_backbone(x)
        return self.output_proj(x)

    def get_output_lengths(self, input_lengths):
        """Compute output lengths after ResBlock 8x downsampling."""
        # Each ResBlock with stride=2 and kernel=3, padding=1:
        # T_out = ceil(T_in / 2) — but Conv1d with stride=2, pad=1, kernel=3:
        # T_out = floor((T_in + 2*1 - 3) / 2) + 1 = floor((T_in - 1) / 2) + 1
        lengths = input_lengths
        for _ in range(3):  # 3 ResBlocks
            lengths = (lengths - 1) // 2 + 1
        return lengths


class ResBlockMambaDecoder(nn.Module):
    """Hybrid: ResBlock encoder → BiMamba sequence processor → CTC."""
    def __init__(self, n_features_per_frame=256, n_classes=41,
                 d_model=512, n_layers=4, d_state=16, d_conv=4, expand=2,
                 dropout=0.4, n_sessions=24, day_hidden=256,
                 weight_tie=True, post_backbone_norm=True):
        super().__init__()

        DaySpecificInputLayer = _import_day_specific()
        self.day_input = DaySpecificInputLayer(
            n_features_per_frame, day_hidden, n_sessions, dropout=dropout
        )

        self.resblock_encoder = ResBlockEncoder(day_hidden, d_model)

        BlockClass = BiMambaBlock if weight_tie else BiMambaBlockSeparate
        self.layers = nn.ModuleList([
            BlockClass(d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        if post_backbone_norm:
            self.post_backbone = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
        else:
            self.post_backbone = nn.Identity()

        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def set_train_sessions(self, sids):
        self.day_input.set_train_sessions(sids)

    def forward(self, x, session_ids):
        """x: (B, T_raw, input_dim)."""
        x = self.day_input(x, session_ids)
        x = self.resblock_encoder(x)
        for layer in self.layers:
            x = layer(x) + x
        x = self.post_backbone(x)
        return self.output_proj(x)

    def get_output_lengths(self, input_lengths):
        lengths = input_lengths
        for _ in range(3):
            lengths = (lengths - 1) // 2 + 1
        return lengths


# ═══════════════════════════════════════════════════════════════════════
# MAE Components (Source: arxiv 2511.21740 — BIT)
# ═══════════════════════════════════════════════════════════════════════

class NeuralMAE(nn.Module):
    """Masked Autoencoder for neural signal pretraining.

    Self-supervised: mask random contiguous temporal spans, reconstruct original.
    Uses contiguous span masking (~50% ratio) aligned with neural dynamics.

    Architecture:
        Raw (B, T, C) → patchify(patch_size) → (B, T//patch_size, C*patch_size)
        → PatchEmbed: LayerNorm → Linear → LayerNorm + pos_embed
        → Mask ~50% of patches (contiguous spans)
        → Replace masked with learnable mask_token
        → Transformer encoder (n_layers, n_heads)
        → Decoder: Linear → GELU → Linear → reconstruct patches
        → MSE loss on masked patches only
    """
    def __init__(self, n_features=256, d_model=256, n_layers=4,
                 n_heads=8, mask_ratio=0.5, patch_size=5,
                 dropout=0.1, max_patches=1000):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.patch_dim = n_features * patch_size

        # Patch embedding
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder (lightweight: reconstruct patches)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.patch_dim),
        )

    def patchify(self, x):
        """(B, T, C) → (B, n_patches, C * patch_size)"""
        B, T, C = x.shape
        n_patches = T // self.patch_size
        T_trim = n_patches * self.patch_size
        x = x[:, :T_trim, :].reshape(B, n_patches, C * self.patch_size)
        return x

    def unpatchify(self, patches):
        """(B, n_patches, C * patch_size) → (B, T, C)"""
        B, N, PD = patches.shape
        return patches.reshape(B, N * self.patch_size, self.n_features)

    def _contiguous_mask(self, B, n_patches, device):
        """Generate contiguous span masks for a batch.

        Returns: (B, n_patches) bool tensor, True = masked
        """
        mask = torch.zeros(B, n_patches, dtype=torch.bool, device=device)
        n_mask = max(1, int(n_patches * self.mask_ratio))
        for b in range(B):
            max_start = max(0, n_patches - n_mask)
            start = torch.randint(0, max_start + 1, (1,)).item()
            mask[b, start:start + n_mask] = True
        return mask

    def forward(self, x):
        """Self-supervised forward: mask, encode, reconstruct.

        Args:
            x: (B, T, C) raw neural features

        Returns:
            loss: MSE reconstruction loss on masked patches only
            n_masked: number of masked patches (for logging)
        """
        # Patchify
        patches = self.patchify(x)  # (B, N, patch_dim)
        B, N, _ = patches.shape

        # Embed patches
        embedded = self.patch_proj(patches)  # (B, N, d_model)

        # Add positional embeddings
        embedded = embedded + self.pos_embed[:, :N, :]

        # Create contiguous span mask
        mask = self._contiguous_mask(B, N, x.device)  # (B, N)

        # Replace masked tokens
        mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        encoder_input = torch.where(
            mask.unsqueeze(-1).expand_as(embedded),
            mask_tokens,
            embedded
        )

        # Encode
        encoded = self.encoder(encoder_input)  # (B, N, d_model)

        # Decode
        reconstructed = self.decoder(encoded)  # (B, N, patch_dim)

        # MSE loss on masked patches only
        loss = F.mse_loss(
            reconstructed[mask],
            patches[mask],
            reduction='mean'
        )

        return loss, mask.sum().item()

    def encode(self, x):
        """Encode without masking — for downstream use.

        Args:
            x: (B, T, C) raw neural features

        Returns:
            (B, n_patches, d_model) encoded representations
        """
        patches = self.patchify(x)
        B, N, _ = patches.shape
        embedded = self.patch_proj(patches)
        embedded = embedded + self.pos_embed[:, :N, :]
        return self.encoder(embedded)


class PatchCTCDecoder(nn.Module):
    """Patch-based encoder for CTC decoding (fine-tuning after MAE).

    Can initialize encoder from pretrained NeuralMAE.

    Architecture:
        Raw 256D → DaySpecific → patchify → PatchEmbed
        → Transformer encoder → Linear(d_model → n_classes) → CTC
    """
    def __init__(self, n_features=256, n_classes=41, d_model=256,
                 n_layers=4, n_heads=8, patch_size=5,
                 dropout=0.1, n_sessions=24, day_hidden=256,
                 max_patches=1000, use_day_specific=True):
        super().__init__()
        self.patch_size = patch_size
        self.n_features = n_features
        self.use_day_specific = use_day_specific
        self.patch_dim = (day_hidden if use_day_specific else n_features) * patch_size

        if use_day_specific:
            DaySpecificInputLayer = _import_day_specific()
            self.day_input = DaySpecificInputLayer(
                n_features, day_hidden, n_sessions, dropout=0.4
            )
            self._enc_input_dim = day_hidden
        else:
            self.day_input = None
            self._enc_input_dim = n_features

        # Patch embedding (same structure as MAE)
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # CTC output head
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def set_train_sessions(self, sids):
        if self.day_input is not None:
            self.day_input.set_train_sessions(sids)

    def load_mae_weights(self, mae_model):
        """Load encoder weights from pretrained NeuralMAE.

        Only loads patch_proj, pos_embed, and encoder — NOT decoder.
        """
        # If day_specific changes the dimension, patch_proj may not match exactly
        # Load what matches
        try:
            self.patch_proj.load_state_dict(mae_model.patch_proj.state_dict())
            print("  Loaded MAE patch_proj weights")
        except RuntimeError as e:
            print(f"  Could not load patch_proj (dim mismatch): {e}")

        # Position embeddings
        mae_pos = mae_model.pos_embed.data
        if mae_pos.shape == self.pos_embed.shape:
            self.pos_embed.data.copy_(mae_pos)
            print("  Loaded MAE pos_embed weights")
        else:
            # Copy up to min length
            min_len = min(mae_pos.shape[1], self.pos_embed.shape[1])
            self.pos_embed.data[:, :min_len, :] = mae_pos[:, :min_len, :]
            print(f"  Partially loaded MAE pos_embed ({min_len} patches)")

        # Encoder
        try:
            self.encoder.load_state_dict(mae_model.encoder.state_dict())
            print("  Loaded MAE encoder weights")
        except RuntimeError as e:
            print(f"  Could not load encoder (structure mismatch): {e}")

    def forward(self, x, session_ids):
        """x: (B, T_raw, input_dim)."""
        if self.use_day_specific and self.day_input is not None:
            x = self.day_input(x, session_ids)  # (B, T, day_hidden)

        # Patchify
        B, T, C = x.shape
        n_patches = T // self.patch_size
        T_trim = n_patches * self.patch_size
        patches = x[:, :T_trim, :].reshape(B, n_patches, C * self.patch_size)

        # Embed
        embedded = self.patch_proj(patches)
        embedded = embedded + self.pos_embed[:, :n_patches, :]

        # Encode
        encoded = self.encoder(embedded)

        return self.output_proj(encoded)

    def get_output_lengths(self, input_lengths):
        """Compute output lengths after patchification."""
        return input_lengths // self.patch_size


# ═══════════════════════════════════════════════════════════════════════
# Auxiliary Losses
# ═══════════════════════════════════════════════════════════════════════

def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """Supervised contrastive loss (supTCon).

    Source: arxiv 2403.05583 (MONA LISA)
    Clusters embeddings with same phoneme label closer together.

    Args:
        embeddings: (N, D) hidden states at CTC timesteps
        labels: (N,) phoneme labels per timestep
        temperature: temperature scaling (0.1 from paper)

    Returns:
        scalar loss
    """
    if embeddings.shape[0] < 2:
        return torch.tensor(0.0, device=embeddings.device)

    embeddings = F.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.T / temperature

    # Numerical stability
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    exp_sim = torch.exp(sim)

    # Positive pair mask: same label, excluding self
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask_pos.fill_diagonal_(0)

    # All pairs mask (excluding self)
    mask_all = torch.ones_like(mask_pos)
    mask_all.fill_diagonal_(0)

    pos_sum = (exp_sim * mask_pos).sum(1)
    all_sum = (exp_sim * mask_all).sum(1)

    # Only compute loss for samples with at least one positive pair
    valid = mask_pos.sum(1) > 0
    if not valid.any():
        return torch.tensor(0.0, device=embeddings.device)

    loss = -torch.log(pos_sum[valid] / (all_sum[valid] + 1e-8))

    # Per-class normalization (MONA LISA innovation)
    valid_labels = labels[valid]
    unique_labels = valid_labels.unique()
    if len(unique_labels) > 0:
        class_losses = []
        for lbl in unique_labels:
            mask_lbl = valid_labels == lbl
            if mask_lbl.any():
                class_losses.append(loss[mask_lbl].mean())
        return torch.stack(class_losses).mean()

    return loss.mean()


def fastemit_regularization(log_probs, blank=40, lambda_fe=0.01):
    """FastEmit regularization — encourages earlier emission of non-blank tokens.

    Source: arxiv 2412.17227 (Linderman Lab)
    Penalizes high blank probability at each timestep.

    Args:
        log_probs: (B, T, V) log-softmax output
        blank: blank token index
        lambda_fe: regularization strength (0.01 from paper)

    Returns:
        scalar penalty
    """
    # Encourage non-blank emission by penalizing blank log-prob
    blank_log_probs = log_probs[:, :, blank]  # (B, T)
    # FastEmit: minimize -log(1 - p_blank) ≈ maximize non-blank prob
    penalty = -torch.log(1 - torch.exp(blank_log_probs) + 1e-8).mean()
    return lambda_fe * penalty


def speckled_mask(x, prob=0.3):
    """Speckled masking — randomly zero individual elements.

    Source: arxiv 2412.17227 (Linderman Lab)
    Unlike SpecAugment (blocks), this masks individual (t, c) entries.

    Args:
        x: (B, T, C) input features
        prob: probability of masking each element

    Returns:
        masked x (same shape)
    """
    if not x.requires_grad:
        # Inference mode, no masking
        return x
    mask = torch.bernoulli(torch.full_like(x, 1 - prob))
    return x * mask / (1 - prob)  # Scale to maintain expected value
