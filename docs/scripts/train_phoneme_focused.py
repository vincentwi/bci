#!/usr/bin/env python3
"""
Phoneme-focused training script — reproduce and improve on Willett et al. (Nature 2023).

Paper methodology (phoneme classification):
  - 128 electrodes from area 6v only (area 44 excluded)
  - 256 features/time step (128 ch × 2: threshold crossings + spike band power)
  - 5-layer GRU (unspecified if bidirectional)
  - 39 phoneme classes
  - Naive Bayes baseline: ~62% accuracy
  - RNN phoneme error rate: 19.7% (i.e., ~80.3% frame-level accuracy)
  - Rolling z-scoring
  - 20ms bins

Our data:
  - 256 channels (all arrays), 1280 features/bin (spikePow + tx1-tx4)
  - Apr 21: 640 trials, Apr 26: 800 trials (40 classes including DO_NOTHING)
  - Per-block z-scoring

IMPROVEMENT ROUNDS:
  Round 0: Paper reproduction — 128ch, 256 features, 5-layer GRU, paper aug
  Round 1: Full features — 1280 features, 5-layer GRU, paper aug
  Round 2: Architecture search — TCN, Transformer, deeper GRU
  Round 3: Training improvements — label smoothing, mixup, cosine warmup
  Round 4: Feature engineering — temporal derivatives, multi-scale bins

Usage:
    CUDA_VISIBLE_DEVICES=4,5 python train_phoneme_focused.py --round 0
    CUDA_VISIBLE_DEVICES=4,5 python train_phoneme_focused.py --round all
    CUDA_VISIBLE_DEVICES=4,5 python train_phoneme_focused.py --round 0,1,2
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    PROCESSED_DIR, MODELS_DIR, SEED,
    DL_EPOCHS, DL_LR, DL_WD, DL_BS, DL_PATIENCE,
    VAL_FRACTION, NUM_WORKERS, PREFETCH_FACTOR,
)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

RESULTS_DIR = MODELS_DIR / "phoneme_focused"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_phoneme_data(merged=True, channels="all"):
    """Load phoneme data with channel selection.

    Args:
        merged: If True, merge both sessions (1440 trials)
        channels: "all" (1280 feat), "6v" (first 128 ch → 256 feat), "spikepow" (256 ch)
    """
    d1 = np.load(PROCESSED_DIR / "tuning_t12.2022.04.21_phonemes.npz", allow_pickle=True)
    d2 = np.load(PROCESSED_DIR / "tuning_t12.2022.04.26_phonemes.npz", allow_pickle=True)

    if merged:
        X = np.concatenate([d1["X"], d2["X"]])
        y = np.concatenate([d1["y"], d2["y"]])
        # Create group_ids: session 0 blocks + session 1 blocks
        g1 = d1["block_ids"]
        g2 = d2["block_ids"] + g1.max() + 1  # Offset to avoid overlap
        group_ids = np.concatenate([g1, g2])
    else:
        X = d2["X"]  # Apr 26, larger session
        y = d2["y"]
        group_ids = d2["block_ids"]

    class_names = list(d1["class_names"])
    n_classes = int(d1["n_classes"])

    # Channel selection
    # Features are: spikePow(256) + tx1(256) + tx2(256) + tx3(256) + tx4(256) = 1280
    if channels == "6v":
        # Paper: 128 channels from area 6v (first 128 of each feature set)
        # Select first 128 of spikePow + first 128 of tx1 = 256 features
        sp_idx = list(range(0, 128))        # spikePow channels 0-127
        tx1_idx = list(range(256, 384))     # tx1 channels 0-127
        sel = sp_idx + tx1_idx
        X = X[:, :, sel]
        feat_desc = f"6v-only: 128ch × 2feat = {X.shape[2]}"
    elif channels == "spikepow":
        X = X[:, :, :256]  # Just spikePow
        feat_desc = f"spikePow only: {X.shape[2]} features"
    else:
        feat_desc = f"all features: {X.shape[2]}"

    print(f"  Data: X={X.shape}, {n_classes} classes, {len(np.unique(group_ids))} groups")
    print(f"  Features: {feat_desc}")
    print(f"  Merged: {merged}")

    return X, y, group_ids, class_names, n_classes


# ═══════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════

class PaperGRU(nn.Module):
    """5-layer GRU matching the paper's architecture."""
    def __init__(self, nc, nt, nk, hidden=512, n_layers=5, dr=0.3, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(nc, hidden, n_layers, batch_first=True,
                          dropout=dr if n_layers > 1 else 0, bidirectional=bidirectional)
        out_dim = hidden * 2 if bidirectional else hidden
        self.fc = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dr),
            nn.Linear(out_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, nk)
        )

    def forward(self, x):
        # x: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.fc(out.mean(dim=1))


class TCN(nn.Module):
    """Temporal Convolutional Network with dilated convolutions."""
    def __init__(self, nc, nt, nk, hidden=128, dr=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(nc, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dr),
            nn.Conv1d(hidden, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dr),
            nn.Conv1d(hidden, hidden, 7, padding=6, dilation=2),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dr),
            nn.Conv1d(hidden, 64, 7, padding=12, dilation=4),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dr),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, nk)

    def forward(self, x):
        return self.fc(self.pool(self.net(x)).squeeze(-1))


class WiderTCN(nn.Module):
    """Wider/deeper TCN with residual connections."""
    def __init__(self, nc, nt, nk, hidden=256, dr=0.3):
        super().__init__()
        self.proj = nn.Conv1d(nc, hidden, 1)
        self.blocks = nn.ModuleList([
            self._make_block(hidden, hidden, 7, dilation=2**i, dr=dr)
            for i in range(5)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(hidden, nk)
        )

    def _make_block(self, in_ch, out_ch, kernel, dilation, dr):
        padding = (kernel - 1) * dilation // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dr),
            nn.Conv1d(out_ch, out_ch, kernel, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dr),
        )

    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)  # Residual
        return self.fc(self.pool(x).squeeze(-1))


class SpeechTransformer(nn.Module):
    """Transformer encoder with learnable CLS token."""
    def __init__(self, nc, nt, nk, d_model=128, nhead=8, num_layers=4, dr=0.3):
        super().__init__()
        self.proj = nn.Linear(nc, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, nt + 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=dr, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dr),
            nn.Linear(d_model, nk)
        )

    def forward(self, x):
        # x: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        x = self.proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb[:, :x.size(1)]
        x = self.encoder(x)
        return self.fc(x[:, 0])


class EEGNet(nn.Module):
    def __init__(self, nc, nt, nk, F1=16, D=2, F2=32, kl=32, dr=0.4):
        super().__init__()
        self.c1 = nn.Conv2d(1, F1, (1, kl), padding=(0, kl // 2), bias=False)
        self.b1 = nn.BatchNorm2d(F1)
        self.dw = nn.Conv2d(F1, F1 * D, (nc, 1), groups=F1, bias=False)
        self.b2 = nn.BatchNorm2d(F1 * D)
        self.p1 = nn.AvgPool2d((1, 4))
        self.d1 = nn.Dropout(dr)
        self.sd = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sp = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.b3 = nn.BatchNorm2d(F2)
        self.p2 = nn.AvgPool2d((1, 8))
        self.d2 = nn.Dropout(dr)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, nc, nt)
            dummy = self._features(dummy)
            self.flat_size = dummy.shape[1]
        self.fc = nn.Linear(self.flat_size, nk)

    def _features(self, x):
        x = self.d1(self.p1(F.elu(self.b2(self.dw(self.b1(self.c1(x)))))))
        x = self.d2(self.p2(F.elu(self.b3(self.sp(self.sd(x))))))
        return x.flatten(1)

    def forward(self, x):
        return self.fc(self._features(x.unsqueeze(1)))


# ═══════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════

def augment_paper(xb):
    """Willett et al. augmentation: white noise SD=1.0, constant offset SD=0.2."""
    xb = xb + torch.randn_like(xb) * 1.0
    offset = torch.randn(xb.shape[0], xb.shape[1], 1, device=xb.device) * 0.2
    return xb + offset


def augment_enhanced(xb, epoch, max_epochs):
    """Enhanced augmentation with curriculum (weaker early, stronger later)."""
    progress = epoch / max(max_epochs, 1)

    # Paper-style noise (always)
    xb = xb + torch.randn_like(xb) * 1.0
    offset = torch.randn(xb.shape[0], xb.shape[1], 1, device=xb.device) * 0.2
    xb = xb + offset

    # Time masking (increases with training)
    if progress > 0.2 and torch.rand(1).item() < 0.3:
        t = xb.shape[2]
        mask_len = int(t * 0.1 * min(progress, 0.5))
        if mask_len > 0:
            start = torch.randint(0, t - mask_len, (1,)).item()
            xb[:, :, start:start+mask_len] = 0

    # Channel dropout (mild)
    if torch.rand(1).item() < 0.15:
        n_drop = max(1, int(xb.shape[1] * 0.05))
        drop_idx = torch.randint(0, xb.shape[1], (n_drop,))
        xb[:, drop_idx, :] = 0

    return xb


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ═══════════════════════════════════════════════════════════════════════
# TRAINING ENGINE (shared across all rounds)
# ═══════════════════════════════════════════════════════════════════════

def _eval_batch(model, X_gpu, batch_size=512):
    """Batched GPU evaluation."""
    model.eval()
    all_logits = []
    with torch.no_grad(), autocast("cuda"):
        for i in range(0, len(X_gpu), batch_size):
            all_logits.append(model(X_gpu[i:i+batch_size]))
    return torch.cat(all_logits, dim=0)


def train_one_fold(X_train, y_train, X_test, y_test,
                   model_cls, model_kw, device,
                   epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
                   augment_fn=augment_paper, use_mixup=False,
                   label_smoothing=0.0, val_fraction=0.15):
    """Train one fold with val-based early stopping.

    Returns: (val_stopped_test_acc, oracle_test_acc, test_preds, test_probs, info)
    """
    n_gpus = torch.cuda.device_count()

    # Split training into train_sub + val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=SEED)
    sub_idx, val_idx = next(sss.split(X_train, y_train))

    X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    Xtr = torch.FloatTensor(X_sub.transpose(0, 2, 1))
    Xval = torch.FloatTensor(X_val.transpose(0, 2, 1))
    Xte = torch.FloatTensor(X_test.transpose(0, 2, 1))
    ytr = torch.LongTensor(y_sub)

    train_ds = TensorDataset(Xtr, ytr)
    eff_bs = bs * max(n_gpus, 1)
    use_workers = NUM_WORKERS if len(train_ds) > eff_bs else 0
    loader_kw = dict(batch_size=eff_bs, shuffle=True, pin_memory=True)
    if use_workers > 0:
        loader_kw.update(num_workers=use_workers, persistent_workers=True,
                         prefetch_factor=PREFETCH_FACTOR)
    train_loader = DataLoader(train_ds, **loader_kw)

    model = model_cls(**model_kw)
    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
    model = model.to(device)
    Xval_gpu = Xval.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(epochs // 3, 10), T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler("cuda")

    best_val_acc = 0
    best_val_state = None
    best_oracle_acc = 0
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # Augmentation
            if augment_fn is not None:
                if augment_fn == augment_enhanced:
                    xb = augment_fn(xb, epoch, epochs)
                else:
                    xb = augment_fn(xb)

            optimizer.zero_grad(set_to_none=True)

            if use_mixup and torch.rand(1).item() < 0.5:
                xb_mix, ya, yb_mix, lam = mixup_data(xb, yb)
                with autocast("cuda"):
                    logits = model(xb_mix)
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb_mix)
            else:
                with autocast("cuda"):
                    loss = criterion(model(xb), yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Val evaluation for early stopping
        val_logits = _eval_batch(model, Xval_gpu)
        val_preds = val_logits.argmax(1).cpu().numpy()
        val_acc = (val_preds == y_val).mean()

        # Track oracle (test peeking — for gap analysis only)
        Xte_gpu = Xte.to(device)
        te_logits = _eval_batch(model, Xte_gpu)
        te_preds = te_logits.argmax(1).cpu().numpy()
        te_acc = (te_preds == y_test).mean()
        del Xte_gpu

        if te_acc > best_oracle_acc:
            best_oracle_acc = te_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            m = model.module if hasattr(model, 'module') else model
            best_val_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    # Final test from val-best checkpoint
    if best_val_state is not None:
        m = model.module if hasattr(model, 'module') else model
        m.load_state_dict(best_val_state)
    model = model.to(device)
    Xte_gpu = Xte.to(device)
    test_logits = _eval_batch(model, Xte_gpu)
    test_preds = test_logits.argmax(1).cpu().numpy()
    test_probs = F.softmax(test_logits, dim=1).cpu().numpy()
    test_acc = (test_preds == y_test).mean()
    del Xte_gpu

    # Cleanup
    del model, Xval_gpu
    torch.cuda.empty_cache()

    return test_acc, best_oracle_acc, test_preds, test_probs, {
        "val_best": best_val_acc,
        "epochs_trained": epoch + 1,
    }


def run_cv(X, y, group_ids, model_cls, model_kw, device, n_classes,
           epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
           augment_fn=augment_paper, use_mixup=False,
           label_smoothing=0.0, val_fraction=0.15, name="model"):
    """Run leave-one-group-out CV with val-based early stopping."""
    unique_groups = np.unique(group_ids)
    n_folds = len(unique_groups)

    all_preds = np.full(len(y), -1, dtype=np.int64)
    all_probs = np.zeros((len(y), n_classes), dtype=np.float32)
    fold_accs = []
    fold_oracle = []

    for fold_i, g in enumerate(unique_groups):
        train_idx = np.where(group_ids != g)[0]
        test_idx = np.where(group_ids == g)[0]
        t0 = time.time()

        test_acc, oracle_acc, preds, probs, info = train_one_fold(
            X[train_idx], y[train_idx], X[test_idx], y[test_idx],
            model_cls, model_kw, device,
            epochs=epochs, lr=lr, wd=wd, bs=bs, patience=patience,
            augment_fn=augment_fn, use_mixup=use_mixup,
            label_smoothing=label_smoothing, val_fraction=val_fraction,
        )

        all_preds[test_idx] = preds
        all_probs[test_idx] = probs
        fold_accs.append(test_acc)
        fold_oracle.append(oracle_acc)

        gap = oracle_acc - test_acc
        print(f"    Fold {fold_i+1}/{n_folds}: "
              f"val-stopped={test_acc:.3f}, oracle={oracle_acc:.3f}, "
              f"gap={gap:+.3f} ({time.time()-t0:.0f}s)")

    overall = np.mean(fold_accs)
    overall_oracle = np.mean(fold_oracle)
    gap = overall_oracle - overall
    print(f"  {name}: val-stopped={overall:.3f}, oracle={overall_oracle:.3f}, gap={gap:+.3f}")
    print(f"    folds: [{' '.join(f'{a:.3f}' for a in fold_accs)}]")

    return {
        "name": name,
        "val_stopped": float(overall),
        "oracle": float(overall_oracle),
        "gap": float(gap),
        "fold_accs": [float(a) for a in fold_accs],
        "fold_oracle": [float(a) for a in fold_oracle],
        "preds": all_preds,
        "probs": all_probs,
    }


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT ROUNDS
# ═══════════════════════════════════════════════════════════════════════

def round0_paper_reproduction(device):
    """Round 0: Reproduce paper's phoneme results as closely as possible.

    Key settings:
      - 128 channels from area 6v (first 128 of spikePow + first 128 of tx1)
      - 5-layer bidirectional GRU (paper architecture)
      - Paper augmentation (noise SD=1.0, offset SD=0.2)
      - Merged sessions (1440 trials, 40 classes)
      - Leave-one-block-out CV
    """
    print("\n" + "=" * 70)
    print("ROUND 0: PAPER REPRODUCTION")
    print("  128ch (area 6v), 5-layer GRU, paper augmentation")
    print("=" * 70)

    X, y, group_ids, class_names, n_classes = load_phoneme_data(
        merged=True, channels="6v")

    nc, nt = X.shape[2], X.shape[1]
    model_kw = dict(nc=nc, nt=nt, nk=n_classes, hidden=512, n_layers=5, bidirectional=True)

    result = run_cv(X, y, group_ids, PaperGRU, model_kw, device, n_classes,
                    epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
                    augment_fn=augment_paper, name="PaperGRU-6v-128ch")

    print(f"\n  Paper NB baseline: 62.0%")
    print(f"  Our PaperGRU (6v): {result['val_stopped']*100:.1f}%")

    return {"round0_paper_gru_6v": result}


def round1_full_features(device):
    """Round 1: Same architecture but with all our features (1280/bin).

    Tests whether our additional features (all 256ch, tx1-tx4) help.
    """
    print("\n" + "=" * 70)
    print("ROUND 1: FULL FEATURES (1280/bin)")
    print("  All 256 channels + tx1-tx4, 5-layer GRU, paper augmentation")
    print("=" * 70)

    X, y, group_ids, class_names, n_classes = load_phoneme_data(
        merged=True, channels="all")

    nc, nt = X.shape[2], X.shape[1]
    results = {}

    # a) Same 5-layer GRU with full features
    model_kw = dict(nc=nc, nt=nt, nk=n_classes, hidden=512, n_layers=5, bidirectional=True)
    r = run_cv(X, y, group_ids, PaperGRU, model_kw, device, n_classes,
               epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
               augment_fn=augment_paper, name="PaperGRU-all-1280feat")
    results["gru_all"] = r

    # b) spikePow only (256 features) for comparison
    X_sp, y_sp, g_sp, _, _ = load_phoneme_data(merged=True, channels="spikepow")
    nc_sp = X_sp.shape[2]
    model_kw_sp = dict(nc=nc_sp, nt=nt, nk=n_classes, hidden=512, n_layers=5, bidirectional=True)
    r = run_cv(X_sp, y_sp, g_sp, PaperGRU, model_kw_sp, device, n_classes,
               epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
               augment_fn=augment_paper, name="PaperGRU-spikePow-256feat")
    results["gru_spikepow"] = r

    return results


def round2_architecture_search(device, only_models=None):
    """Round 2: Try different architectures with full features.

    Compare TCN, Transformer, EEGNet against the GRU baseline.
    Args:
        only_models: If set, list of model names to run (skip others).
    """
    print("\n" + "=" * 70)
    print("ROUND 2: ARCHITECTURE SEARCH")
    print("  Full features, paper augmentation, multiple architectures")
    print("=" * 70)

    X, y, group_ids, class_names, n_classes = load_phoneme_data(
        merged=True, channels="all")

    nc, nt = X.shape[2], X.shape[1]
    results = {}

    models = {
        "TCN-128": (TCN, dict(nc=nc, nt=nt, nk=n_classes, hidden=128)),
        "TCN-256": (WiderTCN, dict(nc=nc, nt=nt, nk=n_classes, hidden=256)),
        "Transformer-4L": (SpeechTransformer, dict(nc=nc, nt=nt, nk=n_classes,
                                                     d_model=128, nhead=8, num_layers=4)),
        "Transformer-6L": (SpeechTransformer, dict(nc=nc, nt=nt, nk=n_classes,
                                                     d_model=192, nhead=8, num_layers=6)),
        "EEGNet": (EEGNet, dict(nc=nc, nt=nt, nk=n_classes, F1=16, D=2, F2=32, kl=32, dr=0.4)),
    }

    for name, (cls, kw) in models.items():
        if only_models is not None and name not in only_models:
            print(f"\n  {name}: SKIPPED (not in --models)")
            continue
        tmp = cls(**kw)
        n_params = sum(p.numel() for p in tmp.parameters())
        print(f"\n  {name}: {n_params:,} params")
        del tmp

        r = run_cv(X, y, group_ids, cls, kw, device, n_classes,
                   epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
                   augment_fn=augment_paper, name=name)
        results[name] = r

    return results


def round3_training_improvements(device):
    """Round 3: Training recipe improvements on the best architecture.

    Tests: label smoothing, mixup, enhanced augmentation.
    Uses the best architecture from round 2 (or TCN as default).
    """
    print("\n" + "=" * 70)
    print("ROUND 3: TRAINING IMPROVEMENTS")
    print("  Label smoothing, mixup, enhanced augmentation")
    print("=" * 70)

    X, y, group_ids, class_names, n_classes = load_phoneme_data(
        merged=True, channels="all")

    nc, nt = X.shape[2], X.shape[1]
    results = {}

    # Use WiderTCN as base (good speed/accuracy tradeoff)
    base_kw = dict(nc=nc, nt=nt, nk=n_classes, hidden=256)

    configs = [
        ("TCN256-paperaug", dict(augment_fn=augment_paper, label_smoothing=0.0, use_mixup=False)),
        ("TCN256-labelsmooth0.1", dict(augment_fn=augment_paper, label_smoothing=0.1, use_mixup=False)),
        ("TCN256-mixup", dict(augment_fn=augment_paper, label_smoothing=0.0, use_mixup=True)),
        ("TCN256-enhanced-aug", dict(augment_fn=augment_enhanced, label_smoothing=0.0, use_mixup=False)),
        ("TCN256-all-tricks", dict(augment_fn=augment_enhanced, label_smoothing=0.1, use_mixup=True)),
    ]

    for name, extra_kw in configs:
        print(f"\n  Config: {name}")
        r = run_cv(X, y, group_ids, WiderTCN, base_kw, device, n_classes,
                   epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
                   name=name, **extra_kw)
        results[name] = r

    return results


def round4_feature_engineering(device):
    """Round 4: Feature engineering — temporal derivatives and multi-scale.

    Adds first-order temporal derivatives as extra channels.
    """
    print("\n" + "=" * 70)
    print("ROUND 4: FEATURE ENGINEERING")
    print("  Temporal derivatives, go-period focus")
    print("=" * 70)

    X, y, group_ids, class_names, n_classes = load_phoneme_data(
        merged=True, channels="all")

    # a) Add temporal derivatives as extra channels
    # X shape: (trials, time, features)
    X_deriv = np.diff(X, axis=1)  # (trials, time-1, features)
    X_deriv = np.pad(X_deriv, ((0,0), (0,1), (0,0)), mode='edge')  # Pad to match
    X_aug = np.concatenate([X, X_deriv], axis=2)  # Double the features

    nc_aug = X_aug.shape[2]
    nt = X_aug.shape[1]

    results = {}

    print(f"\n  With derivatives: {X_aug.shape} ({nc_aug} features)")
    model_kw = dict(nc=nc_aug, nt=nt, nk=n_classes, hidden=256)
    r = run_cv(X_aug, y, group_ids, WiderTCN, model_kw, device, n_classes,
               epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
               augment_fn=augment_paper, name="TCN256-with-derivatives")
    results["with_derivatives"] = r

    # b) Go-period focus: use only bins 10-60 (go period)
    X_go = X[:, 10:60, :]  # 50 bins = 1000ms go period
    nt_go = X_go.shape[1]
    nc_go = X_go.shape[2]
    print(f"\n  Go-period only: {X_go.shape}")
    model_kw = dict(nc=nc_go, nt=nt_go, nk=n_classes, hidden=256)
    r = run_cv(X_go, y, group_ids, WiderTCN, model_kw, device, n_classes,
               epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
               augment_fn=augment_paper, name="TCN256-go-period-only")
    results["go_period_only"] = r

    # c) Wider window: full trial (all 85 bins) — baseline
    model_kw = dict(nc=X.shape[2], nt=X.shape[1], nk=n_classes, hidden=256)
    r = run_cv(X, y, group_ids, WiderTCN, model_kw, device, n_classes,
               epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
               augment_fn=augment_paper, name="TCN256-full-window-baseline")
    results["full_window"] = r

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="all",
                        help="Which rounds to run: 0,1,2,3,4 or 'all'")
    parser.add_argument("--models", default=None,
                        help="For round 2: comma-separated model names to run (e.g. 'EEGNet')")
    parser.add_argument("--merge", action="store_true",
                        help="Merge new results with existing JSON instead of overwriting")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"Device: {device}, GPUs: {n_gpus}")

    if args.round == "all":
        rounds = [0, 1, 2, 3, 4]
    else:
        rounds = [int(r.strip()) for r in args.round.split(",")]

    only_models = None
    if args.models:
        only_models = [m.strip() for m in args.models.split(",")]

    # Load existing results if merging
    results_path = RESULTS_DIR / "phoneme_focused_results.json"
    if args.merge and results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"  Loaded existing results: {list(all_results.keys())}")
    else:
        all_results = {}

    t_start = time.time()

    if 0 in rounds:
        r = round0_paper_reproduction(device)
        all_results.update(r)

    if 1 in rounds:
        r = round1_full_features(device)
        all_results["round1"] = {k: {kk: vv for kk, vv in v.items() if kk != "preds" and kk != "probs"}
                                  for k, v in r.items()}

    if 2 in rounds:
        r = round2_architecture_search(device, only_models=only_models)
        r2_clean = {k: {kk: vv for kk, vv in v.items() if kk != "preds" and kk != "probs"}
                    for k, v in r.items()}
        if args.merge and "round2" in all_results:
            all_results["round2"].update(r2_clean)
        else:
            all_results["round2"] = r2_clean

    if 3 in rounds:
        r = round3_training_improvements(device)
        all_results["round3"] = {k: {kk: vv for kk, vv in v.items() if kk != "preds" and kk != "probs"}
                                  for k, v in r.items()}

    if 4 in rounds:
        r = round4_feature_engineering(device)
        all_results["round4"] = {k: {kk: vv for kk, vv in v.items() if kk != "preds" and kk != "probs"}
                                  for k, v in r.items()}

    # Save results
    def to_json(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=to_json))
    with open(RESULTS_DIR / "phoneme_focused_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Summary
    elapsed = (time.time() - t_start) / 60
    print("\n" + "=" * 70)
    print("PHONEME-FOCUSED RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Paper NB baseline: 62.0% (39 classes)")
    print(f"  Paper RNN PER: 19.7% (i.e., ~80.3% frame accuracy)")
    print(f"  Our task: 40 classes (incl. DO_NOTHING), leave-one-block-out CV")
    print()

    def print_result(key, data):
        if isinstance(data, dict) and "val_stopped" in data:
            print(f"  {data.get('name', key):35s}: {data['val_stopped']*100:.1f}% "
                  f"(oracle={data['oracle']*100:.1f}%, gap={data['gap']*100:+.1f}%)")
        elif isinstance(data, dict):
            for k, v in data.items():
                print_result(k, v)

    for key, data in all_results.items():
        print_result(key, data)

    print(f"\n  Total time: {elapsed:.1f} min")
    print(f"  Results: {RESULTS_DIR / 'phoneme_focused_results.json'}")


if __name__ == "__main__":
    main()
