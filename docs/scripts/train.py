#!/usr/bin/env python3
"""
Train models to predict speech intent from brain signals — ALL ON GPU.
Uses Willett et al. (Nature 2023) preprocessed data.

v2: Fixed test-set peeking. Proper train/val/test splits.
    Reports BOTH val-stopped and oracle accuracies.

Classical ML (PCA + linear/MLP classifiers) → PyTorch on GPU
Deep learning (EEGNet, TCN, Transformer, GRU) → Multi-GPU with AMP

Usage:
    python train.py                          # full pipeline
    python train.py --dataset diagnostic     # default (8 words)
    python train.py --dataset tuning         # phonemes/words
    python train.py --epochs 200
    python train.py --sanity                 # label-shuffle sanity check
    python train.py --augmentation paper     # Willett et al. augmentation
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from pathlib import Path

# Set visible GPUs BEFORE importing torch (env var overrides config)
from config import VISIBLE_GPUS
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPUS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from config import (
    PROCESSED_DIR, FIGURES_DIR, MODELS_DIR,
    N_PCA, SEED,
    EEGNET_KW, TCN_KW, DL_EPOCHS, DL_LR, DL_WD, DL_BS, DL_PATIENCE,
    VAL_FRACTION, NUM_WORKERS, PREFETCH_FACTOR,
)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ─────────────────────────────────────────────────────────────────────
# Device setup
# ─────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"  CUDA devices visible: {n}")
        for i in range(n):
            print(f"    [{i}] {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.0f} GB)")
        return torch.device("cuda:0")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_data(dataset="diagnostic"):
    if dataset == "diagnostic":
        path = PROCESSED_DIR / "diagnostic_processed.npz"
    else:
        path = PROCESSED_DIR / f"{dataset}.npz"
        if not path.exists():
            candidates = sorted(PROCESSED_DIR.glob(f"*{dataset}*.npz"))
            if not candidates:
                candidates = sorted(PROCESSED_DIR.glob(f"tuning_*.npz"))
            if not candidates:
                print(f"ERROR: No processed {dataset} data in {PROCESSED_DIR}")
                sys.exit(1)
            path = candidates[0]

    if not path.exists():
        print(f"ERROR: {path} not found. Run preprocess.py first.")
        sys.exit(1)

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    X_feat = data["X_feat"]
    class_names = list(data["class_names"])
    n_classes = int(data["n_classes"])

    if "session_ids" in data:
        group_ids = data["session_ids"]
    elif "block_ids" in data:
        group_ids = data["block_ids"]
    else:
        group_ids = np.zeros(len(y), dtype=int)

    print(f"Loaded: X={X.shape}, X_feat={X_feat.shape}, "
          f"{n_classes} classes, {len(np.unique(group_ids))} groups")
    print(f"  Classes: {class_names}")
    return X, X_feat, y, group_ids, class_names, n_classes


# ─────────────────────────────────────────────────────────────────────
# Validation split utility
# ─────────────────────────────────────────────────────────────────────

def make_val_split(train_indices, y, val_fraction=VAL_FRACTION, seed=SEED):
    """Split training indices into train_sub/val, stratified by label."""
    from sklearn.model_selection import StratifiedShuffleSplit

    y_train = y[train_indices]
    # Ensure each class has at least 2 samples for stratification
    class_counts = np.bincount(y_train, minlength=y.max() + 1)
    min_count = class_counts[class_counts > 0].min()

    if min_count < 2 or len(train_indices) < 10:
        # Too few samples for stratified split; use random
        rng = np.random.RandomState(seed)
        n_val = max(1, int(len(train_indices) * val_fraction))
        perm = rng.permutation(len(train_indices))
        val_local = perm[:n_val]
        train_local = perm[n_val:]
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                     random_state=seed)
        train_local, val_local = next(sss.split(train_indices, y_train))

    return train_indices[train_local], train_indices[val_local]


# ─────────────────────────────────────────────────────────────────────
# CV fold generation
# ─────────────────────────────────────────────────────────────────────

def make_cv_folds(y, group_ids):
    """Create leave-one-group-out or StratifiedKFold splits."""
    unique_groups = np.unique(group_ids)
    use_group_cv = len(unique_groups) >= 3

    if use_group_cv:
        folds = []
        for g in unique_groups:
            train_idx = np.where(group_ids != g)[0]
            test_idx = np.where(group_ids == g)[0]
            folds.append((train_idx, test_idx))
        return folds, f"Leave-one-group-out ({len(unique_groups)} folds)"
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        folds = [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y)]
        return folds, "StratifiedKFold (5 folds)"


# ─────────────────────────────────────────────────────────────────────
# GPU-accelerated PCA + Standardization
# ─────────────────────────────────────────────────────────────────────

def gpu_pca_transform(X_train, X_test, n_components, device, X_val=None):
    """PCA via truncated SVD on GPU. Fits on X_train only."""
    Xtr = torch.FloatTensor(X_train).to(device) if not isinstance(X_train, torch.Tensor) else X_train
    Xte = torch.FloatTensor(X_test).to(device) if not isinstance(X_test, torch.Tensor) else X_test

    mu = Xtr.mean(dim=0)
    Xtr_c = Xtr - mu
    Xte_c = Xte - mu

    nc = min(n_components, Xtr_c.shape[0] - 1, Xtr_c.shape[1])
    U, S, Vh = torch.linalg.svd(Xtr_c, full_matrices=False)
    components = Vh[:nc]

    Xtr_pca = Xtr_c @ components.T
    Xte_pca = Xte_c @ components.T

    if X_val is not None:
        Xv = torch.FloatTensor(X_val).to(device) if not isinstance(X_val, torch.Tensor) else X_val
        Xv_pca = (Xv - mu) @ components.T
        return Xtr_pca, Xv_pca, Xte_pca

    return Xtr_pca, Xte_pca


def gpu_standardize(X_train, X_test, device, X_val=None):
    """Z-score standardization on GPU. Fits on X_train only."""
    Xtr = torch.FloatTensor(X_train).to(device) if not isinstance(X_train, torch.Tensor) else X_train
    Xte = torch.FloatTensor(X_test).to(device) if not isinstance(X_test, torch.Tensor) else X_test
    mu = Xtr.mean(dim=0)
    sd = Xtr.std(dim=0)
    sd[sd < 1e-8] = 1.0

    result = [(Xtr - mu) / sd, (Xte - mu) / sd]
    if X_val is not None:
        Xv = torch.FloatTensor(X_val).to(device) if not isinstance(X_val, torch.Tensor) else X_val
        result.insert(1, (Xv - mu) / sd)
    return tuple(result)


# ─────────────────────────────────────────────────────────────────────
# GPU classifiers (PyTorch)
# ─────────────────────────────────────────────────────────────────────

class GPULinearClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.fc(x)


class GPUMLPClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class GPUDeepMLP(nn.Module):
    def __init__(self, n_features, n_classes, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_gpu_classifier(model, Xtr, ytr, Xval, yval, Xte, device,
                         epochs=100, lr=1e-3, wd=1e-2, bs=128, patience=20):
    """Train classifier with val-based early stopping. Evaluate test ONCE."""
    model = model.to(device)
    ds = TensorDataset(Xtr, ytr)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    best_val_acc = 0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            with autocast("cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Evaluate on VALIDATION only
        model.eval()
        with torch.no_grad(), autocast("cuda"):
            val_preds = model(Xval).argmax(1).cpu().numpy()
        val_acc = (val_preds == yval.cpu().numpy()).mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    # Load best-val model, evaluate test ONCE
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), autocast("cuda"):
        test_preds = model(Xte).argmax(1).cpu().numpy()

    return test_preds, best_val_acc


# ─────────────────────────────────────────────────────────────────────
# Cross-validation on GPU (classical)
# ─────────────────────────────────────────────────────────────────────

def gpu_cv(X_feat, y, group_ids, model_fn, device, n_pca=N_PCA,
           epochs=100, lr=1e-3, wd=1e-2, bs=128, patience=20,
           val_fraction=VAL_FRACTION):
    """Cross-validation with val-based early stopping for classical models."""
    folds, cv_name = make_cv_folds(y, group_ids)

    yt_all, yp_all, fold_accs, fold_val_accs = [], [], [], []

    for train_idx, test_idx in folds:
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        # Split training into train_sub / val
        train_sub_idx, val_idx = make_val_split(train_idx, y, val_fraction)

        Xtr_np, Xval_np, Xte_np = X_feat[train_sub_idx], X_feat[val_idx], X_feat[test_idx]
        ytr_np, yval_np, yte_np = y[train_sub_idx], y[val_idx], y[test_idx]

        # Standardize (fit on train_sub only)
        Xtr_g, Xval_g, Xte_g = gpu_standardize(Xtr_np, Xte_np, device, X_val=Xval_np)

        # NaN check
        Xtr_g = torch.nan_to_num(Xtr_g)
        Xval_g = torch.nan_to_num(Xval_g)
        Xte_g = torch.nan_to_num(Xte_g)

        # PCA (fit on train_sub only)
        if n_pca and n_pca < Xtr_g.shape[1]:
            Xtr_g, Xval_g, Xte_g = gpu_pca_transform(
                Xtr_g.cpu().numpy(), Xte_g.cpu().numpy(), n_pca, device,
                X_val=Xval_g.cpu().numpy())

        n_feat = Xtr_g.shape[1]
        n_classes = len(np.unique(y))
        model = model_fn(n_feat, n_classes)

        ytr_t = torch.LongTensor(ytr_np).to(device)
        yval_t = torch.LongTensor(yval_np).to(device)

        preds, val_acc = train_gpu_classifier(
            model, Xtr_g, ytr_t, Xval_g, yval_t, Xte_g, device,
            epochs=epochs, lr=lr, wd=wd, bs=bs, patience=patience)

        yt_all.extend(yte_np)
        yp_all.extend(preds)
        test_acc = (preds == yte_np).mean()
        fold_accs.append(test_acc)
        fold_val_accs.append(val_acc)

    return np.array(yt_all), np.array(yp_all), fold_accs, fold_val_accs


def train_classical_gpu(X_feat, y, group_ids, class_names, n_classes, device,
                        val_fraction=VAL_FRACTION):
    """Train classical models with proper val-based early stopping."""
    models = {
        "Linear": (lambda nf, nc: GPULinearClassifier(nf, nc), 80, 1e-2, 1e-2),
        "MLP-256": (lambda nf, nc: GPUMLPClassifier(nf, nc, 256), 120, 1e-3, 1e-2),
        "MLP-512": (lambda nf, nc: GPUMLPClassifier(nf, nc, 512), 120, 1e-3, 1e-2),
        "DeepMLP": (lambda nf, nc: GPUDeepMLP(nf, nc, 512), 150, 5e-4, 1e-2),
    }

    folds_info, cv_name = make_cv_folds(y, group_ids)
    print(f"\n{'='*60}")
    print(f"GPU CLASSIFIERS — {cv_name}")
    print(f"  Val fraction: {val_fraction:.0%} (for early stopping)")
    print(f"{'='*60}")

    results = {}
    for name, (mfn, epochs, lr, wd) in models.items():
        t0 = time.time()
        yt, yp, fa, fva = gpu_cv(X_feat, y, group_ids, mfn, device,
                                  epochs=epochs, lr=lr, wd=wd,
                                  val_fraction=val_fraction)
        acc = (yt == yp).mean() if len(yt) > 0 else 0
        mean_val = np.mean(fva)
        elapsed = time.time() - t0

        results[name] = {
            "acc": float(acc),
            "fold_accs": fa,
            "fold_val_accs": fva,
            "yt": yt,
            "yp": yp,
        }
        folds_str = " ".join(f"{a:.3f}" for a in fa)
        print(f"  {name}: test={acc:.3f} (val={mean_val:.3f}) [{folds_str}] ({elapsed:.1f}s)")

    # Best model report
    from sklearn.metrics import classification_report
    best_name = max(results, key=lambda m: results[m]["acc"])
    yt, yp = results[best_name]["yt"], results[best_name]["yp"]
    print(f"\nBest: {best_name} ({results[best_name]['acc']:.3f})")
    if len(yt) > 0:
        n_actual = len(np.unique(np.concatenate([yt, yp])))
        tgt = class_names[:n_actual]
        print(classification_report(yt, yp, target_names=tgt, zero_division=0))

    return results


# ─────────────────────────────────────────────────────────────────────
# Permutation test (GPU)
# ─────────────────────────────────────────────────────────────────────

def permutation_test_gpu(X_feat, y, group_ids, device, n_perm=500):
    print(f"\nPermutation test ({n_perm}×, Linear on GPU)...")
    t0 = time.time()
    mfn = lambda nf, nc: GPULinearClassifier(nf, nc)

    _, _, fa, _ = gpu_cv(X_feat, y, group_ids, mfn, device, epochs=60, lr=1e-2)
    true_acc = np.mean(fa)

    null_dist = []
    for i in range(n_perm):
        y_shuf = np.random.permutation(y)
        _, _, fa_s, _ = gpu_cv(X_feat, y_shuf, group_ids, mfn, device, epochs=60, lr=1e-2)
        null_dist.append(np.mean(fa_s))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_perm} ({time.time() - t0:.0f}s)")

    null_dist = np.array(null_dist)
    p_val = np.mean(null_dist >= true_acc)
    print(f"  True={true_acc:.3f}, Null={null_dist.mean():.3f}±{null_dist.std():.3f}, p={p_val:.4f}")
    return true_acc, null_dist, p_val


# ─────────────────────────────────────────────────────────────────────
# Deep Learning Models
# ─────────────────────────────────────────────────────────────────────

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


class TCN(nn.Module):
    def __init__(self, nc, nt, nk, dr=0.3, hidden=128):
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


class SpeechTransformer(nn.Module):
    def __init__(self, nc, nt, nk, d_model=128, nhead=8, num_layers=4, dr=0.3):
        super().__init__()
        self.input_proj = nn.Linear(nc, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, nt + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dr, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dr),
            nn.Linear(d_model, nk)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.encoder(x)
        return self.fc(x[:, 0])


class GRUDecoder(nn.Module):
    def __init__(self, nc, nt, nk, hidden=256, n_layers=2, dr=0.3):
        super().__init__()
        self.gru = nn.GRU(nc, hidden, n_layers, batch_first=True,
                          dropout=dr if n_layers > 1 else 0, bidirectional=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dr),
            nn.Linear(hidden * 2, nk)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.fc(out.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────

def apply_augmentation(xb, mode="standard"):
    """Apply data augmentation to a batch on GPU.

    Args:
        xb: (batch, channels, time) tensor on GPU
        mode: "none", "standard", or "paper"
    """
    if mode == "none":
        return xb
    elif mode == "paper":
        # Willett et al.: white noise SD=1.0, constant offset SD=0.2
        xb = xb + torch.randn_like(xb) * 1.0
        offset = torch.randn(xb.shape[0], xb.shape[1], 1, device=xb.device) * 0.2
        return xb + offset
    else:  # standard
        if torch.rand(1).item() < 0.5:
            xb = xb + torch.randn_like(xb) * 0.1
        if torch.rand(1).item() < 0.3:
            shift = torch.randint(-3, 4, (1,)).item()
            if shift != 0:
                xb = torch.roll(xb, shift, dims=2)
        return xb


# ─────────────────────────────────────────────────────────────────────
# DL Training — Multi-GPU with proper val/test separation
# ─────────────────────────────────────────────────────────────────────

def _eval_batch(model, X_gpu, batch_size=512):
    """Evaluate model on GPU data in batches."""
    preds_list = []
    for i in range(0, len(X_gpu), batch_size):
        preds_list.append(model(X_gpu[i:i + batch_size]).argmax(1))
    return torch.cat(preds_list).cpu().numpy()


def train_dl_cv(X, y, group_ids, model_cls, model_kw, device,
                epochs=DL_EPOCHS, lr=DL_LR, wd=DL_WD, bs=DL_BS,
                patience=DL_PATIENCE, val_fraction=VAL_FRACTION,
                augmentation="standard", num_workers=NUM_WORKERS,
                prefetch_factor=PREFETCH_FACTOR):
    """Train DL model with proper val-based early stopping.

    CRITICAL: Test fold is evaluated ONCE at the val-best checkpoint.
    Oracle (best-test) accuracy is also tracked for gap analysis.

    Returns:
        dict with val_stopped_acc, oracle_acc, fold_details, preds, best_state
    """
    folds, cv_name = make_cv_folds(y, group_ids)
    n_folds = len(folds)

    val_stopped_preds = np.full(len(y), -1, dtype=np.int64)
    oracle_preds = np.full(len(y), -1, dtype=np.int64)
    fold_details = []
    best_state = None
    best_overall_val = 0
    n_gpus = torch.cuda.device_count()

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        if len(test_idx) == 0:
            continue

        # Split training into train_sub / val
        train_sub_idx, val_idx = make_val_split(train_idx, y, val_fraction,
                                                 seed=SEED + fold_i)

        # Create tensors: (trials, channels, time)
        Xtr = torch.FloatTensor(X[train_sub_idx].transpose(0, 2, 1))
        Xval = torch.FloatTensor(X[val_idx].transpose(0, 2, 1))
        Xte = torch.FloatTensor(X[test_idx].transpose(0, 2, 1))
        ytr = torch.LongTensor(y[train_sub_idx])
        yval_np = y[val_idx]
        yte_np = y[test_idx]

        # DataLoader with proper parallelism
        train_ds = TensorDataset(Xtr, ytr)
        effective_bs = bs * max(n_gpus, 1)
        use_workers = num_workers if len(train_ds) > effective_bs else 0
        loader_kw = dict(
            batch_size=effective_bs,
            shuffle=True,
            num_workers=use_workers,
            pin_memory=True,
        )
        if use_workers > 0:
            loader_kw["persistent_workers"] = True
            loader_kw["prefetch_factor"] = prefetch_factor
        train_loader = DataLoader(train_ds, **loader_kw)

        # Model setup
        model = model_cls(**model_kw)
        if n_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
        model = model.to(device)

        Xval_gpu = Xval.to(device)
        Xte_gpu = Xte.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(epochs // 3, 10), T_mult=2)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler("cuda")

        best_val_acc = 0
        best_val_state = None
        best_oracle_acc = 0
        best_oracle_preds_fold = None
        wait = 0
        history = {"train_loss": [], "val_acc": [], "test_acc_oracle": []}

        for epoch in range(epochs):
            # ── Train ──
            model.train()
            epoch_loss = 0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                xb = apply_augmentation(xb, mode=augmentation)

                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # ── Evaluate VAL (used for early stopping) ──
            model.eval()
            with torch.no_grad(), autocast("cuda"):
                val_preds = _eval_batch(model, Xval_gpu)
            val_acc = (val_preds == yval_np).mean()

            # ── Evaluate TEST (oracle tracking only — NOT used for decisions) ──
            with torch.no_grad(), autocast("cuda"):
                test_preds_epoch = _eval_batch(model, Xte_gpu)
            test_acc_epoch = (test_preds_epoch == yte_np).mean()

            history["train_loss"].append(avg_loss)
            history["val_acc"].append(val_acc)
            history["test_acc_oracle"].append(test_acc_epoch)

            # Track oracle (for gap reporting)
            if test_acc_epoch > best_oracle_acc:
                best_oracle_acc = test_acc_epoch
                best_oracle_preds_fold = test_preds_epoch.copy()

            # Early stopping on VAL
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                m = model.module if hasattr(model, 'module') else model
                best_val_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break

        # ── Final test evaluation from val-best model (SINGLE evaluation) ──
        if best_val_state is not None:
            m = model.module if hasattr(model, 'module') else model
            m.load_state_dict(best_val_state)
        model.eval()
        with torch.no_grad(), autocast("cuda"):
            final_test_preds = _eval_batch(model, Xte_gpu)
        vs_test_acc = (final_test_preds == yte_np).mean()

        val_stopped_preds[test_idx] = final_test_preds
        if best_oracle_preds_fold is not None:
            oracle_preds[test_idx] = best_oracle_preds_fold

        # Track best model state across folds
        if best_val_acc > best_overall_val:
            best_overall_val = best_val_acc
            best_state = best_val_state

        fold_details.append({
            "val_stopped_test_acc": vs_test_acc,
            "oracle_test_acc": best_oracle_acc,
            "best_val_acc": best_val_acc,
            "epochs_trained": len(history["val_acc"]),
            "gap": best_oracle_acc - vs_test_acc,
        })

        gap = best_oracle_acc - vs_test_acc
        print(f"  Fold {fold_i+1}/{n_folds}: "
              f"val-stopped={vs_test_acc:.3f}, oracle={best_oracle_acc:.3f}, "
              f"gap={gap:+.3f} @ ep {len(history['val_acc'])}")

        # Cleanup GPU
        del Xval_gpu, Xte_gpu, model
        torch.cuda.empty_cache()

    # Overall accuracies
    vs_valid = val_stopped_preds >= 0
    or_valid = oracle_preds >= 0
    vs_acc = (y[vs_valid] == val_stopped_preds[vs_valid]).mean() if vs_valid.sum() > 0 else 0
    or_acc = (y[or_valid] == oracle_preds[or_valid]).mean() if or_valid.sum() > 0 else 0

    return {
        "val_stopped_acc": float(vs_acc),
        "oracle_acc": float(or_acc),
        "gap": float(or_acc - vs_acc),
        "fold_details": fold_details,
        "val_stopped_preds": val_stopped_preds,
        "oracle_preds": oracle_preds,
        "best_state": best_state,
    }


def train_deep_learning(X, y, group_ids, device, epochs, class_names, n_classes,
                        augmentation="standard", num_workers=NUM_WORKERS,
                        val_fraction=VAL_FRACTION):
    nc = X.shape[2]
    nt = X.shape[1]
    n_gpus = torch.cuda.device_count()

    print(f"\n{'='*60}")
    print(f"DEEP LEARNING (GPUs={n_gpus}, AMP=on, epochs={epochs})")
    print(f"  Input: {nc} ch × {nt} time, {n_classes} classes")
    print(f"  Val fraction: {val_fraction:.0%}, Aug: {augmentation}")
    print(f"  Workers: {num_workers}, Prefetch: {PREFETCH_FACTOR}")
    print(f"{'='*60}")

    model_configs = {
        "EEGNet": (EEGNet, dict(nc=nc, nt=nt, nk=n_classes, **EEGNET_KW)),
        "TCN": (TCN, dict(nc=nc, nt=nt, nk=n_classes, **TCN_KW)),
        "GRU": (GRUDecoder, dict(nc=nc, nt=nt, nk=n_classes, hidden=256, n_layers=2)),
        "Transformer": (SpeechTransformer, dict(
            nc=nc, nt=nt, nk=n_classes, d_model=128, nhead=8, num_layers=4
        )),
    }

    for name, (cls, kw) in model_configs.items():
        tmp = cls(**kw)
        print(f"  {name}: {sum(p.numel() for p in tmp.parameters()):,} params")
        del tmp

    dl_results = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, (cls, kw) in model_configs.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        result = train_dl_cv(
            X, y, group_ids, cls, kw, device, epochs=epochs,
            bs=DL_BS if name != "Transformer" else DL_BS // 2,
            augmentation=augmentation, num_workers=num_workers,
            val_fraction=val_fraction,
        )
        elapsed = time.time() - t0

        dl_results[name] = result

        vs_folds = " ".join(f"{d['val_stopped_test_acc']:.3f}" for d in result["fold_details"])
        mean_gap = np.mean([d["gap"] for d in result["fold_details"]])
        print(f"  {name}: val-stopped={result['val_stopped_acc']:.3f}, "
              f"oracle={result['oracle_acc']:.3f}, gap={mean_gap:+.3f}")
        print(f"    folds: [{vs_folds}] ({elapsed:.0f}s)")

        if result["best_state"]:
            torch.save(result["best_state"], MODELS_DIR / f"{name.lower()}_best.pt")

    return dl_results


# ─────────────────────────────────────────────────────────────────────
# Sanity check: label shuffle
# ─────────────────────────────────────────────────────────────────────

def sanity_check_shuffled_labels(X, y, group_ids, device, n_iters=5):
    """Run TCN with shuffled labels to verify model learns real signal."""
    nc, nt = X.shape[2], X.shape[1]
    n_classes = len(np.unique(y))
    model_kw = dict(nc=nc, nt=nt, nk=n_classes, **TCN_KW)

    print(f"\n{'='*60}")
    print(f"SANITY CHECK: Shuffled labels ({n_iters} iterations)")
    print(f"  Expected accuracy: ~{1/n_classes:.3f} (chance = 1/{n_classes})")
    print(f"{'='*60}")

    shuffled_accs = []
    for i in range(n_iters):
        y_shuf = np.random.permutation(y)
        result = train_dl_cv(
            X, y_shuf, group_ids, TCN, model_kw, device,
            epochs=30, patience=10, augmentation="none", num_workers=2)
        shuffled_accs.append(result["val_stopped_acc"])
        print(f"  Iter {i+1}/{n_iters}: {result['val_stopped_acc']:.3f}")

    mean_shuf = np.mean(shuffled_accs)
    std_shuf = np.std(shuffled_accs)
    chance = 1 / n_classes
    print(f"\n  Shuffled: {mean_shuf:.3f} ± {std_shuf:.3f}")
    print(f"  Chance:   {chance:.3f}")
    if mean_shuf > 2 * chance:
        print(f"  ⚠ WARNING: Shuffled accuracy ({mean_shuf:.3f}) is >2× chance ({chance:.3f})!")
        print(f"    This may indicate data leakage.")
    else:
        print(f"  ✓ OK: Shuffled accuracy is near chance level.")

    return shuffled_accs


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def save_figures(results, dl_results, perm_data, class_names, n_classes, y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── All models comparison (val-stopped for DL) ──
    all_names = list(results.keys())
    all_accs = [results[m]["acc"] for m in results]
    colors = ["#3498db"] * len(results)
    oracle_accs = []
    if dl_results:
        for name, res in dl_results.items():
            all_names.append(name)
            all_accs.append(res["val_stopped_acc"])
            oracle_accs.append(res["oracle_acc"])
            colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(all_names))
    bars = ax.bar(x, all_accs, color=colors, edgecolor="white", label="Val-stopped")

    # Show oracle as lighter bars for DL models
    if oracle_accs:
        n_classical = len(results)
        ax.bar(x[n_classical:], oracle_accs, color="none",
               edgecolor="#c0392b", linewidth=2, linestyle="--", label="Oracle (test-peeked)")

    ax.axhline(1/n_classes, color="gray", ls="--", lw=1, label=f"Chance (1/{n_classes})")
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=30, ha="right", fontsize=10)
    ax.set(ylabel="Accuracy", title="Speech Intent Prediction — Val-Stopped vs Oracle", ylim=(0, 1))
    for b, a in zip(bars, all_accs):
        ax.text(b.get_x()+b.get_width()/2, a+0.02, f"{a:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "all_models.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  all_models.png")

    # ── Confusion matrix from best DL model (val-stopped) ──
    best_dl_name = None
    best_dl_acc = 0
    if dl_results:
        for name, res in dl_results.items():
            if res["val_stopped_acc"] > best_dl_acc:
                best_dl_acc = res["val_stopped_acc"]
                best_dl_name = name

    if best_dl_name:
        preds = dl_results[best_dl_name]["val_stopped_preds"]
        valid = preds >= 0
        if valid.sum() > 0:
            yt, yp = y[valid], preds[valid]
            cm = confusion_matrix(yt, yp)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
            n = cm.shape[0]
            names = class_names[:n]

            # Save confusion matrix
            np.save(MODELS_DIR / "confusion_matrix.npy", cm)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            for ax, (data, title, fmt) in zip(axes,
                    [(cm, f"Counts — {best_dl_name}", "d"),
                     (cm_norm, f"Normalized — {best_dl_name}", ".2f")]):
                ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
                if n <= 20:
                    for i in range(n):
                        for j in range(n):
                            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                                    color="white" if data[i, j] > data.max()/2 else "black",
                                    fontsize=max(5, 8 - n // 10))
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(names, rotation=45, ha="right", fontsize=max(5, 8 - n // 10))
                ax.set_yticklabels(names, fontsize=max(5, 8 - n // 10))
                ax.set(xlabel="Predicted", ylabel="True", title=title)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  confusion_matrix.png")

    # ── Gap analysis figure ──
    if dl_results:
        names_dl = list(dl_results.keys())
        vs_accs = [dl_results[n]["val_stopped_acc"] for n in names_dl]
        or_accs = [dl_results[n]["oracle_acc"] for n in names_dl]
        gaps = [o - v for v, o in zip(vs_accs, or_accs)]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names_dl))
        w = 0.35
        ax.bar(x - w/2, vs_accs, w, label="Val-stopped", color="#2980b9")
        ax.bar(x + w/2, or_accs, w, label="Oracle (test-peeked)", color="#e74c3c", alpha=0.7)
        for i, g in enumerate(gaps):
            ax.annotate(f"gap={g:+.3f}", (x[i], max(vs_accs[i], or_accs[i]) + 0.01),
                        ha="center", fontsize=9, color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(names_dl, fontsize=11)
        ax.set(ylabel="Accuracy", title="Test-Set Peeking Gap Analysis")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "gap_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  gap_analysis.png")

    # ── Permutation ──
    if perm_data:
        true_acc, null_dist, p_val = perm_data
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(null_dist, bins=50, color="#95a5a6", edgecolor="white", alpha=0.8)
        ax.axvline(true_acc, color="#e74c3c", lw=2, label=f"Observed: {true_acc:.3f} (p={p_val:.4f})")
        ax.axvline(1/n_classes, color="gray", ls="--", lw=1, label="Chance")
        ax.set(xlabel="Accuracy", ylabel="Count", title="Permutation Test")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "permutation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  permutation.png")


# ─────────────────────────────────────────────────────────────────────
# Paper comparison
# ─────────────────────────────────────────────────────────────────────

def compare_to_paper(results, dl_results, n_classes, class_names):
    print(f"\n{'='*70}")
    print("COMPARISON TO WILLETT ET AL. (NATURE 2023)")
    print(f"{'='*70}")
    print(f"""
Paper reference results:
  - 50-word vocabulary:  9.1% WER (word error rate)
  - 125K vocabulary:    23.8% WER
  - Phoneme classif.:   ~62% accuracy (39 classes, Naive Bayes)
  - Method: GRU RNN + 5-gram language model
  - Channels: 256 intracortical microelectrodes

Our task: {n_classes}-class classification (chance={100/n_classes:.1f}%)
  - No language model (single-trial classification)
  - Using spikePow + threshold crossing features
  - Val-stopped evaluation (no test-set peeking)
""")

    all_models = [(n, results[n]["acc"]) for n in results]
    if dl_results:
        all_models += [(n, dl_results[n]["val_stopped_acc"]) for n in dl_results]

    best_name, best_acc = max(all_models, key=lambda x: x[1])
    print(f"Our best (val-stopped): {best_name} = {best_acc:.3f} ({best_acc*100:.1f}%)")
    print(f"Chance: {1/n_classes:.3f} ({100/n_classes:.1f}%)")
    print(f"Relative: {best_acc / (1/n_classes):.1f}× chance")


# ─────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────

def save_results_json(results, dl_results, perm_data, sanity_data,
                      dataset_tag="results", n_classes=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "version": 2,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "val_fraction": VAL_FRACTION,
            "seed": SEED,
        },
        "classical_gpu": {},
        "dl": {},
        "permutation": None,
        "sanity_check": None,
    }

    for name in results:
        out["classical_gpu"][name] = {
            "acc": float(results[name]["acc"]),
            "fold_accs": [float(a) for a in results[name]["fold_accs"]],
            "fold_val_accs": [float(a) for a in results[name]["fold_val_accs"]],
        }

    if dl_results:
        for name, res in dl_results.items():
            out["dl"][name] = {
                "val_stopped_acc": res["val_stopped_acc"],
                "oracle_acc": res["oracle_acc"],
                "gap": res["gap"],
                "fold_details": [
                    {k: float(v) if isinstance(v, (float, np.floating)) else v
                     for k, v in d.items()}
                    for d in res["fold_details"]
                ],
            }

    if perm_data:
        true_acc, null_dist, p_val = perm_data
        out["permutation"] = {
            "true_acc": float(true_acc),
            "null_mean": float(null_dist.mean()),
            "p_value": float(p_val),
        }

    if sanity_data:
        out["sanity_check"] = {
            "shuffled_accs": [float(a) for a in sanity_data],
            "chance": float(1 / n_classes) if n_classes else None,
        }

    path = MODELS_DIR / f"{dataset_tag}_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train speech intent models (GPU, v2)")
    parser.add_argument("--dataset", default="diagnostic",
                        help="diagnostic, or tuning filename prefix")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--dl-only", action="store_true")
    parser.add_argument("--no-permutation", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--epochs", type=int, default=DL_EPOCHS)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--sanity", action="store_true",
                        help="Run label-shuffle sanity check")
    parser.add_argument("--augmentation", choices=["none", "standard", "paper"],
                        default="standard")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    args = parser.parse_args()

    t_start = time.time()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"train.py v2 — val-based early stopping, no test peeking")
    device = get_device()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X, X_feat, y, group_ids, class_names, n_classes = load_data(args.dataset)

    # Sanity check
    sanity_data = None
    if args.sanity:
        sanity_data = sanity_check_shuffled_labels(X, y, group_ids, device)

    # Classical GPU models
    results = {}
    if not args.dl_only:
        results = train_classical_gpu(X_feat, y, group_ids, class_names, n_classes,
                                      device, val_fraction=args.val_fraction)

    # Permutation test
    perm_data = None
    if not args.no_permutation and not args.dl_only:
        perm_data = permutation_test_gpu(X_feat, y, group_ids, device, args.n_perm)

    # Deep learning
    dl_results = None
    if not args.classical_only:
        dl_results = train_deep_learning(
            X, y, group_ids, device, args.epochs, class_names, n_classes,
            augmentation=args.augmentation, num_workers=args.num_workers,
            val_fraction=args.val_fraction)

    # Figures
    if not args.no_figures:
        print(f"\nSaving figures to {FIGURES_DIR}/")
        save_figures(results, dl_results, perm_data, class_names, n_classes, y)

    # Save
    ds_tag = args.dataset.replace("/", "_").replace(".", "_")
    save_results_json(results, dl_results, perm_data, sanity_data,
                      dataset_tag=ds_tag, n_classes=n_classes)

    # Paper comparison
    compare_to_paper(results, dl_results, n_classes, class_names)

    # Summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'Val-Stopped':>12} {'Oracle':>10} {'Gap':>8}")
    print(f"  {'─'*20} {'─'*12} {'─'*10} {'─'*8}")

    all_models = []
    for n in results:
        all_models.append((n, results[n]["acc"], results[n]["acc"], 0))
    if dl_results:
        for n, res in dl_results.items():
            all_models.append((n, res["val_stopped_acc"], res["oracle_acc"], res["gap"]))

    for n, vs, oracle, gap in sorted(all_models, key=lambda x: -x[1]):
        if gap > 0:
            print(f"  {n:<20} {vs:>12.3f} {oracle:>10.3f} {gap:>+8.3f}")
        else:
            print(f"  {n:<20} {vs:>12.3f}")

    if dl_results:
        mean_gap = np.mean([res["gap"] for res in dl_results.values()])
        print(f"\n  Mean DL gap (oracle - val-stopped): {mean_gap:+.3f}")
        print(f"  This is the inflation from test-set peeking in v1.")

    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
