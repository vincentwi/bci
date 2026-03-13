#!/usr/bin/env python3
"""
Advanced training experiments for paper comparison.

Experiments:
  1. Merged phoneme sessions (1440 trials, 5-fold CV)
  2. Cross-session generalization (train sess1 → test sess2 and vice versa)
  3. Paper-matching augmentation (white noise SD=1.0, offset SD=0.2)
  4. Larger architectures (TCN-256, 5-layer GRU-512)
  5. Save softmax probabilities & confusion matrices per fold

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_advanced.py
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, SEED, VAL_FRACTION, NUM_WORKERS, PREFETCH_FACTOR

np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = MODELS_DIR / "advanced"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Models ──────────────────────────────────────────────────────────

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


class TCN256(nn.Module):
    """Wider TCN with 256 hidden units."""
    def __init__(self, nc, nt, nk, dr=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(nc, 256, 7, padding=3),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dr),
            nn.Conv1d(256, 256, 7, padding=3),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dr),
            nn.Conv1d(256, 256, 7, padding=6, dilation=2),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dr),
            nn.Conv1d(256, 128, 7, padding=12, dilation=4),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dr),
            nn.Conv1d(128, 64, 5, padding=8, dilation=4),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dr),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, nk)

    def forward(self, x):
        return self.fc(self.pool(self.net(x)).squeeze(-1))


class GRU5Layer(nn.Module):
    """5-layer bidirectional GRU with 512 units — closer to paper's architecture."""
    def __init__(self, nc, nt, nk, hidden=512, n_layers=5, dr=0.3):
        super().__init__()
        self.gru = nn.GRU(nc, hidden, n_layers, batch_first=True,
                          dropout=dr, bidirectional=True)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dr),
            nn.Linear(hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, nk)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.fc(out.mean(dim=1))


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


# ── Training function with paper augmentation & softmax saving ──────

def _eval_batch(model, X_gpu, batch_size=256):
    """Batched GPU evaluation to avoid OOM on large test sets."""
    model.eval()
    all_logits = []
    with torch.no_grad(), autocast("cuda"):
        for i in range(0, len(X_gpu), batch_size):
            all_logits.append(model(X_gpu[i:i+batch_size]))
    return torch.cat(all_logits, dim=0)


def train_model(X_train, y_train, X_test, y_test, model_cls, model_kw,
                device, epochs=200, lr=3e-4, wd=1e-2, bs=32, patience=30,
                paper_aug=True, save_softmax=False):
    """Train model with val-based early stopping (NO test peeking).

    Splits X_train into train_sub (85%) and val (15%).
    Early stopping based on val accuracy only.
    Test evaluated ONCE at the end with val-best checkpoint.
    """
    n_gpus = torch.cuda.device_count()

    # --- Split training into train_sub + val ---
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    sub_idx, val_idx = next(sss.split(X_train, y_train))

    X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    Xtr = torch.FloatTensor(X_sub.transpose(0, 2, 1))
    Xval = torch.FloatTensor(X_val.transpose(0, 2, 1))
    Xte = torch.FloatTensor(X_test.transpose(0, 2, 1))
    ytr = torch.LongTensor(y_sub)

    train_ds = TensorDataset(Xtr, ytr)
    eff_bs = bs * max(n_gpus, 1)
    train_loader = DataLoader(train_ds, batch_size=eff_bs, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)

    model = model_cls(**model_kw)
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    Xval_gpu = Xval.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(epochs // 3, 10), T_mult=2)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    best_val_acc = 0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            if paper_aug:
                xb = xb + torch.randn_like(xb) * 1.0
                offset = torch.randn(xb.shape[0], xb.shape[1], 1, device=xb.device) * 0.2
                xb = xb + offset
            else:
                if np.random.random() < 0.5:
                    xb = xb + torch.randn_like(xb) * 0.1
                if np.random.random() < 0.3:
                    shift = np.random.randint(-3, 4)
                    if shift != 0:
                        xb = torch.roll(xb, shift, dims=2)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Early stopping on VALIDATION set only
        val_logits = _eval_batch(model, Xval_gpu)
        val_preds = val_logits.argmax(1).cpu().numpy()
        val_acc = (val_preds == y_val).mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    # --- Evaluate test ONCE with val-best checkpoint ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    Xte_gpu = Xte.to(device)
    test_logits = _eval_batch(model, Xte_gpu)
    test_preds = test_logits.argmax(1).cpu().numpy()
    test_acc = (test_preds == y_test).mean()

    test_probs = None
    if save_softmax:
        test_probs = F.softmax(test_logits, dim=1).cpu().numpy()

    return test_acc, test_preds, test_probs, {"val_best": best_val_acc, "test": test_acc}


# ── Experiments ──────────────────────────────────────────────────────

def exp1_merged_phonemes(device):
    """Merge both phoneme sessions, 5-fold CV."""
    print("\n" + "=" * 70)
    print("EXP 1: MERGED PHONEME SESSIONS (1440 trials, 40 classes, 5-fold CV)")
    print("=" * 70)

    d1 = np.load(PROCESSED_DIR / "tuning_t12.2022.04.21_phonemes.npz", allow_pickle=True)
    d2 = np.load(PROCESSED_DIR / "tuning_t12.2022.04.26_phonemes.npz", allow_pickle=True)

    X = np.concatenate([d1["X"], d2["X"]])
    y = np.concatenate([d1["y"], d2["y"]])
    class_names = list(d1["class_names"])
    n_classes = int(d1["n_classes"])

    print(f"  Merged: X={X.shape}, {n_classes} classes, {len(X)} total trials")
    print(f"  Session 1: {len(d1['X'])} trials, Session 2: {len(d2['X'])} trials")

    nc, nt = X.shape[2], X.shape[1]
    models = {
        "EEGNet": (EEGNet, dict(nc=nc, nt=nt, nk=n_classes)),
        "TCN-128": (TCN, dict(nc=nc, nt=nt, nk=n_classes, hidden=128)),
        "TCN-256": (TCN256, dict(nc=nc, nt=nt, nk=n_classes)),
        "GRU-5L-512": (GRU5Layer, dict(nc=nc, nt=nt, nk=n_classes)),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results = {}

    for name, (cls, kw) in models.items():
        tmp = cls(**kw)
        n_params = sum(p.numel() for p in tmp.parameters())
        print(f"\n  {name}: {n_params:,} params")
        del tmp

        all_preds = np.full(len(y), -1, dtype=np.int64)
        all_probs = np.zeros((len(y), n_classes), dtype=np.float32)
        fold_accs = []

        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
            t0 = time.time()
            acc, preds, probs, hist = train_model(
                X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
                cls, kw, device, epochs=200, patience=30,
                paper_aug=True, save_softmax=True
            )
            all_preds[te_idx] = preds
            if probs is not None:
                all_probs[te_idx] = probs
            fold_accs.append(acc)
            print(f"    Fold {fold+1}/5: {acc:.3f} ({time.time()-t0:.0f}s)")

        overall = (all_preds[all_preds >= 0] == y[all_preds >= 0]).mean()
        results[name] = {"acc": overall, "fold_accs": fold_accs,
                         "preds": all_preds, "probs": all_probs}
        print(f"  {name} overall: {overall:.3f} [{' '.join(f'{a:.3f}' for a in fold_accs)}]")

    # Save confusion matrix for best model
    best_name = max(results, key=lambda m: results[m]["acc"])
    best = results[best_name]
    valid = best["preds"] >= 0
    cm = confusion_matrix(y[valid], best["preds"][valid])

    np.save(RESULTS_DIR / "merged_phoneme_confusion.npy", cm)
    np.save(RESULTS_DIR / "merged_phoneme_probs.npy", best["probs"])
    np.save(RESULTS_DIR / "merged_phoneme_class_names.npy", np.array(class_names))

    print(f"\n  Best merged: {best_name} = {best['acc']:.3f}")
    print(f"  Paper baseline: 62% (Naive Bayes, 128ch)")
    print(f"  Improvement: +{(best['acc'] - 0.62)*100:.1f}%")

    # Save per-class accuracy
    per_class = {}
    yt, yp = y[valid], best["preds"][valid]
    for c in range(n_classes):
        mask = yt == c
        if mask.sum() > 0:
            per_class[class_names[c]] = float((yp[mask] == c).mean())
    print(f"\n  Per-class accuracy (best 5):")
    for name_c, acc in sorted(per_class.items(), key=lambda x: -x[1])[:5]:
        print(f"    {name_c}: {acc:.3f}")
    print(f"  Per-class accuracy (worst 5):")
    for name_c, acc in sorted(per_class.items(), key=lambda x: x[1])[:5]:
        print(f"    {name_c}: {acc:.3f}")

    return results


def exp2_cross_session(device):
    """Cross-session generalization: train sess1→test sess2, then swap."""
    print("\n" + "=" * 70)
    print("EXP 2: CROSS-SESSION GENERALIZATION (phonemes)")
    print("=" * 70)

    d1 = np.load(PROCESSED_DIR / "tuning_t12.2022.04.21_phonemes.npz", allow_pickle=True)
    d2 = np.load(PROCESSED_DIR / "tuning_t12.2022.04.26_phonemes.npz", allow_pickle=True)

    X1, y1 = d1["X"], d1["y"]
    X2, y2 = d2["X"], d2["y"]
    nc, nt = X1.shape[2], X1.shape[1]
    n_classes = int(d1["n_classes"])

    print(f"  Session 1: {X1.shape}, Session 2: {X2.shape}")

    models = {
        "TCN-256": (TCN256, dict(nc=nc, nt=nt, nk=n_classes)),
        "GRU-5L-512": (GRU5Layer, dict(nc=nc, nt=nt, nk=n_classes)),
    }

    for name, (cls, kw) in models.items():
        print(f"\n  {name}:")
        # Sess1 → Sess2
        acc_12, preds_12, probs_12, _ = train_model(
            X1, y1, X2, y2, cls, kw, device, epochs=200, patience=30,
            paper_aug=True, save_softmax=True
        )
        print(f"    Sess1→Sess2: {acc_12:.3f}")

        # Sess2 → Sess1
        acc_21, preds_21, probs_21, _ = train_model(
            X2, y2, X1, y1, cls, kw, device, epochs=200, patience=30,
            paper_aug=True, save_softmax=True
        )
        print(f"    Sess2→Sess1: {acc_21:.3f}")
        print(f"    Mean cross-session: {(acc_12 + acc_21)/2:.3f}")

        # Save
        np.save(RESULTS_DIR / f"cross_session_{name}_s1_to_s2_probs.npy", probs_12)
        np.save(RESULTS_DIR / f"cross_session_{name}_s2_to_s1_probs.npy", probs_21)

    return {"acc_12": acc_12, "acc_21": acc_21}


def exp3_paper_aug_comparison(device):
    """Compare paper augmentation vs standard augmentation."""
    print("\n" + "=" * 70)
    print("EXP 3: AUGMENTATION ABLATION")
    print("=" * 70)

    d = np.load(PROCESSED_DIR / "tuning_t12.2022.04.26_phonemes.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    nc, nt = X.shape[2], X.shape[1]
    n_classes = int(d["n_classes"])

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    model_kw = dict(nc=nc, nt=nt, nk=n_classes)

    for aug_name, paper_aug in [("standard", False), ("paper_aug", True)]:
        fold_accs = []
        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
            acc, _, _, _ = train_model(
                X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
                TCN256, model_kw, device, epochs=200, patience=30,
                paper_aug=paper_aug, save_softmax=False
            )
            fold_accs.append(acc)
        mean_acc = np.mean(fold_accs)
        print(f"  TCN-256 ({aug_name}): {mean_acc:.3f} [{' '.join(f'{a:.3f}' for a in fold_accs)}]")


def main():
    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}, GPUs: {n_gpus}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # Exp 1: Merged phonemes (most important)
    merged_results = exp1_merged_phonemes(device)

    # Exp 2: Cross-session
    cross_results = exp2_cross_session(device)

    # Exp 3: Augmentation ablation
    exp3_paper_aug_comparison(device)

    # Save all results
    summary = {
        "merged_phonemes": {name: {"acc": float(r["acc"]),
                                     "fold_accs": [float(a) for a in r["fold_accs"]]}
                            for name, r in merged_results.items()},
        "cross_session": {k: float(v) for k, v in cross_results.items()},
    }

    with open(RESULTS_DIR / "advanced_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Final comparison
    print("\n" + "=" * 70)
    print("PAPER COMPARISON SUMMARY")
    print("=" * 70)
    best_merged = max(merged_results.values(), key=lambda r: r["acc"])
    best_name = max(merged_results, key=lambda m: merged_results[m]["acc"])
    print(f"""
  Phoneme classification (40 classes, chance = 2.5%):
    Paper (Naive Bayes, 128ch TX):  62.0%
    Our best ({best_name}):       {best_merged['acc']*100:.1f}%
    Improvement:                    +{(best_merged['acc'] - 0.62)*100:.1f}%

  Cross-session generalization:
    Sess1→Sess2: {cross_results['acc_12']*100:.1f}%
    Sess2→Sess1: {cross_results['acc_21']*100:.1f}%
    Mean:        {(cross_results['acc_12']+cross_results['acc_21'])/2*100:.1f}%

  Total experiment time: {(time.time() - t_start) / 60:.1f} min
""")

if __name__ == "__main__":
    main()
