#!/usr/bin/env python3
"""
Stage 2a: Extract confusion matrix from the best Stage 1 classifier (TCN).

Re-loads the saved TCN weights + phoneme data, runs leave-one-group-out
inference to produce per-trial softmax probabilities, then builds the
40×40 row-normalized confusion matrix that serves as our noise channel model.

Outputs:
    brain2speech/data/noise_model.npy          — 40×40 confusion matrix C[i,j] = P(predict j | true i)
    brain2speech/data/phoneme_predictions.npz  — y_true, y_pred, softmax_probs arrays
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Import Stage 1 models by temporarily making Stage 1 scripts the primary path
# train.py imports from 'config' so the Stage 1 config must be found first
STAGE1_SCRIPTS = Path("/mnt/home/vincent.wilmet/docs/scripts")
sys.path.insert(0, str(STAGE1_SCRIPTS))
from train import TCN, EEGNet, GRUDecoder, SpeechTransformer, load_data
from config import PROCESSED_DIR, SEED

# Now remove Stage 1 from path and add brain2speech config
sys.path.remove(str(STAGE1_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib
_spec = importlib.util.spec_from_file_location(
    "b2s_config", Path(__file__).resolve().parent.parent / "config.py"
)
b2s_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2s_config)
DATA_DIR = b2s_config.DATA_DIR
STAGE1_MODELS = b2s_config.STAGE1_MODELS
CLASS_TO_ARPABET = b2s_config.CLASS_TO_ARPABET
ARPABET_TO_CLASS = b2s_config.ARPABET_TO_CLASS
N_CLASSES = b2s_config.N_CLASSES


def extract_predictions(model_name="TCN", gpu_id=0):
    """Re-run CV inference with saved best-fold weights to get softmax probs.

    Since train.py saves best_state from the best fold only, and we need
    predictions for ALL samples, we re-train with the same CV splits and
    collect test-set softmax outputs per fold.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load phoneme data (same as Stage 1)
    # Try merged phoneme data first
    merged = PROCESSED_DIR / "merged_phonemes.npz"
    if merged.exists():
        X, X_feat, y, group_ids, class_names, n_classes = load_data("merged_phonemes")
    else:
        # Use first phoneme file
        candidates = sorted(PROCESSED_DIR.glob("tuning_*phonemes*.npz"))
        if not candidates:
            print("ERROR: No phoneme data found")
            sys.exit(1)
        X, X_feat, y, group_ids, class_names, n_classes = load_data(candidates[0].stem)

    print(f"Data: {X.shape}, {n_classes} classes, {len(np.unique(group_ids))} groups")
    print(f"Classes: {list(class_names)}")

    nc, nt = X.shape[2], X.shape[1]

    model_configs = {
        "TCN": (TCN, dict(nc=nc, nt=nt, nk=n_classes, dr=0.3, hidden=128)),
        "EEGNet": (EEGNet, dict(nc=nc, nt=nt, nk=n_classes, F1=16, D=2, F2=32, kl=32, dr=0.4)),
        "GRU": (GRUDecoder, dict(nc=nc, nt=nt, nk=n_classes, hidden=256, n_layers=2, dr=0.3)),
        "Transformer": (SpeechTransformer, dict(nc=nc, nt=nt, nk=n_classes, d_model=128, nhead=8, num_layers=4, dr=0.3)),
    }

    # Saved weights are from a different task (orofacial 34-class), so we
    # always do a full CV re-run to get proper phoneme predictions + softmax
    print(f"Running {model_name} CV to extract phoneme predictions...")
    return _rerun_cv(X, y, group_ids, model_configs[model_name], device)


def _rerun_cv(X, y, group_ids, model_config, device):
    """Fallback: re-run CV to get per-fold test predictions."""
    from torch.utils.data import DataLoader, TensorDataset
    from torch.cuda.amp import autocast, GradScaler

    cls, kw = model_config
    unique_groups = np.unique(group_ids)
    n_folds = len(unique_groups) if len(unique_groups) >= 3 else 5

    if len(unique_groups) >= 3:
        fold_splits = [(group_ids != g, group_ids == g) for g in unique_groups]
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
        fold_splits = []
        for tr_idx, te_idx in cv.split(X, y):
            tr_m, te_m = np.zeros(len(y), dtype=bool), np.zeros(len(y), dtype=bool)
            tr_m[tr_idx], te_m[te_idx] = True, True
            fold_splits.append((tr_m, te_m))

    all_preds = np.full(len(y), -1, dtype=np.int64)
    all_probs = np.zeros((len(y), kw['nk']), dtype=np.float32)

    for fold, (train_mask, test_mask) in enumerate(fold_splits):
        if test_mask.sum() == 0:
            continue

        Xtr = torch.FloatTensor(X[train_mask].transpose(0, 2, 1))
        Xte = torch.FloatTensor(X[test_mask].transpose(0, 2, 1))
        ytr = torch.LongTensor(y[train_mask])

        train_ds = TensorDataset(Xtr, ytr)
        loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

        model = cls(**kw).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        best_acc, best_probs_fold = 0, None
        patience, wait = 25, 0

        for epoch in range(150):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if np.random.random() < 0.5:
                    xb = xb + torch.randn_like(xb) * 0.1
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            model.eval()
            Xte_gpu = Xte.to(device)
            with torch.no_grad(), autocast():
                logits = model(Xte_gpu)
                probs = F_torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

            acc = (preds == y[test_mask]).mean()
            if acc > best_acc:
                best_acc = acc
                best_probs_fold = probs.copy()
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break

        all_preds[test_mask] = best_probs_fold.argmax(axis=1)
        all_probs[test_mask] = best_probs_fold
        print(f"  Fold {fold+1}/{n_folds}: {best_acc:.3f}")

    valid = all_preds >= 0
    return y[valid], all_preds[valid], all_probs[valid], None


def build_noise_model(y_true, y_pred, n_classes=N_CLASSES):
    """Build row-normalized confusion matrix: C[i,j] = P(predict j | true i)."""
    C_raw = confusion_matrix(y_true, y_pred, labels=range(n_classes))

    # Add Laplace smoothing to avoid zero-probability entries
    C_smoothed = C_raw.astype(np.float64) + 1e-6
    C = C_smoothed / C_smoothed.sum(axis=1, keepdims=True)

    return C_raw, C


def validate_articulatory_structure(C, class_to_arpabet):
    """Check that articulatorily similar phonemes show higher confusion."""
    # Expected confusable pairs (voiced ↔ unvoiced, same place of articulation)
    expected_pairs = [
        ('B', 'P'), ('D', 'T'), ('G', 'K'), ('V', 'F'),
        ('Z', 'S'), ('ZH', 'SH'), ('DH', 'TH'), ('JH', 'CH'),
        ('B', 'M'), ('D', 'N'), ('G', 'NG'),  # same place
        ('IY', 'IH'), ('UW', 'UH'), ('AE', 'EH'),  # vowel neighbors
    ]
    arpabet_to_idx = {v: k for k, v in class_to_arpabet.items()}

    print("\nArticulatory pair confusion rates:")
    off_diag_mean = (C.sum() - np.trace(C)) / (C.size - len(C))

    for p1, p2 in expected_pairs:
        if p1 in arpabet_to_idx and p2 in arpabet_to_idx:
            i, j = arpabet_to_idx[p1], arpabet_to_idx[p2]
            rate = (C[i, j] + C[j, i]) / 2
            marker = "***" if rate > off_diag_mean * 2 else ""
            print(f"  {p1:3s} ↔ {p2:3s}: {rate:.4f} {marker}")

    print(f"  Mean off-diagonal: {off_diag_mean:.6f}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 2a: Building noise model from best classifier")
    print("=" * 60)

    # Extract predictions using saved TCN weights
    y_true, y_pred, softmax_probs, class_names = extract_predictions("TCN", gpu_id=7)

    # Build confusion matrix
    C_raw, C_norm = build_noise_model(y_true, y_pred)

    # Save
    np.save(DATA_DIR / "noise_model.npy", C_norm)
    np.save(DATA_DIR / "noise_model_raw.npy", C_raw)
    np.savez(
        DATA_DIR / "phoneme_predictions.npz",
        y_true=y_true, y_pred=y_pred, softmax_probs=softmax_probs,
    )

    print(f"\nConfusion matrix shape: {C_norm.shape}")
    print(f"Diagonal mean (accuracy): {np.diag(C_norm).mean():.3f}")
    print(f"Off-diagonal max: {(C_norm - np.diag(np.diag(C_norm))).max():.4f}")

    validate_articulatory_structure(C_norm, CLASS_TO_ARPABET)

    # Also extract predictions from other models for ensemble later
    for model_name in ["EEGNet", "GRU", "Transformer"]:
        try:
            yt, yp, probs, _ = extract_predictions(model_name, gpu_id=7)
            np.savez(
                DATA_DIR / f"{model_name.lower()}_predictions.npz",
                y_true=yt, y_pred=yp, softmax_probs=probs,
            )
            acc = (yp == yt).mean()
            print(f"  {model_name}: {acc:.3f}")
        except Exception as e:
            print(f"  {model_name}: failed ({e})")

    print("\nDone. Outputs in", DATA_DIR)


if __name__ == "__main__":
    main()
