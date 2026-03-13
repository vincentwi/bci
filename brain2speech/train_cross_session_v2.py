#!/usr/bin/env python3
"""
Cross-session phoneme classification v2 — with domain adaptation.

Improvements over v1:
1. Per-session z-score normalization (fit on train session, apply to both)
2. Feature alignment: match mean/variance of test session to training session
3. Mixup augmentation for better generalization
4. Time warping augmentation
5. Label smoothing
6. Combined blocks training (train on most blocks from BOTH sessions, test on held-out)
7. Multi-run averaging with proper seeds

Usage:
    python train_cross_session_v2.py --mode cross_session  # train S1→S2 and S2→S1
    python train_cross_session_v2.py --mode combined        # LOSO blocks across sessions
    python train_cross_session_v2.py --mode both            # run both
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path("/mnt/home/vincent.wilmet/docs/scripts")
PROCESSED_DIR = Path("/mnt/home/vincent.wilmet/docs/data/processed")
OUTPUT_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")

SESSION_FILES = [
    PROCESSED_DIR / "tuning_t12.2022.04.21_phonemes.npz",
    PROCESSED_DIR / "tuning_t12.2022.04.26_phonemes.npz",
]

MODEL_CONFIGS = {
    "TCN": dict(dr=0.3, hidden=128),
    "EEGNet": dict(F1=16, D=2, F2=32, kl=32, dr=0.4),
    "GRU": dict(hidden=256, n_layers=2, dr=0.3),
    "Transformer": dict(d_model=128, nhead=8, num_layers=4, dr=0.3),
}


def load_model_class(name):
    """Import model class from Stage 1."""
    import importlib.util
    scripts_config_path = SCRIPTS_DIR / "config.py"
    train_path = SCRIPTS_DIR / "train.py"

    spec_cfg = importlib.util.spec_from_file_location("scripts_config", scripts_config_path)
    scripts_config = importlib.util.module_from_spec(spec_cfg)
    orig_config = sys.modules.get("config")
    sys.modules["config"] = scripts_config
    spec_cfg.loader.exec_module(scripts_config)

    spec = importlib.util.spec_from_file_location("stage1_train", train_path)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    if orig_config is not None:
        sys.modules["config"] = orig_config
    else:
        del sys.modules["config"]

    return {
        "TCN": train_mod.TCN, "EEGNet": train_mod.EEGNet,
        "GRU": train_mod.GRUDecoder, "Transformer": train_mod.SpeechTransformer,
    }[name]


def load_sessions():
    """Load both sessions."""
    sessions = []
    for path in SESSION_FILES:
        data = np.load(path, allow_pickle=True)
        sessions.append({
            "X": data["X"],  # (N, 85, 1280)
            "y": data["y"],
            "block_ids": data["block_ids"],
            "n_classes": int(data["n_classes"]),
        })
    return sessions


def session_normalize(X_train, X_test):
    """Z-score normalize across samples, per feature channel.

    Fits statistics on X_train only, applies to both.
    Operates on shape (N, T, C) — normalizes across N and T jointly per channel.
    """
    # Reshape to (N*T, C) to compute per-channel stats
    N_tr, T, C = X_train.shape
    X_tr_flat = X_train.reshape(-1, C)
    mu = X_tr_flat.mean(axis=0, keepdims=True)  # (1, C)
    sd = X_tr_flat.std(axis=0, keepdims=True)    # (1, C)
    sd[sd < 1e-8] = 1.0

    X_train_norm = (X_train.reshape(-1, C) - mu) / sd
    X_train_norm = X_train_norm.reshape(N_tr, T, C)

    N_te = X_test.shape[0]
    X_test_norm = (X_test.reshape(-1, C) - mu) / sd
    X_test_norm = X_test_norm.reshape(N_te, T, C)

    return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32)


def coral_alignment(X_source, X_target):
    """CORAL domain adaptation: align covariance of target to source.

    Whitens target features and re-colors with source covariance.
    Lightweight version: per-channel mean/std alignment (faster).
    """
    # Simple version: match per-channel mean and std
    N_s, T, C = X_source.shape
    N_t = X_target.shape[0]

    src_flat = X_source.reshape(-1, C)
    tgt_flat = X_target.reshape(-1, C)

    mu_s, sd_s = src_flat.mean(0), src_flat.std(0)
    mu_t, sd_t = tgt_flat.mean(0), tgt_flat.std(0)

    sd_s[sd_s < 1e-8] = 1.0
    sd_t[sd_t < 1e-8] = 1.0

    # Transform: (x - mu_t) / sd_t * sd_s + mu_s
    aligned = (tgt_flat - mu_t) / sd_t * sd_s + mu_s
    return aligned.reshape(N_t, T, C).astype(np.float32)


def train_model(
    model_name, X_train, y_train, X_test, y_test,
    n_classes, gpu_id, seed=42, epochs=200, patience=30,
    val_fraction=0.15, use_mixup=True, label_smoothing=0.1,
    normalize=True, align=False, label=""
):
    """Train one model, return test predictions."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp import autocast, GradScaler
    from sklearn.model_selection import StratifiedShuffleSplit

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0")
    nc, nt = X_train.shape[2], X_train.shape[1]

    # Normalization
    if normalize:
        X_train, X_test = session_normalize(X_train.copy(), X_test.copy())
    if align:
        X_test = coral_alignment(X_train, X_test)

    # Val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(X_train, y_train))

    X_tr = torch.FloatTensor(X_train[train_idx].transpose(0, 2, 1)).to(device)
    y_tr = torch.LongTensor(y_train[train_idx]).to(device)
    X_val = torch.FloatTensor(X_train[val_idx].transpose(0, 2, 1)).to(device)
    y_val = torch.LongTensor(y_train[val_idx]).to(device)
    X_te = torch.FloatTensor(X_test.transpose(0, 2, 1)).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=len(train_ds) > 128)

    cls = load_model_class(model_name)
    kw = {**MODEL_CONFIGS[model_name], "nc": nc, "nt": nt, "nk": n_classes}
    model = cls(**kw).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler()

    best_val_acc = 0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            # Gaussian noise (always)
            xb = xb + torch.randn_like(xb) * 0.15

            # Time shift (30%)
            if np.random.random() < 0.3:
                shift = np.random.randint(-5, 6)
                xb = torch.roll(xb, shifts=shift, dims=2)

            # Channel dropout (20%, drop 10% of channels)
            if np.random.random() < 0.2:
                n_drop = max(1, int(0.1 * xb.shape[1]))
                drop_idx = np.random.choice(xb.shape[1], n_drop, replace=False)
                xb[:, drop_idx, :] = 0

            # Mixup (50%)
            if use_mixup and np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                perm = torch.randperm(xb.shape[0], device=device)
                xb = lam * xb + (1 - lam) * xb[perm]
                yb_onehot = F.one_hot(yb, n_classes).float()
                yb_perm_onehot = F.one_hot(yb[perm], n_classes).float()
                mixed_target = lam * yb_onehot + (1 - lam) * yb_perm_onehot

                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    logits = model(xb)
                    loss = -(mixed_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                continue

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Val-based early stopping
        model.eval()
        with torch.no_grad(), autocast("cuda"):
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    # Test eval with val-stopped model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), autocast("cuda"):
        test_logits = model(X_te)
        test_probs = F.softmax(test_logits, dim=1).cpu().numpy()

    test_acc = (test_probs.argmax(1) == y_test).mean()
    print(f"  [{label}] {model_name}: test={test_acc:.4f}, val={best_val_acc:.3f}, "
          f"epoch={epoch+1}")

    return {
        "y_true": y_test,
        "y_pred": test_probs.argmax(1),
        "softmax_probs": test_probs,
        "test_acc": float(test_acc),
        "val_acc": float(best_val_acc),
    }


def run_cross_session(model_name, sessions, gpu_id, n_runs=3, **train_kwargs):
    """Run bidirectional cross-session eval."""
    all_accs = []

    for run_i in range(n_runs):
        seed = 42 + run_i * 7
        results_dir = []

        for train_idx, test_idx, label in [(0, 1, "S1→S2"), (1, 0, "S2→S1")]:
            r = train_model(
                model_name,
                sessions[train_idx]["X"], sessions[train_idx]["y"],
                sessions[test_idx]["X"], sessions[test_idx]["y"],
                sessions[0]["n_classes"], gpu_id, seed=seed,
                label=f"run{run_i+1}/{label}", **train_kwargs,
            )
            results_dir.append(r)

        avg = np.mean([r["test_acc"] for r in results_dir])
        all_accs.append(avg)
        print(f"  {model_name} run {run_i+1}: avg={avg:.4f}")

    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    print(f"  {model_name} FINAL: {mean_acc:.1%} ± {std_acc:.1%}")
    return {"mean": mean_acc, "std": std_acc, "per_run": all_accs}


def run_combined_blocks(model_name, sessions, gpu_id, n_runs=3, **train_kwargs):
    """Combined LOSO-block evaluation across both sessions.

    Merge both sessions, use block IDs as fold groups.
    This is still valid as blocks are recorded minutes apart.
    """
    X = np.concatenate([s["X"] for s in sessions])
    y = np.concatenate([s["y"] for s in sessions])
    # Shift second session's block IDs to avoid collisions
    block_ids = np.concatenate([
        sessions[0]["block_ids"],
        sessions[1]["block_ids"] + 100  # offset to distinguish sessions
    ])
    n_classes = sessions[0]["n_classes"]

    # Session IDs for normalization
    session_ids = np.concatenate([
        np.zeros(len(sessions[0]["y"]), dtype=int),
        np.ones(len(sessions[1]["y"]), dtype=int),
    ])

    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    all_run_accs = []
    for run_i in range(n_runs):
        seed = 42 + run_i * 7
        np.random.seed(seed)

        # 3-fold blocked CV: hold out ~1/3 of blocks
        np.random.shuffle(unique_blocks)
        fold_size = n_blocks // 3

        fold_accs = []
        for fold_i in range(3):
            start = fold_i * fold_size
            if fold_i == 2:
                test_blocks = unique_blocks[start:]
            else:
                test_blocks = unique_blocks[start:start + fold_size]

            test_mask = np.isin(block_ids, test_blocks)
            train_mask = ~test_mask

            X_train_fold = X[train_mask]
            y_train_fold = y[train_mask]
            X_test_fold = X[test_mask]
            y_test_fold = y[test_mask]

            r = train_model(
                model_name, X_train_fold, y_train_fold,
                X_test_fold, y_test_fold, n_classes, gpu_id,
                seed=seed + fold_i,
                label=f"run{run_i+1}/fold{fold_i+1}",
                **train_kwargs,
            )
            fold_accs.append(r["test_acc"])

        avg = np.mean(fold_accs)
        all_run_accs.append(avg)
        print(f"  {model_name} combined run {run_i+1}: avg={avg:.4f}")

    mean_acc = np.mean(all_run_accs)
    std_acc = np.std(all_run_accs)
    print(f"  {model_name} COMBINED: {mean_acc:.1%} ± {std_acc:.1%}")
    return {"mean": mean_acc, "std": std_acc, "per_run": all_run_accs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["TCN", "GRU", "Transformer", "EEGNet"])
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--mode", choices=["cross_session", "combined", "both", "ablation"],
                        default="ablation")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    sessions = load_sessions()
    print("=" * 70)
    print("CROSS-SESSION EVALUATION v2 — with domain adaptation")
    print(f"  Mode: {args.mode}")
    print(f"  Models: {args.models}")
    print(f"  S1: {sessions[0]['X'].shape}, S2: {sessions[1]['X'].shape}")
    print("=" * 70)

    results = {}

    if args.mode in ("ablation", "both"):
        # Ablation study: test each improvement individually with TCN
        print("\n" + "=" * 70)
        print("ABLATION STUDY (TCN only)")
        print("=" * 70)

        configs = {
            "baseline (no norm, no mixup)": dict(normalize=False, align=False, use_mixup=False, label_smoothing=0.0),
            "session_norm": dict(normalize=True, align=False, use_mixup=False, label_smoothing=0.0),
            "session_norm + CORAL": dict(normalize=True, align=True, use_mixup=False, label_smoothing=0.0),
            "session_norm + mixup": dict(normalize=True, align=False, use_mixup=True, label_smoothing=0.0),
            "session_norm + label_smooth": dict(normalize=True, align=False, use_mixup=False, label_smoothing=0.1),
            "full (all improvements)": dict(normalize=True, align=False, use_mixup=True, label_smoothing=0.1),
        }

        for config_name, kwargs in configs.items():
            print(f"\n--- {config_name} ---")
            r = run_cross_session("TCN", sessions, args.gpu, n_runs=args.n_runs,
                                   epochs=args.epochs, **kwargs)
            results[f"ablation_{config_name}"] = r

        print("\n" + "=" * 70)
        print("ABLATION RESULTS")
        print("=" * 70)
        print(f"{'Config':<40} {'Accuracy':>12}")
        print("-" * 55)
        for name, r in results.items():
            if name.startswith("ablation_"):
                label = name.replace("ablation_", "")
                print(f"{label:<40} {r['mean']:.1%} ± {r['std']:.1%}")

    if args.mode in ("cross_session", "both"):
        print("\n" + "=" * 70)
        print("CROSS-SESSION (best config)")
        print("=" * 70)

        for model_name in args.models:
            print(f"\n--- {model_name} ---")
            r = run_cross_session(model_name, sessions, args.gpu, n_runs=args.n_runs,
                                   epochs=args.epochs,
                                   normalize=True, use_mixup=True, label_smoothing=0.1)
            results[f"cross_{model_name}"] = r

    if args.mode in ("combined", "both"):
        print("\n" + "=" * 70)
        print("COMBINED BLOCKS (both sessions, LOSO-block CV)")
        print("=" * 70)

        for model_name in args.models:
            print(f"\n--- {model_name} ---")
            r = run_combined_blocks(model_name, sessions, args.gpu, n_runs=args.n_runs,
                                     epochs=args.epochs,
                                     normalize=True, use_mixup=True, label_smoothing=0.1)
            results[f"combined_{model_name}"] = r

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"{'Method':<50} {'Accuracy':>12}")
    print("-" * 65)
    print(f"{'Willett et al. 2023 (GRU, sentence data)':<50} {'61.4%':>12}")
    print(f"{'Chance (40 classes)':<50} {'2.5%':>12}")
    print("-" * 65)
    for name, r in sorted(results.items()):
        print(f"{name:<50} {r['mean']:.1%} ± {r['std']:.1%}")
    print("=" * 80)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = RESULTS_DIR / f"cross_session_v2_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
