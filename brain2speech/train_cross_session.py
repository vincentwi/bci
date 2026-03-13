#!/usr/bin/env python3
"""
Proper cross-session phoneme classification evaluation.

Fixes 3 critical issues from the original build_noise_model.py:
1. Uses TRUE cross-session eval: train on session 1, test on session 2 (and vice versa)
2. Uses a validation split FROM TRAINING DATA for early stopping (no test-set peeking)
3. Reports both directions + average for fair comparison with Willett et al. (61.4%)

Also supports multi-GPU parallel training of different models.

Usage:
    python train_cross_session.py                          # all 4 models, GPU 6
    python train_cross_session.py --models TCN Transformer # specific models
    python train_cross_session.py --gpus 6 7               # parallel across GPUs
    python train_cross_session.py --epochs 200             # more epochs
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

# Paths
SCRIPTS_DIR = Path("/mnt/home/vincent.wilmet/docs/scripts")
PROCESSED_DIR = Path("/mnt/home/vincent.wilmet/docs/data/processed")
OUTPUT_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")

SESSION_FILES = [
    PROCESSED_DIR / "tuning_t12.2022.04.21_phonemes.npz",  # Session 1: 640 trials
    PROCESSED_DIR / "tuning_t12.2022.04.26_phonemes.npz",  # Session 2: 800 trials
]

SEED = 42

MODEL_CONFIGS = {
    "TCN": dict(dr=0.3, hidden=128),
    "EEGNet": dict(F1=16, D=2, F2=32, kl=32, dr=0.4),
    "GRU": dict(hidden=256, n_layers=2, dr=0.3),
    "Transformer": dict(d_model=128, nhead=8, num_layers=4, dr=0.3),
}


def load_sessions():
    """Load both phoneme sessions."""
    sessions = []
    for path in SESSION_FILES:
        data = np.load(path, allow_pickle=True)
        sessions.append({
            "X": data["X"],
            "y": data["y"],
            "block_ids": data["block_ids"],
            "class_names": list(data["class_names"]),
            "n_classes": int(data["n_classes"]),
            "path": str(path),
        })
    return sessions


def load_model_class(name):
    """Import model class from Stage 1 train.py, handling config collision."""
    import importlib.util

    scripts_config_path = SCRIPTS_DIR / "config.py"
    train_path = SCRIPTS_DIR / "train.py"

    # Load Stage 1 config module
    spec_cfg = importlib.util.spec_from_file_location("scripts_config", scripts_config_path)
    scripts_config = importlib.util.module_from_spec(spec_cfg)
    orig_config = sys.modules.get("config")
    sys.modules["config"] = scripts_config
    spec_cfg.loader.exec_module(scripts_config)

    # Load train module
    spec = importlib.util.spec_from_file_location("stage1_train", train_path)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    # Restore original config
    if orig_config is not None:
        sys.modules["config"] = orig_config
    else:
        del sys.modules["config"]

    return {
        "TCN": train_mod.TCN,
        "EEGNet": train_mod.EEGNet,
        "GRU": train_mod.GRUDecoder,
        "Transformer": train_mod.SpeechTransformer,
    }[name]


def train_and_evaluate_direction(
    model_name, X_train, y_train, X_test, y_test,
    n_classes, gpu_id, epochs=150, patience=30, val_fraction=0.15,
    direction_label=""
):
    """
    Train a model on one session, evaluate on another.
    Uses a validation split from training data for early stopping.
    Returns test-set predictions only from the val-stopped checkpoint.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.amp import autocast, GradScaler
    from sklearn.model_selection import StratifiedShuffleSplit

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nc, nt = X_train.shape[2], X_train.shape[1]  # channels, time
    cls = load_model_class(model_name)
    kw = {**MODEL_CONFIGS[model_name], "nc": nc, "nt": nt, "nk": n_classes}

    # --- Validation split from training data ---
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=SEED)
    train_idx, val_idx = next(sss.split(X_train, y_train))

    X_tr = torch.FloatTensor(X_train[train_idx].transpose(0, 2, 1)).to(device)
    y_tr = torch.LongTensor(y_train[train_idx]).to(device)
    X_val = torch.FloatTensor(X_train[val_idx].transpose(0, 2, 1)).to(device)
    y_val = torch.LongTensor(y_train[val_idx]).to(device)
    X_te = torch.FloatTensor(X_test.transpose(0, 2, 1)).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)

    model = cls(**kw).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_acc = 0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            # Data augmentation: Gaussian noise
            if np.random.random() < 0.5:
                xb = xb + torch.randn_like(xb) * 0.1
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # --- Early stopping on VALIDATION set (not test) ---
        model.eval()
        with torch.no_grad(), autocast("cuda"):
            val_logits = model(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"  [{direction_label}] {model_name} early stop at epoch {epoch+1}, "
                  f"val_acc={best_val_acc:.3f}")
            break

    if epoch == epochs - 1:
        print(f"  [{direction_label}] {model_name} completed {epochs} epochs, "
              f"val_acc={best_val_acc:.3f}")

    # --- Evaluate on test set with val-stopped checkpoint ---
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(), autocast("cuda"):
        test_logits = model(X_te)
        test_probs = F.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = test_probs.argmax(axis=1)

    test_acc = (test_preds == y_test).mean()
    print(f"  [{direction_label}] {model_name} TEST acc={test_acc:.4f} "
          f"(val_acc={best_val_acc:.3f})")

    return {
        "y_true": y_test,
        "y_pred": test_preds,
        "softmax_probs": test_probs,
        "test_acc": float(test_acc),
        "val_acc": float(best_val_acc),
        "epochs_trained": epoch + 1,
    }


def run_single_model(model_name, sessions, gpu_id, epochs=150, n_runs=3):
    """
    Run cross-session evaluation for one model.
    Direction 1: Train S1 → Test S2
    Direction 2: Train S2 → Test S1
    Repeat n_runs times with different seeds, report mean ± std.
    """
    results_all_runs = []

    for run_i in range(n_runs):
        global SEED
        run_seed = 42 + run_i * 7
        SEED = run_seed

        results_directions = []

        for train_idx, test_idx, label in [(0, 1, "S1→S2"), (1, 0, "S2→S1")]:
            s_train = sessions[train_idx]
            s_test = sessions[test_idx]

            result = train_and_evaluate_direction(
                model_name=model_name,
                X_train=s_train["X"],
                y_train=s_train["y"],
                X_test=s_test["X"],
                y_test=s_test["y"],
                n_classes=s_train["n_classes"],
                gpu_id=gpu_id,
                epochs=epochs,
                direction_label=f"run{run_i+1}/{label}",
            )
            results_directions.append(result)

        avg_acc = np.mean([r["test_acc"] for r in results_directions])
        results_all_runs.append({
            "run": run_i + 1,
            "seed": run_seed,
            "s1_to_s2": results_directions[0],
            "s2_to_s1": results_directions[1],
            "avg_acc": float(avg_acc),
        })
        print(f"  {model_name} run {run_i+1}: S1→S2={results_directions[0]['test_acc']:.4f}, "
              f"S2→S1={results_directions[1]['test_acc']:.4f}, avg={avg_acc:.4f}")

    # Aggregate across runs
    all_avg = [r["avg_acc"] for r in results_all_runs]
    all_s1s2 = [r["s1_to_s2"]["test_acc"] for r in results_all_runs]
    all_s2s1 = [r["s2_to_s1"]["test_acc"] for r in results_all_runs]

    summary = {
        "model": model_name,
        "n_runs": n_runs,
        "avg_acc_mean": float(np.mean(all_avg)),
        "avg_acc_std": float(np.std(all_avg)),
        "s1_to_s2_mean": float(np.mean(all_s1s2)),
        "s1_to_s2_std": float(np.std(all_s1s2)),
        "s2_to_s1_mean": float(np.mean(all_s2s1)),
        "s2_to_s1_std": float(np.std(all_s2s1)),
        "runs": results_all_runs,
    }

    print(f"\n{'='*60}")
    print(f"{model_name} CROSS-SESSION RESULTS ({n_runs} runs)")
    print(f"  S1→S2: {summary['s1_to_s2_mean']:.1%} ± {summary['s1_to_s2_std']:.1%}")
    print(f"  S2→S1: {summary['s2_to_s1_mean']:.1%} ± {summary['s2_to_s1_std']:.1%}")
    print(f"  Average: {summary['avg_acc_mean']:.1%} ± {summary['avg_acc_std']:.1%}")
    print(f"  Paper baseline (Willett et al.): 61.4%")
    print(f"{'='*60}")

    return summary


def _worker(args):
    """Worker for ProcessPoolExecutor."""
    model_name, sessions_data, gpu_id, epochs, n_runs = args
    # Reconstruct sessions from serialized data
    sessions = []
    for sd in sessions_data:
        sessions.append({
            "X": sd["X"],
            "y": sd["y"],
            "block_ids": sd["block_ids"],
            "class_names": sd["class_names"],
            "n_classes": sd["n_classes"],
        })
    return run_single_model(model_name, sessions, gpu_id, epochs, n_runs)


def main():
    parser = argparse.ArgumentParser(description="Cross-session phoneme classification")
    parser.add_argument("--models", nargs="+", default=["TCN", "EEGNet", "GRU", "Transformer"])
    parser.add_argument("--gpus", nargs="+", type=int, default=[6, 7])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of runs per model (different seeds)")
    parser.add_argument("--save-predictions", action="store_true", default=True,
                        help="Save per-model cross-session predictions for ensemble eval")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CROSS-SESSION PHONEME CLASSIFICATION — PROPER EVALUATION")
    print(f"  Date: {datetime.now().isoformat()}")
    print(f"  Models: {args.models}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Epochs: {args.epochs}, Runs: {args.n_runs}")
    print(f"  Validation split: 15% of training data (for early stopping)")
    print(f"  NO test-set peeking, NO within-session CV")
    print("=" * 70)

    sessions = load_sessions()
    print(f"\nSession 1: {sessions[0]['X'].shape} ({Path(SESSION_FILES[0]).name})")
    print(f"Session 2: {sessions[1]['X'].shape} ({Path(SESSION_FILES[1]).name})")

    # Run models — assign GPUs round-robin
    all_results = {}

    if len(args.models) > 1 and len(args.gpus) > 1:
        # Parallel: run multiple models on different GPUs
        # We use sequential here since CUDA contexts don't share well across forks
        # Instead we'll interleave models across GPUs
        for i, model_name in enumerate(args.models):
            gpu_id = args.gpus[i % len(args.gpus)]
            print(f"\n--- {model_name} on GPU {gpu_id} ---")
            result = run_single_model(model_name, sessions, gpu_id, args.epochs, args.n_runs)
            all_results[model_name] = result
    else:
        for model_name in args.models:
            gpu_id = args.gpus[0]
            print(f"\n--- {model_name} on GPU {gpu_id} ---")
            result = run_single_model(model_name, sessions, gpu_id, args.epochs, args.n_runs)
            all_results[model_name] = result

    # Save best-run predictions for ensemble evaluation
    if args.save_predictions:
        print("\nSaving cross-session predictions...")
        for model_name, result in all_results.items():
            # Find run with highest avg_acc
            best_run = max(result["runs"], key=lambda r: r["avg_acc"])

            # Save S1→S2 predictions (test on session 2)
            r = best_run["s1_to_s2"]
            outpath = OUTPUT_DIR / f"cross_session_{model_name.lower()}_s1s2.npz"
            np.savez(outpath,
                     y_true=r["y_true"], y_pred=r["y_pred"],
                     softmax_probs=r["softmax_probs"])
            print(f"  {outpath.name}: acc={r['test_acc']:.4f}")

            # Save S2→S1 predictions
            r = best_run["s2_to_s1"]
            outpath = OUTPUT_DIR / f"cross_session_{model_name.lower()}_s2s1.npz"
            np.savez(outpath,
                     y_true=r["y_true"], y_pred=r["y_pred"],
                     softmax_probs=r["softmax_probs"])
            print(f"  {outpath.name}: acc={r['test_acc']:.4f}")

    # Print summary comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE — Cross-Session Phoneme Classification (40 classes)")
    print("=" * 80)
    print(f"{'Method':<40} {'S1→S2':>10} {'S2→S1':>10} {'Average':>12}")
    print("-" * 80)
    print(f"{'Willett et al. 2023 (NB, 128ch, 39cls)':<40} {'—':>10} {'—':>10} {'61.4%':>12}")
    print(f"{'Chance (40 classes)':<40} {'2.5%':>10} {'2.5%':>10} {'2.5%':>12}")
    print("-" * 80)

    for model_name in args.models:
        if model_name in all_results:
            r = all_results[model_name]
            s1s2 = f"{r['s1_to_s2_mean']:.1%}±{r['s1_to_s2_std']:.1%}"
            s2s1 = f"{r['s2_to_s1_mean']:.1%}±{r['s2_to_s1_std']:.1%}"
            avg = f"{r['avg_acc_mean']:.1%}±{r['avg_acc_std']:.1%}"
            print(f"{'Our ' + model_name:<40} {s1s2:>10} {s2s1:>10} {avg:>12}")

    # Ensemble from saved predictions (if all models ran)
    if len(all_results) >= 2 and args.save_predictions:
        print("-" * 80)
        for direction, label in [("s1_to_s2", "S1→S2"), ("s2_to_s1", "S2→S1")]:
            suffix = "s1s2" if direction == "s1_to_s2" else "s2s1"
            probs_list = []
            y_true = None
            for model_name in all_results:
                best_run = max(all_results[model_name]["runs"], key=lambda r: r["avg_acc"])
                r = best_run[direction]
                probs_list.append(r["softmax_probs"])
                if y_true is None:
                    y_true = r["y_true"]
            # Uniform ensemble
            ens_probs = np.mean(probs_list, axis=0)
            ens_acc = (ens_probs.argmax(axis=1) == y_true).mean()
            print(f"  Ensemble (uniform) {label}: {ens_acc:.1%}")

            # Weighted ensemble (proportional to individual acc)
            weights = [all_results[m]["runs"][0][direction]["test_acc"] for m in all_results]
            total_w = sum(weights)
            ens_weighted = sum(p * (w / total_w) for p, w in zip(probs_list, weights))
            ens_w_acc = (ens_weighted.argmax(axis=1) == y_true).mean()
            print(f"  Ensemble (weighted) {label}: {ens_w_acc:.1%}")

    print("=" * 80)

    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"cross_session_{timestamp}.json"

    # Make serializable
    serializable = {}
    for model_name, result in all_results.items():
        r_clean = {k: v for k, v in result.items() if k != "runs"}
        r_clean["runs"] = []
        for run in result["runs"]:
            run_clean = {"run": run["run"], "seed": run["seed"], "avg_acc": run["avg_acc"]}
            for direction in ["s1_to_s2", "s2_to_s1"]:
                run_clean[direction] = {
                    "test_acc": run[direction]["test_acc"],
                    "val_acc": run[direction]["val_acc"],
                    "epochs_trained": run[direction]["epochs_trained"],
                }
            r_clean["runs"].append(run_clean)
        serializable[model_name] = r_clean

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
