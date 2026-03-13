#!/usr/bin/env python3
"""
Evaluate cross-session predictions from train_cross_session.py.

Computes:
  - Per-model accuracy (S1→S2, S2→S1, average)
  - Ensemble accuracy (uniform + weighted soft voting)
  - Per-class breakdown (consonants vs vowels)
  - Confusion matrix analysis (articulatory pairs)
  - Comparison table with Willett et al. baseline

Usage:
    python evaluate_cross_session.py                     # full evaluation
    python evaluate_cross_session.py --direction s1s2    # one direction only
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_TO_ARPABET, ARPABET_TO_CLASS, N_CLASSES, VOWELS

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODELS = ["TCN", "EEGNet", "GRU", "Transformer"]


def load_cross_session_predictions(direction="s1s2"):
    """Load cross-session prediction files.

    Args:
        direction: "s1s2" (train S1, test S2) or "s2s1"

    Returns:
        y_true, model_probs dict
    """
    y_true = None
    model_probs = {}

    for name in MODELS:
        path = DATA_DIR / f"cross_session_{name.lower()}_{direction}.npz"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping {name}")
            continue

        data = np.load(path)
        probs = data["softmax_probs"]
        yt = data["y_true"]

        if y_true is None:
            y_true = yt
        else:
            assert np.array_equal(y_true, yt), f"y_true mismatch for {name}"

        acc = (data["y_pred"] == yt).mean()
        model_probs[name] = probs
        print(f"  {name} ({direction}): {len(yt)} trials, acc={acc:.1%}")

    return y_true, model_probs


def compute_metrics(y_true, probs):
    """Compute accuracy, top-k, per-class, confusion matrix."""
    from sklearn.metrics import confusion_matrix as sk_cm

    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    acc = (preds == y_true).mean()

    top3 = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = np.mean([y_true[i] in top3[i] for i in range(len(y_true))])
    top5 = np.argsort(probs, axis=1)[:, -5:]
    top5_acc = np.mean([y_true[i] in top5[i] for i in range(len(y_true))])

    per_class = {}
    for cls_idx in range(N_CLASSES):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            arpabet = CLASS_TO_ARPABET.get(cls_idx, f"CLS{cls_idx}")
            per_class[arpabet] = {
                "accuracy": float((preds[mask] == cls_idx).mean()),
                "n_samples": int(mask.sum()),
                "avg_confidence": float(confidences[mask].mean()),
            }

    cm = sk_cm(y_true, preds, labels=range(N_CLASSES))

    return {
        "accuracy": float(acc),
        "top3_accuracy": float(top3_acc),
        "top5_accuracy": float(top5_acc),
        "mean_confidence": float(confidences.mean()),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def print_results(results_by_direction):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print("CROSS-SESSION PHONEME CLASSIFICATION — PROPER EVALUATION")
    print("Train on one session, test on the other. Validation-based early stopping.")
    print("=" * 90)

    print(f"\n{'Method':<40} {'S1→S2':>10} {'S2→S1':>10} {'Average':>10}")
    print("-" * 75)
    print(f"{'Willett et al. 2023 (NB, 128ch, 39cls)':<40} {'—':>10} {'—':>10} {'61.4%':>10}")
    print(f"{'Chance (40 classes)':<40} {'2.5%':>10} {'2.5%':>10} {'2.5%':>10}")
    print("-" * 75)

    for name in MODELS + ["Ensemble_uniform", "Ensemble_weighted"]:
        s1s2_key = f"{name}_s1s2"
        s2s1_key = f"{name}_s2s1"
        if s1s2_key in results_by_direction and s2s1_key in results_by_direction:
            a1 = results_by_direction[s1s2_key]["accuracy"]
            a2 = results_by_direction[s2s1_key]["accuracy"]
            avg = (a1 + a2) / 2
            print(f"{'Our ' + name:<40} {a1:>9.1%} {a2:>9.1%} {avg:>9.1%}")

    print("=" * 90)


def print_per_class_breakdown(metrics, label=""):
    """Print per-phoneme accuracy breakdown."""
    per_class = metrics.get("per_class", {})
    if not per_class:
        return

    print(f"\nPer-phoneme accuracy ({label}, overall {metrics['accuracy']:.1%}):")

    sorted_cls = sorted(per_class.items(), key=lambda x: x[1]["accuracy"])
    print(f"  Worst 10:")
    for ph, info in sorted_cls[:10]:
        ptype = "vowel" if ph in VOWELS else "cons"
        print(f"    {ph:<6} ({ptype}): {info['accuracy']:>6.1%} (n={info['n_samples']})")

    cons_accs = [info["accuracy"] for ph, info in per_class.items()
                 if ph not in VOWELS and ph != "SIL"]
    vowel_accs = [info["accuracy"] for ph, info in per_class.items()
                  if ph in VOWELS]
    if cons_accs and vowel_accs:
        print(f"\n  Consonant avg: {np.mean(cons_accs):.1%}")
        print(f"  Vowel avg:     {np.mean(vowel_accs):.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["s1s2", "s2s1", "both"], default="both")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_all = {}

    directions = ["s1s2", "s2s1"] if args.direction == "both" else [args.direction]

    for direction in directions:
        print(f"\n{'─'*60}")
        print(f"Loading {direction} predictions...")
        y_true, model_probs = load_cross_session_predictions(direction)

        if y_true is None or not model_probs:
            print(f"  No predictions found for {direction}")
            continue

        # Individual models
        for name, probs in model_probs.items():
            metrics = compute_metrics(y_true, probs)
            key = f"{name}_{direction}"
            results_all[key] = metrics
            print(f"  {name}: {metrics['accuracy']:.1%} "
                  f"(top3={metrics['top3_accuracy']:.1%})")

        # Ensemble
        if len(model_probs) >= 2:
            names = list(model_probs.keys())

            # Uniform
            ens_probs = np.mean(list(model_probs.values()), axis=0)
            m = compute_metrics(y_true, ens_probs)
            results_all[f"Ensemble_uniform_{direction}"] = m
            print(f"  Ensemble (uniform): {m['accuracy']:.1%}")

            # Weighted by individual accuracy
            weights = {n: results_all[f"{n}_{direction}"]["accuracy"] for n in names}
            total_w = sum(weights.values())
            ens_w = sum(model_probs[n] * (weights[n] / total_w) for n in names)
            m_w = compute_metrics(y_true, ens_w)
            results_all[f"Ensemble_weighted_{direction}"] = m_w
            print(f"  Ensemble (weighted): {m_w['accuracy']:.1%}")

    if len(directions) == 2:
        print_results(results_all)

        # Per-class for best ensemble direction
        best_key = max(
            (k for k in results_all if "Ensemble" in k),
            key=lambda k: results_all[k]["accuracy"],
            default=None,
        )
        if best_key:
            print_per_class_breakdown(results_all[best_key], best_key)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = RESULTS_DIR / f"cross_session_eval_{timestamp}.json"
    serializable = {}
    for k, v in results_all.items():
        serializable[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                           for kk, vv in v.items()}
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
