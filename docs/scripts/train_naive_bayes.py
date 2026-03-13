#!/usr/bin/env python3
"""
Reproduce the paper's Naive Bayes baseline for phoneme classification.

Willett et al. (Nature 2023) reported ~62% accuracy on 39 phoneme classes
using Gaussian Naive Bayes on threshold crossing (TX) features.

We run GNB on:
  1. Single phoneme session (Apr 26) — 800 trials, 40 classes
  2. Single phoneme session (Apr 21) — 640 trials, 40 classes
  3. Merged phoneme sessions — 1440 trials, 40 classes
  4. 50-word vocabulary — 1020 trials, 51 classes
  5. Orofacial — 680 trials, 34 classes

Uses Stratified K-Fold CV to match the paper's evaluation protocol.
Also runs on GPU features (PCA-reduced) for fair comparison.
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

# No GPU needed for Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import PROCESSED_DIR, MODELS_DIR, SEED

np.random.seed(SEED)

RESULTS_DIR = MODELS_DIR / "naive_bayes"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_naive_bayes(X_feat, y, class_names, n_classes, dataset_name,
                    n_folds=10, n_pca_list=[None, 30, 60, 120]):
    """Run Gaussian Naive Bayes with various PCA dimensions."""
    print(f"\n{'='*70}")
    print(f"NAIVE BAYES: {dataset_name}")
    print(f"  {X_feat.shape[0]} trials, {n_classes} classes, "
          f"{X_feat.shape[1]} features, chance={1/n_classes*100:.1f}%")
    print(f"{'='*70}")

    cv = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    results = {}

    for n_pca in n_pca_list:
        name = f"GNB-PCA{n_pca}" if n_pca else "GNB-raw"

        if n_pca and n_pca >= X_feat.shape[1]:
            continue

        if n_pca:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_pca, random_state=SEED)),
                ("gnb", GaussianNB()),
            ])
        else:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("gnb", GaussianNB()),
            ])

        t0 = time.time()
        fold_accs = []
        all_preds = np.full(len(y), -1)

        for fold, (tr_idx, te_idx) in enumerate(cv.split(X_feat, y)):
            pipe.fit(X_feat[tr_idx], y[tr_idx])
            preds = pipe.predict(X_feat[te_idx])
            acc = accuracy_score(y[te_idx], preds)
            fold_accs.append(acc)
            all_preds[te_idx] = preds

        overall = np.mean(fold_accs)
        elapsed = time.time() - t0
        results[name] = {
            "acc": overall,
            "fold_accs": fold_accs,
            "preds": all_preds,
        }

        folds_str = " ".join(f"{a:.3f}" for a in fold_accs)
        print(f"  {name}: {overall:.3f} [{folds_str}] ({elapsed:.1f}s)")

    # Best model confusion matrix
    best_name = max(results, key=lambda m: results[m]["acc"])
    best = results[best_name]
    valid = best["preds"] >= 0
    cm = confusion_matrix(y[valid], best["preds"][valid])

    # Save
    safe_name = dataset_name.replace(" ", "_").replace("/", "_")
    np.save(RESULTS_DIR / f"{safe_name}_confusion.npy", cm)
    np.save(RESULTS_DIR / f"{safe_name}_preds.npy", best["preds"])

    print(f"\n  Best: {best_name} = {best['acc']:.3f}")

    # Per-class accuracy for best
    yt, yp = y[valid], best["preds"][valid]
    per_class = []
    for c in range(n_classes):
        mask = yt == c
        if mask.sum() > 0:
            pc_acc = (yp[mask] == c).mean()
            per_class.append((class_names[c], pc_acc))

    print(f"  Top 5 classes:")
    for name_c, acc in sorted(per_class, key=lambda x: -x[1])[:5]:
        print(f"    {name_c}: {acc:.3f}")
    print(f"  Bottom 5 classes:")
    for name_c, acc in sorted(per_class, key=lambda x: x[1])[:5]:
        print(f"    {name_c}: {acc:.3f}")

    return results


def main():
    t_start = time.time()
    all_results = {}

    # 1. Phoneme session Apr 26 (800 trials, 40 classes)
    d = np.load(PROCESSED_DIR / "tuning_t12.2022.04.26_phonemes.npz", allow_pickle=True)
    r = run_naive_bayes(d["X_feat"], d["y"], list(d["class_names"]),
                        int(d["n_classes"]), "phonemes_apr26")
    all_results["phonemes_apr26"] = {k: {"acc": v["acc"], "folds": v["fold_accs"]}
                                     for k, v in r.items()}

    # 2. Phoneme session Apr 21 (640 trials, 40 classes)
    d2 = np.load(PROCESSED_DIR / "tuning_t12.2022.04.21_phonemes.npz", allow_pickle=True)
    r = run_naive_bayes(d2["X_feat"], d2["y"], list(d2["class_names"]),
                        int(d2["n_classes"]), "phonemes_apr21")
    all_results["phonemes_apr21"] = {k: {"acc": v["acc"], "folds": v["fold_accs"]}
                                     for k, v in r.items()}

    # 3. Merged phoneme sessions
    X_merged = np.concatenate([d["X_feat"], d2["X_feat"]])
    y_merged = np.concatenate([d["y"], d2["y"]])
    r = run_naive_bayes(X_merged, y_merged, list(d["class_names"]),
                        int(d["n_classes"]), "phonemes_merged")
    all_results["phonemes_merged"] = {k: {"acc": v["acc"], "folds": v["fold_accs"]}
                                       for k, v in r.items()}

    # 4. 50-word vocabulary
    d50 = np.load(PROCESSED_DIR / "tuning_t12.2022.05.03_fiftyWordSet.npz", allow_pickle=True)
    r = run_naive_bayes(d50["X_feat"], d50["y"], list(d50["class_names"]),
                        int(d50["n_classes"]), "fiftyword")
    all_results["fiftyword"] = {k: {"acc": v["acc"], "folds": v["fold_accs"]}
                                 for k, v in r.items()}

    # 5. Orofacial
    doro = np.load(PROCESSED_DIR / "tuning_t12.2022.04.21_orofacial.npz", allow_pickle=True)
    r = run_naive_bayes(doro["X_feat"], doro["y"], list(doro["class_names"]),
                        int(doro["n_classes"]), "orofacial")
    all_results["orofacial"] = {k: {"acc": v["acc"], "folds": v["fold_accs"]}
                                 for k, v in r.items()}

    # Summary
    print("\n" + "=" * 70)
    print("NAIVE BAYES SUMMARY — PAPER COMPARISON")
    print("=" * 70)
    print(f"\n  Paper baseline: ~62% phoneme accuracy (GNB, 39 classes, 128ch TX features)")
    print(f"  Paper features: threshold crossings only (128 channels)")
    print(f"  Our features: spikePow + TX (256 channels, 1280 features/bin)\n")

    for ds_name, ds_results in all_results.items():
        best_name = max(ds_results, key=lambda m: ds_results[m]["acc"])
        best_acc = ds_results[best_name]["acc"]
        print(f"  {ds_name:25s}: {best_name} = {best_acc:.3f} ({best_acc*100:.1f}%)")

    # Save summary
    import json
    # Convert numpy types for JSON serialization
    def to_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary = {}
    for ds_name, ds_results in all_results.items():
        summary[ds_name] = {}
        for model_name, model_data in ds_results.items():
            summary[ds_name][model_name] = {
                "acc": to_json(model_data["acc"]),
                "folds": [to_json(f) for f in model_data["folds"]],
            }

    with open(RESULTS_DIR / "naive_bayes_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR}")
    print(f"  Total time: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
