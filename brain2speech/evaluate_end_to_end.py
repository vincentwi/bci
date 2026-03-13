#!/usr/bin/env python3
"""
Comprehensive end-to-end evaluation of the brain-to-speech pipeline.

Uses SAVED predictions from Stage 1 classifiers (pre-computed softmax
probabilities), then evaluates ensemble voting and LM correction.

Produces a comparison table for the writeup:
    - Paper baseline (Willett et al. 2023): ~61.4%
    - Individual classifiers (TCN, EEGNet, GRU, Transformer)
    - Ensemble (weighted soft voting)
    - Each + LM correction at varying confidence thresholds

Outputs:
    - JSON results file with all metrics
    - Per-model confusion matrices
    - Comparison table (printed + saved)
    - Per-phoneme accuracy breakdown (consonants vs vowels)

Usage:
    python evaluate_end_to_end.py                          # full evaluation
    python evaluate_end_to_end.py --skip-lm                # classifier-only
    python evaluate_end_to_end.py --with-audio --n-audio 5 # include TTS round-trip
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLASS_TO_ARPABET, ARPABET_TO_CLASS, STAGE1_MODELS, STAGE1_DATA,
    DATA_DIR, MODELS_DIR, N_CLASSES, ARPABET_39, VOWELS,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ─────────────────────────────────────────────────────────────────
# Load saved predictions
# ─────────────────────────────────────────────────────────────────

# Mapping from model name to saved prediction file
PREDICTION_FILES = {
    "TCN": DATA_DIR / "phoneme_predictions.npz",      # TCN predictions
    "EEGNet": DATA_DIR / "eegnet_predictions.npz",
    "GRU": DATA_DIR / "gru_predictions.npz",
    "Transformer": DATA_DIR / "transformer_predictions.npz",
}


def load_saved_predictions():
    """Load all saved prediction files from Stage 1 classifiers.

    Each .npz contains:
        y_true: (N,) true labels
        y_pred: (N,) argmax predictions
        softmax_probs: (N, 40) softmax probabilities

    Returns:
        y_true: (N,) ground truth (same across all models)
        model_probs: dict of {model_name: (N, 40) probabilities}
    """
    y_true = None
    model_probs = {}

    for name, path in PREDICTION_FILES.items():
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {name}")
            continue

        data = np.load(path, allow_pickle=True)
        probs = data['softmax_probs']
        yt = data['y_true']

        if y_true is None:
            y_true = yt
        else:
            assert np.array_equal(y_true, yt), \
                f"y_true mismatch between models! {name} has different labels."

        model_probs[name] = probs
        acc = (data['y_pred'] == yt).mean()
        print(f"  Loaded {name}: {len(yt)} trials, acc={acc:.1%}")

    return y_true, model_probs


# ─────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────

def ensemble_predictions(model_probs, weights=None):
    """Weighted average of softmax probabilities.

    Args:
        model_probs: dict of {name: (N, 40) probabilities}
        weights: dict of {name: float} (default: accuracy-proportional)

    Returns:
        (N, 40) ensemble probabilities
    """
    names = list(model_probs.keys())
    if weights is None:
        # Uniform
        weights = {n: 1.0 for n in names}

    total_w = sum(weights[n] for n in names)
    return sum(model_probs[n] * (weights[n] / total_w) for n in names)


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true, probs, label=""):
    """Compute comprehensive classification metrics.

    Returns dict with accuracy, top-k, per-class, confidence stats.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    acc = (preds == y_true).mean()
    top3 = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = np.mean([y_true[i] in top3[i] for i in range(len(y_true))])
    top5 = np.argsort(probs, axis=1)[:, -5:]
    top5_acc = np.mean([y_true[i] in top5[i] for i in range(len(y_true))])

    # Per-class accuracy
    per_class = {}
    for cls_idx in range(N_CLASSES):
        mask = y_true == cls_idx
        if mask.sum() > 0:
            cls_acc = (preds[mask] == cls_idx).mean()
            arpabet = CLASS_TO_ARPABET.get(cls_idx, f"CLS{cls_idx}")
            per_class[arpabet] = {
                "accuracy": float(cls_acc),
                "n_samples": int(mask.sum()),
                "avg_confidence": float(confidences[mask].mean()),
            }

    # Confusion matrix
    cm = sk_cm(y_true, preds, labels=range(N_CLASSES))

    correct_mask = preds == y_true
    conf_correct = float(confidences[correct_mask].mean()) if correct_mask.sum() > 0 else 0
    conf_incorrect = float(confidences[~correct_mask].mean()) if (~correct_mask).sum() > 0 else 0

    return {
        "accuracy": float(acc),
        "top3_accuracy": float(top3_acc),
        "top5_accuracy": float(top5_acc),
        "mean_confidence": float(confidences.mean()),
        "confidence_correct": conf_correct,
        "confidence_incorrect": conf_incorrect,
        "n_samples": int(len(y_true)),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def compute_lm_correction_metrics(y_true, raw_probs, corrector, threshold=0.8):
    """Evaluate LM correction on real neural data predictions.

    Processes phonemes in sliding windows to give the LM context.
    """
    import editdistance

    raw_preds = raw_probs.argmax(axis=1)
    confidences = raw_probs.max(axis=1)

    raw_phonemes = [CLASS_TO_ARPABET[p] for p in raw_preds]
    true_phonemes = [CLASS_TO_ARPABET[t] for t in y_true]
    conf_list = confidences.tolist()

    # Apply LM correction in chunks (gives LM phonotactic context)
    chunk_size = 15
    corrected_phonemes = []
    for i in range(0, len(raw_phonemes), chunk_size):
        chunk_raw = raw_phonemes[i:i + chunk_size]
        chunk_conf = conf_list[i:i + chunk_size]
        chunk_corrected = corrector.correct(chunk_raw, chunk_conf, threshold=threshold)
        corrected_phonemes.extend(chunk_corrected)

    # Convert back to class indices
    corrected_preds = np.array([
        ARPABET_TO_CLASS.get(p, raw_preds[i])
        for i, p in enumerate(corrected_phonemes[:len(y_true)])
    ])

    # Pad if LM returned fewer phonemes
    if len(corrected_preds) < len(y_true):
        corrected_preds = np.concatenate([
            corrected_preds,
            raw_preds[len(corrected_preds):]
        ])

    acc_before = (raw_preds == y_true).mean()
    acc_after = (corrected_preds[:len(y_true)] == y_true).mean()

    n_changed = n_fixed = n_broken = n_errors = n_correct = 0
    for i in range(len(y_true)):
        was_correct = raw_preds[i] == y_true[i]
        is_correct = corrected_preds[i] == y_true[i]
        was_changed = raw_preds[i] != corrected_preds[i]

        if was_changed:
            n_changed += 1
        if was_correct:
            n_correct += 1
            if not is_correct:
                n_broken += 1
        else:
            n_errors += 1
            if is_correct:
                n_fixed += 1

    per_before = editdistance.eval(
        [CLASS_TO_ARPABET[p] for p in raw_preds], true_phonemes
    ) / len(y_true)
    per_after = editdistance.eval(
        corrected_phonemes[:len(y_true)], true_phonemes
    ) / len(y_true)

    return {
        "threshold": threshold,
        "accuracy_before": float(acc_before),
        "accuracy_after": float(acc_after),
        "accuracy_delta": float(acc_after - acc_before),
        "per_before": float(per_before),
        "per_after": float(per_after),
        "per_reduction_relative": float(1 - per_after / max(per_before, 1e-8)),
        "n_changed": n_changed,
        "n_fixed": n_fixed,
        "n_broken": n_broken,
        "n_errors_before": n_errors,
        "n_correct_before": n_correct,
        "correction_precision": n_fixed / max(n_changed, 1),
        "correction_recall": n_fixed / max(n_errors, 1),
        "damage_rate": n_broken / max(n_correct, 1),
    }


# ─────────────────────────────────────────────────────────────────
# Audio round-trip
# ─────────────────────────────────────────────────────────────────

def evaluate_audio_roundtrip(phoneme_sequences, output_dir, n_samples=5):
    """Synthesize via ElevenLabs, transcribe with Whisper, compute PER."""
    from stage3_synthesis.elevenlabs_tts import synthesize_speech
    from stage3_synthesis.evaluate_audio import round_trip_eval
    import whisper

    whisper_model = whisper.load_model("base")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, phonemes in enumerate(phoneme_sequences[:n_samples]):
        audio_path = output_dir / f"sample_{i:03d}.mp3"
        try:
            synthesize_speech(phonemes, str(audio_path))
            per, text, recon = round_trip_eval(audio_path, phonemes, whisper_model)
            results.append({
                "index": i, "input_phonemes": " ".join(phonemes),
                "per": float(per), "recognized_text": text, "success": True,
            })
            print(f"  [{i}] PER={per:.2f}, heard='{text}'")
        except Exception as e:
            results.append({"index": i, "error": str(e), "success": False})
            print(f"  [{i}] ERROR: {e}")

    return results


# ─────────────────────────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────────────────────────

def print_comparison_table(results, include_lm=True):
    """Print formatted comparison table for writeup."""
    print("\n" + "=" * 90)
    print("COMPREHENSIVE RESULTS — Brain Signal → 40-Class Phoneme Decoding")
    print("Dataset: Willett et al. (Nature 2023), T12, cross-session evaluation")
    print("=" * 90)
    print(f"{'Method':<50} {'Acc':>8} {'Top-3':>8} {'Top-5':>8} {'Conf':>8}")
    print("-" * 90)
    print(f"{'Willett et al. 2023 (NB, 128ch, 39cls)':<50} {'61.4%':>8} {'—':>8} {'—':>8} {'—':>8}")
    print(f"{'Chance (40 classes)':<50} {'2.5%':>8} {'7.5%':>8} {'12.5%':>8} {'—':>8}")
    print("-" * 90)

    for name in ["TCN", "Transformer", "GRU", "EEGNet"]:
        key = f"classifier_{name}"
        if key in results:
            m = results[key]
            print(f"{'Our ' + name:<50} {m['accuracy']:>7.1%} {m['top3_accuracy']:>7.1%} "
                  f"{m['top5_accuracy']:>7.1%} {m['mean_confidence']:>7.3f}")

    for ens_key in ["classifier_Ensemble_uniform", "classifier_Ensemble_weighted"]:
        if ens_key in results:
            m = results[ens_key]
            label = ens_key.replace("classifier_", "Our ")
            print(f"{label:<50} {m['accuracy']:>7.1%} {m['top3_accuracy']:>7.1%} "
                  f"{m['top5_accuracy']:>7.1%} {m['mean_confidence']:>7.3f}")

    if include_lm:
        print("-" * 90)
        print(f"{'+ LM Correction':<50} {'Acc':>8} {'Δ Acc':>8} {'PER':>8} {'Prec':>8}")
        print("-" * 90)
        for key in sorted(results.keys()):
            if key.startswith("lm_"):
                m = results[key]
                label = key.replace("lm_", "").replace("_", " τ=")
                print(f"{'  ' + label:<50} {m['accuracy_after']:>7.1%} "
                      f"{m['accuracy_delta']:>+7.1%} {m['per_after']:>7.1%} "
                      f"{m['correction_precision']:>7.1%}")

    print("=" * 90)

    # Correction quality detail
    for key in sorted(results.keys()):
        if key.startswith("lm_"):
            m = results[key]
            print(f"\n{key}:")
            print(f"  Precision (helpful changes): {m['correction_precision']:.1%}")
            print(f"  Recall (errors fixed):       {m['correction_recall']:.1%}")
            print(f"  Damage rate:                 {m['damage_rate']:.1%}")
            print(f"  Changes: {m['n_changed']}, fixed: {m['n_fixed']}, broken: {m['n_broken']}")


def print_per_class_breakdown(results, n_worst=10, n_best=10):
    """Print per-phoneme accuracy for the best classifier."""
    best_key = max(
        (k for k in results if k.startswith("classifier_")),
        key=lambda k: results[k].get("accuracy", 0),
        default=None
    )
    if not best_key:
        return

    per_class = results[best_key].get("per_class", {})
    if not per_class:
        return

    print(f"\n{'='*65}")
    print(f"PER-PHONEME ACCURACY — {best_key} ({results[best_key]['accuracy']:.1%} overall)")
    print(f"{'='*65}")

    sorted_cls = sorted(per_class.items(), key=lambda x: x[1]["accuracy"])

    print(f"\nHardest phonemes (worst {n_worst}):")
    print(f"{'Phoneme':<10} {'Type':<6} {'Accuracy':>10} {'N':>6} {'Confidence':>12}")
    print("-" * 45)
    for ph, info in sorted_cls[:n_worst]:
        ptype = "vowel" if ph in VOWELS else "cons"
        print(f"{ph:<10} {ptype:<6} {info['accuracy']:>10.1%} {info['n_samples']:>6} "
              f"{info['avg_confidence']:>12.3f}")

    print(f"\nEasiest phonemes (best {n_best}):")
    for ph, info in sorted_cls[-n_best:]:
        ptype = "vowel" if ph in VOWELS else "cons"
        print(f"{ph:<10} {ptype:<6} {info['accuracy']:>10.1%} {info['n_samples']:>6} "
              f"{info['avg_confidence']:>12.3f}")

    # Aggregate: consonants vs vowels
    cons_accs = [info["accuracy"] for ph, info in per_class.items()
                 if ph not in VOWELS and ph != "SIL"]
    vowel_accs = [info["accuracy"] for ph, info in per_class.items()
                  if ph in VOWELS]

    if cons_accs and vowel_accs:
        print(f"\nConsonant avg: {np.mean(cons_accs):.1%} ({len(cons_accs)} classes)")
        print(f"Vowel avg:     {np.mean(vowel_accs):.1%} ({len(vowel_accs)} classes)")

    # Articulatory confusion analysis
    print(f"\nVoiced/unvoiced confusion pairs (from confusion matrix):")
    confusion_pairs = [
        ("B", "P"), ("D", "T"), ("G", "K"), ("V", "F"),
        ("Z", "S"), ("ZH", "SH"), ("DH", "TH"), ("JH", "CH"),
    ]
    if "confusion_matrix" in results[best_key]:
        cm = results[best_key]["confusion_matrix"]
        for p1, p2 in confusion_pairs:
            if p1 in ARPABET_TO_CLASS and p2 in ARPABET_TO_CLASS:
                i1, i2 = ARPABET_TO_CLASS[p1], ARPABET_TO_CLASS[p2]
                r1 = cm[i1].sum()
                r2 = cm[i2].sum()
                if r1 > 0 and r2 > 0:
                    c12 = cm[i1, i2] / r1  # P(predict p2 | true p1)
                    c21 = cm[i2, i1] / r2  # P(predict p1 | true p2)
                    print(f"  {p1}↔{p2}: P({p2}|{p1})={c12:.1%}, P({p1}|{p2})={c21:.1%}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end brain-to-speech evaluation")
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument("--n-audio", type=int, default=5)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("END-TO-END BRAIN-TO-SPEECH EVALUATION")
    print(f"  Date: {datetime.now().isoformat()}")
    print(f"  Device: {device}")
    print(f"  Using saved Stage 1 predictions")
    print("=" * 60)

    # ── Load saved predictions ──
    print("\nLoading saved predictions...")
    y_true, model_probs = load_saved_predictions()

    if y_true is None or not model_probs:
        print("ERROR: No predictions found.")
        sys.exit(1)

    # ── Evaluate individual classifiers ──
    results = {}

    for name, probs in model_probs.items():
        print(f"\n{'─'*40}")
        print(f"Computing metrics for {name}...")
        metrics = compute_metrics(y_true, probs, label=name)
        key = f"classifier_{name}"
        results[key] = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        # Keep confusion matrix for analysis but don't serialize
        results[key]["confusion_matrix"] = metrics["confusion_matrix"]
        print(f"  Acc: {metrics['accuracy']:.1%}, Top-3: {metrics['top3_accuracy']:.1%}, "
              f"Top-5: {metrics['top5_accuracy']:.1%}")

    # ── Ensemble — try multiple weighting schemes ──
    if len(model_probs) >= 2:
        print(f"\n{'─'*40}")
        print("Evaluating ensembles...")

        # Uniform weights
        ens_uniform = ensemble_predictions(model_probs, weights=None)
        m_uni = compute_metrics(y_true, ens_uniform)
        results["classifier_Ensemble_uniform"] = {
            k: v for k, v in m_uni.items() if k != "confusion_matrix"
        }
        results["classifier_Ensemble_uniform"]["confusion_matrix"] = m_uni["confusion_matrix"]
        print(f"  Uniform:  Acc={m_uni['accuracy']:.1%}, Top-3={m_uni['top3_accuracy']:.1%}")

        # Accuracy-proportional weights
        acc_weights = {}
        for name in model_probs:
            key = f"classifier_{name}"
            acc_weights[name] = results[key]["accuracy"]
        ens_weighted = ensemble_predictions(model_probs, weights=acc_weights)
        m_w = compute_metrics(y_true, ens_weighted)
        results["classifier_Ensemble_weighted"] = {
            k: v for k, v in m_w.items() if k != "confusion_matrix"
        }
        results["classifier_Ensemble_weighted"]["confusion_matrix"] = m_w["confusion_matrix"]
        print(f"  Weighted: Acc={m_w['accuracy']:.1%}, Top-3={m_w['top3_accuracy']:.1%}")
        print(f"    Weights: {acc_weights}")

        # Pick best ensemble
        best_ens_name = "Ensemble_weighted" if m_w["accuracy"] >= m_uni["accuracy"] else "Ensemble_uniform"
        best_ens_probs = ens_weighted if best_ens_name == "Ensemble_weighted" else ens_uniform

    # ── LM correction ──
    if not args.skip_lm:
        print(f"\n{'─'*40}")
        print("Loading LM corrector...")
        try:
            from stage2_lm_correction.inference_lm import PhonemeCorrector
            corrector = PhonemeCorrector(device="cuda")

            # Evaluate on best single model + best ensemble
            eval_targets = {}
            eval_targets["TCN"] = model_probs.get("TCN")
            if len(model_probs) >= 2:
                eval_targets[best_ens_name] = best_ens_probs

            for model_name, probs in eval_targets.items():
                if probs is None:
                    continue
                for threshold in args.thresholds:
                    print(f"\n  LM correction: {model_name}, τ={threshold}")
                    lm_metrics = compute_lm_correction_metrics(
                        y_true, probs, corrector, threshold=threshold
                    )
                    key = f"lm_{model_name}_{threshold}"
                    results[key] = lm_metrics
                    print(f"    Acc: {lm_metrics['accuracy_before']:.1%} → "
                          f"{lm_metrics['accuracy_after']:.1%} "
                          f"(Δ={lm_metrics['accuracy_delta']:+.1%})")
                    print(f"    PER: {lm_metrics['per_before']:.1%} → "
                          f"{lm_metrics['per_after']:.1%}")
                    print(f"    Fixed: {lm_metrics['n_fixed']}, "
                          f"Broken: {lm_metrics['n_broken']}, "
                          f"Precision: {lm_metrics['correction_precision']:.1%}")

            del corrector
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  LM corrector unavailable: {e}")
            import traceback; traceback.print_exc()
            print("  Skipping LM correction evaluation.")

    # ── Audio round-trip ──
    if args.with_audio:
        print(f"\n{'─'*40}")
        print("Audio round-trip evaluation...")
        best_probs = best_ens_probs if len(model_probs) >= 2 else list(model_probs.values())[0]
        preds = best_probs.argmax(axis=1)
        sequences = []
        for i in range(0, min(args.n_audio * 5, len(preds)), 5):
            seq = [CLASS_TO_ARPABET[p] for p in preds[i:i+5]]
            sequences.append(seq)
        audio_results = evaluate_audio_roundtrip(
            sequences, RESULTS_DIR / "audio_samples", n_samples=args.n_audio
        )
        results["audio_roundtrip"] = audio_results

    # ── Print results ──
    print_comparison_table(results, include_lm=not args.skip_lm)
    print_per_class_breakdown(results)

    # ── Save results ──
    output_path = args.output or str(RESULTS_DIR / f"end_to_end_{timestamp}.json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {
                kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    latest = RESULTS_DIR / "latest_results.json"
    with open(latest, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
