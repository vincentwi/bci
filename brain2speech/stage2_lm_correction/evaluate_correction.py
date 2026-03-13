#!/usr/bin/env python3
"""
Stage 2e: Evaluate phoneme correction quality.

Multi-GPU parallel evaluation: each GPU gets a shard of test data,
results are aggregated at the end.

Metrics:
    - Phoneme Error Rate (PER) before/after correction
    - Correction precision: fraction of changes that are improvements
    - Correction recall: fraction of errors that are fixed
    - Damage rate: fraction of correct predictions broken by LM

Usage:
    python evaluate_correction.py
    python evaluate_correction.py --threshold 0.7 --gpus 0,1,2,3
    python evaluate_correction.py --max_samples 5000 --gpus 0,1,2,3,4,5
"""
import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, LORA_ADAPTER_PATH


def phoneme_error_rate(predicted, reference):
    """PER = edit_distance(pred, ref) / len(ref)."""
    import editdistance
    if len(reference) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    return editdistance.eval(predicted, reference) / len(reference)


def _eval_worker(gpu_id, shard, threshold, result_queue):
    """Worker process: load model on a specific GPU and evaluate a shard."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from inference_lm import PhonemeCorrector
    corrector = PhonemeCorrector(device="cuda")

    results = {
        'per_before': [], 'per_after': [],
        'fixed': 0, 'broken': 0,
        'total_errors': 0, 'total_correct': 0, 'total_changes': 0,
    }

    for i, pair in enumerate(shard):
        noisy = pair['noisy'].split() if isinstance(pair['noisy'], str) else pair['noisy']
        clean = pair['clean'].split() if isinstance(pair['clean'], str) else pair['clean']

        confidences = [0.5] * len(noisy)
        corrected = corrector.correct(noisy, confidences, threshold=threshold)

        results['per_before'].append(phoneme_error_rate(noisy, clean))
        results['per_after'].append(phoneme_error_rate(corrected, clean))

        min_len = min(len(noisy), len(clean), len(corrected))
        for j in range(min_len):
            n, cl, co = noisy[j], clean[j], corrected[j]
            if n != cl:
                results['total_errors'] += 1
                if co == cl:
                    results['fixed'] += 1
            else:
                results['total_correct'] += 1
                if co != cl:
                    results['broken'] += 1
            if n != co:
                results['total_changes'] += 1

        if (i + 1) % 50 == 0:
            per_b = np.mean(results['per_before'])
            per_a = np.mean(results['per_after'])
            print(f"  [GPU {gpu_id}] {i+1}/{len(shard)}: PER {per_b:.3f} → {per_a:.3f}")

    result_queue.put(results)


def evaluate_parallel(test_pairs, gpus, threshold=0.8, max_samples=None):
    """Evaluate across multiple GPUs in parallel."""
    if max_samples:
        test_pairs = test_pairs[:max_samples]

    n_gpus = len(gpus)
    chunk_size = len(test_pairs) // n_gpus
    remainder = len(test_pairs) % n_gpus

    shards = []
    offset = 0
    for i in range(n_gpus):
        n = chunk_size + (1 if i < remainder else 0)
        shards.append(test_pairs[offset:offset + n])
        offset += n

    print(f"Distributing {len(test_pairs)} samples across {n_gpus} GPUs: "
          f"{[len(s) for s in shards]}")

    result_queue = Queue()
    processes = []
    for gpu_id, shard in zip(gpus, shards):
        p = Process(target=_eval_worker, args=(gpu_id, shard, threshold, result_queue))
        p.start()
        processes.append(p)

    # Collect results
    all_results = []
    for _ in processes:
        all_results.append(result_queue.get())

    for p in processes:
        p.join()

    # Aggregate
    merged = {
        'per_before': [], 'per_after': [],
        'fixed': 0, 'broken': 0,
        'total_errors': 0, 'total_correct': 0, 'total_changes': 0,
    }
    for r in all_results:
        merged['per_before'].extend(r['per_before'])
        merged['per_after'].extend(r['per_after'])
        merged['fixed'] += r['fixed']
        merged['broken'] += r['broken']
        merged['total_errors'] += r['total_errors']
        merged['total_correct'] += r['total_correct']
        merged['total_changes'] += r['total_changes']

    per_before = np.mean(merged['per_before'])
    per_after = np.mean(merged['per_after'])

    return {
        'per_before': per_before,
        'per_after': per_after,
        'per_reduction': 1 - per_after / max(per_before, 1e-8),
        'per_reduction_absolute': per_before - per_after,
        'correction_precision': merged['fixed'] / max(merged['total_changes'], 1),
        'correction_recall': merged['fixed'] / max(merged['total_errors'], 1),
        'damage_rate': merged['broken'] / max(merged['total_correct'], 1),
        'total_errors': merged['total_errors'],
        'total_correct': merged['total_correct'],
        'total_changes': merged['total_changes'],
        'total_fixed': merged['fixed'],
        'total_broken': merged['broken'],
        'n_samples': len(test_pairs),
    }


def evaluate_correction(test_pairs, corrector, threshold=0.8, max_samples=None):
    """Single-GPU evaluation (legacy interface)."""
    results = {
        'per_before': [], 'per_after': [],
        'fixed': 0, 'broken': 0,
        'total_errors': 0, 'total_correct': 0, 'total_changes': 0,
    }

    if max_samples:
        test_pairs = test_pairs[:max_samples]

    for i, pair in enumerate(test_pairs):
        noisy = pair['noisy'].split() if isinstance(pair['noisy'], str) else pair['noisy']
        clean = pair['clean'].split() if isinstance(pair['clean'], str) else pair['clean']

        confidences = [0.5] * len(noisy)
        corrected = corrector.correct(noisy, confidences, threshold=threshold)

        results['per_before'].append(phoneme_error_rate(noisy, clean))
        results['per_after'].append(phoneme_error_rate(corrected, clean))

        min_len = min(len(noisy), len(clean), len(corrected))
        for j in range(min_len):
            n, cl, co = noisy[j], clean[j], corrected[j]
            if n != cl:
                results['total_errors'] += 1
                if co == cl:
                    results['fixed'] += 1
            else:
                results['total_correct'] += 1
                if co != cl:
                    results['broken'] += 1
            if n != co:
                results['total_changes'] += 1

        if (i + 1) % 100 == 0:
            per_b = np.mean(results['per_before'])
            per_a = np.mean(results['per_after'])
            print(f"  {i+1}/{len(test_pairs)}: PER {per_b:.3f} → {per_a:.3f}")

    per_before = np.mean(results['per_before'])
    per_after = np.mean(results['per_after'])

    return {
        'per_before': per_before,
        'per_after': per_after,
        'per_reduction': 1 - per_after / max(per_before, 1e-8),
        'per_reduction_absolute': per_before - per_after,
        'correction_precision': results['fixed'] / max(results['total_changes'], 1),
        'correction_recall': results['fixed'] / max(results['total_errors'], 1),
        'damage_rate': results['broken'] / max(results['total_correct'], 1),
        'total_errors': results['total_errors'],
        'total_correct': results['total_correct'],
        'total_changes': results['total_changes'],
        'total_fixed': results['fixed'],
        'total_broken': results['broken'],
        'n_samples': len(test_pairs),
    }


def print_comparison_table(metrics, stage1_acc=None):
    """Print comparison table matching the plan."""
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Method':<40} {'PER':>10}")
    print("-" * 50)
    print(f"{'Paper (Naive Bayes, 128ch, 39 classes)':<40} {'~38%':>10}")
    if stage1_acc:
        per_est = 1.0 - stage1_acc
        print(f"{'Our best classifier (TCN, 1280 feat)':<40} {per_est:>10.1%}")
    print(f"{'Before LM correction':<40} {metrics['per_before']:>10.1%}")
    print(f"{'After LM correction':<40} {metrics['per_after']:>10.1%}")
    print(f"{'Relative PER reduction':<40} {metrics['per_reduction']:>10.1%}")
    print("-" * 50)

    print(f"\nCorrection quality:")
    print(f"  Precision (changes that helped): {metrics['correction_precision']:.1%}")
    print(f"  Recall (errors fixed):           {metrics['correction_recall']:.1%}")
    print(f"  Damage rate (correct→broken):    {metrics['damage_rate']:.1%}")
    print(f"  Total changes: {metrics['total_changes']}, "
          f"fixed: {metrics['total_fixed']}, broken: {metrics['total_broken']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs for parallel eval")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    print("=" * 60)
    print("Stage 2e: Evaluating phoneme correction")
    print("=" * 60)

    # Load test data
    test_path = DATA_DIR / "phoneme_correction_test_raw.jsonl"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. Run generate_pairs.py first.")
        sys.exit(1)

    test_pairs = []
    with open(test_path) as f:
        for line in f:
            test_pairs.append(json.loads(line))
    print(f"Loaded {len(test_pairs)} test pairs")

    # Evaluate
    t0 = time.time()
    if len(gpus) > 1:
        print(f"\nParallel evaluation on GPUs {gpus}...")
        metrics = evaluate_parallel(
            test_pairs, gpus,
            threshold=args.threshold,
            max_samples=args.max_samples,
        )
    else:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        from inference_lm import PhonemeCorrector
        print(f"Loading PhonemeCorrector (threshold={args.threshold})...")
        corrector = PhonemeCorrector()
        print("\nEvaluating...")
        metrics = evaluate_correction(
            test_pairs, corrector,
            threshold=args.threshold,
            max_samples=args.max_samples,
        )
    elapsed = time.time() - t0
    print(f"\nEvaluation took {elapsed:.1f}s")

    # Print results
    print_comparison_table(metrics, stage1_acc=0.988)

    # Save results
    results_path = DATA_DIR / "correction_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
