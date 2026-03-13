#!/usr/bin/env python3
"""Preprocess all data: extract HG, normalize, compute LPC + VAD, cache to HDF5.

Usage:
    python scripts/02_preprocess.py              # process all splits
    python scripts/02_preprocess.py --split train # process only train
    python scripts/02_preprocess.py --verify      # verify existing cache

This is CPU-only work (no GPU needed). Takes ~10-15 minutes for full dataset.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import torch
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from speech_bci import config
from speech_bci.io_utils import load_run, discover_runs
from speech_bci.signal import extract_hg, set_signal_device
from speech_bci.normalization import compute_day_stats, normalize_hg
from speech_bci.vad_labels import energy_based_vad, align_vad_to_hg
from speech_bci.lpc import extract_lpc

# Use GPU for signal processing
if torch.cuda.is_available():
    _device = os.environ.get("PREPROCESS_GPU", "cuda:0")
    set_signal_device(_device)
    print(f"Signal processing on GPU: {_device} ({torch.cuda.get_device_name(int(_device.split(':')[-1]))})")

WORD_TO_ID = {w: i for i, w in enumerate(config.WORDS)}

SPLITS = {
    "train": (config.TRAIN_DIR, config.TRAIN_DAYS),
    "validation": (config.VAL_DIR, [config.VAL_DAY]),
    "test": (config.TEST_DIR, [config.TEST_DAY]),
    "online": (config.ONLINE_DIR, config.ONLINE_DAYS),
}


def build_norm_cache():
    """Build normalization stats for all session days."""
    all_days = config.TRAIN_DAYS + [config.VAL_DAY, config.TEST_DAY] + config.ONLINE_DAYS
    cache = {}
    for day_id in tqdm(all_days, desc="Computing normalization baselines"):
        syll_path = config.SYLLABLE_DIR / day_id / "SyllableRepetition_Overt.mat"
        if syll_path.exists():
            mu, sd = compute_day_stats(syll_path)
            cache[day_id] = (mu, sd)
        else:
            print(f"  WARNING: No syllable baseline for {day_id}")
    return cache


def process_run(data_dir, day_id, run_id, mu, sd):
    """Process a single run → dict of arrays."""
    run = load_run(data_dir, day_id, run_id)

    # HG extraction + normalization
    hg, _ = extract_hg(run.ecog)
    hg_z = normalize_hg(hg, mu, sd)

    # VAD from audio
    vad = energy_based_vad(run.audio, run.fs_audio)
    vad = align_vad_to_hg(vad, len(hg_z))

    # LPC features (Python fallback — no LPCNet build needed)
    lpc = extract_lpc(run.audio, run.fs_audio, method="python")

    # Align to common length
    n = min(len(hg_z), len(lpc), len(vad))

    # Trial boundaries as frame indices
    trials = []
    for _, row in run.trials.iterrows():
        sf = int(row["start"] * 100)
        ef = int(row["end"] * 100)
        wid = WORD_TO_ID.get(row["word"], -1)
        if sf < n and ef <= n and wid >= 0:
            trials.append([sf, ef, wid])

    return {
        "hg": hg_z[:n].astype(np.float32),
        "lpc": lpc[:n].astype(np.float32),
        "vad": vad[:n].astype(np.int8),
        "trials": np.array(trials, dtype=np.int32) if trials else np.zeros((0, 3), dtype=np.int32),
    }


def preprocess(splits_to_process=None, output_path=None):
    """Main preprocessing pipeline."""
    output_path = output_path or config.CACHE_DIR / "features.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splits_to_process = splits_to_process or list(SPLITS.keys())

    print(f"Output: {output_path}")
    print(f"Splits: {splits_to_process}")

    # Build normalization cache
    norm_cache = build_norm_cache()
    print(f"Normalization baselines: {len(norm_cache)} days")

    t0 = time.time()
    total_runs = 0
    total_trials = 0

    with h5py.File(output_path, "a") as f:  # append mode to allow incremental
        for split_name in splits_to_process:
            if split_name not in SPLITS:
                print(f"Unknown split: {split_name}")
                continue

            data_dir, day_ids = SPLITS[split_name]
            print(f"\n=== {split_name} ({len(day_ids)} days) ===")

            for day_id in day_ids:
                if day_id not in norm_cache:
                    print(f"  SKIP {day_id}: no normalization baseline")
                    continue

                day_dir = data_dir / day_id
                if not day_dir.exists():
                    print(f"  SKIP {day_id}: directory not found")
                    continue

                mu, sd = norm_cache[day_id]
                runs = discover_runs(day_dir)

                for run_id in tqdm(runs, desc=f"  {day_id}", leave=False):
                    grp_key = f"{day_id}/{run_id}"
                    if grp_key in f:
                        print(f"    {grp_key}: already cached, skipping")
                        continue

                    try:
                        result = process_run(data_dir, day_id, run_id, mu, sd)
                        grp = f.create_group(grp_key)
                        for key, arr in result.items():
                            grp.create_dataset(key, data=arr)
                        grp.attrs["day_id"] = day_id
                        grp.attrs["run_id"] = run_id
                        grp.attrs["split"] = split_name

                        n_trials = len(result["trials"])
                        total_runs += 1
                        total_trials += n_trials
                        print(f"    {grp_key}: hg={result['hg'].shape}, "
                              f"trials={n_trials}, "
                              f"speech={result['vad'].mean():.1%}")
                    except Exception as e:
                        print(f"    ERROR {grp_key}: {e}")

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nDone: {total_runs} runs, {total_trials} trials, "
          f"{size_mb:.1f} MB, {elapsed:.0f}s")


def verify(output_path=None):
    """Verify HDF5 cache contents."""
    output_path = output_path or config.CACHE_DIR / "features.h5"
    if not output_path.exists():
        print(f"Cache not found: {output_path}")
        return

    with h5py.File(output_path, "r") as f:
        total = 0
        for day_id in sorted(f.keys()):
            for run_id in sorted(f[day_id].keys()):
                grp = f[day_id][run_id]
                n = grp["hg"].shape[0]
                n_ch = grp["hg"].shape[1]
                n_trials = grp["trials"].shape[0]
                split = grp.attrs.get("split", "?")
                print(f"  {day_id}/{run_id}: {n} frames x {n_ch} ch, "
                      f"{n_trials} trials [{split}]")
                total += n_trials
        print(f"\nTotal: {total} trials")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Speech BCI data")
    parser.add_argument("--split", type=str, nargs="+",
                        help="Which splits to process (default: all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output HDF5 path")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing cache")
    args = parser.parse_args()

    if args.verify:
        verify(args.output)
    else:
        preprocess(args.split, args.output)


if __name__ == "__main__":
    main()
