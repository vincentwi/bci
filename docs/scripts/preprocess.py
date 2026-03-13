#!/usr/bin/env python3
"""
Preprocess Willett et al. (Nature 2023) speech neuroprosthesis data.

Handles three data sources:
  1. diagnosticBlocks — 7-word classification across sessions (primary)
  2. tuningTasks — phoneme/word/orofacial tuning data
  3. competitionData — sentence-level data (for advanced models)

Pipeline:
  1. Load .mat files (spikePow, tx features, trial info)
  2. Z-score features per block
  3. Segment trials around go-period onsets
  4. Save processed arrays as .npz

Usage:
    python preprocess.py
    python preprocess.py --dataset diagnostic   # default
    python preprocess.py --dataset tuning
    python preprocess.py --dataset competition
    python preprocess.py --all                   # process all datasets
"""
import argparse
import sys
import os
import tarfile
import numpy as np
import scipy.io

from config import *

np.random.seed(SEED)


def extract_if_needed(tar_path, extract_dir):
    """Extract tar.gz if not already extracted."""
    if not tar_path.exists():
        print(f"  ERROR: {tar_path} not found")
        print(f"  Run: wget -O {tar_path} 'https://zenodo.org/api/records/8047896/files/{tar_path.name}/content'")
        return False
    # Check if already extracted
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  Already extracted: {extract_dir}")
        return True
    print(f"  Extracting {tar_path.name}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)
    print(f"  Done: {sum(1 for _ in extract_dir.rglob('*.mat'))} .mat files")
    return True


def load_diagnostic_blocks():
    """Load diagnostic block data (7-word classification)."""
    tar_path = DRYAD_DIR / "diagnosticBlocks.tar.gz"
    extract_dir = DRYAD_DIR / "diagnosticBlocks"
    if not extract_if_needed(tar_path, extract_dir):
        return None

    mat_files = sorted(extract_dir.rglob("*.mat"))
    if not mat_files:
        print("  No .mat files found in diagnosticBlocks")
        return None

    print(f"\n  Found {len(mat_files)} diagnostic block files:")
    all_data = []
    for mf in mat_files:
        try:
            mat = scipy.io.loadmat(str(mf), squeeze_me=False)
        except Exception as e:
            print(f"    SKIP {mf.name}: {e}")
            continue

        session_name = mf.stem
        sp = mat.get("spikePow", None)
        if sp is None:
            print(f"    SKIP {session_name}: no spikePow")
            continue

        # spikePow is S×1 cell of T×256 or T×256 matrix
        if sp.ndim == 1 or (sp.ndim == 2 and sp.shape[1] == 1 and sp.dtype == object):
            # Cell array — flatten
            sp_flat = sp.flatten()
        elif sp.dtype == object:
            sp_flat = sp.flatten()
        else:
            sp_flat = [sp]  # single matrix

        # Get trial info
        trial_cues = mat.get("trialCues", None)
        cue_list = mat.get("cueList", None)
        go_epochs = mat.get("goTrialEpochs", None)
        block_num = mat.get("blockNum", None)
        trial_state = mat.get("trialState", None)

        if trial_cues is None or cue_list is None:
            print(f"    SKIP {session_name}: missing trialCues or cueList")
            continue

        # Flatten cueList
        if cue_list.dtype == object:
            cue_names = [str(c.flat[0]) if hasattr(c, 'flat') else str(c)
                         for c in cue_list.flatten()]
        else:
            cue_names = [str(c) for c in cue_list.flatten()]

        # Build continuous spikePow matrix
        if len(sp_flat) > 0 and hasattr(sp_flat[0], 'shape'):
            continuous_sp = np.vstack([s for s in sp_flat if hasattr(s, 'shape') and s.size > 0])
        else:
            print(f"    SKIP {session_name}: empty spikePow")
            continue

        n_time, n_ch = continuous_sp.shape
        trial_cues_arr = trial_cues.flatten() if trial_cues.ndim > 1 else trial_cues

        # Get tx features if available
        tx_features = {}
        for tx_name in ["tx1", "tx2", "tx3", "tx4"]:
            tx = mat.get(tx_name, None)
            if tx is not None:
                if tx.dtype == object:
                    tx_flat = tx.flatten()
                    tx_features[tx_name] = np.vstack([t for t in tx_flat if hasattr(t, 'shape') and t.size > 0])
                else:
                    tx_features[tx_name] = tx

        # Block numbers for normalization
        if block_num is not None:
            bn = block_num.flatten()
        else:
            bn = np.ones(n_time, dtype=int)

        info = {
            "session": session_name,
            "spikePow": continuous_sp.astype(np.float32),
            "tx_features": tx_features,
            "trial_cues": trial_cues_arr,
            "cue_names": cue_names,
            "go_epochs": go_epochs if go_epochs is not None else None,
            "block_num": bn[:n_time],
            "trial_state": trial_state.flatten()[:n_time] if trial_state is not None else None,
            "n_channels": n_ch,
        }

        n_trials = len(trial_cues_arr)
        print(f"    {session_name}: {n_time} bins, {n_ch} ch, "
              f"{n_trials} trials, {len(cue_names)} cues")
        all_data.append(info)

    return all_data


def load_tuning_tasks():
    """Load tuning task data (phonemes, words, orofacial)."""
    tar_path = DRYAD_DIR / "tuningTasks.tar.gz"
    extract_dir = DRYAD_DIR / "tuningTasks"
    if not extract_if_needed(tar_path, extract_dir):
        return None

    mat_files = sorted(extract_dir.rglob("*.mat"))
    if not mat_files:
        print("  No .mat files found in tuningTasks")
        return None

    print(f"\n  Found {len(mat_files)} tuning task files:")
    all_data = []
    for mf in mat_files:
        try:
            mat = scipy.io.loadmat(str(mf), squeeze_me=False)
        except Exception as e:
            print(f"    SKIP {mf.name}: {e}")
            continue

        session_name = mf.stem
        sp = mat.get("spikePow", None)
        if sp is None:
            print(f"    SKIP {session_name}: no spikePow")
            continue

        if sp.dtype == object:
            continuous_sp = np.vstack([s for s in sp.flatten() if hasattr(s, 'shape') and s.size > 0])
        else:
            continuous_sp = sp

        trial_cues = mat.get("trialCues", None)
        cue_list = mat.get("cueList", None)
        go_epochs = mat.get("goTrialEpochs", None)
        block_num = mat.get("blockNum", None)
        trial_state = mat.get("trialState", None)

        if trial_cues is None or cue_list is None:
            continue

        if cue_list.dtype == object:
            cue_names = [str(c.flat[0]) if hasattr(c, 'flat') else str(c)
                         for c in cue_list.flatten()]
        else:
            cue_names = [str(c) for c in cue_list.flatten()]

        n_time, n_ch = continuous_sp.shape
        trial_cues_arr = trial_cues.flatten()

        tx_features = {}
        for tx_name in ["tx1", "tx2", "tx3", "tx4"]:
            tx = mat.get(tx_name, None)
            if tx is not None:
                if tx.dtype == object:
                    tx_features[tx_name] = np.vstack([t for t in tx.flatten() if hasattr(t, 'shape') and t.size > 0])
                else:
                    tx_features[tx_name] = tx

        bn = block_num.flatten()[:n_time] if block_num is not None else np.ones(n_time, dtype=int)

        info = {
            "session": session_name,
            "spikePow": continuous_sp.astype(np.float32),
            "tx_features": tx_features,
            "trial_cues": trial_cues_arr,
            "cue_names": cue_names,
            "go_epochs": go_epochs,
            "block_num": bn,
            "trial_state": trial_state.flatten()[:n_time] if trial_state is not None else None,
            "n_channels": n_ch,
        }
        n_trials = len(trial_cues_arr)
        print(f"    {session_name}: {n_time} bins, {n_ch} ch, "
              f"{n_trials} trials, {len(cue_names)} cues: {cue_names[:5]}...")
        all_data.append(info)

    return all_data


def load_competition_data():
    """Load competition data (sentence-level)."""
    tar_path = DRYAD_DIR / "competitionData.tar.gz"
    extract_dir = DRYAD_DIR / "competitionData"
    if not extract_if_needed(tar_path, extract_dir):
        return None

    mat_files = sorted(extract_dir.rglob("*.mat"))
    if not mat_files:
        print("  No .mat files found in competitionData")
        return None

    print(f"\n  Found {len(mat_files)} competition data files:")
    all_data = []
    for mf in mat_files:
        try:
            mat = scipy.io.loadmat(str(mf), squeeze_me=False)
        except Exception as e:
            print(f"    SKIP {mf.name}: {e}")
            continue

        session_name = mf.stem
        sp = mat.get("spikePow", None)
        if sp is None:
            continue

        # spikePow is S×1 cell array where each cell is T×256
        if sp.dtype == object:
            sp_list = [s for s in sp.flatten() if hasattr(s, 'shape') and s.size > 0]
        else:
            sp_list = [sp]

        sent_text = mat.get("sentenceText", None)
        block_idx = mat.get("blockIdx", None)

        tx_lists = {}
        for tx_name in ["tx1", "tx2", "tx3", "tx4"]:
            tx = mat.get(tx_name, None)
            if tx is not None and tx.dtype == object:
                tx_lists[tx_name] = [t for t in tx.flatten() if hasattr(t, 'shape') and t.size > 0]

        n_sentences = len(sp_list)
        n_ch = sp_list[0].shape[1] if sp_list else 0

        if sent_text is not None and sent_text.dtype.kind in ('U', 'S', 'O'):
            sentences = []
            for row in sent_text:
                if hasattr(row, 'flat'):
                    sentences.append(''.join(str(c) for c in row.flat).strip())
                else:
                    sentences.append(str(row).strip())
        else:
            sentences = [f"sentence_{i}" for i in range(n_sentences)]

        info = {
            "session": session_name,
            "spikePow_list": sp_list,
            "tx_lists": tx_lists,
            "sentences": sentences[:n_sentences],
            "block_idx": block_idx.flatten() if block_idx is not None else None,
            "n_channels": n_ch,
            "n_sentences": n_sentences,
        }
        print(f"    {session_name}: {n_sentences} sentences, {n_ch} ch")
        all_data.append(info)

    return all_data


def zscore_by_block(features, block_nums):
    """Z-score features within each block."""
    result = features.copy()
    for b in np.unique(block_nums):
        mask = block_nums == b
        if mask.sum() < 2:
            continue
        block_data = result[mask]
        mu = block_data.mean(axis=0, keepdims=True)
        sd = block_data.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        result[mask] = (block_data - mu) / sd
    return result


def process_diagnostic(all_data):
    """Process diagnostic blocks into trial-segmented arrays."""
    print("\n" + "=" * 60)
    print("Processing Diagnostic Blocks (word classification)")
    print("=" * 60)

    all_X, all_y, all_session_ids = [], [], []
    all_cue_names = None

    for sess_idx, data in enumerate(all_data):
        sp = data["spikePow"]
        trial_cues = data["trial_cues"]
        go_epochs = data["go_epochs"]
        block_num = data["block_num"]
        trial_state = data["trial_state"]
        cue_names = data["cue_names"]
        n_ch = data["n_channels"]

        if all_cue_names is None:
            all_cue_names = cue_names

        # Z-score by block
        sp_z = zscore_by_block(sp, block_num)

        # Stack tx features if available
        feature_list = [sp_z]
        for tx_name in sorted(data["tx_features"].keys()):
            tx = data["tx_features"][tx_name].astype(np.float32)
            tx_z = zscore_by_block(tx[:len(block_num)], block_num)
            feature_list.append(tx_z)

        features = np.concatenate(feature_list, axis=1)

        # Segment trials using go epochs
        if go_epochs is not None and go_epochs.size > 0:
            go_epochs_arr = go_epochs.reshape(-1, 2)
            n_trials = min(len(trial_cues), len(go_epochs_arr))

            trials, labels = [], []
            for i in range(n_trials):
                onset = int(go_epochs_arr[i, 0]) - 1  # MATLAB 1-indexed
                # Take PRE_BINS before onset and POST_BINS after
                start = onset - PRE_BINS
                end = onset + POST_BINS
                if start < 0 or end > len(features):
                    continue
                trial_data = features[start:end]
                trials.append(trial_data)
                labels.append(int(trial_cues[i]))

            if trials:
                X = np.array(trials, dtype=np.float32)
                y = np.array(labels, dtype=np.int64)
                all_X.append(X)
                all_y.append(y)
                all_session_ids.append(np.full(len(y), sess_idx))
                print(f"  {data['session']}: {len(y)} trials, "
                      f"shape={X.shape}, labels={np.unique(y)}")
        elif trial_state is not None:
            # Use trial_state to find go periods
            go_state = (trial_state == 1).astype(int)
            onsets = np.where(np.diff(go_state) == 1)[0] + 1
            n_trials = min(len(trial_cues), len(onsets))

            trials, labels = [], []
            for i in range(n_trials):
                onset = onsets[i]
                start = onset - PRE_BINS
                end = onset + POST_BINS
                if start < 0 or end > len(features):
                    continue
                trials.append(features[start:end])
                labels.append(int(trial_cues[i]))

            if trials:
                X = np.array(trials, dtype=np.float32)
                y = np.array(labels, dtype=np.int64)
                all_X.append(X)
                all_y.append(y)
                all_session_ids.append(np.full(len(y), sess_idx))
                print(f"  {data['session']}: {len(y)} trials, "
                      f"shape={X.shape}")

    if not all_X:
        print("  ERROR: No trials extracted!")
        return

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    session_ids = np.concatenate(all_session_ids)

    # Remap labels to 0-indexed contiguous
    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_mapped = np.array([label_map[yi] for yi in y], dtype=np.int64)
    n_classes = len(unique_labels)

    mapped_names = []
    if all_cue_names:
        for ul in unique_labels:
            idx = ul - 1  # MATLAB 1-indexed
            if 0 <= idx < len(all_cue_names):
                mapped_names.append(all_cue_names[idx])
            else:
                mapped_names.append(f"class_{ul}")
    else:
        mapped_names = [f"class_{ul}" for ul in unique_labels]

    print(f"\n  Total: X={X.shape}, {n_classes} classes, "
          f"{len(np.unique(session_ids))} sessions")
    print(f"  Classes: {list(zip(range(n_classes), mapped_names))}")
    print(f"  Per class: {dict(zip(mapped_names, [(y_mapped == c).sum() for c in range(n_classes)]))}")

    # Build flat features for classical ML (mean over time bins)
    n_trials, n_time, n_feat = X.shape
    # Time-binned features: split into bins of ~5 timesteps (100ms)
    bin_size = 5
    n_bins = n_time // bin_size
    X_binned = np.zeros((n_trials, n_bins, n_feat), dtype=np.float32)
    for b in range(n_bins):
        X_binned[:, b, :] = X[:, b*bin_size:(b+1)*bin_size, :].mean(axis=1)
    X_flat = X_binned.reshape(n_trials, -1)
    # Add temporal derivatives
    X_deriv = np.diff(X_binned, axis=1).reshape(n_trials, -1)
    X_feat = np.nan_to_num(np.hstack([X_flat, X_deriv]))
    print(f"  Classical features: {X_feat.shape}")

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "diagnostic_processed.npz"
    np.savez_compressed(
        out_path,
        X=X, y=y_mapped, session_ids=session_ids,
        X_feat=X_feat,
        class_names=np.array(mapped_names),
        n_classes=n_classes,
        label_map=np.array(list(label_map.items())),
    )
    print(f"\n  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def process_tuning(all_data):
    """Process tuning tasks into trial-segmented arrays."""
    print("\n" + "=" * 60)
    print("Processing Tuning Tasks")
    print("=" * 60)

    for data in all_data:
        sp = data["spikePow"]
        trial_cues = data["trial_cues"]
        go_epochs = data["go_epochs"]
        block_num = data["block_num"]
        trial_state = data["trial_state"]
        cue_names = data["cue_names"]
        session_name = data["session"]

        # Z-score by block
        sp_z = zscore_by_block(sp, block_num)

        feature_list = [sp_z]
        for tx_name in sorted(data["tx_features"].keys()):
            tx = data["tx_features"][tx_name].astype(np.float32)
            tx_z = zscore_by_block(tx[:len(block_num)], block_num)
            feature_list.append(tx_z)
        features = np.concatenate(feature_list, axis=1)

        # Segment using trial_state or go_epochs
        if go_epochs is not None and go_epochs.size > 0:
            go_epochs_arr = go_epochs.reshape(-1, 2)
            n_trials = min(len(trial_cues), len(go_epochs_arr))
            trials, labels, blocks = [], [], []

            for i in range(n_trials):
                onset = int(go_epochs_arr[i, 0]) - 1
                start = onset - PRE_BINS
                end = onset + POST_BINS
                if start < 0 or end > len(features):
                    continue
                trials.append(features[start:end])
                labels.append(int(trial_cues[i]))
                blocks.append(int(block_num[min(onset, len(block_num)-1)]))

            if not trials:
                print(f"  {session_name}: 0 trials extracted, skipping")
                continue

            X = np.array(trials, dtype=np.float32)
            y = np.array(labels, dtype=np.int64)
            block_ids = np.array(blocks)

            unique_labels = np.unique(y)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            y_mapped = np.array([label_map[yi] for yi in y])
            n_classes = len(unique_labels)

            mapped_names = []
            for ul in unique_labels:
                idx = ul - 1
                if 0 <= idx < len(cue_names):
                    mapped_names.append(cue_names[idx])
                else:
                    mapped_names.append(f"class_{ul}")

            # Flat features
            n_tr, n_t, n_f = X.shape
            bin_size = 5
            n_bins = n_t // bin_size
            X_binned = np.zeros((n_tr, n_bins, n_f), dtype=np.float32)
            for b in range(n_bins):
                X_binned[:, b, :] = X[:, b*bin_size:(b+1)*bin_size, :].mean(axis=1)
            X_flat = X_binned.reshape(n_tr, -1)
            X_deriv = np.diff(X_binned, axis=1).reshape(n_tr, -1)
            X_feat = np.nan_to_num(np.hstack([X_flat, X_deriv]))

            out_name = f"tuning_{session_name}.npz"
            out_path = PROCESSED_DIR / out_name
            np.savez_compressed(
                out_path,
                X=X, y=y_mapped, block_ids=block_ids,
                X_feat=X_feat,
                class_names=np.array(mapped_names),
                n_classes=n_classes,
            )
            print(f"  {session_name}: {len(y)} trials, {n_classes} classes, "
                  f"saved ({out_path.stat().st_size / 1e6:.1f} MB)")
            print(f"    Classes: {mapped_names}")


def process_competition(all_data):
    """Process competition data into sentence-level arrays."""
    print("\n" + "=" * 60)
    print("Processing Competition Data (sentences)")
    print("=" * 60)

    for data in all_data:
        session_name = data["session"]
        sp_list = data["spikePow_list"]
        sentences = data["sentences"]
        block_idx = data["block_idx"]
        n_sentences = data["n_sentences"]
        n_ch = data["n_channels"]

        # Pad/truncate all sentences to same length for batching
        lengths = [s.shape[0] for s in sp_list]
        max_len = min(max(lengths), 500)  # Cap at 10 seconds

        X = np.zeros((n_sentences, max_len, n_ch), dtype=np.float32)
        mask = np.zeros((n_sentences, max_len), dtype=bool)

        for i, sp in enumerate(sp_list):
            t = min(sp.shape[0], max_len)
            X[i, :t, :] = sp[:t].astype(np.float32)
            mask[i, :t] = True

        # Z-score per sentence
        for i in range(n_sentences):
            valid = mask[i]
            if valid.sum() < 2:
                continue
            mu = X[i, valid].mean(axis=0, keepdims=True)
            sd = X[i, valid].std(axis=0, keepdims=True)
            sd[sd < 1e-8] = 1.0
            X[i, valid] = (X[i, valid] - mu) / sd

        out_path = PROCESSED_DIR / f"competition_{session_name}.npz"
        np.savez_compressed(
            out_path,
            X=X, mask=mask,
            sentences=np.array(sentences, dtype=object),
            block_idx=block_idx,
            lengths=np.array(lengths),
        )
        print(f"  {session_name}: {n_sentences} sentences, max_len={max_len}, "
              f"saved ({out_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Willett speech data")
    parser.add_argument("--dataset", choices=["diagnostic", "tuning", "competition"],
                        default="diagnostic", help="Which dataset to process")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    datasets_to_process = []
    if args.all:
        datasets_to_process = ["diagnostic", "tuning", "competition"]
    else:
        datasets_to_process = [args.dataset]

    for ds in datasets_to_process:
        if ds == "diagnostic":
            data = load_diagnostic_blocks()
            if data:
                process_diagnostic(data)

        elif ds == "tuning":
            data = load_tuning_tasks()
            if data:
                process_tuning(data)

        elif ds == "competition":
            data = load_competition_data()
            if data:
                process_competition(data)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
