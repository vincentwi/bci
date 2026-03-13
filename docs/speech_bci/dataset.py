"""PyTorch Datasets and DataLoaders for speech BCI training."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import config


class SpeechBCIDataset(Dataset):
    """Trial-level dataset for word classification and acoustic decoding.

    Each item is a variable-length trial segment with HG features and
    corresponding targets (LPC coefficients, mel, or word label).

    Parameters
    ----------
    h5_path : path to precomputed HDF5 feature cache
    day_ids : which session days to include
    channel_mask : indices of selected channels (default: all 128)
    target_type : "lpc", "vad", or "word"
    pre_pad_frames : frames to include before trial start
    post_pad_frames : frames to include after trial end
    """

    def __init__(self, h5_path: Path, day_ids: list[str],
                 channel_mask: np.ndarray | None = None,
                 target_type: str = "lpc",
                 pre_pad_frames: int = 50, post_pad_frames: int = 50):
        self.h5_path = str(h5_path)
        self.channel_mask = channel_mask
        self.target_type = target_type
        self.pre_pad = pre_pad_frames
        self.post_pad = post_pad_frames

        # Build trial index from HDF5
        self.trials = []
        with h5py.File(self.h5_path, "r") as f:
            for day_id in day_ids:
                if day_id not in f:
                    continue
                day_grp = f[day_id]
                for run_id in sorted(day_grp.keys()):
                    run_grp = day_grp[run_id]
                    if "trials" not in run_grp:
                        continue
                    trial_data = run_grp["trials"][:]  # (n_trials, 3): start, end, word_id
                    n_hg = run_grp["hg"].shape[0]
                    for t in range(len(trial_data)):
                        sf = max(0, int(trial_data[t, 0]) - self.pre_pad)
                        ef = min(n_hg, int(trial_data[t, 1]) + self.post_pad)
                        if ef - sf < 10:
                            continue
                        self.trials.append({
                            "day_id": day_id,
                            "run_id": run_id,
                            "start_frame": sf,
                            "end_frame": ef,
                            "word_id": int(trial_data[t, 2]),
                        })

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]
        sf, ef = trial["start_frame"], trial["end_frame"]

        with h5py.File(self.h5_path, "r") as f:
            run_grp = f[trial["day_id"]][trial["run_id"]]
            hg = run_grp["hg"][sf:ef]
            if self.channel_mask is not None:
                hg = hg[:, self.channel_mask]

            if self.target_type == "lpc":
                target = run_grp["lpc"][sf:ef] if "lpc" in run_grp else np.zeros((ef - sf, config.N_LPC))
            elif self.target_type == "vad":
                target = run_grp["vad"][sf:ef] if "vad" in run_grp else np.zeros(ef - sf)
            elif self.target_type == "word":
                target = trial["word_id"]
            else:
                raise ValueError(f"Unknown target_type: {self.target_type}")

        hg_tensor = torch.from_numpy(hg.astype(np.float32))

        if self.target_type == "word":
            return hg_tensor, target
        else:
            target_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32))
            return hg_tensor, target_tensor

    @property
    def n_channels(self):
        if self.channel_mask is not None:
            return len(self.channel_mask)
        return config.N_ECOG


class SequentialFrameDataset(Dataset):
    """Continuous frame-level dataset for nVAD training with truncated BPTT.

    Returns full runs as continuous sequences. The training loop handles
    the truncated BPTT chunking.

    Parameters
    ----------
    h5_path : path to precomputed HDF5 feature cache
    day_ids : which session days to include
    channel_mask : indices of selected channels
    """

    def __init__(self, h5_path: Path, day_ids: list[str],
                 channel_mask: np.ndarray | None = None):
        self.h5_path = str(h5_path)
        self.channel_mask = channel_mask

        # Index all runs
        self.runs = []
        with h5py.File(self.h5_path, "r") as f:
            for day_id in day_ids:
                if day_id not in f:
                    continue
                day_grp = f[day_id]
                for run_id in sorted(day_grp.keys()):
                    if "hg" in day_grp[run_id] and "vad" in day_grp[run_id]:
                        n_frames = day_grp[run_id]["hg"].shape[0]
                        self.runs.append({
                            "day_id": day_id,
                            "run_id": run_id,
                            "n_frames": n_frames,
                        })

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, idx):
        run_info = self.runs[idx]

        with h5py.File(self.h5_path, "r") as f:
            run_grp = f[run_info["day_id"]][run_info["run_id"]]
            hg = run_grp["hg"][:]
            vad = run_grp["vad"][:]

        if self.channel_mask is not None:
            hg = hg[:, self.channel_mask]

        return (
            torch.from_numpy(hg.astype(np.float32)),
            torch.from_numpy(vad.astype(np.float32)),
        )

    @property
    def n_channels(self):
        if self.channel_mask is not None:
            return len(self.channel_mask)
        return config.N_ECOG


def collate_variable_length(batch):
    """Collate variable-length trial sequences with padding.

    Returns
    -------
    padded_hg : (batch, max_len, n_channels) float32
    padded_targets : (batch, max_len, target_dim) or (batch,) for word labels
    lengths : (batch,) int
    """
    hg_list = [item[0] for item in batch]
    target_list = [item[1] for item in batch]
    lengths = torch.tensor([len(h) for h in hg_list])

    # Check if targets are scalars (word classification)
    if isinstance(target_list[0], (int, np.integer)):
        padded_hg = torch.nn.utils.rnn.pad_sequence(hg_list, batch_first=True)
        targets = torch.tensor(target_list, dtype=torch.long)
        return padded_hg, targets, lengths

    # Variable-length targets (LPC, VAD)
    padded_hg = torch.nn.utils.rnn.pad_sequence(hg_list, batch_first=True)
    padded_targets = torch.nn.utils.rnn.pad_sequence(target_list, batch_first=True)
    return padded_hg, padded_targets, lengths


def make_dataloader(dataset: Dataset, batch_size: int = 1,
                    shuffle: bool = True, **kwargs) -> DataLoader:
    """Create a DataLoader with appropriate collate function."""
    if isinstance(dataset, SequentialFrameDataset):
        # Sequential data — no padding needed, batch_size=1
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, **kwargs)
    else:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=collate_variable_length, **kwargs,
        )
