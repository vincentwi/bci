"""I/O utilities for loading BCI2000 .mat files, .wav audio, .lab trials."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.io.wavfile as wavfile

from . import config


@dataclass
class RunData:
    ecog: np.ndarray           # (n_samples, 128) float64, in µV
    audio: np.ndarray          # (n_audio_samples,) int16 or float32
    fs_audio: int              # typically 16000
    trials: pd.DataFrame       # columns: start, end, word
    stim_code: np.ndarray      # (n_samples,) uint16
    day_id: str = ""
    run_id: str = ""


def load_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load BCI2000 .mat file, return (ecog_uv, stimulus_code).

    ecog_uv: (n_samples, 128) float64 in microvolts
    stimulus_code: (n_samples,) uint16 — 0=silence, 1-6=word
    """
    mat = sio.loadmat(str(path))
    ecog = mat["signal"][:, :config.N_ECOG].astype(np.float64) * config.GAIN

    # Handle two different mat file formats for states
    states = mat["states"]
    if states.dtype.names is not None:
        # Structured array: states[0,0]["StimulusCode"]
        stim = states[0, 0]["StimulusCode"].flatten().astype(np.uint16)
    else:
        # Flat array: states IS the stimulus code directly
        stim = states.flatten().astype(np.uint16)
    return ecog, stim


def load_trials(path: Path) -> pd.DataFrame:
    """Parse .lab trial file → DataFrame with columns: start, end, word."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                rows.append({
                    "start": float(parts[0]),
                    "end": float(parts[1]),
                    "word": parts[2],
                })
    return pd.DataFrame(rows)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load .wav file, return (audio_array, sample_rate)."""
    fs, audio = wavfile.read(str(path))
    return audio, fs


def _find_file(directory: Path, pattern: str) -> Path:
    """Find a single file matching a glob pattern in directory."""
    matches = list(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    return matches[0]


def _detect_prefix(day_dir: Path) -> str:
    """Auto-detect whether this day uses KeywordReading or KeywordSynthesis prefix."""
    if list(day_dir.glob("KeywordSynthesis_Overt_*.mat")):
        return config.ONLINE_PREFIX
    return config.TRAIN_PREFIX


def discover_runs(day_dir: Path) -> list[str]:
    """Auto-discover available run IDs in a day directory."""
    prefix = _detect_prefix(day_dir)
    mat_files = sorted(day_dir.glob(f"{prefix}_R*.mat"))
    runs = []
    for f in mat_files:
        # Extract run ID: KeywordReading_Overt_R01.mat → R01
        stem = f.stem  # KeywordReading_Overt_R01
        run_id = stem.split("_")[-1]  # R01
        runs.append(run_id)
    return runs


def load_run(data_dir: Path, day_id: str, run_id: str) -> RunData:
    """Load all files for a single run.

    Auto-detects whether the day uses KeywordReading or KeywordSynthesis prefix.
    """
    day_dir = data_dir / day_id
    prefix = f"{_detect_prefix(day_dir)}_{run_id}"

    ecog, stim = load_mat(day_dir / f"{prefix}.mat")
    audio, fs_a = load_audio(day_dir / f"{prefix}.wav")
    trials = load_trials(day_dir / f"{prefix}_trials.lab")

    return RunData(
        ecog=ecog,
        audio=audio,
        fs_audio=fs_a,
        trials=trials,
        stim_code=stim,
        day_id=day_id,
        run_id=run_id,
    )


def load_day(data_dir: Path, day_id: str,
             runs: list[str] | None = None) -> list[RunData]:
    """Load all runs for a session day. Auto-discovers runs if not specified."""
    day_dir = data_dir / day_id
    if runs is None:
        runs = discover_runs(day_dir)
    if not runs:
        runs = config.RUNS  # fallback
    results = []
    for run_id in runs:
        prefix = _detect_prefix(day_dir)
        mat_path = day_dir / f"{prefix}_{run_id}.mat"
        if mat_path.exists():
            results.append(load_run(data_dir, day_id, run_id))
    return results
