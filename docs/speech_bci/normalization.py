"""Per-day z-score normalization using syllable repetition baselines."""

from pathlib import Path

import numpy as np
import scipy.io as sio

from . import config
from .signal import extract_hg


def compute_day_stats(syllable_mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load SyllableRepetition .mat, extract HG, return (mean, std) per channel.

    Each session day has its own syllable repetition recording that serves
    as a normalization baseline. Electrode impedance drifts across days,
    so raw power values shift — this corrects for that.

    Parameters
    ----------
    syllable_mat_path : path to SyllableRepetition_Overt.mat

    Returns
    -------
    mu : (n_channels,) mean HG power per channel
    sd : (n_channels,) std HG power per channel
    """
    mat = sio.loadmat(str(syllable_mat_path))
    ecog = mat["signal"][:, :config.N_ECOG].astype(np.float64) * config.GAIN
    hg, _ = extract_hg(ecog)
    mu = hg.mean(axis=0)
    sd = hg.std(axis=0)
    return mu, sd


def normalize_hg(hg: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Z-score normalize high-gamma features.

    Parameters
    ----------
    hg : (n_frames, n_channels) — raw HG features
    mu : (n_channels,) — per-channel mean from syllable baseline
    sd : (n_channels,) — per-channel std from syllable baseline

    Returns
    -------
    normalized : (n_frames, n_channels)
    """
    return (hg - mu) / (sd + 1e-10)


def build_normalization_cache(
    syllable_dir: Path,
    day_ids: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Precompute (mu, sd) for every session day.

    Parameters
    ----------
    syllable_dir : directory containing per-day syllable repetition folders
    day_ids : which days to process (defaults to all known days)

    Returns
    -------
    cache : {day_id: (mu, sd)} where mu, sd are (n_channels,) arrays
    """
    day_ids = day_ids or config.ALL_DAYS
    cache = {}
    for day_id in day_ids:
        syll_path = syllable_dir / day_id / "SyllableRepetition_Overt.mat"
        if syll_path.exists():
            mu, sd = compute_day_stats(syll_path)
            cache[day_id] = (mu, sd)
    return cache
