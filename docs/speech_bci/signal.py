"""Signal processing: CAR, high-gamma extraction, bad channel correction.

Supports GPU-accelerated processing via PyTorch when a CUDA device is available.
"""

import numpy as np
import scipy.signal as ssig
import torch

from . import config

# Default GPU device for signal processing (set by caller)
_signal_device = None


def set_signal_device(device: str | None):
    """Set the device used for GPU-accelerated signal processing."""
    global _signal_device
    _signal_device = device


def _get_device():
    if _signal_device is not None:
        return _signal_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def common_average_reference(ecog: np.ndarray,
                              grid1_mask: np.ndarray | None = None,
                              grid2_mask: np.ndarray | None = None) -> np.ndarray:
    """Common average reference per grid.

    Each 8x8 grid (channels 0-63 and 64-127) has its own common noise,
    so CAR is applied separately to each grid.
    """
    car = ecog.copy()
    g1 = slice(0, config.GRID_SIZE)
    g2 = slice(config.GRID_SIZE, config.N_ECOG)
    car[:, g1] -= car[:, g1].mean(axis=1, keepdims=True)
    car[:, g2] -= car[:, g2].mean(axis=1, keepdims=True)
    return car


def _bandpass_fft_gpu(signal_t: torch.Tensor, lo: float, hi: float,
                      fs: int, order: int = 8) -> torch.Tensor:
    """FFT-based bandpass filter on GPU. signal_t: (n_samples, n_channels)."""
    n = signal_t.shape[0]
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs, device=signal_t.device)

    # Build frequency-domain Butterworth-like filter
    f_center = (lo + hi) / 2.0
    f_width = (hi - lo) / 2.0
    # Butterworth magnitude response approximation
    normalized = (freqs - f_center) / f_width
    H = 1.0 / torch.sqrt(1.0 + normalized.pow(2 * order))
    # Zero out frequencies outside passband with margin
    H = H * (freqs >= lo * 0.5).float() * (freqs <= hi * 1.5).float()

    # Apply filter in frequency domain
    S = torch.fft.rfft(signal_t, dim=0)
    filtered = torch.fft.irfft(S * H.unsqueeze(1), n=n, dim=0)
    return filtered


def extract_hg_gpu(ecog: np.ndarray, fs: int = config.FS,
                   win_ms: int = config.HG_WIN_MS,
                   hop_ms: int = config.HG_HOP_MS,
                   device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated HG extraction using FFT bandpass and torch.unfold."""
    device = device or _get_device()
    n_samp, n_ch = ecog.shape

    # Step 1: CAR (fast on CPU, data is small relative to GPU transfer)
    car = common_average_reference(ecog)

    # Move to GPU
    car_t = torch.from_numpy(car.astype(np.float32)).to(device)

    # Step 2+3: FFT bandpass and square on GPU
    power_t = torch.zeros_like(car_t)
    for lo, hi in config.HG_BANDS:
        bp = _bandpass_fft_gpu(car_t, lo, hi, fs, order=config.BUTTER_ORDER)
        power_t += bp ** 2

    # Step 4: Windowed mean using unfold on GPU
    w = int(win_ms / 1000 * fs)
    h = int(hop_ms / 1000 * fs)

    # unfold: (n_samp, n_ch) → transpose to (n_ch, n_samp) → unfold → mean
    power_ch = power_t.t()  # (n_ch, n_samp)
    windows = power_ch.unfold(1, w, h)  # (n_ch, n_frames, w)
    mean_power = windows.mean(dim=2)  # (n_ch, n_frames)

    # Step 5: Log transform
    features_t = torch.log(mean_power + 1e-10).t()  # (n_frames, n_ch)

    n_fr = features_t.shape[0]
    times = (np.arange(n_fr) * h + w // 2) / fs

    return features_t.cpu().numpy().astype(np.float64), times


def extract_hg_cpu(ecog: np.ndarray, fs: int = config.FS,
                   win_ms: int = config.HG_WIN_MS,
                   hop_ms: int = config.HG_HOP_MS) -> tuple[np.ndarray, np.ndarray]:
    """CPU fallback HG extraction using scipy IIR filter."""
    n_samp, n_ch = ecog.shape
    car = common_average_reference(ecog)

    power = np.zeros_like(car)
    for lo, hi in config.HG_BANDS:
        sos = ssig.butter(config.BUTTER_ORDER, [lo, hi],
                          btype="band", fs=fs, output="sos")
        power += ssig.sosfilt(sos, car, axis=0) ** 2

    w = int(win_ms / 1000 * fs)
    h = int(hop_ms / 1000 * fs)
    n_fr = (n_samp - w) // h + 1

    shape = (n_fr, w, n_ch)
    strides = (power.strides[0] * h, power.strides[0], power.strides[1])
    windows = np.lib.stride_tricks.as_strided(power, shape=shape, strides=strides)

    features = np.log(windows.mean(axis=1) + 1e-10)
    times = (np.arange(n_fr) * h + w // 2) / fs

    return features, times


def extract_hg(ecog: np.ndarray, fs: int = config.FS,
               win_ms: int = config.HG_WIN_MS,
               hop_ms: int = config.HG_HOP_MS) -> tuple[np.ndarray, np.ndarray]:
    """Extract high-gamma log-power features. Uses GPU if available."""
    device = _get_device()
    if device != "cpu" and torch.cuda.is_available():
        return extract_hg_gpu(ecog, fs, win_ms, hop_ms, device)
    return extract_hg_cpu(ecog, fs, win_ms, hop_ms)


def get_grid_neighbors(channel_idx: int, grid_shape: tuple = (8, 8)) -> list[int]:
    """Get indices of spatial neighbors for a channel on its grid.

    Channels 0-63 are grid 1, 64-127 are grid 2.
    Each grid is 8x8. Returns up to 8 neighbors (including diagonals).
    """
    grid_offset = 0 if channel_idx < config.GRID_SIZE else config.GRID_SIZE
    local_idx = channel_idx - grid_offset
    row, col = divmod(local_idx, grid_shape[1])

    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
                neighbors.append(grid_offset + nr * grid_shape[1] + nc)
    return neighbors


def correct_bad_channels(hg: np.ndarray, bad_channels: list[int],
                          grid_shape: tuple = (8, 8)) -> np.ndarray:
    """Replace bad channel values with the mean of their spatial neighbors.

    This is a simplified version of the Rousseeuw method mentioned in the paper.

    Parameters
    ----------
    hg : (n_frames, n_channels) — high-gamma features
    bad_channels : list of channel indices to correct
    grid_shape : shape of each electrode grid

    Returns
    -------
    corrected : (n_frames, n_channels) — corrected features
    """
    corrected = hg.copy()
    for ch in bad_channels:
        neighbors = get_grid_neighbors(ch, grid_shape)
        good_neighbors = [n for n in neighbors if n not in bad_channels]
        if good_neighbors:
            corrected[:, ch] = corrected[:, good_neighbors].mean(axis=1)
    return corrected
