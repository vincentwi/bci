#!/usr/bin/env python3
"""
GPU-Accelerated Preprocessing for Speech BCI
=============================================
Uses torch FFT for bandpass filtering (numerically stable),
torch avg_pool1d for windowed averaging, torch STFT for mel.
"""

import numpy as np
import torch
import torch.nn.functional as F
import scipy.signal as ssig
import scipy.io as sio
import scipy.io.wavfile as wavfile
import pandas as pd
from pathlib import Path

from preprocess import build_mel_filterbank

N_ECOG = 128
FS_NEURAL = 1000


def _freq_domain_bandpass(x_ct, fs, f_lo, f_hi, device):
    """Zero-phase bandpass filter via FFT (GPU). Equivalent to filtfilt.

    Args:
        x_ct: (C, T) tensor on device
        fs: sampling rate
        f_lo, f_hi: passband edges in Hz
    Returns:
        filtered: (C, T) tensor on device
    """
    T = x_ct.shape[1]
    freqs = torch.fft.rfftfreq(T, d=1.0/fs, device=device)  # (T//2+1,)

    # Build ideal bandpass mask with smooth roll-off (raised cosine edges)
    rolloff = 2.0  # Hz width of transition band
    mask = torch.zeros_like(freqs)
    # Rising edge
    rise = ((freqs - (f_lo - rolloff)) / rolloff).clamp(0, 1)
    # Falling edge
    fall = (((f_hi + rolloff) - freqs) / rolloff).clamp(0, 1)
    mask = (rise * fall).clamp(0, 1)
    # Apply squared cosine for smoother roll-off
    mask = 0.5 * (1 - torch.cos(mask * torch.pi))

    X = torch.fft.rfft(x_ct, dim=1)  # (C, T//2+1)
    X_filt = X * mask.unsqueeze(0)
    return torch.fft.irfft(X_filt, n=T, dim=1)


def extract_hg_gpu(ecog_np, device, fs=1000, win_ms=50, hop_ms=10):
    """GPU-accelerated high-gamma extraction.

    1. CAR per 64-ch grid
    2. FFT bandpass 70-117 Hz + 123-170 Hz (zero-phase, no numerical issues)
    3. Squared amplitude
    4. avg_pool1d for windowed mean
    5. Log transform
    """
    ecog = torch.from_numpy(ecog_np).float().to(device)  # (T, C)
    T, C = ecog.shape

    # 1. CAR
    ecog[:, :64] -= ecog[:, :64].mean(dim=1, keepdim=True)
    ecog[:, 64:] -= ecog[:, 64:].mean(dim=1, keepdim=True)

    # 2. FFT bandpass on GPU — (C, T) layout
    ecog_ct = ecog.T  # (C, T)
    filt_lo = _freq_domain_bandpass(ecog_ct, fs, 70, 117, device)
    filt_hi = _freq_domain_bandpass(ecog_ct, fs, 123, 170, device)

    # 3. Squared amplitude
    pwr = filt_lo ** 2 + filt_hi ** 2  # (C, T)

    # 4. Windowed log-mean via avg_pool1d
    w = int(win_ms / 1000 * fs)
    h = int(hop_ms / 1000 * fs)
    pwr_3d = pwr.unsqueeze(0)  # (1, C, T)
    pooled = F.avg_pool1d(pwr_3d, kernel_size=w, stride=h)  # (1, C, n_fr)

    # 5. Log
    feat = torch.log(pooled + 1e-10).squeeze(0).T  # (n_fr, C)

    n_fr = feat.shape[0]
    times = (np.arange(n_fr) * h + w // 2) / fs

    return feat.cpu().numpy(), times


def mel_frames_gpu(audio_np, fs_a, device, hop_ms=10, n_mels=40, n_fft=1024, fmax=8000):
    """GPU-accelerated mel spectrogram using torch STFT."""
    hop = int(hop_ms / 1000 * fs_a)
    fb = build_mel_filterbank(fs_a, n_fft, n_mels, fmin=0, fmax=fmax)

    audio_t = torch.from_numpy(audio_np.astype(np.float64)).float().to(device)
    window = torch.hann_window(n_fft, device=device)

    Zxx = torch.stft(audio_t, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                     window=window, return_complex=True)
    S_power = Zxx.abs() ** 2

    fb_t = torch.from_numpy(fb).float().to(device)
    mel_spec = fb_t @ S_power
    mel_db = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
    mel_db -= mel_db.max()

    return mel_db.T.cpu().numpy()


def load_run_gpu(data_dir, run_id):
    """Load raw data from a single run (I/O only)."""
    data_dir = Path(data_dir)
    mat = sio.loadmat(data_dir / f'KeywordReading_Overt_{run_id}.mat')
    ecog = mat['signal'][:, :N_ECOG].astype(np.float64) * 0.25
    try:
        stim = mat['states'][0, 0]['StimulusCode'].flatten()
    except (IndexError, ValueError, TypeError):
        stim = mat['states'].flatten()

    fs_audio, audio = wavfile.read(
        data_dir / f'KeywordReading_Overt_{run_id}.wav')

    trials = []
    lab_path = data_dir / f'KeywordReading_Overt_{run_id}_trials.lab'
    with open(lab_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                trials.append(dict(
                    start=float(parts[0]),
                    end=float(parts[1]),
                    word=parts[2]))

    return ecog, audio, fs_audio, pd.DataFrame(trials), stim


def load_session_gpu(data_base, session_name, device, split='train'):
    """Load all runs from a session with GPU-accelerated preprocessing."""
    if split in ('test', 'validation'):
        session_dir = Path(data_base) / split / session_name
    else:
        session_dir = Path(data_base) / 'train' / session_name

    syll_dir = Path(data_base) / 'syllable' / session_name
    syll_path = syll_dir / 'SyllableRepetition_Overt.mat'

    if not syll_path.exists():
        syll_dates = sorted([d.name for d in (Path(data_base) / 'syllable').iterdir()
                             if d.is_dir() and (d / 'SyllableRepetition_Overt.mat').exists()])
        closest = min(syll_dates, key=lambda d: abs(
            int(d.replace('_', '')) - int(session_name.replace('_', ''))))
        syll_path = Path(data_base) / 'syllable' / closest / 'SyllableRepetition_Overt.mat'
        print(f' (syll:{closest})', end='', flush=True)

    syll = sio.loadmat(str(syll_path))
    syll_ecog = syll['signal'][:, :N_ECOG].astype(np.float64) * 0.25
    syll_hg, _ = extract_hg_gpu(syll_ecog, device, FS_NEURAL)
    hg_mu = syll_hg.mean(axis=0)
    hg_sd = syll_hg.std(axis=0)

    runs = sorted([f.stem.split('_')[-1] for f in session_dir.glob('KeywordReading_Overt_R*.mat')])
    session_data = []

    for run_id in runs:
        ecog, audio, fs_audio, trials, stim = load_run_gpu(session_dir, run_id)
        hg, hg_t = extract_hg_gpu(ecog, device, FS_NEURAL)
        hg_z = (hg - hg_mu) / (hg_sd + 1e-10)
        mel = mel_frames_gpu(audio, fs_audio, device)

        session_data.append(dict(
            hg=hg_z, hg_t=hg_t, mel=mel, trials=trials,
            fs_audio=fs_audio, run=run_id, session=session_name))

    return session_data


if __name__ == '__main__':
    import time

    device = torch.device('cuda:0')
    data_dir = '/mnt/home/vincent.wilmet/data/train/2022_09_22'

    mat = sio.loadmat(f'{data_dir}/KeywordReading_Overt_R01.mat')
    ecog = mat['signal'][:, :N_ECOG].astype(np.float64) * 0.25

    # Warmup
    _ = extract_hg_gpu(ecog[:1000], device)

    # GPU benchmark
    torch.cuda.synchronize(device)
    t0 = time.time()
    hg_gpu, _ = extract_hg_gpu(ecog, device)
    torch.cuda.synchronize(device)
    gpu_time = time.time() - t0
    print(f'GPU extract_hg: {gpu_time:.3f}s  shape={hg_gpu.shape}')

    # CPU benchmark
    from preprocess import extract_hg
    t0 = time.time()
    hg_cpu, _ = extract_hg(ecog, FS_NEURAL)
    cpu_time = time.time() - t0
    print(f'CPU extract_hg: {cpu_time:.3f}s  shape={hg_cpu.shape}')
    print(f'Speedup: {cpu_time/gpu_time:.1f}x')

    # Full session
    print('\nFull session GPU load:')
    t0 = time.time()
    data = load_session_gpu('/mnt/home/vincent.wilmet/data', '2022_09_22', device)
    elapsed = time.time() - t0
    print(f'  4 runs: {elapsed:.1f}s, {sum(len(d["trials"]) for d in data)} trials')

    # CPU full session comparison
    from train_multisession import load_session
    t0 = time.time()
    data_cpu = load_session('/mnt/home/vincent.wilmet/data', '2022_09_22', split='train')
    cpu_elapsed = time.time() - t0
    print(f'  CPU: {cpu_elapsed:.1f}s')
    print(f'  Session speedup: {cpu_elapsed/elapsed:.1f}x')
