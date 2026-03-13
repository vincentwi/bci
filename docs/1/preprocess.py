#!/usr/bin/env python3
"""
Preprocessing Pipeline: Raw ECoG → Training-Ready Features
===========================================================

Reads raw .mat/.wav/.lab files and produces preprocessed .npz files
that the training script can load instantly.

Steps:
  1. Load raw ECoG signal from BCI2000 .mat files
  2. Common average re-reference (per electrode grid)
  3. High-gamma extraction: bandpass 70-170 Hz → squared amplitude →
     windowed log-mean power → 100 Hz frame rate
  4. Z-score normalization using syllable repetition baseline
  5. Mel spectrogram from audio (no librosa — uses scipy STFT)
  6. Segment trials using .lab timing files
  7. Save preprocessed arrays as .npz

Usage:
    python preprocess.py --data-dir data/train/2022_09_22 \\
                         --syll-dir data/syllable/2022_09_22 \\
                         --output-dir data/processed

    # Custom parameters
    python preprocess.py --data-dir data/train/2022_09_22 \\
                         --syll-dir data/syllable/2022_09_22 \\
                         --output-dir data/processed \\
                         --n-mels 40 --hg-win-ms 50 --hg-hop-ms 10 \\
                         --pre-context 0.5 --post-context 0.5

Outputs:
    data/processed/
    ├── config.json         — preprocessing parameters (for reproducibility)
    ├── label_encoder.json  — word → class index mapping
    ├── normalization.npz   — hg_mu, hg_sd from syllable baseline
    ├── R01_hg.npz          — per-run HG features + times
    ├── R01_mel.npz         — per-run mel spectrograms
    ├── ...
    ├── cls_train.npz       — classification training segments + labels
    ├── cls_test.npz        — classification test segments + labels
    ├── ac_train.npz        — acoustic decoder training (HG + mel pairs)
    └── ac_test.npz         — acoustic decoder test (HG + mel pairs)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as ssig
import scipy.io.wavfile as wavfile


# ============================================================
# Constants
# ============================================================
N_ECOG = 128
FS_NEURAL = 1000  # Hz, ECoG sampling rate
N_FFT = 1024

RUNS = ['R01', 'R02', 'R03', 'R04']
TRAIN_RUNS = ['R01', 'R02', 'R03']
TEST_RUN = 'R04'


# ============================================================
# High-Gamma Extraction
# ============================================================
def extract_hg(ecog, fs=1000, win_ms=50, hop_ms=10):
    """Extract log high-gamma power envelope from raw ECoG.

    Pipeline:
      1. Common average re-reference (per 64-channel grid)
      2. Bandpass 70-117 Hz + 123-170 Hz (avoids 120 Hz line noise harmonic)
      3. Squared amplitude (instantaneous power)
      4. Sliding window mean + log transform
      5. Output at 100 Hz (10 ms hop)

    Args:
        ecog: (n_samples, n_channels) raw ECoG, float64
        fs: sampling rate in Hz
        win_ms: analysis window in ms
        hop_ms: hop size in ms

    Returns:
        feat: (n_frames, n_channels) log high-gamma power
        times: (n_frames,) center time of each frame in seconds
    """
    n_samp, n_ch = ecog.shape

    # Step 1: Common average reference per grid
    car = ecog.copy()
    car[:, :64] -= car[:, :64].mean(axis=1, keepdims=True)
    car[:, 64:] -= car[:, 64:].mean(axis=1, keepdims=True)

    # Step 2: Bandpass for high-gamma (split around 120 Hz)
    sos_lo = ssig.butter(8, [70, 117], btype='band', fs=fs, output='sos')
    sos_hi = ssig.butter(8, [123, 170], btype='band', fs=fs, output='sos')
    pwr = ssig.sosfilt(sos_lo, car, axis=0)**2 + ssig.sosfilt(sos_hi, car, axis=0)**2

    # Step 3: Windowed log-mean power (vectorized via stride_tricks)
    w = int(win_ms / 1000 * fs)   # window in samples
    h = int(hop_ms / 1000 * fs)   # hop in samples
    n_fr = (n_samp - w) // h + 1

    shape = (n_fr, w, n_ch)
    strides = (pwr.strides[0] * h, pwr.strides[0], pwr.strides[1])
    windows = np.lib.stride_tricks.as_strided(pwr, shape=shape, strides=strides)
    feat = np.log(windows.mean(axis=1) + 1e-10)

    times = (np.arange(n_fr) * h + w // 2) / fs
    return feat, times


# ============================================================
# Mel Spectrogram (no librosa)
# ============================================================
_MEL_FB_CACHE = {}


def build_mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Build HTK-style triangular mel filterbank.

    Args:
        sr: audio sample rate
        n_fft: FFT size
        n_mels: number of mel bins
        fmin: lowest frequency
        fmax: highest frequency (default sr/2)

    Returns:
        fb: (n_mels, n_fft//2+1) filterbank matrix
    """
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key in _MEL_FB_CACHE:
        return _MEL_FB_CACHE[key]
    if fmax is None:
        fmax = sr / 2.0

    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fft_freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        lo, ctr, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up_slope = (fft_freqs - lo) / (ctr - lo + 1e-10)
        down_slope = (hi - fft_freqs) / (hi - ctr + 1e-10)
        fb[m] = np.maximum(0, np.minimum(up_slope, down_slope))

    _MEL_FB_CACHE[key] = fb
    return fb


def mel_frames(audio, fs_a, hop_ms=10, n_mels=40, n_fft=1024, fmax=8000):
    """Compute mel spectrogram in dB from audio waveform.

    Uses scipy STFT + custom mel filterbank (no librosa dependency).

    Args:
        audio: 1D audio waveform
        fs_a: audio sample rate (Hz)
        hop_ms: hop size in ms
        n_mels: number of mel bins
        n_fft: FFT size
        fmax: max frequency for mel filterbank

    Returns:
        mel_db: (n_frames, n_mels) mel spectrogram in dB, normalized to max=0
    """
    hop = int(hop_ms / 1000 * fs_a)
    fb = build_mel_filterbank(fs_a, n_fft, n_mels, fmin=0, fmax=fmax)
    _, _, Zxx = ssig.stft(audio.astype(np.float64), fs=fs_a,
                          nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft)
    S_power = np.abs(Zxx) ** 2
    mel_spec = fb @ S_power
    mel_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    mel_db -= mel_db.max()
    return mel_db.T


# ============================================================
# Data Loading
# ============================================================
def load_run(data_dir, run_id):
    """Load a single run from BCI2000 .mat format.

    Returns:
        ecog: (n_samples, 128) ECoG in microvolts
        audio: 1D audio waveform
        fs_audio: audio sample rate
        trials: DataFrame with start, end, word columns
        stim: (n_samples,) stimulus code per sample
    """
    data_dir = Path(data_dir)

    mat = sio.loadmat(data_dir / f'KeywordReading_Overt_{run_id}.mat')
    ecog = mat['signal'][:, :N_ECOG].astype(np.float64) * 0.25  # int16 → µV
    stim = mat['states'][0, 0]['StimulusCode'].flatten()

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


def seg_trial(hg, hg_t, row, pre=0.5, post=0.5):
    """Extract HG segment for a single trial with context padding.

    Args:
        hg: (n_frames, n_channels) HG features
        hg_t: (n_frames,) frame timestamps in seconds
        row: trial dict/Series with 'start' and 'end' keys
        pre: seconds of context before trial onset
        post: seconds of context after trial offset
    """
    mask = (hg_t >= row['start'] - pre) & (hg_t < row['end'] + post)
    return hg[mask]


# ============================================================
# Load all runs (for use by training script inline mode)
# ============================================================
def load_all_data(data_dir, syll_dir, hg_win_ms=50, hg_hop_ms=10):
    """Load all runs and compute baseline normalization.

    Returns:
        all_runs: dict of run_id -> {hg, hg_t, trials, audio, fs_audio, stim}
        hg_mu: baseline mean per channel
        hg_sd: baseline std per channel
    """
    data_dir = Path(data_dir)
    syll_dir = Path(syll_dir)

    # Baseline from syllable repetition
    print('Computing baseline from syllable repetition...')
    syll = sio.loadmat(syll_dir / 'SyllableRepetition_Overt.mat')
    syll_ecog = syll['signal'][:, :N_ECOG].astype(np.float64) * 0.25
    syll_hg, _ = extract_hg(syll_ecog, FS_NEURAL, hg_win_ms, hg_hop_ms)
    hg_mu = syll_hg.mean(axis=0)
    hg_sd = syll_hg.std(axis=0)

    all_runs = {}
    for run in RUNS:
        print(f'  Loading {run}...', end=' ', flush=True)
        ecog, audio, fs_audio, trials, stim = load_run(data_dir, run)
        hg, hg_t = extract_hg(ecog, FS_NEURAL, hg_win_ms, hg_hop_ms)
        hg_z = (hg - hg_mu) / (hg_sd + 1e-10)
        all_runs[run] = dict(
            hg=hg_z, hg_t=hg_t, trials=trials,
            audio=audio, fs_audio=fs_audio, stim=stim)
        print(f'{len(trials)} trials, {hg_z.shape[0]} frames')

    return all_runs, hg_mu, hg_sd


# ============================================================
# Main Preprocessing Pipeline
# ============================================================
def preprocess(data_dir, syll_dir, output_dir, n_mels=40, hg_win_ms=50,
               hg_hop_ms=10, pre_context=0.5, post_context=0.5,
               fmax=8000):
    """Run the full preprocessing pipeline.

    1. Compute baseline normalization from syllable repetition
    2. For each run: extract HG, compute mel, save per-run files
    3. Segment trials for classification and acoustic decoding
    4. Save train/test splits
    """
    data_dir = Path(data_dir)
    syll_dir = Path(syll_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Save config for reproducibility ──
    config = dict(
        n_ecog=N_ECOG, fs_neural=FS_NEURAL, n_mels=n_mels, n_fft=N_FFT,
        hg_win_ms=hg_win_ms, hg_hop_ms=hg_hop_ms,
        pre_context=pre_context, post_context=post_context,
        fmax=fmax, runs=RUNS, train_runs=TRAIN_RUNS, test_run=TEST_RUN,
        data_dir=str(data_dir), syll_dir=str(syll_dir))
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Saved config.json')

    # ── Step 1: Syllable baseline ──
    print('\n[1/4] Computing baseline from syllable repetition...')
    syll = sio.loadmat(syll_dir / 'SyllableRepetition_Overt.mat')
    syll_ecog = syll['signal'][:, :N_ECOG].astype(np.float64) * 0.25
    syll_hg, _ = extract_hg(syll_ecog, FS_NEURAL, hg_win_ms, hg_hop_ms)
    hg_mu = syll_hg.mean(axis=0)
    hg_sd = syll_hg.std(axis=0)
    np.savez(output_dir / 'normalization.npz', hg_mu=hg_mu, hg_sd=hg_sd)
    print(f'  Baseline: {syll_hg.shape[0]} frames, saved normalization.npz')

    # ── Step 2: Process each run ──
    print('\n[2/4] Processing runs...')
    all_runs = {}
    for run in RUNS:
        print(f'  {run}:', end=' ', flush=True)

        ecog, audio, fs_audio, trials, stim = load_run(data_dir, run)
        print(f'{len(trials)} trials,', end=' ')

        # HG extraction + normalization
        hg, hg_t = extract_hg(ecog, FS_NEURAL, hg_win_ms, hg_hop_ms)
        hg_z = (hg - hg_mu) / (hg_sd + 1e-10)
        np.savez(output_dir / f'{run}_hg.npz',
                 hg=hg_z, hg_t=hg_t, hg_raw=hg)
        print(f'{hg_z.shape[0]} HG frames,', end=' ')

        # Mel spectrogram
        mel = mel_frames(audio, fs_audio, hop_ms=hg_hop_ms,
                         n_mels=n_mels, fmax=fmax)
        np.savez(output_dir / f'{run}_mel.npz', mel=mel, fs_audio=fs_audio)
        print(f'{mel.shape[0]} mel frames')

        all_runs[run] = dict(
            hg=hg_z, hg_t=hg_t, mel=mel,
            trials=trials, stim=stim, fs_audio=fs_audio)

    # ── Step 3: Segment trials ──
    print('\n[3/4] Segmenting trials...')
    all_words = sorted({w for r in all_runs.values()
                        for w in r['trials']['word']})
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    with open(output_dir / 'label_encoder.json', 'w') as f:
        json.dump(word_to_idx, f, indent=2)
    print(f'  Classes: {all_words}')

    # Classification segments
    cls_train_segs, cls_train_labels = [], []
    cls_test_segs, cls_test_labels = [], []
    cls_train_words, cls_test_words = [], []

    # Acoustic segments
    ac_train_hg, ac_train_mel = [], []
    ac_test_hg, ac_test_mel = [], []

    for run in RUNS:
        r = all_runs[run]
        is_test = (run == TEST_RUN)

        for _, row in r['trials'].iterrows():
            # Classification segment
            seg = seg_trial(r['hg'], r['hg_t'], row,
                            pre=pre_context, post=post_context)
            if len(seg) < 10:
                continue
            label = word_to_idx[row['word']]

            if is_test:
                cls_test_segs.append(seg)
                cls_test_labels.append(label)
                cls_test_words.append(row['word'])
            else:
                cls_train_segs.append(seg)
                cls_train_labels.append(label)
                cls_train_words.append(row['word'])

            # Acoustic segment (frame-aligned HG + mel)
            sf = int(row['start'] * 100)
            ef = int(row['end'] * 100)
            n_com = min(len(r['hg']), len(r['mel']))
            if ef > n_com:
                continue

            hg_seg = r['hg'][:n_com][sf:ef]
            mel_seg = r['mel'][:n_com][sf:ef]

            if is_test:
                ac_test_hg.append(hg_seg)
                ac_test_mel.append(mel_seg)
            else:
                ac_train_hg.append(hg_seg)
                ac_train_mel.append(mel_seg)

    print(f'  Classification — train: {len(cls_train_segs)}, '
          f'test: {len(cls_test_segs)}')
    print(f'  Acoustic — train: {len(ac_train_hg)}, '
          f'test: {len(ac_test_hg)}')

    # ── Step 4: Save segmented data ──
    print('\n[4/4] Saving segmented datasets...')

    # Classification (variable-length segments stored as object arrays)
    np.savez(output_dir / 'cls_train.npz',
             segs=np.array(cls_train_segs, dtype=object),
             labels=np.array(cls_train_labels),
             words=np.array(cls_train_words),
             allow_pickle=True)
    np.savez(output_dir / 'cls_test.npz',
             segs=np.array(cls_test_segs, dtype=object),
             labels=np.array(cls_test_labels),
             words=np.array(cls_test_words),
             allow_pickle=True)

    # Acoustic (variable-length paired segments)
    np.savez(output_dir / 'ac_train.npz',
             hg_segs=np.array(ac_train_hg, dtype=object),
             mel_segs=np.array(ac_train_mel, dtype=object),
             allow_pickle=True)
    np.savez(output_dir / 'ac_test.npz',
             hg_segs=np.array(ac_test_hg, dtype=object),
             mel_segs=np.array(ac_test_mel, dtype=object),
             allow_pickle=True)

    elapsed = time.time() - t0
    total_trials = len(cls_train_segs) + len(cls_test_segs)
    print(f'\nDone in {elapsed:.1f}s. {total_trials} trials preprocessed.')
    print(f'Output: {output_dir.resolve()}')

    # Summary
    print('\n' + '=' * 50)
    print('PREPROCESSING SUMMARY')
    print('=' * 50)
    train_lens = [len(s) for s in cls_train_segs]
    test_lens = [len(s) for s in cls_test_segs]
    print(f'  Classes:           {len(all_words)} ({", ".join(all_words)})')
    print(f'  Train trials:      {len(cls_train_segs)}')
    print(f'  Test trials:       {len(cls_test_segs)}')
    print(f'  Feature dim:       {N_ECOG} channels')
    print(f'  HG frame rate:     {1000 / hg_hop_ms:.0f} Hz')
    print(f'  Mel bins:          {n_mels}')
    print(f'  Train seg lengths: {min(train_lens)}–{max(train_lens)} '
          f'(mean {np.mean(train_lens):.0f})')
    print(f'  Test seg lengths:  {min(test_lens)}–{max(test_lens)} '
          f'(mean {np.mean(test_lens):.0f})')

    return output_dir


# ============================================================
# Load preprocessed data (for use by training script)
# ============================================================
def load_preprocessed(processed_dir):
    """Load preprocessed .npz files into a dict matching prepare_data() output.

    Returns dict with keys: n_cls, classes, word_to_idx,
        train_X, train_y, test_X, test_y,
        tr_hg_s, tr_mel_s, te_hg_s, te_mel_s
    """
    processed_dir = Path(processed_dir)

    with open(processed_dir / 'label_encoder.json') as f:
        word_to_idx = json.load(f)
    classes = sorted(word_to_idx, key=word_to_idx.get)
    n_cls = len(classes)

    with open(processed_dir / 'config.json') as f:
        config = json.load(f)

    # Classification data
    cls_train = np.load(processed_dir / 'cls_train.npz', allow_pickle=True)
    cls_test = np.load(processed_dir / 'cls_test.npz', allow_pickle=True)

    train_X = list(cls_train['segs'])
    train_y = list(cls_train['labels'])
    test_X = list(cls_test['segs'])
    test_y = list(cls_test['labels'])

    # Acoustic data
    ac_train = np.load(processed_dir / 'ac_train.npz', allow_pickle=True)
    ac_test = np.load(processed_dir / 'ac_test.npz', allow_pickle=True)

    tr_hg_s = list(ac_train['hg_segs'])
    tr_mel_s = list(ac_train['mel_segs'])
    te_hg_s = list(ac_test['hg_segs'])
    te_mel_s = list(ac_test['mel_segs'])

    print(f'Loaded preprocessed data from {processed_dir}')
    print(f'  {n_cls} classes: {classes}')
    print(f'  Train: {len(train_X)} cls, {len(tr_hg_s)} acoustic')
    print(f'  Test:  {len(test_X)} cls, {len(te_hg_s)} acoustic')

    return dict(
        n_cls=n_cls, classes=classes, word_to_idx=word_to_idx,
        config=config,
        train_X=train_X, train_y=train_y,
        test_X=test_X, test_y=test_y,
        tr_hg_s=tr_hg_s, tr_mel_s=tr_mel_s,
        te_hg_s=te_hg_s, te_mel_s=te_mel_s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess raw ECoG data for speech BCI training')
    parser.add_argument('--data-dir', type=str,
                        default='data/train/2022_09_22',
                        help='Directory with .mat/.wav/.lab files')
    parser.add_argument('--syll-dir', type=str,
                        default='data/syllable/2022_09_22',
                        help='Directory with SyllableRepetition_Overt.mat')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed',
                        help='Output directory for preprocessed files')
    parser.add_argument('--n-mels', type=int, default=40)
    parser.add_argument('--hg-win-ms', type=int, default=50,
                        help='HG analysis window in ms')
    parser.add_argument('--hg-hop-ms', type=int, default=10,
                        help='HG hop size in ms (determines frame rate)')
    parser.add_argument('--pre-context', type=float, default=0.5,
                        help='Seconds of pre-trial context for classification')
    parser.add_argument('--post-context', type=float, default=0.5,
                        help='Seconds of post-trial context for classification')
    parser.add_argument('--fmax', type=int, default=8000,
                        help='Max frequency for mel filterbank')
    args = parser.parse_args()

    preprocess(
        data_dir=args.data_dir,
        syll_dir=args.syll_dir,
        output_dir=args.output_dir,
        n_mels=args.n_mels,
        hg_win_ms=args.hg_win_ms,
        hg_hop_ms=args.hg_hop_ms,
        pre_context=args.pre_context,
        post_context=args.post_context,
        fmax=args.fmax)
