#!/usr/bin/env python3
"""
Speech BCI Advanced Models — GPU Training Script
=================================================
Compares BiLSTM, TCN, and Transformer architectures for:
  1. Word classification (HG → word label)
  2. Acoustic decoding (HG → mel spectrogram)
With and without data augmentation, plus Griffin-Lim audio reconstruction.

Usage:
    # From raw data (runs preprocessing inline)
    python train_gpu.py --data-dir data/train/2022_09_22 \
                        --syll-dir data/syllable/2022_09_22 \
                        --output-dir results --epochs 80

    # From preprocessed data (faster startup)
    python train_gpu.py --preprocessed data/processed \
                        --output-dir results --epochs 80

Outputs:
    results/metrics.json        — all training metrics
    results/models/*.pt         — best model checkpoints
    results/audio/*.wav         — Griffin-Lim reconstructed audio
    results/plots/*.png         — comparison figures
    results/predictions.npz     — raw predictions for further analysis
"""

import argparse
import json
import copy
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import scipy.signal as ssig
import scipy.io.wavfile as wavfile
import scipy.interpolate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from models import (WordLSTM, AcLSTM, TCNClassifier, TCNDecoder,
                    TFClassifier, TFDecoder, get_classifier, get_decoder,
                    count_params)
from preprocess import (extract_hg, mel_frames, build_mel_filterbank,
                        load_run, load_all_data, seg_trial, load_preprocessed)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams['figure.dpi'] = 120

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# Constants
# ============================================================
N_ECOG = 128
FS = 1000
N_MELS = 40
N_FFT = 1024
RUNS = ['R01', 'R02', 'R03', 'R04']
TRAIN_RUNS = ['R01', 'R02', 'R03']
TEST_RUN = 'R04'


# ============================================================
# Data Preparation (from raw, when --preprocessed not used)
# ============================================================
def prepare_data(all_runs):
    """Prepare classification and acoustic decoding datasets from raw runs."""
    le = LabelEncoder()
    all_words = sorted({w for r in all_runs.values()
                        for w in r['trials']['word']})
    le.fit(all_words)
    n_cls = len(le.classes_)

    train_X, train_y, test_X, test_y = [], [], [], []
    for run in RUNS:
        r = all_runs[run]
        tgt_X = test_X if run == TEST_RUN else train_X
        tgt_y = test_y if run == TEST_RUN else train_y
        for _, row in r['trials'].iterrows():
            seg = seg_trial(r['hg'], r['hg_t'], row)
            if len(seg) < 10:
                continue
            tgt_X.append(seg)
            tgt_y.append(le.transform([row['word']])[0])

    tr_hg_s, tr_mel_s, te_hg_s, te_mel_s = [], [], [], []
    for run in RUNS:
        r = all_runs[run]
        mel = mel_frames(r['audio'], r['fs_audio'])
        n_com = min(len(r['hg']), len(mel))
        th = te_hg_s if run == TEST_RUN else tr_hg_s
        tm = te_mel_s if run == TEST_RUN else tr_mel_s
        for _, row in r['trials'].iterrows():
            sf = int(row['start'] * 100)
            ef = int(row['end'] * 100)
            if ef > n_com:
                continue
            th.append(r['hg'][:n_com][sf:ef])
            tm.append(mel[:n_com][sf:ef])

    print(f'\nClassification: {n_cls} classes {list(le.classes_)}')
    print(f'  Train: {len(train_X)}, Test: {len(test_X)}')
    print(f'Acoustic: Train: {len(tr_hg_s)}, Test: {len(te_hg_s)}')

    return dict(
        le=le, n_cls=n_cls,
        train_X=train_X, train_y=train_y,
        test_X=test_X, test_y=test_y,
        tr_hg_s=tr_hg_s, tr_mel_s=tr_mel_s,
        te_hg_s=te_hg_s, te_mel_s=te_mel_s,
    )


# ============================================================
# Section 4: Data Augmentation
# ============================================================
def augment_time_warp(x, rate_range=(0.9, 1.1)):
    """Stretch/compress along time axis via interpolation."""
    T, C = x.shape
    rate = np.random.uniform(*rate_range)
    new_T = max(2, int(T * rate))
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, new_T)
    f = scipy.interpolate.interp1d(old_t, x, axis=0, kind='linear',
                                   fill_value='extrapolate')
    return f(new_t)


def augment_noise(x, scale=0.1):
    """Add Gaussian noise scaled to per-channel std."""
    std = x.std(axis=0, keepdims=True) + 1e-10
    return x + np.random.randn(*x.shape) * std * scale


def augment_channel_dropout(x, p=0.1):
    """Zero out random channels."""
    mask = (np.random.rand(x.shape[1]) > p).astype(np.float64)
    return x * mask[np.newaxis, :]


# ============================================================
# Section 5: Dataset Classes
# ============================================================
class TrialDS(Dataset):
    """Classification dataset: zero-pad to max_len."""
    def __init__(self, segs, labels, max_len=None):
        self.labels = labels
        self.max_len = max_len or max(len(s) for s in segs)
        self.data = np.zeros((len(segs), self.max_len, segs[0].shape[1]))
        self.lens = []
        for i, s in enumerate(segs):
            L = min(len(s), self.max_len)
            self.data[i, :L] = s[:L]
            self.lens.append(L)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.FloatTensor(self.data[i]), self.lens[i], self.labels[i]


class AugTrialDS(Dataset):
    """Classification dataset with on-the-fly augmentation."""
    def __init__(self, segs, labels, max_len=None, augment=False):
        self.raw_segs = [s.copy() for s in segs]
        self.labels = labels
        self.max_len = max_len or max(len(s) for s in segs)
        self.augment = augment
        self.n_ch = segs[0].shape[1]
        # Pre-compute lengths for non-augmented case
        self.base_lens = [min(len(s), self.max_len) for s in segs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        seg = self.raw_segs[i].copy()
        if self.augment:
            if np.random.rand() < 0.5:
                seg = augment_time_warp(seg)
            if np.random.rand() < 0.5:
                seg = augment_noise(seg)
            if np.random.rand() < 0.5:
                seg = augment_channel_dropout(seg)
        L = min(len(seg), self.max_len)
        padded = np.zeros((self.max_len, self.n_ch))
        padded[:L] = seg[:L]
        return torch.FloatTensor(padded), L, self.labels[i]


class AcDS(Dataset):
    """Acoustic decoder dataset: paired HG + mel, zero-padded."""
    def __init__(self, hgs, mels, max_len=None):
        self.ml = max_len or max(len(s) for s in hgs)
        self.hg = np.zeros((len(hgs), self.ml, hgs[0].shape[1]))
        self.mel = np.zeros((len(mels), self.ml, mels[0].shape[1]))
        self.lens = []
        for i in range(len(hgs)):
            L = min(len(hgs[i]), self.ml)
            self.hg[i, :L] = hgs[i][:L]
            self.mel[i, :L] = mels[i][:L]
            self.lens.append(L)

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, i):
        return (torch.FloatTensor(self.hg[i]),
                torch.FloatTensor(self.mel[i]),
                self.lens[i])


class AugAcDS(Dataset):
    """Acoustic decoder dataset with on-the-fly augmentation.
    Time warp is applied to BOTH HG and mel simultaneously."""
    def __init__(self, hgs, mels, max_len=None, augment=False):
        self.raw_hgs = [h.copy() for h in hgs]
        self.raw_mels = [m.copy() for m in mels]
        self.ml = max_len or max(len(s) for s in hgs)
        self.augment = augment
        self.n_ch = hgs[0].shape[1]
        self.n_mel = mels[0].shape[1]

    def __len__(self):
        return len(self.raw_hgs)

    def __getitem__(self, i):
        hg = self.raw_hgs[i].copy()
        mel = self.raw_mels[i].copy()

        if self.augment:
            # Time warp: same factor for both HG and mel
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.9, 1.1)
                hg = augment_time_warp(hg, (factor, factor))
                mel = augment_time_warp(mel, (factor, factor))
            # Noise and dropout: HG only
            if np.random.rand() < 0.5:
                hg = augment_noise(hg)
            if np.random.rand() < 0.5:
                hg = augment_channel_dropout(hg)

        L = min(len(hg), self.ml)
        hg_pad = np.zeros((self.ml, self.n_ch))
        mel_pad = np.zeros((self.ml, self.n_mel))
        hg_pad[:L] = hg[:L]
        mel_pad[:L] = mel[:L]
        return torch.FloatTensor(hg_pad), torch.FloatTensor(mel_pad), L


# ============================================================
# Section 6: Training Functions
# ============================================================
def train_classifier(model, tr_dl, te_dl, device, n_epochs=80,
                     lr=1e-3, wd=1e-4, patience=15, label=''):
    """Train a word classifier. Returns metrics dict."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=10, factor=0.5)
    crit = nn.CrossEntropyLoss()

    losses, accs = [], []
    best_acc, best_ep = 0, 0
    best_state = None
    no_improve = 0

    for ep in range(n_epochs):
        model.train()
        el = 0
        for xb, lb, yb in tr_dl:
            xb = xb.to(device)
            yb = torch.as_tensor(yb).long().to(device)
            loss = crit(model(xb, lb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item() * len(yb)
        el /= len(tr_dl.dataset)
        losses.append(el)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, lb, yb in te_dl:
                out = model(xb.to(device), lb)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(yb if isinstance(yb, list) else yb.numpy())
        acc = accuracy_score(labels, preds)
        accs.append(acc)
        sched.step(el)

        if acc > best_acc:
            best_acc = acc
            best_ep = ep + 1
            best_state = copy.deepcopy(model.state_dict())
            best_preds = list(preds)
            best_labels = list(labels)
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 10 == 0:
            print(f'  [{label}] Ep {ep+1:3d}  loss={el:.4f}  acc={acc:.1%}')

        if no_improve >= patience:
            print(f'  [{label}] Early stop at epoch {ep+1}')
            break

    print(f'  [{label}] Best: {best_acc:.1%} (ep {best_ep})')

    # Compute final metrics with best model
    model.load_state_dict(best_state)
    f1 = f1_score(best_labels, best_preds, average='macro')
    cm = confusion_matrix(best_labels, best_preds)

    return dict(
        losses=losses, accs=accs,
        best_acc=best_acc, best_f1=f1, best_epoch=best_ep,
        preds=best_preds, labels=best_labels, cm=cm,
        model_state=best_state)


def train_decoder(model, tr_dl, te_dl, device, n_epochs=80,
                  lr=1e-3, wd=1e-4, patience=15, label=''):
    """Train an acoustic decoder. Returns metrics dict."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=10, factor=0.5)
    mse_fn = nn.MSELoss()

    train_losses, test_losses = [], []
    best_mse = float('inf')
    best_ep = 0
    best_state = None
    no_improve = 0

    for ep in range(n_epochs):
        model.train()
        el, ns = 0, 0
        for hb, mb, lb in tr_dl:
            hb, mb = hb.to(device), mb.to(device)
            pred = model(hb, lb)
            T_pred = pred.size(1)
            mb = mb[:, :T_pred]
            mask = torch.zeros_like(mb)
            for i, L in enumerate(lb):
                mask[i, :min(L, T_pred)] = 1.0
            loss = mse_fn(pred * mask, mb * mask) * mask.numel() / mask.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item() * len(lb)
            ns += len(lb)
        train_losses.append(el / ns)

        model.eval()
        tl, nt = 0, 0
        with torch.no_grad():
            for hb, mb, lb in te_dl:
                hb, mb = hb.to(device), mb.to(device)
                pred = model(hb, lb)
                T_pred = pred.size(1)
                mb = mb[:, :T_pred]
                mask = torch.zeros_like(mb)
                for i, L in enumerate(lb):
                    mask[i, :min(L, T_pred)] = 1.0
                loss = mse_fn(pred * mask, mb * mask) * mask.numel() / mask.sum()
                tl += loss.item() * len(lb)
                nt += len(lb)
        test_mse = tl / nt
        test_losses.append(test_mse)
        sched.step(test_mse)

        if test_mse < best_mse:
            best_mse = test_mse
            best_ep = ep + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 10 == 0:
            print(f'  [{label}] Ep {ep+1:3d}  '
                  f'train={train_losses[-1]:.4f}  test={test_mse:.4f}')

        if no_improve >= patience:
            print(f'  [{label}] Early stop at epoch {ep+1}')
            break

    print(f'  [{label}] Best test MSE: {best_mse:.4f} (ep {best_ep})')

    # Compute per-bin Pearson r with best model
    model.load_state_dict(best_state)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for hb, mb, lb in te_dl:
            pred = model(hb.to(device), lb).cpu().numpy()
            for i, L in enumerate(lb):
                T_p = pred.shape[1]
                L_use = min(L, T_p)
                all_pred.append(pred[i, :L_use])
                all_true.append(mb[i, :L_use].numpy())
    AT = np.concatenate(all_true)
    AP = np.concatenate(all_pred)
    overall_r = np.corrcoef(AT.flatten(), AP.flatten())[0, 1]
    bin_r = [np.corrcoef(AT[:, b], AP[:, b])[0, 1] for b in range(N_MELS)]

    return dict(
        train_losses=train_losses, test_losses=test_losses,
        best_mse=best_mse, best_epoch=best_ep,
        overall_r=overall_r, bin_r=bin_r,
        mean_bin_r=float(np.mean(bin_r)),
        all_pred=all_pred, all_true=all_true,
        model_state=best_state)


# ============================================================
# Section 8: Griffin-Lim Audio Reconstruction (no librosa)
# ============================================================
def griffin_lim(mel_db, sr=16000, n_fft=1024, hop_ms=10, n_mels=40,
               fmax=8000, n_iter=60):
    """Reconstruct waveform from mel spectrogram (dB scale).

    1. dB → power
    2. Mel → linear via filterbank pseudo-inverse
    3. Iterative phase estimation (Griffin-Lim)
    4. Return float32 waveform normalized to [-1, 1]
    """
    hop = int(hop_ms / 1000 * sr)

    # dB → power
    mel_power = np.power(10.0, mel_db / 10.0).T  # (n_mels, T)

    # Mel → linear spectrogram
    fb = build_mel_filterbank(sr, n_fft, n_mels, fmin=0, fmax=fmax)
    fb_pinv = np.linalg.pinv(fb)  # (n_fft//2+1, n_mels)
    S_linear = np.maximum(fb_pinv @ mel_power, 0)
    S_mag = np.sqrt(S_linear)

    # Griffin-Lim iteration
    n_freq, n_frames = S_mag.shape
    rng = np.random.RandomState(42)
    angles = np.exp(2j * np.pi * rng.rand(n_freq, n_frames))
    S_complex = S_mag * angles

    for _ in range(n_iter):
        _, reconstructed = ssig.istft(
            S_complex, fs=sr, nperseg=n_fft,
            noverlap=n_fft - hop, nfft=n_fft)
        _, _, Zxx = ssig.stft(
            reconstructed, fs=sr, nperseg=n_fft,
            noverlap=n_fft - hop, nfft=n_fft)
        min_t = min(S_mag.shape[1], Zxx.shape[1])
        angles = np.exp(1j * np.angle(Zxx[:, :min_t]))
        S_complex = S_mag[:, :min_t] * angles

    _, waveform = ssig.istft(
        S_complex, fs=sr, nperseg=n_fft,
        noverlap=n_fft - hop, nfft=n_fft)

    waveform = waveform / (np.abs(waveform).max() + 1e-8)
    return waveform.astype(np.float32)


# ============================================================
# Section 9: Visualization & Comparison
# ============================================================
def plot_classifier_comparison(results, le, output_dir):
    """Bar charts + confusion matrices for all classifier conditions."""
    conditions = list(results.keys())
    accs = [results[c]['best_acc'] for c in conditions]
    f1s = [results[c]['best_f1'] for c in conditions]

    fig, axes = plt.subplots(2, len(conditions) // 2 + 1, figsize=(20, 10))

    # Accuracy bars
    ax = axes[0, 0]
    colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b', '#2ecc71', '#27ae60']
    bars = ax.bar(range(len(conditions)), accs, color=colors[:len(conditions)])
    ax.axhline(1.0 / len(le.classes_), color='gray', ls='--',
               label=f'Chance ({1/len(le.classes_):.1%})')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Word Classification Accuracy', fontweight='bold')
    ax.legend()
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.1%}', ha='center', va='bottom', fontsize=8)

    # F1 bars
    ax = axes[0, 1]
    bars = ax.bar(range(len(conditions)), f1s, color=colors[:len(conditions)])
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Macro F1')
    ax.set_title('Macro F1 Score', fontweight='bold')

    # Confusion matrices
    n_cm = min(len(conditions), 4)
    for idx in range(n_cm):
        c = conditions[idx]
        r = len(conditions) // 2 + 1
        row = 1 if idx >= 2 else 0
        col = (idx % 2) + 2 if idx < 2 else idx - 2
        if row < axes.shape[0] and col < axes.shape[1]:
            ax = axes[row, col]
        else:
            continue
        cm = results[c]['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    ax=ax, linewidths=0.5, cbar=False)
        ax.set_title(f'{c}\n({results[c]["best_acc"]:.1%})',
                     fontsize=9, fontweight='bold')
        ax.set_ylabel('True')
        ax.set_xlabel('Pred')

    # Hide unused axes
    for ax in axes.flat:
        if not ax.has_data() and not ax.get_title():
            ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'classifier_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'  Saved classifier_comparison.png')


def plot_decoder_comparison(results, output_dir):
    """Loss curves, per-bin r, example spectrograms."""
    conditions = list(results.keys())
    n_cond = len(conditions)

    fig, axes = plt.subplots(3, max(n_cond, 3), figsize=(6 * max(n_cond, 3), 15))

    # Row 1: Loss curves
    for idx, c in enumerate(conditions):
        ax = axes[0, idx]
        r = results[c]
        ax.plot(r['train_losses'], label='Train', lw=1.2)
        ax.plot(r['test_losses'], label='Test', lw=1.2, color='#e74c3c')
        ax.set_title(f'{c}\nMSE={r["best_mse"]:.4f}', fontsize=9,
                     fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.legend(fontsize=7)

    # Row 2: Per-bin Pearson r
    for idx, c in enumerate(conditions):
        ax = axes[1, idx]
        ax.bar(range(N_MELS), results[c]['bin_r'], width=0.8, alpha=0.8)
        ax.axhline(0, color='gray', ls='--')
        ax.set_title(f'{c} r={results[c]["overall_r"]:.3f}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Mel Bin')
        ax.set_ylabel('Pearson r')
        ax.set_ylim(-0.2, 1.0)

    # Row 3: Example spectrograms (first test trial)
    for idx, c in enumerate(conditions):
        ax = axes[2, idx]
        if results[c]['all_pred']:
            pred = results[c]['all_pred'][0]
            ax.imshow(pred.T, aspect='auto', origin='lower', cmap='magma')
            ax.set_title(f'{c} Predicted Mel', fontsize=9, fontweight='bold')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Mel Bin')

    # Hide unused
    for ax in axes.flat:
        if not ax.has_data() and not ax.get_title():
            ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'decoder_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'  Saved decoder_comparison.png')


def plot_griffin_lim_results(dec_results, all_runs, output_dir, audio_dir,
                            fs_a=16000, test_words=None):
    """Reconstruct audio from best decoder, save .wav, plot spectrograms."""
    # Find best decoder
    best_key = min(dec_results, key=lambda k: dec_results[k]['best_mse'])
    best = dec_results[best_key]
    print(f'  Best decoder: {best_key} (MSE={best["best_mse"]:.4f})')

    n_show = min(5, len(best['all_pred']))
    if all_runs is not None:
        test_trials = all_runs[TEST_RUN]['trials']
        fs_a = all_runs[TEST_RUN]['fs_audio']
    else:
        test_trials = None

    fig, axes = plt.subplots(n_show, 3, figsize=(18, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        true_mel = best['all_true'][i]
        pred_mel = best['all_pred'][i]
        if test_trials is not None and i < len(test_trials):
            word = test_trials.iloc[i]['word']
        elif test_words is not None and i < len(test_words):
            word = test_words[i]
        else:
            word = f'trial_{i}'

        # Reconstruct audio
        pred_audio = griffin_lim(pred_mel, sr=fs_a)
        true_audio = griffin_lim(true_mel, sr=fs_a)

        # Save .wav
        pred_path = audio_dir / f'pred_{word}_{i}.wav'
        true_path = audio_dir / f'true_{word}_{i}.wav'
        wavfile.write(str(pred_path), fs_a,
                      (pred_audio * 32767).astype(np.int16))
        wavfile.write(str(true_path), fs_a,
                      (true_audio * 32767).astype(np.int16))

        # Plot
        r = np.corrcoef(true_mel.flatten(), pred_mel.flatten())[0, 1]

        axes[i, 0].imshow(true_mel.T, aspect='auto', origin='lower',
                          cmap='magma')
        axes[i, 0].set_title(f"True: '{word}'", fontweight='bold')
        axes[i, 0].set_ylabel('Mel Bin')

        axes[i, 1].imshow(pred_mel.T, aspect='auto', origin='lower',
                          cmap='magma')
        axes[i, 1].set_title(f"Predicted (r={r:.3f})", fontweight='bold')

        axes[i, 2].plot(true_audio, alpha=0.6, label='True GL', lw=0.5)
        axes[i, 2].plot(pred_audio, alpha=0.6, label='Pred GL', lw=0.5)
        axes[i, 2].set_title('Griffin-Lim Waveforms', fontweight='bold')
        axes[i, 2].legend(fontsize=7)
        axes[i, 2].set_xlabel('Sample')

    plt.tight_layout()
    plt.savefig(output_dir / 'griffin_lim_results.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'  Saved griffin_lim_results.png')
    print(f'  Saved {n_show * 2} .wav files to {audio_dir}')


# ============================================================
# Section 10: Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Speech BCI Advanced Training')
    parser.add_argument('--data-dir', type=str,
                        default='data/train/2022_09_22',
                        help='Raw data directory (used if --preprocessed not set)')
    parser.add_argument('--syll-dir', type=str,
                        default='data/syllable/2022_09_22',
                        help='Syllable baseline directory')
    parser.add_argument('--preprocessed', type=str, default=None,
                        help='Path to preprocessed data dir (from preprocess.py)')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_dir = output_dir / 'models'
    audio_dir = output_dir / 'audio'
    plot_dir = output_dir / 'plots'
    for d in [output_dir, model_dir, audio_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Device — use GPUs specified by CUDA_VISIBLE_DEVICES
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    print(f'Device: {device} ({n_gpus} GPUs available)')
    use_dp = n_gpus > 1

    # ── Load data ──
    print('\n═══ Loading Data ═══')
    all_runs = None
    if args.preprocessed:
        data = load_preprocessed(args.preprocessed)
        n_cls = data['n_cls']
    else:
        all_runs, hg_mu, hg_sd = load_all_data(args.data_dir, args.syll_dir)
        data = prepare_data(all_runs)
        n_cls = data['n_cls']

    # Build a LabelEncoder-like object for plotting
    if 'le' in data:
        le = data['le']
    else:
        # From preprocessed data
        le = LabelEncoder()
        le.classes_ = np.array(data['classes'])

    # ── Create dataloaders ──
    print('\n═══ Creating Dataloaders ═══')
    ML_cls = max(
        max(len(s) for s in data['train_X']),
        max(len(s) for s in data['test_X']))
    ML_ac = max(
        max(len(s) for s in data['tr_hg_s']),
        max(len(s) for s in data['te_hg_s']))

    # Classification
    tr_ds = TrialDS(data['train_X'], data['train_y'], ML_cls)
    tr_ds_aug = AugTrialDS(data['train_X'], data['train_y'], ML_cls,
                           augment=True)
    te_ds = TrialDS(data['test_X'], data['test_y'], ML_cls)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    tr_dl_aug = DataLoader(tr_ds_aug, batch_size=args.batch_size, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size)

    # Acoustic
    tr_ads = AcDS(data['tr_hg_s'], data['tr_mel_s'], ML_ac)
    tr_ads_aug = AugAcDS(data['tr_hg_s'], data['tr_mel_s'], ML_ac,
                         augment=True)
    te_ads = AcDS(data['te_hg_s'], data['te_mel_s'], ML_ac)

    tr_adl = DataLoader(tr_ads, batch_size=args.batch_size, shuffle=True)
    tr_adl_aug = DataLoader(tr_ads_aug, batch_size=args.batch_size,
                            shuffle=True)
    te_adl = DataLoader(te_ads, batch_size=args.batch_size)

    print(f'  Cls max_len={ML_cls}, Ac max_len={ML_ac}')
    print(f'  Train cls: {len(tr_ds)}, Test cls: {len(te_ds)}')
    print(f'  Train ac: {len(tr_ads)}, Test ac: {len(te_ads)}')

    # ── Define models ──
    # Assign each architecture to a different GPU for parallel training
    gpu_ids = list(range(n_gpus)) if n_gpus > 0 else [0]
    arch_list = ['BiLSTM', 'TCN', 'Transformer']
    arch_gpu = {arch: torch.device(f'cuda:{gpu_ids[i % len(gpu_ids)]}')
                for i, arch in enumerate(arch_list)}
    print(f'  GPU assignments: {", ".join(f"{a}→GPU{arch_gpu[a].index}" for a in arch_list)}')

    classifier_configs = {
        arch: lambda a=arch: get_classifier(a, n_cls, N_ECOG).to(arch_gpu[a])
        for arch in arch_list
    }
    decoder_configs = {
        arch: lambda a=arch: get_decoder(a, N_MELS, N_ECOG).to(arch_gpu[a])
        for arch in arch_list
    }

    # Print param counts
    print('\n═══ Model Parameters ═══')
    for name, factory in classifier_configs.items():
        m = factory()
        print(f'  {name} classifier: {count_params(m):,} params')
        del m
    for name, factory in decoder_configs.items():
        m = factory()
        print(f'  {name} decoder:    {count_params(m):,} params')
        del m

    # ── Train classifiers ──
    print('\n═══ Training Classifiers ═══')
    cls_results = {}
    for arch, factory in classifier_configs.items():
        dev = arch_gpu[arch]
        for aug, dl in [('no_aug', tr_dl), ('aug', tr_dl_aug)]:
            key = f'{arch}_{aug}'
            print(f'\n  ── {key} (GPU {dev.index}) ──')
            model = factory()
            t0 = time.time()
            cls_results[key] = train_classifier(
                model, dl, te_dl, dev,
                n_epochs=args.epochs, lr=args.lr, patience=args.patience,
                label=key)
            elapsed = time.time() - t0
            print(f'  [{key}] Time: {elapsed:.0f}s')
            # Save model
            torch.save(cls_results[key]['model_state'],
                       model_dir / f'cls_{key}.pt')

    # ── Train decoders ──
    print('\n═══ Training Decoders ═══')
    dec_results = {}
    for arch, factory in decoder_configs.items():
        dev = arch_gpu[arch]
        for aug, dl in [('no_aug', tr_adl), ('aug', tr_adl_aug)]:
            key = f'{arch}_{aug}'
            print(f'\n  ── {key} (GPU {dev.index}) ──')
            model = factory()
            t0 = time.time()
            dec_results[key] = train_decoder(
                model, dl, te_adl, dev,
                n_epochs=args.epochs, lr=args.lr, patience=args.patience,
                label=key)
            elapsed = time.time() - t0
            print(f'  [{key}] Time: {elapsed:.0f}s')
            torch.save(dec_results[key]['model_state'],
                       model_dir / f'dec_{key}.pt')

    # ── Save metrics ──
    print('\n═══ Saving Metrics ═══')
    metrics = {
        'classifiers': {
            k: {
                'best_acc': v['best_acc'],
                'best_f1': v['best_f1'],
                'best_epoch': v['best_epoch'],
                'final_loss': v['losses'][-1] if v['losses'] else None,
            }
            for k, v in cls_results.items()
        },
        'decoders': {
            k: {
                'best_mse': v['best_mse'],
                'overall_r': v['overall_r'],
                'mean_bin_r': v['mean_bin_r'],
                'best_epoch': v['best_epoch'],
            }
            for k, v in dec_results.items()
        }
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'  Saved metrics.json')

    # ── Plots ──
    print('\n═══ Generating Plots ═══')
    plot_classifier_comparison(cls_results, le, plot_dir)
    plot_decoder_comparison(dec_results, plot_dir)
    plot_griffin_lim_results(dec_results, all_runs, plot_dir, audio_dir)

    # ── Summary ──
    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)

    print('\nClassifiers:')
    print(f'  {"Condition":<25} {"Acc":>8} {"F1":>8} {"Epoch":>6}')
    print(f'  {"-"*25} {"-"*8} {"-"*8} {"-"*6}')
    for k, v in cls_results.items():
        print(f'  {k:<25} {v["best_acc"]:>7.1%} {v["best_f1"]:>8.3f} '
              f'{v["best_epoch"]:>6d}')

    print('\nDecoders:')
    print(f'  {"Condition":<25} {"MSE":>8} {"r":>8} {"bin_r":>8} {"Epoch":>6}')
    print(f'  {"-"*25} {"-"*8} {"-"*8} {"-"*8} {"-"*6}')
    for k, v in dec_results.items():
        print(f'  {k:<25} {v["best_mse"]:>8.4f} {v["overall_r"]:>8.3f} '
              f'{v["mean_bin_r"]:>8.3f} {v["best_epoch"]:>6d}')

    # Save predictions for further analysis
    pred_data = {}
    for k, v in dec_results.items():
        pred_data[f'{k}_pred'] = np.concatenate(v['all_pred'])
        pred_data[f'{k}_true'] = np.concatenate(v['all_true'])
    np.savez_compressed(output_dir / 'predictions.npz', **pred_data)
    print(f'\n  Saved predictions.npz')
    print(f'\nAll outputs in: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
