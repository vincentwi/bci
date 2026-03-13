#!/usr/bin/env python3
"""
Multi-Session Speech BCI Training — Matching Paper Protocol
============================================================
Trains on ALL training sessions with per-day syllable normalization,
tests on held-out test session (2022_11_03), validates on 2022_11_04.

Paper reference: Angrick et al. 2023, "Online speech synthesis using a
chronically implanted brain-computer interface in an individual with ALS"

Key paper details:
  - 1,570 training trials (~80 min), 70 validation, 70 test
  - 128 ECoG channels, 64 selected for high-gamma responsiveness
  - BiLSTM decoder: 2 layers, 150 units, dropout 50%
  - High-gamma: 70-170 Hz (notch at 118-122 Hz), 50ms window, 10ms hop
  - Per-day normalization using syllable repetition baseline
  - 6 words: Back, Down, Enter, Left, Right, Up
  - Result: 80% intelligibility, spectrogram r=0.67

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multisession.py \\
        --data-dir /mnt/home/vincent.wilmet/data \\
        --output-dir /mnt/home/vincent.wilmet/results_multisession \\
        --epochs 120
"""

import argparse
import json
import copy
import os
import time
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io.wavfile as wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from models import (get_classifier, get_decoder, count_params)
from preprocess import seg_trial
from preprocess_gpu import (extract_hg_gpu, mel_frames_gpu, load_session_gpu)

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
WORDS = ['Back', 'Down', 'Enter', 'Left', 'Right', 'Up']
WORD_TO_IDX = {w: i for i, w in enumerate(WORDS)}
N_CLS = len(WORDS)

# Session layout matching the paper
TRAIN_SESSIONS = [
    '2022_09_22', '2022_09_23', '2022_09_30',
    '2022_10_05', '2022_10_06', '2022_10_10', '2022_10_27',
]
TEST_SESSION = '2022_11_03'
VAL_SESSION = '2022_11_04'


# ============================================================
# Data Loading — Multi-session with per-day normalization
# ============================================================
# Data loading is handled by preprocess_gpu.load_session_gpu


def extract_segments(session_data_list, pre=0.5, post=0.5):
    """Extract classification and acoustic segments from loaded sessions."""
    cls_segs, cls_labels, cls_words = [], [], []
    ac_hg, ac_mel = [], []

    for run_data in session_data_list:
        hg = run_data['hg']
        hg_t = run_data['hg_t']
        mel = run_data['mel']
        trials = run_data['trials']

        for _, row in trials.iterrows():
            word = row['word']
            if word not in WORD_TO_IDX:
                continue
            label = WORD_TO_IDX[word]

            # Classification segment
            seg = seg_trial(hg, hg_t, row, pre=pre, post=post)
            if len(seg) < 10:
                continue
            cls_segs.append(seg)
            cls_labels.append(label)
            cls_words.append(word)

            # Acoustic segment
            sf = int(row['start'] * 100)
            ef = int(row['end'] * 100)
            n_com = min(len(hg), len(mel))
            if ef > n_com:
                continue
            ac_hg.append(hg[:n_com][sf:ef])
            ac_mel.append(mel[:n_com][sf:ef])

    return cls_segs, cls_labels, cls_words, ac_hg, ac_mel


# ============================================================
# Augmentation
# ============================================================
def augment_time_warp(x, rate_range=(0.9, 1.1)):
    import scipy.interpolate
    T, C = x.shape
    rate = np.random.uniform(*rate_range)
    new_T = max(2, int(T * rate))
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, new_T)
    f = scipy.interpolate.interp1d(old_t, x, axis=0, kind='linear',
                                   fill_value='extrapolate')
    return f(new_t)


def augment_noise(x, scale=0.1):
    std = x.std(axis=0, keepdims=True) + 1e-10
    return x + np.random.randn(*x.shape) * std * scale


def augment_channel_dropout(x, p=0.1):
    mask = (np.random.rand(x.shape[1]) > p).astype(np.float64)
    return x * mask[np.newaxis, :]


# ============================================================
# Dataset Classes
# ============================================================
class TrialDS(Dataset):
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
    def __init__(self, segs, labels, max_len=None, augment=False):
        self.raw_segs = [s.copy() for s in segs]
        self.labels = labels
        self.max_len = max_len or max(len(s) for s in segs)
        self.augment = augment
        self.n_ch = segs[0].shape[1]

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
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.9, 1.1)
                hg = augment_time_warp(hg, (factor, factor))
                mel = augment_time_warp(mel, (factor, factor))
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
# Training Functions
# ============================================================
def train_classifier(model, tr_dl, te_dl, device, n_epochs=120,
                     lr=1e-3, wd=1e-4, patience=20, label='',
                     val_dl=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    crit = nn.CrossEntropyLoss()

    losses, accs, val_accs = [], [], []
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

        # Validation accuracy
        if val_dl is not None:
            vpreds, vlabels = [], []
            with torch.no_grad():
                for xb, lb, yb in val_dl:
                    out = model(xb.to(device), lb)
                    vpreds.extend(out.argmax(1).cpu().numpy())
                    vlabels.extend(yb if isinstance(yb, list) else yb.numpy())
            val_acc = accuracy_score(vlabels, vpreds)
            val_accs.append(val_acc)

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
            msg = f'  [{label}] Ep {ep+1:3d}  loss={el:.4f}  test_acc={acc:.1%}'
            if val_accs:
                msg += f'  val_acc={val_accs[-1]:.1%}'
            print(msg)

        if no_improve >= patience:
            print(f'  [{label}] Early stop at epoch {ep+1}')
            break

    print(f'  [{label}] Best test: {best_acc:.1%} (ep {best_ep})')

    model.load_state_dict(best_state)
    f1 = f1_score(best_labels, best_preds, average='macro')
    cm = confusion_matrix(best_labels, best_preds)

    return dict(
        losses=losses, accs=accs, val_accs=val_accs,
        best_acc=best_acc, best_f1=f1, best_epoch=best_ep,
        preds=best_preds, labels=best_labels, cm=cm,
        model_state=best_state)


def train_decoder(model, tr_dl, te_dl, device, n_epochs=120,
                  lr=1e-3, wd=1e-4, patience=20, label=''):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
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
            mb_t = mb[:, :T_pred]
            mask = torch.zeros_like(mb_t)
            for i, L in enumerate(lb):
                mask[i, :min(L, T_pred)] = 1.0
            loss = mse_fn(pred * mask, mb_t * mask) * mask.numel() / mask.sum()
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
                mb_t = mb[:, :T_pred]
                mask = torch.zeros_like(mb_t)
                for i, L in enumerate(lb):
                    mask[i, :min(L, T_pred)] = 1.0
                loss = mse_fn(pred * mask, mb_t * mask) * mask.numel() / mask.sum()
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
# Visualization
# ============================================================
def plot_results(cls_results, dec_results, output_dir):
    """Generate comparison plots."""
    # Classifier comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    conditions = list(cls_results.keys())
    accs = [cls_results[c]['best_acc'] for c in conditions]
    f1s = [cls_results[c]['best_f1'] for c in conditions]

    colors = plt.cm.Set2(np.linspace(0, 1, len(conditions)))
    ax = axes[0]
    bars = ax.bar(range(len(conditions)), accs, color=colors)
    ax.axhline(1/N_CLS, color='gray', ls='--', label=f'Chance ({1/N_CLS:.1%})')
    ax.axhline(0.80, color='red', ls='--', alpha=0.7, label='Paper (80%)')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Word Classification Accuracy', fontweight='bold')
    ax.legend(fontsize=7)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.1%}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    bars = ax.bar(range(len(conditions)), f1s, color=colors)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Macro F1')
    ax.set_title('Macro F1 Score', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'classifier_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Confusion matrices for best models
    fig, axes = plt.subplots(1, min(3, len(conditions)), figsize=(6*min(3, len(conditions)), 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    # Sort by accuracy, show top 3
    top = sorted(cls_results.keys(), key=lambda k: cls_results[k]['best_acc'], reverse=True)[:3]
    for idx, c in enumerate(top):
        cm = cls_results[c]['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=WORDS, yticklabels=WORDS,
                    ax=axes[idx], linewidths=0.5, cbar=False)
        axes[idx].set_title(f'{c}\n({cls_results[c]["best_acc"]:.1%})',
                            fontsize=9, fontweight='bold')
        axes[idx].set_ylabel('True')
        axes[idx].set_xlabel('Pred')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Decoder comparison
    dec_conditions = list(dec_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mses = [dec_results[c]['best_mse'] for c in dec_conditions]
    rs = [dec_results[c]['overall_r'] for c in dec_conditions]

    ax = axes[0]
    bars = ax.bar(range(len(dec_conditions)), rs, color=colors[:len(dec_conditions)])
    ax.axhline(0.67, color='red', ls='--', alpha=0.7, label='Paper (r=0.67)')
    ax.set_xticks(range(len(dec_conditions)))
    ax.set_xticklabels(dec_conditions, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Pearson r')
    ax.set_title('Acoustic Decoding — Spectrogram Correlation', fontweight='bold')
    ax.legend(fontsize=7)
    for bar, v in zip(bars, rs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    bars = ax.bar(range(len(dec_conditions)), mses, color=colors[:len(dec_conditions)])
    ax.set_xticks(range(len(dec_conditions)))
    ax.set_xticklabels(dec_conditions, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('MSE')
    ax.set_title('Acoustic Decoding — Test MSE', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'decoder_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print('  Saved classifier_comparison.png, confusion_matrices.png, decoder_comparison.png')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Multi-Session Speech BCI Training')
    parser.add_argument('--data-dir', type=str, default='/mnt/home/vincent.wilmet/data')
    parser.add_argument('--output-dir', type=str, default='/mnt/home/vincent.wilmet/results_multisession')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_dir = output_dir / 'models'
    plot_dir = output_dir / 'plots'
    for d in [output_dir, model_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    print(f'Device: {device} ({n_gpus} GPUs)')

    arch_list = ['BiLSTM', 'TCN', 'Transformer']
    gpu_ids = list(range(n_gpus)) if n_gpus > 0 else [0]
    arch_gpu = {arch: torch.device(f'cuda:{gpu_ids[i % len(gpu_ids)]}')
                for i, arch in enumerate(arch_list)}
    print(f'GPU map: {", ".join(f"{a}→cuda:{arch_gpu[a].index}" for a in arch_list)}')

    # Pick a GPU for preprocessing (use first available)
    preproc_dev = torch.device('cuda:0')
    print(f'Preprocessing GPU: {preproc_dev}')

    # ── Load ALL training sessions (GPU-accelerated) ──
    print('\n' + '='*60)
    print('LOADING ALL SESSIONS (GPU preprocessing, per-day normalization)')
    print('='*60)

    all_train_data = []
    for session in TRAIN_SESSIONS:
        session_dir = Path(args.data_dir) / 'train' / session
        if not session_dir.exists():
            print(f'  SKIP {session} (not found)')
            continue
        print(f'  Loading {session}...', end=' ', flush=True)
        t0 = time.time()
        session_data = load_session_gpu(args.data_dir, session, preproc_dev, split='train')
        n_trials = sum(len(d['trials']) for d in session_data)
        n_runs = len(session_data)
        print(f' {n_runs} runs, {n_trials} trials ({time.time()-t0:.1f}s)')
        all_train_data.extend(session_data)

    # Load test session
    print(f'\n  Loading TEST {TEST_SESSION}...', end=' ', flush=True)
    test_data = load_session_gpu(args.data_dir, TEST_SESSION, preproc_dev, split='test')
    n_test_trials = sum(len(d['trials']) for d in test_data)
    print(f' {len(test_data)} runs, {n_test_trials} trials')

    # Load validation session
    print(f'  Loading VAL {VAL_SESSION}...', end=' ', flush=True)
    val_data = load_session_gpu(args.data_dir, VAL_SESSION, preproc_dev, split='validation')
    n_val_trials = sum(len(d['trials']) for d in val_data)
    print(f' {len(val_data)} runs, {n_val_trials} trials')

    # ── Extract segments ──
    print('\n  Extracting segments...')
    tr_cls, tr_labels, tr_words, tr_hg, tr_mel = extract_segments(all_train_data)
    te_cls, te_labels, te_words, te_hg, te_mel = extract_segments(test_data)
    va_cls, va_labels, va_words, va_hg, va_mel = extract_segments(val_data)

    print(f'\n  DATASET SUMMARY (matching paper protocol):')
    print(f'    Train: {len(tr_cls)} classification, {len(tr_hg)} acoustic trials')
    print(f'    Test:  {len(te_cls)} classification, {len(te_hg)} acoustic trials')
    print(f'    Val:   {len(va_cls)} classification, {len(va_hg)} acoustic trials')
    print(f'    Paper: ~1570 train, 70 test, 70 validation')

    # Word distribution
    from collections import Counter
    print(f'\n    Train word distribution: {dict(Counter(tr_words))}')
    print(f'    Test word distribution:  {dict(Counter(te_words))}')

    # ── Create dataloaders ──
    print('\n  Creating dataloaders...')
    ML_cls = max(
        max(len(s) for s in tr_cls),
        max(len(s) for s in te_cls),
        max(len(s) for s in va_cls) if va_cls else 0)

    ML_ac = max(
        max(len(s) for s in tr_hg),
        max(len(s) for s in te_hg))

    tr_ds = TrialDS(tr_cls, tr_labels, ML_cls)
    tr_ds_aug = AugTrialDS(tr_cls, tr_labels, ML_cls, augment=True)
    te_ds = TrialDS(te_cls, te_labels, ML_cls)
    va_ds = TrialDS(va_cls, va_labels, ML_cls) if va_cls else None

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    tr_dl_aug = DataLoader(tr_ds_aug, batch_size=args.batch_size, shuffle=True, num_workers=2)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, num_workers=2) if va_ds else None

    tr_ads = AcDS(tr_hg, tr_mel, ML_ac)
    tr_ads_aug = AugAcDS(tr_hg, tr_mel, ML_ac, augment=True)
    te_ads = AcDS(te_hg, te_mel, ML_ac)

    tr_adl = DataLoader(tr_ads, batch_size=args.batch_size, shuffle=True, num_workers=2)
    tr_adl_aug = DataLoader(tr_ads_aug, batch_size=args.batch_size, shuffle=True, num_workers=2)
    te_adl = DataLoader(te_ads, batch_size=args.batch_size, num_workers=2)

    print(f'    Cls max_len={ML_cls}, Ac max_len={ML_ac}')

    # ── Model params ──
    print('\n' + '='*60)
    print('MODEL PARAMETERS')
    print('='*60)
    for arch in arch_list:
        m = get_classifier(arch, N_CLS, N_ECOG)
        print(f'  {arch} classifier: {count_params(m):,} params')
        del m
        m = get_decoder(arch, N_MELS, N_ECOG)
        print(f'  {arch} decoder:    {count_params(m):,} params')
        del m

    # ── Train classifiers ──
    print('\n' + '='*60)
    print('TRAINING CLASSIFIERS')
    print('='*60)
    cls_results = {}
    for arch in arch_list:
        dev = arch_gpu[arch]
        for aug_name, dl in [('no_aug', tr_dl), ('aug', tr_dl_aug)]:
            key = f'{arch}_{aug_name}'
            print(f'\n  ── {key} (GPU {dev.index}) ──')
            model = get_classifier(arch, N_CLS, N_ECOG).to(dev)
            t0 = time.time()
            cls_results[key] = train_classifier(
                model, dl, te_dl, dev,
                n_epochs=args.epochs, lr=args.lr, patience=args.patience,
                label=key, val_dl=va_dl)
            elapsed = time.time() - t0
            print(f'  [{key}] Time: {elapsed:.0f}s')
            torch.save(cls_results[key]['model_state'], model_dir / f'cls_{key}.pt')

    # ── Train decoders ──
    print('\n' + '='*60)
    print('TRAINING DECODERS')
    print('='*60)
    dec_results = {}
    for arch in arch_list:
        dev = arch_gpu[arch]
        for aug_name, dl in [('no_aug', tr_adl), ('aug', tr_adl_aug)]:
            key = f'{arch}_{aug_name}'
            print(f'\n  ── {key} (GPU {dev.index}) ──')
            model = get_decoder(arch, N_MELS, N_ECOG).to(dev)
            t0 = time.time()
            dec_results[key] = train_decoder(
                model, dl, te_adl, dev,
                n_epochs=args.epochs, lr=args.lr, patience=args.patience,
                label=key)
            elapsed = time.time() - t0
            print(f'  [{key}] Time: {elapsed:.0f}s')
            torch.save(dec_results[key]['model_state'], model_dir / f'dec_{key}.pt')

    # ── Save metrics ──
    print('\n' + '='*60)
    print('SAVING RESULTS')
    print('='*60)
    metrics = {
        'dataset': {
            'train_sessions': TRAIN_SESSIONS,
            'test_session': TEST_SESSION,
            'val_session': VAL_SESSION,
            'n_train_trials': len(tr_cls),
            'n_test_trials': len(te_cls),
            'n_val_trials': len(va_cls),
        },
        'classifiers': {
            k: {
                'best_acc': v['best_acc'],
                'best_f1': v['best_f1'],
                'best_epoch': v['best_epoch'],
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
        },
        'paper_reference': {
            'intelligibility': 0.80,
            'spectrogram_r': 0.67,
            'model': 'BiLSTM (2 layers, 150 units)',
            'note': 'Paper reports human listener accuracy on synthesized speech, '
                    'not direct classification accuracy',
        }
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ──
    plot_results(cls_results, dec_results, plot_dir)

    # ── Summary ──
    print('\n' + '='*60)
    print('RESULTS SUMMARY')
    print('='*60)

    print(f'\nDataset: {len(tr_cls)} train, {len(te_cls)} test, {len(va_cls)} val trials')
    print(f'Paper:   ~1570 train, 70 test, 70 validation trials')

    print('\nClassifiers:')
    print(f'  {"Condition":<25} {"Test Acc":>8} {"F1":>8} {"Epoch":>6}')
    print(f'  {"-"*25} {"-"*8} {"-"*8} {"-"*6}')
    for k, v in cls_results.items():
        print(f'  {k:<25} {v["best_acc"]:>7.1%} {v["best_f1"]:>8.3f} '
              f'{v["best_epoch"]:>6d}')
    print(f'  {"Paper (human intel.)":<25} {"80.0%":>8}')

    print('\nDecoders:')
    print(f'  {"Condition":<25} {"MSE":>8} {"r":>8} {"bin_r":>8} {"Epoch":>6}')
    print(f'  {"-"*25} {"-"*8} {"-"*8} {"-"*8} {"-"*6}')
    for k, v in dec_results.items():
        print(f'  {k:<25} {v["best_mse"]:>8.4f} {v["overall_r"]:>8.3f} '
              f'{v["mean_bin_r"]:>8.3f} {v["best_epoch"]:>6d}')
    print(f'  {"Paper":<25} {"--":>8} {"0.670":>8}')

    print(f'\nAll outputs in: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
