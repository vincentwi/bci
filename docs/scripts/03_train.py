#!/usr/bin/env python3
"""Train nVAD and/or Acoustic Decoder on GPU.

Methodology matches the paper exactly:
  - github.com/cronelab/delayed-speech-synthesis
  - nVAD: truncated BPTT k1=k2=50, RMSprop lr=1e-4, 8 epochs, best by val acc
  - Decoder: full trial forward pass, RMSprop lr=1e-4, 20 epochs, best by val loss
  - Channel selection: paper's exact 64-channel mapping
  - No early stopping, no gradient clipping

Usage:
    python scripts/03_train.py                     # train both models
    python scripts/03_train.py --model nvad         # train nVAD only
    python scripts/03_train.py --model decoder      # train decoder only
    python scripts/03_train.py --tag v2_paper_match # tag this run
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from speech_bci import config
from speech_bci.models import build_nvad, build_acoustic_decoder, count_params
from speech_bci.dataset import (
    SequentialFrameDataset, SpeechBCIDataset, make_dataloader,
)
from speech_bci.training import (
    train_nvad, train_acoustic_decoder,
    evaluate_nvad, evaluate_decoder, compute_spectral_correlation,
)
from speech_bci.channel_selection import get_default_speech_channels

# Results directory for all training runs
RESULTS_DIR = PROJECT_DIR / "results"


def detect_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_multi_gpu_devices():
    n = torch.cuda.device_count()
    if n >= 2:
        return "cuda:0", "cuda:1"
    elif n == 1:
        return "cuda:0", "cuda:0"
    return "cpu", "cpu"


def _serialize(v):
    """Recursively serialize for JSON."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, list):
        return [_serialize(x) for x in v]
    if isinstance(v, dict):
        return {kk: _serialize(vv) for kk, vv in v.items()}
    if isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return str(v)


def save_run_results(run_id: str, tag: str, model_name: str, history: dict,
                     run_config: dict, paper_reference: dict):
    """Save structured training results for future visualization."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "tag": tag,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "config": _serialize(run_config),
        "history": _serialize(history),
        "paper_reference": paper_reference,
    }

    fname = RESULTS_DIR / f"{run_id}_{model_name}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved: {fname}")
    return fname


def train_nvad_model(args, device, h5_path, channel_mask, run_id, tag):
    """Train the nVAD model."""
    print("\n" + "=" * 60)
    print("  TRAINING nVAD (paper methodology)")
    print("  - Truncated BPTT k1=k2=50")
    print("  - No gradient clipping, no early stopping")
    print("  - Best model by validation accuracy")
    print("=" * 60)

    n_ch = len(channel_mask)

    # Datasets
    train_ds = SequentialFrameDataset(h5_path, config.TRAIN_DAYS, channel_mask)
    val_ds = SequentialFrameDataset(h5_path, [config.VAL_DAY], channel_mask)
    print(f"Train: {len(train_ds)} runs, Val: {len(val_ds)} runs")
    print(f"Val day: {config.VAL_DAY}, Test day: {config.TEST_DAY}")

    train_dl = make_dataloader(train_ds, shuffle=True)
    val_dl = make_dataloader(val_ds, shuffle=False)

    # Model
    model = build_nvad(n_electrodes=n_ch)
    ckpt_dir = config.CHECKPOINTS_DIR / "nvad"

    if args.resume:
        best_path = ckpt_dir / "best_model.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, weights_only=True))
            print(f"Resumed from best_model.pt")

    epochs = args.epochs or config.NVAD_EPOCHS
    lr = args.lr or config.NVAD_LR

    run_config = {
        "n_electrodes": n_ch,
        "channel_indices": channel_mask.tolist(),
        "hidden_size": config.NVAD_HIDDEN,
        "num_layers": config.NVAD_LAYERS,
        "dropout": config.NVAD_DROPOUT,
        "epochs": epochs,
        "lr": lr,
        "optimizer": "RMSprop",
        "loss": "CrossEntropyLoss",
        "tbptt_seq_len": 50,
        "train_days": config.TRAIN_DAYS,
        "val_day": config.VAL_DAY,
        "test_day": config.TEST_DAY,
        "n_train_runs": len(train_ds),
        "n_val_runs": len(val_ds),
        "n_params": count_params(model),
        "device": device,
    }

    print(f"Config: epochs={epochs}, lr={lr}, device={device}")
    print(f"Params: {count_params(model):,}")

    t0 = time.time()
    history = train_nvad(
        model, train_dl, val_dl,
        device=device, epochs=epochs, lr=lr,
        checkpoint_dir=ckpt_dir,
    )
    elapsed = time.time() - t0
    history["training_time_s"] = elapsed
    print(f"\nTraining time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best epoch: {history['best_epoch']} (val_acc={history['best_val_accuracy']:.3f})")

    # Test evaluation (using best model)
    test_ds = SequentialFrameDataset(h5_path, [config.TEST_DAY], channel_mask)
    test_dl = make_dataloader(test_ds, shuffle=False)
    if len(test_ds) > 0:
        test_metrics = evaluate_nvad(model, test_dl, device)
        print(f"\nTest results (day {config.TEST_DAY}):")
        print(f"  Accuracy:  {test_metrics['accuracy']:.1%}")
        print(f"  F1:        {test_metrics['f1']:.3f}")
        print(f"  Precision: {test_metrics['precision']:.3f}")
        print(f"  Recall:    {test_metrics['recall']:.3f}")
        history["test"] = test_metrics

    # Paper reference
    paper_ref = {
        "paper": "Angrick et al. (2024) Scientific Reports 14:9617",
        "github": "https://github.com/cronelab/delayed-speech-synthesis",
        "nVAD_val_accuracy": 0.934,
        "note": "Paper reports 93.4% frame-wise accuracy on validation day",
    }

    # Save plots
    plot_dir = PROJECT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], lw=1.5)
    axes[0].set_title("nVAD Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(history["val_accuracy"], label="Ours", lw=1.5)
    axes[1].plot(history["val_f1"], label="F1", lw=1.5, ls="--")
    axes[1].axhline(0.934, color="gray", ls=":", label="Paper (93.4%)")
    axes[1].set_title("nVAD Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "nvad_training.png", dpi=150)
    print(f"Plot saved: {plot_dir / 'nvad_training.png'}")

    # Save structured results
    save_run_results(run_id, tag, "nvad", history, run_config, paper_ref)

    return history


def train_decoder_model(args, device, h5_path, channel_mask, run_id, tag):
    """Train the Acoustic Decoder model."""
    print("\n" + "=" * 60)
    print("  TRAINING ACOUSTIC DECODER (paper methodology)")
    print("  - Full trial forward pass (NO truncated BPTT)")
    print("  - No gradient clipping, no early stopping")
    print("  - Best model by validation MSE")
    print("=" * 60)

    n_ch = len(channel_mask)

    # Datasets
    train_ds = SpeechBCIDataset(h5_path, config.TRAIN_DAYS, channel_mask, target_type="lpc")
    val_ds = SpeechBCIDataset(h5_path, [config.VAL_DAY], channel_mask, target_type="lpc")
    print(f"Train: {len(train_ds)} trials, Val: {len(val_ds)} trials")
    print(f"Val day: {config.VAL_DAY}, Test day: {config.TEST_DAY}")

    train_dl = make_dataloader(train_ds, batch_size=1, shuffle=True)
    val_dl = make_dataloader(val_ds, batch_size=1, shuffle=False)

    # Model
    model = build_acoustic_decoder(n_electrodes=n_ch)
    ckpt_dir = config.CHECKPOINTS_DIR / "decoder"

    if args.resume:
        best_path = ckpt_dir / "best_model.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, weights_only=True))
            print(f"Resumed from best_model.pt")

    epochs = args.epochs or config.DECODER_EPOCHS
    lr = args.lr or config.DECODER_LR

    run_config = {
        "n_electrodes": n_ch,
        "channel_indices": channel_mask.tolist(),
        "hidden_size": config.DECODER_HIDDEN,
        "num_layers": config.DECODER_LAYERS,
        "dropout": config.DECODER_DROPOUT,
        "epochs": epochs,
        "lr": lr,
        "optimizer": "RMSprop",
        "loss": "MSELoss",
        "training_mode": "full_trial_forward_pass",
        "train_days": config.TRAIN_DAYS,
        "val_day": config.VAL_DAY,
        "test_day": config.TEST_DAY,
        "n_train_trials": len(train_ds),
        "n_val_trials": len(val_ds),
        "n_params": count_params(model),
        "device": device,
        "lpc_method": "python_fallback",
        "lpc_note": "Using Python LPC (NOT LPCNet Bark-scale cepstrals)",
    }

    print(f"Config: epochs={epochs}, lr={lr}, device={device}")
    print(f"Params: {count_params(model):,}")

    t0 = time.time()
    history = train_acoustic_decoder(
        model, train_dl, val_dl,
        device=device, epochs=epochs, lr=lr,
        checkpoint_dir=ckpt_dir,
    )
    elapsed = time.time() - t0
    history["training_time_s"] = elapsed
    print(f"\nTraining time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Best epoch: {history['best_epoch']} (val_mse={history['best_val_mse']:.4f})")

    # Test evaluation (using best model)
    test_ds = SpeechBCIDataset(h5_path, [config.TEST_DAY], channel_mask, target_type="lpc")
    test_dl = make_dataloader(test_ds, batch_size=1, shuffle=False)
    if len(test_ds) > 0:
        test_metrics = evaluate_decoder(model, test_dl, device)
        print(f"\nTest results (day {config.TEST_DAY}):")
        print(f"  MSE:       {test_metrics['mse']:.4f}")
        print(f"  Pearson r: {test_metrics['pearson_r_mean']:.4f}")
        print(f"  Per-dim r:")
        for i, r in enumerate(test_metrics["pearson_r_per_dim"]):
            label = f"bark_{i}" if i < 18 else ("pitch_period" if i == 18 else "pitch_corr")
            print(f"    {label}: {r:.3f}")
        history["test"] = {
            "mse": float(test_metrics["mse"]),
            "pearson_r_mean": float(test_metrics["pearson_r_mean"]),
            "pearson_r_per_dim": test_metrics["pearson_r_per_dim"].tolist(),
        }

        # Spectral correlation (paper's metric)
        spec_corr = compute_spectral_correlation(model, test_dl, device)
        print(f"\n  Spectral correlation (LPC-based proxy):")
        print(f"    Mean: {spec_corr['spectral_corr_mean']:.3f} ± {spec_corr['spectral_corr_std']:.3f}")
        print(f"    N trials: {spec_corr['n_trials']}")
        history["test"]["spectral_correlation"] = spec_corr

    # Paper reference
    paper_ref = {
        "paper": "Angrick et al. (2024) Scientific Reports 14:9617",
        "github": "https://github.com/cronelab/delayed-speech-synthesis",
        "spectral_correlation_mean": 0.67,
        "spectral_correlation_std": 0.18,
        "intelligibility_accuracy": 0.80,
        "note": "Paper reports spectral corr on mel spectrograms after LPCNet vocoder synthesis. "
                "Our proxy uses per-trial LPC Pearson r (no vocoder available). "
                "Paper's LPC uses LPCNet Bark-scale cepstrals; ours uses Python LPC fallback.",
    }

    # Save plots
    plot_dir = PROJECT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train", lw=1.5)
    axes[0].plot(history["val_mse"], label="Val", lw=1.5, color="#e74c3c")
    axes[0].set_title("Acoustic Decoder Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[1].plot(history["val_pearson_r"], lw=1.5, color="#2ecc71")
    axes[1].set_title("Val Pearson r (mean LPC dims)")
    axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(plot_dir / "decoder_training.png", dpi=150)
    print(f"Plot saved: {plot_dir / 'decoder_training.png'}")

    # Save structured results
    save_run_results(run_id, tag, "decoder", history, run_config, paper_ref)

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Speech BCI models")
    parser.add_argument("--model", type=str, choices=["nvad", "decoder", "both"],
                        default="both", help="Which model to train")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda/mps/cpu)")
    parser.add_argument("--h5", type=Path, default=None,
                        help="Path to features.h5")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use separate GPUs for nVAD (cuda:0) and decoder (cuda:1)")
    parser.add_argument("--tag", type=str, default="default",
                        help="Tag for this training run (for results tracking)")
    args = parser.parse_args()

    h5_path = args.h5 or config.CACHE_DIR / "features.h5"
    if not h5_path.exists():
        print(f"ERROR: Features cache not found: {h5_path}")
        print("Run 02_preprocess.py first.")
        sys.exit(1)

    channel_mask = get_default_speech_channels()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.multi_gpu or (args.device is None and torch.cuda.device_count() >= 2):
        nvad_device, decoder_device = get_multi_gpu_devices()
        print(f"Multi-GPU mode: nVAD → {nvad_device} ({torch.cuda.get_device_name(0)}), "
              f"Decoder → {decoder_device} ({torch.cuda.get_device_name(1)})")
    else:
        device = detect_device(args.device)
        nvad_device = device
        decoder_device = device
        print(f"Device: {device}")
        if "cuda" in str(device):
            gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
            print(f"GPU: {torch.cuda.get_device_name(gpu_idx)}")

    print(f"Channels: {len(channel_mask)} (paper's exact 64-channel selection)")
    print(f"Channel indices: {channel_mask[:5].tolist()}...{channel_mask[-5:].tolist()}")
    print(f"Run ID: {run_id}, Tag: {args.tag}")

    if args.model in ("nvad", "both"):
        train_nvad_model(args, nvad_device, h5_path, channel_mask, run_id, args.tag)

    if args.model in ("decoder", "both"):
        train_decoder_model(args, decoder_device, h5_path, channel_mask, run_id, args.tag)

    print("\n=== All training complete ===")


if __name__ == "__main__":
    main()
