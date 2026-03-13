#!/usr/bin/env python3
"""End-to-end speech synthesis: ECoG → HG → nVAD → Decoder → LPC → Audio.

Usage:
    python scripts/04_synthesize.py                  # synthesize test day
    python scripts/04_synthesize.py --day 2022_11_03 # specific day
    python scripts/04_synthesize.py --run R01        # specific run
    python scripts/04_synthesize.py --output ./out   # output directory

Requires trained nVAD and decoder checkpoints in checkpoints/ directory.
LPCNet synthesis requires the compiled vocoder (otherwise saves LPC features only).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from speech_bci import config
from speech_bci.io_utils import load_run, discover_runs
from speech_bci.signal import extract_hg
from speech_bci.normalization import compute_day_stats, normalize_hg
from speech_bci.channel_selection import get_default_speech_channels
from speech_bci.models import build_nvad, build_acoustic_decoder


def load_models(device, n_ch):
    """Load trained nVAD and decoder from checkpoints."""
    nvad = build_nvad(n_electrodes=n_ch)
    decoder = build_acoustic_decoder(n_electrodes=n_ch)

    nvad_ckpts = sorted(config.CHECKPOINTS_DIR.glob("nvad/nvad_epoch*.pt"))
    if not nvad_ckpts:
        raise FileNotFoundError("No nVAD checkpoints found. Run 03_train.py first.")
    nvad.load_state_dict(torch.load(nvad_ckpts[-1], weights_only=True, map_location=device))
    print(f"Loaded nVAD: {nvad_ckpts[-1].name}")

    dec_ckpts = sorted(config.CHECKPOINTS_DIR.glob("decoder/decoder_epoch*.pt"))
    if not dec_ckpts:
        raise FileNotFoundError("No decoder checkpoints found. Run 03_train.py first.")
    decoder.load_state_dict(torch.load(dec_ckpts[-1], weights_only=True, map_location=device))
    print(f"Loaded decoder: {dec_ckpts[-1].name}")

    nvad.to(device).eval()
    decoder.to(device).eval()
    return nvad, decoder


def synthesize_trial(hg_segment, nvad, decoder, device):
    """Run full pipeline on a single trial segment.

    Returns (vad_pred, lpc_pred) arrays.
    """
    x = torch.from_numpy(hg_segment.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        vad_logits, _ = nvad(x)
        vad_pred = vad_logits.argmax(dim=-1).cpu().numpy()[0]
        lpc_pred, _ = decoder(x)
        lpc_pred = lpc_pred.cpu().numpy()[0]
    return vad_pred, lpc_pred


def main():
    parser = argparse.ArgumentParser(description="Speech BCI synthesis")
    parser.add_argument("--day", type=str, default=config.TEST_DAY)
    parser.add_argument("--run", type=str, default=None, help="Specific run (default: all)")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "synthesized")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else
                              "mps" if hasattr(torch.backends, "mps") and
                              torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    channel_mask = get_default_speech_channels()
    n_ch = len(channel_mask)

    # Load models
    nvad, decoder = load_models(device, n_ch)

    # Determine data directory for the day
    if args.day in config.TRAIN_DAYS:
        data_dir = config.TRAIN_DIR
    elif args.day == config.VAL_DAY:
        data_dir = config.VAL_DIR
    elif args.day == config.TEST_DAY:
        data_dir = config.TEST_DIR
    elif args.day in config.ONLINE_DAYS:
        data_dir = config.ONLINE_DIR
    else:
        print(f"Unknown day: {args.day}")
        sys.exit(1)

    # Normalization
    syll_path = config.SYLLABLE_DIR / args.day / "SyllableRepetition_Overt.mat"
    if not syll_path.exists():
        print(f"No syllable baseline for {args.day}")
        sys.exit(1)
    mu, sd = compute_day_stats(syll_path)

    # Discover runs
    day_dir = data_dir / args.day
    runs = [args.run] if args.run else discover_runs(day_dir)
    print(f"Processing {args.day}: {runs}")

    args.output.mkdir(parents=True, exist_ok=True)
    all_results = []

    for run_id in runs:
        run = load_run(data_dir, args.day, run_id)
        hg, hg_times = extract_hg(run.ecog)
        hg_z = normalize_hg(hg, mu, sd)
        hg_sel = hg_z[:, channel_mask]

        print(f"\n  {run_id}: {len(run.trials)} trials")

        for idx, row in run.trials.iterrows():
            word = row["word"]
            sf = max(0, int(row["start"] * 100) - 50)
            ef = min(len(hg_sel), int(row["end"] * 100) + 50)

            vad_pred, lpc_pred = synthesize_trial(hg_sel[sf:ef], nvad, decoder, device)

            speech_frames = vad_pred.sum()
            lpc_speech = lpc_pred[vad_pred == 1] if speech_frames > 5 else lpc_pred

            result = {
                "word": word,
                "run": run_id,
                "trial_idx": idx,
                "vad": vad_pred,
                "lpc": lpc_pred,
                "lpc_speech": lpc_speech,
                "speech_frames": int(speech_frames),
            }
            all_results.append(result)

            # Save LPC features
            out_prefix = f"{args.day}_{run_id}_t{idx:02d}_{word}"
            np.save(args.output / f"{out_prefix}_lpc.npy", lpc_speech.astype(np.float32))

            # Try LPCNet synthesis
            try:
                from speech_bci.vocoder import synthesize
                audio = synthesize(lpc_speech.astype(np.float32))
                wavfile.write(str(args.output / f"{out_prefix}.wav"), 16000, audio)
                print(f"    {word}: synthesized {len(audio)/16000:.2f}s")
            except RuntimeError:
                print(f"    {word}: {speech_frames} speech frames (LPC saved, no vocoder)")

    # Save summary plot
    n_show = min(12, len(all_results))
    fig, axes = plt.subplots(n_show, 2, figsize=(16, 2.5 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        r = all_results[i]
        T = len(r["vad"])
        axes[i, 0].fill_between(range(T), r["vad"], alpha=0.4, color="red")
        axes[i, 0].set_title(f"'{r['word']}' — VAD ({r['speech_frames']} frames)")
        axes[i, 0].set_ylabel("Speech")
        axes[i, 1].imshow(r["lpc"][:, :18].T, aspect="auto", origin="lower", cmap="magma")
        axes[i, 1].set_title(f"'{r['word']}' — Predicted LPC")
        axes[i, 1].set_ylabel("Dim")

    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(args.output / f"synthesis_{args.day}.png", dpi=150)
    print(f"\nPlot saved: {args.output / f'synthesis_{args.day}.png'}")
    print(f"LPC features saved to: {args.output}/")
    print(f"\nTotal: {len(all_results)} trials processed")


if __name__ == "__main__":
    main()
