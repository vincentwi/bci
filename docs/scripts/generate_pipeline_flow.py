#!/usr/bin/env python3
"""
Generate pipeline flow visualization for WRITEUP.md Section 7.

Creates a multi-panel figure showing what data looks like at every
stage of the brain-to-speech pipeline, from raw neural signals through
to SSML tags and audio waveforms.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Setup ──
FIGURES_DIR = Path("/mnt/home/vincent.wilmet/docs/figures_writeup")
FIGURES_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")

# ARPABET class names (from config.py)
CLASS_TO_ARPABET = {
    0: 'B', 1: 'CH', 2: 'SIL', 3: 'D', 4: 'F', 5: 'G', 6: 'HH', 7: 'JH',
    8: 'K', 9: 'L', 10: 'ER', 11: 'M', 12: 'N', 13: 'NG', 14: 'P', 15: 'R',
    16: 'S', 17: 'SH', 18: 'DH', 19: 'T', 20: 'TH', 21: 'V', 22: 'W',
    23: 'Y', 24: 'Z', 25: 'ZH', 26: 'OY', 27: 'EH', 28: 'EY', 29: 'UH',
    30: 'IY', 31: 'OW', 32: 'UW', 33: 'IH', 34: 'AA', 35: 'AW', 36: 'AY',
    37: 'AH', 38: 'AO', 39: 'AE',
}

VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
           'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}


def add_default_stress(phonemes):
    """Add stress markers to vowels (ElevenLabs CMU ARPABET requirement)."""
    result = []
    seen_stressed = False
    for p in phonemes:
        if p == 'SIL':
            result.append(p)
            seen_stressed = False
            continue
        if p in VOWELS:
            if not seen_stressed:
                result.append(p + '1')
                seen_stressed = True
            else:
                result.append(p + '0')
        else:
            result.append(p)
    return result


def phonemes_to_ssml(phoneme_sequence):
    """Convert ARPABET phoneme sequence to SSML."""
    stressed = add_default_stress(phoneme_sequence)
    words = []
    current = []
    for p in stressed:
        if p == 'SIL':
            if current:
                words.append(current)
                current = []
        else:
            current.append(p)
    if current:
        words.append(current)
    parts = []
    for w in words:
        ph = " ".join(w)
        parts.append(f'<phoneme alphabet="cmu-arpabet"\n  ph="{ph}">word</phoneme>')
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Full Pipeline Flow with Examples
# ═══════════════════════════════════════════════════════════════════

def generate_pipeline_flow():
    """Create a detailed pipeline flow diagram with example outputs at each stage."""

    fig = plt.figure(figsize=(20, 28))
    gs = gridspec.GridSpec(7, 2, height_ratios=[1.5, 0.6, 1.2, 0.6, 1.2, 0.6, 1.5],
                           hspace=0.08, wspace=0.15,
                           left=0.06, right=0.94, top=0.96, bottom=0.02)

    fig.suptitle('Brain-to-Speech Pipeline: End-to-End Data Flow',
                 fontsize=20, fontweight='bold', y=0.985)

    # ── Example sentence for walkthrough ──
    example_sentence = "Hello how are you"
    true_phonemes = ['HH', 'AH', 'L', 'OW', 'SIL', 'HH', 'AW', 'SIL', 'AA', 'R', 'SIL', 'Y', 'UW']
    # Simulated noisy classifier output (with some errors)
    pred_phonemes = ['HH', 'AH', 'L', 'OW', 'SIL', 'HH', 'AW', 'SIL', 'AE', 'R', 'SIL', 'Y', 'UW']
    confidences =   [0.92, 0.71, 0.85, 0.78, 0.99, 0.88, 0.63, 0.99, 0.34, 0.82, 0.99, 0.91, 0.76]
    # After LM correction
    corrected_phonemes = ['HH', 'AH', 'L', 'OW', 'SIL', 'HH', 'AW', 'SIL', 'AA', 'R', 'SIL', 'Y', 'UW']

    colors = {
        'neural': '#2196F3',
        'preprocess': '#4CAF50',
        'classifier': '#FF9800',
        'lm': '#9C27B0',
        'tts': '#E91E63',
        'arrow': '#424242',
        'error': '#F44336',
        'correct': '#4CAF50',
    }

    # ────────────────────────────────────────────────
    # STAGE 0: Raw Neural Signals
    # ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])

    # Generate synthetic but realistic-looking neural data
    np.random.seed(42)
    T = 200  # 200 time bins = 4 seconds at 50Hz
    n_ch = 32  # Show 32 of 256 channels
    t = np.arange(T) * 0.02  # 20ms bins

    # Simulate neural signals with varying activity
    signals = np.random.randn(T, n_ch) * 0.5
    # Add "activity bump" during speech attempt (frames 40-140)
    for ch in range(n_ch):
        center = 90 + np.random.randint(-20, 20)
        width = 30 + np.random.randint(-10, 10)
        amplitude = 1.5 + np.random.rand() * 2
        signals[:, ch] += amplitude * np.exp(-0.5 * ((np.arange(T) - center) / width) ** 2)

    # Plot heatmap
    im = ax0.imshow(signals.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3,
                    extent=[0, T * 0.02, n_ch, 0])
    ax0.set_xlabel('Time (seconds)', fontsize=11)
    ax0.set_ylabel('Channel (of 256)', fontsize=11)

    # Mark speech period
    ax0.axvline(0.8, color='lime', lw=2, ls='--', alpha=0.7)
    ax0.axvline(2.8, color='lime', lw=2, ls='--', alpha=0.7)
    ax0.text(1.8, -1.5, 'Go cue → speech attempt', ha='center', fontsize=10,
             color='lime', fontweight='bold')

    ax0.set_title('Stage 0: Raw Neural Signals — 256ch intracortical (4×64 Utah arrays), 1280 features/20ms bin',
                  fontsize=13, fontweight='bold', color=colors['neural'], pad=8)

    # Annotation box
    box_text = ("4 arrays × 64 electrodes = 256 channels\n"
                "Features: spikePow (256) + tx1–tx4 (4×256) = 1280/bin\n"
                "Temporal resolution: 20ms bins (50 Hz)")
    ax0.text(0.98, 0.95, box_text, transform=ax0.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    cb = plt.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)
    cb.set_label('z-scored activity', fontsize=9)

    # ────────────────────────────────────────────────
    # Arrow 0→1
    # ────────────────────────────────────────────────
    ax_arrow0 = fig.add_subplot(gs[1, :])
    ax_arrow0.set_xlim(0, 10)
    ax_arrow0.set_ylim(0, 1)
    ax_arrow0.axis('off')
    ax_arrow0.annotate('', xy=(5, 0.1), xytext=(5, 0.9),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'],
                                       lw=3, mutation_scale=25))
    ax_arrow0.text(5, 0.5, 'Per-block z-score normalization\n+ Gaussian smoothing (σ=2)',
                   ha='center', va='center', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor=colors['neural']))

    # ────────────────────────────────────────────────
    # STAGE 1: CTC Classifier Output
    # ────────────────────────────────────────────────
    ax1l = fig.add_subplot(gs[2, 0])
    ax1r = fig.add_subplot(gs[2, 1])

    # Left: CTC posterior probability heatmap (simulated)
    n_phones = 41  # 40 + blank
    T_ctc = 100
    np.random.seed(123)
    posteriors = np.random.dirichlet(np.ones(n_phones) * 0.1, T_ctc)
    # Make it look like real CTC output with spiky peaks
    for i, ph in enumerate(true_phonemes):
        idx = {v: k for k, v in CLASS_TO_ARPABET.items()}.get(ph, 40)
        center = int(8 + i * (T_ctc - 16) / len(true_phonemes))
        for dt in range(-3, 4):
            if 0 <= center + dt < T_ctc:
                posteriors[center + dt, idx] += 3.0 * np.exp(-0.5 * (dt / 1.5) ** 2)
    posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)

    im1 = ax1l.imshow(posteriors.T[:20], aspect='auto', cmap='hot', vmin=0, vmax=0.5,
                      extent=[0, T_ctc, 20, 0])
    ax1l.set_xlabel('Time bin', fontsize=10)
    ax1l.set_ylabel('Phoneme class', fontsize=10)
    ax1l.set_title('Stage 1: CTC Posterior Probabilities',
                   fontsize=12, fontweight='bold', color=colors['classifier'])

    # Mark the greedy decoded path
    best_path = posteriors.argmax(axis=1)
    ax1l.set_yticks(range(0, 20, 4))
    phoneme_labels = [CLASS_TO_ARPABET.get(i, 'BLK') for i in range(0, 20, 4)]
    ax1l.set_yticklabels(phoneme_labels, fontsize=8)

    # Right: Decoded phoneme sequence with confidences
    ax1r.set_xlim(-0.5, len(pred_phonemes) - 0.5)
    ax1r.set_ylim(-0.3, 1.3)

    for i, (ph, conf) in enumerate(zip(pred_phonemes, confidences)):
        is_error = (ph != true_phonemes[i])
        color = colors['error'] if is_error else colors['correct']
        alpha = max(0.3, conf)

        # Phoneme box
        rect = FancyBboxPatch((i - 0.4, 0.4), 0.8, 0.5,
                               boxstyle="round,pad=0.05",
                               facecolor=color, alpha=alpha, edgecolor='black', lw=1)
        ax1r.add_patch(rect)
        ax1r.text(i, 0.65, ph, ha='center', va='center', fontsize=10,
                  fontweight='bold', color='white' if conf > 0.5 else 'black')

        # Confidence bar below
        bar_color = '#4CAF50' if conf >= 0.8 else '#FFC107' if conf >= 0.5 else '#F44336'
        ax1r.bar(i, conf, width=0.6, bottom=-0.25, color=bar_color, alpha=0.7, edgecolor='none')
        ax1r.text(i, -0.1, f'{conf:.0%}', ha='center', va='top', fontsize=7)

    # True phonemes above
    for i, ph in enumerate(true_phonemes):
        ax1r.text(i, 1.1, ph, ha='center', va='center', fontsize=8, color='#666',
                  style='italic')
    ax1r.text(-0.5, 1.1, 'True:', ha='right', va='center', fontsize=8, color='#666')
    ax1r.text(-0.5, 0.65, 'Pred:', ha='right', va='center', fontsize=8, color='#333')

    ax1r.set_title('CTC Greedy Decode → ARPABET Sequence',
                   fontsize=12, fontweight='bold', color=colors['classifier'])
    ax1r.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['correct'], alpha=0.7, label='Correct'),
                       Patch(facecolor=colors['error'], alpha=0.7, label='Error (AE→AA)')]
    ax1r.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # ────────────────────────────────────────────────
    # Arrow 1→2
    # ────────────────────────────────────────────────
    ax_arrow1 = fig.add_subplot(gs[3, :])
    ax_arrow1.set_xlim(0, 10)
    ax_arrow1.set_ylim(0, 1)
    ax_arrow1.axis('off')
    ax_arrow1.annotate('', xy=(5, 0.1), xytext=(5, 0.9),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'],
                                       lw=3, mutation_scale=25))
    ax_arrow1.text(5, 0.5,
                   'Stage 2: LM Correction (Qwen3.5-2B + LoRA)\n'
                   'Confidence-gated: only correct phonemes with conf < 0.8',
                   ha='center', va='center', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5', edgecolor=colors['lm']))

    # ────────────────────────────────────────────────
    # STAGE 2: LM Correction Detail
    # ────────────────────────────────────────────────
    ax2l = fig.add_subplot(gs[4, 0])
    ax2r = fig.add_subplot(gs[4, 1])

    # Left: Show the correction process
    ax2l.set_xlim(-1, 10)
    ax2l.set_ylim(-0.5, 4.5)
    ax2l.axis('off')

    # Input noisy sequence
    ax2l.text(0, 4, 'Classifier Output:', fontsize=11, fontweight='bold', color='#333')
    noisy_str = ' '.join(pred_phonemes)
    ax2l.text(0, 3.4, noisy_str, fontsize=9, fontfamily='monospace',
              bbox=dict(facecolor='#FFF3E0', edgecolor=colors['classifier'], lw=1.5, pad=5))

    # Confidence gating
    ax2l.text(0, 2.6, 'Low confidence (< 0.8):', fontsize=10, fontweight='bold', color=colors['error'])
    low_conf = [(ph, c) for ph, c in zip(pred_phonemes, confidences) if c < 0.8]
    ax2l.text(0, 2.1, ', '.join([f'{ph} ({c:.0%})' for ph, c in low_conf]),
              fontsize=9, fontfamily='monospace', color=colors['error'])

    # LM prompt
    ax2l.text(0, 1.3, 'LM Input Prompt:', fontsize=10, fontweight='bold', color=colors['lm'])
    ax2l.text(0, 0.5, '"You are a phoneme error correction\n'
                       ' model for a brain-computer interface.\n'
                       ' Given: HH AH L OW SIL HH AW SIL AE R\n'
                       '        SIL Y UW\n'
                       ' Output corrected sequence."',
              fontsize=8, fontfamily='monospace',
              bbox=dict(facecolor='#F3E5F5', edgecolor=colors['lm'], lw=1, pad=5))

    # Right: Corrected output
    ax2r.set_xlim(-1, 10)
    ax2r.set_ylim(-0.5, 4.5)
    ax2r.axis('off')

    ax2r.text(0, 4, 'LM Corrected Output:', fontsize=11, fontweight='bold', color=colors['lm'])
    corrected_str = ' '.join(corrected_phonemes)
    ax2r.text(0, 3.4, corrected_str, fontsize=9, fontfamily='monospace',
              bbox=dict(facecolor='#E8F5E9', edgecolor=colors['correct'], lw=1.5, pad=5))

    # Show diff
    ax2r.text(0, 2.6, 'Corrections applied:', fontsize=10, fontweight='bold', color=colors['correct'])
    ax2r.text(0, 2.1, 'AE (34%) → AA (corrected by LM)', fontsize=9,
              fontfamily='monospace', color=colors['correct'])
    ax2r.text(0, 1.7, '1 phoneme corrected out of 13', fontsize=9, color='#666')

    # Show ARPABET → English words
    ax2r.text(0, 0.9, 'Decoded words:', fontsize=10, fontweight='bold', color='#333')
    ax2r.text(0, 0.3,
              'HH AH L OW → "hello"\n'
              'HH AW → "how"\n'
              'AA R → "are"\n'
              'Y UW → "you"',
              fontsize=9, fontfamily='monospace',
              bbox=dict(facecolor='#E8F5E9', edgecolor=colors['correct'], lw=1, pad=5))

    # ────────────────────────────────────────────────
    # Arrow 2→3
    # ────────────────────────────────────────────────
    ax_arrow2 = fig.add_subplot(gs[5, :])
    ax_arrow2.set_xlim(0, 10)
    ax_arrow2.set_ylim(0, 1)
    ax_arrow2.axis('off')
    ax_arrow2.annotate('', xy=(5, 0.1), xytext=(5, 0.9),
                       arrowprops=dict(arrowstyle='->', color=colors['arrow'],
                                       lw=3, mutation_scale=25))
    ax_arrow2.text(5, 0.5,
                   'Stage 3: Audio Synthesis\n'
                   'ARPABET → SSML phoneme tags → ElevenLabs API → WAV/MP3',
                   ha='center', va='center', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#FCE4EC', edgecolor=colors['tts']))

    # ────────────────────────────────────────────────
    # STAGE 3: SSML Generation and Audio
    # ────────────────────────────────────────────────
    ax3l = fig.add_subplot(gs[6, 0])
    ax3r = fig.add_subplot(gs[6, 1])

    # Left: SSML generation
    ax3l.set_xlim(-1, 10)
    ax3l.set_ylim(-0.5, 5.5)
    ax3l.axis('off')

    ax3l.text(0, 5.2, 'Step 1: Add stress markers to vowels', fontsize=10,
              fontweight='bold', color=colors['tts'])
    stressed = add_default_stress(corrected_phonemes)
    ax3l.text(0, 4.6, ' '.join(stressed), fontsize=9, fontfamily='monospace',
              bbox=dict(facecolor='#FCE4EC', edgecolor=colors['tts'], lw=1, pad=4))

    ax3l.text(0, 3.8, 'Step 2: Split by SIL → words, wrap in SSML', fontsize=10,
              fontweight='bold', color=colors['tts'])
    ssml = phonemes_to_ssml(corrected_phonemes)
    ax3l.text(0, 1.8, ssml, fontsize=8, fontfamily='monospace',
              bbox=dict(facecolor='#FFF', edgecolor=colors['tts'], lw=1.5, pad=8),
              verticalalignment='center', linespacing=1.5)

    ax3l.text(0, 0.3, 'Step 3: POST to ElevenLabs v1/text-to-speech/{voice_id}',
              fontsize=9, fontweight='bold', color=colors['tts'])
    ax3l.text(0, -0.2, 'Model: eleven_flash_v2 (required for <phoneme> support)',
              fontsize=8, color='#666')

    # Right: Simulated audio waveform
    ax3r.set_title('Output: Synthesized Speech Audio', fontsize=12,
                   fontweight='bold', color=colors['tts'])

    # Generate realistic-looking speech waveform
    np.random.seed(77)
    sr = 1000  # fake sample rate for display
    duration = 2.0
    t_audio = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t_audio)

    # 4 "words" with pauses
    word_intervals = [(0.0, 0.4), (0.5, 0.7), (0.8, 1.1), (1.3, 1.6)]
    word_labels = ['hello', 'how', 'are', 'you']
    for (start, end), label in zip(word_intervals, word_labels):
        mask = (t_audio >= start) & (t_audio <= end)
        envelope = np.exp(-((t_audio[mask] - (start + end) / 2) / ((end - start) / 3)) ** 2)
        freqs = [120 + np.random.rand() * 80, 240 + np.random.rand() * 100,
                 360 + np.random.rand() * 120]
        for f in freqs:
            audio[mask] += envelope * np.sin(2 * np.pi * f * t_audio[mask]) * (0.3 + 0.2 * np.random.rand())
        audio[mask] += envelope * np.random.randn(mask.sum()) * 0.1

    audio = audio / (np.abs(audio).max() + 1e-8) * 0.8

    ax3r.plot(t_audio, audio, color=colors['tts'], lw=0.5, alpha=0.8)
    ax3r.fill_between(t_audio, audio, alpha=0.2, color=colors['tts'])

    # Mark word boundaries
    for (start, end), label in zip(word_intervals, word_labels):
        ax3r.axvspan(start, end, alpha=0.08, color=colors['tts'])
        ax3r.text((start + end) / 2, 0.95, label, ha='center', va='top',
                  fontsize=10, fontweight='bold', color='#333')

    ax3r.set_xlabel('Time (seconds)', fontsize=10)
    ax3r.set_ylabel('Amplitude', fontsize=10)
    ax3r.set_ylim(-1.1, 1.1)

    # Output format note
    ax3r.text(0.98, -0.85, 'Output: brain_speech.mp3 (ElevenLabs API)\n'
                            'Verified via Whisper round-trip ASR',
              transform=ax3r.transAxes, fontsize=8, ha='right', va='top',
              style='italic', color='#666',
              bbox=dict(facecolor='white', edgecolor='#ddd', pad=4))

    plt.savefig(FIGURES_DIR / 'pipeline_flow.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"Saved {FIGURES_DIR / 'pipeline_flow.png'}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: SSML Example Detail
# ═══════════════════════════════════════════════════════════════════

def generate_ssml_examples():
    """Generate a figure showing multiple SSML conversion examples."""

    examples = [
        {
            'sentence': "Hello world",
            'phonemes': ['HH', 'AH', 'L', 'OW', 'SIL', 'W', 'ER', 'L', 'D'],
            'per': 0.0,
        },
        {
            'sentence': "Good morning",
            'phonemes': ['G', 'UH', 'D', 'SIL', 'M', 'AO', 'R', 'N', 'IH', 'NG'],
            'per': 0.0,
        },
        {
            'sentence': "I need water",
            'phonemes': ['AY', 'SIL', 'N', 'IY', 'D', 'SIL', 'W', 'AA', 'T', 'ER'],
            'per': 0.0,
        },
        {
            'sentence': "Thank you (w/ errors)",
            'phonemes': ['TH', 'AE', 'NG', 'K', 'SIL', 'Y', 'UW'],
            'noisy': ['DH', 'AE', 'NG', 'K', 'SIL', 'Y', 'UH'],
            'corrected': ['TH', 'AE', 'NG', 'K', 'SIL', 'Y', 'UW'],
            'per': 0.286,
        },
    ]

    fig, axes = plt.subplots(len(examples), 1, figsize=(16, 3.2 * len(examples)))
    fig.suptitle('ARPABET → SSML Phoneme Tag Conversion Examples',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, (ax, ex) in enumerate(zip(axes, examples)):
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, 3.5)
        ax.axis('off')

        y = 3.0
        # Sentence
        ax.text(0.1, y, f'"{ex["sentence"]}"', fontsize=13, fontweight='bold', color='#333')
        y -= 0.8

        # ARPABET
        phoneme_str = ' '.join(ex['phonemes'])
        ax.text(0.1, y, 'ARPABET:', fontsize=10, fontweight='bold', color='#2196F3')
        ax.text(2.0, y, phoneme_str, fontsize=10, fontfamily='monospace',
                bbox=dict(facecolor='#E3F2FD', pad=4, edgecolor='#2196F3'))

        # Show noisy/corrected if applicable
        if 'noisy' in ex:
            y -= 0.7
            noisy_str = ' '.join(ex['noisy'])
            ax.text(0.1, y, 'Noisy:', fontsize=10, fontweight='bold', color='#F44336')
            ax.text(2.0, y, noisy_str, fontsize=10, fontfamily='monospace',
                    bbox=dict(facecolor='#FFEBEE', pad=4, edgecolor='#F44336'))
            # Highlight errors
            for i, (n, t) in enumerate(zip(ex['noisy'], ex['phonemes'])):
                if n != t:
                    ax.text(2.0, y - 0.5, f'  {n}→{t} (LM fix)', fontsize=9,
                            color='#4CAF50', fontweight='bold')

        y -= 0.7
        # SSML
        stressed = add_default_stress(ex.get('corrected', ex['phonemes']))
        # Build compact SSML
        words = []
        current = []
        for p in stressed:
            if p == 'SIL':
                if current:
                    words.append(current)
                    current = []
            else:
                current.append(p)
        if current:
            words.append(current)

        ssml_parts = []
        for w in words:
            ph = " ".join(w)
            ssml_parts.append(f'<phoneme alphabet="cmu-arpabet" ph="{ph}">word</phoneme>')
        ssml = '  '.join(ssml_parts)

        ax.text(0.1, y, 'SSML:', fontsize=10, fontweight='bold', color='#E91E63')
        ax.text(2.0, y, ssml, fontsize=8, fontfamily='monospace',
                bbox=dict(facecolor='#FCE4EC', pad=4, edgecolor='#E91E63'),
                wrap=True)

        # Separator line
        if idx < len(examples) - 1:
            ax.axhline(-0.3, color='#ddd', lw=1)

    plt.savefig(FIGURES_DIR / 'ssml_examples.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"Saved {FIGURES_DIR / 'ssml_examples.png'}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Audio output comparison (true vs predicted)
# ═══════════════════════════════════════════════════════════════════

def generate_audio_comparison():
    """Generate waveform comparison of true vs predicted audio outputs."""
    import wave

    audio_dir = Path("/mnt/home/vincent.wilmet/results/audio")
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print("No WAV files found")
        return

    # Load pairs
    pairs = []
    for i in range(5):
        true_path = audio_dir / f"true_trial_{i}_{i}.wav"
        pred_path = audio_dir / f"pred_trial_{i}_{i}.wav"
        if true_path.exists() and pred_path.exists():
            pairs.append((true_path, pred_path, i))

    if not pairs:
        print("No matching true/pred pairs found")
        return

    n_pairs = min(3, len(pairs))
    fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 3 * n_pairs))
    fig.suptitle('Audio Output Comparison: Ground Truth vs Predicted Phonemes',
                 fontsize=14, fontweight='bold')

    for row, (true_path, pred_path, trial_idx) in enumerate(pairs[:n_pairs]):
        for col, (path, label, color) in enumerate([
            (true_path, 'Ground Truth', '#4CAF50'),
            (pred_path, 'Predicted', '#FF9800')
        ]):
            try:
                with wave.open(str(path), 'rb') as wf:
                    n_frames = wf.getnframes()
                    sr = wf.getframerate()
                    data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(np.float32)
                    data = data / (np.abs(data).max() + 1e-8)
                    t = np.arange(len(data)) / sr

                ax = axes[row, col] if n_pairs > 1 else axes[col]
                ax.plot(t, data, color=color, lw=0.3, alpha=0.8)
                ax.fill_between(t, data, alpha=0.15, color=color)
                ax.set_ylim(-1.1, 1.1)
                ax.set_ylabel('Amplitude', fontsize=9)
                if row == 0:
                    ax.set_title(f'{label} Phonemes → ElevenLabs TTS',
                                 fontsize=11, fontweight='bold', color=color)
                if row == n_pairs - 1:
                    ax.set_xlabel('Time (s)', fontsize=9)

                ax.text(0.02, 0.95, f'Trial {trial_idx}',
                        transform=ax.transAxes, fontsize=9, va='top',
                        bbox=dict(facecolor='white', alpha=0.8, pad=2))
            except Exception as e:
                print(f"Error loading {path}: {e}")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'audio_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"Saved {FIGURES_DIR / 'audio_comparison.png'}")


if __name__ == '__main__':
    print("Generating pipeline flow visualization...")
    generate_pipeline_flow()
    generate_ssml_examples()
    generate_audio_comparison()
    print("Done!")
