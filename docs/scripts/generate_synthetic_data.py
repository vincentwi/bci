#!/usr/bin/env python3
"""
Generate synthetic ECoG data that matches the format expected by preprocess.py.

Creates realistic-ish BCI2000-format .mat files with:
- 128 neural channels + 3 analog channels = 131 total
- Embedded class-specific spatiotemporal patterns in high-gamma band
- 6 keyword classes (Up, Down, Left, Right, Enter, Back)
- Multiple runs to support Leave-One-Run-Out CV

The patterns are designed so models can learn speech-intent classification.
"""
import numpy as np
import scipy.io
import scipy.signal
from pathlib import Path
from config import *

np.random.seed(SEED)

N_RUNS = 6
TRIALS_PER_CLASS = 20  # per run
TRIAL_DURATION_S = 2.5
ISI_RANGE = (1.0, 2.0)  # inter-stimulus interval
N_TOTAL_CH = N_NEURAL + 3  # 131

# Each class has a distinct spatial pattern (which channels activate)
# and temporal pattern (when they activate relative to onset)
def generate_class_patterns(n_classes=6, n_channels=128):
    """Generate distinct spatiotemporal patterns for each class."""
    patterns = {}
    for c in range(1, n_classes + 1):
        # Each class activates a different subset of ~20-40 channels
        rng = np.random.RandomState(SEED + c)
        n_active = rng.randint(20, 40)
        active_ch = rng.choice(n_channels, n_active, replace=False)

        # Spatial weights (how strongly each active channel responds)
        spatial = np.zeros(n_channels)
        spatial[active_ch] = rng.exponential(1.0, n_active)
        spatial[active_ch] *= rng.choice([-1, 1], n_active)  # some inhibitory

        # Temporal profile: onset latency + duration vary by class
        onset_ms = 100 + c * 30  # 130-280ms post-stimulus
        duration_ms = 300 + c * 50  # 350-600ms

        patterns[c] = {
            'spatial': spatial,
            'onset_ms': onset_ms,
            'duration_ms': duration_ms,
            'active_ch': active_ch,
        }
    return patterns


def generate_temporal_kernel(onset_ms, duration_ms, fs=FS):
    """Generate a temporal activation kernel."""
    onset_samp = int(onset_ms * fs / 1000)
    dur_samp = int(duration_ms * fs / 1000)
    total = onset_samp + dur_samp + int(0.5 * fs)  # extra tail

    kernel = np.zeros(total)
    # Ramp up
    ramp = int(50 * fs / 1000)
    t_rise = np.linspace(0, 1, ramp)
    kernel[onset_samp:onset_samp + ramp] = t_rise
    # Sustain
    kernel[onset_samp + ramp:onset_samp + dur_samp] = 1.0
    # Decay
    decay_len = int(200 * fs / 1000)
    if onset_samp + dur_samp + decay_len <= total:
        kernel[onset_samp + dur_samp:onset_samp + dur_samp + decay_len] = \
            np.exp(-np.linspace(0, 3, decay_len))

    return kernel


def generate_run(run_idx, patterns, n_classes=6, trials_per_class=TRIALS_PER_CLASS):
    """Generate one run of synthetic ECoG data."""
    rng = np.random.RandomState(SEED + run_idx * 100)

    # Build trial sequence (randomized order)
    trial_labels = np.repeat(np.arange(1, n_classes + 1), trials_per_class)
    rng.shuffle(trial_labels)
    n_trials = len(trial_labels)

    # Calculate total duration
    trial_samples = int(TRIAL_DURATION_S * FS)
    isi_samples = [int(rng.uniform(*ISI_RANGE) * FS) for _ in range(n_trials)]
    total_samples = sum(isi_samples) + n_trials * trial_samples + int(2 * FS)  # padding

    # Generate background neural signal (1/f noise + 60Hz line noise)
    signal = np.zeros((total_samples, N_TOTAL_CH), dtype=np.float32)

    for ch in range(N_NEURAL):
        # Pink noise (1/f)
        white = rng.randn(total_samples)
        # Simple 1/f approximation via filtering
        b, a = scipy.signal.butter(2, 1.0, btype='low', fs=FS)
        pink = scipy.signal.lfilter(b, a, white) * 5
        signal[:, ch] = pink + rng.randn(total_samples) * 2

        # Add 60 Hz line noise
        t = np.arange(total_samples) / FS
        signal[:, ch] += 3.0 * np.sin(2 * np.pi * 60 * t + rng.uniform(0, 2 * np.pi))
        signal[:, ch] += 1.0 * np.sin(2 * np.pi * 120 * t + rng.uniform(0, 2 * np.pi))

        # Add some broadband neural activity
        signal[:, ch] += rng.randn(total_samples) * 0.5

    # Add a couple bad channels (very high variance or kurtosis)
    bad_candidates = [17, 64, 98]  # fixed bad channels
    for bc in bad_candidates:
        if bc < N_NEURAL:
            signal[:, bc] *= 10  # high variance
            # Add spikes for high kurtosis
            spike_times = rng.choice(total_samples, 50, replace=False)
            signal[spike_times, bc] += rng.randn(50) * 100

    # Build stimulus code array and inject class-specific patterns
    stim_code = np.zeros(total_samples, dtype=np.int32)
    pos = int(1.0 * FS)  # start after 1s padding

    for trial_idx, label in enumerate(trial_labels):
        # Mark stimulus period
        stim_code[pos:pos + trial_samples] = label

        pat = patterns[label]
        kernel = generate_temporal_kernel(pat['onset_ms'], pat['duration_ms'])
        kernel_len = min(len(kernel), total_samples - pos)

        # Inject high-gamma band activity pattern
        for ch in range(N_NEURAL):
            if pat['spatial'][ch] != 0:
                # Generate burst of high-gamma activity (70-150 Hz)
                hg_burst = np.zeros(kernel_len)
                for freq in range(70, 151, 10):
                    phase = rng.uniform(0, 2 * np.pi)
                    amp = abs(pat['spatial'][ch]) * (0.3 + 0.7 * rng.random())
                    t_burst = np.arange(kernel_len) / FS
                    hg_burst += amp * np.sin(2 * np.pi * freq * t_burst + phase)

                # Modulate by temporal kernel
                modulated = hg_burst * kernel[:kernel_len] * np.sign(pat['spatial'][ch])

                # Add trial-to-trial variability
                trial_noise = 0.3 + 0.7 * rng.random()
                signal[pos:pos + kernel_len, ch] += modulated * trial_noise

        pos += trial_samples + isi_samples[trial_idx]

    # Trim to actual used length
    signal = signal[:pos + int(1.0 * FS)]
    stim_code = stim_code[:pos + int(1.0 * FS)]

    # Create gain array (typical BCI2000 gains)
    gain = np.ones(N_TOTAL_CH, dtype=np.float64) * 0.0298  # typical µV/bit

    # Scale signal to be in "raw ADC" units (divide by gain, will be multiplied back)
    signal_raw = signal / gain[np.newaxis, :]

    # Build .mat structure matching BCI2000 format
    # parameters.SourceChGain.NumericValue = gain
    # parameters.Stimuli.Value = keyword names
    # signal = raw signal
    # states.StimulusCode = stim_code

    stim_labels = np.array(KEYWORD_NAMES, dtype=object)
    stim_value = np.empty((2, len(stim_labels)), dtype=object)
    stim_value[0] = stim_labels
    stim_value[1] = stim_labels

    gain_struct = np.array([(gain,)], dtype=[('NumericValue', object)])
    stim_struct = np.array([(stim_value,)], dtype=[('Value', object)])
    params = np.array([(gain_struct, stim_struct)],
                       dtype=[('SourceChGain', object), ('Stimuli', object)])

    states = np.array([(stim_code,)], dtype=[('StimulusCode', object)])

    return {
        'signal': signal_raw.astype(np.float64),
        'parameters': params,
        'states': states,
    }


def main():
    print("Generating synthetic ECoG data...")
    print(f"  {N_RUNS} runs × {TRIALS_PER_CLASS} trials/class × {N_CLASSES} classes")
    print(f"  {N_NEURAL} neural channels + 3 analog = {N_TOTAL_CH} total")

    patterns = generate_class_patterns()

    # Print pattern info
    for c in range(1, N_CLASSES + 1):
        p = patterns[c]
        print(f"  Class {c} ({KEYWORD_NAMES[c-1]}): {len(p['active_ch'])} active ch, "
              f"onset={p['onset_ms']}ms, dur={p['duration_ms']}ms")

    # Create output directory
    KEYWORD_DIR.mkdir(parents=True, exist_ok=True)

    for run in range(N_RUNS):
        run_id = f"R0{run + 1}"
        print(f"\n  Generating run {run_id}...")
        mat_data = generate_run(run, patterns)

        filepath = KEYWORD_DIR / f"KeywordReading_Overt_{run_id}.mat"
        scipy.io.savemat(str(filepath), mat_data, do_compression=True)

        n_samples = mat_data['signal'].shape[0]
        print(f"    Saved: {filepath.name} ({n_samples/FS:.1f}s, "
              f"{filepath.stat().st_size/1e6:.1f} MB)")

    # Also generate syllable data
    print("\n  Generating syllable data...")
    SYLLABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Use 12 syllable classes with different patterns
    syl_patterns = {}
    for c in range(1, 13):
        rng = np.random.RandomState(SEED + c + 100)
        n_active = rng.randint(15, 35)
        active_ch = rng.choice(N_NEURAL, n_active, replace=False)
        spatial = np.zeros(N_NEURAL)
        spatial[active_ch] = rng.exponential(0.8, n_active)
        spatial[active_ch] *= rng.choice([-1, 1], n_active)
        syl_patterns[c] = {
            'spatial': spatial,
            'onset_ms': 80 + c * 20,
            'duration_ms': 250 + c * 40,
            'active_ch': active_ch,
        }

    # Generate one long syllable run
    syl_rng = np.random.RandomState(SEED + 999)
    n_syl_trials_per_class = 15
    syl_labels = np.repeat(np.arange(1, 13), n_syl_trials_per_class)
    syl_rng.shuffle(syl_labels)

    trial_samples = int(TRIAL_DURATION_S * FS)
    total_syl = int(2 * FS) + len(syl_labels) * (trial_samples + int(1.5 * FS))

    syl_signal = np.zeros((total_syl, N_TOTAL_CH), dtype=np.float32)
    syl_stim = np.zeros(total_syl, dtype=np.int32)

    for ch in range(N_NEURAL):
        white = syl_rng.randn(total_syl)
        b, a = scipy.signal.butter(2, 1.0, btype='low', fs=FS)
        syl_signal[:, ch] = scipy.signal.lfilter(b, a, white) * 5 + syl_rng.randn(total_syl) * 2
        t = np.arange(total_syl) / FS
        syl_signal[:, ch] += 3.0 * np.sin(2 * np.pi * 60 * t + syl_rng.uniform(0, 2 * np.pi))

    pos = int(1.0 * FS)
    for label in syl_labels:
        syl_stim[pos:pos + trial_samples] = label
        pat = syl_patterns[label]
        kernel = generate_temporal_kernel(pat['onset_ms'], pat['duration_ms'])
        kernel_len = min(len(kernel), total_syl - pos)
        for ch in range(N_NEURAL):
            if pat['spatial'][ch] != 0:
                hg_burst = np.zeros(kernel_len)
                for freq in range(70, 151, 10):
                    t_b = np.arange(kernel_len) / FS
                    hg_burst += abs(pat['spatial'][ch]) * np.sin(
                        2 * np.pi * freq * t_b + syl_rng.uniform(0, 2 * np.pi))
                modulated = hg_burst * kernel[:kernel_len] * np.sign(pat['spatial'][ch])
                syl_signal[pos:pos + kernel_len, ch] += modulated * (0.3 + 0.7 * syl_rng.random())
        pos += trial_samples + int(1.5 * FS)

    syl_signal = syl_signal[:pos + int(FS)]
    syl_stim = syl_stim[:pos + int(FS)]

    gain = np.ones(N_TOTAL_CH, dtype=np.float64) * 0.0298
    syl_signal_raw = syl_signal / gain[np.newaxis, :]

    syl_stim_labels = np.array(SYLLABLE_NAMES, dtype=object)
    syl_stim_value = np.empty((2, len(syl_stim_labels)), dtype=object)
    syl_stim_value[0] = syl_stim_labels
    syl_stim_value[1] = syl_stim_labels

    gain_struct = np.array([(gain,)], dtype=[('NumericValue', object)])
    stim_struct = np.array([(syl_stim_value,)], dtype=[('Value', object)])
    params = np.array([(gain_struct, stim_struct)],
                       dtype=[('SourceChGain', object), ('Stimuli', object)])
    states = np.array([(syl_stim,)], dtype=[('StimulusCode', object)])

    syl_path = SYLLABLE_DIR / "SyllableRepetition_Overt.mat"
    scipy.io.savemat(str(syl_path), {
        'signal': syl_signal_raw.astype(np.float64),
        'parameters': params,
        'states': states,
    }, do_compression=True)
    print(f"    Saved: {syl_path.name} ({syl_signal.shape[0]/FS:.1f}s, "
          f"{syl_path.stat().st_size/1e6:.1f} MB)")

    print("\nSynthetic data generation complete.")
    print(f"  Keyword data: {KEYWORD_DIR}")
    print(f"  Syllable data: {SYLLABLE_DIR}")


if __name__ == "__main__":
    main()
