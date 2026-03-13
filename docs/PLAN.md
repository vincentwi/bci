# Speech BCI: Scaling Plan & Engineering Playbook

> Everything I've learned building the EDA + model notebooks on the Angrick et al. (2024)
> ECoG speech dataset. Written so a competent engineer can pick this up and go.

---

## The Problem

A person with ALS has two 8×8 ECoG grids (128 channels) implanted on their left
sensorimotor cortex. They try to speak a word. We record the electrical activity
from the brain surface at 1,000 Hz. We also record what they actually say with a
microphone at 16 kHz. The goal: given ONLY the brain signal, reconstruct what they
said — either as a classified word label or, better, as actual synthesized audio.

---

## What the Data Actually Is

```
OSF: https://osf.io/49rt7/
Paper: Scientific Reports 14:9617 (2024)
Code: https://github.com/cronelab/delayed-speech-synthesis
```

### File structure (per run)
```
KeywordReading_Overt_R01.mat   — raw ECoG signal (241k samples × 131 channels, int16)
KeywordReading_Overt_R01.wav   — time-aligned microphone audio (16 kHz, float32)
KeywordReading_Overt_R01_trials.lab  — tab-separated: start_sec \t end_sec \t word
KeywordReading_Overt_R01_CI.xlsx     — channel confidence intervals (which channels matter)
```

### Dataset splits
```
train/           8 session days, 4 runs each = ~1,570 trials total
  2022_09_22/    (R01-R04, 240 trials)
  2022_09_23/    ...
  2022_09_28/    ...
  2022_09_30/    ...
  2022_10_05/    ...
  2022_10_06/    ...
  2022_10_10/    ...
  2022_10_27/    ...
validation/      1 session day = ~70 trials
  2022_11_04/
test/            1 session day = ~70 trials
  2022_11_03/
online_sessions/ 3 sessions (closed-loop, 5.5 months after training)
  2023_04_14/
  2023_04_18/
  2023_04_21/

SyllableRepetition/  1 file per session day — baseline for normalization
```

### The .mat structure (BCI2000 format)
```python
mat['signal']           # (n_samples, 131) int16
                        #   channels 0-127: ECoG (gain=0.25 µV/unit)
                        #   channels 128-130: ainp1-3 (analog inputs, gain=0.152593)
mat['parameters']       # nested structured array with BCI2000 params
mat['states']           # contains StimulusCode: uint16 per sample
                        #   0=silence, 1=Up, 2=Down, 3=Left, 4=Right, 5=Enter, 6=Back
```

### CRITICAL: Channel name parsing
The channel names are stored in a deeply nested BCI2000 parameter structure.
This is NOT a simple array — it bit us during development.

```python
# WRONG (what we tried first — gives 1 name):
ch_arr = params['ChannelNames'][0, 0]['Value'][0, 0].flatten()

# RIGHT (actual structure is (1, 131) array of 1-element arrays):
ch_val = params['ChannelNames'][0, 0]['Value']  # shape (1, 131)
ch_names = [str(ch_val[0, i].flat[0]) for i in range(ch_val.shape[1])]
```

The channel names are NOT standard 10-20 positions — they're grid electrode
numbers (chan1-chan128). The mapping to cortical anatomy requires the implant
geometry, not EEG conventions.

---

## The Signal Processing Pipeline

This is where most of the value is. Get this wrong and no model will save you.

### Step 1: Common Average Reference (CAR)
```python
# WHY: removes common electrical noise shared across channels on each grid
# Each grid is physically separate, so they have different common noise
car = ecog.copy()
car[:, :64]  -= car[:, :64].mean(axis=1, keepdims=True)   # grid 1
car[:, 64:]  -= car[:, 64:].mean(axis=1, keepdims=True)   # grid 2
```

### Step 2: High-gamma extraction (70-170 Hz)
```python
# WHY: high-gamma power tracks local cortical population activity
# It's the strongest correlate of speech motor activity in ECoG
# We split the bandpass around 120 Hz to notch out the 2nd harmonic of
# 60 Hz line noise (US power grid)

sos_lo = butter(8, [70, 117], btype='band', fs=1000, output='sos')   # 70-117 Hz
sos_hi = butter(8, [123, 170], btype='band', fs=1000, output='sos')  # 123-170 Hz
power = sosfilt(sos_lo, car, axis=0)**2 + sosfilt(sos_hi, car, axis=0)**2
```

### Step 3: Windowed log power
```python
# WHY: smooths the noisy instantaneous power into stable features
# 50 ms windows at 10 ms hop → 100 Hz output frame rate
# Log transform compresses the dynamic range (power is lognormal)

# PERFORMANCE NOTE: the naive loop over 24k frames is ~30s per run.
# Use np.lib.stride_tricks.as_strided for ~100x speedup:
shape = (n_frames, win_samples, n_channels)
strides = (power.strides[0] * hop_samples, power.strides[0], power.strides[1])
windows = np.lib.stride_tricks.as_strided(power, shape=shape, strides=strides)
features = np.log(windows.mean(axis=1) + 1e-10)
```

### Step 4: Per-day z-score normalization
```python
# WHY: electrode impedance drifts across days, so raw power values shift.
# The syllable repetition task provides a consistent baseline per session.
# Without this, cross-day training is garbage.

syll_hg, _ = extract_hg(syllable_repetition_data)
mu = syll_hg.mean(axis=0)   # per-channel mean
sd = syll_hg.std(axis=0)    # per-channel std
normalized = (keyword_hg - mu) / (sd + 1e-10)
```

### Step 5 (paper): Contaminated channel correction
```python
# The paper uses Roussel's method to detect channels with artifacts
# and replaces them with the average of their 8 neighbors on the grid.
# We skipped this but it matters for robustness.
```

---

## What We Built vs. What The Paper Does

### What we built (1 session day, ~240 trials):
| Model | Arch | Result | Time to train |
|-------|------|--------|---------------|
| Word classifier | BiLSTM-128, 2-layer | ~50% acc (chance=16.7%) | ~5 min |
| nVAD | Uni-LSTM-150, 2-layer | ~80% acc | ~2 min |
| Acoustic decoder | BiLSTM-150, 2-layer | MSE + Pearson r | ~10 min |

### What the paper achieves (8 session days, ~1,570 trials):
| Model | Arch | Result |
|-------|------|--------|
| nVAD | Uni-LSTM-150, 2-layer, 311K params | 93.4% frame accuracy |
| Acoustic decoder | BiLSTM-100/dir, 2-layer, 378K params | 80% word intelligibility (human judged) |

**Key corrections from the paper's GitHub code (vs. what the paper text states):**
- Acoustic decoder uses **100 hidden units per direction**, not 150 as paper text implies
- Optimizer is **RMSprop with lr=0.0001**, not 0.001 as paper states
- Only **64 of 128 channels** are used (speech-relevant selection via CI analysis)
- **No data augmentation** at all — just dropout
- LPC features are **20 dimensions** (18 Bark cepstrals + 2 pitch) extracted by LPCNet's `dump_data`

The gap is mostly **data volume**, **LPC targets vs mel targets**, and **channel selection**.

---

## How to Scale: The Concrete Plan

### Phase 1: Download all data (~5 GB)

```python
# Script to download all 8 training days + val + test + syllable baselines
# Each .mat is ~60-70 MB, each .wav is ~15 MB
# Total: 8 days × 4 runs × (~80 MB) + syllable files ≈ 3-4 GB

# Use OSF API to enumerate:
# https://api.osf.io/v2/nodes/49rt7/files/osfstorage/
# Then curl each download URL.

# IMPORTANT: also download the SyllableRepetition for EVERY session day
# (not just the one we used). Each day needs its own normalization baseline.
```

### Phase 2: Build the full data pipeline

```python
# For each of the 8 training days:
#   1. Load SyllableRepetition_Overt.mat for that day
#   2. Extract HG, compute per-day (mu, sd)
#   3. For each run (R01-R04):
#      a. Load .mat → extract HG → normalize with that day's (mu, sd)
#      b. Load .wav → extract acoustic targets
#      c. Load .lab → segment into trials
#      d. Store as (hg_segment, acoustic_segment, word_label, day_id, run_id)

# Key: NEVER mix normalization stats across days.
# Key: NEVER put same-day data in both train and test.
```

### Phase 3: Replace mel targets with LPC coefficients

The paper doesn't predict mel spectrograms — it predicts **20 LPC coefficients**
per 10 ms frame, which are then fed to the **LPCNet** vocoder for synthesis.

```
LPC coefficients = 18 Bark-scale cepstral coefficients + 2 pitch parameters
                   (pitch period + pitch correlation)

# WHY: LPC is a speech-specific representation that's:
#   1. Lower-dimensional than mel (20 vs 40-80)
#   2. Directly synthesizable by LPCNet (no Griffin-Lim needed)
#   3. Captures vocal tract shape + pitch independently
```

To extract LPC:
```python
# Option A: Use the paper's code (https://github.com/cronelab/delayed-speech-synthesis)
#   They use a custom LPC extraction pipeline with Bark-scale warping
#
# Option B: Use librosa for a simpler LPC extraction:
#   librosa.lpc(audio_frame, order=18)
#   But this misses the Bark-scale warping and pitch extraction
#
# Option C: Use LPCNet's built-in feature extraction:
#   LPCNet ships with a C program (dump_data) that extracts exactly
#   the features it expects. This is the most reliable path.
```

### Phase 4: Scale the models

#### 4a. nVAD (Neural Voice Activity Detection)
```
Architecture (from GitHub, corrected):
  - 2-layer unidirectional LSTM
  - 150 hidden units per layer
  - 50% dropout
  - Binary output: speech vs. silence
  - 311,102 parameters
  - Input: 64 channels (NOT 128 — speech-relevant channels only)

Training:
  - RMSprop optimizer, lr=0.0001 (paper says 0.001 — code says 0.0001)
  - Truncated BPTT: k1=50, k2=100 frames
  - Early stopping on validation set (2022_11_04)
  - Ground truth VAD labels: energy-based detection on the audio waveform

WHY unidirectional: must work causally in real-time.
The nVAD decides when to START buffering neural data for decoding,
and when to STOP and trigger the acoustic decoder.

IMPORTANT: channel selection happens BEFORE training. The CI analysis
(.xlsx files) identifies which 64 of 128 channels carry speech info.
This halves input dimensionality and removes noise channels.
```

#### 4b. Acoustic Decoder
```
Architecture (from GitHub, corrected):
  - 2-layer bidirectional LSTM
  - 100 hidden units per direction (NOT 150 — paper text is wrong)
  - 378,420 parameters total
  - 50% dropout (only regularization — no data augmentation)
  - Output: 20 LPC coefficients per frame

Training:
  - RMSprop optimizer, lr=0.0001 (NOT 0.001)
  - Truncated BPTT: k1=50, k2=100
  - Input: 64-channel HG features for full utterance (nVAD provides boundaries)
  - Output: frame-level LPC coefficients
  - Loss: MSE on LPC coefficients

WHY bidirectional: it decodes AFTER the word is spoken (delayed synthesis).
The full utterance is available, so we can look backward and forward.
This is the key architectural insight — accept latency for accuracy.
```

#### 4c. Proper train/val/test protocol
```
Train:    8 session days (2022_09_22 through 2022_10_27) — ~1,570 trials
Val:      2022_11_04 — ~70 trials (for early stopping + hyperparameter selection)
Test:     2022_11_03 — ~70 trials (for final evaluation)
Online:   2023_04_* — 3 sessions (for closed-loop evaluation)

CRITICAL: the test day is BEFORE the validation day chronologically.
This is intentional — it prevents temporal leakage.

CRITICAL: each day has its own SyllableRepetition normalization.
```

### Phase 5: LPCNet vocoder integration

```
LPCNet is a neural vocoder that converts LPC coefficients → audio waveform.

Setup:
  1. Clone https://github.com/xiph/LPCNet
  2. Compile from C source (requires gcc, needs Linux or careful macOS setup)
  3. The paper uses a Cython wrapper for Python integration
  4. Pre-trained LPCNet weights are available (trained on generic speech)

Pipeline:
  Brain signal → HG extraction → nVAD (detect speech) →
  Acoustic decoder (predict LPC) → LPCNet (synthesize audio)

WHY this works for preserving voice:
  LPCNet is conditioned on LPC features, and LPC captures the vocal tract
  shape of the SPECIFIC speaker. So even though LPCNet was trained on
  generic speech, it reproduces the patient's voice characteristics
  because the LPC coefficients encode their unique vocal tract.
```

---

## Lessons Learned / Mistakes to Avoid

### 1. The CSV on Kaggle/OSF derivatives is FAKE
The `eeg_speech_data.csv` file floating around (which was in our neuralink/
folder) uses standard 10-20 EEG channel names and has 10 "words" — but the
real data has 128 ECoG channels named chan1-chan128 and only 6 words. The CSV
gives 100% classification accuracy. The real data gives ~45-50% with the same
methods. If your model hits >90% accuracy on this task, you have a data leak.

### 2. Mel features in the CSV ARE the label
The "mel_0" through "mel_12" columns in the CSV have ~3x less variance
within-class than between-class. They're essentially a deterministic function
of the word label. Including them as inputs is circular — you're decoding
the answer from the answer.

### 3. The HG extraction loop is a bottleneck
The naive Python loop over frames (`for i in range(24000): ...`) takes ~30s
per run. With 8 days × 4 runs = 32 runs, that's 16 minutes just on feature
extraction. Use `np.lib.stride_tricks.as_strided` for vectorized windowing.

### 4. Per-day normalization is non-negotiable
Without syllable repetition normalization, cross-day training produces garbage.
The electrode impedance drifts enough that raw power values from day 1 vs day 8
are on completely different scales. The model learns day-specific artifacts
instead of speech-specific patterns.

### 5. StimulusCode alignment matters
The .lab file gives trial boundaries in seconds. The StimulusCode in the .mat
gives per-sample labels. They should agree, but the .lab file is "cleaner" —
it was post-hoc corrected. Use .lab for trial segmentation, StimulusCode for
quick frame-level speech/silence labels.

### 6. MPS (Apple Silicon) works but has quirks
PyTorch on MPS handles LSTM fine but pack_padded_sequence can be slow.
For serious training, use CUDA. For development, MPS is fine.

### 7. 60 trials is not enough — but it's enough to validate your pipeline
With only 60 trials (one run), LOO-CV gives ~45% accuracy (chance=16.7%).
That's statistically significant (p < 0.001) but high-variance. The real
gains come from:
  - More trials (240 from one day → ~50%, 1570 from 8 days → ?)
  - Cross-day generalization (the actual hard problem)
  - Better temporal models (LSTM > mean+std aggregation)

### 8. The real task is regression, not classification
Word classification is a useful sanity check, but the paper's actual
contribution is **continuous acoustic parameter prediction** — predicting
a 20-dim LPC vector for every 10 ms frame. That's what enables speech
synthesis. Classification can never produce novel words or sentences.

---

## Architecture Decision Tree

```
Q: Do you need real-time decoding?
├── Yes → Unidirectional LSTM (causal)
│         Use for: nVAD, online word classifier
│         Latency: ~50ms per frame
│
└── No → Bidirectional LSTM (can see full utterance)
          Use for: Acoustic decoder (post-hoc synthesis)
          Accuracy: significantly better than unidirectional

Q: What's your target?
├── Word label → Classification head (softmax over 6 classes)
│                Good for: validation, quick experiments
│                Limitation: can never generalize to unseen words
│
├── Mel spectrogram → Regression head (linear → n_mels)
│                     Good for: visualization, debugging
│                     Limitation: mel → audio requires Griffin-Lim (lossy)
│
└── LPC coefficients → Regression head (linear → 20)
                        Good for: actual speech synthesis
                        Requires: LPCNet vocoder
                        This is what the paper does

Q: How much data do you have?
├── < 100 trials → Use sklearn (LDA, SVM) with mean+std features
│                  LOO-CV is appropriate
│                  Don't bother with deep learning
│
├── 100-500 trials → Small LSTM (64-128 hidden, 1-2 layers)
│                    Heavy dropout (0.5), early stopping
│                    Split by run, not randomly
│
└── > 500 trials → Full paper architecture (150 hidden, 2 layers)
                   Can start to see cross-day generalization
                   Train/val/test by day
```

---

## What I'd Do Differently Starting From Scratch

1. **Start with the GitHub repo, not the data.** The paper's code
   (github.com/cronelab/delayed-speech-synthesis) has `replicate.sh` that
   runs the full pipeline. Understanding the code tells you exactly what
   features to extract and how.

2. **Download ALL data upfront.** We wasted time downloading incrementally.
   Write a script that pulls everything, organizes it, and validates checksums.

3. **Build the data pipeline FIRST.** Before touching any model code, build:
   - A function that takes a session day → returns normalized HG features
   - A function that takes a run → returns aligned (HG, acoustic, labels) tuples
   - A DataLoader that handles variable-length sequences properly
   - Verification plots: does the HG track speech? Does normalization work?

4. **Skip mel, go straight to LPC.** We built a mel-based acoustic decoder
   as a stepping stone, but if your goal is actual speech synthesis, mel is
   a dead end — you need LPC for LPCNet.

5. **Profile your code early.** The HG extraction bottleneck cost us hours
   of wall-clock time before we vectorized it.

6. **Don't trust the CSV.** Any pre-processed derivative dataset should be
   validated against the raw data. 100% accuracy is always a red flag.

7. **Use only 64 channels, not 128.** The paper's code uses CI analysis to
   select speech-relevant channels — roughly half. Our models trained on
   all 128 channels, which means half the input is noise. The `.xlsx` CI
   files in each run's folder indicate which channels pass the significance
   threshold. Use them.

8. **Read the code, not just the paper.** The paper says lr=0.001, the code
   uses lr=0.0001. The paper implies 150 hidden units for the decoder, the
   code uses 100 per direction. Always trust the implementation over the
   manuscript — papers get simplified for readability.

---

## Realistic Timeline

```
Day 1:  Download all data, build data pipeline, verify with plots
Day 2:  Implement LPC extraction, train nVAD on full dataset
Day 3:  Train acoustic decoder on full dataset, evaluate on test day
Day 4:  Integrate LPCNet, end-to-end synthesis evaluation
Day 5:  Hyperparameter tuning, ablation studies, write-up
```

This assumes a single engineer with a GPU. On CPU/MPS, double the training time.
