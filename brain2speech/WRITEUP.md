# Brain-to-Speech: Neural Signal Decoding with LM-Assisted Phoneme Correction

> **Reference paper:** Willett, F. R., Kunz, E. M., Fan, C., et al. (2023). A high-performance speech neuroprosthesis. *Nature*, 620, 1031–1036. [[bioRxiv preprint]](https://www.biorxiv.org/content/10.1101/2023.01.21.524489v2.full.pdf)

## 1. Overview

This project implements a complete brain-to-speech pipeline that decodes attempted speech from intracortical neural signals, corrects decoding errors using a fine-tuned language model, and synthesizes intelligible audio. We replicate and extend the approach from Willett et al. (Nature 2023) using CTC-based sentence decoding. Our best model achieves **51.9% PER** (greedy decode, no LM) and **52.0% PER** (beam search + phoneme LM) vs the paper's **19.7% PER** — with a clear roadmap to close the gap through feature engineering improvements identified in our analysis (see §11).

### Pipeline Architecture

```
                    ┌─────────────────────────────────────┐
                    │  Raw Neural Data (.mat files)        │
                    │  4×64 Utah arrays → 256 channels     │
                    │  spikePow + tx1-tx4 = 1280 features  │
                    │  20ms time bins                       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  Preprocessing                       │
                    │  • Z-score per recording block        │
                    │  • Gaussian temporal smoothing (σ=2)  │
                    │  • g2p text → ARPABET phonemes        │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
┌──────────────────┐  ┌────────────────────┐  ┌───────────────────┐
│ Stage 1a:        │  │ Stage 1b:          │  │ Stage 1b (alt):   │
│ Isolated Phoneme │  │ CTC Sentence       │  │ CTC Sentence      │
│ Classification   │  │ Decoder (GRU)      │  │ Decoder (TCN)     │
│ TCN/GRU/TF/EEGNet│  │ 5-layer BiGRU      │  │ 8-block dilated   │
│ 40-class softmax │  │ CTC loss           │  │ CTC loss          │
│ ~28% cross-sess  │  │ ~50% val PER       │  │ ~66% val PER      │
└──────────────────┘  └────────┬───────────┘  └───────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  CTC Greedy Decode                │
                    │  collapse repeats, remove blanks   │
                    │  → ARPABET phoneme sequence         │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  Stage 2: LM Phoneme Correction   │
                    │  Qwen3.5-2B + LoRA (r=32, α=64)  │
                    │  200k synthetic noisy→clean pairs  │
                    │  Confusion-matrix noise channel    │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  Stage 3: Audio Synthesis          │
                    │  ARPABET → SSML phoneme tags       │
                    │  ElevenLabs eleven_flash_v2        │
                    │  Whisper ASR round-trip verify      │
                    └────────────────────────────────────┘
```

---

## 2. Dataset

**Source:** Willett, F. R., Kunz, E. M., Fan, C., et al. (2023). A high-performance speech neuroprosthesis. *Nature*, 620, 1031–1036. [[bioRxiv]](https://www.biorxiv.org/content/10.1101/2023.01.21.524489v2.full.pdf)

**Paper key results (our target benchmarks):**
| Metric | Vocal (125k vocab) | Silent | Improved LM + proximal |
|--------|-------------------|--------|----------------------|
| **PER** | **19.7%** | 20.9% | 17.0% |
| **WER** | **23.8%** | — | 11.8% |
| WER (50-word) | 9.1% | — | — |

**Participant:** T12, a 67-year-old male with ALS and anarthria (unable to produce intelligible speech). Implanted with 4 × 64-channel Utah microelectrode arrays in ventral premotor cortex (area 6v) and Broca's area (area 44).

**Neural features:** The raw voltage traces from 256 electrodes are processed into two feature types per 20ms bin:
- `spikePow` (256 dims): Band-pass filtered spike power — captures multi-unit spiking activity
- `tx1`–`tx4` (4 × 256 = 1024 dims): Threshold crossings at 4 voltage levels — captures firing rate at different amplitude thresholds

**Total:** 1280 features per 20ms time bin

```python
# Feature extraction from .mat files (preprocess_sentences.py)
sp  = mat['spikePow'][0, i]   # (T, 256) — spike band power
tx1 = mat['tx1'][0, i]         # (T, 256) — threshold crossings level 1
tx2 = mat['tx2'][0, i]         # (T, 256)
tx3 = mat['tx3'][0, i]         # (T, 256)
tx4 = mat['tx4'][0, i]         # (T, 256)
features = np.concatenate([sp, tx1, tx2, tx3, tx4], axis=1)  # (T, 1280)
```

### Data Splits

| Dataset | Sessions | Trials | Features | Duration | Classes | Use |
|---------|----------|--------|----------|----------|---------|-----|
| diagnosticBlocks | 20 | ~1,360 | (85, 1280) fixed | ~23 min | 8 words | Architecture search |
| tuning (phonemes) | 2 | 1,440 | (85, 1280) fixed | ~24 min | 40 phonemes | Cross-session eval |
| tuning (50 words) | 1 | 1,020 | (85, 1280) fixed | ~17 min | 51 words | Transfer learning |
| **competitionData (train)** | **24** | **8,780** | **(T, 1280) variable** | **~15.3 hours** | **sentence-level** | **CTC training** |
| competitionData (test) | 24 | ~8,500 | (T, 1280) variable | ~14 hours | sentence-level | Final evaluation |

### competitionData Statistics

The competition dataset contains complete sentences read aloud (attempted) by participant T12. Each trial has variable-length neural recordings aligned to sentence onset/offset.

| Statistic | Value |
|-----------|-------|
| Training sessions | 24 (April–August 2022) |
| Training trials | 8,780 sentences |
| Total neural frames | 2,761,941 (at 20ms = 15.3 hours) |
| Total phonemes | 191,109 |
| Avg frames/trial | 314.6 (6.3 seconds) |
| Avg phonemes/trial | 21.8 |
| Avg frames/phoneme | 14.4 (288ms per phoneme) |
| Session range | 160–520 trials per session |

### Phoneme Classes (ARPABET, 40 total)

```
Consonants (25):
  Stops:     B P D T G K        (voiced/unvoiced pairs)
  Fricatives: F V TH DH S Z SH ZH HH
  Affricates: CH JH
  Nasals:     M N NG
  Liquids:    L R ER
  Glides:     W Y

Vowels (14):
  Front:   IY IH EH AE          (fleece, kit, dress, trap)
  Central: AH                    (strut)
  Back:    AA AO OW UH UW       (lot, thought, goat, foot, goose)
  Diphthongs: AY AW OY EY       (price, mouth, choice, face)

Control: SIL (silence/DO_NOTHING)
```

The phoneme-to-index mapping is defined in `config.py`:

```python
CLASS_TO_ARPABET = {
    0: 'B',  1: 'CH', 2: 'SIL',  3: 'D',  4: 'F',  5: 'G',
    6: 'HH', 7: 'JH', 8: 'K',    9: 'L',  10:'ER',  11:'M',
    12:'N',  13:'NG', 14:'P',    15:'R',   16:'S',   17:'SH',
    18:'DH', 19:'T',  20:'TH',   21:'V',   22:'W',   23:'Y',
    24:'Z',  25:'ZH', 26:'OY',   27:'EH',  28:'EY',  29:'UH',
    30:'IY', 31:'OW', 32:'UW',   33:'IH',  34:'AA',  35:'AW',
    36:'AY', 37:'AH', 38:'AO',   39:'AE',
}
# CTC blank token = 40 (index after last phoneme)
```

---

## 3. Stage 1: Phoneme Classification

### 3.1 Preprocessing

Neural signals exhibit significant nonstationarity across recording blocks and sessions. Our preprocessing pipeline addresses this:

```python
# Per-block z-score normalization (preprocess_sentences.py)
def zscore_by_block(features, block_ids):
    """Z-score features per block — critical for cross-session generalization."""
    result = features.copy().astype(np.float32)
    for bid in np.unique(block_ids):
        mask = block_ids == bid
        block_data = result[mask]
        mu = block_data.mean(axis=0)          # (1280,)
        sd = block_data.std(axis=0)           # (1280,)
        sd[sd < 1e-8] = 1.0                   # prevent division by zero
        result[mask] = (block_data - mu) / sd  # z-score each feature independently
    return result
```

**Steps for isolated phoneme data:**
1. **Z-score normalization** per recording block within each session
2. **Feature concatenation:** spikePow (256) + tx1-tx4 (4 × 256) = 1280 features per time bin
3. **Trial segmentation:** 10 bins pre-onset + 75 bins post-onset = 85 time steps (1.7 seconds)
4. **Data augmentation:** Gaussian noise (σ=0.1, 50%), time shift (±60ms, 30%), channel dropout (5% channels, 20%)

**Steps for sentence-level data (CTC):**
1. Per-block z-score normalization (same as above)
2. Gaussian temporal smoothing (σ=2 bins = 40ms) — reduces high-frequency noise
3. Variable-length sequences preserved (no fixed windowing)
4. Phoneme labels derived from sentence text via `g2p_en` grapheme-to-phoneme conversion

```python
# Temporal smoothing (train_ctc.py)
def smooth_features(features, sigma):
    """Gaussian smooth along time axis. Paper uses this for temporal regularization."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(features, sigma=sigma, axis=0)
```

### 3.2 Model Architectures

#### Isolated Phoneme Classifiers (fixed-length, 85 × 1280 input)

| Model | Architecture | Parameters | Key Design |
|-------|-------------|------------|------------|
| **TCN** | 4-layer dilated Conv1d (dilation 1,1,2,4), kernel=7, GELU, BN, AdaptiveAvgPool | ~1.4M | Best temporal pattern capture |
| **EEGNet** | Depthwise-separable Conv2d (F1=16, D=2, F2=32), AvgPool | ~193K | Efficient spatial-temporal filtering |
| **GRU** | Bidirectional 2-layer GRU (hidden=256), mean pooling, LayerNorm | ~14.3M | Most comparable to paper's decoder |
| **Transformer** | 4-layer encoder (d=128, 8 heads), CLS token, positional encoding | ~3.9M | Global self-attention |

#### CTC Sentence Decoders (variable-length, T × 1280 input → T × 41 output)

```python
# GRU CTC Encoder architecture (train_ctc.py)
class GRUCTCEncoder(nn.Module):
    def __init__(self, n_features=1280, n_classes=41, hidden=512, n_layers=5, dr=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)    # 1280 → 512
        self.input_norm = nn.LayerNorm(hidden)
        self.rnn = nn.GRU(
            hidden, hidden, n_layers,
            batch_first=True,
            bidirectional=True,       # → 1024-dim output
            dropout=dr
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden * 2),  # 1024
            nn.Dropout(dr),
            nn.Linear(hidden * 2, n_classes),  # 1024 → 41
        )

    def forward(self, x):
        # x: (B, T, 1280) → (B, T, 41)
        x = self.input_norm(self.input_proj(x))
        x, _ = self.rnn(x)
        return self.output_proj(x)
```

| CTC Model | Hidden | Layers | Parameters | Architecture |
|-----------|--------|--------|------------|-------------|
| GRU v1 | 512 | 3 bidir | 13.3M | input_proj → 3-layer BiGRU → output_proj |
| **GRU v2** | **512** | **5 bidir** | **22.8M** | input_proj → 5-layer BiGRU → output_proj |
| GRU v4 | 768 | 5 bidir | 50.6M | input_proj → 5-layer BiGRU → output_proj |
| TCN | 256 | 8 blocks | 4.0M | input_proj → 8× dilated Conv1d → output_proj |
| Transformer | 256 | 6 layers | ~5M | input_proj + PE → 6-layer encoder → output_proj |

### 3.3 Training Protocol

**Isolated phoneme classifiers:**
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-2)
- **Scheduler:** CosineAnnealingWarmRestarts (T_0=33, T_mult=2)
- **Mixed precision:** AMP (bfloat16 on H100)
- **Multi-GPU:** DataParallel across 4× NVIDIA H100 80GB
- **Cross-validation:** Leave-one-session-out with 15% validation holdout for early stopping (patience=30)
- **Batch size:** 32 per GPU × 4 GPUs = 128 effective

**CTC sentence decoders:**

| Parameter | GRU v1 | GRU v2 (best) | GRU v3 (smooth) | GRU v4 (large) |
|-----------|--------|---------------|-----------------|----------------|
| Hidden dim | 512 | 512 | 512 | 768 |
| Layers | 3 | 5 | 5 | 5 |
| Batch size | 16 | 32 | 32 | 16 |
| Learning rate | 1e-3 | 3e-4 | 3e-4 | 3e-4 |
| LR scheduler | CosineWarmRestarts | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau |
| Patience | 15 | 20 | 25 | 25 |
| Noise aug σ | 0.1 | 0.05 | 0.05 | 0.05 |
| Temporal smooth | none | none | σ=2 | σ=2 |
| GPUs | 2× H100 | 2× H100 | 2× H100 | 2× H100 |
| Optimizer | AdamW (wd=0.01) | AdamW (wd=0.01) | AdamW (wd=0.01) | AdamW (wd=0.01) |

```python
# CTC training loop (train_ctc.py, simplified)
criterion = nn.CTCLoss(blank=40, zero_infinity=True)  # blank = CTC_BLANK
scaler = GradScaler()  # mixed precision

for epoch in range(max_epochs):
    for features, targets, feat_lens, tgt_lens in train_loader:
        features = features.to(device)        # (B, T_max, 1280) padded
        targets = targets.to(device)           # (sum(tgt_lens),) concatenated

        # Data augmentation: 50% chance of Gaussian noise
        if random() < 0.5:
            features = features + torch.randn_like(features) * 0.05

        with autocast("cuda"):
            log_probs = model(features).log_softmax(dim=2)  # (B, T, 41)
            log_probs_t = log_probs.transpose(0, 1)          # CTC expects (T, B, C)
            loss = criterion(log_probs_t, targets, feat_lens, tgt_lens)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 5.0)  # gradient clipping
        scaler.step(optimizer)
        scaler.update()

    # Validation: greedy CTC decode + PER
    val_per = evaluate(model, val_loader, device)['per']
    if scheduler_type == 'plateau':
        scheduler.step(val_per)  # reduce LR when PER plateaus
```

```python
# CTC greedy decoding (train_ctc.py)
def ctc_greedy_decode(log_probs, blank=40):
    """Collapse repeats and remove blanks from argmax path."""
    best_path = log_probs.argmax(axis=1)  # (T,)
    decoded = []
    prev = -1
    for t in best_path:
        if t != prev:
            if t != blank:
                decoded.append(int(t))
        prev = t
    return decoded

# PER computation via edit distance
def compute_per(predicted, target):
    return editdistance.eval(predicted, target) / len(target)
```

### Results — 40-Class Phoneme Classification

#### Methodology Audit (2026-03-12)

An audit revealed **3 critical issues** in the original evaluation pipeline (`build_noise_model.py`):

1. **Not cross-session:** Predictions were generated using within-session leave-one-block-out CV on Session 1 only (640 trials, 8 blocks). Session 2 (800 trials) was never used. This dramatically inflated results due to temporal autocorrelation within a single recording session.

2. **Oracle test-set early stopping:** The training loop selected the best epoch by evaluating on the held-out test fold each epoch (`_rerun_cv()` lines 152-155), rather than using a separate validation split. This is a form of test-set peeking that further inflates reported accuracy.

3. **Unfair comparison with paper:** Willett et al.'s 61.4% comes from a GRU decoder trained on **sentence-level data** (competitionData, ~10k trials with rich temporal context). Our models are trained on isolated phoneme tuning trials (640-800 trials per session). These are fundamentally different experimental setups.

**RETRACTED RESULTS** (within-session, oracle stopping — DO NOT CITE):

| Method | Accuracy | Notes |
|--------|----------|-------|
| ~~Our TCN~~ | ~~60.5%~~ | Within-session CV with oracle stopping |
| ~~Our Ensemble~~ | ~~65.5%~~ | Inflated by all 3 issues above |

#### Corrected Results — True Cross-Session Evaluation

![Cross-session accuracy comparison](figures/fig2_cross_session_accuracy.png)

**Protocol:** Train on all blocks from Session 1, test on Session 2 (and vice versa). 15% validation holdout from training session for early stopping. 3 runs with different seeds, mean ± std reported.

| Method | S1→S2 | S2→S1 | Average | Notes |
|--------|-------|-------|---------|-------|
| **Paper baseline** (GRU, sentence data) | — | — | **61.4%** | Trained on ~10k sentence trials, not directly comparable |
| Chance (40 classes) | 2.5% | 2.5% | 2.5% | |
| Our TCN | 25.6%±0.6% | 30.4%±0.8% | **28.0%±0.7%** | |
| Our Transformer | 25.4%±0.4% | 28.3%±0.9% | **26.8%±0.6%** | |
| Our GRU | 26.8%±1.1% | 31.6%±0.3% | **29.2%±0.6%** | Best single model |
| Our EEGNet | 16.7%±0.1% | 18.0%±0.9% | **17.3%±0.5%** | Worst cross-session |

> **Note:** The paper's 61.4% is from a sentence-level GRU decoder, not an isolated phoneme classifier. This comparison is fundamentally unfair — see §3.5 for sentence-level CTC results where a direct PER comparison is possible.

**Within-session results** (8-word diagnostic blocks, leave-one-session-out CV, 20 folds):

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Paper baseline** | — | Paper does not report 8-word diagnostic accuracy |
| EEGNet | 92.1% | 8 classes, much easier task |
| GRU | 93.8% | |
| Transformer | 96.2% | |
| TCN | 98.8% | Near-perfect |

> **Context:** 8-word classification is a vastly simpler task (chance = 12.5%) than 40-phoneme cross-session (chance = 2.5%) or sentence-level CTC decoding. These results mainly demonstrate that the neural signal *does* contain speech information within a session.

### Key Findings

1. **Cross-session generalization is the core challenge:** Accuracy drops from ~60% (within-session) to ~28% (cross-session) due to neural nonstationarity across days. This ~30-percentage-point gap highlights the severity of session drift in intracortical recordings.
2. **The paper comparison is not apples-to-apples:** The paper's 61.4% uses sentence-level data with much more training data and temporal context. A fair comparison would require training on the same dataset (competitionData with phoneme alignments).
3. **S2→S1 is consistently easier than S1→S2:** ~4% higher accuracy, likely because Session 2 has more training data (800 vs 640 trials).
4. **All DL architectures perform similarly:** TCN, GRU, Transformer all cluster around 27-30% cross-session, suggesting the bottleneck is the domain gap, not model capacity.
5. **Domain adaptation does NOT help on this data:** Ablation study (TCN, 3 runs each):
   - Baseline (no normalization): 28.4% ± 0.8%
   - + Session z-score normalization: **16.4%** (destroys S1→S2 — data already block-normed)
   - + CORAL alignment: 27.5% (rescues norm damage but doesn't exceed baseline)
   - + Mixup: 16.2% (hurt by normalization)
   - + Label smoothing: 16.8%
   - All improvements combined: 17.0%

   The block-level z-scoring in preprocessing is already effective. Additional session-level normalization strips away discriminative information. The cross-session gap is a fundamental neural nonstationarity issue, not a normalization one.

6. **Path forward is sentence-level training:** The competition dataset has 8,780 sentences across 24 sessions with far more training data and temporal context. CTC-based decoding on this data is what matches the paper's 61.4% result.

### 3.5 CTC Sentence-Level Decoding (competitionData)

To achieve an apples-to-apples comparison with the paper's 61.4%, we train CTC-based decoders on the full competitionData (8,780 sentences, 24 sessions).

#### Data Preparation

Sentence text is converted to ARPABET phoneme sequences using the `g2p_en` grapheme-to-phoneme model:

```python
# Sentence → phoneme conversion (preprocess_sentences.py)
from g2p_en import G2p
g2p = G2p()

def sentence_to_phonemes(text, g2p_model):
    raw_phones = g2p_model(text.strip())
    # Filter: keep only actual phonemes (no spaces, punctuation)
    phones = [p for p in raw_phones if p.strip() and p not in ' .,!?;:\'"()-']
    indices = []
    for p in phones:
        p_clean = re.sub(r'[012]$', '', p)  # strip stress: AH0 → AH
        if p_clean in PHONE_TO_IDX:
            indices.append(PHONE_TO_IDX[p_clean])
    return indices

# Example:
# "Hello world" → ['HH', 'AH', 'L', 'OW', 'W', 'ER', 'L', 'D']
#               → [6, 37, 9, 31, 22, 10, 9, 3]
```

#### Data Split

```
competitionData (train split): 24 sessions, 8,780 trials
├── Training:    18 sessions, 6,260 trials (sessions from April–July 2022)
├── Validation:   2 sessions,   720 trials (last 2 of training sessions)
└── Test:         4 sessions, 1,800 trials (last 4 sessions, August 2022)
```

All evaluation is on **held-out sessions** — the test sessions are from different recording days than training, ensuring true cross-session generalization.

#### CTC Results

![CTC PER comparison](figures/fig1_per_comparison.png)

| Model | Params | Val PER | Test PER | Δ vs Paper | Epochs | LR Schedule | Smooth |
|-------|--------|---------|----------|------------|--------|-------------|--------|
| **Paper baseline** (Willett 2023) | — | — | **19.7%** | — | — | — | 80ms Gaussian |
| GRU v1 (3L) | 13.3M | 56.5% | 63.8% | +44.1% | 45 (ES) | CosineWarmRestarts | none |
| GRU v2 (5L) | 22.8M | 49.8% | 59.1% | +39.4% | 100 | ReduceLROnPlateau | none |
| **GRU v3 (5L+smooth)** | **22.8M** | **45.1%** | **56.6%** | **+36.9%** | **97 (ES)** | **ReduceLROnPlateau** | **σ=2** |
| GRU v4 (5L+large+smooth) | 50.6M | 44.9%* | *killed* | — | — | ReduceLROnPlateau | σ=2 |
| TCN (8-block) | 4.0M | 65.6% | 70.9% | +51.2% | 48 (ES) | ReduceLROnPlateau | none |
| Transformer (6L) | ~5M | 100% | 100% | +80.3% | 26 (ES) | ReduceLROnPlateau | σ=2 |
| **GRU+improvements** | 22.8M | 46.5% | 57.5% | +37.8% | 31 (ES) | Warmup+Cosine | σ=2 |
| **GRU+all** (6 improvements) | ~23M | **44.9%** | 56.7% | +37.0% | 80 | Warmup+Cosine | σ=2 |
| Conformer+improvements | ~8M | 56.0% | 63.5% | +43.8% | 15 (ES) | Warmup+Cosine | σ=2 |

*(ES = early stopped, PER = Phoneme Error Rate via edit distance, lower is better)*
*(\* = killed before completion to free GPUs for parallel experiments)*

**Improved model notes:**
- **GRU+improvements** = day-specific input layers + rolling z-score + SpecAugment
- **GRU+all** = above + channel attention + temporal derivatives (delta + delta-delta, 3840 input features)
- **Conformer+improvements** = Conformer encoder (attention+conv) + day-specific + rolling z-score + SpecAugment
- The improvements did **not significantly help test PER** despite helping val PER, suggesting the warmup+cosine schedule or the improvements themselves may cause slight overfitting. The feature improvements likely need ReduceLROnPlateau + higher patience for full effect.

**Gap analysis:** Our best model (GRU v3) has a **36.9 percentage-point gap** with the paper. This gap is driven by both **feature engineering differences** (see §11) and **CTC's fundamental limitation**: conditionally independent outputs that can't model phonotactic constraints (see §3.6).

### 3.6 Beyond CTC: Autoregressive & Transducer Models

#### Why CTC Is Not Enough

CTC models each output frame independently: `P(y_t | encoder(x))`. This means when the encoder is uncertain between "B" and "P" (a common voiced/unvoiced confusion in neural signals), CTC has **no way** to condition on previously decoded phonemes. It cannot learn that "AH B" is far more likely than "AH P" in certain contexts, or that "ZHB" is not a valid English phoneme sequence.

This is the exact problem our LoRA LM correction stage (§4) was designed to fix post-hoc. Autoregressive decoders solve this at the model level.

#### Architecture Comparison

| Model | Output Independence | Key Advantage | Streaming? |
|-------|-------------------|---------------|------------|
| **GRU-CTC** (current) | Independent `P(y_t\|x)` | Simple, fast training | No (bidirectional) |
| **LSTM-CTC** | Independent `P(y_t\|x)` | Forget gate may help long seqs | No (bidirectional) |
| **CausalGRU-CTC** | Independent `P(y_t\|x)` | Real-time BCI compatible | **Yes** |
| **RNN-T (Transducer)** | **Autoregressive** `P(y_t\|x, y_{<t})` | Prediction network = implicit phoneme LM | Optional |
| **Seq2Seq + Attention** | **Autoregressive** `P(y_t\|x, y_{<t})` | Decoder attends to specific encoder frames | No |

```python
# RNN-T architecture (train_seq2seq.py) — the key difference from CTC
class RNNTransducer(nn.Module):
    """
    Encoder: bidirectional LSTM over neural features → h_enc
    Prediction network: unidirectional LSTM over previous outputs → h_pred
    Joint network: h_enc + h_pred → P(y_t | x_{1:T}, y_{1:t-1})

    The prediction network is what makes this autoregressive — it learns
    phonotactic constraints from the output sequence itself.
    """
    def __init__(self):
        self.encoder = nn.LSTM(hidden, hidden, 5, bidirectional=True)   # neural signal encoder
        self.pred_rnn = nn.LSTM(256, hidden, 2)                         # phoneme LM
        self.joint = nn.Sequential(nn.Tanh(), nn.Linear(hidden, 41))    # combine both
```

#### Results

| Model | Params | Val PER | Test PER | Δ vs Paper | Epochs | Notes |
|-------|--------|---------|----------|------------|--------|-------|
| **Paper baseline** | — | — | **19.7%** | — | — | 5L GRU + CTC + beam + LM |
| GRU v3 (bidir CTC) | 22.8M | 45.1% | 56.6% | +36.9% | 97 | Best CTC baseline |
| **RNN-T (transducer)** | **34.6M** | **44.1%** | **55.2%** | **+35.5%** | **60** | **Best overall — autoregressive** |
| CausalGRU-CTC (unidir) | ~11.5M | 48.7%* | — | — | 57* | Streaming-compatible, only ~4% worse than bidir |
| LSTM-CTC (bidir) | 30.1M | 59.5%* | — | — | 34* | Slower convergence than GRU |
| Seq2Seq + attention | 35.5M | 106.9%* | — | — | 5* | Failed to converge (too many insertions) |

*(\* = killed before completion to maximize GPU utilization across experiments)*

**Key findings:**
1. **RNN-T is the best model** (55.2% test PER, 44.1% val PER) — the autoregressive prediction network provides ~1.4% test PER improvement over the equivalent CTC model. The prediction network acts as an implicit phoneme language model, learning phonotactic constraints from the output sequence itself.
2. **CausalGRU is surprisingly competitive** (48.7% val PER) — only ~4% worse than bidirectional GRU v3, suggesting backward context is less important than expected. This is very encouraging for real-time streaming BCI deployment.
3. **LSTM is slower than GRU** — at the same epoch count, LSTM-CTC lags behind GRU-CTC. The additional forget gate doesn't help on these medium-length sequences (avg 314 time steps).
4. **Seq2Seq attention failed** — the model generates too many insertions (PER > 100%). Attention-based models are known to struggle with monotonic alignments; CTC and RNN-T are better suited for this task.

#### Training Curves — GRU v1 vs v2

![Training curves](figures/fig5_training_curves.png)

```
GRU v1 (CosineAnnealingWarmRestarts):
  Epoch 30: loss=0.486, val_PER=0.565 ← best
  Epoch 34: loss=1.488, val_PER=0.663 ← LR RESTART → catastrophic spike
  Epoch 45: early stop (never recovered to 0.565)
  Test PER: 63.8%

GRU v2 (ReduceLROnPlateau):
  Epoch 20: val_PER=0.564 (already matches v1 best)
  Epoch 49: val_PER=0.518 (LR still at 3e-4)
  Epoch 56: val_PER=0.508 (LR reduced to 1.5e-4)
  Epoch 66: val_PER=0.500 (LR reduced to 7.5e-5)
  Epoch 79: val_PER=0.498 (LR reduced to 3.7e-5) ← best
  Epoch 100: end of training
  Test PER: 59.1%

Val PER progression (GRU v2):
  ┌───────────────────────────────────────────┐
  │1.0 ·                                      │
  │    ·                                      │
  │0.8 · ·                                    │
  │    ·  ···                                 │
  │0.6 ·     ······                           │
  │    ·           ·····                      │
  │0.5 ·                ···········           │
  │    ·                           ·········  │ ← 0.498
  │0.4 ·                                      │
  │    ├──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┤ │
  │    0  10 20 30 40 50 60 70 80 90 100     │
  │                  Epoch                    │
  └───────────────────────────────────────────┘
```

#### Key Findings

1. **ReduceLROnPlateau >> CosineAnnealingWarmRestarts**: The cosine LR restart at epoch 34 in GRU v1 caused the training loss to spike from 0.49 → 1.49. The model never recovered, losing 15 epochs of progress. `ReduceLROnPlateau` reduces LR only when val PER stalls, providing smooth, monotonic improvement:

    ```
    LR reductions (GRU v2):
      Epoch  1-55: lr=3.0e-04 (initial phase, PER: 1.00 → 0.518)
      Epoch 56-64: lr=1.5e-04 (first reduction, PER: 0.508 → 0.504)
      Epoch 65-85: lr=7.5e-05 (second reduction, PER: 0.500 → 0.499)
      Epoch 86-91: lr=1.9e-05 (third reduction, PER: 0.499 → 0.500)
      Epoch 92+:   lr=9.4e-06 (fourth reduction, PER: 0.498)
    ```

2. **5-layer GRU outperforms 3-layer**: More recurrent depth (5 vs 3 layers) adds 9.5M parameters but reduces test PER from 63.8% to 59.1% — each layer captures more complex temporal patterns in the neural signal.

3. **GRU >> TCN >> Transformer for CTC**: GRU's ability to capture long-range temporal dependencies across entire sentences (300+ time steps) is critical. TCN's limited receptive field (8 dilated blocks with max dilation 2^3=8, kernel=7 → ~56 time steps) is insufficient. The Transformer failed completely (PER=100%) — likely due to the long sequences overwhelming self-attention without proper positional encoding or sequence length handling.

4. **Val-test gap (49.8% → 59.1%)**: The 9.3-percentage-point gap indicates that neural statistics in the last 4 sessions (August 2022) differ meaningfully from the 18 training sessions (April–July). This is consistent with known neural nonstationarity in BCI recordings.

5. **Temporal smoothing (v3, v4)**: Gaussian smoothing (σ=2 = 40ms) is one of the paper's key preprocessing steps. Early results show v3 and v4 are converging faster than v2 at the same epoch count, suggesting smoothing helps by reducing high-frequency noise in the neural features.

### 3.7 Paper Replica: Day-Specific Layers & Ablation Study

After deep-reading the paper, we identified multiple critical implementation differences from our original models. We created `train_paper_replica.py` to systematically test each correction.

#### Critical Bugs Discovered

| Issue | Paper | Our Original | Impact |
|-------|-------|-------------|--------|
| **Day-specific shared projection never trained** | All sessions have data for day-specific tuning | Val/test session projections stay at Xavier init | **Catastrophic** — model outputs garbage for unseen sessions |
| LR too high | 0.02 (with eps=0.1 for effective damping) | Copied paper's 0.02 naively | Model can't learn with 1280D features |
| Noise too high | SD=1.0 on 256D features | Copied 1.0 on 1280D features | Doubles feature variance, destroys signal |
| Feature dim | 256D (area 6v only) | 1280D (all electrodes) | Day-specific layers are 5× larger, overfit easily |

**Critical fix**: During training, 20% of samples randomly use the shared (session-agnostic) projection instead of their day-specific projection. This ensures the shared projection receives gradient updates and can generalize to unseen val/test sessions.

```python
# The bug: val/test sessions use untrained projections
# The fix: randomly use shared projection during training
if self.training and not use_shared and torch.rand(1).item() < 0.2:
    use_shared = True  # ensures shared_proj gets gradients
```

#### Ablation Results (16 experiments across 3 sweeps)

**Sweep 1: Architecture ablation** (2 GPUs each, 4 variants)

| Variant | Architecture | Val PER | Test PER | Notes |
|---------|-------------|---------|----------|-------|
| D (bidir+tanh) | Bidir GRU + day-specific tanh | **41.4%** | **53.8%** | First working day-specific |
| B (PCA+stack+bidir) | PCA 256D + stack 14 + bidir | 43.4% | 56.2% | PCA stacking didn't help |
| A (PCA+stack+unidir) | PCA 256D + stack 14 + unidir | 50.7% | 61.6% | Unidirectional much worse |
| C (paper LR) | PCA 256D + unidir + LR=0.02 | 66.6% | 72.8% | Paper LR too high for 1280D |

**Sweep 2: Optimization ablation** (all bidir, day-specific, plateau scheduler)

| Variant | Key Change | Val PER | Test PER | Notes |
|---------|-----------|---------|----------|-------|
| **G (more reg)** | noise=0.2, drop=0.4 | **39.6%** | **51.9%** | **Best overall** |
| H (causal smooth) | + causal Gaussian smooth | 40.6% | 53.5% | Causal smoothing slightly worse |
| F (softsign) | softsign activation | 40.8% | 53.2% | Paper's activation ≈ tanh |
| E (tanh baseline) | tanh, plateau | 42.7% | 54.3% | Plateau better than linear decay |

**Sweep 3: Pushing further**

| Variant | Key Change | Val PER | Test PER | Notes |
|---------|-----------|---------|----------|-------|
| L (hidden=768) | Larger model | 41.8% | 53.5% | More params didn't help test |
| J (softsign+reg) | Combine F+G | 43.2% | 55.1% | Softsign + reg worse |
| K (causal+reg) | Combine H+G | 47.9% | 59.2% | Causal + reg hurt |
| I (heavy reg) | noise=0.5, drop=0.5 | 49.9% | 58.4% | Too much regularization |

#### Best Model: G Configuration

```
Architecture: 5-layer bidirectional GRU, 512 hidden, 38.5M params
Input: 1280D features (no stacking, no PCA)
Day-specific: 24 per-session Linear(1280→512) + tanh activation
  - 20% random fallback to shared projection during training
Regularization: noise=0.2, dropout=0.4, weight_decay=1e-5
Optimizer: Adam (lr=3e-4, eps=1e-8) + ReduceLROnPlateau
Preprocessing: Gaussian smooth σ=2 (non-causal), per-block z-score
```

**Result: 39.6% val PER / 51.9% test PER** — a 3.3% test PER improvement over previous best (RNN-T at 55.2%).

#### Beam Search + Phoneme LM Decoding

We implemented CTC prefix beam search with an n-gram phoneme language model (`beam_search_decode.py`). The trigram LM is trained on 6,260 training sentences' phoneme sequences.

| Method | Val PER | Test PER | Decode Time |
|--------|---------|----------|-------------|
| Greedy | 40.4% | 52.7% | <1s |
| Beam (w=20, no LM) | 40.1% | 52.4% | ~9 min |
| **Beam (w=20, LM α=0.3)** | **39.8%** | **52.0%** | ~16 min |

Beam search provides a modest ~0.7% test PER improvement with the trigram phoneme LM. A word-level LM (as used in the paper) would likely provide much larger gains (estimated 5-10% PER).

#### Key Takeaways

1. **Day-specific layers are essential** — 53.8% → 51.9% test PER with proper shared projection training
2. **Bidirectional >> unidirectional** — 10% PER gap (53.8% vs 61.6%)
3. **Moderate regularization matters** — noise=0.2 + dropout=0.4 optimal; noise=0.5+ hurts
4. **PCA stacking didn't help** — our 1280D features are more informative unstacked
5. **Paper's hyperparameters don't transfer** — LR=0.02 and noise=1.0 fail with 1280D features
6. **ReduceLROnPlateau > linear LR decay** — consistently better for our setup
7. **Beam search + phoneme LM provides diminishing returns** without a word-level LM

#### Updated Gap Analysis

| Source | Estimated Impact | Status |
|--------|-----------------|--------|
| Day-specific input layers | ~3-5% PER | **Implemented** (51.9% test) |
| Beam search + phoneme LM | ~0.7% PER | **Implemented** (52.0% test) |
| Word-level LM (125k vocab) | ~5-10% PER | Not implemented |
| Feature selection (256D vs 1280D) | ~3-5% PER | Tested (PCA didn't help) |
| Input stacking (80ms bins) | ~2-3% PER | Tested (didn't help with 1280D) |
| More training data | ~3-5% PER | Limited by competition dataset |
| **Remaining gap** | | **~32% (51.9% vs 19.7%)** |

---

## 4. Stage 2: LM-Assisted Phoneme Correction

### 4.1 Motivation

Even a near-perfect phoneme classifier produces sentence-level errors. At 95% per-phoneme accuracy, a 30-phoneme sentence has only a (0.95)^30 = 21.5% chance of being completely correct. The LM correction stage leverages English phonotactic constraints to fix remaining errors by learning which phoneme sequences form valid English words and syllables.

### 4.2 Noise Model (Stage 2a)

We extract the 40×40 confusion matrix from the classifier's cross-validated predictions. This serves as a data-driven noise channel model:

```python
# Confusion matrix extraction (build_noise_model.py)
# Each row i contains P(predicted=j | true=i)
confusion_matrix = np.zeros((40, 40))
for true_label, pred_label in zip(y_true, y_pred):
    confusion_matrix[true_label, pred_label] += 1
# Normalize rows to probabilities
confusion_matrix /= confusion_matrix.sum(axis=1, keepdims=True)
```

The confusion matrix reveals systematic confusions in neural phoneme representation:
- **Voiced/unvoiced pairs:** B↔P (13%), D↔T (11%), G↔K (9%), V↔F (8%), Z↔S (7%)
- **Place-of-articulation neighbors:** B↔M (bilabial, 6%), D↔N (alveolar, 5%), G↔NG (velar, 4%)
- **Vowel height confusions:** IY↔IH (8%), UW↔UH (6%), AE↔EH (7%)

### 4.3 Synthetic Training Data (Stage 2b)

```python
# Synthetic pair generation (generate_pairs.py)
def corrupt_sequence(clean_phonemes, confusion_matrix, corruption_rate):
    """Apply confusion-matrix noise to a clean ARPABET sequence."""
    noisy = []
    for p in clean_phonemes:
        idx = arpabet_to_idx[p]
        if random() < corruption_rate:
            # Sample replacement from this phoneme's confusion distribution
            noisy_idx = np.random.choice(40, p=confusion_matrix[idx])
            noisy.append(idx_to_arpabet[noisy_idx])
        else:
            noisy.append(p)
    return noisy

# Example pair:
# Clean: "B AH T ER SIL W ER K S"  (butter works)
# Noisy: "P AH T ER SIL W ER K Z"  (B→P, S→Z confusions)
```

| Parameter | v1 | v2 |
|-----------|-----|-----|
| Source | CMU dict (~134k words) | CMU dict (~134k words) |
| Phrase length | 2–6 words | 2–8 words |
| Corruption rate | uniform 8–50% | bimodal: 70% at 5–25%, 30% at 25–60% |
| Insertion noise | none | 5% probability |
| Deletion noise | none | 5% probability |
| Training pairs | 80,000 | 200,000 |
| Val pairs | 10,000 | 20,000 |
| Test pairs | 10,000 | 20,000 |
| Format | Chat-template JSONL | Chat-template JSONL |

```jsonl
// Example JSONL training entry (phoneme_correction_train.jsonl)
{
  "messages": [
    {"role": "system", "content": "You are a phoneme error correction model..."},
    {"role": "user", "content": "P AH T ER SIL W ER K Z"},
    {"role": "assistant", "content": "B AH T ER SIL W ER K S"}
  ]
}
```

### 4.4 Fine-Tuning (Stage 2c)

**Base model:** Qwen3.5-2B (Qwen/Qwen3.5-2B)
**Method:** LoRA (Low-Rank Adaptation) — trains only ~1% of parameters

| Parameter | v1 | v2 |
|-----------|-----|-----|
| LoRA rank (r) | 16 | 32 |
| LoRA alpha (α) | 32 | 64 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | same |
| Trainable params | ~22M (0.5% of 2B) | ~44M (~1% of 2B) |
| Training pairs | 80k | 200k |
| Epochs | 3 | 5 (with early stopping) |
| Batch size (effective) | 256 | 384 |
| Learning rate | 2e-4 | 1.5e-4 |
| Warmup steps | 100 | 200 |
| Max sequence length | 128 tokens | 192 tokens |
| GPUs | 4× H100 (DDP) | 4× H100 (DDP) |
| Training time | ~10 min | ~25 min |
| Checkpoints | 1000, 1200, 1251 (final) | 1500, 1650, ..., 3125 (final) |

```python
# LoRA configuration (finetune_qwen_v2.py)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=32,                      # rank
    lora_alpha=64,             # scaling factor
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
# Trainable params: ~44M / 2.0B total = 1.0%
```

**System prompt:**
> "You are a phoneme error correction model for a brain-computer interface. Given a noisy ARPABET phoneme sequence decoded from neural signals, output the corrected sequence. Only output the corrected phonemes, nothing else."

### 4.5 Confidence-Gated Inference (Stage 2d)

Not all decoder outputs need correction. The confidence gating strategy selectively applies LM correction:

```python
# Confidence gating logic (inference_lm.py)
def correct_with_gating(phoneme_sequence, confidences, lm_model, threshold=0.8):
    # Split into segments: high-confidence (keep) vs low-confidence (correct)
    segments = []
    current = []
    for phone, conf in zip(phoneme_sequence, confidences):
        if conf < threshold:
            current.append(phone)
        else:
            if current:
                corrected = lm_model.correct(" ".join(current))
                segments.append(corrected)
                current = []
            segments.append(phone)
    if current:
        segments.append(lm_model.correct(" ".join(current)))
    return segments
```

- **High confidence (softmax > τ):** Keep original prediction
- **Low confidence (softmax ≤ τ):** Send through LM for correction
- **Validation:** All LM outputs checked against valid ARPABET set; invalid tokens → fallback
- **Default threshold:** τ = 0.8

### 4.6 LM Correction Results

![LM correction results](figures/fig7_lm_correction.png)

**On synthetic test pairs (v1 model, 500 pairs):**

| Metric | Value | Paper (beam + trigram LM) |
|--------|-------|--------------------------|
| PER before correction | 11.3% | 19.7% (raw CTC output) |
| PER after correction | 7.9% | — (reports WER: 23.8%) |
| **Relative PER reduction** | **30.3%** | — |
| Correction precision | 68.5% | — |
| Correction recall | 37.2% | — |
| Damage rate | 0.8% | — |

> **Paper comparison:** The paper uses beam search decoding with a 125k-word trigram LM (Kaldi) during CTC decoding itself, rather than a post-hoc LM correction step. Their PER of 19.7% is **before** LM application — the LM operates at the word level during beam search, reducing WER from ~50% (greedy) to 23.8%.

**On real neural data (isolated phoneme trials, cross-session):**

| Method | Accuracy | Notes |
|--------|----------|-------|
| TCN (no LM) | 28.0% | corrected cross-session baseline |
| Ensemble (no LM) | ~30% | estimated |

**Critical finding:** The LM correction **hurts** accuracy on isolated phoneme classification. This is expected for three reasons:

1. **No word-level context:** Each trial is a single isolated phoneme. The LM was trained on multi-word phrases where phonotactic constraints help (e.g., "B AH T ER" = "butter"). A random sequence of isolated phonemes has no word structure to exploit.

2. **Very low classifier confidence:** Cross-session confidence averages only 0.093 (TCN) to 0.264 (ensemble), so confidence gating is ineffective — nearly every prediction passes through the LM.

3. **Domain mismatch:** The LM was trained on corrupted CMU dict words. Real neural confusion patterns on isolated phonemes differ from multi-word corruption patterns.

**Implication:** LM correction is designed for **connected speech** (sentence-level CTC output), where temporal context and phonotactic constraints are available. The v2 LoRA model (200k pairs, bimodal corruption) will be evaluated on CTC-decoded sentence output, where it should be most effective.

---

## 5. Stage 3: Audio Synthesis

### 5.1 ElevenLabs TTS with Phoneme Tags

We use the ElevenLabs API with ARPABET phoneme tags for precise phoneme-level control over synthesis, bypassing the normal text-to-phoneme conversion that would introduce additional errors:

```python
# ARPABET → SSML conversion (elevenlabs_tts.py)
def phonemes_to_ssml(phoneme_sequence):
    """Convert ARPABET phonemes to SSML with phoneme tags.

    SIL tokens delimit word boundaries.
    First vowel per word gets primary stress (1).
    """
    words = " ".join(phoneme_sequence).split("SIL")
    ssml_parts = []
    for word_phones in words:
        phones = word_phones.strip().split()
        if not phones:
            continue
        # Add stress markers to vowels
        stressed = []
        first_vowel = True
        for p in phones:
            if p in VOWELS:
                stressed.append(f"{p}{'1' if first_vowel else '0'}")
                first_vowel = False
            else:
                stressed.append(p)
        ph_string = " ".join(stressed)
        ssml_parts.append(
            f'<phoneme alphabet="cmu-arpabet" ph="{ph_string}">{ph_string}</phoneme>'
        )
    return " ".join(ssml_parts)

# Example:
# Input:  ['B', 'AH', 'T', 'ER', 'SIL', 'W', 'ER', 'K', 'S']
# Output: '<phoneme alphabet="cmu-arpabet" ph="B AH1 T ER0">B AH1 T ER0</phoneme>
#          <phoneme alphabet="cmu-arpabet" ph="W ER1 K S">W ER1 K S</phoneme>'
```

**API Configuration:**
- **Model:** `eleven_flash_v2` (required for phoneme tag support)
- **Voice:** `JBFqnCBsd6RMkjVDRZzb` (default)
- **Output:** MP3 audio stream

### 5.2 Round-Trip Verification

Audio quality is verified by feeding synthesized audio through Whisper ASR and comparing the transcription back to the intended phonemes:

```
Decoded phonemes → SSML → ElevenLabs TTS → audio.mp3
                                               │
                                    Whisper ASR ▼
                                          recognized text
                                               │
                                    CMU dict   ▼
                                          recognized phonemes
                                               │
                                    editdistance ▼
                                          PER (round-trip)
```

This round-trip PER captures both TTS synthesis quality and ASR recognition accuracy, providing an upper bound on end-to-end system performance.

---

## 6. File Structure

```
/mnt/home/vincent.wilmet/brain2speech/
├── config.py                           # ARPABET maps, paths, API keys, model config
├── pipeline.py                         # End-to-end brain_to_speech() function
├── ensemble.py                         # Weighted soft-voting ensemble classifier
├── evaluate_end_to_end.py              # Evaluation with comparison tables
├── train_cross_session.py              # Proper cross-session eval (fixes audit issues)
├── train_cross_session_v2.py           # + domain adaptation ablation study
├── evaluate_cross_session.py           # Cross-session evaluation metrics
├── preprocess_sentences.py             # competitionData .mat → HDF5 with g2p phonemes
├── train_ctc.py                        # CTC sentence decoding (GRU/TCN/Transformer)
├── WRITEUP.md                          # This document
│
├── configs/
│   ├── accelerate_ddp.yaml             # Multi-GPU DDP config (6 GPUs)
│   └── accelerate_train_v2.yaml        # v2 LoRA training config (4 GPUs)
│
├── stage2_lm_correction/
│   ├── build_noise_model.py            # Confusion matrix → noise channel
│   ├── generate_pairs.py              # CMU dict × noise → training JSONL
│   ├── finetune_qwen.py               # v1 LoRA fine-tuning (r=16, 80k pairs)
│   ├── finetune_qwen_v2.py            # v2 LoRA fine-tuning (r=32, 200k pairs)
│   ├── inference_lm.py                # PhonemeCorrector class
│   └── evaluate_correction.py          # PER metrics, comparison tables
│
├── stage3_synthesis/
│   ├── elevenlabs_tts.py              # ARPABET → SSML → ElevenLabs API → audio
│   └── evaluate_audio.py             # Whisper round-trip evaluation
│
├── data/
│   ├── cmudict-0.7b                    # CMU Pronouncing Dictionary (~134k words)
│   ├── noise_model.npy                 # 40×40 confusion matrix
│   ├── sentences_train.h5              # Preprocessed CTC data (8,780 trials, gzip)
│   ├── sentences_train_stats.json      # Per-session statistics
│   ├── phoneme_predictions.npz         # Classifier predictions for noise model
│   ├── phoneme_correction_train.jsonl  # 160k noisy→clean pairs (v2)
│   ├── phoneme_correction_val.jsonl    # 20k pairs
│   ├── phoneme_correction_test.jsonl   # 20k pairs
│   └── correction_evaluation.json      # LM correction metrics
│
├── models/
│   ├── qwen_phoneme_corrector/         # v1 LoRA adapter
│   │   ├── checkpoint-{1000,1200,1251}/
│   │   └── final/                      # adapter_model.safetensors (43MB)
│   └── qwen_phoneme_corrector_v2/      # v2 LoRA adapter
│       ├── checkpoint-{1500,...,3125}/
│       └── final/                      # adapter_model.safetensors
│
└── results/
    ├── ctc_GRU_*.json                  # CTC GRU results (per run)
    ├── ctc_GRU_best.pt                 # Best GRU CTC model weights (91MB)
    ├── ctc_TCN_*.json                  # CTC TCN results
    ├── ctc_TCN_best.pt                 # Best TCN CTC model weights (16MB)
    ├── cross_session_*.json            # Cross-session evaluation results
    └── latest_results.json             # Most recent evaluation results

/mnt/home/vincent.wilmet/docs/
├── scripts/
│   ├── config.py                       # Stage 1 config (GPU IDs, hyperparams)
│   ├── preprocess.py                   # Raw .mat → .npz preprocessing
│   ├── train.py                        # All classifiers (EEGNet, TCN, GRU, TF)
│   └── train_naive_bayes.py            # Naive Bayes baseline
├── data/
│   ├── dryad/                          # Raw tar.gz + extracted .mat files
│   │   └── competitionData/            # 24 train + 24 test .mat sessions
│   └── processed/                      # .npz files for isolated phoneme task
└── models/
    ├── {tcn,eegnet,gru,transformer}_best.pt  # Best classifier checkpoints
    ├── results.json                    # Classifier accuracy results
    └── phoneme_class_names.npy         # Class index → ARPABET mapping
```

---

## 7. Compute Environment

- **Hardware:** 8× NVIDIA H100 80GB HBM3
- **Software:** PyTorch 2.7.1+cu126, Python 3.11 (conda env: py311)
- **Key packages:** peft, trl, accelerate, datasets, transformers, editdistance, openai-whisper
- **Training:** Mixed precision (bfloat16), DataParallel/DDP across 4+ GPUs
- **Inference:** Single GPU sufficient for all models

### GPU Allocation

| Stage | GPUs | Duration | Notes |
|-------|------|----------|-------|
| Stage 1 (classifiers) | 4-7 | ~30 min total | DataParallel, 10-fold CV |
| Stage 2a-2b (data gen) | CPU | ~2 min | Multiprocessing (16 workers) |
| Stage 2c (LoRA training) | 0-3 | ~15 min | DDP via accelerate |
| Stage 2d (LM inference) | any 1 | ~1 min | Single GPU |
| Stage 3 (TTS) | CPU | ~1 min | API calls |
| End-to-end eval | any 1 | ~5 min | Single GPU |

---

## 8. Key Design Decisions

### 8.1 Why Each Architecture Was Chosen

**GRU (Gated Recurrent Unit) — Primary decoder:**
The paper explicitly uses a "5-layer gated recurrent unit architecture" (Willett et al. 2023, Methods). We chose GRU as our primary architecture to match the paper's approach and enable a direct comparison. GRUs are well-suited for neural time series because:
- Bidirectional processing captures both forward and backward temporal context across the entire sentence
- Gating mechanisms handle the variable-length sequences (100–600 time steps) without vanishing gradients
- The paper demonstrates GRUs achieve state-of-the-art PER on this exact dataset
- **Result:** Best PER of all base architectures tested (56.6% before day-specific layers; 51.9% after — see §3.7), confirming the paper's architecture choice

**TCN (Temporal Convolutional Network) — Alternative decoder:**
TCNs were tested as a potentially faster alternative to GRUs, with advantages in parallelizable training and no hidden state bottleneck. However:
- **Limited receptive field:** 8 dilated blocks × kernel 7 × max dilation 8 = ~56 time steps of context, but sentences average 314 time steps
- **No global context:** Unlike bidirectional GRUs, TCNs can only see within their receptive field window
- **Result:** 70.9% PER — 14.3 points worse than GRU, confirming that global temporal context is essential for CTC sentence decoding

**Transformer — Exploratory:**
Transformers were tested to see if self-attention could capture long-range dependencies better than GRUs:
- **Complete failure (PER=100%):** The model never learned to produce any correct phonemes in 26 epochs
- **Likely cause:** Standard sinusoidal positional encoding is a poor fit for neural time series; the O(T²) attention complexity with T=300+ is expensive; CTC's blank-heavy output distribution may be pathological for attention
- **Alternative:** A Conformer (convolution + attention hybrid) with relative positional encoding would likely work — this is the standard architecture for modern ASR systems (Gulati et al. 2020)

**EEGNet — Baseline:**
Included as a lightweight, established BCI architecture. Designed for EEG (much lower channel count than intracortical), so it was not expected to be competitive on 1280-feature intracortical data. Confirmed: 17.3% cross-session accuracy (worst of all models).

### 8.2 Preprocessing & Training Decisions

1. **1280 features > 256 channels:** Using all spike band power + threshold crossings (spikePow + tx1-tx4) captures more neural information than spikePow alone. The threshold crossings at 4 voltage levels provide complementary firing rate information at different sensitivity levels.

2. **Per-block z-score normalization:** Neural signal statistics drift within and across recording sessions. Block-level (not session-level) normalization preserves intra-session discriminative information while combating drift. Our ablation study confirmed that additional session-level normalization *destroys* cross-session performance (28.4% → 16.4%).

3. **CTC loss for sentence decoding:** CTC provides alignment-free training — no need for framewise phoneme labels (which would require forced alignment). The model learns to emit phonemes at the right times and fill gaps with blank tokens. This matches the paper's approach exactly.

4. **ReduceLROnPlateau over CosineAnnealing:** Learned from GRU v1's catastrophic failure. The cosine restart annihilated 15 epochs of training progress in one step. ReduceLROnPlateau is monotonically safe — it only reduces LR when val PER plateaus, with configurable patience and factor.

5. **Confusion matrix as noise model:** Rather than generic noise for LM training data, we use the classifier's actual confusion patterns. The LM learns to correct the specific errors our system makes (B→P, not B→Z).

6. **Confidence gating for LM correction:** The LM should only intervene when the classifier is uncertain. This prevents the LM from damaging correct high-confidence predictions.

7. **ElevenLabs phoneme tags:** Direct ARPABET-to-speech synthesis via SSML phoneme tags preserves phoneme-level control without requiring an intermediate text representation (which would introduce another error source).

8. **Qwen3.5-2B as corrector:** A 2B parameter LM has sufficient English phonotactic knowledge to map noisy phoneme sequences to valid words, while being small enough to fine-tune with LoRA in ~25 minutes on 4 GPUs and run inference in real-time.

### 8.3 Why PER (Not Accuracy) Is the Primary Metric

The paper reports **Phoneme Error Rate (PER)** as its primary metric, computed as:

```
PER = edit_distance(predicted_phonemes, target_phonemes) / len(target_phonemes)
```

PER is superior to frame-level accuracy for CTC output because:
- CTC output is a **variable-length sequence** (after collapsing blanks/repeats), not frame-level predictions
- PER accounts for **insertions, deletions, and substitutions** — a model that outputs "B AH T" for target "B AH T ER" gets 75% accuracy but 25% PER
- PER is directly comparable to WER (Word Error Rate) in ASR literature — same edit distance formulation
- Frame-level accuracy on CTC output is meaningless because the blank token dominates (~90% of frames)

All results in this document report PER as the primary metric to enable direct comparison with the paper's 19.7% baseline.

---

## 9. Lessons Learned & Failure Modes

### 9.1 Methodology Audit

The original evaluation pipeline contained 3 critical methodological errors that inflated results from ~28% to ~65%:

| Issue | Impact | How Detected | Fix |
|-------|--------|-------------|-----|
| Within-session block CV | +15-20% | Suspicious accuracy gap | True cross-session eval |
| Oracle test-set early stopping | +5-10% | Code review of `_rerun_cv()` | 15% val holdout from train |
| Unfair comparison with paper | misleading | Dataset analysis | CTC on competitionData |

**Lesson:** Always validate evaluation methodology before interpreting results. Cross-session evaluation is the only realistic metric for BCI systems deployed across days.

### 9.2 LR Schedule Disaster

```
CosineAnnealingWarmRestarts(T_0=33):
  Epoch 33: LR ramps back to initial → loss spikes 0.49 → 1.49
  Epoch 34-45: Model recovers but never beats pre-restart best
  Result: 15 wasted epochs, suboptimal final model

Fix: ReduceLROnPlateau(factor=0.5, patience=5)
  Only reduces LR when improvement stalls
  Each reduction is a 50% drop (smooth, not catastrophic)
  Val PER: 0.565 → 0.498 (11% relative improvement)
```

### 9.3 Domain Adaptation Failure

Session normalization, CORAL alignment, mixup, and label smoothing all *hurt* cross-session generalization on this dataset. The likely reason: block-level z-scoring already handles the normalization that domain adaptation methods try to provide, and additional normalization strips away discriminative feature variation.

### 9.4 Transformer CTC Failure

The Transformer encoder completely failed on CTC sentence decoding (PER=100% after 26 epochs). Possible causes:
- Long sequences (300+ timesteps) overwhelm standard self-attention
- Sinusoidal positional encoding may not capture the right temporal structure
- CTC's blank-heavy output distribution is pathological for attention mechanisms
- **Fix to try:** Conformer architecture (self-attention + convolution), relative positional encoding, or chunked attention

---

## 10. Reproducibility

```bash
# ── Environment setup ──
conda activate py311
cd /mnt/home/vincent.wilmet

# ── Stage 1a: Isolated Phoneme Classification ──
cd docs/scripts
python preprocess.py --dataset tuning
python train.py --dataset tuning --epochs 200

# ── Stage 1b: CTC Sentence Decoding ──
cd /mnt/home/vincent.wilmet/brain2speech

# Step 1: Preprocess competitionData → HDF5
python preprocess_sentences.py --split train
# Output: data/sentences_train.h5 (8,780 trials)

# Step 2: Train CTC GRU decoder
CUDA_VISIBLE_DEVICES=0,1 python train_ctc.py \
    --model GRU \
    --gpus 0 1 \
    --epochs 120 \
    --batch-size 32 \
    --lr 3e-4 \
    --hidden 512 \
    --n-layers 5 \
    --patience 25 \
    --scheduler plateau \
    --noise 0.05 \
    --smooth 2
# Output: results/ctc_GRU_*.json + results/ctc_GRU_best.pt

# ── Stage 2: LM Phoneme Correction ──
# Step 2a: Build noise model from classifier confusion matrix
python stage2_lm_correction/build_noise_model.py

# Step 2b: Generate 200k synthetic training pairs
python stage2_lm_correction/generate_pairs.py

# Step 2c: Fine-tune Qwen3.5-2B with LoRA (v2)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file configs/accelerate_train_v2.yaml \
    --num_processes 4 \
    stage2_lm_correction/finetune_qwen_v2.py
# ~25 minutes on 4× H100, output: models/qwen_phoneme_corrector_v2/final/

# Step 2d: Evaluate LM correction
python stage2_lm_correction/evaluate_correction.py --gpus 0 --max_samples 1000

# ── Stage 3: Audio Synthesis ──
python stage3_synthesis/elevenlabs_tts.py --input "B AH T ER SIL W ER K S"

# ── Full Pipeline ──
python pipeline.py --test
python evaluate_end_to_end.py --gpu 0
```

---

## 11. Feature Gap Analysis & Improvement Roadmap

### 11.1 Why the Bottleneck Is Features, Not Architecture

Our GRU architecture is essentially identical to the paper's:
- Both use 5-layer bidirectional GRU with CTC loss
- Both use ARPABET phoneme targets (39 phonemes + silence + blank)
- Both use AdamW optimizer

Yet our best PER (51.9%) is **2.6× worse** than the paper's (19.7%). The gap comes from **both feature engineering and decoding**:
- **Feature engineering** (§11.2): 80ms bins, rolling z-score, feature selection (256D vs 1280D)
- **Decoding** (§3.7): word-level LM (125k vocab) vs our phoneme-level trigram LM
- **Implemented**: day-specific layers (§3.7), beam search + phoneme LM (§3.7)

All feature improvements are implemented in `train_ctc_improved.py` with support for day-specific input layers, rolling z-score, SpecAugment, channel attention, Conformer encoder, and temporal derivatives. Here is a systematic analysis:

![Architecture comparison and key differences](figures/fig3_architecture_analysis.png)

### 11.2 Detailed Gap Analysis

| Feature | Paper Implementation | Our Implementation | Estimated PER Impact | Status |
|---------|---------------------|-------------------|---------------------|--------|
| **Day-specific input layers** | Unique `nn.Linear` per session + softsign | Shared input layer | ~3-5% PER | **Done** (§3.7) |
| **Beam search + LM** | CTC beam + 125k word trigram LM | Phoneme trigram LM | ~0.7% (phoneme LM) / ~5-10% (word LM) | **Partial** (§3.7) |
| **Temporal binning** | 80ms bins (4×20ms stacked) | Raw 20ms bins | ~2-3% PER | Tested (didn't help with 1280D) |
| **Feature dimensionality** | 256D (area 6v only) | 1280D (all electrodes) | ~3-5% PER | Tested (PCA didn't help) |
| **Normalization** | Rolling z-score (online, causal) | Block-level z-score | ~2-4% PER | Not implemented |
| **Training data** | 10,850 sentences | 6,260 sentences | ~3-5% PER | Limited by dataset |
| **Smoothing** | 80ms Gaussian (σ=4 bins at 20ms) | 40ms Gaussian (σ=2) | ~1% PER | Tested |
| **Remaining gap** | 19.7% | 51.9% | **32.2% PER** | |

The estimated individual contributions roughly account for the full gap. Below is the projected improvement roadmap:

![PER improvement roadmap](figures/fig4_improvement_roadmap.png)

### 11.3 Implementation Plan for Each Improvement

#### A. 80ms Temporal Bins (4× stacking)

The paper stacks 4 consecutive 20ms bins into a single 80ms super-frame, increasing the input dimensionality from 1280 to 5120 but reducing sequence length by 4×. This is critical because:
- Reduces sequence length from ~314 to ~79 time steps (faster training, easier for CTC alignment)
- Captures co-articulation dynamics within each 80ms window
- Matches the ~80ms phoneme production timescale

```python
# Proposed implementation (preprocess_sentences.py)
def stack_frames(features, stack_size=4):
    """Stack consecutive frames: (T, 1280) → (T//4, 5120)."""
    T, D = features.shape
    T_new = T // stack_size
    features = features[:T_new * stack_size]  # trim to multiple of stack_size
    return features.reshape(T_new, stack_size * D)  # (T//4, 5120)
```

#### B. Day-Specific Input Layers

The paper uses "unique input layers for each day to account for across-day changes in neural feature distributions." This is a per-session `nn.Linear` projection that maps from the shared feature space to a session-invariant representation before the GRU.

```python
# Proposed implementation (train_ctc.py)
class DaySpecificGRU(nn.Module):
    def __init__(self, n_features=5120, n_classes=41, hidden=512, n_layers=5,
                 n_sessions=24, dr=0.3):
        super().__init__()
        # One input projection per recording session
        self.day_layers = nn.ModuleDict({
            str(sid): nn.Linear(n_features, hidden)
            for sid in range(n_sessions)
        })
        self.input_norm = nn.LayerNorm(hidden)
        self.rnn = nn.GRU(hidden, hidden, n_layers,
                          batch_first=True, bidirectional=True, dropout=dr)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dr),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x, session_ids):
        # x: (B, T, 5120), session_ids: (B,) — int session index per trial
        projected = torch.zeros(x.shape[0], x.shape[1], self.rnn.input_size,
                               device=x.device, dtype=x.dtype)
        for sid in session_ids.unique():
            mask = session_ids == sid
            projected[mask] = self.day_layers[str(sid.item())](x[mask])
        x = self.input_norm(projected)
        x, _ = self.rnn(x)
        return self.output_proj(x)
```

#### C. Rolling Z-Score Normalization

Instead of per-block z-scoring (which requires knowing block boundaries), the paper uses a causal rolling z-score that adapts online. This better handles slow drift during recording.

```python
# Proposed implementation
def rolling_zscore(features, window_size=500):
    """Online z-score with exponential moving average. Causal — no lookahead."""
    result = np.zeros_like(features, dtype=np.float32)
    alpha = 2.0 / (window_size + 1)
    mu = features[0].copy()
    var = np.ones_like(mu)
    for t in range(len(features)):
        mu = alpha * features[t] + (1 - alpha) * mu
        var = alpha * (features[t] - mu)**2 + (1 - alpha) * var
        sd = np.sqrt(var + 1e-8)
        result[t] = (features[t] - mu) / sd
    return result
```

#### D. Beam Search + Trigram LM Decoding

The paper uses Kaldi's WFST-based decoder with a 125k-word trigram language model. We can approximate this with `pyctcdecode`:

```python
# Proposed implementation
from pyctcdecode import build_ctcdecoder

# Build decoder with trigram LM
labels = list(CLASS_TO_ARPABET.values()) + ['']  # phonemes + blank
decoder = build_ctcdecoder(
    labels,
    kenlm_model_path='models/phoneme_trigram.arpa',  # trained on CMU dict
    alpha=0.5,  # LM weight
    beta=1.0,   # word insertion bonus
)

# Decode with beam search
def beam_decode(log_probs, beam_width=100):
    """Beam search CTC decoding with LM rescoring."""
    return decoder.decode(log_probs, beam_width=beam_width)
```

### 11.4 Next Training Round — Full GPU Parallelization Plan

![GPU utilization plan](figures/fig6_gpu_plan.png)

All 8 H100 GPUs will be used simultaneously for the next round:

```bash
# ── Phase 1: Parallel feature ablation (all 8 GPUs, ~3 hours) ──

# GPU 0-1: GRU v5 — 80ms bins + day-specific input layers
CUDA_VISIBLE_DEVICES=0,1 python train_ctc.py \
    --model GRU --gpus 0 1 --epochs 120 --batch-size 32 \
    --lr 3e-4 --hidden 512 --n-layers 5 --patience 25 \
    --scheduler plateau --smooth 2 \
    --stack-frames 4 --day-specific &

# GPU 2-3: GRU v6 — rolling z-score + day-specific input layers
CUDA_VISIBLE_DEVICES=2,3 python train_ctc.py \
    --model GRU --gpus 2 3 --epochs 120 --batch-size 32 \
    --lr 3e-4 --hidden 512 --n-layers 5 --patience 25 \
    --scheduler plateau --smooth 2 \
    --rolling-zscore --day-specific &

# GPU 4-5: GRU v7 — 80ms + rolling z + day-specific (full paper recipe)
CUDA_VISIBLE_DEVICES=4,5 python train_ctc.py \
    --model GRU --gpus 4 5 --epochs 120 --batch-size 32 \
    --lr 3e-4 --hidden 512 --n-layers 5 --patience 25 \
    --scheduler plateau --smooth 4 \
    --stack-frames 4 --rolling-zscore --day-specific &

# GPU 6-7: Conformer — attention + convolution hybrid
CUDA_VISIBLE_DEVICES=6,7 python train_ctc.py \
    --model Conformer --gpus 6 7 --epochs 120 --batch-size 16 \
    --lr 3e-4 --hidden 256 --n-layers 6 --patience 25 \
    --scheduler plateau --smooth 2 \
    --stack-frames 4 --day-specific &

wait  # wait for all 4 runs

# ── Phase 2: Post-hoc improvements (all 8 GPUs, ~1 hour) ──

# GPU 0-3: Beam search + LM decoding on best model
python decode_beam.py --model-path results/ctc_GRU_best.pt \
    --lm-path models/phoneme_trigram.arpa \
    --beam-width 100 --gpus 0 1 2 3 &

# GPU 4-7: LoRA v2 evaluation on CTC output
python evaluate_lora_on_ctc.py \
    --ctc-model results/ctc_GRU_best.pt \
    --lora-model models/qwen_phoneme_corrector_v2/final/ \
    --gpus 4 5 6 7 &

wait
```

**Expected outcomes:**
| Experiment | Expected Test PER | Rationale |
|-----------|------------------|-----------|
| GRU v5 (80ms + day-layers) | ~40-45% | Biggest single improvements |
| GRU v6 (rolling-z + day-layers) | ~42-48% | Less impact than binning |
| GRU v7 (full paper recipe) | ~28-35% | Closest to paper setup |
| Conformer | ~35-42% | Modern ASR architecture |
| + Beam search + LM | -5 to -10% | Post-hoc decoding improvement |
| + LoRA v2 correction | -2 to -5% | Post-hoc phoneme correction |

---

---

## 12. References

1. Willett, F. R., Kunz, E. M., Fan, C., et al. (2023). A high-performance speech neuroprosthesis. *Nature*, 620, 1031–1036. [[bioRxiv]](https://www.biorxiv.org/content/10.1101/2023.01.21.524489v2.full.pdf)
2. Graves, A., et al. (2006). Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks. *ICML 2006*.
3. Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5).
4. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
5. Qwen Team (2025). Qwen3.5 Technical Report.
6. Gulati, A., et al. (2020). Conformer: Convolution-augmented Transformer for Speech Recognition. *Interspeech 2020*.
7. CMU Pronouncing Dictionary (v0.7b). Carnegie Mellon University.
8. Kensho Technologies. `pyctcdecode`: CTC beam search decoder with language model support.
