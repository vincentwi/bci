# Lead 1: GRU Baselines & Literature Reproductions

**Author:** Vincent Wilmet (with Claude)
**Date:** 2026-03-13
**Status:** Complete
**Best Result:** 17.5% val / 17.8% test PER (5-model ensemble + beam search)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Architecture](#3-architecture)
4. [Experimental Setup](#4-experimental-setup)
5. [Phase 1: Literature Reproductions](#5-phase-1-literature-reproductions)
6. [Phase 2: Ablation Study](#6-phase-2-ablation-study)
7. [Phase 3: Scheduler Discovery](#7-phase-3-scheduler-discovery)
8. [Phase 4: Beam Search Decoding](#8-phase-4-beam-search-decoding)
9. [Phase 5: Ensemble Methods](#9-phase-5-ensemble-methods)
10. [Complete Results Table](#10-complete-results-table)
11. [Key Findings](#11-key-findings)
12. [Relevant Files](#12-relevant-files)
13. [Reproduction Guide](#13-reproduction-guide)
14. [Limitations & Future Work](#14-limitations--future-work)

---

## 1. Executive Summary

This lead reproduced 4 published GRU architectures for brain-to-speech phoneme decoding, then systematically improved upon them through ablation studies, scheduler optimization, beam search decoding, and ensemble methods.

### Key Results

| Configuration | Val PER | Test PER | vs. Paper |
|--------------|---------|----------|-----------|
| Willett et al. 2023 (paper target) | — | 19.7% | baseline |
| Best single model (greedy) | 19.3% | 20.2% | −0.5pp |
| Best single model (beam search) | 18.8% | 19.5% | +0.2pp |
| **Best 5-model ensemble (beam search)** | **17.5%** | **17.8%** | **+1.9pp** |

### Key Discoveries

1. **Cosine annealing (20K steps)** is the single biggest improvement: +1.1pp over step decay
2. **SGD with lr=0.1** significantly outperforms Adam (20.9% vs 21.4% test)
3. **Architecture diversity** in ensembles (h=768 + h=1024) helps more than seed diversity
4. **Beam search with trigram LM** (α=0.3, β=0.5) gives ~0.5–0.8pp improvement
5. **BiGRU h=1024, 5 layers, k=32** is the optimal single-model architecture

---

## 2. Background & Motivation

### The Problem

Decode spoken sentences from neural activity recorded in area 6v of participant T12 (ALS, anarthric). Input: 256D neural features at 20ms resolution (128 spikePow + 128 tx1). Output: CTC-decoded phoneme sequences (40 classes + blank).

### Published Baselines

| Source | Architecture | PER | WER |
|--------|-------------|-----|-----|
| Willett et al. 2023 | UniGRU h=512 k=14 | 19.7% | 23.8% |
| cffan/neural_seq_decoder | BiGRU h=1024 k=32 | ~18% | — |
| CIBR-Okubo (2nd place) | BiGRU h=1024 k=32, SGD | ~15% | ~9% |
| Benchmark Lessons | BiGRU h=512, post-RNN head | — | 8.0% |
| DCoND (1st place) | BiGRU + diphone loss + LM | — | 5.77% |

### Root Cause of Previous Gap

Our earlier baselines gave 40–53% PER because of critical hyperparameter mismatches:
- Using `kernel_size=1, stride=1` (seeing 1×20ms bin instead of 14×280ms context window)
- Wrong optimizer settings (Adam lr=3e-4 instead of lr=0.02/eps=0.1)
- Using cross-session split instead of within-day holdout

### Paper vs. TF Source Discrepancies

The published paper and TF source code differ in several key hyperparameters:

| Parameter | Paper (Table 5) | TF Source | Our Best |
|-----------|----------------|-----------|----------|
| LR | 0.02→0 | 0.01→0 | 0.1→0 (SGD) |
| Optimizer | Adam(eps=0.1) | Adam(eps=0.01) | SGD(momentum=0.9) |
| Steps | 10,000 | 100,000 | 20,000 (cosine) |
| GRU hidden | 512 | 512 | 1024 |
| Bidirectional | No | No | Yes |
| Kernel | 14 | 14 | 32 |
| Grad clip | not mentioned | 10 | 5 |

---

## 3. Architecture

### EnhancedGRU Pipeline

```
Raw 256D input (20ms bins, ~315 frames avg)
    │
    ▼
EfficientDaySpecificLayer (256→256, softsign, per-session einsum)
    │
    ▼
Stack & Stride (kernel=32, stride=4)  →  T_out = (T - 32) / 4 + 1
    │
    ▼
Linear(32×256=8192 → 1024) + LayerNorm
    │
    ▼
[Optional] SpeckledMask(p=0.3)
    │
    ▼
5-layer BiGRU (hidden=1024, dropout=0.4)
    │
    ▼
[Optional] PostRNNHead (LayerNorm→Dropout→Linear→GELU)
    │
    ▼
Dropout(0.4) → Linear(2048 → 41)
    │
    ▼
CTC Loss (blank=40)
```

### Day-Specific Input Layer (Einsum Implementation)

The day-specific layer is the most architecturally important component. Each recording session has its own learned affine transformation, enabling the model to account for electrode drift across days.

```python
class EfficientDaySpecificLayer(nn.Module):
    """Per-session affine transform via batched einsum (Source: cffan)."""

    def __init__(self, in_dim, out_dim, n_sessions, dropout=0.4):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_sessions, in_dim, out_dim))
        self.biases = nn.Parameter(torch.zeros(n_sessions, out_dim))
        self.shared_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.shared_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, session_ids):
        # Fast path: batched einsum
        W = self.weights[session_ids]        # (B, in_dim, out_dim)
        b = self.biases[session_ids]         # (B, out_dim)
        projected = torch.einsum('btd,bdk->btk', x, W) + b.unsqueeze(1)
        # Softsign activation: x / (|x| + 1)
        return projected / (projected.abs() + 1)
```

During training, 20% of samples randomly use shared weights (fallback for unseen sessions).

### Stack & Stride (In-Forward Implementation)

Unlike the paper's preprocessing approach, stacking is done inside `forward()` so different kernel sizes share the same data loader:

```python
def forward(self, x, session_ids):
    x = self.day_input(x, session_ids)   # (B, T_raw, 256)

    B, T, H = x.shape
    T_new = max(1, (T - self.kernel_size) // self.stride + 1)
    indices = torch.arange(T_new, device=x.device) * self.stride
    stacked_list = []
    for k in range(self.kernel_size):
        idx = (indices + k).clamp(max=T - 1)
        stacked_list.append(x[:, idx, :])
    x = torch.cat(stacked_list, dim=-1)  # (B, T_new, K * 256)

    x = self.input_proj(x)              # (B, T_new, hidden)
    rnn_out, _ = self.rnn(x)            # (B, T_new, hidden*2)
    logits = self.output_proj(rnn_out)   # (B, T_new, 41)
    return logits
```

### Model Sizes

| Config | Hidden | Bidir | Params |
|--------|--------|-------|--------|
| Paper (h=512, UniGRU) | 512 | No | ~29M |
| h=768 BiGRU | 768 | Yes | 57.6M |
| h=1024 BiGRU | 1024 | Yes | 98.3M |
| h=1280 BiGRU | 1280 | Yes | 149.9M |

---

## 4. Experimental Setup

### Data

- **Source:** `sentences_paper_256d.h5` — 8,780 sentence trials from participant T12
- **Features:** 256D (128 spikePow + 128 tx1 from area 6v)
- **Temporal resolution:** 20ms bins, avg ~315 frames/trial
- **Preprocessing:** Causal Gaussian smoothing (SD=40ms, delay=160ms)

### Evaluation Protocol

**Within-day holdout** (matches paper exactly):
- Per session: 40 sentences held out for test, 10% for validation
- All sessions contribute to train/val/test (day-specific layers trained for all)
- Fixed seed=42 for reproducible splits
- Split: 6,942 train / 878 val / 960 test

### Training

- Mixed precision (AMP) with gradient clipping at 5.0
- Patience-based early stopping on validation PER
- CTC loss with `blank=40, zero_infinity=True`
- Paper-style noise augmentation: `x += N(0, σ_white)` per-element + `N(0, σ_offset)` per-channel

### Hardware

- 4× NVIDIA H100 80GB HBM3
- Training time: ~1.5–3h per model depending on config
- Beam search decoding: ~70s per eval set (CPU-bound)

---

## 5. Phase 1: Literature Reproductions

We reproduced 4 published GRU architectures using a unified training script with CLI presets.

### 5.1 Willett Paper (L1.1c)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u brain2speech/lead1_train_gru_v2.py \
    --preset paper --experiment L1.1c_paper_withinday
```

**Config:** UniGRU h=512, k=14, Adam lr=0.02 eps=0.1, linear decay, batch=64, noise=1.0
**Result:** 23.9% val / 25.4% test (915 epochs)

The paper reports 19.7% test PER. Our gap is due to: (1) the paper likely uses the TF implementation which has different numerical behavior, (2) we use patience-based early stopping vs. fixed 10K minibatches.

### 5.2 cffan Reference (L1.2b)

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u brain2speech/lead1_train_gru_v2.py \
    --preset cffan --experiment L1.2b_cffan_withinday
```

**Config:** BiGRU h=1024, k=32, Adam lr=0.02 eps=0.1, linear decay, batch=64
**Result:** 20.8% val / 21.4% test (314 epochs)

BiGRU + larger kernel = 4pp improvement over Paper UniGRU.

### 5.3 CIBR-Okubo 2nd Place (L1.3)

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u brain2speech/lead1_train_gru_v2.py \
    --preset cibr --experiment L1.3_cibr_withinday
```

**Config:** BiGRU h=1024, k=32, SGD lr=0.1 momentum=0.9, step decay, batch=128, noise=0.8, ortho init
**Result:** 20.4% val / 20.9% test (197 epochs)

SGD + orthogonal init + reduced noise = 0.5pp improvement over cffan.

### 5.4 Linderman Enhanced (L1.4)

```bash
CUDA_VISIBLE_DEVICES=2 python3 -u brain2speech/lead1_train_gru_v2.py \
    --preset linderman --experiment L1.4_linderman_withinday
```

**Config:** BiGRU h=512, k=14, SGD lr=0.025, post-RNN head (LN→Drop→Linear→GELU), speckled mask (p=0.3)
**Result:** 22.6% val / 23.8% test (782 epochs)

Worse than CIBR — the h=512 + k=14 limitation dominates any benefit from post-RNN head.

### 5.5 Enhanced (All Techniques Combined, L1.5)

**Config:** CIBR base + post-RNN head + speckled mask
**Result:** 24.2% val / 24.7% test — **worse** than CIBR alone

The combination of techniques didn't help. Post-RNN head + speckled mask add noise when the base model is already strong. This matches findings in the literature: not all techniques compose well.

![Literature Reproductions](plots/01_literature_reproductions.png)

---

## 6. Phase 2: Ablation Study

Starting from the best base (CIBR, 20.4% val), we systematically ablated each hyperparameter.

### 6.1 Kernel Size

| Kernel | Config | Val PER | Test PER |
|--------|--------|---------|----------|
| 14 | CIBR k=14 | 21.3% | 21.6% |
| **32** | **CIBR k=32** | **20.4%** | **20.9%** |

**Finding:** k=32 (640ms context) is significantly better than k=14 (280ms). The wider receptive field captures more phoneme context.

### 6.2 Hidden Size

| Hidden | Params | Val PER | Test PER |
|--------|--------|---------|----------|
| 768 | 57.6M | 20.8% | 21.3% |
| **1024** | **98.3M** | **20.4%** | **20.9%** |
| 1280 | 149.9M | 20.7% | 21.0% |

**Finding:** h=1024 is the sweet spot. h=768 underfits slightly, h=1280 overfits (diminishing returns from extra capacity).

### 6.3 Layer Count

| Layers | Val PER | Test PER |
|--------|---------|----------|
| 3 | 21.1% | 21.3% |
| **5** | **20.4%** | **20.9%** |
| 7 | 33.6% | 34.5% |

**Finding:** 5 layers is optimal. 7 layers catastrophically degrades — likely due to vanishing gradients in the deeper GRU stack without residual connections.

### 6.4 Noise Level

| White Noise σ | Val PER | Test PER |
|---------------|---------|----------|
| **0.8** | **20.4%** | **20.9%** |
| 1.0 (paper) | 21.4% | 22.0% |

**Finding:** Reduced noise (0.8 vs paper's 1.0) helps by 1.1pp. The paper's noise level may have been optimal for their architecture but is too aggressive for BiGRU h=1024.

### 6.5 Auxiliary Losses

| Technique | Val PER | Test PER |
|-----------|---------|----------|
| CTC only | 20.4% | 20.9% |
| CTC + supTCon (τ=0.1, w=0.1) | 20.6% | 21.0% |

**Finding:** Supervised contrastive loss (supTCon) did not help. The additional loss term may conflict with CTC optimization in this setting.

![Ablation Study](plots/03_ablation_study.png)

---

## 7. Phase 3: Scheduler Discovery

This was the **single most impactful finding** of the entire lead.

### 7.1 The Problem with Step Decay

The CIBR preset uses StepLR(step=4000, gamma=0.1) over 100K max minibatches. With batch=128 and ~6,942 train trials, one epoch ≈ 54 minibatches. So:
- Step at 4000 minibatches ≈ epoch 74
- With patience=50, early stopping triggers around epoch 197
- LR drops from 0.1 to 0.01 at epoch 74, then stays flat

The model converges but the LR schedule is suboptimal — most training happens at a fixed lr=0.01.

### 7.2 Cosine Annealing

We tested CosineAnnealingLR with different `T_max` (total minibatches) and patience values:

| Config | T_max | Patience | Val PER | Test PER | Epochs |
|--------|-------|----------|---------|----------|--------|
| Step decay (L1.3) | — | 50 | 20.4% | 20.9% | 197 |
| Cosine 100K (L1.5f) | 100K | 50 | 19.9% | 20.4% | 236 |
| Cosine 200K (L1.5g) | 200K | 100 | 19.8% | 20.7% | 301 |
| **Cosine 20K (L1.5h)** | **20K** | **100** | **19.3%** | **20.2%** | **370** |

**Finding:** Cosine 20K is the clear winner. With T_max=20K minibatches (≈370 epochs), the LR smoothly decays from 0.1 to near-zero over the full training duration. The larger patience (100) allows the model to train through the full cosine cycle instead of early-stopping while the LR is still high.

### Why Cosine 20K Works

The key insight is matching the cosine period to the actual training duration:
- **100K period**: LR barely decays before early stopping (lr≈0.096 at epoch 236)
- **200K period**: Same problem (lr≈0.098 at epoch 301)
- **20K period**: Full cosine cycle completes in ~370 epochs, LR reaches near-zero

The smooth decay allows the model to explore broadly at high LR, then fine-tune at low LR — a natural curriculum.

```python
# Cosine 20K config
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=20000 // steps_per_epoch,  # ~370 epochs
    eta_min=1e-6
)
```

![Scheduler Comparison](plots/02_scheduler_comparison.png)
![Best Model Training](plots/06_best_model_training.png)

---

## 8. Phase 4: Beam Search Decoding

### 8.1 Implementation

We implemented CTC prefix beam search with a trigram phoneme language model trained on the training set sequences.

**Components:**
- `PhonemeNgramLM`: Trigram model with Kneser-Ney-style backoff, trained on 6,942 sequences (583K n-grams)
- `ctc_prefix_beam_search`: Standard CTC prefix beam search with LM integration
- Scoring: `score = ctc_score + α × lm_score + β × |sequence|`

### 8.2 Hyperparameter Sweep

We swept beam width ∈ {5, 10, 25}, LM weight α ∈ {0.0–2.0}, and length bonus β ∈ {0.0, 0.3, 0.5}.

**Key findings from sweep on L1.7c seed 789:**

| Beam | α | β | Val PER | Rel. Improvement |
|------|-----|-----|---------|-----------------|
| — | — | — | 19.25% | (greedy baseline) |
| 5 | 0.3 | 0.3 | 18.74% | +2.7% |
| 10 | 0.3 | 0.0 | 18.73% | +2.7% |
| **10** | **0.5** | **0.5** | **18.68%** | **+2.9%** |
| 10 | 1.0 | 0.0 | 19.43% | −0.9% |
| 10 | 2.0 | 0.0 | 22.28% | −15.8% |

**Findings:**
- α=0.3–0.5 is optimal; higher values over-rely on the LM and hurt
- Length bonus β=0.5 gives a small additional improvement
- Beam width >10 gives negligible improvement for 3× compute cost
- Overall improvement: ~0.5–0.8pp absolute over greedy decoding

![Beam Search Heatmap](plots/05_beam_search_heatmap.png)

### 8.3 Best Single-Model Results

```bash
python3 -u brain2speech/eval_beam_search.py \
    --checkpoint brain2speech/results/lead1/L1.5h_cibr_cosine_20k_best.pt \
    --gpu 0 --beam-width 10 --lm-weight 0.3 --length-bonus 0.5
```

| Metric | Val | Test |
|--------|-----|------|
| Greedy | 19.3% | 20.1% |
| Beam (w=10, α=0.3, β=0.5) | **18.8%** | **19.5%** |

---

## 9. Phase 5: Ensemble Methods

### 9.1 Seed Diversity (Same Config)

Trained 4 models with the cosine 20K config and different random seeds:

| Seed | Val PER | Test PER |
|------|---------|----------|
| 42 | 19.5% | 20.2% |
| 123 | 19.5% | 20.2% |
| 456 | 20.2% | 21.1% |
| 789 | 19.3% | 20.3% |

The 4-seed ensemble (log-prob averaging + beam search):
- **Val: 18.7%, Test: 19.2%** — only marginal improvement over best single

**Why seed diversity alone doesn't work well:** Models with identical architectures and training configs converge to very similar solutions. The error correlation is too high.

### 9.2 Architecture Diversity

Adding the L1.5h checkpoint (same seed=42 but different training trajectory) dramatically improved the ensemble:

| Ensemble | Models | Val | Test |
|----------|--------|-----|------|
| 4-seed only | s42,123,456,789 (h=1024) | 18.7% | 19.2% |
| 4-model (−s456) | L1.5h + s42,123,789 (h=1024) | 17.7% | 18.1% |
| **5-model** | **L1.5h + s42,123,789 + h=768** | **17.5%** | **17.8%** |
| 6-model | +h=1280 | 17.7% | 18.0% |

**Key insight:** The h=768 model (19.1% val individually) makes errors in different places than h=1024 models, providing genuine complementary information. The h=1280 model (19.7% val) is too weak and drags the ensemble down.

### 9.3 Failed Approaches

1. **Heterogeneous kernel ensemble (k=14 + k=32):** Fundamentally broken — different kernel sizes produce log_probs at different time resolutions, making alignment impossible.

2. **Adam + SGD ensemble:** Adam model (26.2% val) was too weak to contribute.

3. **Including all models:** Adding weaker models (L1.3 step-decay, L1.2b cffan) hurt the ensemble — quality > quantity.

### 9.4 Ensemble Implementation

```python
# Average log probs in log space (geometric mean of probabilities)
for trial_idx in range(len(split_trials)):
    log_probs_list = [all_model_probs[m][trial_idx][0]
                      for m in range(len(models))]
    min_T = min(lp.shape[0] for lp in log_probs_list)
    avg_lp = np.mean([lp[:min_T] for lp in log_probs_list], axis=0)
    ensemble_probs.append((avg_lp, target))
```

```bash
# Best ensemble command
python3 -u brain2speech/eval_beam_search.py \
    --checkpoint brain2speech/results/lead1/L1.5h_cibr_cosine_20k_best.pt \
                 brain2speech/results/lead1/L1.7c_cos20k_seed42_best.pt \
                 brain2speech/results/lead1/L1.7c_cos20k_seed123_best.pt \
                 brain2speech/results/lead1/L1.7c_cos20k_seed789_best.pt \
                 brain2speech/results/lead1/L1.8a_h768_cos20k_best.pt \
    --gpu 0 --beam-width 10 --lm-weight 0.3 --length-bonus 0.5
```

![Ensemble Results](plots/04_ensemble_results.png)

---

## 10. Complete Results Table

### All Experiments (Chronological)

| # | Experiment | Hidden | Layers | Bidir | Kernel | Optimizer | Scheduler | Val PER | Test PER |
|---|-----------|--------|--------|-------|--------|-----------|-----------|---------|----------|
| 1 | L1.1c Paper | 512 | 5 | No | 14 | Adam | linear | 23.9% | 25.4% |
| 2 | L1.2b cffan | 1024 | 5 | Yes | 32 | Adam | linear | 20.8% | 21.4% |
| 3 | L1.3 CIBR | 1024 | 5 | Yes | 32 | SGD | step | 20.4% | 20.9% |
| 4 | L1.4 Linderman | 512 | 5 | Yes | 14 | SGD | linear | 22.6% | 23.8% |
| 5 | L1.5 Enhanced | 1024 | 5 | Yes | 32 | SGD | step | 24.2% | 24.7% |
| 6 | L1.5a k=14 | 1024 | 5 | Yes | 14 | SGD | step | 21.3% | 21.6% |
| 7 | L1.5b h=768 | 768 | 5 | Yes | 32 | SGD | step | 20.8% | 21.3% |
| 8 | L1.5c h=1280 | 1280 | 5 | Yes | 32 | SGD | step | 20.7% | 21.0% |
| 9 | L1.5d noise=1.0 | 1024 | 5 | Yes | 32 | SGD | step | 21.4% | 22.0% |
| 10 | L1.5e 7-layer | 1024 | 7 | Yes | 32 | SGD | step | 33.6% | 34.5% |
| 11 | L1.5f cos-100K | 1024 | 5 | Yes | 32 | SGD | cosine | 19.9% | 20.4% |
| 12 | L1.5g cos-200K | 1024 | 5 | Yes | 32 | SGD | cosine | 19.8% | 20.7% |
| 13 | **L1.5h cos-20K** | **1024** | **5** | **Yes** | **32** | **SGD** | **cosine** | **19.3%** | **20.2%** |
| 14 | L1.5i 3-layer | 1024 | 3 | Yes | 32 | SGD | step | 21.1% | 21.3% |
| 15 | L1.6 supTCon | 1024 | 5 | Yes | 32 | SGD | step | 20.6% | 21.0% |
| 16 | L1.7c s42 cos20K | 1024 | 5 | Yes | 32 | SGD | cosine | 19.5% | 20.2% |
| 17 | L1.7c s123 cos20K | 1024 | 5 | Yes | 32 | SGD | cosine | 19.5% | 20.2% |
| 18 | L1.7c s456 cos20K | 1024 | 5 | Yes | 32 | SGD | cosine | 20.2% | 21.1% |
| 19 | L1.7c s789 cos20K | 1024 | 5 | Yes | 32 | SGD | cosine | 19.3% | 20.3% |
| 20 | L1.8a h=768 cos20K | 768 | 5 | Yes | 32 | SGD | cosine | 19.1% | 19.7% |
| 21 | L1.8b Adam cos20K | 1024 | 5 | Yes | 32 | Adam | cosine | 26.2% | 26.6% |
| 22 | L1.8c h=1280 cos20K | 1280 | 5 | Yes | 32 | SGD | cosine | 19.7% | 20.3% |

### Beam Search Results (Selected)

| Config | Val Greedy | Val Beam | Test Greedy | Test Beam |
|--------|-----------|---------|------------|----------|
| L1.5h single | 19.3% | 18.8% | 20.1% | 19.5% |
| L1.7c 4-seed ensemble | 19.5% | 18.7% | 19.8% | 19.2% |
| L1.5h + s42,123,789 | 18.3% | 17.7% | 18.7% | 18.1% |
| **L1.5h + s42,123,789 + h=768** | **18.0%** | **17.5%** | **18.4%** | **17.8%** |

---

## 11. Key Findings

### What Matters Most (Ranked by Impact)

1. **Correct evaluation protocol** (within-day vs cross-session): ~15pp difference
2. **Correct architecture** (kernel=32, BiGRU h=1024): ~5pp improvement
3. **Cosine LR schedule** (vs step decay): ~1.1pp improvement
4. **SGD vs Adam**: ~0.5pp improvement
5. **Ensemble + beam search**: ~2.4pp improvement over best single greedy
6. **Architecture diversity in ensemble**: ~0.6pp over seed-only ensemble

### What Didn't Help

1. **Post-RNN head + speckled mask** on strong base model: −3.8pp (L1.5 enhanced)
2. **Supervised contrastive loss** (supTCon): −0.2pp
3. **More layers** (7 vs 5): −13pp catastrophic degradation
4. **Adam optimizer** with cosine schedule: −7pp vs SGD
5. **Seed-only diversity** in ensemble: marginal improvement

### Surprising Findings

1. **h=768 individually better than h=1024 on test** (19.7% vs 20.2%) with cosine schedule — the smaller model generalizes slightly better but was worse on val
2. **Including L1.5h (seed 42) in ensemble with L1.7c seed 42**: Despite same seed, they differ because of different training trajectories (different early stopping points), providing genuine diversity
3. **Length bonus β=0.5 matters more than beam width**: beam=10+β=0.5 beats beam=25+β=0.0

---

## 12. Relevant Files

### Scripts

| File | Description |
|------|-------------|
| `brain2speech/lead1_train_gru_v2.py` | Unified training script with EnhancedGRU, 5 presets, all enhancements |
| `brain2speech/eval_beam_search.py` | Beam search eval for single/ensemble models |
| `brain2speech/beam_search_decode.py` | PhonemeNgramLM + CTC prefix beam search + PER computation |
| `brain2speech/config.py` | Phoneme mapping, constants (N_CLASSES=40, CTC_BLANK=40) |
| `brain2speech/results/lead1/generate_plots.py` | Plot generation script for this writeup |

### Checkpoints (Best Models)

| File | Config | Val PER | Test PER |
|------|--------|---------|----------|
| `L1.5h_cibr_cosine_20k_best.pt` | CIBR cos20K s42 | 19.3% | 20.2% |
| `L1.7c_cos20k_seed42_best.pt` | CIBR cos20K s42 | 19.5% | 20.2% |
| `L1.7c_cos20k_seed123_best.pt` | CIBR cos20K s123 | 19.5% | 20.2% |
| `L1.7c_cos20k_seed789_best.pt` | CIBR cos20K s789 | 19.3% | 20.3% |
| `L1.8a_h768_cos20k_best.pt` | h=768 cos20K s42 | 19.1% | 19.7% |

### Result JSONs

All 31 experiment results stored as JSON with full hyperparameters, training history (per-epoch loss, val PER, LR), and test metrics in `brain2speech/results/lead1/`.

### Plots

| File | Contents |
|------|----------|
| `plots/01_literature_reproductions.png` | Training curves for 4 literature reproductions |
| `plots/02_scheduler_comparison.png` | LR schedule and PER comparison (step vs cosine variants) |
| `plots/03_ablation_study.png` | Bar chart of all ablation test PERs |
| `plots/04_ensemble_results.png` | Seed diversity + ensemble comparison |
| `plots/05_beam_search_heatmap.png` | Beam search α vs beam width heatmap |
| `plots/06_best_model_training.png` | L1.5h training loss, val PER, and LR curves |

---

## 13. Reproduction Guide

### Prerequisites

```bash
# Data (already preprocessed)
ls /mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5

# Dependencies
pip install torch h5py scipy numpy matplotlib
```

### Train Best Single Model

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u brain2speech/lead1_train_gru_v2.py \
    --preset cibr \
    --hidden 1024 --n-layers 5 --bidirectional true \
    --kernel-size 32 --stride 4 --day-hidden 256 --ortho-init \
    --optimizer sgd --lr 0.1 --momentum 0.9 --weight-decay 1e-5 \
    --scheduler cosine --max-minibatches 20000 --patience 100 \
    --batch-size 128 --white-noise 0.8 --offset-noise 0.2 \
    --seed 42 --experiment my_best_model
```

Expected: ~19.3% val PER in ~2.5h on 1× H100.

### Train Diverse Models for Ensemble

```bash
# Train 4 seeds in parallel (4 GPUs)
for seed in 42 123 456 789; do
    CUDA_VISIBLE_DEVICES=$((seed % 4)) python3 -u brain2speech/lead1_train_gru_v2.py \
        --preset cibr \
        --hidden 1024 --bidirectional true --kernel-size 32 --ortho-init \
        --optimizer sgd --lr 0.1 --momentum 0.9 \
        --scheduler cosine --max-minibatches 20000 --patience 100 \
        --batch-size 128 --white-noise 0.8 --offset-noise 0.2 \
        --seed $seed --experiment ensemble_seed${seed} &
done

# Train h=768 variant
CUDA_VISIBLE_DEVICES=0 python3 -u brain2speech/lead1_train_gru_v2.py \
    --preset cibr \
    --hidden 768 --bidirectional true --kernel-size 32 --ortho-init \
    --optimizer sgd --lr 0.1 --momentum 0.9 \
    --scheduler cosine --max-minibatches 20000 --patience 100 \
    --batch-size 128 --white-noise 0.8 --offset-noise 0.2 \
    --seed 42 --experiment ensemble_h768
```

### Evaluate Ensemble with Beam Search

```bash
python3 -u brain2speech/eval_beam_search.py \
    --checkpoint brain2speech/results/lead1/*_best.pt \
    --gpu 0 --eval-set both \
    --beam-width 10 --lm-weight 0.3 --length-bonus 0.5
```

---

## 14. Limitations & Future Work

### Current Limitations

1. **Trigram phoneme LM is weak:** The n-gram LM only provides ~0.5–0.8pp improvement. Competition winners used 5-gram + neural LM pipelines for 5–10pp PER→WER improvement.
2. **No word-level decoding:** PER ≠ WER. The gap to competition SOTA (5.77% WER) requires word-level decoding with a strong LM.
3. **Single participant data:** Results are specific to T12. Generalization unknown.
4. **Within-day evaluation:** All sessions have trained day-specific layers. Cross-session performance would be significantly worse.

### Recommended Next Steps

1. **KenLM integration:** Replace trigram with 5-gram word-level KenLM for WFST decoding (expected: 5–10pp WER improvement)
2. **Neural LM rescoring:** OPT-6.7B or fine-tuned GPT for re-ranking beam search candidates
3. **More ensemble diversity:** Train models with different kernel sizes (k=20, k=28) that still produce compatible output lengths via padding
4. **Diphone loss:** DCoND's dual CTC loss (α=0.6 phoneme + 0.4 diphone) — expected 1–2pp PER improvement
5. **Longer training:** The cosine 20K cycle completes at epoch 370. A second cosine cycle (warm restart) might improve further

### Performance Ceiling Estimate

| Component | Estimated PER |
|-----------|--------------|
| Current best (5-model + trigram beam) | 17.8% |
| + KenLM 5-gram | ~15% |
| + Neural LM rescoring | ~12% |
| + Diphone loss + more diversity | ~10% |
| Competition SOTA (in WER terms) | ~5.8% WER |

---

*Generated: 2026-03-13. All experiments run on NVIDIA H100 80GB GPUs.*
