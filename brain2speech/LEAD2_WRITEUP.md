# Lead 2: DCoND Diphone System — Complete Writeup

## Summary

Lead 2 reproduces the DCoND (Divide-and-Conquer Neural Decoding) diphone system from the 1st-place Brain-to-Text 2024 competition entry (arxiv:2411.10657). The system uses a 1601-class diphone CTC decoder with marginalized monophone loss, followed by a phoneme-to-text LLM (LIFT) for final transcription.

**Best Results:**
| Metric | Val | Test |
|--------|-----|------|
| PER (seed45 greedy) | 18.7% | 20.9% |
| WER (LIFT v1) | 28.1% | 34.6% |
| WER (4-seed majority vote) | 27.7% | — |
| WER (oracle 4-seed) | 22.6% | — |

## Architecture

### DCoND Neural Decoder (PaperExactDCoND)
- **Architecture:** Unidirectional GRU, 5 layers, hidden=512, kernel_size=14, stride=4
- **Output:** 1601 classes (1600 diphones + 1 CTC blank)
- **Loss:** α·L_diphone + (1-α)·L_monophone, α=0.6
- **Optimizer:** Adam (lr=0.02, eps=0.1) — SGD failed for this architecture
- **Day-specific input layer:** 256D → per-session affine → 256D
- **Marginalization:** 1601→41 via M_ext matrix for monophone CTC loss and inference

### Phoneme-to-Text LIFT (v1)
- **Base model:** Qwen3.5-2B
- **Fine-tuning:** LoRA r=32, alpha=64, 3 epochs on 6,226 examples
- **Training data:** Decoded phonemes from seed45 DCoND + ground truth text
- **Prompt:** `Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:`

### DCoND-LIFT Corrector
- **Base model:** Qwen3.5-2B
- **Fine-tuning:** LoRA r=32, alpha=64, 3 epochs on ~6,200 (LIFT_output + phonemes → GT) pairs
- **Prompt:** `Correct this brain-computer interface transcription using both the decoded text and raw phonemes.\n\nDecoded text: {text}\nPhonemes: {phones}\n\nCorrected text:`
- **Status:** Trained, evaluation pending

### LIFT v2 (in progress)
- **Base model:** Qwen3.5-2B
- **Fine-tuning:** LoRA r=64, alpha=128, 5 epochs on 31,300 examples (all 4 seeds)
- **Multi-GPU:** DDP across 4 GPUs
- **Status:** Training at ~44% completion

## File Inventory

### Training Scripts
| File | Purpose | Lines |
|------|---------|-------|
| `brain2speech/lead2_train_dcond.py` | DCoND diphone decoder training | ~1100 |
| `brain2speech/lead2_train_lift.py` | LIFT v1 phoneme→text fine-tuning | ~130 |
| `brain2speech/lead2_train_lift_v2.py` | LIFT v2 training (31K examples) | ~160 |
| `brain2speech/lead2_train_lift_v2_multigpu.py` | LIFT v2 multi-GPU DDP version | ~100 |
| `brain2speech/lead2_train_dcond_lift_corrector.py` | DCoND-LIFT corrector training | ~200 |
| `brain2speech/lead2_build_kenlm.py` | KenLM 5-gram phoneme LM builder | ~70 |

### Evaluation & Ablation Scripts
| File | Purpose |
|------|---------|
| `brain2speech/lead2_dcond_lift.py` | Full end-to-end pipeline |
| `brain2speech/lead2_eval_corrector.py` | Evaluate LIFT v1 vs corrector |
| `brain2speech/lead2_eval_lift_v2.py` | Evaluate LIFT v1 vs v2 |
| `brain2speech/lead2_ensemble.py` | Multi-seed DCoND ensemble decode |
| `brain2speech/lead2_decode_kenlm.py` | KenLM beam search + LIFT comparison |
| `brain2speech/lead2_beam_search_lift_v2.py` | pyctcdecode beam + N-best LIFT |
| `brain2speech/lead2_multi_seed_lift.py` | Multi-seed LIFT + majority vote |
| `brain2speech/lead2_lift_sampling.py` | Temperature sampling + oracle/majority |
| `brain2speech/lead2_lift_rerank_logprob.py` | Log-probability reranking |
| `brain2speech/lead2_lift_prompt_ablation.py` | 5 prompt variants comparison |
| `brain2speech/lead2_qwen_selector.py` | Base Qwen perplexity selection |
| `brain2speech/lead2_lift_multi_input.py` | Multi-input prompt (4 seeds in 1) |
| `brain2speech/lead2_second_stage_correction.py` | Base model correction (failed) |
| `brain2speech/lead2_rescore_opt.py` | OPT-6.7B rescoring (not used) |

### Result Files
| File | Contents |
|------|----------|
| `brain2speech/results/L2_pipeline_summary.json` | Complete results summary |
| `brain2speech/results/L2_phoneme_lift_test_val_results.json` | LIFT v1 test+val WER |
| `brain2speech/results/L2_sampling_lift_comparison.json` | Sampling experiments |
| `brain2speech/results/L2_lift_logprob_rerank.json` | Log-prob reranking results |
| `brain2speech/results/L2_ensemble_decode_comparison.json` | Ensemble decode comparison |
| `brain2speech/results/L2_multi_seed_lift.json` | Multi-seed majority vote |
| `brain2speech/results/L2_qwen_selector.json` | Perplexity selector results |
| `brain2speech/results/L2_multi_input_lift.json` | Multi-input prompt results |
| `brain2speech/results/L2_kenlm_beam_lift_comparison.json` | KenLM beam + LIFT |

## Model Weights

### DCoND Checkpoints (58MB each)
| File | Val PER | Notes |
|------|---------|-------|
| `results/L2_L2_pe_dcond_adam_s42_long_best.pt` | 19.2% | Seed 42 |
| `results/L2_L2_pe_dcond_adam_s43_best.pt` | 18.8% | Seed 43 |
| `results/L2_L2_pe_dcond_adam_s44_best.pt` | 19.1% | Seed 44 |
| `results/L2_L2_pe_dcond_adam_s45_best.pt` | 18.7% | **BEST** Seed 45 |

### LoRA Adapters
| Directory | Size | Base Model | Purpose |
|-----------|------|------------|---------|
| `models/phoneme_lift_qwen/final/` | 603MB | Qwen3.5-2B | LIFT v1 (6K examples, r=32) |
| `models/phoneme_lift_qwen_v2/` | 1.0GB | Qwen3.5-2B | LIFT v2 (31K examples, r=64, IN PROGRESS) |
| `models/dcond_lift_corrector/final/` | 103MB | Qwen3.5-2B | DCoND-LIFT corrector (r=32) |

### Other Model Files
| File | Size | Purpose |
|------|------|---------|
| `data/phoneme_5gram.arpa` | ~10MB | KenLM 5-gram phoneme LM |

### Failed Experiment Checkpoints (can be deleted)
All files matching `results/L2_L2_dcond_*_best.pt` and `results/L2_L2_pe_dcond_*_best.pt`
EXCEPT the 4 seed checkpoints above. These are from failed architecture experiments
(h=1024, k=32, bidirectional, SGD, dual-head, etc.) that achieved >60% PER.

Total failed checkpoint size: ~3.5GB (can be cleaned up).

## Upload Instructions

### Model Weights to Storage

```bash
# Create tarball of essential weights only
cd /mnt/home/vincent.wilmet/brain2speech

# DCoND checkpoints (4 × 58MB = 232MB)
tar czf lead2_dcond_checkpoints.tar.gz \
    results/L2_L2_pe_dcond_adam_s42_long_best.pt \
    results/L2_L2_pe_dcond_adam_s43_best.pt \
    results/L2_L2_pe_dcond_adam_s44_best.pt \
    results/L2_L2_pe_dcond_adam_s45_best.pt

# LoRA adapters (603MB + 103MB = ~706MB, v2 TBD)
tar czf lead2_lora_adapters.tar.gz \
    models/phoneme_lift_qwen/ \
    models/dcond_lift_corrector/

# KenLM model
tar czf lead2_kenlm.tar.gz data/phoneme_5gram.arpa

# All results JSONs
tar czf lead2_results.tar.gz results/L2_*.json

# Upload to storage (adjust path as needed)
cp lead2_dcond_checkpoints.tar.gz /mnt/home/vincent.wilmet/data/
cp lead2_lora_adapters.tar.gz /mnt/home/vincent.wilmet/data/
cp lead2_kenlm.tar.gz /mnt/home/vincent.wilmet/data/
cp lead2_results.tar.gz /mnt/home/vincent.wilmet/data/
```

### After LIFT v2 completes:
```bash
# Add LIFT v2 adapter
tar czf lead2_lift_v2_adapter.tar.gz models/phoneme_lift_qwen_v2/
cp lead2_lift_v2_adapter.tar.gz /mnt/home/vincent.wilmet/data/
```

## Comprehensive Ablation Results

### Stage Ablation
| Stage | Val PER | Val WER | Notes |
|-------|---------|---------|-------|
| Single DCoND (seed45, greedy) | 18.7% | — | Best single model |
| Ensemble 4 seeds (greedy) | 20.2% | — | WORSE: flattens CTC peaks |
| Ensemble weighted (greedy) | 19.4% | — | Still worse than single |
| KenLM beam search | 18.3% | — | Marginal improvement |
| CMUDict words | — | 57.0% | Lossy phoneme→word conversion |
| CMUDict + OPT rescore | — | 50.5% | Still very lossy |
| **Phoneme LIFT v1** | — | **28.1%** | **BEST single-model** |
| KenLM beam + LIFT | — | 28.3% | No improvement after LIFT |
| LIFT v1 + majority (4 seeds) | — | **27.7%** | **BEST overall** |

### Reranking Experiments
| Strategy | Val WER | Notes |
|----------|---------|-------|
| LIFT greedy (baseline) | 28.2% | Single seed, greedy decode |
| Best-of-6 oracle (sampling) | 24.2% | Oracle upper bound, 15% relative |
| LIFT sampling + majority | 28.2% | No improvement |
| Log-prob reranking | 31.7% | WORSE: degenerate outputs score high |
| Multi-seed oracle | 22.6% | Oracle upper bound, 20% relative |
| **Multi-seed majority** | **27.7%** | **Best automatic method** |
| Base Qwen perplexity | 28.8% | WORSE: perplexity not useful |
| Multi-input prompt | 30.6% | WORSE: model not trained for multi-input |

### Prompt Ablation
| Prompt | Val WER | Notes |
|--------|---------|-------|
| **v1_original** | **28.5%** | BEST: model trained with this prompt |
| v2_simple | 29.2% | Slightly worse |
| v3_context | 29.0% | Slightly worse |
| v4_examples | 30.5% | Worse |
| v5_error_aware | 43.4% | CATASTROPHIC: confuses model |

### Failed Architecture Experiments
| Architecture | Val PER | Notes |
|-------------|---------|-------|
| PE DCoND h=1024 | 87.1% | Never converged |
| PE DCoND k=32 | 93.5% | Never converged |
| DCoND bidirectional | 98.9% | CTC collapse, loss=0 |
| DCoND SGD Linderman | 95.8% | SGD + reverse alpha stuck |
| DCoND mono pretrain 50ep | 92.0% | Pretrained weights didn't transfer |
| DCoND dual-head | 61.8% | Converged but not useful |
| DCoND Adam s47 | 98.9% | CTC collapse |

### PER-Bucketed WER (shows PER is the bottleneck)
| PER Bucket | Val N | Val WER | Test N | Test WER |
|-----------|-------|---------|--------|----------|
| 0-10% | 156 | 5.5% | 264 | 6.0% |
| 10-15% | 149 | 18.8% | 316 | 19.2% |
| 15-20% | 126 | 25.4% | 316 | 27.5% |
| 20-30% | 195 | 38.4% | 532 | 40.6% |
| 30-100% | 94 | 62.9% | 372 | 65.3% |

## Key Findings

1. **PaperExactDCoND (h=512, ks=14, unidir, Adam) is the ONLY working architecture.** All larger/different architectures fail with CTC collapse or never converge. The 1601-class softmax is extremely sensitive.

2. **Phoneme-to-text LIFT is the key innovation.** It bypasses the lossy CMUDict+WFST pipeline entirely. 28% WER vs 57% with CMUDict.

3. **For PER < 10%, the system achieves competition-level WER (5.5-6%).** The bottleneck is PER, not the LIFT stage.

4. **Ensemble HURTS CTC greedy decode.** Averaging logits flattens CTC peaks, increasing PER from 18.5% to 20.2%.

5. **No automatic reranking method works.** Log-probability, perplexity, multi-input all hurt. Only majority vote gives a modest +2% relative improvement.

6. **Oracle gap shows 20% improvement potential** from better candidate selection (22.6% vs 28.2% WER), but no automatic method captures it.

## What Remains / Integration with Other Leads

### Pending
- LIFT v2 evaluation (training in progress)
- DCoND-LIFT corrector evaluation (running)
- Full test set evaluation with corrector

### Required from Other Leads
- **Lead 1:** Diverse GRU checkpoints for cross-lead ensemble
- **Lead 3:** BiMamba/S4 checkpoints for ensemble diversity
- **Lead 4:** Cross-lead ensemble evaluation using our 4 DCoND seeds

### Provided to Other Leads
- 4 DCoND seed checkpoints (58MB each) → Lead 4 ensemble
- KenLM 5-gram phoneme model → All leads
- LIFT v1 adapter → Lead 4 can reuse
- Pipeline summary JSON → All leads for comparison
