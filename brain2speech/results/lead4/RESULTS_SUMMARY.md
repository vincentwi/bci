# Lead 4: LM Decoding + Ensemble Results

## Best Configuration
- **10-model CTC ensemble** (5 cffan/Adam + 5 cibr/SGD, different seeds)
- **Proper mixture model**: log(sum(w_i * p_i)) instead of sum(w_i * log(p_i))
- **5-gram KenLM**: alpha=0.005, beam_width=20
- **Test PER: 7.33%** (val: 7.41%)

## Results Progression

| Stage | Val PER | Test PER | Relative Improvement |
|-------|---------|----------|---------------------|
| Single model greedy (cibr_s43) | 20.2% | ~20.2% | baseline |
| Single + KenLM | 19.7% | ~19.7% | 2.5% |
| 5-model ensemble greedy | 13.5% | 13.9% | 33% |
| 5-model + KenLM | 12.9% | 13.2% | 36% |
| 10-model ensemble greedy | 10.5% | 10.7% | 47% |
| 10-model + KenLM (weighted log-prob) | 9.9% | 10.0% | 50% |
| **10-model MIXTURE + KenLM** | **7.41%** | **7.33%** | **63%** |

## Model Inventory

| Model | Preset | Seed | Val PER |
|-------|--------|------|---------|
| L4_cffan_baseline | cffan (Adam) | 42 | 20.7% |
| L4_cffan_s42 | cffan (Adam) | 42 | 20.7% |
| L4_cffan_s43 | cffan (Adam) | 43 | 20.5% |
| L4_cffan_s44 | cffan (Adam) | 44 | 20.8% |
| L4_cffan_s45 | cffan (Adam) | 45 | 20.5% |
| L4_cibr_s42 | cibr (SGD) | 42 | 20.4% |
| L4_cibr_s43 | cibr (SGD) | 43 | 20.2% |
| L4_cibr_s44 | cibr (SGD) | 44 | 21.2% |
| L4_cibr_s45 | cibr (SGD) | 45 | 20.1% |
| L4_cibr_s46 | cibr (SGD) | 46 | 20.9% |

## Key Findings

1. **Ensemble is the biggest contributor**: Single 20.2% -> 10-model 10.7% (47% relative)
2. **Proper mixture model is critical**: log(sum(w*p)) >> sum(w*log(p)) with KenLM
   - Weighted log-prob: 10.0% test PER
   - Mixture model: 7.4% test PER (26% relative improvement)
3. **KenLM synergizes with mixture**: 12.0% greedy -> 7.4% with LM (38.8% relative)
4. **Optimizer diversity matters**: Mixing Adam (cffan) and SGD (cibr) models improves ensemble
5. **Qwen-2B correction not effective at low PER**: At 7.36% PER, the 2B model damages more than it fixes. Source 6 (MONA LISA) used GPT-3.5-turbo for this stage. Need larger LLM or word-level WFST approach.
6. **OPT-6.7B verified working**: 8-bit quantization loads on 2 GPUs (~7GB). Correctly distinguishes valid vs invalid English. Bottleneck is word-level N-best generation, not the model itself.

## Pipeline Components

| File | Purpose | Status |
|------|---------|--------|
| `lead4_build_phoneme_lm.py` | Build KenLM 4/5-gram from training phonemes | Done |
| `lead4_decode_kenlm.py` | KenLM-backed CTC beam search + N-best + sweep | Done |
| `lead4_phoneme_to_words.py` | CMU dict reverse lookup for phoneme→word | Done |
| `lead4_rescore_opt.py` | OPT-6.7B N-best rescoring with weight sweep | Done (code ready, needs GPU) |
| `lead4_qwen_correction_v3.py` | Enhanced Qwen LoRA with 7 prompt templates, DCoND-LIFT | Done |
| `lead4_generate_correction_pairs.py` | Real decoder error pairs for Qwen training | Done |
| `lead4_build_wfst.py` | WFST T-L-G construction + simplified decoder | Done |
| `lead4_ensemble_ctc.py` | Multi-model CTC ensemble with weighted voting | Done |
| `lead4_full_pipeline.py` | End-to-end 4-stage evaluation pipeline | Done |
| `configs/lead4_ensemble_10.json` | 10-model ensemble config | Done |
| `configs/cross_lead_ensemble.json` | Cross-lead ensemble (L4 models, L1-3 placeholder) | Ready |

## Stage 3+4 Evaluation (OPT + Qwen)

### OPT-6.7B Rescoring
- **Status**: OPT loads and scores sentences correctly (fp16, ~13GB on 2 GPUs)
- **Result**: OPT rescoring **degrades** PER from 7.41% to 10.55% on val set
  - 413 trials improved, 224 unchanged, 241 worse (but worse trials much worse)
  - WER: 32.9% — phoneme→word conversion produces gibberish
- **Root cause**: Phoneme-level N-best → word conversion via CMU dict reverse lookup is too noisy
  - Many phoneme subsequences don't map to dictionary words
  - OPT scores on garbage word text, reranks incorrectly
- **Fix needed**: WFST-based word-level N-best (Willett pipeline approach) where decoding directly produces valid words

### Qwen LoRA Correction
- **Adapter v1**: `brain2speech/models/L4_qwen_B_real/final/` (28.5MB, trained on 6,942 real decoder pairs)
  - Trained on high-PER single-model errors (~15-20% PER input)
  - On ensemble 7.36% PER output: PER 17.4% → 25.3% (**worse**, +45.6% relative)
  - DCoND-LIFT dual-input: PER 20.8% → 23.0% (still worse, but less bad)
- **Adapter v2**: `brain2speech/models/L4_qwen_ensemble_retrain/final/` (retrained on 874 ensemble error pairs)
  - Train loss: 1.7 → 0.69, token accuracy 82.5%
  - On test: PER 13.6% → 15.2% (**still worse**, +11.2% relative)
  - 6 improved, 133 unchanged, 61 worse out of 200 trials
- **Conclusion**: Qwen-2B is insufficient for correcting already-good (7.36% PER) predictions. Need GPT-3.5/4-class model (Source 6: GPT-3.5 fine-tuned on 100 examples → 44% improvement)

### Zero-shot Word-level Correction
- Base Qwen without fine-tuning on word-level text: WER doubled (66% → 132%)
- Not viable without fine-tuning

## Next Steps

1. **Cross-lead ensemble**: Add Lead 1-3 checkpoints to `cross_lead_ensemble.json` — most promising remaining improvement
2. **WFST word-level decoding**: Use `SimplifiedWFSTDecoder` to generate word-level N-best for OPT rescoring
3. **Larger LLM correction**: Consider using a larger model (7B+) for the correction stage, or API-based models
4. **More ensemble diversity**: Train models with Linderman-style post-RNN norm, speckled masking, FastEmit

## Configs
- Ensemble: `brain2speech/configs/lead4_ensemble_10.json`
- KenLM: `brain2speech/data/phoneme_5gram.arpa`, alpha=0.05, beam=20
- Best command:
```bash
python3 brain2speech/lead4_full_pipeline.py \
    --stage ensemble+kenlm --mixture \
    --ensemble-config brain2speech/configs/lead4_ensemble_10.json \
    --kenlm brain2speech/data/phoneme_5gram.arpa \
    --beam-width 20 --alpha 0.005 \
    --eval-set test
```
