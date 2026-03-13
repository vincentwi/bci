# Lead 4: LM Decoding + Ensemble Results

## Best Configuration
- **10-model CTC ensemble** (5 cffan/Adam + 5 cibr/SGD, different seeds)
- **Proper mixture model**: log(sum(w_i * p_i)) instead of sum(w_i * log(p_i))
- **5-gram KenLM**: alpha=0.05, beam_width=20
- **Test PER: 7.36%** (val: 7.70%)

## Results Progression

| Stage | Val PER | Test PER | Relative Improvement |
|-------|---------|----------|---------------------|
| Single model greedy (cibr_s43) | 20.2% | ~20.2% | baseline |
| Single + KenLM | 19.7% | ~19.7% | 2.5% |
| 5-model ensemble greedy | 13.5% | 13.9% | 33% |
| 5-model + KenLM | 12.9% | 13.2% | 36% |
| 10-model ensemble greedy | 10.5% | 10.7% | 47% |
| 10-model + KenLM (weighted log-prob) | 9.9% | 10.0% | 50% |
| **10-model MIXTURE + KenLM** | **7.7%** | **7.4%** | **63%** |

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
5. **Qwen phoneme correction not effective**: Too conservative at phoneme level, need word-level approach

## Configs
- Ensemble: `brain2speech/configs/lead4_ensemble_10.json`
- KenLM: `brain2speech/data/phoneme_5gram.arpa`, alpha=0.05, beam=20
- Best command:
```bash
python3 brain2speech/lead4_full_pipeline.py \
    --stage ensemble+kenlm --mixture \
    --ensemble-config brain2speech/configs/lead4_ensemble_10.json \
    --kenlm brain2speech/data/phoneme_5gram.arpa \
    --beam-width 20 --alpha 0.05 \
    --eval-set test
```
