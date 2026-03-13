# Cross-Lead Integration Handoff

**Date:** 2026-03-13
**From:** Lead 1 (GRU Baselines)
**To:** Lead 2 (DCoND), Lead 4 (LM Pipeline)

---

## Current Best Results

| Lead | Best PER | Method | Checkpoints |
|------|----------|--------|-------------|
| Lead 1 | 17.8% test | 5-model ensemble + trigram beam (α=0.3) | 5 .pt files, 1.72GB |
| Lead 3 | 43.2% test | ResBlock+GRU (negative result: GRU >> Mamba) | Not useful for ensemble |
| Lead 4 | 7.33% test | 10-model mixture ensemble + KenLM 5-gram (α=0.005) | 23 .pt files, 7.9GB |

**Lead 4's 7.33% is the best result**, but uses step-decay models (20% PER singles). Lead 1's cosine-20K models (19.3% PER singles) should improve this further.

---

## What Lead 1 Has Prepared for Integration

### 1. Checkpoints (copied to shared storage)

```
/mnt/home/vincent.wilmet/data/lead1_checkpoints/ensemble/
├── L1.5h_cibr_cosine_20k_best.pt   (375MB, val 19.3%, test 20.2%)
├── L1.7c_cos20k_seed42_best.pt     (375MB, val 19.5%, test 20.2%)
├── L1.7c_cos20k_seed123_best.pt    (375MB, val 19.5%, test 20.2%)
├── L1.7c_cos20k_seed789_best.pt    (375MB, val 19.3%, test 20.3%)
└── L1.8a_h768_cos20k_best.pt       (220MB, val 19.1%, test 19.7%)  ← NOTE: h=768
```

### 2. JSON config for Lead 4's ensemble system

```
brain2speech/configs/lead1_for_cross_lead.json
```

This file contains model specs in the same format as `cross_lead_ensemble.json`. Note the h=768 model has `"hidden": 768` — Lead 4's `CTCEnsemble` loads model_kwargs per-checkpoint, so this just works.

### 3. Data extraction (in progress)

Lead 1 is extracting:
- `languageModel.tar.gz` → `/mnt/home/vincent.wilmet/data/languageModel/` (word-level KenLM, WFST vocab)
- `sentences.tar.gz` → `/mnt/home/vincent.wilmet/data/sentences/` (raw .mat files, needed by Lead 2)

### 4. Preprocessed data (shared)

```
/mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5  (1.6GB, 8780 trials, 256D)
```

All leads use this same file. It's outside the git repo.

---

## Detailed Instructions for Lead 4

### Task L4-A: Integrate Lead 1 checkpoints into ensemble (HIGH PRIORITY)

**Why:** Lead 1's cosine-20K models (19.1-19.5% val) are 0.6-1.7pp better than Lead 4's step-decay models (20.1-20.8% val). Replacing the weakest Lead 4 models with Lead 1's should improve ensemble PER significantly.

**Steps:**

1. **Merge Lead 1 models into cross_lead_ensemble.json:**

```bash
cd /mnt/home/vincent.wilmet/bci/.claude/worktrees/rbl4

# Read Lead 1's config
cat brain2speech/configs/lead1_for_cross_lead.json

# Option A: Replace weakest L4 models with L1 models
# Current L4 models: 5 cffan (Adam, ~20.5-20.8%) + 5 cibr (SGD, ~20.1-21.2%)
# Replace the 5 cffan models (weakest) with 5 L1 cosine-20K models
# Keep 5 L4 cibr for optimizer diversity

# Option B: Add all L1 models to get 15-model ensemble
# More models = better, IF each contributes unique errors
```

2. **Test the integrated ensemble:**

```bash
# First test with greedy to check PER improvement
CUDA_VISIBLE_DEVICES=0 python brain2speech/lead4_full_pipeline.py \
    --stage greedy \
    --ensemble-config brain2speech/configs/cross_lead_ensemble_v2.json \
    --mixture --split within-day --eval-set val

# Then with KenLM
CUDA_VISIBLE_DEVICES=0 python brain2speech/lead4_full_pipeline.py \
    --stage kenlm \
    --ensemble-config brain2speech/configs/cross_lead_ensemble_v2.json \
    --kenlm brain2speech/data/phoneme_5gram.arpa \
    --mixture --alpha 0.005 --beam-width 20 \
    --split within-day --eval-set both
```

3. **Expected improvement:**
   - Lead 4's 10-model mixture: 7.33% test PER
   - With 5 L1 cosine models replacing 5 L4 Adam models: estimated **5-6% test PER**
   - With all 15 models: depends on diversity, test empirically

**Critical detail:** Lead 1's h=768 model has `hidden=768` not `hidden=1024`. The `CTCEnsemble` code already handles per-model kwargs from the config JSON, so this should load correctly. But verify that `_get_mono_logits()` handles different hidden sizes (it should — all models output 41-class logits regardless of hidden size).

### Task L4-B: Word-level WFST decoding (HIGH PRIORITY)

**Why:** The OPT-6.7B rescoring failed because phoneme→word conversion via CMUdict reverse lookup produces garbage. The Willett paper uses WFST (T-L-G) decoding to produce valid word sequences directly.

**Data now available:**
```
/mnt/home/vincent.wilmet/data/languageModel/   ← being extracted now
```

This should contain:
- Word-level KenLM models (.arpa or .bin)
- Lexicon (L.fst or phoneme→word mapping)
- Vocabulary file

**Steps:**

1. **Check extracted languageModel contents:**
```bash
ls -la /mnt/home/vincent.wilmet/data/languageModel/
find /mnt/home/vincent.wilmet/data/languageModel/ -name "*.arpa" -o -name "*.bin" -o -name "*.fst" -o -name "*.txt" | head -20
```

2. **Build WFST decoder:**
   - Code exists: `brain2speech/lead4_build_wfst.py` (23KB)
   - Needs: T.fst (CTC topology), L.fst (lexicon), G.fst (word LM)
   - Compose: T ∘ L ∘ G → TLG.fst
   - Decode: shortest path through TLG given acoustic scores

3. **Generate word-level N-best lists:**
   - Instead of phoneme beam → CMUdict lookup, use WFST to directly produce word sequences
   - This gives valid English word sequences for OPT rescoring

4. **Re-run OPT-6.7B rescoring on valid word N-best:**
```bash
CUDA_VISIBLE_DEVICES=0,1 python brain2speech/lead4_full_pipeline.py \
    --stage full \
    --ensemble-config brain2speech/configs/cross_lead_ensemble_v2.json \
    --mixture --kenlm [word_level_lm_path] \
    --opt facebook/opt-6.7b \
    --split within-day --eval-set both
```

### Task L4-C: Compute actual WER (MEDIUM PRIORITY)

**Why:** We only have PER numbers. The competition metric is WER. PER ≠ WER because phoneme→word conversion is lossy.

**Steps:**
1. With WFST decoding producing word sequences, WER is computed directly
2. Without WFST: use `lead4_phoneme_to_words.py` with CMUdict, but results will be noisy
3. The original data has reference text in the .mat files. Check:
```bash
# After sentences.tar.gz extraction:
python -c "
import scipy.io as sio
d = sio.loadmat('/mnt/home/vincent.wilmet/data/sentences/2022_09_22/sentences_block1.mat')
print([k for k in d.keys() if not k.startswith('_')])
"
```

### Task L4-D: Re-evaluate Qwen correction at lower PER (LOW PRIORITY)

**Why:** Qwen-2B correction made things worse at 7.36% PER. With the L1+L4 cross-lead ensemble (expected ~5% PER), there's even less room for LLM correction to help. Skip this unless WFST + OPT gets WER below 10%.

**If pursued:** Use larger model (Qwen-7B or GPT-3.5-turbo) and operate at word level, not phoneme level.

### Task L4-E: Final evaluation protocol (LOW PRIORITY)

1. Train all models on train+val combined
2. Report test PER and test WER
3. Compare to published results:
   - Willett 2023: 19.7% PER / 23.8% WER
   - DCoND-LIFT: 5.77% WER
   - CIBR-Okubo: 8.26% WER

---

## Detailed Instructions for Lead 2

### Context

Lead 2 has **code written but never executed**. All 6 scripts are on the `lead2` branch:
- `lead2_train_dcond.py` — DCoND diphone CTC with progressive alpha
- `lead2_build_kenlm.py` — Phoneme n-gram LM builder
- `lead2_decode_kenlm.py` — KenLM beam search decoding
- `lead2_rescore_opt.py` — OPT-6.7B rescoring
- `lead2_ensemble.py` — Multi-model ensemble
- `lead2_dcond_lift.py` — Diphone→monophone marginalization + LIFT

### Task L2-A: Verify data availability (PREREQUISITE)

```bash
# Check H5 file exists
ls -lh /mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5
# Should show: 1.6GB file

# Check the script's DATA_DIR
grep "^DATA_DIR" brain2speech/lead2_train_dcond.py
# If it points to wrong path, update to:
# DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")

# Check that raw sentences are extracted (needed for phoneme sequences)
ls /mnt/home/vincent.wilmet/data/sentences/
# Should show dated directories: 2022_09_22, 2022_09_23, etc.
```

### Task L2-B: Train DCoND baseline (4 GPUs, ~4-8 hours)

**What DCoND does:** Instead of predicting 40 monophones, it predicts 1600 diphone (bigram) classes. This gives the CTC decoder more phonotactic information. The loss is:

```
L = α × L_monophone_ctc + (1-α) × L_diphone_ctc
```

With progressive alpha: start at α=0 (diphone only), increase +0.1 every 10 epochs to α=0.6.

**Steps:**

1. **Sanity check the script:**
```bash
# Read the script to understand its args
python brain2speech/lead2_train_dcond.py --help

# Quick sanity run (2 epochs, 50 trials)
CUDA_VISIBLE_DEVICES=0 python -u brain2speech/lead2_train_dcond.py \
    --max-trials 50 --max-epochs 2 \
    --experiment L2.0_sanity 2>&1 | head -50
```

2. **Train 4 seeds in parallel:**
```bash
# Based on DCoND paper: Adam lr=0.02, batch=32, 120 epochs
for gpu in 0 1 2 3; do
    seed=$((42 + gpu))
    CUDA_VISIBLE_DEVICES=$gpu python -u brain2speech/lead2_train_dcond.py \
        --hidden 1024 --n-layers 5 --bidirectional true \
        --kernel-size 32 --stride 4 \
        --optimizer adam --lr 0.02 --adam-eps 0.1 \
        --batch-size 32 --max-epochs 120 \
        --alpha-schedule progressive --alpha-max 0.6 \
        --seed $seed --experiment L2.1_dcond_s${seed} \
        2>&1 | tee brain2speech/results/lead2/log_L2.1_s${seed}.txt &
done
wait
```

**If the script doesn't have `--alpha-schedule`:** Check the actual CLI args. The progressive alpha may be hardcoded in the training loop.

3. **Also try SGD variant (Lead 1 found SGD >> Adam):**
```bash
CUDA_VISIBLE_DEVICES=0 python -u brain2speech/lead2_train_dcond.py \
    --hidden 1024 --n-layers 5 --bidirectional true \
    --kernel-size 32 --stride 4 \
    --optimizer sgd --lr 0.1 --momentum 0.9 \
    --scheduler cosine --max-minibatches 20000 --patience 100 \
    --batch-size 128 --white-noise 0.8 --ortho-init \
    --alpha-schedule progressive --alpha-max 0.6 \
    --seed 42 --experiment L2.2_dcond_sgd_cosine \
    2>&1 | tee brain2speech/results/lead2/log_L2.2.txt
```

This combines Lead 1's best training recipe (SGD + cosine 20K) with Lead 2's diphone loss.

### Task L2-C: Alpha ablation (2 GPUs, ~6 hours)

```bash
# Test different alpha values
for alpha in 0.2 0.4 0.6 0.8 1.0; do
    CUDA_VISIBLE_DEVICES=$((RANDOM % 4)) python -u brain2speech/lead2_train_dcond.py \
        --hidden 1024 --bidirectional true --kernel-size 32 \
        --optimizer sgd --lr 0.1 --momentum 0.9 \
        --scheduler cosine --max-minibatches 20000 \
        --alpha-fixed $alpha \
        --seed 42 --experiment L2.3_alpha${alpha} \
        2>&1 | tee brain2speech/results/lead2/log_L2.3_a${alpha}.txt &
done
wait
```

### Task L2-D: Diphone→monophone marginalization + ensemble

DCoND produces 1600 diphone logits. To ensemble with Lead 1/4's monophone models, you need to marginalize:

```
P(phone_i) = sum_j P(diphone_{i,j})  # sum over all diphones starting with phone_i
```

The code for this is in `lead2_dcond_lift.py`. After marginalization, diphone log-probs become compatible with monophone log-probs for ensemble averaging.

```bash
# Evaluate standalone
python brain2speech/lead2_decode_kenlm.py \
    --model brain2speech/results/lead2/L2.1_dcond_s42_best.pt \
    --lm brain2speech/data/phoneme_5gram.arpa \
    --beam-width 50 --alpha 0.5

# For cross-lead ensemble: marginalize and save monophone log-probs
python brain2speech/lead2_dcond_lift.py \
    --checkpoint brain2speech/results/lead2/L2.1_dcond_s42_best.pt \
    --output brain2speech/results/lead2/L2.1_dcond_s42_mono_logprobs.pt
```

### Task L2-E: Provide checkpoints to Lead 4

After training, copy best checkpoints to shared storage:
```bash
mkdir -p /mnt/home/vincent.wilmet/data/lead2_checkpoints
cp brain2speech/results/lead2/*_best.pt /mnt/home/vincent.wilmet/data/lead2_checkpoints/
```

Create `brain2speech/configs/lead2_for_cross_lead.json` following the format in `brain2speech/configs/lead1_for_cross_lead.json`.

**Key difference:** DCoND models output 1600 diphone classes, not 41 monophone classes. Lead 4's ensemble needs the marginalization step from `lead2_dcond_lift.py` before these can be mixed with L1/L4 monophone models.

---

## Dependency Graph (All Independent Execution)

```
Lead 1 (DONE)
  ├── 5 cosine-20K checkpoints → /mnt/home/vincent.wilmet/data/lead1_checkpoints/ensemble/
  ├── lead1_for_cross_lead.json config
  └── Extracted languageModel.tar.gz and sentences.tar.gz

Lead 2 (INDEPENDENT — can start immediately)
  ├── Uses: sentences_paper_256d.h5 (already exists)
  ├── Trains: DCoND diphone models (4 seeds × 4 GPUs)
  ├── Ablates: alpha schedule
  ├── Outputs: Marginalized monophone checkpoints → /mnt/home/vincent.wilmet/data/lead2_checkpoints/
  └── lead2_for_cross_lead.json config

Lead 4 (INDEPENDENT — can start immediately)
  ├── Task A: Merge Lead 1 checkpoints into ensemble (no GPU needed, 30 min)
  ├── Task B: Build WFST from languageModel.tar.gz (no GPU needed, 1-2 hours)
  ├── Task C: Evaluate cross-lead ensemble + KenLM (1 GPU, 1 hour)
  ├── Task D: Word-level N-best + OPT rescoring (2 GPUs, 2-4 hours)
  └── Task E: Final WER evaluation

Lead 4 + Lead 2 integration (AFTER L2 completes)
  ├── Add Lead 2 marginalized checkpoints to cross-lead ensemble
  └── Re-evaluate full pipeline with all 3 leads
```

No lead blocks any other. Lead 2 and Lead 4 can run in parallel starting now.

---

## Shared Resources

### Data Files
| Path | Size | Status | Used By |
|------|------|--------|---------|
| `/mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5` | 1.6GB | EXISTS | All leads |
| `/mnt/home/vincent.wilmet/data/languageModel/` | ~14GB | EXTRACTING NOW | Lead 4 (WFST, word KenLM) |
| `/mnt/home/vincent.wilmet/data/sentences/` | ~14GB | EXTRACTING NOW | Lead 2 (raw .mat files if needed) |
| `/mnt/home/vincent.wilmet/data/competitionData.tar.gz` | 3.7GB | Partially extracted to `docs/data/dryad/` | Lead 4 (eval data) |
| `brain2speech/data/phoneme_5gram.arpa` | In repo | EXISTS (lead4 branch) | Leads 2, 4 |

### Checkpoints
| Path | Models | Size | Quality |
|------|--------|------|---------|
| `/mnt/home/vincent.wilmet/data/lead1_checkpoints/ensemble/` | 5 | 1.72GB | 19.1-19.5% val (cosine-20K) |
| `/mnt/home/vincent.wilmet/bci/.claude/worktrees/rbl4/brain2speech/results/lead4/` | 23 | 7.9GB | 20.1-20.8% val (step-decay) |
| `/mnt/home/vincent.wilmet/data/lead2_checkpoints/` | TBD | TBD | TBD (awaiting Lead 2 training) |

### GPU Allocation (suggested)
| GPUs | Lead | Task |
|------|------|------|
| 0-3 | Lead 2 | DCoND diphone training (4 seeds parallel) |
| 4-5 | Lead 4 | Cross-lead ensemble eval + OPT rescoring |
| 6-7 | Free | Additional seed training or ablations |
