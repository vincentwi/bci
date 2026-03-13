#!/usr/bin/env python3
"""
Lead 2: DCoND-LIFT — Full pipeline with LLM correction using text + phonemes.

The LIFT innovation (from arxiv 2411.10657):
  - Standard LLM correction: feed only text candidates
  - DCoND-LIFT: feed BOTH text AND decoded phoneme sequences
  - Phoneme info helps disambiguate homophones (e.g., "their/there/they're")
  - Result: 7.29% → 5.77% WER (21% relative improvement)

For our reproduction:
  - Use Qwen2.5-1.5B (Source: BIT paper — 1.5B optimal for BCI)
  - Reuse existing stage2_lm_correction/ LoRA infrastructure
  - New training data format: (candidates + phonemes) → correct_transcription

Modes:
  - generate-data: Generate LIFT training data from DCoND + KenLM output
  - finetune: Fine-tune Qwen with LoRA on LIFT data
  - evaluate: Run LIFT correction on test data
  - full-pipeline: End-to-end DCoND → KenLM → OPT → LIFT → WER

Usage:
    # Generate training data
    python brain2speech/lead2_dcond_lift.py --mode generate-data \
        --model results/L2_dcond_best.pt \
        --lm data/phoneme_5gram.bin

    # Fine-tune Qwen for LIFT
    CUDA_VISIBLE_DEVICES=4,5 python brain2speech/lead2_dcond_lift.py \
        --mode finetune --base-model Qwen/Qwen2.5-1.5B \
        --train-data data/dcond_lift_train.json

    # Full pipeline evaluation
    CUDA_VISIBLE_DEVICES=4,5,6,7 python brain2speech/lead2_dcond_lift.py \
        --mode full-pipeline \
        --ensemble-paths "results/L2_dcond_seed42.pt,results/L2_dcond_seed43.pt" \
        --kenlm data/phoneme_5gram.bin \
        --opt facebook/opt-6.7b \
        --lift-model models/dcond_lift_qwen/final
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CLASS_TO_ARPABET, ARPABET_TO_CLASS, N_CLASSES

DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")
MODELS_DIR = SCRIPT_DIR / "models"


# ═══════════════════════════════════════════════════════════════════════
# DCOND-LIFT PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════

TEMPLATES = {
    'v1_text_only': """You are correcting speech decoded from a brain-computer interface.

Given these candidate transcriptions:
{candidates_block}

Select or produce the most accurate transcription. Consider grammar, common phrases, and plausibility.
Output ONLY the corrected transcription, nothing else.""",

    'v2_text_phones': """You are correcting speech decoded from a brain-computer interface.

Given these candidate transcriptions with their decoded phoneme sequences:
{candidates_block}

Select or produce the most accurate transcription. Consider:
1. Text plausibility (grammar, common phrases)
2. Phoneme-text consistency (phonemes should match the words)
3. Similar-sounding word alternatives

Output ONLY the corrected transcription, nothing else.""",

    'v3_text_phones_conf': """You are correcting speech decoded from a brain-computer interface.

Given these candidate transcriptions with their decoded phoneme sequences and confidence:
{candidates_block}

Select or produce the most accurate transcription. Pay special attention to:
1. Low-confidence phonemes (marked with *) — these are likely errors
2. Phoneme-text consistency
3. Grammar and common phrases

Output ONLY the corrected transcription, nothing else.""",

    'v4_nbest_select': """You are selecting the best transcription from brain-computer interface decoder output.

Candidates (ranked by decoder confidence):
{candidates_block}

Select the most plausible transcription. Output ONLY the selected text, nothing else.""",
}


def format_candidates_v1(candidates):
    """Text-only format."""
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f'{i+1}. "{c["word_text"]}"')
    return '\n'.join(lines)


def format_candidates_v2(candidates):
    """Text + phoneme format (DCoND-LIFT key innovation)."""
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f'Candidate {i+1}: "{c["word_text"]}"')
        lines.append(f'Phonemes {i+1}: {c.get("phone_text", "")}')
        lines.append('')
    return '\n'.join(lines)


def format_candidates_v3(candidates):
    """Text + phonemes + confidence markers."""
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f'Candidate {i+1}: "{c["word_text"]}"')
        # Mark low-confidence phonemes with *
        phones = c.get('phone_text', '')
        conf = c.get('phone_confidence', [])
        if conf and phones:
            tokens = phones.split()
            marked = []
            for j, tok in enumerate(tokens):
                if j < len(conf) and conf[j] < 0.5:
                    marked.append(f'*{tok}*')
                else:
                    marked.append(tok)
            phones = ' '.join(marked)
        lines.append(f'Phonemes {i+1}: {phones}')
        lines.append('')
    return '\n'.join(lines)


def format_candidates_v4(candidates):
    """Simple numbered list for selection."""
    lines = []
    for i, c in enumerate(candidates):
        score = c.get('combined_score', 0)
        lines.append(f'{i+1}. "{c["word_text"]}" (conf: {score:.2f})')
    return '\n'.join(lines)


FORMAT_FNS = {
    'v1_text_only': format_candidates_v1,
    'v2_text_phones': format_candidates_v2,
    'v3_text_phones_conf': format_candidates_v3,
    'v4_nbest_select': format_candidates_v4,
}


def build_lift_prompt(candidates, template_name='v2_text_phones'):
    """Build a DCoND-LIFT prompt from candidates."""
    template = TEMPLATES[template_name]
    format_fn = FORMAT_FNS[template_name]
    block = format_fn(candidates)
    return template.format(candidates_block=block)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_lift_training_data(nbest_json, ground_truth_texts,
                                template_name='v2_text_phones',
                                n_candidates=5, output_path=None):
    """Generate (prompt, target) pairs for LIFT fine-tuning.

    For each trial:
      - Input: LIFT prompt with top-N candidates + phonemes
      - Target: ground truth text

    Args:
        nbest_json: N-best results from lead2_decode_kenlm + lead2_rescore_opt
        ground_truth_texts: list of ground truth text strings
        template_name: which prompt template to use
        n_candidates: how many candidates to include in prompt
        output_path: where to save training data JSON
    """
    training_data = []

    for trial_data, gt_text in zip(nbest_json, ground_truth_texts):
        candidates = trial_data.get('nbest', [])[:n_candidates]
        if not candidates:
            continue

        prompt = build_lift_prompt(candidates, template_name)
        training_data.append({
            'prompt': prompt,
            'target': gt_text,
            'n_candidates': len(candidates),
        })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"LIFT training data: {output_path} ({len(training_data)} examples)")

    return training_data


# ═══════════════════════════════════════════════════════════════════════
# QWEN FINE-TUNING FOR LIFT
# ═══════════════════════════════════════════════════════════════════════

def finetune_qwen_lift(train_data_path, base_model='Qwen/Qwen2.5-1.5B',
                       output_dir=None, lora_r=64, lora_alpha=128,
                       lr=5e-5, epochs=8, batch_size=16):
    """Fine-tune Qwen for DCoND-LIFT correction with LoRA.

    Source: BIT paper (arxiv 2511.21740) — 1.5B optimal for BCI.
    """
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer)
    from peft import LoraConfig, get_peft_model, TaskType

    if output_dir is None:
        output_dir = str(MODELS_DIR / 'dcond_lift_qwen')

    print(f"Fine-tuning {base_model} for DCoND-LIFT")
    print(f"  LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"  LR: {lr}, Epochs: {epochs}, Batch: {batch_size}")

    # Load training data
    with open(train_data_path) as f:
        train_data = json.load(f)
    print(f"  Training examples: {len(train_data)}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map='auto')

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize data
    from torch.utils.data import Dataset

    class LIFTDataset(Dataset):
        def __init__(self, data, tokenizer, max_len=512):
            self.examples = []
            for item in data:
                text = item['prompt'] + '\n' + item['target'] + tokenizer.eos_token
                enc = tokenizer(text, truncation=True, max_length=max_len,
                                padding='max_length', return_tensors='pt')
                # Create labels: mask prompt tokens
                prompt_enc = tokenizer(item['prompt'] + '\n',
                                       truncation=True, max_length=max_len)
                prompt_len = len(prompt_enc.input_ids)

                labels = enc.input_ids.clone().squeeze(0)
                labels[:prompt_len] = -100  # Mask prompt
                # Mask padding
                labels[labels == tokenizer.pad_token_id] = -100

                self.examples.append({
                    'input_ids': enc.input_ids.squeeze(0),
                    'attention_mask': enc.attention_mask.squeeze(0),
                    'labels': labels,
                })

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    dataset = LIFTDataset(train_data, tokenizer)

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        fp16=True,
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=2,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        report_to='none',
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save final
    final_dir = os.path.join(output_dir, 'final')
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved: {final_dir}")


# ═══════════════════════════════════════════════════════════════════════
# LIFT INFERENCE
# ═══════════════════════════════════════════════════════════════════════

class LIFTCorrector:
    """DCoND-LIFT text correction using fine-tuned Qwen.

    Loads LoRA adapter and generates corrected transcription from
    candidates + phonemes prompt.
    """

    def __init__(self, model_path, base_model='Qwen/Qwen2.5-1.5B'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"Loading LIFT corrector: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16, device_map='auto')
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()

    def correct(self, candidates, template_name='v2_text_phones',
                max_new_tokens=128):
        """Generate corrected transcription from candidates.

        Args:
            candidates: list of dicts with 'word_text' and optionally 'phone_text'
            template_name: which prompt template

        Returns: corrected text string
        """
        import torch

        prompt = build_lift_prompt(candidates, template_name)
        inputs = self.tokenizer(prompt + '\n', return_tensors='pt')
        input_ids = inputs.input_ids.to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated = output[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_full_pipeline(args):
    """Run complete DCoND-LIFT pipeline: ensemble → KenLM → OPT → LIFT → WER.

    This is Phase L2.7 of the plan.
    """
    import torch
    from train_beyond_paper import load_h5_dataset, compute_per, CTC_BLANK
    from lead2_train_dcond import (
        DCoNDDecoder, build_marginalization_matrix_ext, collate_dcond,
    )
    from lead2_ensemble import DCoNDEnsemble

    device = torch.device('cuda:0')

    # Load ensemble
    checkpoint_paths = args.ensemble_paths.split(',')
    print(f"\nLoading {len(checkpoint_paths)}-model DCoND ensemble...")
    ensemble = DCoNDEnsemble(checkpoint_paths, device=device)

    M_ext = build_marginalization_matrix_ext()

    # Load data
    trials = load_h5_dataset(DATA_DIR / args.data, load_raw=True)
    session_names = sorted(set(t['session'] for t in trials))
    session_to_idx = {s: i for i, s in enumerate(session_names)}
    for t in trials:
        t['session_idx'] = session_to_idx[t['session']]

    sessions = {}
    for t in trials:
        sessions.setdefault(t['session'], []).append(t)

    test_sessions = session_names[-4:]
    test_trials = [t for s in test_sessions for t in sessions[s]]
    print(f"Test: {len(test_trials)} trials")

    # Stage 1: Ensemble → monophone log probs
    print("\n[Stage 1] Ensemble inference → monophone log probs")
    all_mono_lp = ensemble.get_all_monophone_log_probs(
        test_trials, M_ext, device, batch_size=args.batch_size)

    # Greedy baseline
    from train_beyond_paper import ctc_greedy_decode
    greedy_pers = []
    for lp, target in all_mono_lp:
        decoded = ctc_greedy_decode(lp, blank=CTC_BLANK)
        greedy_pers.append(compute_per(decoded, target))
    print(f"  Greedy PER: {np.mean(greedy_pers):.1%}")

    kenlm_pers = None  # sentinel for conditional reporting below

    # Stage 2: KenLM beam search
    print(f"\n[Stage 2] KenLM beam search (beam={args.beam_width})")
    try:
        from pyctcdecode import build_ctcdecoder
        from lead2_decode_kenlm import build_vocab, phoneme_string_to_indices

        vocab = build_vocab()
        decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model_path=args.kenlm,
            alpha=args.lm_weight,
        )

        kenlm_pers = []
        all_nbest = []
        for idx, (lp, target) in enumerate(all_mono_lp):
            probs = np.exp(lp)
            beams = decoder.decode_beams(probs, beam_width=args.beam_width)
            nbest = []
            for beam in beams[:args.top_n]:
                nbest.append({
                    'phone_text': beam[0],
                    'logit_score': float(beam[2]) if len(beam) > 2 else 0,
                    'lm_score': float(beam[3]) if len(beam) > 3 else 0,
                    'combined_score': float(
                        (beam[2] if len(beam) > 2 else 0) +
                        (beam[3] if len(beam) > 3 else 0)),
                })
            all_nbest.append({'nbest': nbest, 'target': target})

            best = phoneme_string_to_indices(beams[0][0]) if beams else []
            kenlm_pers.append(compute_per(best, target))

        print(f"  KenLM PER: {np.mean(kenlm_pers):.1%}")
    except ImportError:
        print("  SKIPPED: pyctcdecode not available")
        all_nbest = None

    # Stage 3: Phoneme→word + OPT rescoring (if available)
    if all_nbest and args.opt:
        print(f"\n[Stage 3] OPT-6.7B rescoring")
        try:
            from lead2_rescore_opt import CMUDictTrie, OPTRescorer
            trie = CMUDictTrie()
            trie_path = DATA_DIR / 'cmudict_trie.pkl'
            if trie_path.exists():
                trie.load(str(trie_path))
            else:
                trie.load_cmudict()
                trie.save(str(trie_path))

            # Convert phonemes to words
            for trial in all_nbest:
                for hyp in trial['nbest']:
                    phones = hyp['phone_text'].strip().split()
                    words = trie.phonemes_to_words(phones)
                    hyp['word_text'] = ' '.join(words)

            # OPT rescore
            rescorer = OPTRescorer(args.opt)
            for trial in all_nbest:
                candidates = [h['word_text'] for h in trial['nbest'] if h['word_text']]
                ctc_scores = [h['combined_score'] for h in trial['nbest'] if h['word_text']]
                if candidates:
                    rescored = rescorer.rescore_nbest(
                        candidates, ctc_scores, opt_weight=args.opt_weight)
                    trial['opt_best'] = rescored[0]['text'] if rescored else ''

            print(f"  OPT rescoring complete")
        except Exception as e:
            print(f"  OPT rescoring failed: {e}")

    # Stage 4: LIFT correction (if model available)
    if all_nbest and args.lift_model and os.path.exists(args.lift_model):
        print(f"\n[Stage 4] DCoND-LIFT correction")
        try:
            corrector = LIFTCorrector(args.lift_model)
            for trial in all_nbest:
                candidates = trial['nbest'][:5]
                corrected = corrector.correct(candidates,
                                              template_name=args.prompt_template)
                trial['lift_corrected'] = corrected
            print(f"  LIFT correction complete")
        except Exception as e:
            print(f"  LIFT correction failed: {e}")

    # Report results
    print(f"\n{'=' * 60}")
    print("PIPELINE RESULTS")
    print(f"{'=' * 60}")
    print(f"  Stage 0 (Greedy):  PER = {np.mean(greedy_pers):.1%}")
    if kenlm_pers is not None:
        print(f"  Stage 2 (KenLM):   PER = {np.mean(kenlm_pers):.1%}")

    # Save ablation
    ablation = {
        'greedy_per': float(np.mean(greedy_pers)),
        'n_models': len(checkpoint_paths),
        'beam_width': args.beam_width,
        'lm_weight': args.lm_weight,
    }
    if kenlm_pers is not None:
        ablation['kenlm_per'] = float(np.mean(kenlm_pers))

    ablation_path = RESULTS_DIR / 'L2_pipeline_ablation.json'
    with open(ablation_path, 'w') as f:
        json.dump(ablation, f, indent=2)
    print(f"\nAblation saved: {ablation_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Lead 2: DCoND-LIFT Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['generate-data', 'finetune', 'evaluate',
                                 'full-pipeline'],
                        help='Pipeline mode')

    # Data generation
    parser.add_argument('--model', type=str, default=None,
                        help='DCoND model checkpoint')
    parser.add_argument('--data', type=str, default='sentences_paper_256d.h5')
    parser.add_argument('--train-data', type=str, default=None,
                        help='LIFT training data JSON')

    # Fine-tuning
    parser.add_argument('--base-model', type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--lora-r', type=int, default=64)
    parser.add_argument('--lora-alpha', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)

    # Full pipeline
    parser.add_argument('--ensemble-paths', type=str, default=None)
    parser.add_argument('--kenlm', type=str,
                        default=str(DATA_DIR / 'phoneme_5gram.bin'))
    parser.add_argument('--opt', type=str, default=None,
                        help='OPT model name (e.g., facebook/opt-6.7b)')
    parser.add_argument('--opt-weight', type=float, default=0.5)
    parser.add_argument('--lift-model', type=str, default=None,
                        help='Path to fine-tuned LIFT Qwen model')
    parser.add_argument('--prompt-template', type=str, default='v2_text_phones',
                        choices=list(TEMPLATES.keys()))
    parser.add_argument('--beam-width', type=int, default=5000)
    parser.add_argument('--lm-weight', type=float, default=2.0)
    parser.add_argument('--top-n', type=int, default=100)

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == 'generate-data':
        # Load N-best from existing decode
        nbest_path = RESULTS_DIR / 'L2_kenlm_nbest.json'
        if not nbest_path.exists():
            print(f"ERROR: Run lead2_decode_kenlm.py first to generate {nbest_path}")
            sys.exit(1)

        with open(nbest_path) as f:
            nbest_data = json.load(f)

        # Get ground truth texts indexed by trial_idx stored in N-best data.
        # If trial_idx is not stored, use the trial's own 'target' phoneme
        # indices (which lead2_decode_kenlm.py stores per trial).
        import h5py
        h5_path = DATA_DIR / args.data
        all_gt_texts = []
        with h5py.File(h5_path, 'r') as f:
            n_trials = f.attrs['n_trials']
            for i in range(n_trials):
                all_gt_texts.append(f[f'trial_{i:05d}'].attrs['text'])

        # Build ground truth for each split by matching trial indices.
        # The N-best data from lead2_decode_kenlm stores 'trial_idx' per
        # trial entry (relative to the eval split). We need to map these
        # back to the correct HDF5 trial. Since the eval split uses the
        # same session-based split, reconstruct the split mapping.
        session_names_list = []
        with h5py.File(h5_path, 'r') as f:
            for i in range(f.attrs['n_trials']):
                session_names_list.append(f[f'trial_{i:05d}'].attrs['session'])

        session_names_sorted = sorted(set(session_names_list))
        test_sessions = set(session_names_sorted[-4:])
        remaining = session_names_sorted[:-4]
        val_sessions = set(remaining[-2:])

        split_trial_indices = {'val': [], 'test': []}
        for i, sess in enumerate(session_names_list):
            if sess in val_sessions:
                split_trial_indices['val'].append(i)
            elif sess in test_sessions:
                split_trial_indices['test'].append(i)

        output = str(DATA_DIR / 'dcond_lift_train.json')
        for split_name, trials in nbest_data.items():
            # Get correctly-aligned ground truth texts for this split
            indices = split_trial_indices.get(split_name, [])
            gt_texts = [all_gt_texts[idx] for idx in indices]
            # Truncate to match number of N-best trials
            gt_texts = gt_texts[:len(trials)]
            generate_lift_training_data(
                trials, gt_texts,
                template_name=args.prompt_template,
                output_path=output)

    elif args.mode == 'finetune':
        if not args.train_data:
            args.train_data = str(DATA_DIR / 'dcond_lift_train.json')
        finetune_qwen_lift(
            args.train_data, base_model=args.base_model,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)

    elif args.mode == 'evaluate':
        if not args.lift_model:
            print("ERROR: --lift-model required for evaluate mode")
            sys.exit(1)

        corrector = LIFTCorrector(args.lift_model, args.base_model)

        # Load test data
        nbest_path = RESULTS_DIR / 'L2_kenlm_nbest.json'
        with open(nbest_path) as f:
            nbest_data = json.load(f)

        for template in list(TEMPLATES.keys()):
            print(f"\nTemplate: {template}")
            # Run correction on first N examples
            for split_name, trials in nbest_data.items():
                for trial in trials[:10]:
                    candidates = trial.get('nbest', [])[:5]
                    if candidates:
                        corrected = corrector.correct(candidates, template)
                        print(f"  {corrected}")

    elif args.mode == 'full-pipeline':
        if not args.ensemble_paths:
            print("ERROR: --ensemble-paths required")
            sys.exit(1)
        run_full_pipeline(args)

    print("\nDone.")


if __name__ == '__main__':
    main()
