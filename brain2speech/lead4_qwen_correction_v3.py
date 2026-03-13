#!/usr/bin/env python3
"""Enhanced Qwen LoRA correction with DCoND-LIFT-style dual input.

Implements 6 prompt templates (A-F), 4 data strategies, multiple LoRA configs,
and iterative correction. Builds on existing stage2_lm_correction infrastructure.

Usage:
    # Evaluate with DCoND-LIFT prompt
    python brain2speech/lead4_qwen_correction_v3.py \
        --mode eval --prompt B_dcond --eval-data brain2speech/results/L4_kenlm_decoded.json

    # Fine-tune with real decoder errors
    python brain2speech/lead4_qwen_correction_v3.py \
        --mode finetune --prompt B_dcond --data-source real_decoder \
        --lr 2e-5 --epochs 3 --lora-r 16 --lora-alpha 32

    # Fine-tune with N-best candidates (DCoND-LIFT style)
    python brain2speech/lead4_qwen_correction_v3.py \
        --mode finetune --prompt F_full_dcond --data-source nbest_candidates \
        --n-candidates 10 --lr 2e-5 --epochs 5

References:
    - Source 8 (DCoND): GPT-3.5 receives BOTH text AND phonemes → 5.77% WER
    - Source 6 (MONA LISA): fine-tuned on only 100 examples → 44% improvement
    - Source 10 (BIT): optimal LLM ~1.5B; LoRA on q_proj, v_proj + FF
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (CLASS_TO_ARPABET, ARPABET_TO_CLASS, ARPABET_39,
                    QWEN_MODEL_NAME, SYSTEM_PROMPT, N_CLASSES)


# ═══════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════

PROMPTS = {
    # A: Current baseline (phonemes only) — existing v1/v2 approach
    'A': (
        "You are a phoneme error correction model for a brain-computer interface. "
        "Given a noisy ARPABET phoneme sequence decoded from neural signals, "
        "output the corrected sequence. Only output the corrected phonemes, nothing else."
    ),

    # B: DCoND-LIFT dual input (Source 8 — key innovation)
    'B_dcond': (
        "You are correcting brain-computer interface decoding output. "
        "You receive both decoded phonemes and an approximate text transcription. "
        "Correct the phoneme sequence so it matches a natural English sentence. "
        "Output only the corrected ARPABET phonemes, space-separated."
    ),

    # C: Confidence-aware
    'C_conf': (
        "You are correcting phonemes from a brain-computer interface. "
        "Each phoneme has a confidence score (0.0-1.0). "
        "Lower confidence means more likely wrong. "
        "Correct low-confidence phonemes while preserving high-confidence ones. "
        "Output only corrected ARPABET phonemes, space-separated."
    ),

    # D: N-best selection (Source 6 LISA prompt)
    'D_nbest': (
        "Choose the transcription that is most accurate, ensuring it is "
        "contextually and grammatically correct. Focus on key differences "
        "in the options that change the meaning or correctness. "
        "Avoid repetitive or nonsensical phrases. "
        "Output only the chosen transcription."
    ),

    # E: Multi-candidate with phonemes (Source 8 + Source 6)
    'E_multi': (
        "A brain-computer interface produced these candidate decodings. "
        "Each includes decoded text and its phoneme sequence. "
        "Select or correct to produce the most accurate transcription. "
        "Output only the corrected sentence text."
    ),

    # F: Full DCoND-LIFT reproduction (Source 8 exact)
    'F_full_dcond': (
        "Perform automatic speech recognition on the following decoded neural signals. "
        "Translate each subgroup of phonemes enclosed by two SIL symbols into one single word. "
        "Remove SIL symbols at the start or the end. "
        "Output the refined transcription and its corresponding phoneme representation only, "
        "without any introductory text."
    ),

    # G: Word-level correction
    'G_word': (
        "You are correcting text from a brain-computer interface. "
        "The text may contain word errors due to phoneme decoding mistakes. "
        "Correct the text to form a natural English sentence. "
        "Output only the corrected text."
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# DATA FORMATTING
# ═══════════════════════════════════════════════════════════════════════

def format_prompt_A(noisy_phonemes, **kwargs):
    """Prompt A: Simple phoneme correction."""
    return {'system': PROMPTS['A'], 'user': noisy_phonemes}


def format_prompt_B(noisy_phonemes, approx_text='', **kwargs):
    """Prompt B: DCoND-LIFT dual input (phonemes + text)."""
    user = f"Phonemes: {noisy_phonemes}\nText: {approx_text}"
    return {'system': PROMPTS['B_dcond'], 'user': user}


def format_prompt_C(noisy_phonemes, confidences=None, **kwargs):
    """Prompt C: Confidence-aware."""
    if confidences:
        phones = noisy_phonemes.split()
        parts = [f"{p}({c:.2f})" for p, c in zip(phones, confidences)]
        user = ' '.join(parts)
    else:
        user = noisy_phonemes
    return {'system': PROMPTS['C_conf'], 'user': user}


def format_prompt_D(candidates_text, **kwargs):
    """Prompt D: N-best text selection."""
    lines = [f"{i+1}: {t}" for i, t in enumerate(candidates_text)]
    user = '\n'.join(lines)
    return {'system': PROMPTS['D_nbest'], 'user': user}


def format_prompt_E(candidates, **kwargs):
    """Prompt E: Multi-candidate with phonemes.

    candidates: list of {'text': ..., 'phonemes': ...}
    """
    lines = [f"{i+1}: {c['text']} | {c['phonemes']}" for i, c in enumerate(candidates)]
    user = '\n'.join(lines)
    return {'system': PROMPTS['E_multi'], 'user': user}


def format_prompt_F(candidates, **kwargs):
    """Prompt F: Full DCoND-LIFT reproduction.

    candidates: list of {'text': ..., 'phonemes': ...}
    """
    lines = [f"Candidate {i+1}: {c['text']} | {c['phonemes']}"
             for i, c in enumerate(candidates)]
    user = '\n'.join(lines)
    return {'system': PROMPTS['F_full_dcond'], 'user': user}


def format_prompt_G(approx_text, **kwargs):
    """Prompt G: Word-level correction."""
    return {'system': PROMPTS['G_word'], 'user': approx_text}


FORMATTERS = {
    'A': format_prompt_A,
    'B_dcond': format_prompt_B,
    'C_conf': format_prompt_C,
    'D_nbest': format_prompt_D,
    'E_multi': format_prompt_E,
    'F_full_dcond': format_prompt_F,
    'G_word': format_prompt_G,
}


# ═══════════════════════════════════════════════════════════════════════
# LORA CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════

LORA_CONFIGS = {
    'minimal': {
        'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.05,
        'target_modules': ['q_proj', 'v_proj'],
    },
    'bit_style': {  # Source 10 (BIT paper)
        'r': 16, 'lora_alpha': 32, 'lora_dropout': 0.05,
        'target_modules': ['q_proj', 'v_proj', 'gate_proj', 'up_proj'],
    },
    'medium': {
        'r': 16, 'lora_alpha': 32, 'lora_dropout': 0.05,
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    },
    'current_v2': {  # Current production (from finetune_qwen_v2.py)
        'r': 32, 'lora_alpha': 64, 'lora_dropout': 0.08,
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                           'gate_proj', 'up_proj', 'down_proj'],
    },
    'large': {
        'r': 64, 'lora_alpha': 128, 'lora_dropout': 0.1,
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                           'gate_proj', 'up_proj', 'down_proj'],
    },
}

MODEL_CONFIGS = {
    'qwen_0.5b': 'Qwen/Qwen2.5-0.5B',
    'qwen_1.5b': 'Qwen/Qwen2.5-1.5B',
    'qwen_2b': 'Qwen/Qwen3.5-2B',
    'qwen_3b': 'Qwen/Qwen2.5-3B',
}


# ═══════════════════════════════════════════════════════════════════════
# TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_pairs(n_pairs=10000, prompt_key='A'):
    """Strategy A: Synthetic pairs from confusion matrix (existing approach).

    Reuses infrastructure from stage2_lm_correction/generate_pairs.py
    """
    from stage2_lm_correction.generate_pairs import (
        download_cmudict, parse_cmudict, filter_cmudict,
        generate_pairs, format_as_chat
    )
    import numpy as np

    cmudict_path = download_cmudict(str(Path(__file__).parent / 'data' / 'cmudict-0.7b'))
    cmudict = parse_cmudict(cmudict_path)
    cmudict = filter_cmudict(cmudict, set(ARPABET_39) | {'SIL'})

    noise_model_path = Path(__file__).parent / 'data' / 'noise_model.npy'
    if noise_model_path.exists():
        confusion = np.load(str(noise_model_path))
    else:
        # Build uniform confusion matrix if noise model not available
        confusion = np.eye(N_CLASSES) * 0.9 + np.ones((N_CLASSES, N_CLASSES)) * 0.1 / N_CLASSES
        confusion = confusion / confusion.sum(axis=1, keepdims=True)

    pairs = generate_pairs(cmudict, confusion, n_pairs=n_pairs)
    return pairs


def generate_real_pairs(model, dataset, device='cuda', prompt_key='B_dcond'):
    """Strategy B: Real decoder error pairs from actual CTC predictions.

    Runs CTC decoder on training data to get (predicted, true) pairs
    with confidence scores and approximate text.
    """
    from lead4_phoneme_to_words import PhonemeToWordConverter
    from lead4_decode_kenlm import CTC_BLANK

    converter = PhonemeToWordConverter()
    pairs = []
    model.eval()

    for trial in dataset:
        if 'features_raw' in trial:
            features = torch.FloatTensor(trial['features_raw']).unsqueeze(0).to(device)
        else:
            features = torch.FloatTensor(trial['features']).unsqueeze(0).to(device)

        session_id = torch.LongTensor([trial.get('session_idx', 0)]).to(device)

        with torch.no_grad():
            logits = model(features, session_id)
            probs = torch.softmax(logits, dim=-1)

        # Greedy decode
        pred_indices = logits[0].argmax(dim=-1).cpu().tolist()
        confidences = probs[0].max(dim=-1).values.cpu().tolist()

        # CTC collapse
        decoded_indices = []
        decoded_confs = []
        prev = -1
        for t_idx, t in enumerate(pred_indices):
            if t != prev:
                if t != CTC_BLANK:
                    decoded_indices.append(t)
                    decoded_confs.append(confidences[t_idx])
            prev = t

        pred_phones = [CLASS_TO_ARPABET[d] for d in decoded_indices if d in CLASS_TO_ARPABET]
        true_phones_arr = trial.get('phoneme_indices', [])
        if hasattr(true_phones_arr, 'tolist'):
            true_phones_arr = true_phones_arr.tolist()
        true_phones = [CLASS_TO_ARPABET[int(t)] for t in true_phones_arr
                       if int(t) in CLASS_TO_ARPABET]

        # Get approximate text
        approx_text = converter.convert(' '.join(pred_phones))
        true_text = converter.convert(' '.join(true_phones))

        pair = {
            'input_phonemes': ' '.join(pred_phones),
            'true_phonemes': ' '.join(true_phones),
            'confidences': decoded_confs[:len(pred_phones)],
            'approx_text': approx_text,
            'true_text': true_text,
        }
        pairs.append(pair)

    return pairs


def generate_nbest_pairs(model, dataset, kenlm_decoder, device='cuda',
                          n_candidates=10, converter=None):
    """Strategy C: Multi-beam N-best pairs (Source 8 DCoND-LIFT).

    Each training example contains N candidates with phonemes + text.
    """
    import numpy as np
    from lead4_phoneme_to_words import PhonemeToWordConverter

    if converter is None:
        converter = PhonemeToWordConverter()

    pairs = []
    model.eval()

    for trial in dataset:
        if 'features_raw' in trial:
            features = torch.FloatTensor(trial['features_raw']).unsqueeze(0).to(device)
        else:
            features = torch.FloatTensor(trial['features']).unsqueeze(0).to(device)

        session_id = torch.LongTensor([trial.get('session_idx', 0)]).to(device)

        with torch.no_grad():
            logits = model(features, session_id)
            probs_np = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)
            probs_np = probs_np / probs_np.sum(axis=-1, keepdims=True)

        beams = kenlm_decoder.decode_beams(probs_np, beam_width=150)

        candidates = []
        for b in beams[:n_candidates]:
            phone_str = b[0]
            text = converter.convert(phone_str)
            score = b[3] if len(b) > 3 else 0.0
            candidates.append({
                'phonemes': phone_str,
                'text': text,
                'score': float(score),
            })

        true_phones_arr = trial.get('phoneme_indices', [])
        if hasattr(true_phones_arr, 'tolist'):
            true_phones_arr = true_phones_arr.tolist()
        true_phones = [CLASS_TO_ARPABET[int(t)] for t in true_phones_arr
                       if int(t) in CLASS_TO_ARPABET]
        true_text = converter.convert(' '.join(true_phones))

        pairs.append({
            'candidates': candidates,
            'true_phonemes': ' '.join(true_phones),
            'true_text': true_text,
        })

    return pairs


# ═══════════════════════════════════════════════════════════════════════
# TRAINING DATA FORMATTING
# ═══════════════════════════════════════════════════════════════════════

def format_training_data(pairs, prompt_key='A', max_examples=None):
    """Format pairs into chat training format for SFTTrainer.

    Args:
        pairs: List of pair dicts (different format per data source)
        prompt_key: Which prompt template to use
        max_examples: Limit number of examples

    Returns:
        List of {"messages": [...]} dicts
    """
    formatted = []

    if max_examples:
        pairs = pairs[:max_examples]

    for pair in pairs:
        system_prompt = PROMPTS.get(prompt_key, PROMPTS['A'])

        if prompt_key == 'A':
            user_msg = pair.get('input_phonemes', pair.get('noisy', ''))
            assistant_msg = pair.get('true_phonemes', pair.get('clean', ''))

        elif prompt_key == 'B_dcond':
            user_msg = (f"Phonemes: {pair.get('input_phonemes', '')}\n"
                        f"Text: {pair.get('approx_text', '')}")
            assistant_msg = pair.get('true_phonemes', '')

        elif prompt_key == 'C_conf':
            phones = pair.get('input_phonemes', '').split()
            confs = pair.get('confidences', [0.5] * len(phones))
            parts = [f"{p}({c:.2f})" for p, c in zip(phones, confs)]
            user_msg = ' '.join(parts)
            assistant_msg = pair.get('true_phonemes', '')

        elif prompt_key in ('E_multi', 'F_full_dcond'):
            candidates = pair.get('candidates', [])
            if prompt_key == 'F_full_dcond':
                lines = [f"Candidate {i+1}: {c['text']} | {c['phonemes']}"
                         for i, c in enumerate(candidates)]
                assistant_msg = f"{pair['true_text']} | {pair['true_phonemes']}"
            else:
                lines = [f"{i+1}: {c['text']} | {c['phonemes']}"
                         for i, c in enumerate(candidates)]
                assistant_msg = pair.get('true_text', '')
            user_msg = '\n'.join(lines)

        elif prompt_key == 'D_nbest':
            candidates = pair.get('candidates', [])
            lines = [f"{i+1}: {c['text']}" for i, c in enumerate(candidates)]
            user_msg = '\n'.join(lines)
            assistant_msg = pair.get('true_text', '')

        elif prompt_key == 'G_word':
            user_msg = pair.get('approx_text', '')
            assistant_msg = pair.get('true_text', '')

        else:
            user_msg = pair.get('input_phonemes', '')
            assistant_msg = pair.get('true_phonemes', '')

        if not user_msg or not assistant_msg:
            continue

        formatted.append({
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_msg},
                {'role': 'assistant', 'content': assistant_msg},
            ]
        })

    print(f"Formatted {len(formatted)} training examples with prompt {prompt_key}")
    return formatted


# ═══════════════════════════════════════════════════════════════════════
# FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════

def finetune(training_data, output_dir, model_name=None,
             lora_config_name='current_v2', lr=1.5e-4, epochs=5,
             batch_size=32, warmup_steps=100, max_length=192,
             grad_accum=2, weight_decay=0.01):
    """Fine-tune Qwen with LoRA on formatted training data.

    Builds on existing infrastructure from finetune_qwen_v2.py.
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    if model_name is None:
        model_name = QWEN_MODEL_NAME

    lora_cfg = LORA_CONFIGS[lora_config_name]

    print(f"\nFine-tuning {model_name}")
    print(f"  LoRA: r={lora_cfg['r']}, α={lora_cfg['lora_alpha']}, "
          f"targets={lora_cfg['target_modules']}")
    print(f"  Training: lr={lr}, epochs={epochs}, batch={batch_size}")
    print(f"  Data: {len(training_data)} examples")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        target_modules=lora_cfg['target_modules'],
        bias='none',
    )

    # Create dataset
    dataset = Dataset.from_list(training_data)

    # Split into train/val
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']

    # Training arguments
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type='cosine',
        logging_steps=50,
        eval_strategy='steps',
        eval_steps=200,
        save_strategy='steps',
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        bf16=True,
        max_length=max_length,
        report_to='none',
        remove_unused_columns=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Train
    trainer.train()

    # Save
    final_dir = output_dir / 'final'
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nModel saved: {final_dir}")

    return str(final_dir)


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE / CORRECTION
# ═══════════════════════════════════════════════════════════════════════

class EnhancedPhonemeCorrector:
    """Enhanced corrector supporting all prompt templates."""

    def __init__(self, base_model=None, adapter_path=None, device='cuda'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if base_model is None:
            base_model = QWEN_MODEL_NAME

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            device_map='auto', trust_remote_code=True)

        if adapter_path and Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Loaded LoRA adapter: {adapter_path}")

        self.model.eval()
        self.device = device

    def correct(self, prompt_data, max_new_tokens=256, num_beams=5):
        """Run correction with given prompt data.

        Args:
            prompt_data: Dict with 'system' and 'user' keys
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width

        Returns:
            Generated text string
        """
        messages = [
            {'role': 'system', 'content': prompt_data['system']},
            {'role': 'user', 'content': prompt_data['user']},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                repetition_penalty=1.1,
            )

        # Extract only newly generated tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return result

    def correct_phonemes(self, noisy_phonemes, approx_text=None,
                          confidences=None, prompt_key='A'):
        """Convenience method for phoneme correction with specified prompt.

        Args:
            noisy_phonemes: Space-separated ARPABET string
            approx_text: Approximate text (for dual-input prompts)
            confidences: Per-phoneme confidence scores
            prompt_key: Prompt template key

        Returns:
            Corrected phoneme string
        """
        formatter = FORMATTERS[prompt_key]

        if prompt_key == 'A':
            prompt_data = formatter(noisy_phonemes)
        elif prompt_key == 'B_dcond':
            prompt_data = formatter(noisy_phonemes, approx_text=approx_text or '')
        elif prompt_key == 'C_conf':
            prompt_data = formatter(noisy_phonemes, confidences=confidences)
        elif prompt_key == 'G_word':
            prompt_data = formatter(approx_text or '')
        else:
            prompt_data = formatter(noisy_phonemes)

        result = self.correct(prompt_data)

        # Validate output against valid ARPABET phonemes
        if prompt_key not in ('D_nbest', 'E_multi', 'G_word'):
            valid_phonemes = set(ARPABET_39) | {'SIL'}
            tokens = result.split()
            validated = [t for t in tokens if t in valid_phonemes]
            if validated:
                return ' '.join(validated)

        return result

    def correct_dcond_lift(self, candidates, top_text=None, top_phones=None):
        """DCoND-LIFT style correction with multiple candidates.

        Args:
            candidates: Either a formatted string or list of dicts
            top_text: Best text candidate (optional)
            top_phones: Best phoneme candidate (optional)

        Returns:
            Corrected text or phoneme string
        """
        if isinstance(candidates, str):
            user_msg = candidates
        else:
            lines = [f"Candidate {i+1}: {c.get('text', '')} | {c.get('phonemes', '')}"
                     for i, c in enumerate(candidates)]
            user_msg = '\n'.join(lines)

        prompt_data = {
            'system': PROMPTS['F_full_dcond'],
            'user': user_msg,
        }
        return self.correct(prompt_data)


def iterative_correction(corrector, input_phonemes, n_passes=3,
                          approx_text=None, prompt_key='A'):
    """Run correction N times, stopping if converged.

    Source 6: single pass usually sufficient.
    """
    current = input_phonemes
    history = [current]

    for i in range(n_passes):
        current = corrector.correct_phonemes(
            current, approx_text=approx_text, prompt_key=prompt_key)
        history.append(current)
        if current == history[-2]:  # Converged
            break

    return current, history


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_corrector(corrector, test_pairs, prompt_key='A', max_samples=500):
    """Evaluate corrector on test pairs.

    Returns dict with PER before/after, precision, recall, damage rate.
    """
    import editdistance

    total_errors_before = 0
    total_errors_after = 0
    total_len = 0
    total_changes = 0
    total_fixed = 0
    total_broken = 0
    n_samples = 0

    for pair in test_pairs[:max_samples]:
        # Handle both flat dict and chat messages format
        if 'messages' in pair:
            msgs = pair['messages']
            user_content = msgs[1]['content'] if len(msgs) > 1 else ''
            clean = msgs[2]['content'] if len(msgs) > 2 else ''
            # Parse user content based on prompt format
            if 'Phonemes:' in user_content:
                lines = user_content.split('\n')
                noisy = lines[0].replace('Phonemes: ', '')
                approx_text = lines[1].replace('Text: ', '') if len(lines) > 1 else ''
            else:
                noisy = user_content
                approx_text = ''
            confidences = None
        else:
            noisy = pair.get('input_phonemes', pair.get('noisy', ''))
            clean = pair.get('true_phonemes', pair.get('clean', ''))
            approx_text = pair.get('approx_text', '')
            confidences = pair.get('confidences', None)

        if not noisy or not clean:
            continue

        corrected = corrector.correct_phonemes(
            noisy, approx_text=approx_text, confidences=confidences,
            prompt_key=prompt_key)

        noisy_list = noisy.split()
        clean_list = clean.split()
        corrected_list = corrected.split()

        per_before = editdistance.eval(noisy_list, clean_list)
        per_after = editdistance.eval(corrected_list, clean_list)

        total_errors_before += per_before
        total_errors_after += per_after
        total_len += len(clean_list)

        # Count changes, fixes, and damage
        for i in range(min(len(noisy_list), len(corrected_list))):
            if i < len(noisy_list) and i < len(corrected_list):
                if noisy_list[i] != corrected_list[i]:
                    total_changes += 1
                    if i < len(clean_list):
                        if corrected_list[i] == clean_list[i] and noisy_list[i] != clean_list[i]:
                            total_fixed += 1
                        elif noisy_list[i] == clean_list[i] and corrected_list[i] != clean_list[i]:
                            total_broken += 1

        n_samples += 1
        if n_samples % 100 == 0:
            print(f"  Evaluated {n_samples}/{min(len(test_pairs), max_samples)}")

    per_before = total_errors_before / max(total_len, 1)
    per_after = total_errors_after / max(total_len, 1)

    metrics = {
        'per_before': per_before,
        'per_after': per_after,
        'per_reduction_relative': (per_before - per_after) / max(per_before, 1e-10),
        'per_reduction_absolute': per_before - per_after,
        'precision': total_fixed / max(total_changes, 1),
        'recall': total_fixed / max(total_errors_before, 1),
        'damage_rate': total_broken / max(total_len - total_errors_before, 1),
        'total_changes': total_changes,
        'total_fixed': total_fixed,
        'total_broken': total_broken,
        'n_samples': n_samples,
    }

    print(f"\nResults (prompt={prompt_key}, n={n_samples}):")
    print(f"  PER before: {per_before:.4f}")
    print(f"  PER after:  {per_after:.4f}")
    print(f"  Reduction:  {metrics['per_reduction_relative']*100:.1f}% relative")
    print(f"  Precision:  {metrics['precision']*100:.1f}%")
    print(f"  Recall:     {metrics['recall']*100:.1f}%")
    print(f"  Damage:     {metrics['damage_rate']*100:.2f}%")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Enhanced Qwen Correction v3')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['finetune', 'eval', 'generate_data'],
                        help='Operation mode')
    parser.add_argument('--prompt', type=str, default='A',
                        choices=list(PROMPTS.keys()),
                        help='Prompt template')
    parser.add_argument('--experiment', type=str, default='L4_qwen_v3',
                        help='Experiment name')

    # Model
    parser.add_argument('--model-config', type=str, default='qwen_2b',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model size config')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to existing adapter')

    # LoRA
    parser.add_argument('--lora-config', type=str, default='current_v2',
                        choices=list(LORA_CONFIGS.keys()))
    parser.add_argument('--lora-r', type=int, default=None,
                        help='Override LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=None,
                        help='Override LoRA alpha')

    # Training
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--max-length', type=int, default=192)

    # Data
    parser.add_argument('--data-source', type=str, default='synthetic',
                        choices=['synthetic', 'real_decoder', 'nbest_candidates', 'jsonl'])
    parser.add_argument('--train-data', type=str, default=None,
                        help='Path to pre-generated JSONL training data')
    parser.add_argument('--n-train-examples', type=int, default=None,
                        help='Limit training examples')
    parser.add_argument('--n-candidates', type=int, default=10,
                        help='N-best candidates for DCoND-LIFT training')

    # Eval
    parser.add_argument('--eval-data', type=str, default=None,
                        help='Path to eval data JSON')
    parser.add_argument('--max-eval-samples', type=int, default=500)
    parser.add_argument('--n-correction-passes', type=int, default=1,
                        help='Iterative correction passes')

    # CTC model (for real/nbest data generation)
    parser.add_argument('--base-model', type=str, default=None,
                        help='Path to CTC model checkpoint for generating pairs')
    parser.add_argument('--data', type=str, default=None,
                        help='H5 data path')
    parser.add_argument('--kenlm', type=str, default=None,
                        help='KenLM model path (for nbest generation)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'brain2speech/models/{args.experiment}'

    model_name = MODEL_CONFIGS[args.model_config]

    if args.mode == 'generate_data':
        # Generate training data
        if args.data_source == 'synthetic':
            pairs = generate_synthetic_pairs(
                n_pairs=args.n_train_examples or 10000,
                prompt_key=args.prompt)
        else:
            print("For real/nbest data, provide --base-model")
            return

        formatted = format_training_data(pairs, prompt_key=args.prompt,
                                          max_examples=args.n_train_examples)
        out_path = f'{args.output_dir}/training_data.jsonl'
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            for item in formatted:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(formatted)} examples to {out_path}")

    elif args.mode == 'finetune':
        # Generate or load training data
        if args.data_source == 'jsonl' and args.train_data:
            # Load pre-generated JSONL pairs (from lead4_generate_correction_pairs.py)
            with open(args.train_data, 'r') as f:
                formatted = [json.loads(line) for line in f]
            if args.n_train_examples:
                formatted = formatted[:args.n_train_examples]
            print(f"Loaded {len(formatted)} training examples from {args.train_data}")
        elif args.data_source == 'synthetic':
            pairs = generate_synthetic_pairs(
                n_pairs=args.n_train_examples or 10000)
            formatted = format_training_data(pairs, prompt_key=args.prompt,
                                              max_examples=args.n_train_examples)
        else:
            print(f"Data source '{args.data_source}' requires --train-data or --base-model")
            return

        # Override LoRA config if specified
        lora_config_name = args.lora_config
        if args.lora_r or args.lora_alpha:
            LORA_CONFIGS[lora_config_name] = dict(LORA_CONFIGS[lora_config_name])
            if args.lora_r:
                LORA_CONFIGS[lora_config_name]['r'] = args.lora_r
            if args.lora_alpha:
                LORA_CONFIGS[lora_config_name]['lora_alpha'] = args.lora_alpha

        finetune(
            training_data=formatted,
            output_dir=args.output_dir,
            model_name=model_name,
            lora_config_name=lora_config_name,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            max_length=args.max_length,
        )

    elif args.mode == 'eval':
        # Load corrector
        corrector = EnhancedPhonemeCorrector(
            base_model=model_name,
            adapter_path=args.model_path,
            device=f'cuda:{args.gpu}',
        )

        # Load test data
        if args.eval_data and Path(args.eval_data).exists():
            with open(args.eval_data, 'r') as f:
                if args.eval_data.endswith('.jsonl'):
                    test_pairs = [json.loads(line) for line in f]
                else:
                    test_pairs = json.load(f)
        else:
            # Generate synthetic test pairs
            pairs = generate_synthetic_pairs(n_pairs=args.max_eval_samples)
            test_pairs = []
            for p in pairs:
                test_pairs.append({
                    'input_phonemes': p[0] if isinstance(p, (list, tuple)) else p.get('noisy', ''),
                    'true_phonemes': p[1] if isinstance(p, (list, tuple)) else p.get('clean', ''),
                })

        metrics = evaluate_corrector(corrector, test_pairs,
                                      prompt_key=args.prompt,
                                      max_samples=args.max_eval_samples)

        # Save metrics
        out_path = f'{args.output_dir}/eval_metrics.json'
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved: {out_path}")


if __name__ == '__main__':
    main()
