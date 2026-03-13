#!/usr/bin/env python3
"""
Stage 2d: Confidence-gated phoneme correction using fine-tuned Qwen3.5-2B.

PhonemeCorrector loads the LoRA adapter and corrects noisy ARPABET sequences.
High-confidence classifier predictions are kept; low-confidence ones are
sent through the LM for correction.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    QWEN_MODEL_NAME, LORA_ADAPTER_PATH, SYSTEM_PROMPT,
    CLASS_TO_ARPABET, ARPABET_39, N_CLASSES,
)


class PhonemeCorrector:
    """Corrects noisy ARPABET phoneme sequences using a fine-tuned LM."""

    def __init__(self, base_model=QWEN_MODEL_NAME,
                 adapter_path=str(LORA_ADAPTER_PATH),
                 device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16
        )
        self.model = PeftModel.from_pretrained(base, adapter_path).to(device).eval()
        self.device = device
        self.valid_phonemes = set(ARPABET_39) | {'SIL'}

    def correct(self, noisy_phonemes, confidences, threshold=0.8):
        """
        Correct a noisy phoneme sequence.

        Args:
            noisy_phonemes: list of ARPABET strings from classifier
            confidences: per-phoneme max softmax probability
            threshold: only correct phonemes below this confidence

        Returns:
            corrected: list of ARPABET strings
        """
        # Skip LM if all predictions are high-confidence
        if all(c >= threshold for c in confidences):
            return list(noisy_phonemes)

        input_text = " ".join(noisy_phonemes)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=len(noisy_phonemes) * 3,
                num_beams=5,
                do_sample=False,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        corrected = response.strip().split()

        # Validate: all tokens must be valid ARPABET phonemes
        validated = []
        for i, p in enumerate(corrected[:len(noisy_phonemes)]):
            if p in self.valid_phonemes:
                validated.append(p)
            elif i < len(noisy_phonemes):
                validated.append(noisy_phonemes[i])

        # Pad/truncate to match input length
        while len(validated) < len(noisy_phonemes):
            validated.append(noisy_phonemes[len(validated)])
        validated = validated[:len(noisy_phonemes)]

        # Confidence gating: keep high-confidence originals
        for i, conf in enumerate(confidences):
            if conf >= threshold:
                validated[i] = noisy_phonemes[i]

        return validated

    def correct_batch(self, sequences, confidences_list, threshold=0.8):
        """Correct a batch of phoneme sequences."""
        return [
            self.correct(seq, conf, threshold)
            for seq, conf in zip(sequences, confidences_list)
        ]


def classifier_to_phoneme_sequence(softmax_probs, class_to_arpabet=CLASS_TO_ARPABET,
                                    top_k=3):
    """
    Convert per-trial classifier softmax outputs into phoneme sequences.

    Args:
        softmax_probs: (n_trials, 40) array
        class_to_arpabet: dict mapping class index → ARPABET string
        top_k: number of candidates to return per trial

    Returns:
        phonemes: list of ARPABET strings (argmax per trial)
        confidences: list of floats (max softmax per trial)
        top_k_candidates: list of [(phoneme, prob), ...] per trial
    """
    phonemes = []
    confidences = []
    top_k_candidates = []

    for probs in softmax_probs:
        top_indices = np.argsort(probs)[-top_k:][::-1]
        best_idx = top_indices[0]

        phonemes.append(class_to_arpabet[best_idx])
        confidences.append(float(probs[best_idx]))
        top_k_candidates.append([
            (class_to_arpabet[idx], float(probs[idx])) for idx in top_indices
        ])

    return phonemes, confidences, top_k_candidates


def brain_to_corrected_phonemes(neural_data, classifier_model, corrector,
                                 class_to_arpabet=CLASS_TO_ARPABET,
                                 threshold=0.8):
    """
    End-to-end: neural data → corrected phoneme sequence.

    Args:
        neural_data: (n_trials, time, channels) or (n_trials, channels, time)
        classifier_model: trained phoneme classifier with forward() returning logits
        corrector: PhonemeCorrector instance
        class_to_arpabet: mapping dict
        threshold: confidence gating threshold

    Returns:
        corrected_phonemes, raw_phonemes, confidences
    """
    import torch.nn.functional as F

    # Get classifier predictions
    if isinstance(neural_data, np.ndarray):
        neural_data = torch.FloatTensor(neural_data)
    if neural_data.device != next(classifier_model.parameters()).device:
        neural_data = neural_data.to(next(classifier_model.parameters()).device)

    classifier_model.eval()
    with torch.no_grad():
        logits = classifier_model(neural_data)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    # Convert to phonemes
    raw_phonemes, confidences, candidates = classifier_to_phoneme_sequence(
        probs, class_to_arpabet
    )

    # LM correction
    corrected = corrector.correct(raw_phonemes, confidences, threshold=threshold)

    return corrected, raw_phonemes, confidences


if __name__ == "__main__":
    # Quick test
    print("Testing PhonemeCorrector...")
    try:
        corrector = PhonemeCorrector()
        test_noisy = ['P', 'AH', 'T', 'AH', 'SIL', 'W', 'ER', 'D']
        test_conf = [0.3, 0.9, 0.9, 0.3, 0.99, 0.9, 0.9, 0.9]
        result = corrector.correct(test_noisy, test_conf, threshold=0.8)
        print(f"  Input:  {' '.join(test_noisy)}")
        print(f"  Output: {' '.join(result)}")
    except Exception as e:
        print(f"  Error (expected if adapter not trained yet): {e}")
