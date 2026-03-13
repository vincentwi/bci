#!/usr/bin/env python3
"""
End-to-end brain-to-speech pipeline.

Neural signals → Phoneme classifier → LM correction → ElevenLabs TTS → Audio

Usage:
    # Full pipeline with pre-loaded models:
    python pipeline.py

    # Test with synthetic data:
    python pipeline.py --test
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from config import (
    CLASS_TO_ARPABET, STAGE1_MODELS, STAGE1_DATA, DATA_DIR, MODELS_DIR,
)


def _import_stage1_models():
    """Import model classes from Stage 1, avoiding config.py name collision."""
    import importlib.util
    train_path = Path("/mnt/home/vincent.wilmet/docs/scripts/train.py")
    scripts_config_path = Path("/mnt/home/vincent.wilmet/docs/scripts/config.py")

    spec_cfg = importlib.util.spec_from_file_location("scripts_config", scripts_config_path)
    scripts_config = importlib.util.module_from_spec(spec_cfg)
    sys.modules["scripts_config"] = scripts_config
    orig_config = sys.modules.get("config")
    sys.modules["config"] = scripts_config
    spec_cfg.loader.exec_module(scripts_config)

    spec = importlib.util.spec_from_file_location("stage1_train", train_path)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    if orig_config is not None:
        sys.modules["config"] = orig_config
    else:
        del sys.modules["config"]
    return train_mod


def load_classifier(model_name="TCN", device="cuda"):
    """Load the best Stage 1 classifier."""
    train_mod = _import_stage1_models()
    TCN = train_mod.TCN
    EEGNet = train_mod.EEGNet
    GRUDecoder = train_mod.GRUDecoder
    SpeechTransformer = train_mod.SpeechTransformer

    # Standard config for phoneme classification
    nc, nt, nk = 1280, 85, 40

    model_configs = {
        "TCN": (TCN, dict(nc=nc, nt=nt, nk=nk, dr=0.3, hidden=128)),
        "EEGNet": (EEGNet, dict(nc=nc, nt=nt, nk=nk, F1=16, D=2, F2=32, kl=32, dr=0.4)),
        "GRU": (GRUDecoder, dict(nc=nc, nt=nt, nk=nk, hidden=256, n_layers=2, dr=0.3)),
        "Transformer": (SpeechTransformer, dict(nc=nc, nt=nt, nk=nk, d_model=128, nhead=8, num_layers=4, dr=0.3)),
    }

    cls, kw = model_configs[model_name]
    model = cls(**kw)

    weight_path = STAGE1_MODELS / f"{model_name.lower()}_best.pt"
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    print(f"Loaded {model_name} classifier from {weight_path}")
    return model


def load_corrector(device="cuda"):
    """Load the fine-tuned phoneme corrector."""
    from stage2_lm_correction.inference_lm import PhonemeCorrector
    return PhonemeCorrector(device=device)


def brain_to_speech(neural_data, classifier, corrector,
                    output_path="brain_speech.mp3",
                    confidence_threshold=0.8):
    """
    Complete pipeline: neural signals → audio.

    Args:
        neural_data: (n_trials, 85, 1280) or (n_trials, 1280, 85) array
        classifier: trained phoneme classifier
        corrector: PhonemeCorrector instance
        output_path: where to save audio
        confidence_threshold: gating threshold for LM correction

    Returns:
        dict with audio_bytes, corrected_phonemes, raw_phonemes, confidences, ssml
    """
    import torch.nn.functional as F
    from stage2_lm_correction.inference_lm import classifier_to_phoneme_sequence
    from stage3_synthesis.elevenlabs_tts import synthesize_speech, phonemes_to_ssml

    # Ensure correct shape: (N, channels, time)
    if isinstance(neural_data, np.ndarray):
        neural_data = torch.FloatTensor(neural_data)

    if neural_data.dim() == 2:
        neural_data = neural_data.unsqueeze(0)

    # Models expect (batch, channels, time) = (N, 1280, 85)
    if neural_data.shape[1] != 1280 and neural_data.shape[2] == 1280:
        neural_data = neural_data.transpose(1, 2)

    device = next(classifier.parameters()).device
    neural_data = neural_data.to(device)

    # Step 1: Classify
    classifier.eval()
    with torch.no_grad():
        logits = classifier(neural_data)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    # Step 2: Convert to phonemes
    raw_phonemes, confidences, candidates = classifier_to_phoneme_sequence(
        probs, CLASS_TO_ARPABET
    )

    print(f"Raw phonemes ({len(raw_phonemes)}): {' '.join(raw_phonemes)}")
    print(f"Avg confidence: {np.mean(confidences):.3f}")

    # Step 3: LM correction
    corrected_phonemes = corrector.correct(
        raw_phonemes, confidences, threshold=confidence_threshold
    )

    n_changed = sum(1 for r, c in zip(raw_phonemes, corrected_phonemes) if r != c)
    print(f"Corrected phonemes: {' '.join(corrected_phonemes)} ({n_changed} changed)")

    # Step 4: Synthesize audio
    ssml = phonemes_to_ssml(corrected_phonemes)
    audio_bytes = synthesize_speech(corrected_phonemes, output_path)

    return {
        'audio_bytes': audio_bytes,
        'corrected_phonemes': corrected_phonemes,
        'raw_phonemes': raw_phonemes,
        'confidences': confidences,
        'candidates': candidates,
        'ssml': ssml,
        'output_path': output_path,
        'n_changed': n_changed,
    }


def test_pipeline():
    """Test with a real sample from the phoneme dataset."""
    print("=" * 60)
    print("Testing brain-to-speech pipeline")
    print("=" * 60)

    # Load a sample from phoneme data
    candidates = sorted(STAGE1_DATA.glob("tuning_*phonemes*.npz"))
    if not candidates:
        print("No phoneme data found for testing")
        return

    data = np.load(candidates[0], allow_pickle=True)
    X = data['X']
    y = data['y']
    class_names = list(data['class_names'])
    print(f"Loaded {X.shape[0]} trials, shape {X.shape}")

    # Pick a few random trials to form a "sentence"
    np.random.seed(42)
    n_trials = 5
    indices = np.random.choice(len(X), n_trials, replace=False)
    neural_data = X[indices]  # (5, 85, 1280)
    true_labels = y[indices]
    true_phonemes = [CLASS_TO_ARPABET[l] for l in true_labels]
    print(f"True phonemes: {' '.join(true_phonemes)}")

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = load_classifier("TCN", device)

    try:
        corrector = load_corrector(device)
    except Exception as e:
        print(f"LM corrector not available ({e}), using passthrough")
        # Passthrough corrector for testing
        class PassthroughCorrector:
            def correct(self, phonemes, confidences, threshold=0.8):
                return list(phonemes)
        corrector = PassthroughCorrector()

    # Run pipeline
    result = brain_to_speech(
        neural_data, classifier, corrector,
        output_path="brain2speech_test.mp3",
    )

    print(f"\nResults:")
    print(f"  True:      {' '.join(true_phonemes)}")
    print(f"  Raw:       {' '.join(result['raw_phonemes'])}")
    print(f"  Corrected: {' '.join(result['corrected_phonemes'])}")
    print(f"  Changed:   {result['n_changed']}")
    print(f"  Audio:     {result['output_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test pipeline")
    args = parser.parse_args()

    if args.test:
        test_pipeline()
    else:
        print("Use --test to run test pipeline, or import brain_to_speech()")
