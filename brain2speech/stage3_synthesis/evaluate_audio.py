#!/usr/bin/env python3
"""
Round-trip evaluation: synthesized audio → Whisper ASR → phoneme comparison.

Tests end-to-end intelligibility: if the brain-decoded phonemes produce
audio that Whisper can transcribe back to the intended words, the pipeline
is working.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR


def load_cmudict():
    """Load CMU dict for word→phoneme lookup."""
    cmudict_path = DATA_DIR / "cmudict-0.7b"
    if not cmudict_path.exists():
        return {}
    entries = {}
    with open(cmudict_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith(';;;') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].split('(')[0].upper()
            phonemes = [p.rstrip('012') for p in parts[1:]]
            if word not in entries:
                entries[word] = phonemes
    return entries


def round_trip_eval(audio_path, expected_phonemes, whisper_model=None):
    """
    Feed synthesized audio through Whisper, compare recognized text to expected.

    Returns:
        per: phoneme error rate of round-trip
        recognized_text: what Whisper heard
        recognized_phonemes: Whisper text → CMU dict → phonemes
    """
    import editdistance

    if whisper_model is None:
        import whisper
        whisper_model = whisper.load_model("base")

    result = whisper_model.transcribe(str(audio_path))
    recognized_text = result["text"].strip().lower()

    cmudict = load_cmudict()

    # Convert recognized words to phonemes
    recognized_phonemes = []
    for word in recognized_text.split():
        clean = word.strip(".,!?;:'\"").upper()
        if clean in cmudict:
            recognized_phonemes.extend(cmudict[clean])
            recognized_phonemes.append('SIL')

    if recognized_phonemes and recognized_phonemes[-1] == 'SIL':
        recognized_phonemes = recognized_phonemes[:-1]

    # Filter SIL from expected for comparison
    expected_clean = [p for p in expected_phonemes if p != 'SIL']
    recognized_clean = [p for p in recognized_phonemes if p != 'SIL']

    if len(expected_clean) == 0:
        per = 0.0 if len(recognized_clean) == 0 else 1.0
    else:
        per = editdistance.eval(recognized_clean, expected_clean) / len(expected_clean)

    return per, recognized_text, recognized_phonemes


def batch_round_trip_eval(audio_dir, phoneme_sequences, whisper_model_size="base"):
    """Evaluate a batch of synthesized audio files."""
    import whisper

    model = whisper.load_model(whisper_model_size)
    results = []

    audio_files = sorted(Path(audio_dir).glob("*.mp3"))

    for audio_path, expected in zip(audio_files, phoneme_sequences):
        per, text, recon_phonemes = round_trip_eval(audio_path, expected, model)
        results.append({
            'file': str(audio_path),
            'per': per,
            'recognized_text': text,
            'expected_phonemes': expected,
            'recognized_phonemes': recon_phonemes,
        })
        print(f"  {audio_path.name}: PER={per:.2f}, heard='{text}'")

    avg_per = np.mean([r['per'] for r in results])
    print(f"\nAverage round-trip PER: {avg_per:.3f}")
    return results, avg_per


if __name__ == "__main__":
    print("Round-trip evaluation module. Use batch_round_trip_eval() or import.")
