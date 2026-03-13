#!/usr/bin/env python3
"""
Stage 3: Audio synthesis via ElevenLabs TTS with ARPABET phoneme tags.

Converts corrected ARPABET phoneme sequences to SSML with <phoneme> tags,
sends to ElevenLabs API (eleven_flash_v2 model), returns audio.

Usage:
    python elevenlabs_tts.py                           # test with sample
    python elevenlabs_tts.py --phonemes "B AH T ER"    # specific phonemes
"""
import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ELEVENLABS_API_KEY, ELEVENLABS_BASE_URL,
    ELEVENLABS_MODEL_ID, ELEVENLABS_VOICE_ID, VOWELS,
)


def add_default_stress(phonemes):
    """Add stress markers to vowels (required by ElevenLabs CMU ARPABET).

    Strategy: first vowel in each 'word' (between SIL tokens) gets primary
    stress (1), all other vowels get no stress (0).
    """
    result = []
    seen_stressed = False
    for p in phonemes:
        if p == 'SIL':
            result.append(p)
            seen_stressed = False
            continue
        if p in VOWELS:
            if not seen_stressed:
                result.append(p + '1')
                seen_stressed = True
            else:
                result.append(p + '0')
        else:
            result.append(p)
    return result


def phonemes_to_ssml(phoneme_sequence):
    """Convert ARPABET phoneme sequence to SSML with phoneme tags.

    Each word (delimited by SIL) becomes one <phoneme> tag.
    ElevenLabs uses the phoneme attribute, not the text content.
    """
    stressed = add_default_stress(phoneme_sequence)

    # Split by SIL into words
    words = []
    current_word = []
    for p in stressed:
        if p == 'SIL':
            if current_word:
                words.append(current_word)
                current_word = []
        else:
            current_word.append(p)
    if current_word:
        words.append(current_word)

    if not words:
        return ""

    # Build SSML — each word gets a phoneme tag
    ssml_parts = []
    for word_phonemes in words:
        ph_str = " ".join(word_phonemes)
        ssml_parts.append(
            f'<phoneme alphabet="cmu-arpabet" ph="{ph_str}">word</phoneme>'
        )

    return " ".join(ssml_parts)


def synthesize_speech(phoneme_sequence, output_path="output.mp3",
                      voice_id=ELEVENLABS_VOICE_ID,
                      model_id=ELEVENLABS_MODEL_ID):
    """
    Convert ARPABET phoneme sequence to audio via ElevenLabs API.

    Args:
        phoneme_sequence: list of ARPABET strings, e.g. ['B', 'AH', 'T', 'ER']
        output_path: where to save audio file
        voice_id: ElevenLabs voice ID
        model_id: must be 'eleven_flash_v2' or 'eleven_english_v1' for phoneme tags

    Returns:
        audio_bytes: raw audio content
    """
    ssml_text = phonemes_to_ssml(phoneme_sequence)
    if not ssml_text:
        raise ValueError("Empty phoneme sequence")

    print(f"  SSML: {ssml_text[:200]}{'...' if len(ssml_text) > 200 else ''}")

    response = requests.post(
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "text": ssml_text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.75,
            },
        },
        timeout=30,
    )

    if response.status_code == 200:
        audio_bytes = response.content
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        print(f"  Audio saved: {output_path} ({len(audio_bytes)} bytes)")
        return audio_bytes
    else:
        raise RuntimeError(
            f"ElevenLabs API error {response.status_code}: {response.text}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phonemes", type=str, default=None,
                        help="Space-separated ARPABET phonemes")
    parser.add_argument("--output", type=str, default="brain2speech_test.mp3")
    args = parser.parse_args()

    if args.phonemes:
        phonemes = args.phonemes.split()
    else:
        # Test: "butter word" → B AH T ER SIL W ER D
        phonemes = ['B', 'AH', 'T', 'ER', 'SIL', 'W', 'ER', 'D']

    print(f"Phonemes: {' '.join(phonemes)}")
    print(f"SSML preview: {phonemes_to_ssml(phonemes)}")

    audio = synthesize_speech(phonemes, args.output)
    print(f"Success! Audio: {args.output} ({len(audio)} bytes)")


if __name__ == "__main__":
    main()
