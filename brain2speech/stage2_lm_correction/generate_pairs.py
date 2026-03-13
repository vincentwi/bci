#!/usr/bin/env python3
"""
Stage 2b: Generate synthetic noisy→clean phoneme pairs for LM fine-tuning.

Uses the confusion matrix (Stage 2a) as a noise channel model to corrupt
clean ARPABET sequences from the CMU Pronouncing Dictionary. Produces
JSONL files in chat format for SFTTrainer.

Outputs:
    brain2speech/data/phoneme_correction_train.jsonl  (80k pairs)
    brain2speech/data/phoneme_correction_val.jsonl    (10k pairs)
    brain2speech/data/phoneme_correction_test.jsonl   (10k pairs)
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR, CLASS_TO_ARPABET, ARPABET_TO_CLASS, N_CLASSES, SEED, SYSTEM_PROMPT,
)

np.random.seed(SEED)


# ── CMU Pronouncing Dictionary ──

def download_cmudict(output_path):
    """Download CMU Pronouncing Dictionary."""
    if output_path.exists():
        print(f"CMU dict already exists at {output_path}")
        return
    urls = [
        "https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b",
        "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict-0.7b",
    ]
    for url in urls:
        try:
            print(f"Downloading CMU dict from {url[:60]}...")
            urllib.request.urlretrieve(url, str(output_path))
            print("Done.")
            return
        except Exception as e:
            print(f"  Failed: {e}")
    raise RuntimeError("Could not download CMU dict from any source")


def parse_cmudict(path):
    """Parse CMU dict → {word: [phoneme_list]}. Strip stress markers."""
    entries = {}
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith(';;;') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].split('(')[0]  # Remove variant markers like WORD(2)
            # Strip stress: AH0→AH, AH1→AH, AH2→AH
            phonemes = [p.rstrip('012') for p in parts[1:]]
            if word not in entries:
                entries[word] = phonemes
    return entries


def filter_cmudict(cmudict, valid_phonemes):
    """Keep only entries whose phonemes are in our 39-phoneme set."""
    filtered = {}
    for word, phones in cmudict.items():
        if all(p in valid_phonemes for p in phones):
            filtered[word] = phones
    return filtered


# ── Corruption ──

def corrupt_sequence(clean_phonemes, confusion_matrix, arpabet_to_idx, idx_to_arpabet,
                     corruption_rate=0.4):
    """Apply confusion-matrix noise to a phoneme sequence."""
    noisy = []
    for p in clean_phonemes:
        if p == 'SIL':
            noisy.append('SIL')
            continue
        if p not in arpabet_to_idx:
            noisy.append(p)
            continue
        idx = arpabet_to_idx[p]
        if np.random.random() < corruption_rate:
            # Sample from this phoneme's confusion distribution
            noisy_idx = np.random.choice(len(confusion_matrix), p=confusion_matrix[idx])
            noisy.append(idx_to_arpabet[noisy_idx])
        else:
            noisy.append(p)
    return noisy


def _generate_chunk(args):
    """Worker function for parallel pair generation."""
    chunk_id, n_pairs, words, word_phonemes, confusion_matrix, seed = args
    rng = np.random.RandomState(seed)
    idx_to_arpabet = CLASS_TO_ARPABET
    arpabet_to_idx = ARPABET_TO_CLASS
    n_words_arr = len(words)

    pairs = []
    for _ in range(n_pairs):
        n_w = rng.randint(2, 7)
        chosen_idx = rng.choice(n_words_arr, n_w, replace=False)

        clean = []
        chosen_names = []
        for wi in chosen_idx:
            clean.extend(word_phonemes[wi])
            clean.append('SIL')
            chosen_names.append(words[wi])
        clean = clean[:-1]

        rate = rng.uniform(0.08, 0.50)
        noisy = []
        for p in clean:
            if p == 'SIL' or p not in arpabet_to_idx:
                noisy.append(p)
                continue
            idx = arpabet_to_idx[p]
            if rng.random() < rate:
                noisy_idx = rng.choice(len(confusion_matrix), p=confusion_matrix[idx])
                noisy.append(idx_to_arpabet[noisy_idx])
            else:
                noisy.append(p)

        pairs.append({
            "noisy": " ".join(noisy),
            "clean": " ".join(clean),
            "words": " ".join(chosen_names),
            "corruption_rate": float(rate),
            "n_phonemes": len(clean),
        })
    return pairs


def generate_pairs(cmudict, confusion_matrix, n_pairs=100000, n_workers=None):
    """Generate noisy/clean phoneme sequence pairs using multiprocessing."""
    from multiprocessing import Pool, cpu_count

    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    words = list(cmudict.keys())
    word_phonemes = [cmudict[w] for w in words]

    chunk_size = n_pairs // n_workers
    remainder = n_pairs % n_workers

    args_list = []
    for i in range(n_workers):
        n = chunk_size + (1 if i < remainder else 0)
        args_list.append((i, n, words, word_phonemes, confusion_matrix, SEED + i))

    print(f"  Generating {n_pairs} pairs across {n_workers} workers...")
    with Pool(n_workers) as pool:
        chunks = pool.map(_generate_chunk, args_list)

    pairs = []
    for chunk in chunks:
        pairs.extend(chunk)

    print(f"  Generated {len(pairs)} pairs total")
    return pairs


def format_as_chat(pair):
    """Format a pair as a chat message for SFTTrainer."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["noisy"]},
            {"role": "assistant", "content": pair["clean"]},
        ]
    }


def save_jsonl(data, path):
    """Save list of dicts as JSONL."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved {len(data)} examples to {path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 2b: Generating synthetic training data")
    print("=" * 60)

    # Download + parse CMU dict
    cmudict_path = DATA_DIR / "cmudict-0.7b"
    download_cmudict(cmudict_path)
    cmudict = parse_cmudict(cmudict_path)
    print(f"CMU dict: {len(cmudict)} entries")

    # Filter to our phoneme set
    valid_phonemes = set(CLASS_TO_ARPABET.values()) - {'SIL'}
    cmudict = filter_cmudict(cmudict, valid_phonemes)
    print(f"After filtering: {len(cmudict)} entries with valid phonemes")

    # Load confusion matrix
    noise_model_path = DATA_DIR / "noise_model.npy"
    if noise_model_path.exists():
        C = np.load(noise_model_path)
        print(f"Loaded confusion matrix: {C.shape}")
    else:
        print("WARNING: No noise model found. Using synthetic confusion matrix.")
        print("Run build_noise_model.py first for best results.")
        # Create a synthetic confusion matrix based on articulatory similarity
        C = _make_synthetic_confusion_matrix()

    # Generate pairs
    print("\nGenerating 100k noisy/clean pairs...")
    pairs = generate_pairs(cmudict, C, n_pairs=100000)

    # Shuffle and split
    np.random.shuffle(pairs)
    train_pairs = pairs[:80000]
    val_pairs = pairs[80000:90000]
    test_pairs = pairs[90000:]

    # Format as chat messages and save
    print("\nSaving JSONL files...")
    save_jsonl([format_as_chat(p) for p in train_pairs],
               DATA_DIR / "phoneme_correction_train.jsonl")
    save_jsonl([format_as_chat(p) for p in val_pairs],
               DATA_DIR / "phoneme_correction_val.jsonl")
    save_jsonl([format_as_chat(p) for p in test_pairs],
               DATA_DIR / "phoneme_correction_test.jsonl")

    # Also save raw pairs for evaluation
    save_jsonl(test_pairs, DATA_DIR / "phoneme_correction_test_raw.jsonl")

    # Stats
    avg_len = np.mean([p["n_phonemes"] for p in pairs])
    avg_rate = np.mean([p["corruption_rate"] for p in pairs])
    print(f"\nStats: avg sequence length = {avg_len:.1f} phonemes, "
          f"avg corruption rate = {avg_rate:.2f}")
    print("Done.")


def _make_synthetic_confusion_matrix():
    """Fallback: create confusion matrix from articulatory similarity priors."""
    C = np.eye(N_CLASSES) * 0.92

    # Add confusion between articulatory neighbors
    confusable = [
        ('B', 'P', 0.02), ('D', 'T', 0.02), ('G', 'K', 0.02),
        ('V', 'F', 0.02), ('Z', 'S', 0.02), ('ZH', 'SH', 0.02),
        ('DH', 'TH', 0.02), ('JH', 'CH', 0.02),
        ('B', 'M', 0.01), ('D', 'N', 0.01), ('G', 'NG', 0.01),
        ('IY', 'IH', 0.015), ('UW', 'UH', 0.015), ('AE', 'EH', 0.015),
        ('AA', 'AO', 0.015), ('AH', 'AE', 0.01),
    ]

    for p1, p2, rate in confusable:
        i, j = ARPABET_TO_CLASS.get(p1), ARPABET_TO_CLASS.get(p2)
        if i is not None and j is not None:
            C[i, j] += rate
            C[j, i] += rate

    # Normalize rows
    C = C / C.sum(axis=1, keepdims=True)
    return C


if __name__ == "__main__":
    main()
