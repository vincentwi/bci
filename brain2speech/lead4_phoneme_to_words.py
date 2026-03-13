#!/usr/bin/env python3
"""Phoneme-to-word conversion via CMU dictionary reverse lookup.

Splits phoneme sequences at SIL tokens, then looks up each word-chunk
in a reverse CMU dict (phoneme tuple → word).

References:
    - Source 9 (tbenst): uses cmudict.txt for reverse mapping
    - Source 8 (DCoND): SIL tokens mark word boundaries;
      "translate each subgroup of phonemes enclosed by two SIL symbols
       into one single word"

Usage:
    from lead4_phoneme_to_words import PhonemeToWordConverter
    converter = PhonemeToWordConverter()
    text = converter.convert("B AH T ER SIL W ER D")
    # → "butter word"
"""
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR


CMUDICT_URLS = [
    "https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b",
    "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict-0.7b",
]

# Known local locations
CMUDICT_LOCAL_PATHS = [
    Path("/mnt/home/vincent.wilmet/brain2speech/data/cmudict-0.7b"),
]


def download_cmudict(output_path=None):
    """Download CMU Pronouncing Dictionary if not present."""
    if output_path is None:
        output_path = DATA_DIR / "cmudict-0.7b"
    output_path = Path(output_path)
    if output_path.exists():
        return str(output_path)

    # Check known local paths
    for local in CMUDICT_LOCAL_PATHS:
        if local.exists():
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(local), str(output_path))
            print(f"Copied CMU dict from {local}")
            return str(output_path)

    # Try downloading
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for url in CMUDICT_URLS:
        try:
            print(f"Downloading CMU dict from {url[:60]}...")
            urllib.request.urlretrieve(url, str(output_path))
            print(f"Downloaded: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"  Failed: {e}")

    raise RuntimeError("Could not find or download CMU dict")


def load_cmudict(path=None):
    """Load CMU dict → (word_to_phones, phone_to_words).

    Returns:
        word_to_phones: {WORD: [tuple of phonemes]}  (may have multiple pronunciations)
        phone_to_words: {tuple of phonemes: [WORD, ...]}
    """
    if path is None:
        path = download_cmudict()

    word_to_phones = defaultdict(list)
    phone_to_words = defaultdict(list)

    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;;'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # Remove variant number: WORD(2) → WORD
            word = re.sub(r'\(\d+\)$', '', parts[0]).upper()
            # Strip stress markers from vowels: AH1 → AH
            phones = tuple(re.sub(r'\d', '', p) for p in parts[1:])
            word_to_phones[word].append(phones)
            phone_to_words[phones].append(word)

    print(f"Loaded CMU dict: {len(word_to_phones)} words, "
          f"{len(phone_to_words)} unique pronunciations")
    return word_to_phones, phone_to_words


class PhonemeToWordConverter:
    """Convert phoneme sequences to words using CMU dict reverse lookup."""

    def __init__(self, cmudict_path=None):
        self.word_to_phones, self.phone_to_words = load_cmudict(cmudict_path)

    def convert(self, phoneme_str_or_list):
        """Convert phoneme sequence to text.

        Args:
            phoneme_str_or_list: Either a space-separated string of ARPABET
                phonemes (with SIL as word boundaries) or a list of phoneme strings.

        Returns:
            Lowercase text string.
        """
        if isinstance(phoneme_str_or_list, str):
            phones = phoneme_str_or_list.strip().split()
        else:
            phones = list(phoneme_str_or_list)

        return self._phonemes_to_text(phones)

    def _phonemes_to_text(self, phone_list):
        """Split at SIL tokens, look up each word-chunk.

        Source 8 (DCoND): "translate each subgroup of phonemes enclosed by
        two SIL symbols into one single word"
        """
        words = []
        current_chunk = []

        for p in phone_list:
            if p == 'SIL':
                if current_chunk:
                    matched = self._match_chunk(current_chunk)
                    words.extend(matched)
                    current_chunk = []
            else:
                current_chunk.append(p)

        # Handle trailing chunk (no final SIL)
        if current_chunk:
            matched = self._match_chunk(current_chunk)
            words.extend(matched)

        return ' '.join(words)

    def _match_chunk(self, phones):
        """Match a phoneme chunk to word(s).

        Strategy:
        1. Try exact match first
        2. Fall back to greedy longest-prefix matching
        """
        key = tuple(phones)
        if key in self.phone_to_words:
            return [self.phone_to_words[key][0].lower()]

        # Greedy longest-prefix match
        return self._longest_match(phones)

    def _longest_match(self, phones):
        """Greedy longest-first matching for multi-word chunks."""
        words = []
        i = 0
        while i < len(phones):
            best_len, best_word = 0, None
            # Try longest possible match first (max word length ~15 phonemes)
            max_end = min(i + 15, len(phones))
            for end in range(max_end, i, -1):
                key = tuple(phones[i:end])
                if key in self.phone_to_words:
                    best_len = end - i
                    best_word = self.phone_to_words[key][0].lower()
                    break

            if best_word:
                words.append(best_word)
                i += best_len
            else:
                # Skip unmatched phoneme (or could try single-phoneme words)
                i += 1

        return words

    def words_to_phonemes(self, text):
        """Convert text to phoneme sequence (for reference/evaluation).

        Args:
            text: Lowercase text string

        Returns:
            List of ARPABET phoneme strings with SIL between words
        """
        words = text.upper().split()
        phonemes = []
        for i, word in enumerate(words):
            word = re.sub(r'[^\w]', '', word)
            if word in self.word_to_phones:
                if i > 0:
                    phonemes.append('SIL')
                phonemes.extend(self.word_to_phones[word][0])
            # Skip unknown words
        return phonemes


def main():
    """Test phoneme-to-word conversion."""
    converter = PhonemeToWordConverter()

    test_cases = [
        "B AH T ER SIL W ER D",
        "DH AH SIL K AE T SIL S AE T SIL AA N SIL DH AH SIL M AE T",
        "HH EH L OW SIL W ER L D",
        "AY SIL W AA N T SIL T UW SIL G OW SIL HH OW M",
    ]

    print("Phoneme-to-Word Conversion Tests:")
    print("=" * 60)
    for phonemes in test_cases:
        text = converter.convert(phonemes)
        print(f"  {phonemes}")
        print(f"  → {text}")
        print()


if __name__ == '__main__':
    main()
