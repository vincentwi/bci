#!/usr/bin/env python3
"""
Lead 2: OPT-6.7B N-best rescoring for DCoND decoder.

Pipeline:
  1. DCoND → marginalized monophone log probs
  2. KenLM beam search → top-100 phoneme hypotheses
  3. Phoneme→word conversion via CMUdict reverse lookup
  4. OPT-6.7B scores each word-level candidate by log-probability
  5. Re-rank by: ctc_score + lm_weight * opt_score

Sources:
  - fwillett/speechBCI LanguageModelDecoder/
  - arxiv 2411.10657 Section 3.3
  - arxiv 2412.17227: OPT-6.7B rescoring universal across top teams

Usage:
    # Rescore N-best list from KenLM beam search
    CUDA_VISIBLE_DEVICES=6,7 python brain2speech/lead2_rescore_opt.py \
        --nbest brain2speech/results/L2_kenlm_nbest.json \
        --opt-weight 0.5

    # Build CMUdict trie
    python brain2speech/lead2_rescore_opt.py --build-cmudict

    # Sweep OPT weight
    CUDA_VISIBLE_DEVICES=6,7 python brain2speech/lead2_rescore_opt.py \
        --nbest brain2speech/results/L2_kenlm_nbest.json \
        --sweep-weights 0.1,0.2,0.3,0.5,1.0
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CLASS_TO_ARPABET, ARPABET_TO_CLASS, N_CLASSES

DATA_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/data")
RESULTS_DIR = Path("/mnt/home/vincent.wilmet/brain2speech/results")


# ═══════════════════════════════════════════════════════════════════════
# CMUDICT PHONEME→WORD CONVERSION
# ═══════════════════════════════════════════════════════════════════════

class CMUDictTrie:
    """Reverse trie for phoneme→word lookup using CMU Pronouncing Dictionary.

    Builds a trie where keys are phoneme sequences and values are words.
    Used to convert decoded phoneme sequences back to English words.
    """

    def __init__(self):
        self.trie = {}  # nested dict, leaves have '_words' key
        self.word_to_phones = {}  # word → list of phoneme lists

    def load_cmudict(self):
        """Load CMU Pronouncing Dictionary.

        Tries nltk first, falls back to downloading directly.
        Strips stress markers from vowels (AH0→AH, IY1→IY).
        """
        entries = []

        try:
            import nltk
            try:
                cmudict = nltk.corpus.cmudict.dict()
            except LookupError:
                nltk.download('cmudict', quiet=True)
                cmudict = nltk.corpus.cmudict.dict()

            for word, pronunciations in cmudict.items():
                for pron in pronunciations:
                    # Strip stress markers
                    clean = [re.sub(r'\d+$', '', p) for p in pron]
                    entries.append((word.lower(), clean))

        except ImportError:
            print("nltk not available, downloading CMUdict directly...")
            import urllib.request
            url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
            response = urllib.request.urlopen(url)
            for line in response:
                line = line.decode('utf-8').strip()
                if not line or line.startswith(';;;'):
                    continue
                parts = line.split()
                word = parts[0].lower()
                # Remove variant marker (e.g., "WORD(2)")
                word = re.sub(r'\(\d+\)$', '', word)
                pron = [re.sub(r'\d+$', '', p) for p in parts[1:]]
                entries.append((word, pron))

        # Map our ARPABET symbols to CMUdict symbols
        our_to_cmu = {}
        for idx, symbol in CLASS_TO_ARPABET.items():
            our_to_cmu[symbol] = symbol  # Most are the same

        # Build trie
        n_added = 0
        for word, pron in entries:
            # Filter to only phonemes in our vocabulary
            mapped = []
            valid = True
            for p in pron:
                if p in ARPABET_TO_CLASS:
                    mapped.append(p)
                else:
                    valid = False
                    break
            if not valid or not mapped:
                continue

            # Add to trie
            node = self.trie
            for phone in mapped:
                if phone not in node:
                    node[phone] = {}
                node = node[phone]
            if '_words' not in node:
                node['_words'] = []
            node['_words'].append(word)

            # Store forward mapping too
            if word not in self.word_to_phones:
                self.word_to_phones[word] = []
            self.word_to_phones[word].append(mapped)
            n_added += 1

        print(f"CMUDict trie built: {n_added} entries, "
              f"{len(self.word_to_phones)} unique words")

    def lookup(self, phoneme_seq):
        """Look up a phoneme sequence in the trie.

        Returns list of matching words, or empty list if no match.
        """
        node = self.trie
        for phone in phoneme_seq:
            if phone not in node:
                return []
            node = node[phone]
        return node.get('_words', [])

    def phonemes_to_words(self, phoneme_seq, sil_token='SIL'):
        """Convert a full phoneme sequence to word sequence.

        Uses greedy longest-match with SIL as word boundary.
        Returns list of words.
        """
        # Split on SIL tokens
        segments = []
        current = []
        for p in phoneme_seq:
            if p == sil_token:
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(p)
        if current:
            segments.append(current)

        words = []
        for segment in segments:
            matched = self._greedy_match(segment)
            words.extend(matched)

        return words

    def _greedy_match(self, phones):
        """Greedy longest-match phoneme→word conversion for one segment."""
        if not phones:
            return []

        words = []
        i = 0
        while i < len(phones):
            # Try longest match first
            best_word = None
            best_len = 0

            for end in range(min(i + 15, len(phones)), i, -1):
                subseq = phones[i:end]
                matches = self.lookup(subseq)
                if matches:
                    best_word = matches[0]  # Take first match
                    best_len = end - i
                    break

            if best_word:
                words.append(best_word)
                i += best_len
            else:
                # No match found — skip this phoneme
                i += 1

        return words

    def save(self, path):
        """Save trie to pickle."""
        with open(path, 'wb') as f:
            pickle.dump({'trie': self.trie,
                         'word_to_phones': self.word_to_phones}, f)
        print(f"CMUDict trie saved: {path}")

    def load(self, path):
        """Load trie from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.trie = data['trie']
        self.word_to_phones = data['word_to_phones']
        print(f"CMUDict trie loaded: {len(self.word_to_phones)} words")


# ═══════════════════════════════════════════════════════════════════════
# OPT-6.7B RESCORER
# ═══════════════════════════════════════════════════════════════════════

class OPTRescorer:
    """OPT-6.7B sequence log-probability rescorer.

    Scores text candidates by their log-probability under OPT-6.7B.
    Used to re-rank N-best hypotheses from KenLM beam search.

    Source: fwillett/speechBCI LanguageModelDecoder
    """

    def __init__(self, model_name='facebook/opt-6.7b', device='cuda'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map='auto')
        self.model.eval()
        print(f"  Model loaded on {self.model.device}")

    def score(self, text):
        """Compute sequence log-probability under OPT-6.7B.

        Returns normalized log-prob (per token) to avoid length bias.
        """
        import torch

        if not text.strip():
            return -float('inf')

        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs.input_ids.to(self.model.device)

        if input_ids.shape[1] < 2:
            return -float('inf')

        with torch.no_grad():
            outputs = self.model(input_ids)
            log_probs = outputs.logits.log_softmax(dim=-1)
            # Score each token given prefix
            token_scores = log_probs[0, :-1].gather(
                1, input_ids[0, 1:].unsqueeze(1)).squeeze(1)
            # Normalized by length
            return token_scores.sum().item() / token_scores.shape[0]

    def score_batch(self, texts, max_len=128):
        """Score multiple texts in a batch for efficiency."""
        import torch

        scores = []
        for text in texts:
            scores.append(self.score(text))
        return scores

    def rescore_nbest(self, candidates, ctc_scores, opt_weight=0.5):
        """Re-rank N-best list using OPT-6.7B scores.

        Final score = ctc_score + opt_weight * opt_score

        Args:
            candidates: list of text strings
            ctc_scores: list of CTC+LM scores from beam search
            opt_weight: weight for OPT score

        Returns: list of (text, combined_score, opt_score) sorted by score
        """
        opt_scores = []
        for text in candidates:
            s = self.score(text)
            opt_scores.append(s)

        results = []
        for text, ctc_s, opt_s in zip(candidates, ctc_scores, opt_scores):
            combined = ctc_s + opt_weight * opt_s
            results.append({
                'text': text,
                'ctc_score': ctc_s,
                'opt_score': opt_s,
                'combined_score': combined,
            })

        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Lead 2: OPT-6.7B N-best rescoring')
    parser.add_argument('--nbest', type=str, default=None,
                        help='Path to N-best JSON from lead2_decode_kenlm.py')
    parser.add_argument('--opt-model', type=str, default='facebook/opt-6.7b')
    parser.add_argument('--opt-weight', type=float, default=0.5,
                        help='OPT rescoring weight')
    parser.add_argument('--sweep-weights', type=str, default=None,
                        help='Comma-separated OPT weights to sweep')
    parser.add_argument('--build-cmudict', action='store_true',
                        help='Build and save CMUdict trie')
    parser.add_argument('--cmudict', type=str,
                        default=str(DATA_DIR / 'cmudict_trie.pkl'),
                        help='Path to CMUdict trie pickle')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON for rescored results')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.build_cmudict:
        trie = CMUDictTrie()
        trie.load_cmudict()
        trie.save(args.cmudict)

        # Test
        test_words = ['the', 'cat', 'sat', 'hello', 'world']
        print("\nTest lookups:")
        for w in test_words:
            if w in trie.word_to_phones:
                phones = trie.word_to_phones[w][0]
                print(f"  {w} → {' '.join(phones)}")
                # Reverse lookup
                found = trie.lookup(phones)
                print(f"    → {found}")
        return

    if not args.nbest:
        print("ERROR: --nbest is required (or use --build-cmudict)")
        sys.exit(1)

    # Load N-best results
    print(f"Loading N-best: {args.nbest}")
    with open(args.nbest) as f:
        nbest_data = json.load(f)

    # Load CMUdict trie
    trie = CMUDictTrie()
    if os.path.exists(args.cmudict):
        trie.load(args.cmudict)
    else:
        print(f"CMUdict not found at {args.cmudict}, building...")
        trie.load_cmudict()
        trie.save(args.cmudict)

    # Convert phoneme hypotheses to words
    print("\nConverting phonemes to words...")
    for split_name, trials in nbest_data.items():
        for trial in trials:
            for hyp in trial.get('nbest', []):
                phone_str = hyp.get('text', '')
                if phone_str:
                    phones = phone_str.strip().split()
                    words = trie.phonemes_to_words(phones)
                    hyp['word_text'] = ' '.join(words)
                else:
                    hyp['word_text'] = ''

    # Load OPT-6.7B
    print(f"\nLoading rescorer: {args.opt_model}")
    rescorer = OPTRescorer(args.opt_model)

    # Rescore
    weights = [args.opt_weight]
    if args.sweep_weights:
        weights = [float(w) for w in args.sweep_weights.split(',')]

    for opt_weight in weights:
        print(f"\n{'=' * 60}")
        print(f"OPT weight: {opt_weight}")
        print(f"{'=' * 60}")

        for split_name, trials in nbest_data.items():
            correct = 0
            total = 0
            t0 = time.time()

            for trial_idx, trial in enumerate(trials):
                candidates = []
                ctc_scores = []
                for hyp in trial.get('nbest', []):
                    word_text = hyp.get('word_text', '')
                    if word_text:
                        candidates.append(word_text)
                        ctc_scores.append(hyp.get('combined_score', 0.0))

                if not candidates:
                    continue

                rescored = rescorer.rescore_nbest(
                    candidates, ctc_scores, opt_weight=opt_weight)

                trial['rescored_best'] = rescored[0] if rescored else None
                total += 1

                if (trial_idx + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"    {trial_idx+1}/{len(trials)} "
                          f"({elapsed:.1f}s)")

            elapsed = time.time() - t0
            print(f"\n  {split_name}: {total} trials rescored ({elapsed:.1f}s)")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(nbest_data, f, indent=2)
        print(f"\nRescored results: {output_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
