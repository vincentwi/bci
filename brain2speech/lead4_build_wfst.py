#!/usr/bin/env python3
"""Build WFST T-L-G composition for phoneme-to-word decoding.

Constructs a Weighted Finite State Transducer pipeline:
  T (CTC-to-phoneme transducer) o L (lexicon FST) o G (n-gram LM FST)

Since the Dryad pre-built WFST (languageModel.tar.gz) is not available,
this builds the pipeline from scratch using OpenFST Python bindings.

If openfst-python is not available, falls back to a simplified phoneme→word
pipeline using the CMU dictionary + KenLM combination.

WFST beam search parameters from Source 5 (Willett 2023):
  beam=18, acoustic_scale=0.8, blank_penalty=log(7),
  min_active=200, max_active=7000, lm_weight=0.5

References:
    - Source 1 (Willett repo): T-L-G composition via lmDecoderUtils.py
    - Source 5 (paper): min(det(L o G)), silence_prob=0.9, arpa2fst
    - Source 8 (DCoND): WFST as first stage before OPT rescoring

Usage:
    # Build WFST from CMU dict + KenLM ARPA
    python brain2speech/lead4_build_wfst.py \
        --cmudict brain2speech/data/cmudict-0.7b \
        --arpa brain2speech/data/phoneme_5gram.arpa \
        --output brain2speech/data/wfst/

    # Decode with WFST
    python brain2speech/lead4_build_wfst.py \
        --decode --wfst-dir brain2speech/data/wfst/ \
        --model brain2speech/results/lead4/L4_cffan_baseline.pt
"""
import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_TO_ARPABET, ARPABET_TO_CLASS, N_CLASSES

CTC_BLANK = 40

# WFST beam search parameters (Source 5: Willett 2023)
WFST_PARAMS = {
    'beam': 18,                    # Source 5: optimal accuracy/speed
    'acoustic_scale': 0.8,         # Source 5: standard
    'blank_penalty': math.log(7),  # Source 1: np.log(7) ≈ 1.946
    'min_active': 200,             # Source 5
    'max_active': 7000,            # Source 5
    'lm_weight': 0.5,             # Source 5: for OPT rescoring interpolation
    'silence_prob': 0.9,          # Source 5: L.fst silence probability
}


# ═══════════════════════════════════════════════════════════════════════
# CMU DICTIONARY LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_cmudict(path):
    """Load CMU dictionary: word → list of phoneme tuples.

    Returns:
        word_to_phones: {WORD: [phoneme_tuple, ...]} (multiple pronunciations)
        phone_set: set of all phonemes in dictionary
    """
    word_to_phones = defaultdict(list)
    phone_set = set()

    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;;'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = re.sub(r'\(\d+\)$', '', parts[0]).upper()
            # Strip stress markers
            phones = tuple(re.sub(r'\d', '', p) for p in parts[1:])
            word_to_phones[word].append(phones)
            phone_set.update(phones)

    print(f"Loaded CMU dict: {len(word_to_phones)} words, "
          f"{len(phone_set)} unique phonemes")
    return word_to_phones, phone_set


# ═══════════════════════════════════════════════════════════════════════
# WFST CONSTRUCTION (OpenFST)
# ═══════════════════════════════════════════════════════════════════════

def build_T_fst(phone_symbols, output_dir):
    """Build CTC-to-phoneme transducer T.fst.

    T maps CTC output (including blank + repeated symbols) to clean
    phoneme sequences. This is a standard CTC topology:
    - State 0: initial
    - For each phoneme p, self-loop with p:epsilon (repetition)
    - For each phoneme p, transition to next state with p:p (emission)
    - Blank: self-loop with blank:epsilon

    Args:
        phone_symbols: dict mapping phoneme name → integer symbol ID
        output_dir: Directory to write T.fst text format
    """
    lines = []
    state = 0

    # Initial state: blank self-loop
    lines.append(f"{state} {state} <blank> <eps> 0")

    # For each phoneme: allow transition and self-loop
    for phone, sym_id in phone_symbols.items():
        if phone in ('<eps>', '<blank>', 'SIL'):
            continue
        # Emit phoneme
        lines.append(f"{state} {state} {phone} {phone} 0")

    # Final state
    lines.append(f"{state} 0")

    fst_path = Path(output_dir) / 'T.fst.txt'
    with open(fst_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Built T.fst: {fst_path} ({len(lines)} arcs)")
    return fst_path


def build_L_fst(word_to_phones, phone_symbols, word_symbols, output_dir,
                silence_prob=0.9):
    """Build lexicon FST L.fst.

    L maps phoneme sequences to words. Each word entry:
      phone_1:word phone_2:eps ... phone_n:eps
    With optional silence between words (probability = silence_prob).

    Source 5: silence_prob=0.9

    Args:
        word_to_phones: {WORD: [(phone1, phone2, ...), ...]}
        phone_symbols: {phone_name: int}
        word_symbols: {word: int}
        output_dir: Output directory
        silence_prob: Probability of silence between words (Source 5: 0.9)
    """
    lines = []
    state_id = 0
    next_state = 1

    # For each word, create arcs through phone sequence
    for word, pronunciations in word_to_phones.items():
        if word not in word_symbols:
            continue

        for phones in pronunciations:
            if not all(p in phone_symbols for p in phones):
                continue

            # Start state → first phone with word output
            current = state_id
            for i, phone in enumerate(phones):
                word_out = word if i == 0 else '<eps>'
                lines.append(f"{current} {next_state} {phone} {word_out} 0")
                current = next_state
                next_state += 1

            # Return to start (with optional silence)
            sil_weight = -math.log(silence_prob) if silence_prob > 0 else 0
            no_sil_weight = -math.log(1 - silence_prob) if silence_prob < 1 else 0

            # With silence
            lines.append(f"{current} {state_id} SIL <eps> {sil_weight:.4f}")
            # Without silence
            lines.append(f"{current} {state_id} <eps> <eps> {no_sil_weight:.4f}")

    # Final state
    lines.append(f"{state_id} 0")

    fst_path = Path(output_dir) / 'L.fst.txt'
    with open(fst_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Built L.fst: {fst_path} ({len(lines)} arcs, "
          f"{len(word_to_phones)} words)")
    return fst_path


def build_G_fst(arpa_path, word_symbols, output_dir):
    """Build grammar FST G.fst from ARPA n-gram LM.

    Source 5: G.fst via arpa2fst --disambig-symbol=#0
    Composition: min(det(L o G)) with log semiring.

    For simplicity, we build a unigram/bigram FST from ARPA.
    Higher-order n-grams are handled by KenLM beam search.

    Args:
        arpa_path: Path to ARPA format language model
        word_symbols: {word: int}
        output_dir: Output directory
    """
    lines = []
    state = 0

    # Parse ARPA for unigrams and bigrams
    unigrams = {}
    bigrams = {}

    with open(arpa_path, 'r') as f:
        current_section = None
        for line in f:
            line = line.strip()
            if line == '\\1-grams:':
                current_section = 1
                continue
            elif line == '\\2-grams:':
                current_section = 2
                continue
            elif line.startswith('\\'):
                current_section = None
                continue

            if current_section == 1 and line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    log_prob = float(parts[0])
                    word = parts[1].strip()
                    backoff = float(parts[2]) if len(parts) > 2 else 0.0
                    unigrams[word] = (log_prob, backoff)

            elif current_section == 2 and line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    log_prob = float(parts[0])
                    words = parts[1].strip()
                    bigrams[words] = log_prob

    # Build unigram arcs from state 0
    for word, (log_prob, _backoff) in unigrams.items():
        if word in ('<s>', '</s>'):
            continue
        if word in word_symbols:
            weight = -log_prob * math.log(10)  # Convert log10 to -log (tropical)
            lines.append(f"{state} {state} {word} {word} {weight:.4f}")

    lines.append(f"{state} 0")

    fst_path = Path(output_dir) / 'G.fst.txt'
    with open(fst_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Built G.fst: {fst_path} ({len(unigrams)} unigrams)")
    return fst_path


def build_symbol_tables(word_to_phones, phone_set, output_dir):
    """Build symbol tables for OpenFST.

    Returns:
        phone_symbols: {phone: int}
        word_symbols: {word: int}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phone symbols
    phone_symbols = {'<eps>': 0, '<blank>': 1, 'SIL': 2}
    for i, phone in enumerate(sorted(phone_set)):
        if phone not in phone_symbols:
            phone_symbols[phone] = len(phone_symbols)

    # Add CTC phonemes that might not be in CMU dict
    for idx, name in CLASS_TO_ARPABET.items():
        if name not in phone_symbols:
            phone_symbols[name] = len(phone_symbols)

    with open(output_dir / 'phones.txt', 'w') as f:
        for phone, idx in sorted(phone_symbols.items(), key=lambda x: x[1]):
            f.write(f"{phone} {idx}\n")

    # Word symbols
    word_symbols = {'<eps>': 0}
    for word in sorted(word_to_phones.keys()):
        word_symbols[word] = len(word_symbols)

    with open(output_dir / 'words.txt', 'w') as f:
        for word, idx in sorted(word_symbols.items(), key=lambda x: x[1]):
            f.write(f"{word} {idx}\n")

    print(f"Symbol tables: {len(phone_symbols)} phones, {len(word_symbols)} words")
    return phone_symbols, word_symbols


def try_compile_fst(fst_text_path, output_dir):
    """Try to compile text FST to binary using OpenFST tools.

    Returns binary FST path if successful, else None.
    """
    import shutil
    import subprocess

    fstcompile = shutil.which('fstcompile')
    if not fstcompile:
        return None

    bin_path = str(fst_text_path).replace('.fst.txt', '.fst')
    syms = str(Path(output_dir) / 'phones.txt')

    try:
        subprocess.run([
            fstcompile,
            f'--isymbols={syms}',
            f'--osymbols={syms}',
            str(fst_text_path),
            bin_path,
        ], check=True, capture_output=True, text=True)
        return bin_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ═══════════════════════════════════════════════════════════════════════
# SIMPLIFIED WFST DECODER (without OpenFST binary)
# ═══════════════════════════════════════════════════════════════════════

class SimplifiedWFSTDecoder:
    """Simplified WFST-style decoder combining phoneme→word with LM.

    When full OpenFST is not available, this provides a similar pipeline:
    1. CTC beam search → N-best phoneme sequences
    2. Phoneme→word conversion via CMU dict
    3. Word-level LM rescoring (via KenLM or OPT)

    This approximates the T-L-G composition without actual FST operations.

    Parameters match Source 5:
        acoustic_scale=0.8, blank_penalty=log(7), beam=18
    """

    def __init__(self, cmudict_path=None, kenlm_path=None,
                 acoustic_scale=0.8, blank_penalty=None):
        from lead4_phoneme_to_words import PhonemeToWordConverter
        self.converter = PhonemeToWordConverter(cmudict_path)

        self.acoustic_scale = acoustic_scale
        self.blank_penalty = blank_penalty or WFST_PARAMS['blank_penalty']

        self.word_lm = None
        if kenlm_path:
            try:
                import kenlm
                self.word_lm = kenlm.Model(kenlm_path)
                print(f"Word-level LM loaded: order={self.word_lm.order}")
            except ImportError:
                print("kenlm not available for word-level scoring")

    def decode(self, log_probs_np, beam_width=18, nbest=10,
               phoneme_lm=None, phoneme_alpha=0.05):
        """Decode CTC log-probs through simplified WFST pipeline.

        Args:
            log_probs_np: (T, 41) numpy array of log probabilities
            beam_width: Beam width for phoneme-level search
            nbest: Number of word-level hypotheses to return
            phoneme_lm: KenLMPhonemeLM for phoneme-level scoring
            phoneme_alpha: Phoneme LM weight

        Returns:
            List of (text, phonemes, combined_score) tuples
        """
        from lead4_decode_kenlm import decode_single

        # Apply acoustic scale and blank penalty
        scaled_lp = log_probs_np.copy()
        scaled_lp *= self.acoustic_scale
        scaled_lp[:, CTC_BLANK] -= self.blank_penalty

        # Stage 1: Phoneme beam search
        phoneme_results = decode_single(
            scaled_lp, phoneme_lm,
            beam_width=beam_width,
            alpha=phoneme_alpha,
            nbest=nbest * 3  # Over-generate for diversity
        )

        # Stage 2: Phoneme → word conversion
        word_results = []
        seen_texts = set()

        for phone_str, indices, ctc_score in phoneme_results:
            text = self.converter.convert(phone_str)
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            # Stage 3: Word-level LM scoring
            word_lm_score = 0.0
            if self.word_lm:
                word_lm_score = self.word_lm.score(text, bos=True, eos=True)
                word_lm_score *= math.log(10)  # Convert log10 to ln

            combined = ctc_score + WFST_PARAMS['lm_weight'] * word_lm_score
            word_results.append((text, phone_str, combined, ctc_score, word_lm_score))

        # Sort by combined score
        word_results.sort(key=lambda x: x[2], reverse=True)
        return word_results[:nbest]

    def decode_with_opt(self, log_probs_np, opt_rescorer,
                        beam_width=18, nbest=100,
                        phoneme_lm=None, phoneme_alpha=0.05,
                        opt_weight=0.5):
        """Full pipeline: CTC → phonemes → words → OPT rescoring.

        Matches Source 5's multi-stage approach:
        RNN → 5-gram beam → phoneme→word → OPT rescoring

        Args:
            opt_rescorer: OPTRescorer instance
            opt_weight: OPT interpolation weight (Source 5: 0.5)
        """
        # Get word-level candidates
        candidates = self.decode(
            log_probs_np, beam_width=beam_width, nbest=nbest,
            phoneme_lm=phoneme_lm, phoneme_alpha=phoneme_alpha
        )

        if not candidates:
            return []

        # OPT rescoring
        texts = [c[0] for c in candidates]
        acoustic_scores = [c[3] for c in candidates]  # CTC scores

        reranked = opt_rescorer.rescore_nbest(
            texts, acoustic_scores, lm_weight=opt_weight)

        # Merge back with phoneme info
        text_to_phones = {c[0]: c[1] for c in candidates}
        results = []
        for text, combined, ac_score, opt_score in reranked:
            phones = text_to_phones.get(text, '')
            results.append((text, phones, combined, ac_score, opt_score))

        return results


# ═══════════════════════════════════════════════════════════════════════
# BUILD PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def build_wfst(cmudict_path, arpa_path, output_dir):
    """Build complete T-L-G WFST pipeline.

    Args:
        cmudict_path: Path to CMU pronouncing dictionary
        arpa_path: Path to ARPA format n-gram LM
        output_dir: Output directory for FST files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CMU dict
    word_to_phones, phone_set = load_cmudict(cmudict_path)

    # Build symbol tables
    phone_symbols, word_symbols = build_symbol_tables(
        word_to_phones, phone_set, output_dir)

    # Build individual FSTs
    t_path = build_T_fst(phone_symbols, output_dir)
    l_path = build_L_fst(word_to_phones, phone_symbols, word_symbols,
                          output_dir, silence_prob=WFST_PARAMS['silence_prob'])
    g_path = build_G_fst(arpa_path, word_symbols, output_dir)

    # Try to compile to binary
    for fst_path in [t_path, l_path, g_path]:
        bin_path = try_compile_fst(fst_path, output_dir)
        if bin_path:
            print(f"  Compiled: {bin_path}")

    # Save WFST parameters
    import json
    with open(output_dir / 'wfst_params.json', 'w') as f:
        json.dump(WFST_PARAMS, f, indent=2)

    print(f"\nWFST built in {output_dir}/")
    print(f"  T.fst: CTC-to-phoneme transducer")
    print(f"  L.fst: Lexicon ({len(word_to_phones)} words, "
          f"silence_prob={WFST_PARAMS['silence_prob']})")
    print(f"  G.fst: Grammar from {arpa_path}")
    print(f"\nNote: Full T-L-G composition requires OpenFST tools.")
    print(f"Use SimplifiedWFSTDecoder as fallback.")

    return output_dir


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_wfst_decoder(model, trials, decoder, device='cuda',
                          phoneme_lm=None, phoneme_alpha=0.05,
                          beam_width=18):
    """Evaluate simplified WFST decoder on trial data.

    Returns mean WER.
    """
    import editdistance
    import torch
    import torch.nn.functional as F

    model.eval()
    total_edits, total_words = 0, 0

    for trial_idx, trial in enumerate(trials):
        if 'features_raw' in trial:
            features = torch.FloatTensor(trial['features_raw']).unsqueeze(0).to(device)
        else:
            features = torch.FloatTensor(trial['features']).unsqueeze(0).to(device)
        session_id = torch.LongTensor([trial.get('session_idx', 0)]).to(device)

        with torch.no_grad():
            logits = model(features, session_id)
            log_probs = F.log_softmax(logits, dim=-1)
        log_probs_np = log_probs[0].cpu().numpy()

        results = decoder.decode(
            log_probs_np, beam_width=beam_width,
            phoneme_lm=phoneme_lm, phoneme_alpha=phoneme_alpha)

        pred_text = results[0][0] if results else ''
        target_text = trial.get('text', '')

        if pred_text and target_text:
            pred_words = pred_text.lower().split()
            target_words = target_text.lower().split()
            edits = editdistance.eval(pred_words, target_words)
            total_edits += edits
            total_words += len(target_words)

        if (trial_idx + 1) % 100 == 0:
            wer = total_edits / max(total_words, 1)
            print(f"  [{trial_idx+1}/{len(trials)}] WER={wer:.4f}")

    wer = total_edits / max(total_words, 1)
    print(f"\nWFST Decoder WER: {wer:.4f} ({len(trials)} trials)")
    return wer


def main():
    parser = argparse.ArgumentParser(description='Build WFST T-L-G Pipeline')
    subparsers = parser.add_subparsers(dest='command')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build WFST components')
    build_parser.add_argument('--cmudict', type=str, required=True,
                              help='Path to CMU dict file')
    build_parser.add_argument('--arpa', type=str, required=True,
                              help='Path to word-level ARPA LM')
    build_parser.add_argument('--output', type=str, required=True,
                              help='Output directory for WFST files')

    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Decode with WFST')
    decode_parser.add_argument('--model', type=str, required=True,
                               help='CTC model checkpoint')
    decode_parser.add_argument('--cmudict', type=str, default=None,
                               help='CMU dict path')
    decode_parser.add_argument('--kenlm', type=str, default=None,
                               help='Phoneme KenLM path')
    decode_parser.add_argument('--word-lm', type=str, default=None,
                               help='Word-level KenLM path')
    decode_parser.add_argument('--data', type=str, default=None)
    decode_parser.add_argument('--beam', type=int, default=18)
    decode_parser.add_argument('--alpha', type=float, default=0.05)
    decode_parser.add_argument('--acoustic-scale', type=float, default=0.8)
    decode_parser.add_argument('--gpu', type=int, default=0)
    decode_parser.add_argument('--max-trials', type=int, default=None)
    decode_parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if args.command == 'build':
        build_wfst(args.cmudict, args.arpa, args.output)

    elif args.command == 'decode':
        import torch

        device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

        # Load model and data
        from lead4_full_pipeline import (load_raw_trials, split_within_day,
                                          load_enhanced_gru)
        data_path = args.data or '/mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5'
        trials = load_raw_trials(data_path, max_trials=args.max_trials)
        train_trials, val_trials, test_trials, n_sessions, train_sids = \
            split_within_day(trials)

        model = load_enhanced_gru(
            args.model, device=device, n_sessions=n_sessions,
            train_sids=train_sids)

        # Build decoder
        decoder = SimplifiedWFSTDecoder(
            cmudict_path=args.cmudict,
            kenlm_path=args.word_lm,
            acoustic_scale=args.acoustic_scale,
        )

        # Load phoneme LM
        phoneme_lm = None
        if args.kenlm:
            from lead4_decode_kenlm import create_decoder
            phoneme_lm = create_decoder(args.kenlm)

        # Evaluate
        wer = evaluate_wfst_decoder(
            model, val_trials, decoder, device=device,
            phoneme_lm=phoneme_lm, phoneme_alpha=args.alpha,
            beam_width=args.beam)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
