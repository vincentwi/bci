#!/usr/bin/env python3
"""OPT-6.7B N-best rescoring pipeline.

Rescores N-best phoneme/word hypotheses from KenLM beam search using
OPT-6.7B language model in 8-bit quantization.

Usage:
    # Rescore N-best lists
    python brain2speech/lead4_rescore_opt.py \
        --nbest brain2speech/results/L4_nbest_words.json \
        --output brain2speech/results/L4_opt_rescored.json

    # Sweep interpolation weights
    python brain2speech/lead4_rescore_opt.py \
        --nbest brain2speech/results/L4_nbest_words.json \
        --sweep-weights 0.0,0.1,0.2,0.3,0.5,0.7,1.0

References:
    - Source 1 (Willett repo): lmDecoderUtils.build_opt(load_in_8bit=True)
    - Source 5 (paper): N-best=100, interpolation_weight=0.5, length_penalty=0
    - Source 8 (DCoND): OPT as Stage 2 in 3-stage pipeline
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


class OPTRescorer:
    """OPT-6.7B language model rescorer.

    Source 5: 8-bit quantization, interpolation weight=0.5, length_penalty=0.
    """

    def __init__(self, model_name='facebook/opt-6.7b', device_map='auto',
                 load_in_8bit=True, cache_dir=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {model_name} (8-bit={load_in_8bit})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir)

        load_kwargs = {
            'device_map': device_map,
            'cache_dir': cache_dir,
        }
        if load_in_8bit:
            load_kwargs['load_in_8bit'] = True
        else:
            load_kwargs['torch_dtype'] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs)
        self.model.eval()
        print(f"OPT loaded. Device map: {getattr(self.model, 'hf_device_map', 'N/A')}")

    def score_sentence(self, text):
        """Compute log-likelihood of a sentence.

        Source 5: length_penalty=0 means we use raw log-likelihood
        (no normalization by length).

        Args:
            text: Text string to score

        Returns:
            Total log-likelihood (sum of per-token log probs)
        """
        if not text or not text.strip():
            return -1e6

        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs.input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            # outputs.loss is mean cross-entropy per token
            # Total log-likelihood = -loss * n_tokens
            n_tokens = input_ids.shape[1]
            log_likelihood = -outputs.loss.item() * n_tokens

        return log_likelihood

    def score_sentences_batch(self, texts):
        """Score multiple sentences (sequential, could be parallelized).

        Returns:
            List of log-likelihoods
        """
        scores = []
        for text in texts:
            scores.append(self.score_sentence(text))
        return scores

    def rescore_nbest(self, candidates, acoustic_scores=None, lm_weight=0.5):
        """Rescore N-best list with OPT and combine with acoustic scores.

        Source 5: lm_weight=0.5 (50/50 interpolation).

        Args:
            candidates: List of text strings
            acoustic_scores: List of acoustic/beam scores (from KenLM decoder).
                If None, uses OPT score alone.
            lm_weight: Interpolation weight for OPT score

        Returns:
            List of (text, combined_score, acoustic_score, opt_score) sorted by combined_score
        """
        opt_scores = self.score_sentences_batch(candidates)

        if acoustic_scores is None:
            acoustic_scores = [0.0] * len(candidates)

        scored = []
        for text, ac_score, opt_score in zip(candidates, acoustic_scores, opt_scores):
            combined = ac_score + lm_weight * opt_score
            scored.append((text, combined, ac_score, opt_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def sweep_weights(self, candidates, acoustic_scores,
                      weights=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)):
        """Pre-compute OPT scores once, sweep interpolation weight.

        Returns:
            Dict mapping weight → sorted list of (text, combined_score)
        """
        opt_scores = self.score_sentences_batch(candidates)

        results = {}
        for w in weights:
            combined = [
                (t, ac + w * opt, ac, opt)
                for t, ac, opt in zip(candidates, acoustic_scores, opt_scores)
            ]
            combined.sort(key=lambda x: x[1], reverse=True)
            results[w] = combined

        return results


def rescore_nbest_file(nbest_path, rescorer, converter, lm_weight=0.5,
                       output_path=None):
    """Rescore N-best phoneme lists by converting to words and scoring with OPT.

    Args:
        nbest_path: Path to JSON with N-best phoneme lists
        rescorer: OPTRescorer instance
        converter: PhonemeToWordConverter instance
        lm_weight: OPT interpolation weight
        output_path: Output JSON path

    Returns:
        List of rescored results per trial
    """
    with open(nbest_path, 'r') as f:
        nbest_data = json.load(f)

    results = []
    for trial_idx, trial in enumerate(nbest_data):
        candidates = trial['candidates']

        # Convert phonemes to words for OPT scoring
        texts = []
        acoustic_scores = []
        for c in candidates:
            phoneme_str = c['phonemes']
            text = converter.convert(phoneme_str)
            texts.append(text)
            # Use combined logit + LM score as acoustic score
            acoustic_scores.append(c.get('logit_score', 0.0) + c.get('lm_score', 0.0))

        # Rescore with OPT
        reranked = rescorer.rescore_nbest(texts, acoustic_scores, lm_weight=lm_weight)

        results.append({
            'trial_idx': trial.get('trial_idx', trial_idx),
            'target_phonemes': trial.get('target_phonemes', ''),
            'top1_text': reranked[0][0] if reranked else '',
            'top1_combined_score': reranked[0][1] if reranked else 0.0,
            'reranked': [
                {
                    'text': text,
                    'combined_score': float(combined),
                    'acoustic_score': float(ac),
                    'opt_score': float(opt),
                }
                for text, combined, ac, opt in reranked[:10]  # Keep top 10
            ],
        })

        if (trial_idx + 1) % 20 == 0:
            print(f"  Rescored {trial_idx + 1}/{len(nbest_data)} trials")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Rescored results saved: {output_path}")

    return results


def evaluate_rescoring(results, converter):
    """Evaluate WER of rescored results.

    Args:
        results: List of rescored trial dicts
        converter: PhonemeToWordConverter for target conversion

    Returns:
        Mean WER
    """
    import editdistance

    total_edits, total_words = 0, 0
    for r in results:
        pred_words = r['top1_text'].lower().split()
        # Convert target phonemes to words
        target_text = converter.convert(r['target_phonemes'])
        target_words = target_text.lower().split()

        edits = editdistance.eval(pred_words, target_words)
        total_edits += edits
        total_words += len(target_words)

    wer = total_edits / max(total_words, 1)
    return wer


def main():
    parser = argparse.ArgumentParser(description='OPT-6.7B N-best Rescoring')
    parser.add_argument('--nbest', type=str, required=True,
                        help='Path to N-best JSON from lead4_decode_kenlm.py')
    parser.add_argument('--model', type=str, default='facebook/opt-6.7b',
                        help='OPT model name (default: facebook/opt-6.7b)')
    parser.add_argument('--no-8bit', action='store_true',
                        help='Disable 8-bit quantization')
    parser.add_argument('--lm-weight', type=float, default=0.5,
                        help='OPT interpolation weight (default: 0.5)')
    parser.add_argument('--sweep-weights', type=str, default=None,
                        help='Comma-separated weights to sweep')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='HuggingFace cache directory')
    args = parser.parse_args()

    from lead4_phoneme_to_words import PhonemeToWordConverter
    converter = PhonemeToWordConverter()

    rescorer = OPTRescorer(
        model_name=args.model,
        load_in_8bit=not args.no_8bit,
        cache_dir=args.cache_dir,
    )

    if args.sweep_weights:
        weights = [float(w) for w in args.sweep_weights.split(',')]
        print(f"\nSweeping OPT weights: {weights}")

        with open(args.nbest, 'r') as f:
            nbest_data = json.load(f)

        # For each weight, rescore and evaluate
        for w in weights:
            results = rescore_nbest_file(args.nbest, rescorer, converter,
                                         lm_weight=w)
            wer = evaluate_rescoring(results, converter)
            print(f"  weight={w:.2f} → WER={wer:.4f}")

    else:
        output = args.output or 'brain2speech/results/L4_opt_rescored.json'
        results = rescore_nbest_file(args.nbest, rescorer, converter,
                                     lm_weight=args.lm_weight,
                                     output_path=output)
        wer = evaluate_rescoring(results, converter)
        print(f"\nOPT rescoring WER: {wer:.4f} (weight={args.lm_weight})")


if __name__ == '__main__':
    main()
