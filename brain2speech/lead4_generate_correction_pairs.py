#!/usr/bin/env python3
"""Generate training pairs for Qwen LoRA correction from real decoder outputs.

Instead of synthetic corruption, this uses actual CTC decoder predictions
paired with ground truth to create training data. This produces more
representative error distributions than confusion-matrix-based noise.

Strategies:
  A) Single greedy: (greedy_predicted, ground_truth) pairs
  B) KenLM beam: (beam_decoded, ground_truth) pairs
  C) N-best candidates: (top-N candidates, ground_truth) for DCoND-LIFT training

References:
    - Source 6 (MONA LISA): Only 100 fine-tuning examples → 44% relative WER improvement
    - Source 8 (DCoND): 10 candidates per training sentence
    - Source 10 (BIT): Smaller LLMs (1.5B) better than larger for BCI correction

Usage:
    # Generate greedy prediction pairs
    python brain2speech/lead4_generate_correction_pairs.py \
        --model brain2speech/results/lead4/L4_cffan_baseline.pt \
        --strategy greedy \
        --output brain2speech/data/correction_pairs_greedy.jsonl

    # Generate DCoND-LIFT style N-best pairs
    python brain2speech/lead4_generate_correction_pairs.py \
        --model brain2speech/results/lead4/L4_cffan_baseline.pt \
        --strategy nbest --kenlm brain2speech/data/phoneme_4gram.arpa \
        --output brain2speech/data/correction_pairs_nbest.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_TO_ARPABET, N_CLASSES

CTC_BLANK = 40

# ═══════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES (matching lead4_qwen_correction_v3.py)
# ═══════════════════════════════════════════════════════════════════════

PROMPTS = {
    'A': (
        "You are a phoneme error correction model for a brain-computer interface. "
        "Given a noisy ARPABET phoneme sequence decoded from neural signals, "
        "output the corrected sequence. Only output the corrected phonemes, nothing else."
    ),
    'B_dcond': (
        "You are correcting brain-computer interface decoding output. "
        "You receive both decoded phonemes and an approximate text transcription. "
        "Correct the phoneme sequence so it matches a natural English sentence. "
        "Output only the corrected ARPABET phonemes, space-separated."
    ),
    'D_nbest': (
        "Choose the transcription that is most accurate, ensuring it is "
        "contextually and grammatically correct. Focus on key differences "
        "in the options that change the meaning or correctness. "
        "Avoid repetitive or nonsensical phrases. "
        "Output only the chosen transcription."
    ),
    'F_full_dcond': (
        "Perform automatic speech recognition on the following decoded neural signals. "
        "Translate each subgroup of phonemes enclosed by two SIL symbols into one single word. "
        "Remove SIL symbols at the start or the end. "
        "Output the refined transcription and its corresponding phoneme representation only, "
        "without any introductory text."
    ),
    'G_word': (
        "You are correcting text from a brain-computer interface. "
        "The text may contain word errors due to phoneme decoding mistakes. "
        "Correct the text to form a natural English sentence. "
        "Output only the corrected text."
    ),
}


def greedy_decode(log_probs_np):
    """CTC greedy decode."""
    best_path = log_probs_np.argmax(axis=1)
    decoded = []
    prev = -1
    for t in best_path:
        if t != prev and t != CTC_BLANK:
            decoded.append(int(t))
        prev = t
    return [CLASS_TO_ARPABET[d] for d in decoded if d in CLASS_TO_ARPABET]


def generate_greedy_pairs(model, trials, device='cuda', converter=None,
                          prompt_key='A'):
    """Strategy A: Greedy prediction → ground truth pairs."""
    pairs = []
    model.eval()

    for i, trial in enumerate(trials):
        features = torch.FloatTensor(trial['features_raw']).unsqueeze(0).to(device)
        session_id = torch.LongTensor([trial.get('session_idx', 0)]).to(device)

        with torch.no_grad():
            logits = model(features, session_id)
            log_probs = F.log_softmax(logits, dim=-1)
        log_probs_np = log_probs[0].cpu().numpy()

        # Greedy decode
        pred_phones = greedy_decode(log_probs_np)

        # Ground truth
        target_arr = trial.get('phoneme_indices', [])
        if hasattr(target_arr, 'tolist'):
            target_arr = target_arr.tolist()
        target_phones = [CLASS_TO_ARPABET[int(t)] for t in target_arr
                         if int(t) in CLASS_TO_ARPABET]

        pred_str = ' '.join(pred_phones)
        target_str = ' '.join(target_phones)

        if prompt_key == 'A':
            user_content = pred_str
            assistant_content = target_str
        elif prompt_key == 'B_dcond':
            pred_text = converter.convert(pred_str) if converter else ''
            user_content = f"Phonemes: {pred_str}\nText: {pred_text}"
            assistant_content = target_str
        elif prompt_key == 'G_word':
            pred_text = converter.convert(pred_str) if converter else ''
            target_text = trial.get('text', converter.convert(target_str) if converter else '')
            user_content = pred_text
            assistant_content = target_text
        else:
            user_content = pred_str
            assistant_content = target_str

        pair = {
            'messages': [
                {'role': 'system', 'content': PROMPTS[prompt_key]},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': assistant_content},
            ]
        }
        pairs.append(pair)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i+1}/{len(trials)} pairs")

    return pairs


def generate_dcond_pairs(model, trials, lm, device='cuda', converter=None,
                         beam_width=50, alpha=0.3, nbest=5, prompt_key='B_dcond'):
    """Strategy B/C: Beam decoded pairs with optional N-best candidates."""
    from lead4_decode_kenlm import decode_single

    pairs = []
    model.eval()

    for i, trial in enumerate(trials):
        features = torch.FloatTensor(trial['features_raw']).unsqueeze(0).to(device)
        session_id = torch.LongTensor([trial.get('session_idx', 0)]).to(device)

        with torch.no_grad():
            logits = model(features, session_id)
            log_probs = F.log_softmax(logits, dim=-1)
        log_probs_np = log_probs[0].cpu().numpy()

        # Beam search decode
        results = decode_single(log_probs_np, lm, beam_width=beam_width,
                                alpha=alpha, beta=0.0, nbest=nbest)

        # Ground truth
        target_arr = trial.get('phoneme_indices', [])
        if hasattr(target_arr, 'tolist'):
            target_arr = target_arr.tolist()
        target_phones = [CLASS_TO_ARPABET[int(t)] for t in target_arr
                         if int(t) in CLASS_TO_ARPABET]
        target_str = ' '.join(target_phones)
        target_text = trial.get('text', '')

        if prompt_key == 'B_dcond':
            # DCoND dual-input: best beam decoded + text
            if results:
                pred_str = results[0][0]
            else:
                pred_str = ' '.join(greedy_decode(log_probs_np))
            pred_text = converter.convert(pred_str) if converter else ''
            user_content = f"Phonemes: {pred_str}\nText: {pred_text}"
            assistant_content = target_str
        elif prompt_key == 'F_full_dcond':
            # Full DCoND-LIFT: all candidates
            candidate_lines = []
            for rank, (ph_str, indices, score) in enumerate(results, 1):
                text = converter.convert(ph_str) if converter else ''
                candidate_lines.append(f"Candidate {rank}: {text} | {ph_str}")
            user_content = '\n'.join(candidate_lines)
            if target_text and converter:
                target_ph_str = target_str
                assistant_content = f"{target_text} | {target_ph_str}"
            else:
                assistant_content = target_str
        elif prompt_key == 'D_nbest':
            # N-best text selection
            candidate_lines = []
            for rank, (ph_str, indices, score) in enumerate(results, 1):
                text = converter.convert(ph_str) if converter else ph_str
                candidate_lines.append(f"{rank}: {text}")
            user_content = '\n'.join(candidate_lines)
            assistant_content = target_text if target_text else target_str
        else:
            if results:
                user_content = results[0][0]
            else:
                user_content = ' '.join(greedy_decode(log_probs_np))
            assistant_content = target_str

        pair = {
            'messages': [
                {'role': 'system', 'content': PROMPTS[prompt_key]},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': assistant_content},
            ]
        }
        pairs.append(pair)

        if (i + 1) % 200 == 0:
            print(f"  Generated {i+1}/{len(trials)} pairs")

    return pairs


def main():
    parser = argparse.ArgumentParser(description='Generate correction training pairs')
    parser.add_argument('--model', type=str, required=True,
                        help='CTC model checkpoint path')
    parser.add_argument('--strategy', type=str, default='greedy',
                        choices=['greedy', 'beam', 'nbest'],
                        help='Pair generation strategy')
    parser.add_argument('--prompt', type=str, default='B_dcond',
                        choices=list(PROMPTS.keys()),
                        help='Prompt template to use')
    parser.add_argument('--kenlm', type=str, default=None,
                        help='KenLM ARPA path (for beam/nbest strategies)')
    parser.add_argument('--beam-width', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--nbest', type=int, default=5)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--split', type=str, default='within-day',
                        choices=['within-day', 'cross-session'])
    parser.add_argument('--set', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Which split to generate pairs from')
    parser.add_argument('--max-trials', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL path')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Load data
    from lead4_full_pipeline import (load_raw_trials, split_within_day,
                                      split_cross_session, load_enhanced_gru)
    data_path = args.data or '/mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5'
    trials = load_raw_trials(data_path, max_trials=args.max_trials)

    if args.split == 'within-day':
        train_trials, val_trials, test_trials, n_sessions, train_sids = \
            split_within_day(trials, seed=args.seed)
    else:
        train_trials, val_trials, test_trials, n_sessions, train_sids = \
            split_cross_session(trials)

    if args.set == 'train':
        gen_trials = train_trials
    elif args.set == 'val':
        gen_trials = val_trials
    else:
        gen_trials = test_trials

    print(f"Generating pairs from {len(gen_trials)} {args.set} trials")

    # Load model
    model = load_enhanced_gru(args.model, device=device, n_sessions=n_sessions,
                               train_sids=train_sids)

    # Load converter for text-based prompts
    converter = None
    if args.prompt in ('B_dcond', 'D_nbest', 'F_full_dcond', 'G_word'):
        from lead4_phoneme_to_words import PhonemeToWordConverter
        converter = PhonemeToWordConverter()

    # Generate pairs
    t_start = time.time()
    if args.strategy == 'greedy':
        pairs = generate_greedy_pairs(model, gen_trials, device=device,
                                       converter=converter, prompt_key=args.prompt)
    else:
        if not args.kenlm:
            print("ERROR: --kenlm required for beam/nbest strategies")
            sys.exit(1)
        from lead4_decode_kenlm import create_decoder
        lm = create_decoder(args.kenlm)
        pairs = generate_dcond_pairs(model, gen_trials, lm, device=device,
                                      converter=converter, beam_width=args.beam_width,
                                      alpha=args.alpha, nbest=args.nbest,
                                      prompt_key=args.prompt)

    elapsed = time.time() - t_start
    print(f"Generated {len(pairs)} pairs in {elapsed:.0f}s")

    # Save as JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    print(f"Saved: {output_path}")

    # Show sample
    print(f"\nSample pair:")
    sample = pairs[0]
    for msg in sample['messages']:
        role = msg['role']
        content = msg['content'][:100] + ('...' if len(msg['content']) > 100 else '')
        print(f"  [{role}]: {content}")


if __name__ == '__main__':
    main()
