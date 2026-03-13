#!/usr/bin/env python3
"""DCoND-LIFT: Full end-to-end pipeline for Lead 2.

Pipeline:
  1. DCoND decoder (PaperExactDCoND, seed45 best) → diphone logits (1601-class)
  2. Marginalize diphones → monophone probs (41-class) via M_ext matrix
  3. CTC greedy decode → phoneme sequence (ARPABET)
  4. LIFT v1 (Qwen3.5-2B LoRA) → phoneme-to-text translation
  5. Optional: DCoND-LIFT corrector (text + phonemes → corrected text)
  6. Optional: Multi-seed majority vote (4 DCoND seeds × LIFT)

Results (val / test):
  - Single seed45 greedy + LIFT v1: 28.1% / 34.6% WER
  - 4-seed majority vote + LIFT v1: 27.7% / — WER
  - Oracle from 4 seeds: 22.6% WER (upper bound)

Key findings:
  - Phoneme-to-text LIFT bypasses lossy CMUDict: 28% val WER vs 57% with CMUDict
  - For PER < 10%, WER is 5.5-6.0% (competition-level)
  - PER is the primary bottleneck: reducing PER directly reduces WER
  - Ensemble HURTS greedy CTC decode (flattens peaks, 18.5→20.2% PER)
  - Only PaperExactDCoND h=512 ks=14 unidir architecture works
  - No automatic reranking works: log-prob, perplexity, multi-input all hurt
  - Majority vote across 4 seeds is the only helpful selection method (+2% relative)
"""
import json, sys, time, re
import numpy as np
import h5py
import torch
from collections import Counter
from Levenshtein import distance as lev_distance

sys.path.insert(0, 'brain2speech')
from config import CLASS_TO_ARPABET, N_CLASSES
from train_beyond_paper import compute_per, ctc_greedy_decode, DATA_DIR, RESULTS_DIR, CTC_BLANK
from lead2_train_dcond import PaperExactDCoND, build_marginalization_matrix_ext


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


def load_dcond_model(ckpt_path, device='cuda:0'):
    """Load a PaperExactDCoND checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model = PaperExactDCoND(**ckpt['model_kwargs'])
    state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt.get('val_per', None)


def decode_with_dcond(model, trials, train_sids, device='cuda:0', batch_size=32):
    """Decode trials with a DCoND model, returning phoneme strings."""
    model.set_train_sessions(train_sids)
    M_ext = build_marginalization_matrix_ext()

    results = []
    for i in range(0, len(trials), batch_size):
        batch = trials[i:i+batch_size]
        features = [torch.FloatTensor(t['features_raw']) for t in batch]
        feat_lens = [f.shape[0] for f in features]
        max_T = max(feat_lens)
        C = features[0].shape[1]
        padded = torch.zeros(len(batch), max_T, C).to(device)
        for j, f in enumerate(features):
            padded[j, :f.shape[0], :] = f
        session_ids = torch.LongTensor([t.get('session_idx', 0) for t in batch]).to(device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits = model(padded, session_ids)
        probs = logits.float().softmax(dim=-1).cpu()
        mono_probs = probs @ M_ext
        mono_lp = (mono_probs + 1e-10).log().numpy()

        for j, t in enumerate(batch):
            T_raw = feat_lens[j]
            T_out = max(1, (T_raw - model.kernel_size) // model.stride + 1)
            T_out = min(T_out, mono_lp.shape[1])
            decoded = ctc_greedy_decode(mono_lp[j, :T_out], blank=CTC_BLANK)
            target = t['phoneme_indices'].tolist()
            results.append({
                'decoded_phones': ' '.join([CLASS_TO_ARPABET[p] for p in decoded]),
                'gt_text': t['text'],
                'per': compute_per(decoded, target),
            })
    return results


def run_lift(data_list, lift_model, tok, device='cuda:0'):
    """Run LIFT on decoded phonemes to get text translations."""
    PROMPT = "Convert these decoded brain-computer interface phonemes to English text.\n\nPhonemes: {phones}\n\nEnglish:"

    results = []
    for d in data_list:
        if not d['gt_text']:
            d['lift_text'] = ''
            continue
        prompt = PROMPT.format(phones=d['decoded_phones'])
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = lift_model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
        d['lift_text'] = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        gt_words = normalize_text(d['gt_text'])
        lift_words = normalize_text(d['lift_text'])
        wer = lev_distance(lift_words, gt_words) / len(gt_words) if gt_words else 0.0
        d['wer'] = wer
    return data_list


def multi_seed_majority_vote(seed_results_list):
    """Select best candidate via majority vote across seeds.

    seed_results_list: list of lists, each inner list has dicts with 'lift_text', 'gt_text'
    Returns: list of selected texts and WERs
    """
    n_trials = len(seed_results_list[0])
    results = []
    for i in range(n_trials):
        gt_text = seed_results_list[0][i]['gt_text']
        if not gt_text:
            continue
        gt_words = normalize_text(gt_text)
        if not gt_words:
            continue

        candidates = [sr[i]['lift_text'] for sr in seed_results_list]
        normalized = [' '.join(normalize_text(c)) for c in candidates]
        counter = Counter(normalized)
        majority = counter.most_common(1)[0][0]
        majority_words = majority.split()
        wer = lev_distance(majority_words, gt_words) / len(gt_words)
        results.append({'text': majority, 'wer': wer, 'gt': gt_text})
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'multi-seed', 'full'], default='single')
    parser.add_argument('--split', choices=['val', 'test', 'both'], default='val')
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--lift-model', default='phoneme_lift_qwen/final')
    args = parser.parse_args()

    # Load data
    h5_path = DATA_DIR / 'sentences_paper_256d.h5'
    print(f"Loading data from {h5_path}...")
    trials = []
    with h5py.File(h5_path, 'r') as f:
        n_trials = f.attrs['n_trials']
        for i in range(n_trials):
            grp = f[f'trial_{i:05d}']
            trials.append({
                'features_raw': grp['features'][:],
                'phoneme_indices': grp['phoneme_indices'][:],
                'session': grp.attrs['session'],
                'text': grp.attrs.get('text', ''),
            })

    session_names = sorted(set(t['session'] for t in trials))
    session_to_idx = {s: i for i, s in enumerate(session_names)}
    for t in trials:
        t['session_idx'] = session_to_idx[t['session']]
    sessions = {}
    for t in trials:
        sessions.setdefault(t['session'], []).append(t)

    train_sessions = session_names[:-6]
    val_sessions = session_names[-6:-4]
    test_sessions = session_names[-4:]
    train_sids = {session_to_idx[s] for s in train_sessions}

    eval_trials = {}
    if args.split in ('val', 'both'):
        eval_trials['val'] = [t for s in val_sessions for t in sessions[s]]
    if args.split in ('test', 'both'):
        eval_trials['test'] = [t for s in test_sessions for t in sessions[s]]

    device = torch.device('cuda:0')

    if args.mode == 'single':
        ckpt_path = str(RESULTS_DIR / f'L2_L2_pe_dcond_adam_s{args.seed}_best.pt')
        if args.seed == 42:
            ckpt_path = str(RESULTS_DIR / f'L2_L2_pe_dcond_adam_s42_long_best.pt')

        model, val_per = load_dcond_model(ckpt_path, device)
        print(f"Loaded seed{args.seed}, val PER: {val_per}")

        for split_name, trial_list in eval_trials.items():
            data = decode_with_dcond(model, trial_list, train_sids, device)
            per = np.mean([d['per'] for d in data])
            print(f"{split_name}: {len(data)} trials, PER={per:.1%}")

        del model; torch.cuda.empty_cache()

        # Load LIFT
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        adapter_path = f'/mnt/home/vincent.wilmet/brain2speech/models/{args.lift_model}'
        tok = AutoTokenizer.from_pretrained(adapter_path)
        lift_model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen3.5-2B', dtype=torch.bfloat16, device_map='cuda:0')
        lift_model = PeftModel.from_pretrained(lift_model, adapter_path)
        lift_model.eval()

        for split_name, trial_list in eval_trials.items():
            data = decode_with_dcond(model if 'model' in dir() else load_dcond_model(ckpt_path, device)[0],
                                     trial_list, train_sids, device)
            data = run_lift(data, lift_model, tok, device)
            wers = [d.get('wer', 0) for d in data if d.get('wer') is not None]
            print(f"\n{split_name}: Mean WER={np.mean(wers):.1%}, Median={np.median(wers):.1%}")

    print("\nDone.")
