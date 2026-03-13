#!/usr/bin/env python3
"""End-to-end: Ensemble → KenLM → OPT-6.7B → Qwen LoRA → PER/WER.

Matches DCoND-LIFT architecture (Source 8):
  Stage 1: N decoders → CTC log probs (ensemble or diverse candidates)
  Stage 2: KenLM 5-gram beam → N-best phoneme hypotheses
  Stage 3: phoneme→word + OPT-6.7B rescoring → re-ranked word hypotheses
  Stage 4: Qwen LoRA correction (DCoND-LIFT dual input) → final text

Usage:
    # Greedy CTC only (within-day split, EnhancedGRU)
    python brain2speech/lead4_full_pipeline.py --stage greedy \
        --model brain2speech/results/lead4/L4_cffan_baseline.pt \
        --model-type enhanced --split within-day

    # + KenLM beam
    python brain2speech/lead4_full_pipeline.py --stage kenlm \
        --model brain2speech/results/lead4/L4_cffan_baseline.pt \
        --model-type enhanced --split within-day \
        --kenlm brain2speech/data/phoneme_4gram.arpa

    # Full pipeline
    python brain2speech/lead4_full_pipeline.py --stage full \
        --ensemble-config brain2speech/configs/lead4_ensemble.json \
        --split within-day \
        --kenlm brain2speech/data/phoneme_5gram.arpa \
        --opt facebook/opt-6.7b \
        --qwen models/qwen_phoneme_corrector/best/

References:
    - Source 8 (DCoND): 5.77% WER with 10 decoders + 5-gram + OPT + GPT-3.5
    - Source 7 (Benchmark): DCoND-LIFT pipeline details
    - Source 10 (BIT): 6.35% WER with cascaded n-gram LM
"""
import argparse
import json
import sys
import time
from pathlib import Path

import editdistance
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_TO_ARPABET, N_CLASSES

CTC_BLANK = 40


def compute_wer(predicted_text, reference_text):
    """Word Error Rate."""
    pred_words = predicted_text.lower().split()
    ref_words = reference_text.lower().split()
    if not ref_words:
        return 1.0 if pred_words else 0.0
    return editdistance.eval(pred_words, ref_words) / len(ref_words)


def compute_per(predicted_phones, reference_phones):
    """Phoneme Error Rate."""
    if isinstance(predicted_phones, str):
        predicted_phones = predicted_phones.split()
    if isinstance(reference_phones, str):
        reference_phones = reference_phones.split()
    if not reference_phones:
        return 1.0 if predicted_phones else 0.0
    return editdistance.eval(predicted_phones, reference_phones) / len(reference_phones)


def greedy_decode(log_probs_np):
    """CTC greedy decode from log-probabilities."""
    best_path = log_probs_np.argmax(axis=1)
    decoded = []
    prev = -1
    for t in best_path:
        if t != prev and t != CTC_BLANK:
            decoded.append(int(t))
        prev = t
    return [CLASS_TO_ARPABET[d] for d in decoded if d in CLASS_TO_ARPABET]


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING (supports both cross-session and within-day splits)
# ═══════════════════════════════════════════════════════════════════════

def load_raw_trials(h5_path, max_trials=None):
    """Load raw (un-stacked) features from H5 dataset."""
    import h5py
    from train_beyond_paper import causal_gaussian_smooth

    trials = []
    with h5py.File(h5_path, 'r') as f:
        n_trials = f.attrs['n_trials']
        if max_trials:
            n_trials = min(n_trials, max_trials)
        t_start = time.time()
        for i in range(n_trials):
            if i % 2000 == 0 and i > 0:
                print(f"  Loading {i}/{n_trials} ({time.time()-t_start:.1f}s)...")
            grp = f[f'trial_{i:05d}']
            features = grp['features'][:].astype(np.float32)
            features = causal_gaussian_smooth(features, sd_bins=2, delay_bins=8)
            trials.append({
                'features_raw': features,
                'phoneme_indices': grp['phoneme_indices'][:],
                'session': grp.attrs['session'],
                'text': grp.attrs['text'],
                'n_frames_raw': features.shape[0],
                'n_phonemes': grp.attrs['n_phonemes'],
            })

    elapsed = time.time() - t_start
    print(f"Loaded {len(trials)} trials ({elapsed:.1f}s) — raw {trials[0]['features_raw'].shape[1]}D")
    return trials


def split_within_day(trials, seed=42, val_ratio=0.1):
    """Within-day split: 40 test sentences/session, 10% val, rest train.

    This matches the competition protocol where all sessions are available
    during training but some sentences from each session are held out.
    """
    # Assign session indices
    session_names = sorted(set(t['session'] for t in trials))
    session_to_idx = {s: i for i, s in enumerate(session_names)}
    n_sessions = len(session_names)
    for t in trials:
        t['session_idx'] = session_to_idx[t['session']]

    # Group by session
    sessions_dict = {}
    for t in trials:
        sessions_dict.setdefault(t['session'], []).append(t)

    rng = np.random.RandomState(seed)
    train_trials, val_trials, test_trials = [], [], []
    for s_name in session_names:
        s_trials = sessions_dict[s_name]
        rng.shuffle(s_trials)
        n_test = 40  # Paper: 40 sentences per day held out for test
        n_val = max(1, int(len(s_trials) * val_ratio))
        test_trials.extend(s_trials[:n_test])
        val_trials.extend(s_trials[n_test:n_test + n_val])
        train_trials.extend(s_trials[n_test + n_val:])

    # All sessions are "train" sessions (day-specific layers trained for all)
    train_sids = set(range(n_sessions))

    print(f"Within-day split: {len(train_trials)} train, {len(val_trials)} val, "
          f"{len(test_trials)} test ({n_sessions} sessions)")
    return train_trials, val_trials, test_trials, n_sessions, train_sids


def split_cross_session(trials):
    """Cross-session split: last 4 sessions test, prev 2 val, rest train."""
    session_names = sorted(set(t['session'] for t in trials))
    session_to_idx = {s: i for i, s in enumerate(session_names)}
    n_sessions = len(session_names)
    for t in trials:
        t['session_idx'] = session_to_idx[t['session']]

    sessions_dict = {}
    for t in trials:
        sessions_dict.setdefault(t['session'], []).append(t)

    test_sessions = session_names[-4:]
    remaining = session_names[:-4]
    val_sessions = remaining[-2:]
    train_sessions = remaining[:-2]

    train_trials = [t for s in train_sessions for t in sessions_dict[s]]
    val_trials = [t for s in val_sessions for t in sessions_dict[s]]
    test_trials = [t for s in test_sessions for t in sessions_dict[s]]
    train_sids = {session_to_idx[s] for s in train_sessions}

    print(f"Cross-session split: {len(train_trials)} train, {len(val_trials)} val, "
          f"{len(test_trials)} test ({n_sessions} sessions)")
    return train_trials, val_trials, test_trials, n_sessions, train_sids


# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_enhanced_gru(checkpoint_path, device='cuda', n_sessions=24,
                      hidden=1024, n_layers=5, bidirectional=True,
                      kernel_size=32, stride=4, day_hidden=256,
                      input_dim=256, train_sids=None,
                      use_post_rnn_head=False, head_dim=256,
                      use_speckled_mask=False, speckled_p=0.3,
                      ortho_init=False):
    """Load EnhancedGRU model from lead1_train_gru_v2.py checkpoint."""
    from lead1_train_gru_v2 import EnhancedGRU

    # Load checkpoint (may be wrapped dict or raw state_dict)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        # Override kwargs from saved hyperparams if available
        hp = ckpt.get('hyperparams', {})
        if hp:
            hidden = hp.get('hidden', hidden)
            n_layers = hp.get('n_layers', n_layers)
            bidirectional = hp.get('bidirectional', bidirectional)
            kernel_size = hp.get('kernel_size', kernel_size)
            stride = hp.get('stride', stride)
            day_hidden = hp.get('day_hidden', day_hidden)
            input_dim = hp.get('input_dim', input_dim)
            use_post_rnn_head = hp.get('use_post_rnn_head', use_post_rnn_head)
            head_dim = hp.get('head_dim', head_dim)
            use_speckled_mask = hp.get('use_speckled_mask', use_speckled_mask)
            speckled_p = hp.get('speckled_p', speckled_p)
            ortho_init = hp.get('ortho_init', ortho_init)
        n_sessions = ckpt.get('n_sessions', n_sessions)
    else:
        state_dict = ckpt

    # Handle bidirectional stored as string
    if isinstance(bidirectional, str):
        bidirectional = bidirectional.lower() == 'true'

    model = EnhancedGRU(
        input_dim=input_dim,
        n_classes=N_CLASSES + 1,  # 41
        hidden=hidden,
        n_layers=n_layers,
        dropout=0.4,
        n_sessions=n_sessions,
        kernel_size=kernel_size,
        stride=stride,
        bidirectional=bidirectional,
        day_hidden=day_hidden,
        use_post_rnn_head=use_post_rnn_head,
        head_dim=head_dim,
        use_speckled_mask=use_speckled_mask,
        speckled_p=speckled_p,
        ortho_init=ortho_init,
    )

    model.load_state_dict(state_dict)
    model.to(device).eval()

    if train_sids is not None:
        model.day_input.set_train_sessions(train_sids)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded EnhancedGRU: {n_params/1e6:.1f}M params, "
          f"hidden={hidden}, bidir={bidirectional}, kernel={kernel_size}")
    return model


def load_paper_gru(checkpoint_path, device='cuda', n_sessions=24,
                   train_sids=None):
    """Load PaperGRUSimple model from beam_search_decode.py."""
    from beam_search_decode import load_model
    model = load_model(checkpoint_path, device=device, n_sessions=n_sessions,
                       n_features_stacked=3584, bidirectional=True,
                       activation='softsign',
                       train_session_idxs=train_sids)
    return model


def load_beyond_paper_gru(checkpoint_path, device='cuda', n_sessions=24,
                          hidden=512, n_layers=5, bidirectional=True,
                          kernel_size=14, stride=4, day_hidden=256,
                          input_dim=256, train_sids=None):
    """Load BeyondPaperGRU model from train_beyond_paper.py."""
    from train_beyond_paper import BeyondPaperGRU

    model = BeyondPaperGRU(
        n_features_per_frame=input_dim,
        n_classes=N_CLASSES + 1,  # 41
        hidden=hidden,
        n_layers=n_layers,
        dropout=0.4,
        n_sessions=n_sessions,
        kernel_size=kernel_size,
        stride=stride,
        bidirectional=bidirectional,
        day_hidden=day_hidden,
    )

    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    if train_sids is not None:
        model.day_input.set_train_sessions(train_sids)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded BeyondPaperGRU: {n_params/1e6:.1f}M params, "
          f"hidden={hidden}, bidir={bidirectional}, kernel={kernel_size}")
    return model


def auto_detect_model_type(checkpoint_path):
    """Try to detect model type from checkpoint state dict keys."""
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    keys = set(state.keys())

    if any('post_rnn_head' in k for k in keys):
        return 'enhanced'
    elif any('day_input.day_weights' in k for k in keys):
        # Both BeyondPaperGRU and EnhancedGRU have this
        if any('efficient' in k.lower() or 'speckled' in k for k in keys):
            return 'enhanced'
        return 'beyond'
    elif any('day_projections' in k for k in keys):
        return 'paper'
    else:
        return 'enhanced'  # default


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_pipeline(
    stage='greedy',
    model_path=None,
    model_type='enhanced',
    model_kwargs=None,
    ensemble_config=None,
    kenlm_path=None,
    kenlm_alpha=2.0,
    kenlm_beta=0.0,
    beam_width=500,
    nbest=100,
    opt_model_name=None,
    opt_weight=0.5,
    opt_load_in_8bit=True,
    qwen_adapter=None,
    qwen_prompt='B_dcond',
    use_diverse_candidates=False,
    use_mixture=False,
    data_path=None,
    split='within-day',
    eval_set='val',
    max_trials=None,
    device='cuda',
    output_path=None,
    seed=42,
):
    """Evaluate pipeline at specified stage.

    Stages (cumulative):
        greedy: CTC greedy decode only
        kenlm: + KenLM beam search
        kenlm+opt: + OPT-6.7B rescoring
        ensemble: ensemble greedy
        ensemble+kenlm: ensemble + KenLM
        ensemble+kenlm+opt: ensemble + KenLM + OPT
        full: ensemble + KenLM + OPT + Qwen correction
    """
    # ── Load data ──
    if data_path is None:
        data_path = '/mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5'

    use_raw = model_type in ('enhanced', 'beyond')

    if use_raw:
        trials = load_raw_trials(data_path, max_trials=max_trials)
    else:
        from beam_search_decode import load_trials
        trials = load_trials(data_path, kernel_size=14, stride=4, max_trials=max_trials)

    if split == 'within-day':
        train_trials, val_trials, test_trials, n_sessions, train_sids = \
            split_within_day(trials, seed=seed)
    else:
        train_trials, val_trials, test_trials, n_sessions, train_sids = \
            split_cross_session(trials)

    if eval_set == 'val':
        eval_trials = val_trials
    elif eval_set == 'test':
        eval_trials = test_trials
    else:
        eval_trials = val_trials + test_trials

    print(f"Evaluating {len(eval_trials)} {eval_set} trials")

    # ── Load model(s) ──
    use_ensemble = 'ensemble' in stage
    if use_ensemble and ensemble_config:
        from lead4_ensemble_ctc import CTCEnsemble, load_ensemble_config
        configs = load_ensemble_config(ensemble_config)
        ensemble = CTCEnsemble(configs, device=device)
    elif model_path:
        kwargs = model_kwargs or {}
        if model_type == 'enhanced':
            model = load_enhanced_gru(
                model_path, device=device, n_sessions=n_sessions,
                train_sids=train_sids, **kwargs)
        elif model_type == 'beyond':
            model = load_beyond_paper_gru(
                model_path, device=device, n_sessions=n_sessions,
                train_sids=train_sids, **kwargs)
        elif model_type == 'paper':
            model = load_paper_gru(
                model_path, device=device, n_sessions=n_sessions,
                train_sids=train_sids)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    else:
        raise ValueError("Need --model or --ensemble-config")

    # ── Load KenLM LM (if needed) ──
    kenlm_lm = None
    if 'kenlm' in stage or stage == 'full':
        if kenlm_path:
            from lead4_decode_kenlm import create_decoder
            kenlm_lm = create_decoder(kenlm_path)
        else:
            print("WARNING: KenLM stage requested but no --kenlm path provided")

    # ── Load OPT rescorer (if needed) ──
    opt_rescorer = None
    if 'opt' in stage or stage == 'full':
        if opt_model_name:
            from lead4_rescore_opt import OPTRescorer
            opt_rescorer = OPTRescorer(model_name=opt_model_name,
                                          load_in_8bit=opt_load_in_8bit)
        else:
            print("WARNING: OPT stage requested but no --opt model provided")

    # ── Load phoneme-to-word converter ──
    converter = None
    if opt_rescorer or 'word' in stage or stage == 'full':
        from lead4_phoneme_to_words import PhonemeToWordConverter
        converter = PhonemeToWordConverter()

    # ── Load Qwen corrector (if needed) ──
    qwen_corrector = None
    if stage == 'full' and qwen_adapter:
        from lead4_qwen_correction_v3 import EnhancedPhonemeCorrector
        qwen_corrector = EnhancedPhonemeCorrector(adapter_path=qwen_adapter)

    # ── Run pipeline ──
    all_per = []
    all_wer = []
    results_log = []
    t_start = time.time()

    for trial_idx, trial in enumerate(eval_trials):
        # Get features (raw or stacked depending on model type)
        if 'features_raw' in trial:
            features = torch.FloatTensor(trial['features_raw']).unsqueeze(0).to(device)
        elif 'features_stacked' in trial:
            features = torch.FloatTensor(trial['features_stacked']).unsqueeze(0).to(device)
        else:
            features = torch.FloatTensor(trial['features']).unsqueeze(0).to(device)
        session_id = torch.LongTensor([trial.get('session_idx', 0)]).to(device)

        target_phones_arr = trial.get('phoneme_indices', [])
        if hasattr(target_phones_arr, 'tolist'):
            target_phones_arr = target_phones_arr.tolist()
        target_phones = [CLASS_TO_ARPABET[int(t)] for t in target_phones_arr
                         if int(t) in CLASS_TO_ARPABET]

        # === Stage 1: CTC decode ===
        if use_ensemble:
            if use_mixture:
                log_probs = ensemble.get_ensemble_log_probs_mixture(features, session_id)
            else:
                log_probs = ensemble.get_ensemble_log_probs(features, session_id)
            log_probs_np = log_probs[0].cpu().numpy()
        else:
            with torch.no_grad():
                logits = model(features, session_id)
                log_probs = F.log_softmax(logits, dim=-1)
            log_probs_np = log_probs[0].cpu().numpy()

        # Greedy decode (always computed for comparison)
        greedy_phones = greedy_decode(log_probs_np)
        greedy_per = compute_per(greedy_phones, target_phones)

        if stage in ('greedy', 'ensemble'):
            pred_phones = greedy_phones
            pred_text = converter.convert(' '.join(pred_phones)) if converter else ''
        elif kenlm_lm:
            # === Stage 2: KenLM beam search ===
            from lead4_decode_kenlm import decode_single
            results = decode_single(log_probs_np, kenlm_lm,
                                     beam_width=beam_width, alpha=kenlm_alpha,
                                     beta=kenlm_beta, nbest=nbest)

            if results:
                # results[0] = (phoneme_names_str, phoneme_indices, score)
                pred_phones = results[0][0].split()
            else:
                pred_phones = greedy_phones

            kenlm_per = compute_per(pred_phones, target_phones)

            if opt_rescorer and converter:
                # === Stage 3: OPT rescoring ===
                texts = []
                acoustic_scores = []
                phone_candidates = []
                for ph_str, indices, score in results[:nbest]:
                    text = converter.convert(ph_str)
                    texts.append(text)
                    acoustic_scores.append(float(score))
                    phone_candidates.append(ph_str)

                reranked = opt_rescorer.rescore_nbest(texts, acoustic_scores,
                                                      lm_weight=opt_weight)
                pred_text = reranked[0][0]
                top_idx = texts.index(pred_text) if pred_text in texts else 0
                pred_phones = phone_candidates[top_idx].split()
            else:
                pred_text = converter.convert(' '.join(pred_phones)) if converter else ''
        else:
            pred_phones = greedy_phones
            pred_text = ''

        # === Stage 4: Qwen correction ===
        if qwen_corrector and stage == 'full':
            if use_diverse_candidates and use_ensemble and kenlm_lm:
                # DCoND-LIFT: get diverse candidates from ensemble
                diverse = ensemble.get_diverse_candidates(
                    features, session_id, kenlm_lm, beam_width=150)
                corrected = qwen_corrector.correct_dcond_lift(diverse)
            elif converter:
                # Dual-input: phonemes + text
                corrected = qwen_corrector.correct_phonemes(
                    ' '.join(pred_phones),
                    approx_text=pred_text,
                    prompt_key=qwen_prompt)
            else:
                corrected = qwen_corrector.correct_phonemes(
                    ' '.join(pred_phones), prompt_key='A')

            # Parse corrected output
            if qwen_prompt in ('D_nbest', 'E_multi', 'G_word', 'F_full_dcond'):
                # Text output
                pred_text = corrected
            else:
                pred_phones = corrected.split()

        # === Compute metrics ===
        final_per = compute_per(pred_phones, target_phones)
        all_per.append(final_per)

        target_text = converter.convert(' '.join(target_phones)) if converter else ''
        if pred_text and target_text:
            wer = compute_wer(pred_text, target_text)
            all_wer.append(wer)

        result = {
            'trial': trial_idx,
            'greedy_per': greedy_per,
            'final_per': final_per,
            'pred_phonemes': ' '.join(pred_phones),
            'target_phonemes': ' '.join(target_phones),
        }
        if pred_text:
            result['pred_text'] = pred_text
        if target_text:
            result['target_text'] = target_text
            result['wer'] = compute_wer(pred_text, target_text)

        results_log.append(result)

        if (trial_idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            avg_per = np.mean(all_per)
            avg_wer = np.mean(all_wer) if all_wer else float('nan')
            print(f"[{trial_idx+1}/{len(eval_trials)}] "
                  f"PER={avg_per:.4f} WER={avg_wer:.4f} "
                  f"({elapsed:.0f}s)")

    # ── Summary ──
    mean_per = np.mean(all_per)
    mean_wer = np.mean(all_wer) if all_wer else float('nan')
    greedy_pers = [r['greedy_per'] for r in results_log]
    mean_greedy_per = np.mean(greedy_pers)

    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"Stage: {stage}")
    print(f"Split: {split}")
    print(f"Trials: {len(eval_trials)}")
    print(f"Greedy PER:  {mean_greedy_per:.4f}")
    print(f"Final PER:   {mean_per:.4f}")
    print(f"PER improvement: {mean_greedy_per - mean_per:.4f} "
          f"({(mean_greedy_per - mean_per)/mean_greedy_per*100:.1f}% relative)")
    if all_wer:
        print(f"Final WER:   {mean_wer:.4f}")
    print(f"Time: {elapsed:.0f}s ({elapsed/len(eval_trials):.1f}s/trial)")

    # Save results
    if output_path:
        summary = {
            'stage': stage,
            'split': split,
            'model_type': model_type,
            'n_trials': len(eval_trials),
            'mean_greedy_per': float(mean_greedy_per),
            'mean_per': float(mean_per),
            'mean_wer': float(mean_wer) if all_wer else None,
            'per_improvement_abs': float(mean_greedy_per - mean_per),
            'per_improvement_rel': float((mean_greedy_per - mean_per) / mean_greedy_per) if mean_greedy_per > 0 else 0,
            'elapsed_seconds': elapsed,
            'results': results_log,
        }
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved: {output_path}")

    return mean_per, mean_wer, results_log


def main():
    parser = argparse.ArgumentParser(description='Full Lead 4 Pipeline Evaluation')
    parser.add_argument('--stage', type=str, required=True,
                        choices=['greedy', 'kenlm', 'kenlm+opt',
                                 'ensemble', 'ensemble+kenlm',
                                 'ensemble+kenlm+opt', 'full'],
                        help='Pipeline stage to evaluate')

    # Model
    parser.add_argument('--model', type=str, default=None,
                        help='Single CTC model checkpoint')
    parser.add_argument('--model-type', type=str, default='enhanced',
                        choices=['enhanced', 'beyond', 'paper', 'auto'],
                        help='Model architecture type')
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--kernel-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--no-bidirectional', action='store_false', dest='bidirectional')
    parser.add_argument('--ensemble-config', type=str, default=None,
                        help='Ensemble config JSON')

    # KenLM
    parser.add_argument('--kenlm', type=str, default=None,
                        help='KenLM model path (.arpa or .bin)')
    parser.add_argument('--beam-width', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.005,
                        help='KenLM weight (0.005 optimal for mixture ensemble, 2.0 for single model)')
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--nbest', type=int, default=100)

    # OPT
    parser.add_argument('--opt', type=str, default=None,
                        help='OPT model name (e.g., facebook/opt-6.7b)')
    parser.add_argument('--opt-weight', type=float, default=0.5)
    parser.add_argument('--no-8bit', action='store_true',
                        help='Disable 8-bit quantization for OPT (use fp16 instead)')

    # Qwen correction
    parser.add_argument('--qwen', type=str, default=None,
                        help='Qwen LoRA adapter path')
    parser.add_argument('--qwen-prompt', type=str, default='B_dcond')
    parser.add_argument('--dcond-lift', action='store_true',
                        help='Use DCoND-LIFT diverse candidate correction')
    parser.add_argument('--mixture', action='store_true',
                        help='Use proper mixture model for ensemble (log-sum-exp)')

    # Data
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--split', type=str, default='within-day',
                        choices=['within-day', 'cross-session'],
                        help='Data split strategy')
    parser.add_argument('--eval-set', type=str, default='val',
                        choices=['val', 'test', 'both'])
    parser.add_argument('--max-trials', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    model_type = args.model_type
    if model_type == 'auto' and args.model:
        model_type = auto_detect_model_type(args.model)
        print(f"Auto-detected model type: {model_type}")

    model_kwargs = {
        'hidden': args.hidden,
        'n_layers': args.n_layers,
        'kernel_size': args.kernel_size,
        'stride': args.stride,
        'bidirectional': args.bidirectional,
    }

    evaluate_pipeline(
        stage=args.stage,
        model_path=args.model,
        model_type=model_type,
        model_kwargs=model_kwargs,
        ensemble_config=args.ensemble_config,
        kenlm_path=args.kenlm,
        kenlm_alpha=args.alpha,
        kenlm_beta=args.beta,
        beam_width=args.beam_width,
        nbest=args.nbest,
        opt_model_name=args.opt,
        opt_weight=args.opt_weight,
        opt_load_in_8bit=not args.no_8bit,
        qwen_adapter=args.qwen,
        qwen_prompt=args.qwen_prompt,
        use_diverse_candidates=args.dcond_lift,
        use_mixture=args.mixture,
        data_path=args.data,
        split=args.split,
        eval_set=args.eval_set,
        max_trials=args.max_trials,
        device=device,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
