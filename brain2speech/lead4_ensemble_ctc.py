#!/usr/bin/env python3
"""Multi-model CTC ensemble with weighted log-probability averaging.

Supports loading multiple trained GRU/Conformer models and combining their
CTC outputs via weighted log-probability averaging or diverse candidate
generation (DCoND-LIFT style).

Usage:
    python brain2speech/lead4_ensemble_ctc.py \
        --config brain2speech/configs/lead4_ensemble.json \
        --eval-set val --output brain2speech/results/L4_ensemble.json

References:
    - Source 6 (MONA): 10 models → 13.7% → 7.3% with LLM
    - Source 8 (DCoND): 10 decoders, different seeds → 8.09% → 5.77%
    - Source 10 (BIT): 10-ensemble → 10.22% (vs 15.67% single), ~35% relative
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_TO_ARPABET, N_CLASSES

CTC_BLANK = 40


class CTCEnsemble:
    """Multi-model CTC ensemble.

    Supports two modes:
    1. Averaged log-probs: weighted average of log softmax outputs
    2. Diverse candidates: each model produces independent top-1 for DCoND-LIFT
    """

    def __init__(self, checkpoint_configs, device='cuda'):
        """
        Args:
            checkpoint_configs: List of dicts with keys:
                - path: checkpoint file path
                - model_class: string name of model class
                - model_kwargs: dict of model constructor args
                - weight: ensemble weight (default: 1.0)
            device: torch device
        """
        self.device = device
        self.models = []
        self.weights = []
        self.names = []

        total_w = sum(c.get('weight', 1.0) for c in checkpoint_configs)

        for cfg in checkpoint_configs:
            model = self._load_model(cfg)
            model.to(device).eval()
            self.models.append(model)
            self.weights.append(cfg.get('weight', 1.0) / total_w)
            self.names.append(cfg.get('name', Path(cfg['path']).stem))

        print(f"Ensemble loaded: {len(self.models)} models")
        for name, w in zip(self.names, self.weights):
            print(f"  {name}: weight={w:.3f}")

    def _load_model(self, cfg):
        """Load a model from config dict."""
        model_class_name = cfg.get('model_class', 'BeyondPaperGRU')
        model_kwargs = cfg.get('model_kwargs', {})

        # Import model class
        if model_class_name == 'BeyondPaperGRU':
            from train_beyond_paper import BeyondPaperGRU
            model = BeyondPaperGRU(**model_kwargs)
        elif model_class_name == 'ConformerDecoder':
            from train_beyond_paper import ConformerDecoder
            model = ConformerDecoder(**model_kwargs)
        elif model_class_name == 'PaperGRUSimple':
            from train_paper_replica import PaperGRUSimple
            model = PaperGRUSimple(**model_kwargs)
        elif model_class_name == 'ImprovedGRUCTCEncoder':
            from train_ctc_improved import ImprovedGRUCTCEncoder
            model = ImprovedGRUCTCEncoder(**model_kwargs)
        elif model_class_name == 'ConformerCTCEncoder':
            from train_ctc_improved import ConformerCTCEncoder
            model = ConformerCTCEncoder(**model_kwargs)
        elif model_class_name == 'EnhancedGRU':
            from lead1_train_gru_v2 import EnhancedGRU
            model = EnhancedGRU(**model_kwargs)
        elif model_class_name == 'PaperExactDCoND':
            from lead2_train_dcond import PaperExactDCoND
            model = PaperExactDCoND(**model_kwargs)
        elif model_class_name == 'DCoNDDecoder':
            from lead2_train_dcond import DCoNDDecoder
            model = DCoNDDecoder(**model_kwargs)
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")

        # Load weights
        state = torch.load(cfg['path'], map_location='cpu', weights_only=False)
        # Handle wrapped state dicts (from lead1_train_gru_v2.py)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state, strict=False)

        # Set train sessions if available
        if hasattr(model, 'set_train_sessions') and 'train_session_idxs' in cfg:
            model.set_train_sessions(cfg['train_session_idxs'])

        return model

    def _get_mono_logits(self, model, features, session_ids):
        """Get monophone logits from any model type.

        DCoND models have forward_mono() for 41-class output.
        Other models return 41-class directly from forward().
        """
        if hasattr(model, 'forward_mono'):
            return model.forward_mono(features, session_ids)
        return model(features, session_ids)

    def get_ensemble_log_probs(self, features, session_ids):
        """Weighted average of log probs across all models.

        Args:
            features: (B, T, C) input tensor
            session_ids: (B,) session indices

        Returns:
            (B, T_out, n_classes+1) averaged log probabilities
        """
        all_log_probs = []

        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                logits = self._get_mono_logits(model, features, session_ids)
                log_probs = F.log_softmax(logits, dim=-1)
                all_log_probs.append(log_probs * weight)

        # Sum weighted log probs (approximation of mixture model)
        # More accurate would be: log(sum(w_i * exp(log_p_i)))
        # But for CTC with similar models, weighted sum of log_probs is standard
        combined = torch.stack(all_log_probs).sum(dim=0)
        return combined

    def get_ensemble_log_probs_mixture(self, features, session_ids):
        """Proper mixture model: log(sum(w_i * p_i)).

        More expensive but theoretically correct for combining distributions.
        """
        all_probs = []

        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                logits = self._get_mono_logits(model, features, session_ids)
                probs = F.softmax(logits, dim=-1) * weight
                all_probs.append(probs)

        combined_probs = torch.stack(all_probs).sum(dim=0)
        # Avoid log(0)
        combined_probs = combined_probs.clamp(min=1e-10)
        return torch.log(combined_probs)

    def get_diverse_candidates(self, features, session_ids, lm=None,
                                beam_width=150, alpha=0.3, beta=0.0):
        """Get top-1 from each model for DCoND-LIFT style correction.

        Source 8: Feed all 10 candidates to correction LLM.

        Args:
            features: (B, T, C) — only B=1 supported
            session_ids: (B,)
            lm: KenLMPhonemeLM instance (or None for greedy)
            beam_width: Per-model beam width
            alpha: LM weight
            beta: Length bonus

        Returns:
            List of dicts with 'model_name', 'phonemes', 'score'
        """
        from lead4_decode_kenlm import decode_single

        candidates = []

        with torch.no_grad():
            for i, model in enumerate(self.models):
                logits = self._get_mono_logits(model, features, session_ids)
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs_np = log_probs[0].cpu().numpy()

                results = decode_single(log_probs_np, lm,
                                         beam_width=beam_width,
                                         alpha=alpha, beta=beta, nbest=1)
                if results:
                    phoneme_str, indices, score = results[0]
                    candidates.append({
                        'model_idx': i,
                        'model_name': self.names[i],
                        'phonemes': phoneme_str,
                        'score': float(score),
                    })

        return candidates

    def evaluate(self, trials, device=None):
        """Evaluate ensemble on trial data.

        Returns per-model PER and ensemble PER.
        """
        import editdistance

        if device is None:
            device = self.device

        # Per-model results
        model_edits = [0] * len(self.models)
        model_lens = [0] * len(self.models)
        ensemble_edits, ensemble_lens = 0, 0

        for trial in trials:
            if 'features_raw' in trial:
                features = torch.FloatTensor(trial['features_raw']).unsqueeze(0).to(device)
            else:
                features = torch.FloatTensor(trial['features']).unsqueeze(0).to(device)
            session_id = torch.LongTensor([trial.get('session_idx', 0)]).to(device)

            target = trial.get('phoneme_indices', [])
            if hasattr(target, 'tolist'):
                target = target.tolist()
            target_phones = [CLASS_TO_ARPABET[int(t)] for t in target
                             if int(t) in CLASS_TO_ARPABET]

            # Per-model greedy decode
            for m_idx, model in enumerate(self.models):
                with torch.no_grad():
                    logits = model(features, session_id)
                    pred = logits[0].argmax(dim=-1).cpu().tolist()

                decoded = []
                prev = -1
                for t in pred:
                    if t != prev and t != CTC_BLANK:
                        decoded.append(CLASS_TO_ARPABET.get(t, '?'))
                    prev = t

                edits = editdistance.eval(decoded, target_phones)
                model_edits[m_idx] += edits
                model_lens[m_idx] += len(target_phones)

            # Ensemble greedy decode
            with torch.no_grad():
                combined = self.get_ensemble_log_probs(features, session_id)
                pred = combined[0].argmax(dim=-1).cpu().tolist()

            decoded = []
            prev = -1
            for t in pred:
                if t != prev and t != CTC_BLANK:
                    decoded.append(CLASS_TO_ARPABET.get(t, '?'))
                prev = t

            edits = editdistance.eval(decoded, target_phones)
            ensemble_edits += edits
            ensemble_lens += len(target_phones)

        # Print results
        print("\nPer-model PER:")
        for m_idx, name in enumerate(self.names):
            per = model_edits[m_idx] / max(model_lens[m_idx], 1)
            print(f"  {name}: {per:.4f}")

        ensemble_per = ensemble_edits / max(ensemble_lens, 1)
        print(f"\nEnsemble PER: {ensemble_per:.4f}")

        return ensemble_per


def load_ensemble_config(config_path):
    """Load ensemble configuration from JSON file.

    Expected format:
    {
        "models": [
            {
                "name": "gru_baseline",
                "path": "brain2speech/results/L4_baseline_best.pt",
                "model_class": "BeyondPaperGRU",
                "model_kwargs": {
                    "n_features_per_frame": 256,
                    "n_classes": 41,
                    "hidden": 512,
                    "n_layers": 5,
                    "n_sessions": 24,
                    "bidirectional": true
                },
                "weight": 1.0
            },
            ...
        ]
    }
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['models']


def main():
    parser = argparse.ArgumentParser(description='CTC Ensemble Evaluator')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to ensemble config JSON')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to H5 dataset')
    parser.add_argument('--eval-set', type=str, default='val',
                        choices=['val', 'test', 'both'])
    parser.add_argument('--max-trials', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Load config and build ensemble
    configs = load_ensemble_config(args.config)
    ensemble = CTCEnsemble(configs, device=device)

    # Load data
    from beam_search_decode import load_trials, split_trials
    data_path = args.data or '/mnt/home/vincent.wilmet/brain2speech/data/sentences_paper_256d.h5'
    trials = load_trials(data_path, max_trials=args.max_trials)
    train_trials, val_trials, test_trials, n_sessions, train_sids = split_trials(trials)

    if args.eval_set == 'val':
        eval_trials = val_trials
    elif args.eval_set == 'test':
        eval_trials = test_trials
    else:
        eval_trials = val_trials + test_trials

    # Set train sessions for all models
    for model in ensemble.models:
        if hasattr(model, 'set_train_sessions'):
            model.set_train_sessions(train_sids)

    # Evaluate
    ensemble_per = ensemble.evaluate(eval_trials, device=device)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'ensemble_per': ensemble_per,
                'n_models': len(ensemble.models),
                'model_names': ensemble.names,
                'weights': ensemble.weights,
            }, f, indent=2)
        print(f"Results saved: {args.output}")


if __name__ == '__main__':
    main()
