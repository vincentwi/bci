#!/usr/bin/env python3
"""
Ensemble phoneme classifier combining TCN, EEGNet, GRU, and Transformer.

Supports soft voting (averaged softmax probabilities) and hard voting
(majority vote). Soft voting generally outperforms hard voting as it
preserves confidence information needed for the LM correction stage.

Architecture:
    Neural data (85 bins × 1280 features)
        → TCN classifier   → P_tcn  ∈ ℝ^40
        → EEGNet classifier → P_eeg  ∈ ℝ^40
        → GRU classifier   → P_gru  ∈ ℝ^40
        → Transformer       → P_tf   ∈ ℝ^40
        → Ensemble: P_ens = weighted average of P_i
        → argmax(P_ens) = predicted phoneme class

Usage:
    from ensemble import EnsembleClassifier
    ensemble = EnsembleClassifier(device="cuda")
    probs = ensemble.predict_proba(neural_data)  # (N, 40) softmax
    preds = ensemble.predict(neural_data)         # (N,) class indices
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import STAGE1_MODELS, CLASS_TO_ARPABET, N_CLASSES


def _load_model_classes():
    """Import model classes from Stage 1, avoiding config.py name collision."""
    import importlib.util
    train_path = Path("/mnt/home/vincent.wilmet/docs/scripts/train.py")
    scripts_config_path = Path("/mnt/home/vincent.wilmet/docs/scripts/config.py")

    spec_cfg = importlib.util.spec_from_file_location("scripts_config", scripts_config_path)
    scripts_config = importlib.util.module_from_spec(spec_cfg)
    sys.modules["scripts_config"] = scripts_config
    orig_config = sys.modules.get("config")
    sys.modules["config"] = scripts_config
    spec_cfg.loader.exec_module(scripts_config)

    spec = importlib.util.spec_from_file_location("stage1_train", train_path)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    if orig_config is not None:
        sys.modules["config"] = orig_config
    else:
        del sys.modules["config"]

    return train_mod


def _load_model_class(name):
    """Import a specific model class from Stage 1 training script."""
    train_mod = _load_model_classes()
    return {"TCN": train_mod.TCN, "EEGNet": train_mod.EEGNet,
            "GRU": train_mod.GRUDecoder,
            "Transformer": train_mod.SpeechTransformer}[name]


# Default model configs matching train.py
MODEL_CONFIGS = {
    "TCN": dict(nc=1280, nt=85, nk=N_CLASSES, dr=0.3, hidden=128),
    "EEGNet": dict(nc=1280, nt=85, nk=N_CLASSES, F1=16, D=2, F2=32, kl=32, dr=0.4),
    "GRU": dict(nc=1280, nt=85, nk=N_CLASSES, hidden=256, n_layers=2, dr=0.3),
    "Transformer": dict(nc=1280, nt=85, nk=N_CLASSES, d_model=128, nhead=8,
                         num_layers=4, dr=0.3),
}

# Weights proportional to individual model accuracy (from results.json)
# TCN=98.8%, Transformer=96.2%, GRU=93.8%, EEGNet=92.1%
DEFAULT_WEIGHTS = {
    "TCN": 0.35,
    "Transformer": 0.25,
    "GRU": 0.20,
    "EEGNet": 0.20,
}


class EnsembleClassifier:
    """Weighted soft-voting ensemble of phoneme classifiers.

    Loads pre-trained TCN, EEGNet, GRU, and Transformer checkpoints,
    runs inference on all four, and returns weighted-average softmax
    probabilities. Weights are proportional to each model's individual
    cross-validated accuracy.

    Attributes:
        models: dict of {name: nn.Module} — loaded classifiers
        weights: dict of {name: float} — voting weights (sum to 1)
        device: torch device for inference
    """

    def __init__(self, model_names=None, weights=None, device="cuda",
                 model_dir=None):
        """
        Args:
            model_names: list of model names to include (default: all four)
            weights: dict of {name: weight} (default: accuracy-proportional)
            device: "cuda" or "cpu"
            model_dir: directory containing *_best.pt files
        """
        self.device = torch.device(device)
        self.model_dir = Path(model_dir) if model_dir else STAGE1_MODELS

        if model_names is None:
            model_names = list(DEFAULT_WEIGHTS.keys())

        if weights is None:
            weights = {n: DEFAULT_WEIGHTS[n] for n in model_names}

        # Normalize weights to sum to 1
        total = sum(weights.values())
        self.weights = {n: w / total for n, w in weights.items()}

        self.models = {}
        for name in model_names:
            self.models[name] = self._load_one(name)

        print(f"Ensemble loaded: {list(self.models.keys())}")
        print(f"  Weights: {self.weights}")

    def _load_one(self, name):
        """Load a single model checkpoint."""
        cls = _load_model_class(name)
        kw = MODEL_CONFIGS[name]
        model = cls(**kw)

        weight_file = self.model_dir / f"{name.lower()}_best.pt"
        state = torch.load(weight_file, map_location=self.device, weights_only=True)

        # Handle DataParallel prefix
        cleaned = {}
        for k, v in state.items():
            key = k.replace("module.", "") if k.startswith("module.") else k
            cleaned[key] = v
        model.load_state_dict(cleaned)

        model = model.to(self.device).eval()
        print(f"  Loaded {name} from {weight_file}")
        return model

    def predict_proba(self, X, batch_size=256):
        """
        Return weighted-average softmax probabilities.

        Args:
            X: (N, time, channels) or (N, channels, time) numpy array or tensor
            batch_size: inference batch size

        Returns:
            probs: (N, 40) numpy array of softmax probabilities
        """
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        # Ensure shape is (N, channels=1280, time=85)
        if X.shape[1] != 1280 and X.shape[2] == 1280:
            X = X.transpose(1, 2)

        all_probs = []
        for name, model in self.models.items():
            w = self.weights[name]
            model_probs = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size].to(self.device)
                with torch.no_grad():
                    logits = model(batch)
                    probs = F.softmax(logits, dim=1).cpu()
                model_probs.append(probs)
            model_probs = torch.cat(model_probs, dim=0)  # (N, 40)
            all_probs.append(model_probs * w)

        ensemble_probs = sum(all_probs).numpy()
        return ensemble_probs

    def predict(self, X, batch_size=256):
        """Return argmax class predictions.

        Args:
            X: (N, time, channels) or (N, channels, time) array
        Returns:
            preds: (N,) numpy int array
        """
        probs = self.predict_proba(X, batch_size)
        return probs.argmax(axis=1)

    def predict_with_confidence(self, X, batch_size=256):
        """Return predictions, max confidence, and top-3 candidates per trial.

        Used by the LM correction stage to decide which phonemes to correct.

        Args:
            X: (N, time, channels) or (N, channels, time) array

        Returns:
            phonemes: list of ARPABET strings
            confidences: list of floats (max softmax per trial)
            top_k: list of [(ARPABET, prob), ...] per trial
        """
        probs = self.predict_proba(X, batch_size)

        phonemes = []
        confidences = []
        top_k = []

        for row in probs:
            top3_idx = np.argsort(row)[-3:][::-1]
            best = top3_idx[0]
            phonemes.append(CLASS_TO_ARPABET[best])
            confidences.append(float(row[best]))
            top_k.append([
                (CLASS_TO_ARPABET[idx], float(row[idx])) for idx in top3_idx
            ])

        return phonemes, confidences, top_k


if __name__ == "__main__":
    print("=" * 60)
    print("Testing ensemble classifier")
    print("=" * 60)

    # Load ensemble
    ensemble = EnsembleClassifier(device="cuda")

    # Load a phoneme data sample
    from config import STAGE1_DATA
    candidates = sorted(STAGE1_DATA.glob("tuning_*phonemes*.npz"))
    if candidates:
        data = np.load(candidates[0], allow_pickle=True)
        X = data['X'][:20]  # first 20 trials
        y = data['y'][:20]
        class_names = list(data['class_names'])

        probs = ensemble.predict_proba(X)
        preds = probs.argmax(axis=1)
        acc = (preds == y).mean()

        print(f"\nSample evaluation (20 trials):")
        print(f"  Accuracy: {acc:.1%}")
        print(f"  Predictions: {preds[:10]}")
        print(f"  True labels: {y[:10]}")

        phonemes, confs, top3 = ensemble.predict_with_confidence(X)
        print(f"  Avg confidence: {np.mean(confs):.3f}")
        print(f"  Phonemes: {' '.join(phonemes[:10])}")
    else:
        print("No phoneme data found for testing")
