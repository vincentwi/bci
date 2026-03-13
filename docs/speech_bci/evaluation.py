"""Evaluation metrics for speech BCI models."""

import numpy as np
import torch

from . import config


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Pearson correlation between flattened arrays."""
    pred_f = pred.flatten()
    target_f = target.flatten()
    if np.std(pred_f) < 1e-10 or np.std(target_f) < 1e-10:
        return 0.0
    return float(np.corrcoef(pred_f, target_f)[0, 1])


def pearson_r_per_dim(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute per-dimension Pearson correlation.

    Parameters
    ----------
    pred : (T, D) predictions
    target : (T, D) ground truth

    Returns
    -------
    r : (D,) per-dimension correlation
    """
    D = pred.shape[1]
    r = np.zeros(D)
    for d in range(D):
        if np.std(pred[:, d]) > 1e-10 and np.std(target[:, d]) > 1e-10:
            r[d] = np.corrcoef(pred[:, d], target[:, d])[0, 1]
    return r


def frame_accuracy(pred_vad: np.ndarray, true_vad: np.ndarray) -> float:
    """Frame-level VAD accuracy."""
    n = min(len(pred_vad), len(true_vad))
    return float((pred_vad[:n] == true_vad[:n]).mean())


def classification_accuracy(pred_labels: np.ndarray,
                             true_labels: np.ndarray) -> float:
    """Word classification accuracy."""
    return float((pred_labels == true_labels).mean())


def mcd(pred_lpc: np.ndarray, target_lpc: np.ndarray,
        n_cepstral: int = 18) -> float:
    """Mel Cepstral Distortion between predicted and target LPC features.

    Only uses the first n_cepstral dimensions (excluding pitch).
    Lower is better.

    Parameters
    ----------
    pred_lpc : (T, 20) predicted LPC
    target_lpc : (T, 20) ground truth LPC

    Returns
    -------
    mcd : float, in dB
    """
    diff = pred_lpc[:, :n_cepstral] - target_lpc[:, :n_cepstral]
    frame_dist = np.sqrt(2 * np.sum(diff ** 2, axis=1))
    return float(np.mean(frame_dist)) * (10.0 / np.log(10.0))


@torch.no_grad()
def predict_all(model, dataloader, device="mps"):
    """Run model on entire dataset, return concatenated predictions and targets.

    Returns
    -------
    all_preds : np.ndarray
    all_targets : np.ndarray
    trial_info : list of dicts with word labels and lengths
    """
    model.eval()
    all_preds = []
    all_targets = []
    trial_info = []

    for batch in dataloader:
        if len(batch) == 3:
            hg, targets, lengths = batch
        else:
            hg, targets = batch
            lengths = [hg.shape[1]]

        for i in range(len(hg)):
            L = lengths[i] if isinstance(lengths, list) else lengths[i].item()
            hg_i = hg[i:i+1, :L].to(device)

            if hasattr(model, "forward"):
                output = model(hg_i)
                if isinstance(output, tuple):
                    output = output[0]

            all_preds.append(output.cpu().numpy()[0])
            if isinstance(targets, torch.Tensor) and targets.dim() > 1:
                all_targets.append(targets[i, :L].numpy())
            else:
                all_targets.append(targets[i].item() if isinstance(targets, torch.Tensor) else targets[i])
            trial_info.append({"length": L})

    if isinstance(all_targets[0], (int, float, np.integer, np.floating)):
        return np.array([p.argmax(-1) if p.ndim > 0 else p for p in all_preds]), \
               np.array(all_targets), trial_info

    return np.concatenate(all_preds, axis=0), \
           np.concatenate(all_targets, axis=0), trial_info
