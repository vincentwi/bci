"""Energy-based Voice Activity Detection ground truth from audio."""

import numpy as np
import librosa

from . import config


def energy_based_vad(audio: np.ndarray, fs: int = config.AUDIO_FS,
                     frame_ms: int = config.HG_HOP_MS,
                     n_mels: int = 40, threshold_db: float = -40.0) -> np.ndarray:
    """Compute frame-level VAD labels from audio energy.

    Uses log mel filterbank energy (Kaldi-style), which is what the paper's
    EnergyBasedVad uses. Frames with energy above the threshold are labeled
    as speech.

    Parameters
    ----------
    audio : (n_samples,) audio waveform
    fs : sample rate in Hz
    frame_ms : frame hop in milliseconds (matches HG frame rate)
    n_mels : number of mel bands for energy computation
    threshold_db : energy threshold in dB (relative to max)

    Returns
    -------
    vad : (n_frames,) binary int8 — 1=speech, 0=silence
    """
    audio_float = audio.astype(np.float32) / (np.abs(audio).max() + 1e-10)

    hop_length = int(frame_ms / 1000 * fs)
    S = librosa.feature.melspectrogram(
        y=audio_float, sr=fs, n_mels=n_mels,
        n_fft=1024, hop_length=hop_length, fmax=8000,
    )
    log_energy = librosa.power_to_db(S.sum(axis=0), ref=np.max)

    vad = (log_energy > threshold_db).astype(np.int8)

    # Smooth: remove isolated speech/silence frames (median filter)
    from scipy.ndimage import median_filter
    vad = median_filter(vad, size=5).astype(np.int8)

    return vad


def align_vad_to_hg(vad_labels: np.ndarray, n_hg_frames: int) -> np.ndarray:
    """Align VAD labels to HG frame count.

    Both should be at 100 Hz (10 ms hop), but edge effects from different
    window sizes may cause slight length mismatches. This handles that.

    Parameters
    ----------
    vad_labels : (n_vad_frames,) binary labels
    n_hg_frames : target number of HG frames

    Returns
    -------
    aligned : (n_hg_frames,) binary labels
    """
    n_vad = len(vad_labels)
    if n_vad == n_hg_frames:
        return vad_labels
    elif n_vad > n_hg_frames:
        return vad_labels[:n_hg_frames]
    else:
        # Pad with silence
        return np.pad(vad_labels, (0, n_hg_frames - n_vad),
                      mode="constant", constant_values=0)


def vad_from_stim_code(stim_code: np.ndarray, hg_times: np.ndarray,
                        fs: int = config.FS) -> np.ndarray:
    """Fallback: derive VAD labels from StimulusCode.

    Less accurate than energy-based (includes silence within trial windows),
    but doesn't require audio. Useful for quick validation.

    Parameters
    ----------
    stim_code : (n_samples,) uint16 — 0=silence, >0=word
    hg_times : (n_frames,) float64 — HG frame center times in seconds

    Returns
    -------
    vad : (n_frames,) binary int8
    """
    vad = np.zeros(len(hg_times), dtype=np.int8)
    for i, t in enumerate(hg_times):
        sample_idx = int(t * fs)
        if sample_idx < len(stim_code) and stim_code[sample_idx] > 0:
            vad[i] = 1
    return vad
