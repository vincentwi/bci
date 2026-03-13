"""LPC feature extraction for LPCNet vocoder.

Three extraction paths:
  A. Cython wrapper (matches paper exactly) — needs LPCNet build
  B. dump_data CLI (same features, no Cython) — needs compiled LPCNet binary
  C. Pure Python approximation (fallback) — does NOT match Bark-scale warping

Use extract_lpc() which auto-selects the best available method.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa

from . import config

# Will be set if Cython wrapper is available
_cython_encoder = None
_lpcnet_bin = None


def _try_cython_import():
    """Attempt to import the Cython LPCNet wrapper."""
    global _cython_encoder
    try:
        from extensions.lpcnet.LPCNet import LPCFeatureEncoder
        _cython_encoder = LPCFeatureEncoder
        return True
    except ImportError:
        return False


def _find_lpcnet_binary() -> Path | None:
    """Find the compiled dump_data binary."""
    global _lpcnet_bin
    candidates = [
        config.PROJECT_DIR / "extensions" / "lpcnet" / "LPCNet" / "dump_data",
        config.PROJECT_DIR / "extensions" / "lpcnet" / "LPCNet" / "src" / "dump_data",
        Path("/usr/local/bin/dump_data"),
    ]
    for p in candidates:
        if p.exists():
            _lpcnet_bin = p
            return p
    return None


def extract_lpc_cython(audio: np.ndarray, fs: int = config.AUDIO_FS) -> np.ndarray:
    """Extract LPC features using the Cython wrapper (Path A).

    Parameters
    ----------
    audio : (n_samples,) int16 PCM audio at 16 kHz
    fs : sample rate

    Returns
    -------
    features : (n_frames, 20) float32 — 18 Bark cepstrals + 2 pitch params
    """
    if _cython_encoder is None:
        if not _try_cython_import():
            raise RuntimeError("Cython LPCNet wrapper not available. Build it first.")

    encoder = _cython_encoder()
    features = encoder.compute_LPC_features(audio.astype(np.int16))
    return features[:, :config.N_LPC].astype(np.float32)


def extract_lpc_cli(audio_path: Path,
                    lpcnet_bin: Path | None = None) -> np.ndarray:
    """Extract LPC features using dump_data CLI (Path B).

    Parameters
    ----------
    audio_path : path to 16 kHz, 16-bit signed PCM .wav or .raw file
    lpcnet_bin : path to compiled dump_data binary

    Returns
    -------
    features : (n_frames, 20) float32
    """
    bin_path = lpcnet_bin or _lpcnet_bin or _find_lpcnet_binary()
    if bin_path is None:
        raise RuntimeError("dump_data binary not found. Compile LPCNet first.")

    with tempfile.NamedTemporaryFile(suffix=".pcm") as pcm_f, \
         tempfile.NamedTemporaryFile(suffix=".f32") as feat_f:

        # Convert to raw PCM if needed
        if str(audio_path).endswith(".wav"):
            import scipy.io.wavfile as wavfile
            fs, audio = wavfile.read(str(audio_path))
            audio.astype(np.int16).tofile(pcm_f.name)
        else:
            # Assume already raw PCM
            import shutil
            shutil.copy(str(audio_path), pcm_f.name)

        subprocess.run(
            [str(bin_path), "-test", pcm_f.name, feat_f.name],
            check=True, capture_output=True,
        )

        # dump_data outputs 36 features per frame; we want the first 20
        raw = np.fromfile(feat_f.name, dtype=np.float32)
        n_features_per_frame = 36
        features = raw.reshape(-1, n_features_per_frame)[:, :config.N_LPC]

    return features.astype(np.float32)


def extract_lpc_python(audio: np.ndarray, fs: int = config.AUDIO_FS,
                       order: int = 18,
                       hop_ms: int = config.HG_HOP_MS) -> np.ndarray:
    """Pure Python LPC extraction (Path C — fallback/approximation).

    WARNING: Does NOT match LPCNet's Bark-scale warping. The coefficients
    are standard linear prediction coefficients, not Bark-scale cepstrals.
    Use only as a development placeholder.

    Parameters
    ----------
    audio : (n_samples,) audio waveform
    fs : sample rate
    order : LPC order (18 to match paper's 18 cepstral coefficients)
    hop_ms : frame hop in milliseconds

    Returns
    -------
    features : (n_frames, 20) float32 — 18 LPC coefficients + 2 pitch params
    """
    audio_float = audio.astype(np.float32) / (np.abs(audio).max() + 1e-10)

    hop_length = int(hop_ms / 1000 * fs)
    win_length = int(0.025 * fs)  # 25 ms analysis window
    n_frames = (len(audio_float) - win_length) // hop_length + 1

    lpc_features = np.zeros((n_frames, config.N_LPC), dtype=np.float32)

    # Extract pitch using pyin
    f0, voiced_flag, _ = librosa.pyin(
        audio_float, fmin=50, fmax=500, sr=fs,
        hop_length=hop_length,
    )
    # Convert to pitch period (samples) and pitch correlation
    pitch_period = np.where(
        (f0 > 0) & np.isfinite(f0),
        fs / f0,
        0.0,
    )
    pitch_corr = voiced_flag.astype(np.float32) if voiced_flag is not None else np.zeros(n_frames)

    # Truncate to match frame count
    n_common = min(n_frames, len(pitch_period), len(pitch_corr))
    lpc_features = np.zeros((n_common, config.N_LPC), dtype=np.float32)

    # Extract LPC per frame
    for i in range(n_common):
        start = i * hop_length
        end = start + win_length
        if end > len(audio_float):
            break
        frame = audio_float[start:end]

        # Apply Hamming window
        frame = frame * np.hamming(len(frame))

        # LPC analysis
        try:
            a = librosa.lpc(frame, order=order)
            lpc_features[i, :order] = a[1:]  # skip a[0] which is always 1
        except Exception:
            pass  # leave as zeros for silent frames

        lpc_features[i, 18] = pitch_period[i]
        lpc_features[i, 19] = pitch_corr[i]

    return lpc_features


def preprocess_audio_for_lpc(audio: np.ndarray, fs: int,
                              vad_labels: np.ndarray | None = None,
                              target_db: float = -3.0) -> np.ndarray:
    """Preprocess audio before LPC extraction.

    From the paper's prepare_corpus.py:
    1. Normalize loudness to target_db on speech segments
    2. Apply 16 ms zero-padding for LPCNet filter delay compensation

    Parameters
    ----------
    audio : (n_samples,) audio waveform
    fs : sample rate
    vad_labels : optional frame-level VAD for speech-segment normalization
    target_db : target loudness in dB

    Returns
    -------
    processed : (n_samples,) preprocessed audio
    """
    audio = audio.astype(np.float32).copy()

    # Loudness normalization on speech segments
    if vad_labels is not None:
        hop = int(config.HG_HOP_MS / 1000 * fs)
        speech_mask = np.zeros(len(audio), dtype=bool)
        for i, v in enumerate(vad_labels):
            if v > 0:
                start = i * hop
                end = min(start + hop, len(audio))
                speech_mask[start:end] = True

        speech_energy = np.sqrt(np.mean(audio[speech_mask] ** 2) + 1e-10)
        target_energy = 10 ** (target_db / 20)
        if speech_energy > 1e-10:
            audio *= target_energy / speech_energy
    else:
        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        target_energy = 10 ** (target_db / 20)
        if rms > 1e-10:
            audio *= target_energy / rms

    # 16 ms filter delay compensation
    delay_samples = int(0.016 * fs)
    audio = np.pad(audio, (delay_samples, 0), mode="constant")

    return audio


def extract_lpc(audio: np.ndarray, fs: int = config.AUDIO_FS,
                audio_path: Path | None = None,
                method: str = "auto") -> np.ndarray:
    """Extract LPC features using the best available method.

    Parameters
    ----------
    audio : (n_samples,) audio waveform
    fs : sample rate
    audio_path : path to audio file (needed for CLI method)
    method : "auto", "cython", "cli", or "python"

    Returns
    -------
    features : (n_frames, 20) float32
    """
    if method == "auto":
        if _try_cython_import():
            method = "cython"
        elif _find_lpcnet_binary() is not None:
            method = "cli"
        else:
            method = "python"

    if method == "cython":
        return extract_lpc_cython(audio, fs)
    elif method == "cli":
        if audio_path is None:
            raise ValueError("audio_path required for CLI method")
        return extract_lpc_cli(audio_path)
    else:
        return extract_lpc_python(audio, fs)
