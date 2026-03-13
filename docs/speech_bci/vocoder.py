"""LPCNet vocoder integration for audio synthesis.

Converts predicted LPC coefficients → audio waveform.
Supports Cython wrapper, CLI, and Griffin-Lim fallback.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from . import config


def synthesize_lpcnet_cython(lpc_features: np.ndarray) -> np.ndarray:
    """Synthesize audio using the Cython LPCNet wrapper.

    Parameters
    ----------
    lpc_features : (n_frames, 20) float32 predicted LPC coefficients

    Returns
    -------
    audio : (n_frames * 160,) int16 PCM at 16 kHz
    """
    try:
        from extensions.lpcnet.LPCNet import LPCNet as LPCNetDecoder
    except ImportError:
        raise RuntimeError("Cython LPCNet wrapper not available. Build it first.")

    decoder = LPCNetDecoder()
    audio_chunks = []
    for frame in lpc_features:
        samples = decoder.synthesize(frame.astype(np.float32))
        audio_chunks.append(samples)
    return np.concatenate(audio_chunks).astype(np.int16)


def synthesize_lpcnet_cli(lpc_features: np.ndarray,
                           lpcnet_bin: Path | None = None,
                           output_path: Path | None = None) -> np.ndarray:
    """Synthesize audio using LPCNet CLI.

    Parameters
    ----------
    lpc_features : (n_frames, 20) float32
    lpcnet_bin : path to lpcnet_demo binary
    output_path : optional path to save .wav output

    Returns
    -------
    audio : (n_samples,) int16
    """
    if lpcnet_bin is None:
        candidates = [
            config.PROJECT_DIR / "extensions" / "lpcnet" / "LPCNet" / "lpcnet_demo",
            Path("/usr/local/bin/lpcnet_demo"),
        ]
        for p in candidates:
            if p.exists():
                lpcnet_bin = p
                break
        if lpcnet_bin is None:
            raise RuntimeError("lpcnet_demo binary not found")

    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as feat_f, \
         tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as pcm_f:

        # Pad to 36 features (LPCNet expects 36, uses first 20)
        padded = np.zeros((len(lpc_features), 36), dtype=np.float32)
        padded[:, :config.N_LPC] = lpc_features
        padded.tofile(feat_f.name)

        subprocess.run(
            [str(lpcnet_bin), "-synthesis", feat_f.name, pcm_f.name],
            check=True, capture_output=True,
        )

        audio = np.fromfile(pcm_f.name, dtype=np.int16)

    if output_path:
        wavfile.write(str(output_path), config.AUDIO_FS, audio)

    return audio


def synthesize_griffin_lim(mel_spectrogram: np.ndarray,
                            sr: int = config.AUDIO_FS,
                            n_fft: int = 1024,
                            hop_length: int = 160,
                            n_iter: int = 64) -> np.ndarray:
    """Griffin-Lim fallback for mel spectrogram → audio.

    Only use this with mel targets (NOT LPC). Quality is lower than LPCNet
    but doesn't require any external compilation.
    """
    import librosa

    # mel_spectrogram should be in dB → convert back to power
    S = librosa.db_to_power(mel_spectrogram.T)
    audio = librosa.feature.inverse.mel_to_audio(
        S, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter,
    )
    return (audio * 32767).astype(np.int16)


def synthesize(lpc_features: np.ndarray, method: str = "auto",
               output_path: Path | None = None) -> np.ndarray:
    """Synthesize audio from LPC features using best available method.

    Parameters
    ----------
    lpc_features : (n_frames, 20) float32
    method : "auto", "cython", "cli"
    output_path : optional path to save .wav

    Returns
    -------
    audio : int16 PCM at 16 kHz
    """
    if method == "auto":
        try:
            audio = synthesize_lpcnet_cython(lpc_features)
        except RuntimeError:
            try:
                audio = synthesize_lpcnet_cli(lpc_features)
            except RuntimeError:
                raise RuntimeError(
                    "No LPCNet backend available. Build the Cython wrapper or "
                    "compile lpcnet_demo. See extensions/lpcnet/README."
                )
    elif method == "cython":
        audio = synthesize_lpcnet_cython(lpc_features)
    elif method == "cli":
        audio = synthesize_lpcnet_cli(lpc_features)
    else:
        raise ValueError(f"Unknown method: {method}")

    if output_path:
        wavfile.write(str(output_path), config.AUDIO_FS, audio)

    return audio
