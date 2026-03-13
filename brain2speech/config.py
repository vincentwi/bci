"""
Configuration for the brain-to-speech pipeline.
Stages 2 (LM correction) and 3 (audio synthesis).
"""
from pathlib import Path
import os

# ── Paths ──
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Stage 1 artifacts (from docs/scripts training pipeline)
STAGE1_ROOT = Path("/mnt/home/vincent.wilmet/docs")
STAGE1_MODELS = STAGE1_ROOT / "models"
STAGE1_DATA = STAGE1_ROOT / "data" / "processed"

# ── ARPABET mapping ──
# Maps class index (from phoneme_class_names.npy) → standard ARPABET symbol
# Verified order: ['Bah','Chah','DO_NOTHING','Dah','Fah','Gah','Hah','Jah',
#   'Kah','Lah','LettER','Mah','Nah','Ngah','Pah','Rah','Sah','Shah','THe',
#   'Tah','Thah','Vah','Wah','Yah','Zah','Zhah','chOIce','drEss','fAce',
#   'fOOt','flEEce','gOAt','gOOse','kIt','lOt','mOUth','prIce','strUt',
#   'thOUGHt','trAp']
CLASS_TO_ARPABET = {
    0: 'B', 1: 'CH', 2: 'SIL',   # DO_NOTHING → silence
    3: 'D', 4: 'F', 5: 'G', 6: 'HH', 7: 'JH', 8: 'K', 9: 'L',
    10: 'ER', 11: 'M', 12: 'N', 13: 'NG', 14: 'P', 15: 'R',
    16: 'S', 17: 'SH', 18: 'DH', 19: 'T', 20: 'TH',
    21: 'V', 22: 'W', 23: 'Y', 24: 'Z', 25: 'ZH',
    26: 'OY', 27: 'EH', 28: 'EY', 29: 'UH', 30: 'IY',
    31: 'OW', 32: 'UW', 33: 'IH', 34: 'AA', 35: 'AW',
    36: 'AY', 37: 'AH', 38: 'AO', 39: 'AE',
}

ARPABET_TO_CLASS = {v: k for k, v in CLASS_TO_ARPABET.items()}

# All 39 ARPABET phonemes (excluding SIL)
ARPABET_39 = sorted(set(CLASS_TO_ARPABET.values()) - {'SIL'})

# Vowels (need stress markers for ElevenLabs)
VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
           'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}

# ── ElevenLabs ──
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "8fdaac8c5eee2990402a7304e3b9acef")
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
ELEVENLABS_MODEL_ID = "eleven_flash_v2"  # Required for phoneme tag support
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # Default voice

# ── LM correction ──
QWEN_MODEL_NAME = "Qwen/Qwen3.5-2B"
LORA_ADAPTER_PATH = MODELS_DIR / "qwen_phoneme_corrector" / "final"
SYSTEM_PROMPT = (
    "You are a phoneme error correction model for a brain-computer interface. "
    "Given a noisy ARPABET phoneme sequence decoded from neural signals, "
    "output the corrected sequence. Only output the corrected phonemes, nothing else."
)

# ── Training ──
N_CLASSES = 40
SEED = 42
