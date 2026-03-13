from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
ONLINE_DIR = DATA_DIR / "online_sessions"
SYLLABLE_DIR = DATA_DIR / "syllable"
CACHE_DIR = DATA_DIR / "cache"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"

# ── ECoG constants ─────────────────────────────────────────────────────
FS = 1000               # ECoG sample rate (Hz)
N_ECOG = 128            # Total ECoG channels (2 x 8x8 grids)
N_GRIDS = 2
GRID_SIZE = 64           # Channels per grid
GAIN = 0.25              # µV per int16 unit
ANALOG_CHANNELS = [128, 129, 130]  # ainp1-3 (not used)

# ── High-gamma extraction ──────────────────────────────────────────────
HG_WIN_MS = 50           # Window size for log-power
HG_HOP_MS = 10           # Hop size → 100 Hz output frame rate
HG_BANDS = [(70, 117), (123, 170)]  # Split around 120 Hz (2nd harmonic of 60 Hz)
BUTTER_ORDER = 8

# ── LPC features ──────────────────────────────────────────────────────
N_LPC = 20               # 18 Bark cepstrals + 2 pitch params
AUDIO_FS = 16000         # Expected audio sample rate

# ── Dataset splits ─────────────────────────────────────────────────────
TRAIN_DAYS = [
    "2022_09_22", "2022_09_23", "2022_09_28", "2022_09_30",
    "2022_10_05", "2022_10_06", "2022_10_10", "2022_10_27",
]
VAL_DAY = "2022_11_04"
TEST_DAY = "2022_11_03"
ONLINE_DAYS = ["2023_04_14", "2023_04_18", "2023_04_21"]
ALL_DAYS = TRAIN_DAYS + [VAL_DAY, TEST_DAY] + ONLINE_DAYS

RUNS = ["R01", "R02", "R03", "R04"]  # default; some days have more or fewer

# Online sessions use a different file prefix
ONLINE_PREFIX = "KeywordSynthesis_Overt"
TRAIN_PREFIX = "KeywordReading_Overt"
WORDS = ["Back", "Down", "Enter", "Left", "Right", "Up"]
N_CLASSES = len(WORDS)

# ── Model hyperparameters (from paper's GitHub, corrected) ─────────────
# nVAD
NVAD_HIDDEN = 150
NVAD_LAYERS = 2
NVAD_DROPOUT = 0.5
NVAD_LR = 1e-4
NVAD_EPOCHS = 8

# Acoustic decoder
DECODER_HIDDEN = 100     # Per direction (NOT 150 as paper text states)
DECODER_LAYERS = 2
DECODER_DROPOUT = 0.5
DECODER_LR = 1e-4
DECODER_EPOCHS = 20

# Truncated BPTT
TBPTT_K1 = 50            # Update every k1 frames
TBPTT_K2 = 100           # Gradient flows through k2 frames

# ── OSF dataset ────────────────────────────────────────────────────────
OSF_NODE_ID = "49rt7"
OSF_API_BASE = "https://api.osf.io/v2"
