"""
Shared constants and utilities for the speech-intent decoding pipeline.
Adapted for Willett et al. (Nature 2023) dataset from Dryad/Zenodo.
"""
from pathlib import Path

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DRYAD_DIR = DATA_DIR / "dryad"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "figures_v2"
MODELS_DIR = ROOT / "models"

# ── Recording (Willett dataset) ──
FS = 50                 # Sampling rate Hz (20 ms bins)
N_CHANNELS = 256        # Neural channels (spikePow features)
N_TX_FEATURES = 4       # Threshold crossing feature sets (tx1-tx4)

# ── Task: Diagnostic blocks (word classification) ──
WORD_NAMES = None       # Loaded from cueList in data
N_CLASSES = None        # Set dynamically

# ── Trial segmentation ──
PRE_BINS = 10           # 200 ms pre-onset (10 × 20ms)
POST_BINS = 75          # 1500 ms post-onset (75 × 20ms)
GO_DURATION_BINS = 50   # Expected go-period duration (1000ms)

# ── ML ──
N_PCA = 60
SEED = 42

# ── DL ──
EEGNET_KW = dict(F1=16, D=2, F2=32, kl=32, dr=0.4)
TCN_KW = dict(dr=0.3, hidden=128)
DL_EPOCHS = 150
DL_LR = 3e-4
DL_WD = 1e-2
DL_BS = 32
DL_PATIENCE = 25

# ── Validation ──
VAL_FRACTION = 0.15         # Held-out validation within each CV fold

# ── DataLoader ──
NUM_WORKERS = 6             # DataLoader workers per GPU
PREFETCH_FACTOR = 3         # Batches to prefetch per worker

# ── GPU ──
VISIBLE_GPUS = "4,5,6,7"  # Use last 4 of 8 GPUs
