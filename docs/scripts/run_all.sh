#!/bin/bash
# Master script: run the full Speech BCI pipeline on a GPU VM.
#
# Usage:
#   scp -r neuralink/ vm:~/speech_bci/
#   ssh vm 'cd ~/speech_bci && bash scripts/run_all.sh'
#
# Or run individual steps:
#   bash scripts/00_setup.sh
#   python scripts/01_download.py
#   python scripts/02_preprocess.py
#   python scripts/03_train.py --model both
#   python scripts/04_synthesize.py
set -e

echo "============================================"
echo "  Speech BCI — Full Pipeline"
echo "============================================"

# Step 0: Setup (if not already done)
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo ">>> Step 0: Setting up environment..."
    bash scripts/00_setup.sh
fi

# Step 1: Download data (~4 GB)
echo ""
echo ">>> Step 1: Downloading data from OSF..."
python scripts/01_download.py

# Step 2: Preprocess (CPU — ~15 min)
echo ""
echo ">>> Step 2: Preprocessing (HG + normalization + LPC + VAD → HDF5)..."
python scripts/02_preprocess.py

# Step 3: Train models (GPU)
echo ""
echo ">>> Step 3: Training models..."
python scripts/03_train.py --model both

# Step 4: Synthesize test day
echo ""
echo ">>> Step 4: Synthesizing test day..."
python scripts/04_synthesize.py

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Checkpoints: checkpoints/"
echo "  Plots:       plots/"
echo "  Synthesis:   synthesized/"
echo "============================================"
