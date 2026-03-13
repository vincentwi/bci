#!/bin/bash
# Setup script for GPU VM
# Run this first to install dependencies and verify CUDA
set -e

echo "=== Speech BCI VM Setup ==="

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No CUDA GPU detected. Training will be slow."
fi

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas scikit-learn matplotlib seaborn
pip install librosa h5py tqdm requests openpyxl pydub

# Verify
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo "=== Setup complete ==="
