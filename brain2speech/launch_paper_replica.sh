#!/bin/bash
# v5: Push regularization further + combine best features
# G (best) was: bidir, tanh, noise=0.2, drop=0.4, plateau = 39.6% val / 51.9% test
set -e
source /mnt/home/vincent.wilmet/miniconda/bin/activate py311
cd /mnt/home/vincent.wilmet

LOGDIR="brain2speech/results"

echo "=== v5: Push beyond 39.6% val PER ==="
echo ""

# GPU 0-1: I) G + even more reg (noise=0.5, dropout=0.5)
CUDA_VISIBLE_DEVICES=0,1 python -u brain2speech/train_paper_replica.py \
  --gpus 0 1 --batch-size 64 --lr 3e-4 --adam-eps 1e-8 \
  --white-noise 0.5 --offset-noise 0.1 --dropout 0.5 \
  --kernel-size 1 --stride 1 --bidirectional \
  --activation tanh --patience 15 --scheduler plateau \
  > $LOGDIR/log_v5_I_heavy_reg.txt 2>&1 &
echo "GPU 0-1: I) Heavy reg (noise=0.5, drop=0.5), PID=$!"

# GPU 2-3: J) Best combo: softsign + more reg (combine F+G)
CUDA_VISIBLE_DEVICES=2,3 python -u brain2speech/train_paper_replica.py \
  --gpus 0 1 --batch-size 64 --lr 3e-4 --adam-eps 1e-8 \
  --white-noise 0.2 --offset-noise 0.05 --dropout 0.4 \
  --kernel-size 1 --stride 1 --bidirectional \
  --activation softsign --patience 15 --scheduler plateau \
  > $LOGDIR/log_v5_J_softsign_reg.txt 2>&1 &
echo "GPU 2-3: J) Softsign + more reg (combine F+G), PID=$!"

# GPU 4-5: K) Causal smooth + more reg (combine H+G)
CUDA_VISIBLE_DEVICES=4,5 python -u brain2speech/train_paper_replica.py \
  --gpus 0 1 --batch-size 64 --lr 3e-4 --adam-eps 1e-8 \
  --white-noise 0.2 --offset-noise 0.05 --dropout 0.4 \
  --kernel-size 1 --stride 1 --bidirectional \
  --activation tanh --patience 15 --scheduler plateau \
  --causal-smooth \
  > $LOGDIR/log_v5_K_causal_reg.txt 2>&1 &
echo "GPU 4-5: K) Causal smooth + more reg (combine H+G), PID=$!"

# GPU 6-7: L) G but with larger hidden (768)
CUDA_VISIBLE_DEVICES=6,7 python -u brain2speech/train_paper_replica.py \
  --gpus 0 1 --batch-size 64 --lr 3e-4 --adam-eps 1e-8 \
  --white-noise 0.2 --offset-noise 0.05 --dropout 0.4 \
  --hidden 768 \
  --kernel-size 1 --stride 1 --bidirectional \
  --activation tanh --patience 15 --scheduler plateau \
  > $LOGDIR/log_v5_L_large.txt 2>&1 &
echo "GPU 6-7: L) G + hidden=768, PID=$!"

echo ""
echo "Monitor: tail -f brain2speech/results/log_v5_*.txt"
