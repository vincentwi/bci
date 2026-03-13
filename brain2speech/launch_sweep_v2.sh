#!/bin/bash
# Sweep v2: Feature improvements + new architectures, all 8 GPUs
# Uses train_ctc_improved.py for improved features
# Uses train_seq2seq.py for new architectures (vanilla, for comparison)
set -e
source /mnt/home/vincent.wilmet/miniconda/bin/activate py311
cd /mnt/home/vincent.wilmet

LOGDIR="brain2speech/results"

echo "=== Sweep v2: 4 experiments on 8× H100 ==="
echo "  patience=10 for fast iteration"
echo ""

# GPU 0-1: Improved GRU (day_specific + rolling_zscore + specaugment)
# This is the key experiment — matches paper's feature recipe
CUDA_VISIBLE_DEVICES=0,1 python -u brain2speech/train_ctc_improved.py \
  --model GRU --gpus 0 1 --epochs 80 --batch-size 32 --lr 3e-4 \
  --hidden 512 --n-layers 5 --patience 10 --smooth 2 \
  --improvements day_specific,rolling_zscore,specaugment \
  > $LOGDIR/log_GRU_improved.txt 2>&1 &
echo "GPU 0-1: Improved GRU (day+zscore+spec), PID=$!"

# GPU 2-3: Conformer (day_specific + rolling_zscore + specaugment)
# Modern ASR architecture — should beat Transformer where it failed
CUDA_VISIBLE_DEVICES=2,3 python -u brain2speech/train_ctc_improved.py \
  --model Conformer --gpus 2 3 --epochs 80 --batch-size 16 --lr 3e-4 \
  --hidden 256 --n-layers 6 --patience 10 --smooth 2 \
  --improvements day_specific,rolling_zscore,specaugment,conformer \
  > $LOGDIR/log_Conformer.txt 2>&1 &
echo "GPU 2-3: Conformer (day+zscore+spec), PID=$!"

# GPU 4-5: RNN-T (vanilla features, autoregressive)
# Tests if autoregressive decoding alone closes the gap
CUDA_VISIBLE_DEVICES=4,5 python -u brain2speech/train_seq2seq.py \
  --model RNN-T --gpus 4 5 --epochs 80 --batch-size 16 --lr 3e-4 \
  --hidden 512 --n-layers 5 --patience 10 --smooth 2 \
  > $LOGDIR/log_RNNT_v2.txt 2>&1 &
echo "GPU 4-5: RNN-T (autoregressive), PID=$!"

# GPU 6-7: Improved GRU + channel_attention + temporal_deriv
# Tests if delta features and channel attention add value
CUDA_VISIBLE_DEVICES=6,7 python -u brain2speech/train_ctc_improved.py \
  --model GRU --gpus 6 7 --epochs 80 --batch-size 16 --lr 3e-4 \
  --hidden 512 --n-layers 5 --patience 10 --smooth 2 \
  --improvements day_specific,rolling_zscore,specaugment,channel_attention,temporal_deriv \
  > $LOGDIR/log_GRU_all_improvements.txt 2>&1 &
echo "GPU 6-7: GRU (ALL improvements), PID=$!"

echo ""
echo "All 4 launched. Monitor:"
echo "  tail -f brain2speech/results/log_*.txt"
