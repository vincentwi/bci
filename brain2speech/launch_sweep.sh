#!/bin/bash
# Launch 4 model experiments in parallel across 8 GPUs
# Usage: bash launch_sweep.sh

set -e
source /mnt/home/vincent.wilmet/miniconda/bin/activate py311
cd /mnt/home/vincent.wilmet

SCRIPT="brain2speech/train_seq2seq.py"
LOGDIR="brain2speech/results"

echo "=== Launching 4 models on 8× H100 ==="

# GPU 0-1: RNN-T (autoregressive transducer)
CUDA_VISIBLE_DEVICES=0,1 python -u $SCRIPT \
  --model RNN-T --gpus 0 1 --epochs 80 --batch-size 16 --lr 3e-4 \
  --hidden 512 --n-layers 5 --patience 10 --smooth 2 \
  > $LOGDIR/log_RNNT.txt 2>&1 &
echo "RNN-T on GPU 0-1, PID=$!"

# GPU 2-3: Seq2Seq + Attention (encoder-decoder)
CUDA_VISIBLE_DEVICES=2,3 python -u $SCRIPT \
  --model Seq2Seq --gpus 2 3 --epochs 80 --batch-size 16 --lr 3e-4 \
  --hidden 512 --n-layers 5 --patience 10 --smooth 2 \
  > $LOGDIR/log_Seq2Seq.txt 2>&1 &
echo "Seq2Seq on GPU 2-3, PID=$!"

# GPU 4-5: Causal GRU + CTC (unidirectional, streaming)
CUDA_VISIBLE_DEVICES=4,5 python -u $SCRIPT \
  --model CausalGRU-CTC --gpus 4 5 --epochs 80 --batch-size 32 --lr 3e-4 \
  --hidden 512 --n-layers 5 --patience 10 --smooth 2 \
  > $LOGDIR/log_CausalGRU.txt 2>&1 &
echo "CausalGRU on GPU 4-5, PID=$!"

# GPU 6-7: LSTM + CTC (bidirectional, compare to GRU)
CUDA_VISIBLE_DEVICES=6,7 python -u $SCRIPT \
  --model LSTM-CTC --gpus 6 7 --epochs 80 --batch-size 32 --lr 3e-4 \
  --hidden 512 --n-layers 5 --patience 10 --smooth 2 \
  > $LOGDIR/log_LSTM.txt 2>&1 &
echo "LSTM-CTC on GPU 6-7, PID=$!"

echo ""
echo "All 4 launched. Monitor with:"
echo "  tail -f brain2speech/results/log_*.txt"
echo "  watch -n10 'for f in brain2speech/results/log_*.txt; do echo \$(basename \$f); tail -1 \$f; echo; done'"
