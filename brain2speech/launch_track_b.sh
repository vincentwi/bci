#!/bin/bash
# Track B: Beyond Paper — Launch all experiments
# Researcher B — GPUs 4-7
#
# Prerequisites: Run preprocessing first!
#   python preprocess_from_h5.py
#
# Usage:
#   bash launch_track_b.sh phase1    # B1 + B2 (highest priority)
#   bash launch_track_b.sh phase2    # B3a-B3d (exploratory)
#   bash launch_track_b.sh all       # Everything

set -e
cd "$(dirname "$0")"

PHASE="${1:-phase1}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/track_b_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Track B: Beyond Paper — $PHASE"
echo "Logs: $LOG_DIR/"
echo "============================================"

if [[ "$PHASE" == "phase1" || "$PHASE" == "all" ]]; then
    echo ""
    echo "--- B1: Bidirectional GRU + 256D (GPUs 4,5) ---"
    CUDA_VISIBLE_DEVICES=4,5 python train_beyond_paper.py \
        --experiment B1_bidir_256d \
        --gpus 0 1 \
        --data sentences_paper_256d.h5 \
        --input-dim 256 \
        --bidirectional \
        --hidden 512 --n-layers 5 \
        --lr 0.02 --adam-eps 0.1 \
        --white-noise 1.0 --offset-noise 0.2 \
        --dropout 0.4 --l2-reg 1e-5 \
        --scheduler linear --warmup-steps 500 \
        --batch-size 64 --max-minibatches 10000 \
        --patience 15 \
        2>&1 | tee "$LOG_DIR/B1_bidir_256d.log" &
    B1_PID=$!

    echo "--- B2: 1280D + Correct Arch + Bidir (GPUs 6,7) ---"
    CUDA_VISIBLE_DEVICES=6,7 python train_beyond_paper.py \
        --experiment B2_1280d_bidir \
        --gpus 0 1 \
        --data sentences_paper_1280d.h5 \
        --input-dim 1280 \
        --bidirectional \
        --hidden 512 --n-layers 5 \
        --day-hidden 256 \
        --lr 0.02 --adam-eps 0.1 \
        --white-noise 1.0 --offset-noise 0.2 \
        --dropout 0.4 --l2-reg 1e-5 \
        --scheduler linear --warmup-steps 500 \
        --batch-size 64 --max-minibatches 10000 \
        --patience 15 \
        2>&1 | tee "$LOG_DIR/B2_1280d_bidir.log" &
    B2_PID=$!

    echo "Waiting for B1 (PID=$B1_PID) and B2 (PID=$B2_PID)..."
    wait $B1_PID $B2_PID
    echo "Phase 1 complete!"
fi

if [[ "$PHASE" == "phase2" || "$PHASE" == "all" ]]; then
    echo ""
    echo "--- B3a: Larger GRU (GPU 4) ---"
    CUDA_VISIBLE_DEVICES=4 python train_beyond_paper.py \
        --experiment B3a_large_gru \
        --gpus 0 \
        --data sentences_paper_256d.h5 \
        --input-dim 256 \
        --bidirectional \
        --hidden 768 --n-layers 6 \
        --lr 0.02 --adam-eps 0.1 \
        --white-noise 1.0 --offset-noise 0.2 \
        --dropout 0.4 --l2-reg 1e-5 \
        --scheduler linear --warmup-steps 500 \
        --batch-size 64 --max-minibatches 10000 \
        --patience 15 \
        2>&1 | tee "$LOG_DIR/B3a_large_gru.log" &

    echo "--- B3b: Lower noise + higher dropout (GPU 5) ---"
    CUDA_VISIBLE_DEVICES=5 python train_beyond_paper.py \
        --experiment B3b_low_noise \
        --gpus 0 \
        --data sentences_paper_256d.h5 \
        --input-dim 256 \
        --bidirectional \
        --hidden 512 --n-layers 5 \
        --lr 0.02 --adam-eps 0.1 \
        --white-noise 0.5 --offset-noise 0.1 \
        --dropout 0.5 --l2-reg 1e-5 \
        --scheduler linear --warmup-steps 500 \
        --batch-size 64 --max-minibatches 10000 \
        --patience 15 \
        2>&1 | tee "$LOG_DIR/B3b_low_noise.log" &

    echo "--- B3c: Conformer (GPU 6) ---"
    CUDA_VISIBLE_DEVICES=6 python train_beyond_paper.py \
        --experiment B3c_conformer \
        --gpus 0 \
        --data sentences_paper_256d.h5 \
        --input-dim 256 \
        --model conformer \
        --hidden 256 --n-layers 6 --n-heads 8 \
        --lr 0.001 --adam-eps 1e-8 \
        --white-noise 0.5 --offset-noise 0.2 \
        --dropout 0.1 --l2-reg 1e-5 \
        --scheduler cosine --warmup-steps 1000 \
        --batch-size 64 --max-minibatches 10000 \
        --patience 20 \
        2>&1 | tee "$LOG_DIR/B3c_conformer.log" &

    echo "--- B3d: Plateau scheduler + lower LR (GPU 7) ---"
    CUDA_VISIBLE_DEVICES=7 python train_beyond_paper.py \
        --experiment B3d_plateau_lr \
        --gpus 0 \
        --data sentences_paper_256d.h5 \
        --input-dim 256 \
        --bidirectional \
        --hidden 512 --n-layers 5 \
        --lr 3e-4 --adam-eps 1e-8 \
        --white-noise 0.2 --offset-noise 0.2 \
        --dropout 0.4 --l2-reg 1e-5 \
        --scheduler plateau --warmup-steps 0 \
        --batch-size 64 --max-minibatches 10000 \
        --patience 15 \
        2>&1 | tee "$LOG_DIR/B3d_plateau_lr.log" &

    echo "Waiting for B3a-B3d..."
    wait
    echo "Phase 2 complete!"
fi

echo ""
echo "============================================"
echo "Track B complete! Check results in results/"
echo "============================================"
