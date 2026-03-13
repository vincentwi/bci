#!/bin/bash
# Download BCI speech dataset from OSF (https://osf.io/49rt7/)
set -e

DATA_DIR="/mnt/home/vincent.wilmet/data"
TRAIN_DIR="$DATA_DIR/train/2022_09_22"
SYLL_DIR="$DATA_DIR/syllable/2022_09_22"

mkdir -p "$TRAIN_DIR" "$SYLL_DIR"

echo "=== Downloading KeywordReading/train/2022_09_22 ==="

# R01
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R01.mat" "https://osf.io/download/9gb6x/" &
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R01.wav" "https://osf.io/download/wbcjp/" &
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R01_trials.lab" "https://osf.io/download/zxcna/" &

# R02
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R02.mat" "https://osf.io/download/662959df80d25c1392f919c3/" &
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R02.wav" "https://osf.io/download/662959e1f49cdf17cfd253a9/" &
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R02_trials.lab" "https://osf.io/download/662959e1c5851a1395f67008/" &

# R03
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R03.mat" "https://osf.io/download/662959e5c5851a1395f6700d/" &
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R03.wav" "https://osf.io/download/662959e4f49cdf17cfd253ab/" &
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R03_trials.lab" "https://osf.io/download/662959e43145c6120a6a831c/" &

# R04
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R04.mat" "https://osf.io/download/662959e73145c612066a8299/" &
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R04.wav" "https://osf.io/download/662959e9f49cdf17cfd253b5/" &
wget -c -O "$TRAIN_DIR/KeywordReading_Overt_R04_trials.lab" "https://osf.io/download/662959e7d8960717ab1b1439/" &

wait
echo "=== KeywordReading done ==="

echo "=== Downloading SyllableRepetition/2022_09_22 ==="
wget -c -O "$SYLL_DIR/SyllableRepetition_Overt.mat" "https://osf.io/download/gfhzw/"
echo "=== SyllableRepetition done ==="

echo "=== All downloads complete ==="
ls -lh "$TRAIN_DIR"
ls -lh "$SYLL_DIR"
