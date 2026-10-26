#!/bin/bash
#SBATCH --job-name=qa-cnn-3class
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=qa_cnn_3class_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# Train 3-class QA CNN: clean_no_snow(0) / snow(1) / non_usable(2).
#
# Step 1: regenerate qa_features.csv with label3 column (train_qa.py)
# Step 2: train TinyQACNN on label3 (train_qa_cnn.py)
#
# Usage:
#   sbatch run_train_qa_cnn.sbatch
#   EPOCHS=100 sbatch run_train_qa_cnn.sbatch

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
CONTAINER="${SNOW_CONTAINER_IMAGE:-/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"

mkdir -p /shared/home/${CLUSTER_USER}/output

echo "============================================================"
echo "QA CNN — 3-CLASS TRAINING (clean_no_snow / snow / non_usable)"
echo "============================================================"
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Node:    $(hostname)"
echo "  Epochs:  ${EPOCHS}  BS=${BATCH_SIZE}  LR=${LR}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    echo 'Started at \$(date)'
    nvidia-smi
    echo ''

    echo '=== Step 1: regenerate qa_features.csv with label3 ==='
    python ${MASTER_DIR}/qa/train_qa.py \
        --ann-dir   ${MASTER_DIR}/data/annotations \
        --img-dir   ${MASTER_DIR}/data/balanced_snow \
        --meteo     ${MASTER_DIR}/data/meteo_hourly.csv \
        --output-dir ${MASTER_DIR}/qa/checkpoints

    echo ''
    echo '=== Step 2: train 3-class CNN ==='
    python ${MASTER_DIR}/qa/train_qa_cnn.py \
        --features-csv ${MASTER_DIR}/qa/checkpoints/qa_features.csv \
        --img-dir      ${MASTER_DIR}/data/balanced_snow \
        --output-dir   ${MASTER_DIR}/qa/checkpoints \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR}

    echo ''
    ls -lh ${MASTER_DIR}/qa/checkpoints/qa_cnn.pt \
            ${MASTER_DIR}/qa/checkpoints/qa_cnn.ckpt

    echo ''
    echo 'Finished at \$(date)'
  "

echo "Job finished!"
