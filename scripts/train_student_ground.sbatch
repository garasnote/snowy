#!/bin/bash
#SBATCH --job-name=student-ground
#SBATCH --partition=frida
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/student_ground_%j.out

# Train student ground segmentation on ensemble pseudo-labels.
#
# Usage:
#   sbatch scripts/frida/train_student_ground.sbatch
#
#   # Custom ensemble / model:
#   ENSEMBLE=majority_3 MODEL=dinov2_large sbatch scripts/frida/train_student_ground.sbatch
#
#   # RGBD (depth as 4th channel):
#   DEPTH_DIR=/shared/home/juan.osorio/master/preprocessing/segmentation/depth_output/shared_depth \
#     sbatch scripts/frida/train_student_ground.sbatch

MODEL="${MODEL:-mobilenet}"
ENSEMBLE="${ENSEMBLE:-majority_3}"
DATA_DIR="${DATA_DIR:-/shared/home/juan.osorio/master/data}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEPTH_DIR="${DEPTH_DIR:-}"

# If depth is enabled, append _rgbd to model variant
if [ -n "${DEPTH_DIR}" ] && [[ "${MODEL}" != *_rgbd ]]; then
    MODEL="${MODEL}_rgbd"
fi

# HuggingFace token (required for gated models like DINOv3)
HF_TOKEN_FILE="/shared/home/juan.osorio/.hf_token"
if [ -f "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN=$(cat "${HF_TOKEN_FILE}" | tr -d '[:space:]')
    export HF_TOKEN
fi

CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"
if [ -z "${CONTAINER}" ] || [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container image not found at ${CONTAINER}"
    echo "Set SNOW_CONTAINER_IMAGE in your environment."
    exit 1
fi

# Parse ensemble name into threshold + weighted flag
if [[ "${ENSEMBLE}" == weighted_* ]]; then
    THRESHOLD="${ENSEMBLE#weighted_}"
    WEIGHTED_FLAG="--ensemble-weighted"
else
    THRESHOLD="${ENSEMBLE#majority_}"
    WEIGHTED_FLAG=""
fi

echo "============================================================"
echo "TRAIN STUDENT GROUND SEGMENTATION"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         ${SLURM_GPUS_PER_NODE}"
echo "  Model:       ${MODEL}"
echo "  Ensemble:    ${ENSEMBLE}"
echo "  Depth:       ${DEPTH_DIR:-disabled}"
echo "  Epochs:      ${EPOCHS}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

DEPTH_FLAG=""
if [ -n "${DEPTH_DIR}" ]; then
    DEPTH_FLAG="--depth-dir ${DEPTH_DIR}"
fi

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=/shared/home/juan.osorio/master \
  bash -c "
    set -e
    export HF_TOKEN='${HF_TOKEN}'
    echo 'Starting training at \$(date)'
    nvidia-smi
    echo ''

    python models/student_ground/train.py \
        --model-variant ${MODEL} \
        --ensemble-threshold ${THRESHOLD} \
        ${WEIGHTED_FLAG} \
        --data-dir ${DATA_DIR} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --precision 16-mixed \
        --num-workers 8 \
        ${DEPTH_FLAG}

    echo ''
    echo 'Training completed at \$(date)'
  "
