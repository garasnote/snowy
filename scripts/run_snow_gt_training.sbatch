#!/bin/bash
#SBATCH --job-name=snow-gt
#SBATCH --partition=frida
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/snow_gt_%j.out

# Train student snow segmentation on GT annotations (FRIDA).
# All 3 bugs fixed: freeze_encoder, negative samples, loss target.
#
# Usage:
#   sbatch scripts/frida/run_snow_gt_training.sbatch
#   MODEL=convnext sbatch scripts/frida/run_snow_gt_training.sbatch
#   MODEL=vit_lora UNFREEZE=1 sbatch scripts/frida/run_snow_gt_training.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
ANN_DIR="${ANN_DIR:-${MASTER_DIR}/data/annotations}"
MODEL="${MODEL:-mobilenet}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
UNFREEZE="${UNFREEZE:-1}"
FOCAL="${FOCAL:-0}"
PRECISION="${PRECISION:-16-mixed}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"

SCRIPT="${MASTER_DIR}/models/student_snow/train.py"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

# HuggingFace token
HF_TOKEN_FILE="/shared/home/juan.osorio/.hf_token"
HF_TOKEN=""
if [ -f "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN=$(cat "${HF_TOKEN_FILE}" | tr -d '[:space:]')
fi

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

UNFREEZE_FLAG=""
if [ "${UNFREEZE}" -eq 1 ] 2>/dev/null; then
    UNFREEZE_FLAG="--unfreeze"
fi

FOCAL_FLAG=""
if [ "${FOCAL}" -eq 1 ] 2>/dev/null; then
    FOCAL_FLAG="--focal"
fi

PRETRAIN_FLAG=""
if [ -n "${PRETRAIN_CKPT}" ]; then
    PRETRAIN_FLAG="--pretrain-ckpt ${PRETRAIN_CKPT}"
fi

echo "============================================================"
echo "STUDENT SNOW — GT TRAINING (FRIDA)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         ${SLURM_GPUS_PER_NODE}"
echo "  Model:       ${MODEL}"
echo "  Unfreeze:    ${UNFREEZE}"
echo "  Epochs:      ${EPOCHS}"
echo "  LR:          ${LR}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  Data dir:    ${DATA_DIR}"
echo "  Ann dir:     ${ANN_DIR}"
echo "  Precision:   ${PRECISION}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    export HF_TOKEN='${HF_TOKEN}'
    pip install --break-system-packages --quiet tensorboard 2>/dev/null || true
    echo 'Starting GT training at \$(date)'
    nvidia-smi
    echo ''

    python ${SCRIPT} \
        --gt \
        --model-variant ${MODEL} \
        --data-dir ${DATA_DIR} \
        --ann-dir ${ANN_DIR} \
        --fallback-img-dirs ${MASTER_DIR}/night_images \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --batch-size ${BATCH_SIZE} \
        --precision ${PRECISION} \
        --num-workers 8 \
        ${UNFREEZE_FLAG} \
        ${FOCAL_FLAG} \
        ${PRETRAIN_FLAG}

    echo ''
    echo 'GT training completed at \$(date)'
  "

echo ""
echo "Job finished!"
