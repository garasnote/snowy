#!/bin/bash
#SBATCH --job-name=student-snow
#SBATCH --partition=frida
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/student_snow_%j.out

# Train student snow segmentation on ensemble pseudo-labels (FRIDA).
# Requires teacher outputs + ensemble voting to be done first.
#
# Usage:
#   sbatch scripts/frida/run_student_snow_train.sbatch
#   MODEL=convnext EPOCHS=200 sbatch scripts/frida/run_student_snow_train.sbatch
#   MODEL=mobilenet UNFREEZE=1 sbatch scripts/frida/run_student_snow_train.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
MODEL="${MODEL:-mobilenet}"
THRESHOLD="${THRESHOLD:-3}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
UNFREEZE="${UNFREEZE:-0}"
FOCAL="${FOCAL:-0}"
PRECISION="${PRECISION:-16-mixed}"
CAMERAS="${CAMERAS:-CF00001 CM00009 CM00014 CM00070 CM00075 CM00080 CM00102 CM00108 CM00112}"
MIN_SNOW_PCT="${MIN_SNOW_PCT:-1}"

SCRIPT="${MASTER_DIR}/models/student_snow/train.py"

# HuggingFace token
HF_TOKEN_FILE="/shared/home/juan.osorio/.hf_token"
HF_TOKEN=""
if [ -f "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN=$(cat "${HF_TOKEN_FILE}" | tr -d '[:space:]')
fi

CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"
if [ -z "${CONTAINER}" ] || [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container image not found at ${CONTAINER}"
    echo "Set SNOW_CONTAINER_IMAGE in your environment."
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "STUDENT SNOW SEGMENTATION — TRAINING (FRIDA)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         ${SLURM_GPUS_PER_NODE}"
echo "  Model:       ${MODEL}"
echo "  Threshold:   majority_${THRESHOLD}"
echo "  Epochs:      ${EPOCHS}"
echo "  Batch size:  ${BATCH_SIZE:-default}"
echo "  LR:          ${LR:-default}"
echo "  Unfreeze:    ${UNFREEZE}"
echo "  Focal:       ${FOCAL}"
echo "  Cameras:     ${CAMERAS}"
echo "  Min snow %:  ${MIN_SNOW_PCT}"
echo "  Precision:   ${PRECISION}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

BATCH_FLAG=""
if [ -n "${BATCH_SIZE}" ]; then
    BATCH_FLAG="--batch-size ${BATCH_SIZE}"
fi

LR_FLAG=""
if [ -n "${LR}" ]; then
    LR_FLAG="--lr ${LR}"
fi

UNFREEZE_FLAG=""
if [ "${UNFREEZE}" -eq 1 ] 2>/dev/null; then
    UNFREEZE_FLAG="--unfreeze"
fi

FOCAL_FLAG=""
if [ "${FOCAL}" -eq 1 ] 2>/dev/null; then
    FOCAL_FLAG="--focal"
fi

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    export HF_TOKEN='${HF_TOKEN}'
    pip install --break-system-packages --quiet tensorboard 2>/dev/null || true
    echo 'Starting training at \$(date)'
    nvidia-smi
    echo ''

    for thresh in 2 3 4 5; do
      echo ''
      echo '============================================================'
      echo \"Training majority_\${thresh} at \$(date)\"
      echo '============================================================'
      python ${SCRIPT} \
          --model-variant ${MODEL} \
          --ensemble-threshold \${thresh} \
          --data-dir ${DATA_DIR} \
          --cameras ${CAMERAS} \
          --epochs ${EPOCHS} \
          --precision ${PRECISION} \
          --num-workers 8 \
          ${BATCH_FLAG} \
          ${LR_FLAG} \
          --min-snow-pct ${MIN_SNOW_PCT} \
          ${UNFREEZE_FLAG} \
          ${FOCAL_FLAG}
      echo \"Finished majority_\${thresh} at \$(date)\"
    done

    echo ''
    echo 'All thresholds completed at \$(date)'
  "

echo ""
echo "Job finished!"
