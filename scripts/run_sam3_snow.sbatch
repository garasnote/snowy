#!/bin/bash
#SBATCH --job-name=sam3-snow
#SBATCH --partition=frida
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/sam3_snow_%j.out

# SAM3 text-prompted snow segmentation on FRIDA.
#
# Usage:
#   sbatch scripts/frida/run_sam3_snow.sbatch
#   CAMERAS="CF00001 CM00009" sbatch scripts/frida/run_sam3_snow.sbatch
#   SAMPLE=50 DEBUG=1 sbatch scripts/frida/run_sam3_snow.sbatch
#   GROUND_MASK_DIR=/shared/home/juan.osorio/master/preprocessing/segmentation/student_ground_output/mobilenet_ft \
#     sbatch scripts/frida/run_sam3_snow.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${MASTER_DIR}/preprocessing/segmentation/sam3_snow_output}"
CAMERAS="${CAMERAS:-}"
THRESHOLD="${THRESHOLD:-0.4}"
SAMPLE="${SAMPLE:-0}"
DEBUG="${DEBUG:-0}"
GROUND_MASK_DIR="${GROUND_MASK_DIR:-}"
NO_SKY="${NO_SKY:-0}"

SCRIPT="${MASTER_DIR}/segmentation/snow_teachers.py"

# HuggingFace token
HF_TOKEN_FILE="/shared/home/juan.osorio/.hf_token"
HF_TOKEN=""
if [ -f "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN=$(cat "${HF_TOKEN_FILE}" | tr -d '[:space:]')
    echo "  HF token:    loaded (${#HF_TOKEN} chars)"
fi

CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"
if [ -z "${CONTAINER}" ] || [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container image not found at ${CONTAINER}"
    echo "Set SNOW_CONTAINER_IMAGE in your environment."
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "SAM3 — SNOW SEGMENTATION (FRIDA)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         ${SLURM_GPUS_PER_NODE}"
echo "  Data dir:    ${DATA_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Cameras:     ${CAMERAS:-all}"
echo "  Threshold:   ${THRESHOLD}"
echo "  Sample:      ${SAMPLE} (0=all)"
echo "  Debug:       ${DEBUG}"
echo "  Ground mask: ${GROUND_MASK_DIR:-disabled}"
echo "  No sky mask: ${NO_SKY}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

CAMERAS_FLAG=""
if [ -n "${CAMERAS}" ]; then
    CAMERAS_FLAG="--cameras ${CAMERAS}"
fi

SAMPLE_FLAG=""
if [ "${SAMPLE}" -gt 0 ] 2>/dev/null; then
    SAMPLE_FLAG="--sample ${SAMPLE}"
fi

DEBUG_FLAG=""
if [ "${DEBUG}" -eq 1 ] 2>/dev/null; then
    DEBUG_FLAG="--debug"
fi

GROUND_FLAG=""
if [ -n "${GROUND_MASK_DIR}" ]; then
    GROUND_FLAG="--ground-mask-dir ${GROUND_MASK_DIR}"
fi

SKY_FLAG=""
if [ "${NO_SKY}" -eq 1 ] 2>/dev/null; then
    SKY_FLAG="--no-sky-mask"
fi

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    export HF_TOKEN='${HF_TOKEN}'
    echo 'Started at \$(date)'
    nvidia-smi
    echo ''

    python ${SCRIPT} sam3 \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --threshold ${THRESHOLD} \
        ${CAMERAS_FLAG} \
        ${SAMPLE_FLAG} \
        ${DEBUG_FLAG} \
        ${GROUND_FLAG} \
        ${SKY_FLAG}

    echo ''
    echo 'Finished at \$(date)'
  "

echo ""
echo "Job finished!"
