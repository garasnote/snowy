#!/bin/bash
#SBATCH --job-name=clipseg-snow
#SBATCH --partition=frida
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/clipseg_snow_%j.out

# CLIPSeg text-prompted snow segmentation on FRIDA.
# Lightweight CLIP-based dense prediction — different feature space from SAM/YOLO.
#
# Usage:
#   sbatch scripts/frida/run_clipseg_snow.sbatch
#   CAMERAS="CF00001 CM00009" sbatch scripts/frida/run_clipseg_snow.sbatch
#   THRESHOLD=0.4 SAMPLE=50 DEBUG=1 sbatch scripts/frida/run_clipseg_snow.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${MASTER_DIR}/preprocessing/segmentation/clipseg_snow_output}"
CAMERAS="${CAMERAS:-}"
THRESHOLD="${THRESHOLD:-0.5}"
TEXT_PROMPT="${TEXT_PROMPT:-snow}"
SAMPLE="${SAMPLE:-0}"
DEBUG="${DEBUG:-0}"

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
echo "CLIPSEG — SNOW SEGMENTATION (FRIDA)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         ${SLURM_GPUS_PER_NODE}"
echo "  Data dir:    ${DATA_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Cameras:     ${CAMERAS:-all}"
echo "  Threshold:   ${THRESHOLD}"
echo "  Prompt:      ${TEXT_PROMPT}"
echo "  Sample:      ${SAMPLE} (0=all)"
echo "  Debug:       ${DEBUG}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

# Build optional flags
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

    python ${SCRIPT} clipseg \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --text-prompt '${TEXT_PROMPT}' \
        --threshold ${THRESHOLD} \
        ${CAMERAS_FLAG} \
        ${SAMPLE_FLAG} \
        ${DEBUG_FLAG}

    echo ''
    echo 'Finished at \$(date)'
  "

echo ""
echo "Job finished!"
