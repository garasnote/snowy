#!/bin/bash
#SBATCH --job-name=gsam2-snow
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=groundedsam2_snow_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# Grounded SAM2 snow segmentation on the cluster.
# GroundingDINO detects "snow" boxes → SAM2 refines to pixel masks.
#
# Usage:
#   sbatch scripts/run_groundedsam2_snow.sbatch
#   CAMERAS="CF00001 CM00009" sbatch scripts/run_groundedsam2_snow.sbatch
#   SAMPLE=50 DEBUG=1 sbatch scripts/run_groundedsam2_snow.sbatch

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${MASTER_DIR}/preprocessing/segmentation/groundedsam2_snow_output}"
CAMERAS="${CAMERAS:-}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
SAMPLE="${SAMPLE:-0}"
DEBUG="${DEBUG:-0}"

SCRIPT="${MASTER_DIR}/segmentation/snow_teachers.py"

# HuggingFace token
HF_TOKEN_FILE="/shared/home/${CLUSTER_USER}/.hf_token"
HF_TOKEN=""
if [ -f "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN=$(cat "${HF_TOKEN_FILE}" | tr -d '[:space:]')
    echo "  HF token:    loaded (${#HF_TOKEN} chars)"
fi

CONTAINER="/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs"
if [ -z "${CONTAINER}" ] || [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container image not found at ${CONTAINER}"
    echo "Set SNOW_CONTAINER_IMAGE in your environment."
    exit 1
fi

mkdir -p "$(dirname ${SLURM_OUTPUT:-/shared/home/${CLUSTER_USER}/output/dummy})" 2>/dev/null
mkdir -p "/shared/home/${CLUSTER_USER}/output"

echo "============================================================"
echo "GROUNDED SAM2 — SNOW SEGMENTATION (cluster)"
echo "============================================================"
echo "  Job ID:         ${SLURM_JOB_ID}"
echo "  Node:           $(hostname)"
echo "  GPU:            ${SLURM_GPUS_PER_NODE}"
echo "  Data dir:       ${DATA_DIR}"
echo "  Output dir:     ${OUTPUT_DIR}"
echo "  Cameras:        ${CAMERAS:-all}"
echo "  Box threshold:  ${BOX_THRESHOLD}"
echo "  Text threshold: ${TEXT_THRESHOLD}"
echo "  Sample:         ${SAMPLE} (0=all)"
echo "  Debug:          ${DEBUG}"
echo "  Container:      ${CONTAINER}"
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
    pip install --break-system-packages --quiet sam-2 2>/dev/null || \
      pip install --break-system-packages --quiet git+https://github.com/facebookresearch/sam2.git 2>/dev/null || true
    echo 'Started at \$(date)'
    nvidia-smi
    echo ''

    python ${SCRIPT} groundedsam2 \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --box-threshold ${BOX_THRESHOLD} \
        --text-threshold ${TEXT_THRESHOLD} \
        ${CAMERAS_FLAG} \
        ${SAMPLE_FLAG} \
        ${DEBUG_FLAG}

    echo ''
    echo 'Finished at \$(date)'
  "

echo ""
echo "Job finished!"
