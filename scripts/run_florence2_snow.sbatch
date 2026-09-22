#!/bin/bash
#SBATCH --job-name=flor2-snow
#SBATCH --partition=frida
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/florence2_snow_%j.out

# Florence-2 referring expression snow segmentation on FRIDA.
#
# Usage:
#   sbatch scripts/frida/run_florence2_snow.sbatch
#   CAMERAS="CF00001 CM00009" sbatch scripts/frida/run_florence2_snow.sbatch
#   MODEL_SIZE=base SAMPLE=50 DEBUG=1 sbatch scripts/frida/run_florence2_snow.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${MASTER_DIR}/preprocessing/segmentation/florence2_snow_output}"
CAMERAS="${CAMERAS:-}"
MODEL_SIZE="${MODEL_SIZE:-large}"
TEXT_PROMPT="${TEXT_PROMPT:-snow}"
SAMPLE="${SAMPLE:-0}"
DEBUG="${DEBUG:-0}"

SCRIPT="${MASTER_DIR}/segmentation/florence2_snow_segmentation.py"

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
echo "FLORENCE-2 — SNOW SEGMENTATION (FRIDA)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         ${SLURM_GPUS_PER_NODE}"
echo "  Model:       Florence-2-${MODEL_SIZE}"
echo "  Prompt:      ${TEXT_PROMPT}"
echo "  Data dir:    ${DATA_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Cameras:     ${CAMERAS:-all}"
echo "  Sample:      ${SAMPLE} (0=all)"
echo "  Debug:       ${DEBUG}"
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

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    export HF_TOKEN='${HF_TOKEN}'
    pip install --break-system-packages --quiet 'transformers==4.51.3' 2>/dev/null || true
    echo 'Started at \$(date)'
    nvidia-smi
    echo ''

    python ${SCRIPT} \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --model-size ${MODEL_SIZE} \
        --text-prompt '${TEXT_PROMPT}' \
        ${CAMERAS_FLAG} \
        ${SAMPLE_FLAG} \
        ${DEBUG_FLAG}

    echo ''
    echo 'Finished at \$(date)'
  "

echo ""
echo "Job finished!"
