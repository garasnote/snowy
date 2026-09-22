#!/bin/bash
#SBATCH --job-name=seggpt-snow
#SBATCH --partition=frida
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/seggpt_snow_%j.out

# SegGPT few-shot snow segmentation (7th teacher).
# Uses GT annotations as in-context prompts.
#
# Usage:
#   sbatch scripts/frida/run_seggpt_snow.sbatch
#   SAMPLE=100 sbatch scripts/frida/run_seggpt_snow.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
ANN_DIR="${ANN_DIR:-${MASTER_DIR}/data/annotations}"
SAMPLE="${SAMPLE:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
DEBUG="${DEBUG:-0}"

SCRIPT="${MASTER_DIR}/segmentation/snow_teachers.py"
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

DEBUG_FLAG=""
if [ "${DEBUG}" -eq 1 ] 2>/dev/null; then
    DEBUG_FLAG="--debug"
fi

SAMPLE_FLAG=""
if [ "${SAMPLE}" -gt 0 ] 2>/dev/null; then
    SAMPLE_FLAG="--sample ${SAMPLE}"
fi

echo "============================================================"
echo "SEGGPT — FEW-SHOT SNOW SEGMENTATION (7th teacher)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         ${SLURM_GPUS_PER_NODE}"
echo "  Num prompts: ${NUM_PROMPTS}"
echo "  Sample:      ${SAMPLE:-all}"
echo "  Data dir:    ${DATA_DIR}"
echo "  Ann dir:     ${ANN_DIR}"
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
    pip install --break-system-packages --quiet transformers>=4.36.0 2>/dev/null || true
    echo 'Starting SegGPT snow segmentation at \$(date)'
    nvidia-smi
    echo ''

    python ${SCRIPT} seggpt \
        --data-dir ${DATA_DIR} \
        --ann-dir ${ANN_DIR} \
        --fallback-img-dirs ${MASTER_DIR}/night_images \
        --num-prompts ${NUM_PROMPTS} \
        --resume \
        ${SAMPLE_FLAG} \
        ${DEBUG_FLAG}

    echo ''
    echo 'SegGPT completed at \$(date)'
  "

echo ""
echo "Job finished!"
