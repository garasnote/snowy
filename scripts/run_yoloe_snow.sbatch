#!/bin/bash
#SBATCH --job-name=yoloe-snow
#SBATCH --partition=frida
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/yoloe_snow_%j.out

# YOLOe-26 open-vocabulary snow segmentation on FRIDA.
#
# Usage:
#   sbatch scripts/frida/run_yoloe_snow.sbatch
#   CAMERAS="CF00001 CM00009" sbatch scripts/frida/run_yoloe_snow.sbatch
#   MODEL_SIZE=m SAMPLE=50 DEBUG=1 sbatch scripts/frida/run_yoloe_snow.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${MASTER_DIR}/preprocessing/segmentation/yoloe_snow_output}"
CAMERAS="${CAMERAS:-}"
MODEL_SIZE="${MODEL_SIZE:-l}"
CONF="${CONF:-0.25}"
SAMPLE="${SAMPLE:-0}"
DEBUG="${DEBUG:-0}"

SCRIPT="${MASTER_DIR}/segmentation/snow_teachers.py"

CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"
if [ -z "${CONTAINER}" ] || [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container image not found at ${CONTAINER}"
    echo "Set SNOW_CONTAINER_IMAGE in your environment."
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "YOLOe-26 — SNOW SEGMENTATION (FRIDA)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         ${SLURM_GPUS_PER_NODE}"
echo "  Model:       yoloe-26${MODEL_SIZE}-seg"
echo "  Confidence:  ${CONF}"
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
    pip install --break-system-packages --quiet ultralytics 2>/dev/null || true
    pip install --break-system-packages --quiet openai-clip 2>/dev/null || \
      pip install --break-system-packages --quiet git+https://github.com/ultralytics/CLIP.git 2>/dev/null || true
    echo 'Started at \$(date)'
    nvidia-smi
    echo ''

    python ${SCRIPT} yoloe \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --model-size ${MODEL_SIZE} \
        --conf ${CONF} \
        ${CAMERAS_FLAG} \
        ${SAMPLE_FLAG} \
        ${DEBUG_FLAG}

    echo ''
    echo 'Finished at \$(date)'
  "

echo ""
echo "Job finished!"
