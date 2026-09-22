#!/bin/bash
#SBATCH --job-name=hsv-snow
#SBATCH --partition=amd
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/shared/home/juan.osorio/output/hsv_snow_%j.out

# HSV white-pixel snow baseline on FRIDA. CPU-only (no GPU needed).
#
# Usage:
#   sbatch scripts/frida/run_hsv_baseline.sbatch
#   CAMERAS="CF00001 CM00009" sbatch scripts/frida/run_hsv_baseline.sbatch
#   GROUND_MASK_DIR=/shared/home/juan.osorio/master/preprocessing/segmentation/student_ground_output/mobilenet_ft \
#     sbatch scripts/frida/run_hsv_baseline.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${MASTER_DIR}/preprocessing/segmentation/hsv_snow_output_mobilenet_ft}"
CAMERAS="${CAMERAS:-}"
SAT_MAX="${SAT_MAX:-30}"
VAL_MIN="${VAL_MIN:-200}"
GROUND_MASK_DIR="${GROUND_MASK_DIR:-${MASTER_DIR}/preprocessing/segmentation/student_ground_output/mobilenet_ft}"

SCRIPT="${MASTER_DIR}/segmentation/hsv_snow_baseline.py"

CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"
if [ -z "${CONTAINER}" ] || [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container image not found at ${CONTAINER}"
    echo "Set SNOW_CONTAINER_IMAGE in your environment."
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "HSV — SNOW BASELINE (FRIDA)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  Data dir:    ${DATA_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Cameras:     ${CAMERAS:-all}"
echo "  Sat max:     ${SAT_MAX}"
echo "  Val min:     ${VAL_MIN}"
echo "  Ground mask: ${GROUND_MASK_DIR}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

CAMERAS_FLAG=""
if [ -n "${CAMERAS}" ]; then
    CAMERAS_FLAG="--cameras ${CAMERAS}"
fi

GROUND_FLAG=""
if [ -n "${GROUND_MASK_DIR}" ]; then
    GROUND_FLAG="--ground-mask-dir ${GROUND_MASK_DIR}"
fi

srun \
  --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    echo 'Started at \$(date)'
    echo ''

    python ${SCRIPT} \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --sat-max ${SAT_MAX} \
        --val-min ${VAL_MIN} \
        ${CAMERAS_FLAG} \
        ${GROUND_FLAG}

    echo ''
    echo 'Finished at \$(date)'
  "

echo ""
echo "Job finished!"
