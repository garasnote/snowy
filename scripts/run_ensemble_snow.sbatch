#!/bin/bash
#SBATCH --job-name=ensemble-snow
#SBATCH --partition=amd
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/ensemble_snow_%j.out

# Ensemble snow voting across all teacher methods on FRIDA. CPU-only.
#
# Usage:
#   sbatch scripts/frida/run_ensemble_snow.sbatch
#   MIN_VOTES=3 sbatch scripts/frida/run_ensemble_snow.sbatch
#   WEIGHTED=1 MIN_VOTES=4 WEIGHTS="hsv=1 sam3=2 yoloe=2 florence2=2 groundedsam2=2 clipseg=1" \
#     sbatch scripts/frida/run_ensemble_snow.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
MIN_VOTES="${MIN_VOTES:-3}"
MIN_METHODS="${MIN_METHODS:-3}"
WEIGHTED="${WEIGHTED:-0}"
WEIGHTS="${WEIGHTS:-}"
CAMERAS="${CAMERAS:-}"
WORKERS="${WORKERS:-16}"
DEBUG="${DEBUG:-0}"

SCRIPT="${MASTER_DIR}/segmentation/snow_utils.py"

CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"
if [ -z "${CONTAINER}" ] || [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container image not found at ${CONTAINER}"
    echo "Set SNOW_CONTAINER_IMAGE in your environment."
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "ENSEMBLE SNOW VOTING (FRIDA)"
echo "============================================================"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Node:         $(hostname)"
echo "  Min votes:    ${MIN_VOTES}"
echo "  Min methods:  ${MIN_METHODS}"
echo "  Weighted:     ${WEIGHTED}"
echo "  Weights:      ${WEIGHTS:-default}"
echo "  Cameras:      ${CAMERAS:-all}"
echo "  Workers:      ${WORKERS}"
echo "  Debug:        ${DEBUG}"
echo "  Container:    ${CONTAINER}"
echo "============================================================"
echo ""

CAMERAS_FLAG=""
if [ -n "${CAMERAS}" ]; then
    CAMERAS_FLAG="--cameras ${CAMERAS}"
fi

WEIGHTED_FLAG=""
WEIGHTS_FLAG=""
if [ "${WEIGHTED}" -eq 1 ] 2>/dev/null; then
    WEIGHTED_FLAG="--weighted"
    if [ -n "${WEIGHTS}" ]; then
        WEIGHTS_FLAG="--weights ${WEIGHTS}"
    fi
fi

DEBUG_FLAG=""
if [ "${DEBUG}" -eq 1 ] 2>/dev/null; then
    DEBUG_FLAG="--debug"
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

    python ${SCRIPT} ensemble \
        --min-votes ${MIN_VOTES} \
        --min-methods ${MIN_METHODS} \
        --workers ${WORKERS} \
        ${WEIGHTED_FLAG} \
        ${WEIGHTS_FLAG} \
        ${CAMERAS_FLAG} \
        ${DEBUG_FLAG}

    echo ''
    echo 'Finished at \$(date)'
  "

echo ""
echo "Job finished!"
