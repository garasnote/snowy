#!/bin/bash
#SBATCH --job-name=dinov2-pca-ground
#SBATCH --partition=frida
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/dinov2_pca_ground_%j.out

# DINOv2 PCA ground segmentation on FRIDA (zero-shot, no training, no labels).
#
# This is the ground-teacher wrapper the release was missing: the six snow
# teachers ship run_*_snow.sbatch files, the ground teachers shipped none, which
# is how dinov2_pca_output ended up covering CF00001 only.
#
# Note the default output directory: ground_teachers.py writes dinov2_pca into
# WORKSPACE_DIR/segmentation/ground/dinov2_pca, which is exactly where
# fusion_cnn.py looks for that channel - so this job feeds the fusion model and
# the pseudo-label sweep directly, with no symlinking step.
#
# Usage:
#   sbatch scripts/run_dinov2_pca_ground.sbatch
#   CAMERAS="CF00001 CM00009" sbatch scripts/run_dinov2_pca_ground.sbatch
#   MODEL=facebook/dinov2-large SAMPLE=6 DEBUG=1 sbatch scripts/run_dinov2_pca_ground.sbatch

MASTER_DIR="${MASTER_DIR:-/shared/home/juan.osorio/master}"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${MASTER_DIR}/workspace/segmentation/ground/dinov2_pca}"
CAMERAS="${CAMERAS:-}"
MODEL="${MODEL:-facebook/dinov2-base}"
INPUT_SIZE="${INPUT_SIZE:-518}"
SAMPLE="${SAMPLE:-0}"
DEBUG="${DEBUG:-0}"

SCRIPT="${SCRIPT:-${MASTER_DIR}/segmentation/ground_teachers.py}"

HF_TOKEN_FILE="/shared/home/juan.osorio/.hf_token"
HF_TOKEN=""
if [ -f "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN=$(cat "${HF_TOKEN_FILE}" | tr -d '[:space:]')
    echo "  HF token:    loaded (${#HF_TOKEN} chars)"
fi

CONTAINER="${SNOW_CONTAINER_IMAGE:-/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs}"
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container image not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output" "${OUTPUT_DIR}"

echo "============================================================"
echo "DINOV2 PCA - GROUND SEGMENTATION (FRIDA)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  Data dir:    ${DATA_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Cameras:     ${CAMERAS:-all}"
echo "  Model:       ${MODEL}"
echo "  Input size:  ${INPUT_SIZE}"
echo "  Sample:      ${SAMPLE} (0=all)"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

CAMERAS_FLAG=""
[ -n "${CAMERAS}" ] && CAMERAS_FLAG="--cameras ${CAMERAS}"

SAMPLE_FLAG=""
[ "${SAMPLE}" -gt 0 ] 2>/dev/null && SAMPLE_FLAG="--sample ${SAMPLE}"

DEBUG_FLAG=""
[ "${DEBUG}" -eq 1 ] 2>/dev/null && DEBUG_FLAG="--debug"

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

    python ${SCRIPT} dinov2_pca \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --model ${MODEL} \
        --input-size ${INPUT_SIZE} \
        ${CAMERAS_FLAG} \
        ${SAMPLE_FLAG} \
        ${DEBUG_FLAG}

    echo ''
    echo 'Finished at \$(date)'
  "

echo ""
echo "Job finished!"
