#!/bin/bash
#SBATCH --job-name=subspace-snow
#SBATCH --partition=frida
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/subspace_ad_snow_%j.out

# SubspaceAD snow segmentation: DINOv2 PCA subspace anomaly detection.
# Step 1: fit PCA on normal (summer) images
# Step 2: inference on all images
# Step 3: evaluate against GT
#
# Usage:
#   sbatch scripts/frida/run_subspace_ad_snow.sbatch
#   CAMERA=CF00001 sbatch scripts/frida/run_subspace_ad_snow.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
ANN_DIR="${ANN_DIR:-${MASTER_DIR}/data/annotations}"
CAMERA="${CAMERA:-}"
K_SHOT="${K_SHOT:-4}"
N_SEEDS="${N_SEEDS:-5}"
STRATIFY="${STRATIFY:-}"

FIT_SCRIPT="${MASTER_DIR}/models/subspace_ad/fit.py"
INFER_SCRIPT="${MASTER_DIR}/models/subspace_ad/inference.py"
EVAL_SCRIPT="${MASTER_DIR}/models/subspace_ad/eval_iou.py"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

CAMERA_FLAG=""
if [ -n "${CAMERA}" ]; then
    CAMERA_FLAG="--camera ${CAMERA}"
else
    CAMERA_FLAG="--all-cameras"
fi

STRATIFY_FLAG=""
if [ "${STRATIFY}" = "1" ]; then
    STRATIFY_FLAG="--stratify"
fi

echo "============================================================"
echo "SUBSPACE-AD — SNOW SEGMENTATION (FRIDA)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  Camera:      ${CAMERA:-all}"
echo "  K-shot:      ${K_SHOT}"
echo "  Stratify:    ${STRATIFY:-no}"
echo "  N seeds:     ${N_SEEDS}"
echo "  Data dir:    ${DATA_DIR}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    echo 'Starting SubspaceAD at \$(date)'
    nvidia-smi
    echo ''

    echo '=== Step 1: Fit PCA subspace ==='
    python ${FIT_SCRIPT} \
        --img-dir ${DATA_DIR} \
        --k-shot ${K_SHOT} \
        --n-seeds ${N_SEEDS} \
        ${STRATIFY_FLAG} \
        ${CAMERA_FLAG}

    echo ''
    echo '=== Step 2: Inference ==='
    python ${INFER_SCRIPT} \
        --img-dir ${DATA_DIR} \
        --k-shot ${K_SHOT} \
        ${CAMERA_FLAG}

    echo ''
    echo '=== Step 3: Evaluate vs GT ==='
    python ${EVAL_SCRIPT} \
        --ann-dir ${ANN_DIR}

    echo ''
    echo 'SubspaceAD completed at \$(date)'
  "

echo ""
echo "Job finished!"
