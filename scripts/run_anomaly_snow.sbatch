#!/bin/bash
#SBATCH --job-name=anomaly-snow
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=anomaly_snow_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# Anomaly detection for snow segmentation (PatchCore + EfficientAD).
# Step 1: prepare anomalib Folder dataset (normal=summer, anomaly=GT snow)
# Step 2: train + evaluate per camera
#
# Usage:
#   sbatch scripts/run_anomaly_snow.sbatch
#   METHOD=efficientad sbatch scripts/run_anomaly_snow.sbatch
#   CAMERA=CF00001 sbatch scripts/run_anomaly_snow.sbatch

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
ANN_DIR="${ANN_DIR:-${MASTER_DIR}/data/annotations}"
METHOD="${METHOD:-both}"
CAMERA="${CAMERA:-}"
MAX_NORMAL="${MAX_NORMAL:-500}"
IMG_SIZE="${IMG_SIZE:-256}"

PREP_SCRIPT="${MASTER_DIR}/models/anomaly_snow/prepare_dataset.py"
TRAIN_SCRIPT="${MASTER_DIR}/models/anomaly_snow/train_eval.py"
CONTAINER="/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/${CLUSTER_USER}/output"

CAMERA_FLAG=""
if [ -n "${CAMERA}" ]; then
    CAMERA_FLAG="--camera ${CAMERA}"
fi

echo "============================================================"
echo "ANOMALY DETECTION — SNOW SEGMENTATION (cluster)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  Method:      ${METHOD}"
echo "  Camera:      ${CAMERA:-all}"
echo "  Max normal:  ${MAX_NORMAL}"
echo "  Img size:    ${IMG_SIZE}"
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
    pip install --break-system-packages --quiet anomalib 2>/dev/null || true
    echo 'Starting anomaly snow detection at \$(date)'
    nvidia-smi
    echo ''

    echo '=== Step 1: Prepare dataset ==='
    python ${PREP_SCRIPT} \
        --img-dir ${DATA_DIR} \
        --ann-dir ${ANN_DIR} \
        --meteo ${DATA_DIR}/meteo_hourly.csv \
        --max-normal ${MAX_NORMAL} \
        --img-size ${IMG_SIZE} \
        ${CAMERA_FLAG}

    echo ''
    echo '=== Step 2: Train + Evaluate ==='
    python ${TRAIN_SCRIPT} \
        --method ${METHOD} \
        --img-size ${IMG_SIZE} \
        ${CAMERA_FLAG}

    echo ''
    echo 'Anomaly detection completed at \$(date)'
  "

echo ""
echo "Job finished!"
