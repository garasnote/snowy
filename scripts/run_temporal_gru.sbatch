#!/bin/bash
#SBATCH --job-name=temporal-gru
#SBATCH --partition=frida
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/temporal_gru_%j.out

# Temporal GRU snow refinement pipeline:
#   Step 1: Extract image features (CPU, ~10ms/image)
#   Step 2: Rebuild snow timeseries CSV (merge features + meteo)
#   Step 3: Train GRU temporal model
#
# Usage:
#   sbatch scripts/frida/run_temporal_gru.sbatch
#   ARCH=lstm BIDIRECTIONAL=1 sbatch scripts/frida/run_temporal_gru.sbatch
#   SEQ_LEN=48 sbatch scripts/frida/run_temporal_gru.sbatch
#   SKIP_FEATURES=1 sbatch scripts/frida/run_temporal_gru.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
SNOW_CKPT="${SNOW_CKPT:-${MASTER_DIR}/models/student_snow/checkpoints/snow_gt_mobilenet_20260509_134400/best-epoch=78-iouval_iou=0.837.ckpt}"
SNOW_VARIANT="${SNOW_VARIANT:-mobilenet}"
SKIP_INFERENCE="${SKIP_INFERENCE:-1}"
ARCH="${ARCH:-gru}"
HIDDEN="${HIDDEN:-128}"
LAYERS="${LAYERS:-2}"
SEQ_LEN="${SEQ_LEN:-72}"
STRIDE="${STRIDE:-12}"
KERNEL="${KERNEL:-7}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
DROPOUT="${DROPOUT:-0.3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SPLIT="${SPLIT:-temporal}"
SKIP_FEATURES="${SKIP_FEATURES:-0}"
NO_IMAGE_FEATURES="${NO_IMAGE_FEATURES:-0}"
WORKERS="${WORKERS:-8}"

CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

NO_IMG_FLAG=""
if [ "${NO_IMAGE_FEATURES}" -eq 1 ] 2>/dev/null; then
    NO_IMG_FLAG="--no-image-features"
fi

echo "============================================================"
echo "TEMPORAL GRU — SNOW REFINEMENT (FRIDA)"
echo "============================================================"
echo "  Job ID:         ${SLURM_JOB_ID}"
echo "  Node:           $(hostname)"
echo "  Snow ckpt:      ${SNOW_CKPT}"
echo "  Snow variant:   ${SNOW_VARIANT}"
echo "  Skip inference: ${SKIP_INFERENCE}"
echo "  Arch:           ${ARCH}"
echo "  Hidden:         ${HIDDEN}"
echo "  Layers:         ${LAYERS}"
echo "  Seq len:        ${SEQ_LEN}"
echo "  Stride:         ${STRIDE}"
echo "  Epochs:         ${EPOCHS}"
echo "  Kernel:         ${KERNEL}"
echo "  Bidirectional:  both (forward + bidir)"
echo "  Image features: $([ ${NO_IMAGE_FEATURES} -eq 1 ] && echo 'disabled' || echo 'enabled')"
echo "  Skip features:  ${SKIP_FEATURES}"
echo "  Container:      ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    pip install --break-system-packages --quiet tensorboard 2>/dev/null || true
    echo 'Starting temporal GRU pipeline at \$(date)'
    nvidia-smi
    echo ''

    # Step 0: Run student snow inference on ALL images (GPU)
    if [ ${SKIP_INFERENCE} -eq 0 ]; then
        echo '=== Step 0: Student snow inference (all images) ==='
        python ${MASTER_DIR}/models/student_snow/inference.py \
            --checkpoint ${SNOW_CKPT} \
            --model-variant ${SNOW_VARIANT} \
            --data-dir ${DATA_DIR} \
            --output-dir ${MASTER_DIR}/preprocessing/segmentation/student_snow_output/mobilenet_gt \
            --batch-size 16
        echo ''
    else
        echo '=== Step 0: SKIPPED (SKIP_INFERENCE=1) ==='
        echo ''
    fi

    # Step 1: Extract image features (CPU, ~10ms/image)
    if [ ${SKIP_FEATURES} -eq 0 ]; then
        echo '=== Step 1: Extract image features ==='
        python ${MASTER_DIR}/analysis/extract_image_features.py \
            --data-dir ${DATA_DIR} \
            --workers ${WORKERS}
        echo ''

        echo '=== Step 2: Rebuild timeseries CSV ==='
        python ${MASTER_DIR}/analysis/build_snow_timeseries.py \
            --data-dir ${DATA_DIR} \
            --workers ${WORKERS} \
            --skip-qa
        echo ''
    else
        echo '=== Steps 1-2: SKIPPED (SKIP_FEATURES=1) ==='
        echo ''
    fi

    echo '=== Step 3: Train smoother (forward-only, deployment-realistic) ==='
    python ${MASTER_DIR}/models/temporal_snow/train_smoother.py \
        --csv ${MASTER_DIR}/analysis/snow_timeseries.csv \
        --arch ${ARCH} \
        --hidden-size ${HIDDEN} \
        --num-layers ${LAYERS} \
        --seq-len ${SEQ_LEN} \
        --stride ${STRIDE} \
        --kernel-size ${KERNEL} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --dropout ${DROPOUT} \
        --weight-decay ${WEIGHT_DECAY} \
        --split ${SPLIT} \
        --num-workers ${WORKERS} \
        ${NO_IMG_FLAG}

    echo ''
    echo '=== Step 4: Train smoother (bidirectional, upper bound) ==='
    python ${MASTER_DIR}/models/temporal_snow/train_smoother.py \
        --csv ${MASTER_DIR}/analysis/snow_timeseries.csv \
        --arch ${ARCH} \
        --hidden-size ${HIDDEN} \
        --num-layers ${LAYERS} \
        --seq-len ${SEQ_LEN} \
        --stride ${STRIDE} \
        --kernel-size ${KERNEL} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --dropout ${DROPOUT} \
        --weight-decay ${WEIGHT_DECAY} \
        --split ${SPLIT} \
        --num-workers ${WORKERS} \
        --bidirectional \
        ${NO_IMG_FLAG}

    echo ''
    echo 'Temporal GRU pipeline completed at \$(date)'
  "

echo ""
echo "Job finished!"
