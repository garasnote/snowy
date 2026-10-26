#!/bin/bash
#SBATCH --job-name=finetune-ground
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=finetune_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# Fine-tune student ground segmentation on real GT annotations.
# Run AFTER train_student_ground.sbatch to get the base checkpoint.
#
# Usage:
#   # Auto-detects the latest best-*.ckpt in checkpoints/:
#   sbatch scripts/finetune_student_ground.sbatch
#
#   # Chain after training (auto-detect picks up the new checkpoint):
#   JOB1=$(ENSEMBLE=majority_3 sbatch --parsable scripts/train_student_ground.sbatch)
#   sbatch --dependency=afterok:${JOB1} scripts/finetune_student_ground.sbatch
#
#   # Or explicit checkpoint:
#   CKPT=/path/to/best-checkpoint.ckpt sbatch scripts/finetune_student_ground.sbatch
#
#   # Camera-specific fine-tuning (only GT from these cameras):
#   CAMERAS="CM00075 CM00108" sbatch scripts/finetune_student_ground.sbatch
#
#   # RGBD fine-tuning:
#   DEPTH_DIR=/shared/home/${CLUSTER_USER}/master/preprocessing/segmentation/depth_output/shared_depth \
#     sbatch scripts/finetune_student_ground.sbatch

MODEL="${MODEL:-mobilenet}"
DEPTH_DIR="${DEPTH_DIR:-}"

# If depth is enabled, append _rgbd to model variant
if [ -n "${DEPTH_DIR}" ] && [[ "${MODEL}" != *_rgbd ]]; then
    MODEL="${MODEL}_rgbd"
fi
CKPT_DIR="/shared/home/${CLUSTER_USER}/master/models/student_ground/checkpoints"
ANNOTATIONS="${ANNOTATIONS:-/shared/home/${CLUSTER_USER}/master/data/annotations/ground_annotations.json}"

# Auto-detect latest best checkpoint if CKPT not explicitly set
if [ -z "${CKPT}" ]; then
    # Use the full MODEL name (including _rgbd) so mobilenet_rgbd matches
    # only mobilenet_rgbd_* dirs, not mobilenet_majority_3_* dirs.
    # For non-RGBD models, exclude _rgbd directories to avoid cross-matches.
    if [[ "${MODEL}" == *_rgbd ]]; then
        CKPT=$(find "${CKPT_DIR}" -path "*/${MODEL}_*" -name "best-*.ckpt" -printf '%T@ %p\n' 2>/dev/null \
               | sort -rn | head -1 | cut -d' ' -f2-)
    else
        CKPT=$(find "${CKPT_DIR}" -path "*/${MODEL}_*" -not -path "*_rgbd_*" -name "best-*.ckpt" -printf '%T@ %p\n' 2>/dev/null \
               | sort -rn | head -1 | cut -d' ' -f2-)
    fi
    if [ -z "${CKPT}" ]; then
        echo "ERROR: No best-*.ckpt found in ${CKPT_DIR} for model '${MODEL}'"
        echo "Run train_student_ground.sbatch first, or set CKPT= explicitly."
        exit 1
    fi
    echo "Auto-detected checkpoint: ${CKPT}"
fi
DATA_DIR="${DATA_DIR:-/shared/home/${CLUSTER_USER}/master/data}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CAMERAS="${CAMERAS:-}"

# HuggingFace token (required for gated models like DINOv3)
HF_TOKEN_FILE="/shared/home/${CLUSTER_USER}/.hf_token"
if [ -f "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN=$(cat "${HF_TOKEN_FILE}" | tr -d '[:space:]')
    export HF_TOKEN
fi

CONTAINER="/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs"
if [ -z "${CONTAINER}" ] || [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container image not found at ${CONTAINER}"
    echo "Set SNOW_CONTAINER_IMAGE in your environment."
    exit 1
fi

echo "============================================================"
echo "FINE-TUNE STUDENT GROUND SEGMENTATION"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  GPU:         ${SLURM_GPUS_PER_NODE}"
echo "  Model:       ${MODEL}"
echo "  Checkpoint:  ${CKPT}"
echo "  Annotations: ${ANNOTATIONS}"
echo "  Depth:       ${DEPTH_DIR:-disabled}"
echo "  Cameras:     ${CAMERAS:-all}"
echo "  Epochs:      ${EPOCHS}"
echo "  LR:          ${LR}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

DEPTH_FLAG=""
if [ -n "${DEPTH_DIR}" ]; then
    DEPTH_FLAG="--depth-dir ${DEPTH_DIR}"
fi

CAMERAS_FLAG=""
if [ -n "${CAMERAS}" ]; then
    CAMERAS_FLAG="--cameras ${CAMERAS}"
fi

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=/shared/home/${CLUSTER_USER}/master \
  bash -c "
    set -e
    export HF_TOKEN='${HF_TOKEN}'
    echo 'Starting fine-tuning at \$(date)'
    nvidia-smi
    echo ''

    python models/student_ground/train.py \
        --finetune \
        --checkpoint ${CKPT} \
        --annotations ${ANNOTATIONS} \
        --data-dir ${DATA_DIR} \
        --model-variant ${MODEL} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --batch-size ${BATCH_SIZE} \
        --precision 16-mixed \
        --num-workers 8 \
        ${DEPTH_FLAG} \
        ${CAMERAS_FLAG}

    echo ''
    echo 'Fine-tuning completed at \$(date)'
  "
