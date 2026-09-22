#!/bin/bash
#SBATCH --job-name=multihead-dinov3
#SBATCH --partition=frida
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=output/multihead/train_%j.out

# DINOv3 multi-head training: QA + ground seg + snow seg — one backbone.
#
# Usage:
#   sbatch run_multihead_train.sbatch                         # dinov3_base, defaults
#   VARIANT=dinov3_large BATCH_SIZE=8 sbatch run_multihead_train.sbatch
#   PATIENCE=15 MAX_EPOCHS=150 sbatch run_multihead_train.sbatch
#   SMOKE=1 sbatch run_multihead_train.sbatch                 # 2-epoch sanity check
#   SKIP_PSEUDO_CKPT=/shared/.../checkpoints/dinov3_base/pseudo_phase1.pt sbatch run_multihead_train.sbatch
#                                                               # skip Phase 1, load saved decoder weights

VARIANT="${VARIANT:-dinov3_base}"
RUN_TAG="${RUN_TAG:-}"          # suffix for checkpoints/{variant}_{tag}/ — avoids clobbering a run already in progress
MAX_EPOCHS="${MAX_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-0}"   # 0 = use model default
LR="${LR:-0}"                   # 0 = use model default
NUM_WORKERS="${NUM_WORKERS:-12}"
PRECISION="${PRECISION:-16-mixed}"
SMOKE="${SMOKE:-0}"
PSEUDO_GROUND="${PSEUDO_GROUND:-/shared/home/juan.osorio/master/preprocessing/segmentation/sam3_dino_output/text}"
PSEUDO_SNOW="${PSEUDO_SNOW:-/shared/home/juan.osorio/master/preprocessing/segmentation/fusion_snow_output}"
PSEUDO_EPOCHS="${PSEUDO_EPOCHS:-30}"
PSEUDO_GROUND_CACHE="${PSEUDO_GROUND_CACHE:-/shared/home/juan.osorio/master/models/multi_head/cache/pseudo_ground_valid.tsv}"
PSEUDO_SNOW_CACHE="${PSEUDO_SNOW_CACHE:-/shared/home/juan.osorio/master/models/multi_head/cache/pseudo_snow_valid.tsv}"
SKIP_PSEUDO_CKPT="${SKIP_PSEUDO_CKPT:-}"   # set to a pseudo_phase1.pt path to skip Phase 1 entirely

MASTER_DIR="/shared/home/juan.osorio/master"
SCRIPT_PATH="${MASTER_DIR}/models/multi_head/train_multihead.py"
MLFLOW_DIR="${HOME}/mlruns"

mkdir -p output/multihead

echo "============================================================"
echo "MULTIHEAD DINOV3 — TRAINING"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo ""
echo "Variant:     ${VARIANT}"
echo "Max epochs:  ${MAX_EPOCHS}  Patience: ${PATIENCE}"
echo "Batch size:  ${BATCH_SIZE:-model default}"
echo "LR:          ${LR:-model default}"
echo "MLflow dir:  ${MLFLOW_DIR}"
echo "============================================================"
echo ""

# Smoke test: 2 epochs, small patience
if [ "${SMOKE}" -eq 1 ] 2>/dev/null; then
    MAX_EPOCHS=2
    PATIENCE=2
    echo "SMOKE TEST MODE"
fi

BATCH_FLAG=""
if [ "${BATCH_SIZE}" -gt 0 ] 2>/dev/null; then
    BATCH_FLAG="--batch-size ${BATCH_SIZE}"
fi

LR_FLAG=""
if [ "${LR}" != "0" ] && [ -n "${LR}" ]; then
    LR_FLAG="--lr ${LR}"
fi

SKIP_PSEUDO_FLAG=""
if [ -n "${SKIP_PSEUDO_CKPT}" ]; then
    SKIP_PSEUDO_FLAG="--skip-pseudo-ckpt ${SKIP_PSEUDO_CKPT}"
    echo "Skip Phase 1: loading ${SKIP_PSEUDO_CKPT}"
fi

RUN_TAG_FLAG=""
if [ -n "${RUN_TAG}" ]; then
    RUN_TAG_FLAG="--run-tag ${RUN_TAG}"
    echo "Run tag:     ${RUN_TAG}  (checkpoints -> checkpoints/${VARIANT}_${RUN_TAG}/)"
fi

srun \
  --container-image=${SNOW_CONTAINER_IMAGE} \
  --container-mounts=${PWD}:${PWD},/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  --export=ALL,HF_TOKEN=$(cat /shared/home/juan.osorio/.hf_token) \
  bash -c "
    set -e
    echo 'Started at \$(date)'
    nvidia-smi
    echo ''

    pip install --break-system-packages --quiet mlflow

    export MLFLOW_TRACKING_URI=file://${MLFLOW_DIR}
    export MLFLOW_ALLOW_FILE_STORE=true

    python ${SCRIPT_PATH} \
        --variant ${VARIANT} \
        --max-epochs ${MAX_EPOCHS} \
        --patience ${PATIENCE} \
        --num-workers ${NUM_WORKERS} \
        --mlflow-uri file://${MLFLOW_DIR} \
        --experiment ${EXPERIMENT:-multihead_dinov3_unet} \
        --pseudo-ground ${PSEUDO_GROUND} \
        --pseudo-snow   ${PSEUDO_SNOW} \
        --pseudo-epochs ${PSEUDO_EPOCHS} \
        --pseudo-ground-cache ${PSEUDO_GROUND_CACHE} \
        --pseudo-snow-cache   ${PSEUDO_SNOW_CACHE} \
        ${BATCH_FLAG} \
        ${LR_FLAG} \
        ${SKIP_PSEUDO_FLAG} \
        ${RUN_TAG_FLAG}

    echo ''
    echo "Finished at $(date)"
  "

echo "Job finished!"
