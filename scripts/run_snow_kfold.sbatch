#!/bin/bash
#SBATCH --job-name=snow-kfold
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=snow_kfold_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# K-fold CV finetuning on GT snow annotations (5-fold stratified).
# Initializes from pseudo-label checkpoint (majority_3), trains all 5 folds sequentially.
#
# Usage:
#   MODEL=mobilenet PRETRAIN_CKPT=<path> sbatch scripts/run_snow_kfold.sbatch
#   MODEL=convnext  PRETRAIN_CKPT=<path> sbatch scripts/run_snow_kfold.sbatch
#   MODEL=vit_lora  PRETRAIN_CKPT=<path> sbatch scripts/run_snow_kfold.sbatch

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
ANN_DIR="${ANN_DIR:-${MASTER_DIR}/data/annotations}"
MODEL="${MODEL:-mobilenet}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
UNFREEZE="${UNFREEZE:-1}"
PRECISION="${PRECISION:-16-mixed}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
KFOLD_K="${KFOLD_K:-5}"

SCRIPT="${MASTER_DIR}/models/student_snow/train.py"
CONTAINER="/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs"

HF_TOKEN_FILE="/shared/home/${CLUSTER_USER}/.hf_token"
HF_TOKEN=""
if [ -f "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN=$(cat "${HF_TOKEN_FILE}" | tr -d '[:space:]')
fi

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/${CLUSTER_USER}/output"

UNFREEZE_FLAG=""
if [ "${UNFREEZE}" -eq 1 ] 2>/dev/null; then
    UNFREEZE_FLAG="--unfreeze"
fi

PRETRAIN_FLAG=""
if [ -n "${PRETRAIN_CKPT}" ]; then
    PRETRAIN_FLAG="--pretrain-ckpt ${PRETRAIN_CKPT}"
fi

echo "============================================================"
echo "SNOW K-FOLD CV — GT Finetuning (cluster)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Model:       ${MODEL}"
echo "  Pretrain:    ${PRETRAIN_CKPT:-none}"
echo "  Epochs:      ${EPOCHS}"
echo "  LR:          ${LR}"
echo "  Folds:       ${KFOLD_K}"
echo "============================================================"
echo ""

for fold in 0 1 2 3 4; do
    echo ""
    echo ">>> FOLD $((fold+1))/${KFOLD_K} at $(date)"
    echo ""

    srun \
      --container-image=${CONTAINER} \
      --container-mounts=/shared:/shared \
      --container-workdir=${MASTER_DIR} \
      bash -c "
        set -e
        export HF_TOKEN='${HF_TOKEN}'
        pip install --break-system-packages --quiet tensorboard 2>/dev/null || true

        python ${SCRIPT} \
            --gt \
            --model-variant ${MODEL} \
            --data-dir ${DATA_DIR} \
            --ann-dir ${ANN_DIR} \
            --epochs ${EPOCHS} \
            --lr ${LR} \
            --batch-size ${BATCH_SIZE} \
            --precision ${PRECISION} \
            --num-workers 8 \
            --kfold-idx ${fold} \
            --kfold-k ${KFOLD_K} \
            ${UNFREEZE_FLAG} \
            ${PRETRAIN_FLAG}

        echo 'Fold ${fold} finished at \$(date)'
      "
done

echo ""
echo "============================================================"
echo "K-FOLD CV Complete — All folds trained"
echo "============================================================"
