#!/bin/bash
#SBATCH --job-name=subspace-auroc
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=subspace_ad_auroc_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
FIT_SCRIPT="${MASTER_DIR}/models/subspace_ad/fit.py"
EVAL_SCRIPT="${MASTER_DIR}/models/subspace_ad/evaluate.py"
CONTAINER="/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs"

K_SHOT="${K_SHOT:-4}"
N_SEEDS="${N_SEEDS:-5}"
STRATIFY="${STRATIFY:-}"

STRATIFY_FLAG=""
if [ "${STRATIFY}" = "1" ]; then
    STRATIFY_FLAG="--stratify"
fi

mkdir -p "/shared/home/${CLUSTER_USER}/output"

echo "============================================================"
echo "SUBSPACE-AD — FIT + EVAL (AUROC + IoU)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  K-shot:      ${K_SHOT}"
echo "  Stratify:    ${STRATIFY:-no}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    nvidia-smi
    echo ''

    echo '=== Step 1: Fit PCA ==='
    python ${FIT_SCRIPT} \
        --all-cameras \
        --k-shot ${K_SHOT} \
        --n-seeds ${N_SEEDS} \
        ${STRATIFY_FLAG}

    echo ''
    echo '=== Step 2: Evaluate ==='
    python ${EVAL_SCRIPT} \
        --k-shot ${K_SHOT} \
        --n-seeds ${N_SEEDS} \
        --save-maps
  "

echo ""
echo "Job finished!"
