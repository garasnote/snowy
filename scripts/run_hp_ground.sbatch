#!/bin/bash
#SBATCH --job-name=hp-ground
#SBATCH --partition=frida
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/hp_ground_%j.out

# Optuna TPE hyperparameter search for Ground MobileNet (majority-3 pseudo-labels).
# Single job — Optuna runs trials sequentially.
# SQLite DB enables crash-resume: resubmit same sbatch to continue.
#
# Ground has 89K images per trial (~30-60min each).
# 50 TPE trials covers the space far better than 192 grid configs.
#
# Usage:
#   sbatch scripts/frida/run_hp_ground.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "GROUND MOBILENET HP SEARCH — Optuna TPE (majority-3)"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       $(hostname)"
echo "  Split:      temporal (day-boundary, 1-day gap)"
echo "  Container:  ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    pip install --break-system-packages --quiet optuna tensorboard 2>/dev/null || true
    echo 'Starting Ground Optuna HP search at \$(date)'
    nvidia-smi
    echo ''

    python ${MASTER_DIR}/scripts/hp_search_ground.py \
        --n-trials 20 \
        --epochs 50 \
        --ensemble-threshold 3 \
        --num-workers 4 \
        --db ${MASTER_DIR}/scripts/optuna_ground.db

    echo ''
    echo 'HP search completed at \$(date)'
  "

echo ""
echo "Job finished!"
