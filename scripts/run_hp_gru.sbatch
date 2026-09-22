#!/bin/bash
#SBATCH --job-name=hp-gru
#SBATCH --partition=frida
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/hp_gru_%j.out

# Optuna TPE hyperparameter search for GRU temporal smoother.
# Single job — Optuna runs trials sequentially within process.
# SQLite DB enables crash-resume: resubmit same sbatch to continue.
#
# Usage:
#   sbatch scripts/frida/run_hp_gru.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "GRU HP SEARCH — Optuna TPE"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       $(hostname)"
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
    echo 'Starting GRU Optuna HP search at \$(date)'
    nvidia-smi
    echo ''

    python ${MASTER_DIR}/scripts/hp_search_gru.py \
        --n-trials 80 \
        --epochs 100 \
        --num-workers 4 \
        --csv ${MASTER_DIR}/analysis/snow_timeseries.csv \
        --db ${MASTER_DIR}/scripts/optuna_gru.db

    echo ''
    echo 'HP search completed at \$(date)'
  "

echo ""
echo "Job finished!"
