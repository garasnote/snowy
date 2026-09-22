#!/bin/bash
#SBATCH --job-name=hp-snow
#SBATCH --partition=frida
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/hp_snow_%j.out

# Optuna TPE hyperparameter search for Snow MobileNet (GT labels).
# Single job — Optuna runs trials sequentially.
# SQLite DB enables crash-resume: resubmit same sbatch to continue.
#
# Usage:
#   sbatch scripts/frida/run_hp_snow.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "SNOW MOBILENET HP SEARCH — Optuna TPE (GT labels)"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       $(hostname)"
echo "  Cameras:    CF00001 CM00009 CM00014 CM00070 CM00075 CM00080 CM00102 CM00108 CM00112"
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
    echo 'Starting Snow Optuna HP search at \$(date)'
    nvidia-smi
    echo ''

    python ${MASTER_DIR}/scripts/hp_search_snow.py \
        --n-trials 60 \
        --epochs 200 \
        --num-workers 4 \
        --db ${MASTER_DIR}/scripts/optuna_snow.db

    echo ''
    echo 'HP search completed at \$(date)'
  "

echo ""
echo "Job finished!"
