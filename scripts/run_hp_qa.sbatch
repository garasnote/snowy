#!/bin/bash
#SBATCH --job-name=hp-qa
#SBATCH --partition=amd
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/shared/home/juan.osorio/output/hp_qa_%j.out

# Hyperparameter search for QA HistGradientBoosting.
# Uses sklearn RandomizedSearchCV with 5-fold CV.
# CPU-only — no GPU needed.
#
# Usage:
#   sbatch scripts/frida/run_hp_qa.sbatch
#   HP_ITER=100 sbatch scripts/frida/run_hp_qa.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
HP_ITER="${HP_ITER:-500}"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/cpu_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "QA HYPERPARAMETER SEARCH (FRIDA)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  HP iters:    ${HP_ITER}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    echo 'Starting QA HP search at \$(date)'
    echo ''

    python ${MASTER_DIR}/preprocessing/qa/train_qa.py \
        --ann-dir ${MASTER_DIR}/data/annotations \
        --img-dir ${MASTER_DIR}/data\
        --meteo ${MASTER_DIR}/data/meteo_hourly.csv \
        --output-dir ${MASTER_DIR}/preprocessing/qa/checkpoints \
        --hp-search \
        --hp-iter ${HP_ITER}

    echo ''
    echo 'QA HP search completed at \$(date)'
  "

echo ""
echo "Job finished!"
