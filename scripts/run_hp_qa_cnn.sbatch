#!/bin/bash
#SBATCH --job-name=hp-qa-cnn
#SBATCH --partition=frida
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/shared/home/juan.osorio/output/hp_qa_cnn_%j.out

# Optuna HP search for tiny QA CNN classifier.
# Searches architecture + training HPs with 3-fold CV AUROC.
# SQLite DB enables crash-resume: resubmit same sbatch to continue.
#
# Usage:
#   sbatch scripts/frida/run_hp_qa_cnn.sbatch
#   N_TRIALS=50 sbatch scripts/frida/run_hp_qa_cnn.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
N_TRIALS="${N_TRIALS:-30}"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "QA CNN HP SEARCH — Optuna TPE"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  N trials:    ${N_TRIALS}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    pip install --break-system-packages --quiet optuna 2>/dev/null || true
    echo 'Starting QA CNN HP search at \$(date)'
    nvidia-smi
    echo ''

    python ${MASTER_DIR}/preprocessing/qa/hp_search_qa_cnn.py \
        --n-trials ${N_TRIALS} \
        --img-dir ${MASTER_DIR}/data/balanced_snow \
        --output-dir ${MASTER_DIR}/preprocessing/qa/checkpoints \
        --db ${MASTER_DIR}/preprocessing/qa/optuna_qa_cnn.db

    echo ''
    echo 'HP search completed at \$(date)'
  "

echo ""
echo "Job finished!"
