#!/bin/bash
#SBATCH --job-name=qa-train
#SBATCH --partition=amd
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=qa_train_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# Train HistGradientBoosting QA classifier.
# CPU-only — no GPU needed. Produces qa_gradboost.pkl for predict_qa.py.

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
CONTAINER="/shared/workspace/lkm/${CLUSTER_USER}/container/cpu_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/${CLUSTER_USER}/output"

echo "============================================================"
echo "QA CLASSIFIER — HISTGRADIENTBOOSTING (cluster)"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    echo 'Starting QA training at \$(date)'
    echo ''

    python ${MASTER_DIR}/qa/train_qa.py \
        --ann-dir ${MASTER_DIR}/data/annotations \
        --img-dir ${MASTER_DIR}/data/balanced_snow \
        --meteo ${MASTER_DIR}/data/meteo_hourly.csv \
        --output-dir ${MASTER_DIR}/qa/checkpoints \
        --n-estimators 300 \
        --max-depth 5 \
        --plot-mistakes

    echo ''
    echo 'Verifying output...'
    ls -la ${MASTER_DIR}/qa/checkpoints/qa_gradboost.pkl

    echo ''
    echo 'QA training completed at \$(date)'
  "

echo ""
echo "Job finished!"
