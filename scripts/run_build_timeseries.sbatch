#!/bin/bash
#SBATCH --job-name=build-ts
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=build_timeseries_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# Rebuild snow_timeseries.csv with CNN QA filtering.
# GPU needed for CNN QA inference (~70K images).
#
# Usage:
#   sbatch scripts/run_build_timeseries.sbatch

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
CONTAINER="/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/${CLUSTER_USER}/output"

echo "============================================================"
echo "BUILD SNOW TIMESERIES — with CNN QA"
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
    echo 'Starting timeseries build at \$(date)'
    nvidia-smi
    echo ''

    python ${MASTER_DIR}/analysis/build_features.py timeseries \
        --data-dir ${MASTER_DIR}/data \
        --workers 8 \
        --qa-threshold 0.5

    echo ''
    echo 'Verifying output...'
    wc -l ${MASTER_DIR}/analysis/snow_timeseries.csv
    python -c \"
import pandas as pd
df = pd.read_csv('${MASTER_DIR}/analysis/snow_timeseries.csv')
n_rej = (df['qa_usable']==0).sum()
print(f'Total: {len(df)} rows')
print(f'QA rejected: {n_rej} ({100*n_rej/len(df):.1f}%)')
print(f'Cameras: {sorted(df[\"camera\"].unique())}')
\"

    echo ''
    echo 'Build completed at \$(date)'
  "

echo ""
echo "Job finished!"
