#!/bin/bash
#SBATCH --job-name=eval-qa
#SBATCH --partition=frida
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/eval_qa_%j.out

MASTER_DIR="/shared/home/juan.osorio/master"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"
TYPES="${TYPES:-gradboost cnn patchcore}"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "QA EVALUATION (FRIDA)"
echo "============================================================"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Types:     ${TYPES}"
echo "  Container: ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    pip install --break-system-packages --quiet anomalib 2>/dev/null || true
    echo 'Starting QA eval at \$(date)'
    nvidia-smi
    echo ''

    python eval_all.py --task qa --types ${TYPES}

    echo ''
    echo 'QA eval completed at \$(date)'
  "

echo ""
echo "Job finished!"
