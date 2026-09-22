#!/bin/bash
#SBATCH --job-name=anomaly-iou
#SBATCH --partition=frida
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/shared/home/juan.osorio/output/anomaly_iou_%j.out

MASTER_DIR="/shared/home/juan.osorio/master"
METHOD="${METHOD:-both}"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "ANOMALY DETECTION — COMPUTE IoU"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Method:      ${METHOD}"
echo "============================================================"

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR}/models/anomaly_snow \
  bash -c "
    set -e
    pip install --break-system-packages --quiet anomalib 2>/dev/null || true
    python compute_iou.py --method ${METHOD}
  "

echo ""
echo "Job finished!"
