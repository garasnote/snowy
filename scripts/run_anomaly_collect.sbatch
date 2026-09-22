#!/bin/bash
#SBATCH --job-name=anomaly-collect
#SBATCH --partition=frida
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/anomaly_collect_%j.out

MASTER_DIR="/shared/home/juan.osorio/master"
METHOD="${METHOD:-both}"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "ANOMALY DETECTION — COLLECT RESULTS"
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
    python collect_results.py --method ${METHOD} --from-checkpoints
  "

echo ""
echo "Job finished!"
