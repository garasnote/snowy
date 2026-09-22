#!/bin/bash
#SBATCH --job-name=baseline-ground
#SBATCH --partition=amd
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/shared/home/juan.osorio/output/baseline_ground_%j.out

MASTER_DIR="/shared/home/juan.osorio/master"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/cpu_env.sqfs"

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "BASELINE GROUND SEGMENTATION (fixed + sobel)"
echo "============================================================"

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    python segmentation/baseline_ground.py --data-dir data/
  "

echo ""
echo "Job finished!"
