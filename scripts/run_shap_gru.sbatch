#!/bin/bash
#SBATCH --job-name=shap-gru
#SBATCH --partition=frida
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/shared/home/juan.osorio/output/shap_gru_%j.out

MASTER_DIR="/shared/home/juan.osorio/master"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    pip install --break-system-packages --quiet shap 2>/dev/null || true
    echo 'SHAP analysis started at \$(date)'
    python models/temporal_snow/shap_gru.py
    echo 'SHAP analysis finished at \$(date)'
  "
