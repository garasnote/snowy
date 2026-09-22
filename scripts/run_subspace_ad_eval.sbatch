#!/bin/bash
#SBATCH --job-name=subspace-eval
#SBATCH --partition=frida
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/shared/home/juan.osorio/output/subspace_ad_eval_%j.out

MASTER_DIR="/shared/home/juan.osorio/master"
ANN_DIR="${ANN_DIR:-${MASTER_DIR}/data/annotations}"
EVAL_SCRIPT="${MASTER_DIR}/models/subspace_ad/eval_iou.py"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

mkdir -p "/shared/home/juan.osorio/output"

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    python ${EVAL_SCRIPT} --ann-dir ${ANN_DIR}
  "
