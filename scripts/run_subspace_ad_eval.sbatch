#!/bin/bash
#SBATCH --job-name=subspace-eval
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=subspace_ad_eval_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
ANN_DIR="${ANN_DIR:-${MASTER_DIR}/data/annotations}"
EVAL_SCRIPT="${MASTER_DIR}/models/subspace_ad/eval_iou.py"
CONTAINER="/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs"

mkdir -p "/shared/home/${CLUSTER_USER}/output"

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    python ${EVAL_SCRIPT} --ann-dir ${ANN_DIR}
  "
