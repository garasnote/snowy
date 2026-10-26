#!/bin/bash
#SBATCH --job-name=smoother-train
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=smoother_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# Train temporal smoother (GRU/LSTM, uni/bi).
# Usage:
#   sbatch run_train_smoother.sbatch                          # uni-GRU (default)
#   ARCH=lstm BI=1 sbatch run_train_smoother.sbatch          # bi-LSTM
#   ARCH=gru  BI=1 sbatch run_train_smoother.sbatch          # bi-GRU

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
CONTAINER="/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs"

ARCH="${ARCH:-gru}"
BI="${BI:-0}"
HIDDEN="${HIDDEN:-128}"
LAYERS="${LAYERS:-2}"
SEQ_LEN="${SEQ_LEN:-96}"
EPOCHS="${EPOCHS:-150}"
LR="${LR:-1e-3}"
DROPOUT="${DROPOUT:-0.3}"
KERNEL="${KERNEL:-7}"

BI_FLAG=""
if [ "${BI}" = "1" ]; then BI_FLAG="--bidirectional"; fi

echo "============================================================"
echo "SMOOTHER TRAINING — ${ARCH} bi=${BI} h=${HIDDEN} seq=${SEQ_LEN}"
echo "============================================================"
echo "  Job:  ${SLURM_JOB_ID}  Node: $(hostname)"
echo "============================================================"

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    python models/temporal_snow/train_smoother.py \
        --arch ${ARCH} \
        --hidden-size ${HIDDEN} \
        --num-layers ${LAYERS} \
        --seq-len ${SEQ_LEN} \
        --kernel-size ${KERNEL} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --dropout ${DROPOUT} \
        ${BI_FLAG}
    echo 'Done at \$(date)'
  "
