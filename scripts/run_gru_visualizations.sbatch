#!/bin/bash
#SBATCH --job-name=gru-viz
#SBATCH --partition=frida
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/shared/home/juan.osorio/output/gru_viz_%j.out

# Generate animations + calendar heatmaps from trained GRU smoother.
#
# Usage:
#   CKPT=models/temporal_snow/checkpoints/<run>/best-*.ckpt sbatch scripts/frida/run_gru_visualizations.sbatch
#   CKPT=... CAMERAS="CM00014 CF00001" sbatch scripts/frida/run_gru_visualizations.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
CKPT="${CKPT:?ERROR: set CKPT=path/to/smoother/checkpoint.ckpt}"
CSV="${CSV:-${MASTER_DIR}/analysis/snow_timeseries.csv}"
IMG_DIR="${IMG_DIR:-${MASTER_DIR}/data}"
SEQ_LEN="${SEQ_LEN:-72}"
CAMERAS="${CAMERAS:-}"
FPS="${FPS:-4}"
MAX_PERIODS="${MAX_PERIODS:-3}"

CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

mkdir -p "/shared/home/juan.osorio/output"

CAM_FLAG=""
if [ -n "${CAMERAS}" ]; then
    CAM_FLAG="--cameras ${CAMERAS}"
fi

echo "============================================================"
echo "GRU SMOOTHER — VISUALIZATIONS (FRIDA)"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Checkpoint: ${CKPT}"
echo "  CSV:        ${CSV}"
echo "  Seq len:    ${SEQ_LEN}"
echo "  Cameras:    ${CAMERAS:-all}"
echo "  Container:  ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    echo 'Starting visualizations at \$(date)'
    nvidia-smi
    echo ''

    echo '=== Calendar heatmaps ==='
    python ${MASTER_DIR}/models/temporal_snow/visualize_smoother.py calendar \
        --checkpoint ${CKPT} \
        --csv ${CSV} \
        --seq-len ${SEQ_LEN} \
        ${CAM_FLAG}
    echo ''

    echo '=== Full station timelines ==='
    python ${MASTER_DIR}/models/temporal_snow/visualize_smoother.py timeline \
        --checkpoint ${CKPT} \
        --csv ${CSV} \
        --seq-len ${SEQ_LEN} \
        --summary \
        ${CAM_FLAG}
    echo ''

    echo '=== Full camera timelapses ==='
    python ${MASTER_DIR}/models/temporal_snow/visualize_smoother.py timelapse \
        --checkpoint ${CKPT} \
        --csv ${CSV} \
        --img-dir ${IMG_DIR} \
        --seq-len ${SEQ_LEN} \
        --fps ${FPS} \
        ${CAM_FLAG}
    echo ''

    echo '=== Period animations ==='
    python ${MASTER_DIR}/models/temporal_snow/animate_periods.py \
        --checkpoint ${CKPT} \
        --csv ${CSV} \
        --img-dir ${IMG_DIR} \
        --seq-len ${SEQ_LEN} \
        --fps ${FPS} \
        --max-periods-per-camera ${MAX_PERIODS} \
        ${CAM_FLAG}
    echo ''

    echo 'Visualizations completed at \$(date)'
  "

echo ""
echo "Job finished!"
