#!/bin/bash
#SBATCH --job-name=student-snow-eval
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=student_snow_eval_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# Evaluate student snow checkpoint against GT annotations.
# Step 1: inference on all images
# Step 2: eval_iou against GT
#
# Usage:
#   CKPT=checkpoints/snow_mobilenet_.../best.ckpt sbatch scripts/run_student_snow_eval.sbatch
#   CKPT=... VARIANT=convnext sbatch scripts/run_student_snow_eval.sbatch

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
DATA_DIR="${DATA_DIR:-${MASTER_DIR}/data}"
ANN_DIR="${ANN_DIR:-${MASTER_DIR}/data/annotations}"
CKPT="${CKPT:?ERROR: set CKPT=path/to/checkpoint.ckpt}"
VARIANT="${VARIANT:-mobilenet}"
THRESHOLD="${THRESHOLD:-0.5}"

INFER_SCRIPT="${MASTER_DIR}/models/student_snow/inference.py"
EVAL_SCRIPT="${MASTER_DIR}/models/subspace_ad/eval_iou.py"
CONTAINER="/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs"

# Resolve relative checkpoint path
if [[ "${CKPT}" != /* ]]; then
    CKPT="${MASTER_DIR}/models/student_snow/${CKPT}"
fi

OUTPUT_DIR="${MASTER_DIR}/preprocessing/segmentation/student_snow_output/${VARIANT}"

mkdir -p "/shared/home/${CLUSTER_USER}/output"

echo "============================================================"
echo "STUDENT SNOW — EVAL vs GT"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Checkpoint:  ${CKPT}"
echo "  Variant:     ${VARIANT}"
echo "  Threshold:   ${THRESHOLD}"
echo "  Output:      ${OUTPUT_DIR}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    echo 'Step 1: Inference'
    python ${INFER_SCRIPT} \
        --checkpoint ${CKPT} \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --model-variant ${VARIANT} \
        --threshold ${THRESHOLD}

    echo ''
    echo 'Step 2: Eval vs GT'
    python ${EVAL_SCRIPT} \
        --mask-dir ${OUTPUT_DIR} \
        --ann-dir ${ANN_DIR}

    echo ''
    echo 'Eval completed at \$(date)'
  "

echo ""
echo "Job finished!"
