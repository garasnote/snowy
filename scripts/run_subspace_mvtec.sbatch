#!/bin/bash
#SBATCH --job-name=subspace-mvtec
#SBATCH --partition=frida
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/subspace_mvtec_%j.out

# Evaluate SubspaceAD on MVTec AD benchmark.
#
# Usage:
#   sbatch scripts/frida/run_subspace_mvtec.sbatch
#   K_SHOT=8 sbatch scripts/frida/run_subspace_mvtec.sbatch
#   VARIANT=dinov2_vitl14 sbatch scripts/frida/run_subspace_mvtec.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
SCRIPT="${MASTER_DIR}/models/subspace_ad/eval_mvtec.py"
CONTAINER="/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs"

MVTEC_DIR="${MVTEC_DIR:-/shared/home/juan.osorio/ml/data/mvtec_anomaly_detection}"

K_SHOT="${K_SHOT:-4}"
N_SEEDS="${N_SEEDS:-3}"
VARIANT="${VARIANT:-dinov2_vitb14}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"

mkdir -p "/shared/home/juan.osorio/output"

echo "============================================================"
echo "SUBSPACE-AD — MVTec AD Benchmark"
echo "============================================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  Variant:     ${VARIANT}"
echo "  K-shot:      ${K_SHOT}"
echo "  Seeds:       ${N_SEEDS}"
echo "  Image size:  ${IMAGE_SIZE}"
echo "  MVTec dir:   ${MVTEC_DIR}"
echo "  Container:   ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    set -e
    nvidia-smi
    echo ''

    python ${SCRIPT} \
        --mvtec-dir ${MVTEC_DIR} \
        --variant ${VARIANT} \
        --image-size ${IMAGE_SIZE} \
        --k-shot ${K_SHOT} \
        --n-seeds ${N_SEEDS}
  "

echo ""
echo "Job finished!"
