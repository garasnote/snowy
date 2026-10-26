#!/bin/bash
#SBATCH --job-name=snow-train
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=output/train_%x_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# Training script for snow detection
# Usage: sbatch train_snow_detection.sbatch

SNOW_CONTAINER_IMAGE="/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs"

# Check if container exists
if [ ! -f "${SNOW_CONTAINER_IMAGE}" ]; then
    echo "ERROR: Container image not found at ${SNOW_CONTAINER_IMAGE}"
    echo "Please run 'sbatch setup_environment.sbatch' first to create the environment."
    exit 1
fi

echo "========================================="
echo "Snow Detection Training Job"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${SLURM_GPUS_PER_NODE}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "========================================="
echo ""

# Run training
srun \
  --container-image=${SNOW_CONTAINER_IMAGE} \
  --container-mounts=${PWD}:${PWD},${SCRATCH}:/scratch \
  --container-workdir=${PWD} \
  bash -c '
    echo "Starting training at $(date)"
    echo ""

    # Display GPU info
    nvidia-smi
    echo ""

    # Run your training script
    python train.py

    echo ""
    echo "Training completed at $(date)"
  '

echo ""
echo "Job finished!"
