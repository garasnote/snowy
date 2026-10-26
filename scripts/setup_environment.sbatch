#!/bin/bash
# This script sets up the environment and saves it as a container image
# SAM3 Requirements: Python >=3.12, PyTorch >=2.7, CUDA >=12.6
# Run with: bash setup_environment.sbatch (or sbatch setup_environment.sbatch)
#
# Not written for a generic HPC layout — invoke from the parent directory
# of this repo (so "master/requirements.txt" below resolves), with
# SNOW_CONTAINER_IMAGE set to where the built image should be saved.
# Outside the cluster (Slurm + Pyxis/enroot), use the repo's Dockerfile instead —
# it mirrors this recipe exactly and builds with plain `docker build`.

echo "========================================="
echo "Setting up Snow Detection Environment"
echo "SAM3 Prerequisites: Python 3.12+, PyTorch 2.7+, CUDA 12.6+"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Running on node: ${SLURMD_NODENAME:-$(hostname)}"
echo ""

# Start with a base Ubuntu container that has Python 3.12+ and CUDA 12.6
# Using Ubuntu 24.04 which has Python 3.12
srun \
  --partition=dev \
  --time=01:30:00 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --container-image=nvcr.io#nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04 \
  --container-mounts=${PWD}:${PWD} \
  --container-workdir=${PWD} \
  --container-save=${SNOW_CONTAINER_IMAGE} \
  bash -c '
    set -e

    echo "Installing Python 3.12 and system dependencies..."
    apt-get update
    apt-get install -y python3.12 python3.12-venv python3-pip git libgl1 libglib2.0-0

    # Make python3.12 the default
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

    echo ""
    echo "Container environment info:"
    echo "  Python: $(python --version)"
    python -c "import sys; assert sys.version_info >= (3, 12), \"Python 3.12+ required for SAM3\""
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    nvcc --version | grep "release" || echo "  CUDA Toolkit: (will be verified after PyTorch install)"
    echo ""

    echo "Checking pip version..."
    pip --version

    echo ""
    echo "Installing PyTorch 2.7.0 with CUDA 12.6 (SAM3 requirement)..."
    pip install --break-system-packages torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

    echo ""
    echo "Verifying PyTorch installation..."
    python -c "import torch; assert torch.__version__.startswith(\"2.7\"), \"PyTorch 2.7+ required\""
    python -c "import torch; assert torch.version.cuda >= \"12.6\", \"CUDA 12.6+ required\""
    python -c "import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}\")"

    echo ""
    echo "Installing dependencies from requirements.txt..."
    pip install --break-system-packages -r master/requirements.txt

    echo ""
    echo "Cloning and installing SAM3..."
    if [ ! -d "master/sam3" ]; then
        git clone https://github.com/facebookresearch/sam3.git
        cd sam3
        pip install --break-system-packages -e .
        echo ""
        echo "Installing SAM3 notebook dependencies (includes huggingface-hub for checkpoint access)..."
        pip install --break-system-packages -e ".[notebooks]"
        cd ..
    else
        echo "SAM3 directory already exists, skipping clone..."
        cd master/sam3
        pip install --break-system-packages -e .
        pip install --break-system-packages -e ".[notebooks]"
        cd ..
    fi

    echo ""
    echo "Final verification..."
    python -c "import torch; print(f\"✓ PyTorch: {torch.__version__}\")"
    python -c "import torch; print(f\"✓ CUDA available: {torch.cuda.is_available()}\")"
    python -c "import torch; print(f\"✓ CUDA version: {torch.version.cuda}\")"
    python -c "import cv2; print(f\"✓ OpenCV: {cv2.__version__}\")"
    python -c "import numpy; print(f\"✓ NumPy: {numpy.__version__}\")"
    python -c "import pandas; print(f\"✓ Pandas: {pandas.__version__}\")"
    python -c "import lightning; print(f\"✓ Lightning: {lightning.__version__}\")"
    python -c "from sam3.model_builder import build_sam3_image_model; print(\"✓ SAM3 installed successfully\")"

    echo ""
    echo "========================================="
    echo "Environment setup complete!"
    echo "========================================="
    echo ""
    echo "Remember to authenticate with: hf auth login for SAM3"
    echo ""
  '

echo ""
echo "========================================="
echo "Container saved to: ${SNOW_CONTAINER_IMAGE}"
echo "========================================="
echo ""
echo "Use this container for your jobs with:"
echo "  --container-image=${SNOW_CONTAINER_IMAGE}"
