#!/bin/bash
#SBATCH --job-name=multihead-hp
#SBATCH --partition=dev
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=output/multihead/hp_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

# Optuna TPE hyperparameter search for DINOv3 multi-head model.
# Each trial trains with EarlyStopping; Shapley importances logged at end.
#
# Usage:
#   sbatch run_multihead_hp.sbatch                    # 20 trials, 100 epochs each
#   N_TRIALS=5 MAX_EPOCHS=5 sbatch run_multihead_hp.sbatch   # quick test
#   VARIANT=dinov3_large sbatch run_multihead_hp.sbatch

VARIANT="${VARIANT:-dinov3_base}"
N_TRIALS="${N_TRIALS:-20}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STUDY_NAME="${STUDY_NAME:-multihead_tpe}"

MASTER_DIR="/shared/home/${CLUSTER_USER}/master"
SCRIPT_PATH="${MASTER_DIR}/models/multi_head/hp_search_multihead.py"
MLFLOW_DIR="/shared/home/${CLUSTER_USER}/mlruns"
DB_PATH="${MASTER_DIR}/models/multi_head/checkpoints/hp_search/optuna.db"

mkdir -p output/multihead
mkdir -p "$(dirname ${DB_PATH})"

echo "============================================================"
echo "MULTIHEAD HP SEARCH — OPTUNA TPE"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo ""
echo "Variant:     ${VARIANT}"
echo "Trials:      ${N_TRIALS}"
echo "Max epochs:  ${MAX_EPOCHS} (ceiling — patience decides early stop)"
echo "Study name:  ${STUDY_NAME}"
echo "Optuna DB:   ${DB_PATH}"
echo "MLflow dir:  ${MLFLOW_DIR}"
echo "============================================================"
echo ""

srun \
  --container-image=${SNOW_CONTAINER_IMAGE} \
  --container-mounts=${PWD}:${PWD},/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  --export=ALL,HF_TOKEN=$(cat /shared/home/${CLUSTER_USER}/.hf_token) \
  bash -c "
    set -e
    echo 'Started at \$(date)'
    nvidia-smi
    echo ''

    pip install --break-system-packages --quiet mlflow optuna optuna-integration

    export MLFLOW_TRACKING_URI=file://${MLFLOW_DIR}
    export MLFLOW_ALLOW_FILE_STORE=true

    python ${SCRIPT_PATH} \
        --variant ${VARIANT} \
        --n-trials ${N_TRIALS} \
        --max-epochs ${MAX_EPOCHS} \
        --num-workers ${NUM_WORKERS} \
        --mlflow-uri file://${MLFLOW_DIR} \
        --experiment multihead_${VARIANT} \
        --study-name ${STUDY_NAME} \
        --storage sqlite:///${DB_PATH}

    echo ''
    echo "Finished at $(date)"
  "

echo "Job finished!"
