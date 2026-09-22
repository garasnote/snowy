#!/bin/bash
#SBATCH --job-name=eval-ood
#SBATCH --partition=frida
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/shared/home/juan.osorio/output/eval_ood_%j.out

# OOD evaluation on new cohort: CM00054, CM00071, CM00115
#
# Usage:
#   sbatch run_eval_ood.sbatch                         # all OOD types
#   GROUND_TYPES="baselines_ood students_ood ood" sbatch run_eval_ood.sbatch
#   SNOW_TYPES="baselines_ood students_ood students_gt_ood" sbatch run_eval_ood.sbatch
#   QA_TYPES="cnn gradboost qa_ood" sbatch run_eval_ood.sbatch

MASTER_DIR="/shared/home/juan.osorio/master"
CONTAINER="${SNOW_CONTAINER_IMAGE:-/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs}"

GROUND_TYPES="${GROUND_TYPES:-baselines_ood students_ood ood}"
SNOW_TYPES="${SNOW_TYPES:-baselines_ood students_pseudo_ood students_ood students_gt_ood}"
QA_TYPES="${QA_TYPES:-cnn gradboost qa_ood}"

mkdir -p /shared/home/juan.osorio/output

echo "============================================================"
echo "OOD EVALUATION — CM00054 / CM00071 / CM00115"
echo "============================================================"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Node:         $(hostname)"
echo "  Ground types: ${GROUND_TYPES}"
echo "  Snow types:   ${SNOW_TYPES}"
echo "  QA types:     ${QA_TYPES}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  --export=ALL,HF_TOKEN=$(cat /shared/home/juan.osorio/.hf_token) \
  bash -c "
    set -e
    echo 'Started at \$(date)'
    nvidia-smi
    echo ''

    echo '=== GROUND OOD ==='
    python eval_all.py --task ground --types ${GROUND_TYPES}

    echo ''
    echo '=== SNOW OOD ==='
    python eval_all.py --task snow --types ${SNOW_TYPES}

    echo ''
    echo '=== QA OOD ==='
    python eval_all.py --task qa --types ${QA_TYPES}

    echo ''
    echo 'Finished at \$(date)'
  "

echo "Job finished!"
