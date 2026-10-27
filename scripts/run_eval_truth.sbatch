#!/bin/bash
#SBATCH --job-name=eval-truth
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=eval_truth_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"
#
# Manuscript-vs-measured truth run: evaluates the tasks the paper reports on the
# full in-distribution frame set and audits whether each checkpoint actually
# loads into the model the eval builds. Purpose is to find every place where the
# shipped code and the manuscript disagree *before* a reviewer does.
#
# Why this exists: a CPU run of just the snow students appeared to be ~0.06 off
# the manuscript for two of three rows (an artefact of --val-only vs the full
# set), and the third (ViT-LoRA) silently loaded a model with 84 randomly
# initialised layers and reported an IoU anyway. See
# paper/eval_reproduction_findings.md.
#
# What this covers: ground, snow, qa, smoother on the full frame set; the snow
# students on the temporal val split; the ground teacher/ensemble/baseline
# per-camera breakdown; and the checkpoint-key parity audit.
#
# What this does NOT cover: the OOD columns (ground PL/FT OOD, snow OOD,
# per-camera OOD, QA OOD). Those come from the *_ood types and are already a
# separate, working job in this repo:
#     cd ~/master && sbatch scripts/run_eval_ood.sbatch
# They were briefly appended here as well; the run hit its wall-clock limit at
# that section and lost it, which is why they belong in their own job.
#
# Usage:
#   sbatch run_eval_truth.sbatch
#   TASKS="snow" sbatch run_eval_truth.sbatch          # just one task
#
# Prerequisite: the checkpoint-key audit script, pushed to the master dir. If it
# is absent the audit is skipped *silently*, so check for the PARITY section in
# the log:
#   the cluster-rsync push ~/Documents/Uni/Thesis/paper-code-release/diag_keys.py \
#                    /shared/home/${CLUSTER_USER}/master/

MASTER_DIR="${MASTER_DIR:-/shared/home/${CLUSTER_USER}/master}"
CONTAINER="${SNOW_CONTAINER_IMAGE:-/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs}"
TASKS="${TASKS:-ground snow qa smoother}"

mkdir -p /shared/home/${CLUSTER_USER}/output

echo "============================================================"
echo "EVAL TRUTH RUN"
echo "  Job ID: ${SLURM_JOB_ID}   Node: $(hostname)"
echo "  Tasks:  ${TASKS}"
echo "============================================================"

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  --export=ALL,HF_TOKEN=$(cat /shared/home/${CLUSTER_USER}/.hf_token) \
  bash -c "
    set -e
    echo 'Job start:'; date

    # 1) Do the checkpoints load whole? Silent partial loads are the failure mode
    #    that produced the ViT-LoRA problem, so this runs first and is not fatal.
    if [ -f diag_keys.py ]; then
      echo ''
      echo '=== CHECKPOINT / MODEL KEY PARITY ==='
      python -u diag_keys.py || true
    else
      echo ''
      echo '=== CHECKPOINT / MODEL KEY PARITY: SKIPPED, diag_keys.py not in ${MASTER_DIR} ==='
    fi

    for T in ${TASKS}; do
      echo ''
      echo '==============================================='
      echo \"=== TASK: \${T} (in-distribution, full frame set) ===\"
      echo '==============================================='
      python -u eval_all.py --task \${T} --device cuda || echo \"  [task \${T} exited non-zero]\"
    done

    echo ''
    echo '==============================================='
    echo '=== SNOW STUDENTS ON THE TEMPORAL VAL SPLIT ==='
    echo '==============================================='
    python -u eval_all.py --task snow --types students --device cuda || true

    echo ''
    echo '==============================================='
    echo '=== GROUND TEACHERS / ENSEMBLES / BASELINES, PER-CAMERA BREAKDOWN ==='
    echo '==============================================='
    # NOTE: --breakdown only produces a per-camera table for methods that write
    # prediction PNGs (teachers, ensembles, baselines). Combining it with
    # --types students yields an all-dash table with no warning, so it is not
    # used on the student run above.
    python -u eval_all.py --task ground --types teachers ensembles baselines --breakdown --device cuda || true

    echo ''
    echo 'OOD columns intentionally not run here: use scripts/run_eval_ood.sbatch'
    echo 'Job end:'; date
  "

echo "Job finished!"
