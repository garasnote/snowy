#!/bin/bash
#SBATCH --job-name=bad-img-rm
#SBATCH --partition=frida
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=output/bad_img_rm_%j.out

# Move bad images (night/dark, and corrupt/truncated) to a separate folder.
#
# Usage:
#   sbatch run_remove_bad_images.sbatch                    # move all bad images
#   DRY_RUN=1 sbatch run_remove_bad_images.sbatch          # preview only
#   MARGIN=30 sbatch run_remove_bad_images.sbatch           # 30 min twilight buffer
#   NO_CORRUPT_CHECK=1 sbatch run_remove_bad_images.sbatch  # skip corrupt-image check

# ── configurable defaults ────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/shared/home/juan.osorio/master/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/shared/home/juan.osorio/master/night_images}"
MARGIN="${MARGIN:-0}"
DRY_RUN="${DRY_RUN:-0}"
NO_CORRUPT_CHECK="${NO_CORRUPT_CHECK:-0}"

SCRIPT_PATH="/shared/home/juan.osorio/master/preprocessing/remove_bad_images.py"

# ── container check ──────────────────────────────────────────────────────────
if [ ! -f "${SNOW_CONTAINER_IMAGE}" ]; then
    echo "ERROR: Container image not found at ${SNOW_CONTAINER_IMAGE}"
    echo "Please run 'sbatch setup_environment.sbatch' first."
    exit 1
fi

mkdir -p output

echo "========================================="
echo "Bad Image Removal (night + corrupt)"
echo "========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo ""
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Margin:      ${MARGIN} min"
echo "Dry run:     ${DRY_RUN}"
echo "Corrupt check: $([ "${NO_CORRUPT_CHECK}" -eq 1 ] && echo disabled || echo enabled)"
echo "========================================="
echo ""

# Build flags
DRY_RUN_FLAG=""
if [ "${DRY_RUN}" -eq 1 ]; then
    DRY_RUN_FLAG="--dry-run"
fi
NO_CORRUPT_CHECK_FLAG=""
if [ "${NO_CORRUPT_CHECK}" -eq 1 ]; then
    NO_CORRUPT_CHECK_FLAG="--no-corrupt-check"
fi

srun \
  --container-image=${SNOW_CONTAINER_IMAGE} \
  --container-mounts=${PWD}:${PWD},/shared:/shared \
  --container-workdir=/shared/home/juan.osorio/master \
  --export=ALL \
  bash -c "
    set -e
    echo 'Started at \$(date)'

    python ${SCRIPT_PATH} \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --margin ${MARGIN} \
        ${DRY_RUN_FLAG} \
        ${NO_CORRUPT_CHECK_FLAG}

    echo ''
    echo 'Finished at \$(date)'
  "

echo ""
echo "Job finished!"
