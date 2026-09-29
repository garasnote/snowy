#!/bin/bash
#SBATCH --job-name=teacher-gaps
#SBATCH --partition=frida
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/shared/home/juan.osorio/output/teacher_gaps_%j.out

# Fill the snow masks that five teachers never produced for annotated frames.
#
# tools/fill_teacher_gaps.py runs ONE teacher over an explicit frame list and
# writes into the same preprocessing/segmentation/<teacher>_snow_output/ layout
# the pipeline already reads. This job runs it once per teacher, sequentially,
# so one failing teacher cannot abort the others.
#
# No transformers overlay here on purpose: that is only needed for Florence-2's
# remote code. These five load in the container exactly as they did when they
# produced the masks we already have.
#
# Usage:
#   sbatch run_teacher_gapfill.sbatch                  # all five, all frames
#   TEACHERS=clipseg sbatch run_teacher_gapfill.sbatch
#   LIMIT=5 sbatch run_teacher_gapfill.sbatch          # gauge the rate first

MASTER_DIR="${MASTER_DIR:-/shared/home/juan.osorio/master}"
HOME_ROOT="${HOME_ROOT:-/shared/home/juan.osorio}"
RELEASE_DIR="${RELEASE_DIR:-${MASTER_DIR}/release_check}"
SEG_ROOT="${SEG_ROOT:-${MASTER_DIR}/preprocessing/segmentation}"
FRAMES_DIR="${FRAMES_DIR:-${MASTER_DIR}/tools}"
ANN_DIR="${ANN_DIR:-${RELEASE_DIR}/data/annotations}"
DATA_DIRS="${DATA_DIRS:-${MASTER_DIR}/data ${MASTER_DIR}/night_images}"
TEACHERS="${TEACHERS:-clipseg groundedsam2 sam3 yoloe seggpt}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-0}"
CONTAINER="${SNOW_CONTAINER_IMAGE:-/shared/workspace/lkm/juan.osorio/container/snow_detector_env.sqfs}"

DRIVER="${RELEASE_DIR}/tools/fill_teacher_gaps.py"
HF_TOKEN_FILE="/shared/home/juan.osorio/.hf_token"

[ -f "${CONTAINER}" ] || { echo "ERROR: container not found at ${CONTAINER}"; exit 1; }
[ -f "${DRIVER}" ] || {
    echo "ERROR: ${DRIVER} missing."
    echo "       Push the release first (tools/fill_teacher_gaps.py must exist)."
    exit 1
}

LIMIT_FLAG=""
[ "${LIMIT}" -gt 0 ] 2>/dev/null && LIMIT_FLAG="--limit ${LIMIT}"

DATA_DIR_FLAGS=""
for d in ${DATA_DIRS}; do
    DATA_DIR_FLAGS="${DATA_DIR_FLAGS} --data-dir ${d}"
done

mkdir -p /shared/home/juan.osorio/output

# Build the inner script with the values already substituted, so the container
# step is a plain `bash <file>` rather than a nested quoting puzzle. Inner-loop
# variables are escaped (\${T}) so the outer shell leaves them alone.
INNER="${MASTER_DIR}/tools/gapfill_inner.sh"
cat > "${INNER}" <<INNER_EOF
#!/bin/bash
set -u
cd "${RELEASE_DIR}"
export PYTHONPATH="${RELEASE_DIR}"

# Hugging Face cache + token. The container runs as root, so its HOME is /root
# and it cannot see the token or the model cache that the login node's
# huggingface-cli wrote under the user's home - which is why a gated repo like
# facebook/sam3 answered 401. Point HF at the mounted home cache instead, and
# take the token from .hf_token if present, else from the hub token file.
export HF_HOME="\${HF_CACHE:-${MASTER_DIR}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="\${HF_HOME}/hub"
export TRANSFORMERS_CACHE="\${HF_HOME}/hub"
TOKEN_FILE=""
# Candidates in order: <master>/.hf_token, the home dir's .hf_token (where this
# cluster actually keeps it), then the hub token file huggingface-cli writes.
for cand in "${MASTER_DIR}/.hf_token" "${HOME_ROOT}/.hf_token" "\${HF_HOME}/token"; do
    if [ -z "\${TOKEN_FILE}" ] && [ -s "\${cand}" ]; then TOKEN_FILE="\${cand}"; fi
done
if [ -n "\${TOKEN_FILE}" ]; then
    export HF_TOKEN="\$(tr -d '[:space:]' < "\${TOKEN_FILE}")"
    echo "  HF token  : loaded from \${TOKEN_FILE} (\${#HF_TOKEN} chars)"
else
    echo "  HF token  : NOT FOUND (looked in ${MASTER_DIR}/.hf_token, ${HOME_ROOT}/.hf_token, \${HF_HOME}/token)"
    echo "              gated repos such as facebook/sam3 will fail with 401"
fi
echo "  HF_HOME   : \${HF_HOME}"
echo ""
for T in ${TEACHERS}; do
    FF="${FRAMES_DIR}/teacher_gaps_\${T}.txt"
    OUT="${SEG_ROOT}/\${T}_snow_output"
    echo "============================================================"
    echo "== teacher : \${T}"
    echo "== frames  : \${FF}"
    echo "== out     : \${OUT}"
    echo "============================================================"
    if [ ! -f "\${FF}" ]; then
        echo "== ERROR: frame list \${FF} missing - push teacher_gaps_\${T}.txt"
        continue
    fi
    mkdir -p "\${OUT}"
    EXTRA=""
    if [ "\${T}" = "seggpt" ]; then EXTRA="--ann-dir ${ANN_DIR}"; fi
    if python -u "${DRIVER}" --teacher "\${T}" --frames-file "\${FF}" \
        ${DATA_DIR_FLAGS} --out-dir "\${OUT}" --device "${DEVICE}" \
        ${LIMIT_FLAG} \${EXTRA}; then
        echo "== \${T}: finished"
    else
        echo "== \${T}: FAILED - continuing with the next teacher"
    fi
    echo ""
done
echo "all teachers attempted"
INNER_EOF

echo "============================================================"
echo "TEACHER GAP FILL (FRIDA)"
echo "============================================================"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Node:         $(hostname)"
echo "  Teachers:     ${TEACHERS}"
echo "  Data dirs:    ${DATA_DIRS}"
echo "  Frames dir:   ${FRAMES_DIR}"
echo "  Seg root:     ${SEG_ROOT}"
echo "  Release code: ${RELEASE_DIR}"
echo "  Limit:        ${LIMIT} (0=all missing)"
echo "  Container:    ${CONTAINER}"
echo "============================================================"
echo ""

srun \
  --container-image=${CONTAINER} \
  --container-mounts=/shared:/shared \
  --container-workdir=${MASTER_DIR} \
  bash -c "
    export HF_TOKEN='$(cat \"${HF_TOKEN_FILE}\" 2>/dev/null | tr -d '[:space:]')'
    echo \"Started at \$(date)\"
    nvidia-smi
    echo ''
    bash '${INNER}'
    echo ''
    echo \"Finished at \$(date)\"
  "

echo ""
echo "Job finished!"
echo "New masks land in ${SEG_ROOT}/<teacher>_snow_output/<camera>/<stem>_snow.png"
echo "Then re-run: python ${RELEASE_DIR}/tools/verify_fusion_inputs.py"
