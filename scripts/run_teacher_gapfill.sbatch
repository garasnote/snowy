#!/bin/bash
#SBATCH --job-name=teacher-gaps
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
# GPU model matters here. The container's PyTorch is built for sm_50..sm_90, while
# the B200 (ixb1-3) and B300 (ixb5/6/8) nodes are sm_100+, where every kernel call
# fails with "no kernel image is available for execution on the device". The A100
# (axa) and H100 (ixh) nodes are sm_80/sm_90 and work. Exclude by node name rather
# than by gres type: naming a type (e.g. gpu:L4:1) is what produced "Invalid
# generic resource (gres) specification" on this cluster.
#SBATCH --exclude=ixb1,ixb2,ixb3,ixb4,ixb5,ixb6,ixb7,ixb8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=teacher_gaps_%j.out

CLUSTER_USER="${CLUSTER_USER:-$(whoami)}"

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

MASTER_DIR="${MASTER_DIR:-/shared/home/${CLUSTER_USER}/master}"
HOME_ROOT="${HOME_ROOT:-/shared/home/${CLUSTER_USER}}"
RELEASE_DIR="${RELEASE_DIR:-${MASTER_DIR}/release_check}"
SEG_ROOT="${SEG_ROOT:-${MASTER_DIR}/preprocessing/segmentation}"
FRAMES_DIR="${FRAMES_DIR:-${MASTER_DIR}/tools}"
ANN_DIR="${ANN_DIR:-${RELEASE_DIR}/data/annotations}"
DATA_DIRS="${DATA_DIRS:-${MASTER_DIR}/data ${MASTER_DIR}/night_images}"
TEACHERS="${TEACHERS:-clipseg groundedsam2 sam3 yoloe seggpt}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-0}"
# Private overlay for packages the image lacks (see the ultralytics block below).
UL_OVERLAY="${UL_OVERLAY:-${MASTER_DIR}/ultralytics_overlay}"
CONTAINER="${SNOW_CONTAINER_IMAGE:-/shared/workspace/lkm/${CLUSTER_USER}/container/snow_detector_env.sqfs}"

DRIVER="${RELEASE_DIR}/tools/fill_teacher_gaps.py"
HF_TOKEN_FILE="/shared/home/${CLUSTER_USER}/.hf_token"

[ -f "${CONTAINER}" ] || { echo "ERROR: container not found at ${CONTAINER}"; exit 1; }
[ -f "${DRIVER}" ] || {
    echo "ERROR: ${DRIVER} missing."
    echo "       Push the release first (tools/fill_teacher_gaps.py must exist)."
    exit 1
}

LIMIT_FLAG=""
[ "${LIMIT}" -gt 0 ] 2>/dev/null && LIMIT_FLAG="--limit ${LIMIT}"

# Optional extra import path (PY_EXTRA): set it when a teacher's package lives in
# a site-packages that the image does not have on its own path, e.g.
#   PY_EXTRA=/shared/home/${CLUSTER_USER}/someenv/lib/python3.12/site-packages
PY_EXTRA="${PY_EXTRA:-}"
PY_EXTRA_PART=""
[ -n "${PY_EXTRA}" ] && PY_EXTRA_PART=":${PY_EXTRA}"

DATA_DIR_FLAGS=""
for d in ${DATA_DIRS}; do
    DATA_DIR_FLAGS="${DATA_DIR_FLAGS} --data-dir ${d}"
done

mkdir -p /shared/home/${CLUSTER_USER}/output

# Build the inner script with the values already substituted, so the container
# step is a plain `bash <file>` rather than a nested quoting puzzle. Inner-loop
# variables are escaped (\${T}) so the outer shell leaves them alone.
INNER="${MASTER_DIR}/tools/gapfill_inner.sh"
cat > "${INNER}" <<INNER_EOF
#!/bin/bash
set -u
cd "${RELEASE_DIR}"
export PYTHONPATH="${RELEASE_DIR}${PY_EXTRA_PART}"

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

# ultralytics is absent from the image: yoloe was only ever run on the older
# cluster (its script still points at /d/hpc/home/jo83525/...). Installing it
# INTO the image would risk pulling a pip torch/torchvision over the container's
# CUDA build, so install --no-deps into a private overlay and put that overlay
# first on PYTHONPATH. Only ultralytics and its tiny thop dependency come from
# there; torch, torchvision and opencv stay the image's own.
UL_OVERLAY="${UL_OVERLAY}"
if [ -d "\${UL_OVERLAY}/ultralytics" ]; then
    echo "  ultralytics: overlay present at \${UL_OVERLAY}"
else
    echo "  ultralytics: installing overlay (one time) into \${UL_OVERLAY}"
    mkdir -p "\${UL_OVERLAY}"
    if command -v uv >/dev/null 2>&1; then
        uv pip install --no-cache --system --target "\${UL_OVERLAY}" --no-deps ultralytics ultralytics-thop
    elif [ -x /usr/local/bin/uv ]; then
        /usr/local/bin/uv pip install --no-cache --system --target "\${UL_OVERLAY}" --no-deps ultralytics ultralytics-thop
    elif python3 -m pip --version >/dev/null 2>&1; then
        python3 -m pip install --no-cache-dir --break-system-packages --target "\${UL_OVERLAY}" --no-deps ultralytics ultralytics-thop
    else
        echo "  ultralytics: no installer found - yoloe will fail to import"
    fi
fi
if [ -d "\${UL_OVERLAY}/ultralytics" ]; then
    echo "  ultralytics: \$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import ultralytics; print(ultralytics.__version__)' "\${UL_OVERLAY}" 2>/dev/null || echo 'overlay present, version unknown')"
fi
# set_classes(["snow"]) builds CLIP text embeddings, so ultralytics needs the
# The ultralytics text model needs the 'clip' package (its own fork) plus ftfy and
# regex. Its AutoUpdate path cannot work here: the image's python is externally
# managed, so pip refuses and the loader dies on 'import clip'. Same overlay, same
# rule: --no-deps, image untouched.
if [ ! -d "\${UL_OVERLAY}/clip" ]; then
    echo "  clip      : installing ultralytics CLIP fork into the overlay"
    if command -v git >/dev/null 2>&1; then
        CLIP_SPEC="git+https://github.com/ultralytics/CLIP.git"
    else
        echo "  clip      : no git in the image, using the source tarball instead"
        curl -sL https://github.com/ultralytics/CLIP/archive/refs/heads/main.tar.gz -o /tmp/ultralytics_clip.tar.gz \\
            && CLIP_SPEC="/tmp/ultralytics_clip.tar.gz"
    fi
    if [ -n "\${CLIP_SPEC:-}" ]; then
        if command -v uv >/dev/null 2>&1; then
            uv pip install --no-cache --system --target "\${UL_OVERLAY}" --no-deps ftfy regex "\${CLIP_SPEC}"
        elif [ -x /usr/local/bin/uv ]; then
            /usr/local/bin/uv pip install --no-cache --system --target "\${UL_OVERLAY}" --no-deps ftfy regex "\${CLIP_SPEC}"
        else
            python3 -m pip install --no-cache-dir --break-system-packages --target "\${UL_OVERLAY}" --no-deps ftfy regex "\${CLIP_SPEC}"
        fi
    fi
    if [ -d "\${UL_OVERLAY}/clip" ]; then
        echo "  clip      : overlay now provides clip"
    else
        echo "  clip      : could not be installed - yoloe will fail at set_classes"
    fi
fi
# groundedsam2 imports the 'sam2' package (facebookresearch/sam2). The image does
# not carry it (like ultralytics, it was only ever needed on the older cluster),
# and it is not in <master>, /workspace or /opt. Installing it with pip from git
# tries to build its CUDA extension and stalls, so unpack the source tree instead
# and put it on PYTHONPATH: sam2/__init__.py needs only hydra, and the compiled
# _C extension is touched in exactly one leaf function (sam2/utils/misc.py:61)
# that the image-predictor path never calls.
if [ ! -f "\${UL_OVERLAY}/sam2_src/sam2/__init__.py" ]; then
    echo "  sam2      : cloning facebookresearch/sam2 into the overlay"
    rm -rf "\${UL_OVERLAY}/sam2_src"
    git clone --depth 1 https://github.com/facebookresearch/sam2.git "\${UL_OVERLAY}/sam2_src" 2>&1 | tail -2 || true
fi
# Installed unconditionally: an idempotent ~5s cost that cannot go stale the way a
# "if antlr4 is missing" guard can (the first version of this guarded on hydra and
# silently kept the overlay short of antlr4, which is what failed job 175963).
echo "  sam2      : ensuring its config deps (hydra-core, iopath, omegaconf, antlr4, portalocker, pyyaml, packaging)"
if true; then
    if command -v uv >/dev/null 2>&1; then
        uv pip install --no-cache --system --target "\${UL_OVERLAY}" --no-deps hydra-core iopath omegaconf antlr4-python3-runtime==4.9.3 portalocker pyyaml packaging
    elif [ -x /usr/local/bin/uv ]; then
        /usr/local/bin/uv pip install --no-cache --system --target "\${UL_OVERLAY}" --no-deps hydra-core iopath omegaconf antlr4-python3-runtime==4.9.3 portalocker pyyaml packaging
    else
        python3 -m pip install --no-cache-dir --break-system-packages --target "\${UL_OVERLAY}" --no-deps hydra-core iopath omegaconf antlr4-python3-runtime==4.9.3 portalocker pyyaml packaging
    fi
fi
if [ -f "\${UL_OVERLAY}/sam2_src/sam2/__init__.py" ]; then
    echo "  sam2      : source tree ready at \${UL_OVERLAY}/sam2_src"
else
    echo "  sam2      : could not be unpacked - groundedsam2 will still fail"
fi
export PYTHONPATH="\${UL_OVERLAY}:\${UL_OVERLAY}/sam2_src:${RELEASE_DIR}${PY_EXTRA_PART}"
echo ""

# The scheduler can hand this job a GPU the container's PyTorch cannot drive: an
# NVIDIA B200 is sm_100 while the image's torch is built for up to sm_90, and then
# every teacher dies with "CUDA error: no kernel image is available for execution
# on the device". These five are small enough for CPU, so probe once and fall back
# rather than lose the whole run to a node assignment.
DEVICE="${DEVICE}"
if [ "\${DEVICE}" = "cuda" ]; then
    if python3 -c "import torch; torch.zeros(2, device='cuda')" >/dev/null 2>&1; then
        echo "  Device    : cuda (probe passed)"
    else
        echo "  Device    : cuda unusable on this node (GPU newer than the image's"
        echo "              torch build) -> falling back to cpu"
        DEVICE=cpu
    fi
else
    echo "  Device    : \${DEVICE}"
fi
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
        ${DATA_DIR_FLAGS} --out-dir "\${OUT}" --device "\${DEVICE}" \\
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
echo "TEACHER GAP FILL (cluster)"
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
