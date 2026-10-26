# Mirrors scripts/setup_environment.sbatch — the exact environment this
# project was developed and trained against on the the cluster cluster (Slurm +
# Pyxis/enroot containers). SAM3 requires Python >=3.12, PyTorch >=2.7,
# CUDA >=12.6; this image satisfies that with room to spare.
#
# Build (full, matches the cluster exactly):
#   docker build -t snow-detector .
#
# Build (slim — skip SAM3, SAM2, and the OpenMMLab teacher-generation
# stack, i.e. skip everything under segmentation/'s zero-shot teachers +
# the not-reported anomaly-detection eval type; keeps ground/snow/QA/
# temporal-smoother student training, inference, and eval):
#   docker build --build-arg INSTALL_SAM3=false --build-arg INSTALL_TEACHERS=false -t snow-detector:slim .
#
# Run (GPU):
#   docker run --gpus all -it --rm -v "$(pwd)/data:/workspace/data" snow-detector
#
# Then inside the container, authenticate for SAM3/SAM2 checkpoint downloads:
#   hf auth login
#
# Multi-stage: compiles against the CUDA *devel* image (mmcv needs nvcc to
# build its CUDA ops), then copies only the installed venv into a *runtime*
# base — the compiler toolchain and headers never ship in the final image.
# Uses uv instead of pip (drop-in compatible, much faster resolution/
# installs) — see https://github.com/astral-sh/uv.

ARG INSTALL_SAM3=true
ARG INSTALL_TEACHERS=true

# ── Builder ──────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04 AS builder
ARG INSTALL_SAM3
ARG INSTALL_TEACHERS
ENV DEBIAN_FRONTEND=noninteractive
# uv's default 30s HTTP timeout is too short for some of the larger
# nvidia-*-cu12 wheels (100-200MB+) on anything less than a very fast/
# stable connection — bump it so a slow link doesn't fail the whole build.
ENV UV_HTTP_TIMEOUT=300

# uv (Rust-based pip replacement — parallel resolution/downloads, one
# static binary, no bootstrap install needed) instead of pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev git libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python3.12 -c "import sys; assert sys.version_info >= (3, 12), 'Python 3.12+ required for SAM3'"

RUN uv venv --python 3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

# ── PyTorch 2.7.0 / CUDA 12.6 (SAM3 requirement, installed separately from
#    requirements.txt exactly as on the cluster). torchvision/torchaudio
#    pinned to their known-compatible releases for torch==2.7.0 — leaving
#    them unpinned let the resolver pick a torchaudio build with a
#    mismatched ABI (`undefined symbol: torch_library_impl` the first
#    time anything imported it — transformers pulls it in transitively
#    via audio_utils.py even for pure vision models like SegFormer,
#    nothing in this repo uses torchaudio directly). ────────────────────────
RUN uv pip install --no-cache \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && python -c "import torch; assert torch.__version__.startswith('2.7'), 'PyTorch 2.7+ required'" \
    && python -c "import torch; assert torch.version.cuda >= '12.6', 'CUDA 12.6+ required'" \
    && python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# ── Repo dependencies. requirements.txt itself is never modified — the
#    slim path filters it at build time, so there's one authoritative file
#    for both Docker and manual installs.
#
#    mmcv's legacy setup.py needs pkg_resources at build time but doesn't
#    declare it as a build dependency — pip's looser build isolation
#    happens to work anyway, uv's stricter isolation doesn't. Pre-install
#    setuptools (provides pkg_resources) and build without isolation so
#    mmcv sees it. ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN if [ "$INSTALL_TEACHERS" = "true" ]; then \
        uv pip install --no-cache setuptools wheel \
        && uv pip install --no-cache --no-build-isolation -r requirements.txt; \
    else \
        grep -vE '^(scikit-image|mmengine|mmcv|mmsegmentation|plotly|streamlit|anomalib)([<>=!].*)?$' \
            requirements.txt > requirements-slim.txt \
        && uv pip install --no-cache -r requirements-slim.txt; \
    fi

# ── Florence-2 requires transformers <5 ─────────────────────────────────────
# Verified: 4.57.6 loads Florence-2 and produces a mask; from 5.0.0 it cannot
# load at all. Florence-2's remote vision tower calls .item() while
# transformers builds the model on the meta device, and 5.x removed the
# low_cpu_mem_usage opt-out, so no flag or shim recovers it (the
# forced_bos_token_id shim in segmentation/snow_teachers.py handles the
# separate 4.58+ config break, but nothing handles this one).
# requirements.txt keeps >=4.56,<5.0 as the floor because DINOv3 needs >=4.56
# and the range is shared with other packages; this pin is the version the
# image is actually built and tested at.
RUN uv pip install --no-cache "transformers==4.57.6" \
    && python -c "import transformers; assert transformers.__version__ == '4.57.6', transformers.__version__; print('PASS transformers:', transformers.__version__)"

# ── SAM3 (zero-shot ground/snow teacher; segmentation/*.py's `sam3`
#    subcommands depend on it). Always creates /workspace/sam3 (even empty)
#    so the runtime stage's COPY below never fails. ─────────────────────────
RUN if [ "$INSTALL_SAM3" = "true" ]; then \
        git clone --depth 1 https://github.com/facebookresearch/sam3.git /workspace/sam3 \
        && cd /workspace/sam3 \
        && uv pip install --no-cache -e . \
        && uv pip install --no-cache -e ".[notebooks]"; \
    else \
        mkdir -p /workspace/sam3; \
    fi

# ── SAM2 (GroundedSAM2 snow teacher's mask-refinement step;
#    segmentation/snow_teachers.py's `groundedsam2` subcommand depends on
#    it). Like SAM3, no clean PyPI release — git clone + editable install.
#    Gated on INSTALL_TEACHERS (it's part of the same zero-shot
#    teacher-generation stack the slim build skips), not INSTALL_SAM3 —
#    it's an unrelated model despite the similar name. Cloned to /opt (NOT
#    /workspace) deliberately: sam2's own build_sam.py refuses to import if
#    the cwd contains a "sam2" subdirectory (it assumes it's shadowing the
#    installed package — a false positive here since the editable install
#    resolves via absolute path, but the check fires regardless), and every
#    script in this repo is run from /workspace. Always creates the
#    directory (even empty) so the runtime COPY below never fails. ────────
RUN if [ "$INSTALL_TEACHERS" = "true" ]; then \
        git clone --depth 1 https://github.com/facebookresearch/sam2.git /opt/sam2 \
        && cd /opt/sam2 \
        && uv pip install --no-cache -e . ; \
    else \
        mkdir -p /opt/sam2; \
    fi

# ── Verification (fails the build if anything's actually broken, same
#    core checks setup_environment.sbatch runs) ─────────────────────────────
RUN python -c "import torch; print(f'PASS PyTorch: {torch.__version__}')" \
    && python -c "import torch; print(f'PASS CUDA version: {torch.version.cuda}')" \
    && python -c "import cv2; print(f'PASS OpenCV: {cv2.__version__}')" \
    && python -c "import numpy; print(f'PASS NumPy: {numpy.__version__}')" \
    && python -c "import pandas; print(f'PASS Pandas: {pandas.__version__}')" \
    && python -c "import lightning; print(f'PASS Lightning: {lightning.__version__}')"
RUN if [ "$INSTALL_SAM3" = "true" ]; then \
        python -c "from sam3.model_builder import build_sam3_image_model; print('PASS SAM3 installed successfully')"; \
    fi
RUN if [ "$INSTALL_TEACHERS" = "true" ]; then \
        python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; print('PASS SAM2 installed successfully')"; \
    fi

# ── Runtime ──────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY --from=builder /opt/sam2 /opt/sam2

WORKDIR /workspace
COPY --from=builder /workspace/sam3 /workspace/sam3
COPY . /workspace

CMD ["/bin/bash"]
