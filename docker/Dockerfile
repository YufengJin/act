# =============================================================================
# act Docker image (uv-managed Python environment)
# Build from repo root:  docker build -f docker/Dockerfile .
# =============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# -----------------------------------------------------------------------------
# 1. System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl wget tar procps bzip2 ca-certificates git build-essential cmake unzip tmux \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    libegl1 libosmesa6 libgl1 libglib2.0-0 ffmpeg lsof psmisc tmux vim \
    && rm -rf /var/lib/apt/lists/*



# 
# Uncomment if project needs MuJoCo / robosuite / EGL rendering:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libx11-6 libxext6 libxrender1 libsm6 libegl1 libosmesa6 \
#     && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. uv (fast Python package manager)
# -----------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# Alternative (faster, always latest):
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# -----------------------------------------------------------------------------
# 3. Python venv via uv
# -----------------------------------------------------------------------------
RUN uv venv --python 3.9 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"



# -----------------------------------------------------------------------------
# 4. Non-Python / make-based third-party dependencies → /opt
# -----------------------------------------------------------------------------
# 
# Example:
# WORKDIR /opt/third_party
# RUN git clone https://github.com/<org>/<repo>.git && \
#     cd <repo> && git checkout <commit> && \
#     mkdir build && cd build && cmake .. && make -j$(nproc) && make install
#
# If the dep has Python bindings:
# RUN uv pip install -e /opt/third_party/<repo>

# -----------------------------------------------------------------------------
# 5. Install block — driven by InstallationPlan (<repo>/.nautilus/install_plan.json)
#    or, when no plan exists, by the legacy probe-only Option A/C heuristic.
#
# NOTE: Every `uv sync` / `uv pip install` line carries `--index-strategy
# unsafe-best-match`. uv defaults to "first-index-wins": once any extra index
# (e.g. the PyTorch CUDA wheel index) mentions a package name, uv treats that
# index as the SOLE source and refuses to consult PyPI for a better version.
# This breaks pins like `packaging==23.0` because the PyTorch index ships
# partial mirrors at different versions. `unsafe-best-match` tells uv to
# consider all indexes and pick the best match — what users intuitively
# expect. Harmless when no extra-index is configured. See:
# https://docs.astral.sh/uv/concepts/indexes/
# -----------------------------------------------------------------------------
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-install-workspace

# Re-install torch explicitly to ensure version.py and all wheel files are present.
# uv may omit small generated files (e.g. torch/version.py) from large wheels on some
# uv versions; pip install --no-deps is the reliable fallback for completeness.
RUN uv pip install --index-url https://download.pytorch.org/whl/cu118 \
    --no-deps \
    torch==2.0.0+cu118 torchvision==0.15.0+cu118

# 
# Slot for downstream sub-skills (benchmark / policy generators) to inject
# extra pip installs via extend_base.py. Leave the placeholder line above —
# it is the insertion anchor.

# -----------------------------------------------------------------------------
# 6. Extension point for sub-skills (benchmark / policy / robot env-generator)
# -----------------------------------------------------------------------------
# Sub-skills insert RUN lines ABOVE this marker via their extend_base.py script.
# Do NOT remove the marker line — it is how downstream tooling locates the insertion point.
# <<<EXTENSION_INSERT_ABOVE>>>

# -----------------------------------------------------------------------------
# 7. Workspace (project source mounted at runtime, not copied)
# -----------------------------------------------------------------------------
WORKDIR /workspace/act
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
