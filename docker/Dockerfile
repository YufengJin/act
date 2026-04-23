# =========================================================
# act Dockerfile (uv-based, GPU-ready)
# MuJoCo + dm_control + WebSocket policy client (no Vulkan, no CUDA-ext build).
# Source is mounted at runtime to /workspace; act + detr are installed
# editable by the entrypoint.
# =========================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# ---------------------------------------------------------
# System dependencies (MuJoCo/dm_control rendering + utils)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libegl1 \
    libosmesa6 \
    ffmpeg \
    lsof \
    psmisc \
    tmux \
    vim \
    x11-apps \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install uv (copy static binary from official image)
# ---------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ---------------------------------------------------------
# Create Python 3.9 venv managed by uv
# ---------------------------------------------------------
RUN uv venv /opt/venv --python 3.9

# ---------------------------------------------------------
# Pre-install project dependencies (leverages Docker layer cache).
# Only pyproject.toml + uv.lock (root + detr workspace member) are
# copied so this layer is reused unless dependencies change.
# ---------------------------------------------------------
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
COPY detr/pyproject.toml ./detr/pyproject.toml
RUN uv sync --frozen --no-install-project --no-install-workspace

# ---------------------------------------------------------
# Entrypoint (act + detr are installed editable at container start
# from the mounted workspace)
# ---------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
