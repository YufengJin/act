#!/usr/bin/env bash
# Entrypoint for act container.
# - Detects the act repo under /workspace (direct or nested mount).
# - Installs act + detr editable from the mounted workspace (uv workspace).
set -e

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"

cd /workspace

# ---------------------------------------------------------
# Project root detection
#   direct mount:  /workspace/pyproject.toml (act repo root)
#   nested mount:  /workspace/act/pyproject.toml
# ---------------------------------------------------------
ACT_ROOT=""
if [ -f /workspace/pyproject.toml ] && [ -d /workspace/detr ]; then
  ACT_ROOT="/workspace"
elif [ -f /workspace/act/pyproject.toml ] && [ -d /workspace/act/detr ]; then
  ACT_ROOT="/workspace/act"
fi

if [ -n "${ACT_ROOT}" ]; then
  echo ">> Installing act + detr (editable, uv workspace) from ${ACT_ROOT} ..."
  # --inexact: do not purge pre-installed dependency layer from the image.
  # Without it, `uv sync` inside the mount would uninstall the pre-synced
  # deps because they are not present in any .lock on the host venv path.
  (cd "${ACT_ROOT}" && uv sync --frozen --inexact)
else
  echo "WARN: could not locate act repo under /workspace; skipping editable install."
fi

# ---------------------------------------------------------
# Exec forwarded command (venv is already on PATH via Dockerfile ENV).
# ---------------------------------------------------------
if [ $# -eq 0 ]; then
  exec bash
else
  exec "$@"
fi
