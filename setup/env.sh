#!/usr/bin/env bash
# Source this file to activate the workspace-local DL environment:
#   source setup/env.sh
# Everything (interpreter, caches, jupyter data) lives inside this repo, so
# nothing is written to ~/Library, /usr/local or ~/.cache.

DSH_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export DSH_ENV_ROOT

# --- uv (standalone binary, workspace-local) ---
export UV_PYTHON_INSTALL_DIR="$DSH_ENV_ROOT/tools/python"
export UV_CACHE_DIR="$DSH_ENV_ROOT/tools/uv-cache"
export UV_TOOL_DIR="$DSH_ENV_ROOT/tools/uv-tools"
export UV_PYTHON_BIN_DIR="$DSH_ENV_ROOT/tools/bin"   # never touch ~/.local/bin
export PATH="$DSH_ENV_ROOT/tools/bin:$DSH_ENV_ROOT/.venv/bin:$PATH"

# --- model / dataset caches (keep downloads inside the workspace) ---
export HF_HOME="$DSH_ENV_ROOT/tools/hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$DSH_ENV_ROOT/tools/torch-cache"
export MPLCONFIGDIR="$DSH_ENV_ROOT/tools/mpl-cache"

# --- jupyter: kernel specs live in .jupyter/kernels, all state is local ---
export JUPYTER_PATH="$DSH_ENV_ROOT/.jupyter"
export JUPYTER_DATA_DIR="$DSH_ENV_ROOT/tools/jupyter-data"
export JUPYTER_CONFIG_DIR="$DSH_ENV_ROOT/tools/jupyter-config"
export JUPYTER_RUNTIME_DIR="$DSH_ENV_ROOT/tools/jupyter-runtime"
export IPYTHONDIR="$DSH_ENV_ROOT/tools/ipython"

if [ -x "$DSH_ENV_ROOT/.venv/bin/python" ]; then
  # shellcheck disable=SC1091
  source "$DSH_ENV_ROOT/.venv/bin/activate"
fi

echo "[env] root=$DSH_ENV_ROOT"
echo "[env] python=$("$DSH_ENV_ROOT/.venv/bin/python" -V 2>/dev/null || echo missing)"
