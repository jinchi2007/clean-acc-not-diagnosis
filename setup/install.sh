#!/usr/bin/env bash
# install.sh - rebuild the whole workspace-local DL environment from zero.
#
#   bash setup/install.sh
#
# Downloads a standalone uv binary into tools/bin, installs a private CPython
# 3.12 into tools/python, creates .venv, installs all packages (Tsinghua mirror
# first, official PyPI as fallback), registers the workspace-local Jupyter
# kernel and finally runs setup/verify_env.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/setup/env.sh"

echo "== [1/6] uv binary =="
mkdir -p "$ROOT/tools/bin" "$ROOT/tools/downloads"
if [ ! -x "$ROOT/tools/bin/uv" ]; then
  curl -fsSL --retry 3 -o "$ROOT/tools/downloads/uv-aarch64-apple-darwin.tar.gz" \
    https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz
  tar xzf "$ROOT/tools/downloads/uv-aarch64-apple-darwin.tar.gz" -C "$ROOT/tools/downloads"
  install -m 755 "$ROOT/tools/downloads/uv-aarch64-apple-darwin/uv" "$ROOT/tools/bin/uv"
fi
"$ROOT/tools/bin/uv" --version

echo "== [2/6] python 3.12 (workspace-local) =="
"$ROOT/tools/bin/uv" python install 3.12 --no-bin
"$ROOT/tools/bin/uv" python list --only-installed

echo "== [3/6] venv =="
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$ROOT/tools/bin/uv" venv --python 3.12 "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -V

echo "== [4/6] packages =="
PKGS="torch torchvision numpy pandas scipy scikit-learn matplotlib seaborn jupyterlab ipykernel tqdm pyyaml timm einops torchinfo captum ptflops openpyxl umap-learn"
# shellcheck disable=SC2086
if ! "$ROOT/tools/bin/uv" pip install --python "$ROOT/.venv/bin/python" \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple $PKGS; then
  echo "Tsinghua mirror failed -> retrying with official PyPI" >&2
  "$ROOT/tools/bin/uv" pip install --python "$ROOT/.venv/bin/python" \
    --index-url https://pypi.org/simple $PKGS
fi

echo "== [5/6] jupyter kernel (workspace-local) =="
bash "$ROOT/setup/register_kernel.sh"

echo "== [6/6] verification =="
"$ROOT/.venv/bin/python" "$ROOT/setup/verify_env.py"
