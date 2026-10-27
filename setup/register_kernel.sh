#!/usr/bin/env bash
# register_kernel.sh - install the ipykernel spec inside the workspace.
#
# The spec is written to .jupyter/kernels/dsh-pytorch/kernel.json, and
# JUPYTER_PATH is set to <repo>/.jupyter, so "jupyter lab" (started from a
# shell that sourced setup/env.sh) picks it up without touching ~/Library.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
KERNEL_DIR="$ROOT/.jupyter/kernels/dsh-pytorch"
PYBIN="$ROOT/.venv/bin/python"

[ -x "$PYBIN" ] || { echo "error: $PYBIN missing - run setup/install.sh first" >&2; exit 1; }

mkdir -p "$KERNEL_DIR"
cat > "$KERNEL_DIR/kernel.json" <<JSON
{
  "argv": ["$PYBIN", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "display_name": "Python 3.12 (dsh MPS)",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
JSON

echo "kernel spec written: $KERNEL_DIR/kernel.json"
# shellcheck disable=SC1091
source "$ROOT/setup/env.sh"
"$ROOT/.venv/bin/jupyter" kernelspec list
