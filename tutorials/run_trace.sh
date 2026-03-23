#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${1:-proj_aie_tuned}"
TARGET="${2:-trace-all}"

source /home/z.ma/tools/Xilinx/2025.2/Vitis/settings64.sh

# Keep the active conda env's tools and shared libraries ahead of system paths.
# This is needed on this server for helpers like readelf, which may require
# env-provided shared libraries such as libdebuginfod.so.1.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

cd "$SCRIPT_DIR"
make -f Makefile.trace "PROJ=${PROJ}" "${TARGET}"
