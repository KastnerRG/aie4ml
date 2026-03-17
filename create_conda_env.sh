#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH" >&2
  exit 1
fi

# Keep conda's pip phase from resolving packages out of ~/.local during env creation.
PYTHONNOUSERSITE=1 conda env create "$@" -f "$repo_dir/environment.aie4ml.yml"
