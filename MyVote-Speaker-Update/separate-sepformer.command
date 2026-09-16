#!/bin/bash
set -euo pipefail
UPDATE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash separate-sepformer.command /path/to/MyVote-Mac-Demo --input clip.wav --model-dir /path/to/sepformer --output-dir result'
  exit 2
fi
DEMO_DIR="$1"
shift
PYTHON_BIN="$DEMO_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo 'Existing Mac demo Python was not found.' >&2
  exit 1
fi
exec "$PYTHON_BIN" -I "$UPDATE_DIR/scripts/mac_update_sepformer.py" "$@"
