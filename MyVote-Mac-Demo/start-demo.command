#!/bin/bash
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$KIT_DIR/.venv/bin/python" ]]; then
  PYTHON="$KIT_DIR/.venv/bin/python"
else
  PYTHON=""
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$(command -v "$candidate")"
      break
    fi
  done
  if [[ -z "$PYTHON" ]]; then
    echo "Python 3.11 이상을 먼저 설치한 뒤 다시 실행하세요." >&2
    exit 1
  fi
fi
exec "$PYTHON" -I "$KIT_DIR/scripts/mac_demo_setup.py" --host 192.168.219.103 "$@"
