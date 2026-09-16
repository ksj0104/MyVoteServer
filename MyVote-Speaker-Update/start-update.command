#!/bin/bash
set -euo pipefail
UPDATE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo '사용법: bash start-update.command "$HOME/Downloads/MyVote-Mac-Demo"' >&2
  exit 1
fi
DEMO_DIR="$1"
shift
if [[ ! -x "$DEMO_DIR/.venv/bin/python" ]]; then
  echo '기존 Mac 데모 폴더의 .venv/bin/python을 찾을 수 없습니다.' >&2
  exit 1
fi
exec "$DEMO_DIR/.venv/bin/python" -I "$UPDATE_DIR/scripts/mac_update_start.py" --demo-dir "$DEMO_DIR" "$@"
