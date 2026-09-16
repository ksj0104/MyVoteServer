#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$PROJECT_DIR/.venv-server/bin/python" ]]; then
  printf '먼저 bash scripts/bootstrap.command 를 실행하세요.\n' >&2
  exit 1
fi
exec "$PROJECT_DIR/.venv-server/bin/python" "$PROJECT_DIR/main.py" start "$@"
