#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ENV="$PROJECT_DIR/.venv-server"

compatible_python() {
  "$1" -c 'import platform, sys; sys.exit(not (sys.version_info[:2] in ((3, 12), (3, 13)) and platform.machine() == "arm64"))' >/dev/null 2>&1
}

if [[ -e "$SERVER_ENV" ]]; then
  if [[ -x "$SERVER_ENV/bin/python" ]] && compatible_python "$SERVER_ENV/bin/python"; then
    "$SERVER_ENV/bin/python" --version
    printf '서버 준비용 가상환경 사용: %s\n' "$SERVER_ENV"
    exit 0
  fi
  printf '기존 %s가 호환되지 않습니다. 자동 덮어쓰기를 하지 않습니다.\n' "$SERVER_ENV" >&2
  exit 1
fi

for candidate in python3.12 python3.13 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
  /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13; do
  if command -v "$candidate" >/dev/null 2>&1 && compatible_python "$candidate"; then
    "$candidate" -m venv "$SERVER_ENV"
    printf '서버 준비용 가상환경 생성: %s\n' "$SERVER_ENV"
    exit 0
  fi
done
printf 'native arm64 Python 3.12 또는 3.13을 먼저 설치하세요.\n' >&2
exit 1
