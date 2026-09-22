#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if command -v python3 >/dev/null 2>&1; then
    exec python3 "$repo_root/deploy/setup_local_mcp.py"
elif command -v python >/dev/null 2>&1; then
    exec python "$repo_root/deploy/setup_local_mcp.py"
else
    echo "setup_local_mcp.sh failed: no python interpreter found" >&2
    exit 127
fi