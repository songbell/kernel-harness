#!/usr/bin/env bash
# run_ckh.sh -- start the ckh MCP HTTP server on the GPU box. Runs from the install dir
# (which contains src/ and kernels/). Used by ckh.service (systemd) or directly via nohup.
# No secrets are stored here; the token arrives via the EnvironmentFile below.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Works whether or not `pip install -e .` succeeded (offline hosts): the package and the
# kernels/ descriptors are both importable straight from the install dir.
export PYTHONPATH="$HERE/src:$HERE${PYTHONPATH:+:$PYTHONPATH}"
export CKH_HTTP_HOST="${CKH_HTTP_HOST:-0.0.0.0}"
export CKH_HTTP_PORT="${CKH_HTTP_PORT:-8791}"

# MCP_AUTH_TOKEN is written by deploy_ckh.sh as a chmod-600 file; absent on a first run.
set -a; [ -f "$HERE/ckh.env" ] && . "$HERE/ckh.env"; set +a

PY="$HERE/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

exec "$PY" -m ckh.mcp --http
