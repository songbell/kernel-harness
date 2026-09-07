#!/usr/bin/env bash
# configure_mcp_client.sh -- render .vscode/mcp.json from a committed template + .env, so the
# server location has ONE source of truth. deploy_ckh.sh calls this at the end; run it by hand
# after changing CKH_REMOTE_HOST / CKH_HTTP_PORT / CKH_MCP_URL.
#
# Lane is inferred, not asked for: CKH_REMOTE_HOST (or CKH_MCP_URL) set -> http client;
# otherwise -> local stdio client. Only the .template files are committed; mcp.json is not.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/deploy/_env.sh"

# VS Code only reads .vscode/mcp.json from a WORKSPACE FOLDER root. When the harness is
# opened as a sub-folder of a bigger workspace (e.g. a repos umbrella dir), its own
# .vscode/ is invisible, so allow the config to be written where the workspace actually is.
CONFIG_DIR="$(env_get CKH_MCP_CONFIG_DIR)"; CONFIG_DIR="${CONFIG_DIR:-$REPO_ROOT}"
OUT="$CONFIG_DIR/.vscode/mcp.json"
mkdir -p "$CONFIG_DIR/.vscode"

HOST="$(env_get CKH_REMOTE_HOST)"
PORT="$(env_get CKH_HTTP_PORT)";     PORT="${PORT:-8791}"
SCHEME="$(env_get CKH_HTTP_SCHEME)"; SCHEME="${SCHEME:-http}"
URL="$(env_get CKH_MCP_URL)"

if [[ -n "$URL" || -n "$HOST" ]]; then
  URL="${URL:-$SCHEME://$HOST:$PORT/mcp}"
  sed "s#@CKH_MCP_URL@#$URL#g" "$REPO_ROOT/.vscode/mcp.json.remote.template" > "$OUT"
  echo "Wrote $OUT  (remote lane -> $URL)"
else
  # Prefer the repo venv so the client does not depend on whatever python is on PATH.
  PY="$REPO_ROOT/.venv/bin/python"; [[ -x "$PY" ]] || PY="python3"
  sed -e "s#@CKH_PYTHON@#$PY#g" -e "s#@CKH_REPO_ROOT@#$REPO_ROOT#g" \
    "$REPO_ROOT/.vscode/mcp.json.local.template" > "$OUT"
  echo "Wrote $OUT  (local lane -> stdio, $PY -m ckh.mcp)"
fi
