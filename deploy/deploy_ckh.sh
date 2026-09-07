#!/usr/bin/env bash
# deploy_ckh.sh -- stand up the ckh MCP server so a model (GitHub Copilot) can drive the
# harness: `doctor`, `bench`, `equiv`, `round`, `snapshot`, `ledger`.
#
# All target config comes from the repo-root .env (copy .env.example -> .env and fill it).
# NOTHING is hardcoded here -- switching machines = edit .env, re-run this.
#
#   bash deploy/deploy_ckh.sh --local     # GPU is in THIS box -> venv + stdio client config
#   bash deploy/deploy_ckh.sh             # CKH_REMOTE_HOST set -> ship to that GPU box over ssh
#
# Why the lane matters: every ckh tool compiles and runs a CM kernel, so the server has to
# live where the GPU is. There is no "deploy the harness to a server and point it at a GPU".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/deploy/_env.sh"

LANE="auto"
[[ "${1:-}" == "--local" ]] && LANE="local"
[[ "${1:-}" == "--remote" ]] && LANE="remote"

REMOTE_HOST="$(env_get CKH_REMOTE_HOST)"
REMOTE_USER="$(env_get CKH_REMOTE_USER)"
REMOTE_DIR="$(env_get CKH_REMOTE_DIR)"; REMOTE_DIR="${REMOTE_DIR:-cm-kernel-harness}"
PORT="$(env_get CKH_HTTP_PORT)";        PORT="${PORT:-8791}"
TOKEN="$(env_get MCP_AUTH_TOKEN)"
PIP_PROXY="$(env_get CKH_PIP_PROXY)"
PIP_PROXY_ARG=""; [[ -n "$PIP_PROXY" ]] && PIP_PROXY_ARG="--proxy $PIP_PROXY"

[[ "$LANE" == "auto" ]] && { [[ -n "$REMOTE_HOST" ]] && LANE="remote" || LANE="local"; }

# ---------------------------------------------------------------- local lane ----------
if [[ "$LANE" == "local" ]]; then
  echo "== local lane: the GPU in this box, served to VS Code over stdio =="

  echo "== 1/4  venv + editable install =="
  [[ -d "$REPO_ROOT/.venv" ]] || python3 -m venv "$REPO_ROOT/.venv"
  "$REPO_ROOT/.venv/bin/pip" -q install $PIP_PROXY_ARG -U pip
  "$REPO_ROOT/.venv/bin/pip" -q install $PIP_PROXY_ARG -e "$REPO_ROOT[dev]"

  echo "== 2/4  platform.toml =="
  if [[ -f "$REPO_ROOT/platform.toml" ]]; then
    echo "   present (kept as-is -- it is machine-specific and never overwritten)"
  else
    cp "$REPO_ROOT/platform.example.toml" "$REPO_ROOT/platform.toml"
    echo "   seeded from platform.example.toml -- EDIT IT before measuring anything"
  fi

  echo "== 3/4  handshake check (no GPU needed) =="
  # A broken MCP server fails silently from the model's side, so prove the handshake and the
  # tool list here rather than in a chat session. Response id 2 is the tools/list reply.
  printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize"}' \
                '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
    | "$REPO_ROOT/.venv/bin/python" -m ckh.mcp 2>/dev/null \
    | "$REPO_ROOT/.venv/bin/python" -c '
import json, sys
names = [t["name"] for l in sys.stdin if l.strip()
         for m in [json.loads(l)] if m.get("id") == 2
         for t in m["result"]["tools"]]
if not names:
    sys.exit("   FAILED: server returned no tools")
print("   ok:", ", ".join(names))
'

  echo "== 4/4  client config =="
  bash "$REPO_ROOT/deploy/configure_mcp_client.sh"

  echo ""
  echo "Deployed locally. Reload VS Code, approve the 'ckh' MCP server, then ask for"
  echo "  ckh doctor   -- it will tell you if the rig is fit to measure on."
  exit 0
fi

# --------------------------------------------------------------- remote lane ----------
echo "== remote lane: ship the harness to the GPU box and serve it over HTTP =="

[[ -n "$REMOTE_HOST" ]] || { echo "ERROR: set CKH_REMOTE_HOST in .env (the box with the GPU)"; exit 1; }
[[ -n "$REMOTE_USER" ]] || { echo "ERROR: set CKH_REMOTE_USER in .env (your ssh user / IDSID)"; exit 1; }
command -v sshpass >/dev/null || { echo "ERROR: install sshpass (sudo apt install sshpass)"; exit 1; }
command -v rsync   >/dev/null || { echo "ERROR: install rsync"; exit 1; }

# The remote lane binds 0.0.0.0, and every tool on that port executes kernel code. Refuse to
# do that silently without a bearer token.
if [[ -z "$TOKEN" && "$(env_get CKH_ALLOW_UNAUTHENTICATED)" != "1" ]]; then
  echo "ERROR: MCP_AUTH_TOKEN is empty. The remote lane listens on 0.0.0.0 and every tool"
  echo "       compiles and runs code on that box -- an open port is a remote-execution"
  echo "       primitive. Set MCP_AUTH_TOKEN in .env (e.g. \`openssl rand -hex 32\`), or set"
  echo "       CKH_ALLOW_UNAUTHENTICATED=1 if the port is genuinely unreachable off-host."
  exit 1
fi

# --- password: CKH_SSH_PASSWORD (.env/env) > ./.ckh_deploy.pass > SSHPASS ---
PASS_VALUE="$(env_get CKH_SSH_PASSWORD)"
PASS_FILE="$REPO_ROOT/.ckh_deploy.pass"
if [[ -n "$PASS_VALUE" ]]; then
  export SSHPASS="$PASS_VALUE"; SSHPASS_ARG=(-e)
elif [[ -r "$PASS_FILE" ]]; then
  SSHPASS_ARG=(-f "$PASS_FILE")
elif [[ -n "${SSHPASS:-}" ]]; then
  SSHPASS_ARG=(-e)
else
  echo "ERROR: no ssh password. Set CKH_SSH_PASSWORD in .env, fill ./.ckh_deploy.pass, or export SSHPASS"
  exit 1
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password)
SSH=(sshpass "${SSHPASS_ARG[@]}" ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST")
RSH="sshpass ${SSHPASS_ARG[*]} ssh ${SSH_OPTS[*]}"

echo "== target: $REMOTE_USER@$REMOTE_HOST  dir: $REMOTE_DIR  port: $PORT =="

if [[ "$REMOTE_DIR" == /* ]]; then
  INSTALL_DIR="$REMOTE_DIR"
else
  REMOTE_HOME="$("${SSH[@]}" 'echo "$HOME"')"
  REMOTE_DIR="${REMOTE_DIR#\~/}"
  [[ "$REMOTE_DIR" == "~" ]] && REMOTE_DIR=""
  INSTALL_DIR="${REMOTE_HOME%/}${REMOTE_DIR:+/$REMOTE_DIR}"
fi
echo "   install dir (absolute): $INSTALL_DIR"

echo "== 1/7  create $INSTALL_DIR =="
"${SSH[@]}" "mkdir -p '$INSTALL_DIR'"

echo "== 2/7  sync code (no secrets, no venv, no results, no platform.toml) =="
# platform.toml is deliberately excluded: it names paths on the *target* box, so shipping
# this machine's copy would point the remote harness at directories that do not exist.
rsync -az --delete -e "$RSH" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
  "$REPO_ROOT/src" "$REPO_ROOT/kernels" "$REPO_ROOT/deploy" \
  "$REMOTE_USER@$REMOTE_HOST:$INSTALL_DIR/"
rsync -az -e "$RSH" \
  "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/platform.example.toml" \
  "$REMOTE_USER@$REMOTE_HOST:$INSTALL_DIR/"
# Findings are append-only history worth carrying over, but never worth deleting remotely.
[[ -d "$REPO_ROOT/ledger" ]] && rsync -az -e "$RSH" "$REPO_ROOT/ledger" "$REMOTE_USER@$REMOTE_HOST:$INSTALL_DIR/"

echo "== 3/7  push auth token (EnvironmentFile, chmod 600, never in argv) =="
# Streamed over stdin, so the token never appears in a command line, process list, or log.
TOKEN_ENV=""
[[ -n "$TOKEN" ]] && TOKEN_ENV="MCP_AUTH_TOKEN=$TOKEN"$'\n'
printf '%s' "$TOKEN_ENV" \
  | "${SSH[@]}" "umask 077 && cat > '$INSTALL_DIR/ckh.env' && chmod 600 '$INSTALL_DIR/ckh.env'"
echo "   token: $([[ -n "$TOKEN" ]] && echo set || echo '<DISABLED -- port is unauthenticated>')"

echo "== 4/7  isolated python venv =="
"${SSH[@]}" "cd '$INSTALL_DIR' && python3 -m venv .venv && ./.venv/bin/pip -q install $PIP_PROXY_ARG -U pip && ./.venv/bin/pip -q install $PIP_PROXY_ARG -e ." \
  || echo "   WARNING: editable install failed (offline host / no proxy). run_ckh.sh still works via PYTHONPATH; set CKH_PIP_PROXY in .env if the host has a proxy."

echo "== 5/7  platform.toml on the target =="
"${SSH[@]}" "cd '$INSTALL_DIR' && if [ -f platform.toml ]; then echo '   present (kept)'; else cp platform.example.toml platform.toml && echo '   SEEDED -- edit $INSTALL_DIR/platform.toml on the target before measuring'; fi"

echo "== 6/7  install + (re)start the service =="
UNIT="$(sed -e "s#@INSTALL_DIR@#$INSTALL_DIR#g" -e "s#@PORT@#$PORT#g" "$REPO_ROOT/deploy/ckh.service")"
"${SSH[@]}" "chmod +x '$INSTALL_DIR/deploy/run_ckh.sh'"
if "${SSH[@]}" "systemctl --user --version" >/dev/null 2>&1; then
  # Re-deploy safe: stop the unit AND any stray non-systemd process, so the fresh start
  # picks up the newly synced code without a port clash. '[c]kh.mcp' does not match this
  # pkill's own argv.
  "${SSH[@]}" "systemctl --user stop ckh.service 2>/dev/null || true; pkill -f '[c]kh.mcp' 2>/dev/null || true; sleep 1 || true"
  printf '%s\n' "$UNIT" | "${SSH[@]}" "mkdir -p ~/.config/systemd/user && cat > ~/.config/systemd/user/ckh.service && systemctl --user daemon-reload && systemctl --user enable --now ckh.service && (loginctl enable-linger '$REMOTE_USER' || true)"
  "${SSH[@]}" "systemctl --user status ckh.service --no-pager -l | head -12 || true"
else
  echo "   systemd --user unavailable; using nohup (detached via setsid)"
  "${SSH[@]}" "cd '$INSTALL_DIR' && pkill -f '[c]kh.mcp' 2>/dev/null || true; CKH_HTTP_PORT='$PORT' setsid nohup bash deploy/run_ckh.sh > '$INSTALL_DIR/ckh.log' 2>&1 < /dev/null & sleep 2; tail -3 '$INSTALL_DIR/ckh.log' || true"
fi

echo "== 7/7  health check =="
"${SSH[@]}" "curl -s --noproxy '*' http://127.0.0.1:$PORT/healthz && echo"

bash "$REPO_ROOT/deploy/configure_mcp_client.sh" || true

echo ""
echo "Deployed. MCP clients point at:  http://$REMOTE_HOST:$PORT/mcp"
echo "If colleagues cannot reach it, open TCP $PORT on the host firewall."
echo "Then run the 'doctor' tool FIRST -- a rig that drifts invalidates every later number."
