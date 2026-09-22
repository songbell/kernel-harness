# `deploy/` — serve the harness to a model over MCP

Turns `ckh` into an MCP server so GitHub Copilot (VS Code Chat or the Copilot CLI) can call
`doctor` / `bench` / `equiv` / `round` / `snapshot` / `ledger` directly, instead of a human
relaying CLI output into a chat. All values come from the repo-root `.env`
([`../.env.example`](../.env.example)) — **no hostnames, users, ports or passwords are
hardcoded**.

| File | Role |
|------|------|
| `setup_local_mcp.py` | canonical one-run local setup with platform auto-detection: `.venv`, editable install, `platform.toml`, `.mcp.json`, `.vscode/mcp.json`, handshake |
| `setup_local_mcp.sh` | Unix-like wrapper for `setup_local_mcp.py` |
| `setup_local_mcp.ps1` | Windows wrapper for `setup_local_mcp.py` |
| `deploy_ckh.sh` | remote Linux deployment: rsync → remote venv → systemd/nohup → health check |
| `run_ckh.sh` | the runner on the GPU box (`python -m ckh.mcp --http`), env-driven |
| `ckh.service` | systemd user unit template (`@INSTALL_DIR@` / `@PORT@` substituted at deploy) |
| `configure_mcp_client.sh` | render `.vscode/mcp.json` from the committed template + `.env` |
| `_env.sh` | the `.env` parsing rules, stated once |

## Local setup

Use one canonical local bootstrap on both Windows and Unix-like systems:

```bash
python deploy/setup_local_mcp.py
```

Platform-native wrappers are also available:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\setup_local_mcp.ps1
```

```bash
bash deploy/setup_local_mcp.sh
```

The script auto-detects the platform and configures the local stdio MCP lane without asking
the user to choose an OS-specific path.

## Remote lane

Note: the remote HTTP deployment path is still under development. Prefer the local stdio lane
unless you specifically need a separate Linux GPU host and are prepared to validate that
environment yourself.

The server must run **on the box that has the GPU** — every tool compiles and runs a CM
kernel, so there is no "deploy it to a server and point it at a GPU".

```bash
cp .env.example .env
bash deploy/deploy_ckh.sh            # CKH_REMOTE_HOST set -> ship to that box, serve HTTP
```

Re-running is the update path: it re-syncs, restarts the unit, and re-checks health.

## Two things that are not like the Librarian deployment

**The port is a remote-execution primitive.** A knowledge-base server serves rows; this one
compiles and runs code. So the bind defaults to `127.0.0.1`, and the remote lane **refuses to
deploy** without `MCP_AUTH_TOKEN` (override with `CKH_ALLOW_UNAUTHENTICATED=1` only if the
port is genuinely unreachable off-host).

**`platform.toml` is never synced.** It names paths on the machine it describes, so shipping
this box's copy would point the remote harness at directories that do not exist. The deploy
seeds it from `platform.example.toml` on the target and tells you to edit it there.

**Secrets:** the SSH password is read from `.env`, a gitignored `.ckh_deploy.pass`, or
`SSHPASS` — never a CLI argument, never printed. `.vscode/mcp.json` is generated and
gitignored; only the `.template` files are committed.

## After deploying

Run the **`doctor`** tool first. It reports whether `clops` resolves and whether other GPU
work is running — on this rig, a competing process once produced a physically impossible
result ordering and invalidated a whole batch. `bench`, `equiv` and `round` refuse to run
while it is present, rather than returning numbers that look fine.
