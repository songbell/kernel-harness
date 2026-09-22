from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / ".venv"
PLATFORM_TOML = REPO_ROOT / "platform.toml"
PLATFORM_EXAMPLE = REPO_ROOT / "platform.example.toml"
WORKSPACE_MCP = REPO_ROOT / ".mcp.json"
VSCODE_DIR = REPO_ROOT / ".vscode"
VSCODE_MCP = VSCODE_DIR / "mcp.json"


def _print(step: str) -> None:
    print(step, flush=True)


def _is_windows() -> bool:
    return os.name == "nt"


def _venv_python() -> Path:
    if _is_windows():
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(command: list[str], cwd: Path | None = None, input_text: str | None = None) -> str:
    proc = subprocess.run(
        command,
        cwd=str(cwd or REPO_ROOT),
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(command)}\n{proc.stdout}{proc.stderr}"
        )
    return proc.stdout


def _bootstrap_python() -> list[str]:
    if _is_windows():
        py = shutil.which("py")
        if py:
            return [py, "-3"]
    python = shutil.which("python")
    if python:
        return [python]
    python3 = shutil.which("python3")
    if python3:
        return [python3]
    raise RuntimeError("No Python launcher found. Install Python 3.11+ and add it to PATH.")


def _write_mcp_config() -> None:
    VSCODE_DIR.mkdir(parents=True, exist_ok=True)
    if _is_windows():
        server = {
            "type": "stdio",
            "command": "${workspaceFolder}/.venv/Scripts/python.exe",
            "args": ["-m", "ckh.mcp"],
            "cwd": "${workspaceFolder}",
        }
    else:
        server = {
            "type": "stdio",
            "command": "${workspaceFolder}/deploy/ckh-mcp-vscode",
            "cwd": "${workspaceFolder}",
        }
    payload = {"servers": {"ckh": server}}
    text = json.dumps(payload, indent=2) + "\n"
    WORKSPACE_MCP.write_text(text, encoding="utf-8")
    VSCODE_MCP.write_text(text, encoding="utf-8")
    print(f"Wrote {WORKSPACE_MCP}")
    print(f"Wrote {VSCODE_MCP}")


def _handshake_check() -> None:
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "setup-local-mcp", "version": "1"},
        },
    }
    tools = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    payload = json.dumps(init) + "\n" + json.dumps(tools) + "\n"
    output = _run([str(_venv_python()), "-m", "ckh.mcp"], input_text=payload)
    tool_names: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        if message.get("id") == 2:
            tool_names = [tool["name"] for tool in message["result"]["tools"]]
    if not tool_names:
        raise RuntimeError("MCP handshake succeeded but tools/list returned no tools.")
    print("ok: " + ", ".join(tool_names))


def main() -> int:
    label = "Windows" if _is_windows() else "Unix-like"
    _print(f"== local MCP setup ({label}) ==")

    if not _venv_python().exists():
        _print("== 1/5 create .venv ==")
        _run(_bootstrap_python() + ["-m", "venv", str(VENV_DIR)])
    else:
        _print("== 1/5 create .venv == present")

    _print("== 2/5 editable install ==")
    _run([str(_venv_python()), "-m", "pip", "install", "-U", "pip"])
    _run([str(_venv_python()), "-m", "pip", "install", "-e", f"{REPO_ROOT}[dev]"])

    _print("== 3/5 platform.toml ==")
    if not PLATFORM_TOML.exists():
        shutil.copy2(PLATFORM_EXAMPLE, PLATFORM_TOML)
        print("Seeded platform.toml from platform.example.toml. Edit it before measuring.")
    else:
        print("platform.toml already present; kept as-is.")

    _print("== 4/5 MCP client config ==")
    _write_mcp_config()

    _print("== 5/5 handshake check ==")
    _handshake_check()

    print()
    print("Done. Reload VS Code, approve the MCP trust prompt, then run `ckh doctor` first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())