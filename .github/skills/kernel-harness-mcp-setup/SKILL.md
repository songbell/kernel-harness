---
name: kernel-harness-mcp-setup
description: 'Connect this cm-kernel-harness repository to GitHub Copilot through its local ckh MCP server. Use when setting up Copilot for the repository, adding or repairing .vscode/mcp.json, troubleshooting ckh MCP startup, or onboarding another developer to the kernel measurement workflow.'
---

# Kernel Harness MCP Setup

Use this skill when Copilot needs access to the repository's `ckh` tools. The MCP server
exposes the same guarded workflow as the CLI: `doctor`, `list_kernels`, `validate`, `equiv`,
`bench`, `round`, snapshots, and ledger operations.

## Setup

1. Work from the repository root: `~/bell/kernel-harness`.
2. Check that the editable package and MCP entry point are available:

   ```bash
   command -v ckh
   command -v ckh-mcp
   python -c 'import ckh; print(ckh.__file__)'
   ```

3. Do not assume the shell's `python` is the right interpreter. Select the interpreter that
   can import `ckh`, for example `/home/openvino-ci-97/miniforge3/bin/python`.
4. Create or update the workspace-local `.vscode/mcp.json`:

   ```json
   {
     "servers": {
       "ckh": {
         "type": "stdio",
         "command": "/absolute/path/to/python",
         "args": ["-m", "ckh.mcp"],
         "cwd": "/absolute/path/to/kernel-harness"
       }
     }
   }
   ```

   Use absolute paths. A VS Code-launched MCP server may inherit a different `PATH` from the
   interactive shell, so a bare `python` can silently select an environment without `ckh`.

5. Reload the VS Code window. The configuration is workspace-local and should not be treated as
   a portable machine-independent file when it contains user-specific paths.

## Verification

Before measuring anything, verify the server and then the rig:

```bash
python -m json.tool .vscode/mcp.json >/dev/null
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"vscode-check","version":"1"}}}' \
  | /absolute/path/to/python -m ckh.mcp
ckh doctor
```

The initialization response must identify server `ckh`. `ckh doctor` must be run before
`bench`, `equiv`, or `round`; it checks the configured repositories, `clops`, and competing GPU
work. A busy GPU invalidates timing measurements.

## Troubleshooting

| Symptom | Action |
|---|---|
| `ModuleNotFoundError: ckh` | Replace the MCP command with the interpreter that imports `ckh`; do not rely on `PATH`. |
| MCP starts but `ckh` tools fail | Run `ckh doctor` from the repository root and fix `platform.toml`. |
| `libclangFEWrapper.so` cannot be found | Put its directory in `exec.ld_library_path`; on the reference machine this is `/usr/lib/x86_64-linux-gnu`. |
| MCP config is not detected | Confirm the file is `.vscode/mcp.json`, reload the VS Code window, and inspect the MCP server status. |
| Measurements are refused | Read the `competing_gpu_work` result from `ckh doctor`; stop unrelated GPU workloads before measuring. |

Do not commit a user-specific `.vscode/mcp.json` unless the repository explicitly chooses to
standardize the interpreter and checkout path. Keep the tracked setup knowledge in this skill and
use the existing `.vscode/mcp.json.local.template` or `.vscode/mcp.json.remote.template` as the
portable starting point.