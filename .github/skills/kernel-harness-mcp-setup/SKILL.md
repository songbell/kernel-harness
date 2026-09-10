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
4. Use the tracked workspace MCP files. `.mcp.json` is read natively by Agent Host, and
   `.vscode/mcp.json` is read by VS Code workspace MCP support. Both should point at the repository
   wrapper script:

   ```json
   {
     "servers": {
       "ckh": {
         "type": "stdio",
         "command": "${workspaceFolder}/deploy/ckh-mcp-vscode",
         "cwd": "${workspaceFolder}"
       }
     }
   }
   ```

   The wrapper selects `.venv/bin/python`, then `CONDA_PREFIX/bin/python`, then `python3` or
   `python`, and adds `src` to `PYTHONPATH`. This avoids a bare `ckh-mcp` failing when the VS Code
   process has a different `PATH` from the shell.

5. Reload the VS Code window. The configuration is workspace-local and should not be treated as
   a portable machine-independent file when it contains user-specific paths.

## Verification

Before measuring anything, verify the server and then the rig:

```bash
python -m json.tool .mcp.json >/dev/null
python -m json.tool .vscode/mcp.json >/dev/null
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"vscode-check","version":"1"}}}' \
  | ./deploy/ckh-mcp-vscode
ckh doctor
```

The initialization response must identify server `ckh`. `ckh doctor` must be run before
`bench`, `equiv`, or `round`; it checks the configured repositories, `clops`, and competing GPU
work. A busy GPU invalidates timing measurements.

## Troubleshooting

| Symptom | Action |
|---|---|
| `ModuleNotFoundError: ckh` | Run `pip install -e .` in the Python environment used by VS Code, or create `.venv` in the repo root. |
| MCP starts but `ckh` tools fail | Run `ckh doctor` from the repository root and fix `platform.toml`. |
| `libclangFEWrapper.so` cannot be found | Put its directory in `exec.ld_library_path`; on the reference machine this is `/usr/lib/x86_64-linux-gnu`. |
| MCP config is not detected | Open the repository root as a VS Code workspace folder, reload the window, then run **MCP: List Servers**. |
| Measurements are refused | Read the `competing_gpu_work` result from `ckh doctor`; stop unrelated GPU workloads before measuring. |

Do not replace the tracked MCP files with user-specific absolute paths. Put machine-specific Python
selection in a repo-local `.venv`, the active Conda environment, or the terminal environment used to
launch VS Code.