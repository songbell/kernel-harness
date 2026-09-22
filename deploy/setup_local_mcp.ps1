Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$script = Join-Path $repoRoot 'deploy\setup_local_mcp.py'

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $script
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $script
    exit $LASTEXITCODE
}

throw 'No Python launcher found. Install Python 3.11+ or add `py`/`python` to PATH.'