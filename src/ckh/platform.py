"""Machine facts, loaded once from platform.toml instead of rediscovered every session.

Re-deriving repo paths, loader paths and build commands cost a noticeable fraction of the
tokens in the work this harness came from. They are facts about the box, so they live in a
file.

Execution is deliberately pluggable and defaults to `local`. Isolating measurement in a
container, a cpuset, a fixed clock, etc. are *remedies for an unstable rig* -- valid, but
specific to a machine. The portable part is the discipline in rig.py: measure the noise
floor, and refuse to report an effect smaller than it.
"""
from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Platform:
    backend: str
    ld_library_path: str
    sandbox: Path
    production: Path
    sandbox_pythonpath: list[str]
    noise_floor_pct: float
    rounds: int
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Platform":
        p = path or REPO_ROOT / "platform.toml"
        if not p.exists():
            raise SystemExit(
                f"{p} not found. Copy platform.example.toml to platform.toml and edit it."
            )
        d = tomllib.loads(p.read_text())
        return cls(
            backend=d["exec"].get("backend", "local"),
            ld_library_path=d["exec"].get("ld_library_path", ""),
            sandbox=Path(d["repos"]["sandbox"]),
            production=Path(d["repos"]["production"]),
            sandbox_pythonpath=d["repos"].get("sandbox_pythonpath", []),
            noise_floor_pct=float(d.get("rig", {}).get("noise_floor_pct", 2.0)),
            rounds=int(d.get("rig", {}).get("rounds", 3)),
            raw=d,
        )

    # -- execution -------------------------------------------------------------------

    def env(self) -> dict[str, str]:
        e = dict(os.environ)
        pypath = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
        pypath += [str(self.sandbox / sub) for sub in self.sandbox_pythonpath]
        e["PYTHONPATH"] = os.pathsep.join(pypath + [e.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        if self.ld_library_path:
            e["LD_LIBRARY_PATH"] = os.pathsep.join(
                [self.ld_library_path, e.get("LD_LIBRARY_PATH", "")]
            ).rstrip(os.pathsep)
        return e

    def run_module(self, module: str, args: list[str], timeout: int = 1800):
        """Run a measurement module. Its stdout stays here and never reaches the caller
        unless it is the one JSON line we asked for -- see runner.RESULT_PREFIX."""
        if self.backend != "local":
            raise SystemExit(
                f"exec.backend='{self.backend}' is not implemented in the core. "
                f"Backends beyond 'local' are machine-specific; add one under ckh/backends/."
            )
        return subprocess.run(
            ["python", "-m", module, *args],
            capture_output=True, text=True, timeout=timeout,
            cwd=self.sandbox / "opencl" / "tests" / "pageatten", env=self.env(),
        )

    # -- doctor ----------------------------------------------------------------------

    def competing_gpu_work(self) -> list[str]:
        """Anything else on the GPU makes every number meaningless.

        A benchmark running in parallel once produced `ablation_off > ablation_on` -- a
        physically impossible ordering -- and invalidated a whole batch of results. Check
        before every batch, not once per session.
        """
        pat = "benchmark|visual_language|pytest|clops"
        r = subprocess.run(["bash", "-lc", f"ps -eo pid,args | grep -iE '{pat}' | grep -v grep"],
                           capture_output=True, text=True)
        return [l.strip()[:110] for l in r.stdout.splitlines() if l.strip()]
