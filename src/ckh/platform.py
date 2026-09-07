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
    # Interpreter every measurement subprocess runs under. Bare "python" is whatever PATH
    # resolves to -- fine from a shell with the right env activated, wrong when the harness
    # is launched by an editor or a systemd unit that inherited a different PATH.
    python: str
    # Path to a checkout that provides `clops` (the CM compile/run binding every measurement
    # ultimately calls) -- e.g. your own aboutSHW's `opencl/` directory. Added to PYTHONPATH
    # ahead of everything else if set. Leave unset/empty to rely on a normally pip-installed
    # `clops` instead (clops ships its own setup.py, so `pip install -e /path/to/opencl` in
    # whatever checkout you have is a one-time step outside this config entirely). Either
    # way there must be exactly one `clops` reachable -- `check_clops()` reports which.
    clops_path: str | None
    noise_floor_pct: float
    rounds: int
    # Where `ckh kernelgen` drops a kernel ported out of the plugin, relative to `sandbox`.
    kernelgen_dest: str
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
            clops_path=d["exec"].get("clops_path") or None,
            python=d["exec"].get("python") or "python",
            sandbox=Path(d["repos"]["sandbox"]),
            production=Path(d["repos"]["production"]),
            sandbox_pythonpath=d["repos"].get("sandbox_pythonpath", []),
            noise_floor_pct=float(d.get("rig", {}).get("noise_floor_pct", 2.0)),
            rounds=int(d.get("rig", {}).get("rounds", 3)),
            kernelgen_dest=d.get("kernelgen", {}).get("dest", "opencl/tests/pageatten"),
            raw=d,
        )

    # -- execution -------------------------------------------------------------------

    def env(self) -> dict[str, str]:
        e = dict(os.environ)
        pypath = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
        if self.clops_path:
            pypath.append(self.clops_path)
        pypath += [str(self.sandbox / sub) for sub in self.sandbox_pythonpath]
        e["PYTHONPATH"] = os.pathsep.join(pypath + [e.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        if self.ld_library_path:
            e["LD_LIBRARY_PATH"] = os.pathsep.join(
                [self.ld_library_path, e.get("LD_LIBRARY_PATH", "")]
            ).rstrip(os.pathsep)
        return e

    def check_clops(self) -> tuple[bool, str]:
        """Is `clops` actually importable in the environment every measurement runs in?

        Run as a real subprocess with `env()`, not an in-process import: importing here would
        pollute this process's own sys.modules and could report success against a stale or
        unrelated clops already loaded by something else in this interpreter.
        """
        r = subprocess.run(
            [self.python, "-c", "import clops; print(clops.__file__)"],
            capture_output=True, text=True, env=self.env(), timeout=60,
        )
        if r.returncode == 0:
            # Report where it actually resolved from rather than guessing which of
            # exec.clops_path / sandbox_pythonpath / a real pip install is responsible --
            # Python's import system doesn't tell us that, and guessing wrong is worse than
            # not claiming it. Last line only: importing clops itself prints a build/GPU
            # banner to stdout ahead of our own print(clops.__file__).
            resolved = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "<no output>"
            return True, f"clops OK: {resolved}"
        detail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["unknown error"]
        return False, (
            f"clops NOT importable ({detail[0]}). Either set exec.clops_path to a checkout "
            f"that provides it (e.g. your own aboutSHW's opencl/ dir), or `pip install -e "
            f"<path>/opencl` once outside this config."
        )

    def run_module(self, module: str, args: list[str], timeout: int = 1800,
                   cwd: Path | None = None):
        """Run a measurement module. Its stdout stays here and never reaches the caller
        unless it is the one JSON line we asked for -- see runner.RESULT_PREFIX.

        `cwd` defaults to the historical pa_small_q-shaped path: fine as long as there is one
        kernel family, wrong the moment a second one lives elsewhere. Callers with a
        KernelSpec should pass `self.sandbox / (spec.cwd or "opencl/tests/pageatten")`.
        """
        if self.backend != "local":
            raise SystemExit(
                f"exec.backend='{self.backend}' is not implemented in the core. "
                f"Backends beyond 'local' are machine-specific; add one under ckh/backends/."
            )
        return subprocess.run(
            [self.python, "-m", module, *args],
            capture_output=True, text=True, timeout=timeout,
            cwd=cwd or (self.sandbox / "opencl" / "tests" / "pageatten"), env=self.env(),
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
