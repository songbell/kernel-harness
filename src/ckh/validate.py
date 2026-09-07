"""`ckh validate` -- everything that can be wrong before a single measurement is worth taking.

Two kinds of check, deliberately kept apart:

  STATIC   pure arithmetic on the KernelSpec -- jit values present, gws/lws well-formed and
           divisible, spec-declared constraints. No GPU, no clops, fully unit-testable.
  IN-ENV   device limits and a real compile (validate_runner.py). Compiles, never enqueues.

The ordering is the point. A gws not divisible by its lws does not raise; it quietly rounds
and every subsequent number describes a different dispatch than the one you think you are
measuring. Finding that here costs nothing. Finding it after a bench round costs the round
plus however long the wrong number was believed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .bench import shape_label
from .platform import Platform
from .validate_runner import RESULT_PREFIX


@dataclass
class Problem:
    severity: str          # "error" blocks; "warn" is reported and does not
    code: str
    detail: str

    def as_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "detail": self.detail}


# -- static: no GPU, no clops ----------------------------------------------------------

def check_jit(spec, shape) -> list[Problem]:
    """A macro left at None reaches the compiler as `-DFOO=None`, which is a C identifier and
    so frequently compiles into something meaningless rather than failing."""
    out = []
    try:
        jit = spec.jit(shape)
    except Exception as e:                              # noqa: BLE001
        return [Problem("error", "jit-raised", f"{type(e).__name__}: {e}")]
    for k, v in jit.items():
        if v is None or v == "":
            out.append(Problem("error", "jit-unset",
                               f"{k} has no value -- `ckh kernelgen` leaves these as TODO"))
        elif not isinstance(v, (int, float, str)):
            out.append(Problem("error", "jit-not-scalar",
                               f"{k}={v!r} ({type(v).__name__}) cannot become a -D"))
    return out


def check_dispatch(spec, shape) -> list[Problem]:
    out = []
    try:
        gws, lws = spec.dispatch(shape)
    except NotImplementedError:
        return [Problem("error", "dispatch-todo",
                        "dispatch() is still the kernelgen scaffold -- see kernel-onboarder")]
    except Exception as e:                              # noqa: BLE001
        return [Problem("error", "dispatch-raised", f"{type(e).__name__}: {e}")]

    gws, lws = list(gws), list(lws)
    if len(gws) != len(lws):
        return [Problem("error", "dispatch-rank",
                        f"gws has {len(gws)} dimensions, lws has {len(lws)}")]
    if not 1 <= len(gws) <= 3:
        out.append(Problem("error", "dispatch-rank", f"{len(gws)} dimensions; OpenCL allows 3"))
    for i, (g, l) in enumerate(zip(gws, lws)):
        if not isinstance(g, int) or not isinstance(l, int):
            out.append(Problem("error", "dispatch-not-int", f"dim {i}: gws={g!r} lws={l!r}"))
            continue
        if g <= 0 or l <= 0:
            out.append(Problem("error", "dispatch-nonpositive", f"dim {i}: gws={g} lws={l}"))
            continue
        if g % l:
            out.append(Problem("error", "dispatch-indivisible",
                               f"dim {i}: gws={g} is not a multiple of lws={l} -- this does "
                               f"not raise, it silently dispatches a different grid"))
    if all(isinstance(x, int) and x > 0 for x in lws) and math.prod(lws) == 1 and len(gws) > 1:
        out.append(Problem("warn", "lws-all-ones",
                           "lws is all 1s: one work item per group. Correct for some kernels, "
                           "and a common transcription slip for others -- confirm it is meant"))
    return out


def check_constraints(spec, shape) -> list[Problem]:
    """Kernel-specific rules the spec declares itself.

    They live on the spec rather than in this module because they are facts about one kernel
    (`Q_head_chunk_size * TILE_Q <= 8` is a DPAS RepeatCount cap, true for pa_small_q and
    meaningless for anything else). A tool that hardcoded them would be wrong for the second
    kernel and would have to be edited to adopt it.
    """
    out = []
    for fn in getattr(spec, "constraints", None) or []:
        try:
            msg = fn(shape)
        except Exception as e:                          # noqa: BLE001
            out.append(Problem("error", "constraint-raised",
                               f"{getattr(fn, '__name__', fn)}: {type(e).__name__}: {e}"))
            continue
        if msg:
            out.append(Problem("error", "constraint-violated",
                               f"{getattr(fn, '__name__', 'constraint')}: {msg}"))
    return out


def static_checks(spec, shapes: list[dict]) -> dict[str, list[Problem]]:
    return {shape_label(spec, s): (check_jit(spec, _shape(s))
                                   + check_dispatch(spec, _shape(s))
                                   + check_constraints(spec, _shape(s)))
            for s in shapes}


def _shape(values: dict):
    from .kernel import Shape
    return Shape(values)


# -- in-env: device limits + a real compile --------------------------------------------

def env_checks(plat: Platform, spec, shapes: list[dict]) -> dict:
    items = [{"label": shape_label(spec, s), "shape": s} for s in shapes]
    cwd = plat.sandbox / (spec.cwd or "opencl/tests/pageatten")
    r = plat.run_module("ckh.validate_runner", [
        "--spec", spec.name,
        "--shapes", json.dumps(items),
        "--source", str(plat.sandbox / spec.source),
    ], cwd=cwd)
    for line in r.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
    raise SystemExit(f"validate runner produced no result.\n{tail}")


# -- rendering -------------------------------------------------------------------------

def render(static: dict[str, list[Problem]], env: dict | None) -> tuple[str, int]:
    """Returns (table, error_count). Warnings are shown and do not block."""
    env_shapes = (env or {}).get("shapes", {})
    lines, errors, warns = [], 0, 0
    for label in static:
        probs = [p.as_dict() for p in static[label]] + env_shapes.get(label, [])
        if not probs:
            lines.append(f"{label}  OK")
            continue
        lines.append(f"{label}")
        for p in probs:
            errors += p["severity"] == "error"
            warns += p["severity"] == "warn"
            mark = "ERROR" if p["severity"] == "error" else "warn "
            lines.append(f"  {mark}  {p['code']}: {p['detail']}")

    if env is None:
        lines.append("\n(static checks only -- the compile and device limits were skipped)")
    if errors:
        lines.append(f"\n{errors} error(s), {warns} warning(s) -- do NOT measure this yet. "
                     f"A bench run on an invalid dispatch produces a number, not an error.")
    else:
        lines.append(f"\n0 errors, {warns} warning(s)"
                     + ("" if not warns else " -- warnings do not block, but read them"))
    return "\n".join(lines), errors
