"""In-environment validation worker: compile the kernel, ask the device its real limits.

Split from validate.py for the same reason runner.py is split from bench.py -- this half needs
clops and a GPU present, the other half is pure arithmetic on the spec and must stay testable
without either. Emits exactly one machine-readable line.

It compiles but never enqueues. That is the whole point of `ckh validate`: a constraint
violation found by the compiler costs a compile, and one found by a bench run costs a bench
run plus the time spent believing the number it produced.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

RESULT_PREFIX = "CKH_VALIDATE "


def _problem(sev: str, code: str, detail: str) -> dict:
    return {"severity": sev, "code": code, "detail": detail}


def _device_limits(cl) -> dict:
    try:
        return dict(cl.dev_info())
    except Exception:                                   # noqa: BLE001
        return {}


def _check_device(limits: dict, lws: list[int]) -> list[dict]:
    """A work group larger than the device allows fails at enqueue, not at compile -- and the
    failure surfaces as a generic CL error a long way from its cause."""
    out = []
    max_wg = limits.get("CL_DEVICE_MAX_WORK_GROUP_SIZE")
    if max_wg and math.prod(lws) > int(max_wg):
        out.append(_problem("error", "lws-too-large",
                            f"lws {lws} has {math.prod(lws)} work items, device max is "
                            f"{max_wg}"))
    sizes = limits.get("CL_DEVICE_MAX_WORK_ITEM_SIZES")
    if sizes:
        for i, (got, cap) in enumerate(zip(lws, sizes)):
            if got > int(cap):
                out.append(_problem("error", "lws-dim-too-large",
                                    f"lws[{i}]={got} exceeds device max {cap}"))
    return out


def _check_arity(spec, shape, source: str) -> list[dict]:
    """Compare the argument list against the ported kernel's recorded signature.

    Only possible because `ckh kernelgen` wrote the parsed signature down. Getting the count
    wrong does not fail loudly at enqueue -- the kernel reads whatever is in the next slot --
    so this is exactly the class of bug that otherwise shows up as "the kernel is subtly
    wrong" three steps later.
    """
    manifest = Path(source).with_suffix("").with_suffix("")
    manifest = manifest.parent / (Path(source).stem + ".kernelgen.json")
    if not manifest.exists():
        return []
    try:
        params = json.loads(manifest.read_text())["params"]
        data = spec.inputs(shape)
        outs = spec.outputs(shape) if spec.outputs else {}
        got = len(spec.args(shape, data, outs))
    except Exception as e:                              # noqa: BLE001
        return [_problem("warn", "arity-unknown",
                         f"could not compare against {manifest.name}: {type(e).__name__}: {e}")]

    jit = spec.jit(shape)
    # A guarded parameter is only in the signature when its guard macro is truthy, so the
    # expected count depends on this shape's jit values, not on the file alone.
    expected = sum(1 for p in params
                   if not p["guard"] or str(jit.get(p["guard"].strip(), "0")) not in ("0", ""))
    if got != expected:
        names = ", ".join(p["name"] for p in params)
        return [_problem("error", "arity-mismatch",
                         f"args() returned {got} arguments, the entry point takes {expected} "
                         f"for this shape ({names})")]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--shapes", required=True, help="JSON list of {label, shape}")
    ap.add_argument("--source", required=True)
    a = ap.parse_args()

    from clops import cl

    from ckh.kernel import Shape
    spec = importlib.import_module(f"kernels.{a.spec}").SPEC
    limits = _device_limits(cl)

    out: dict[str, list[dict]] = {}
    for item in json.loads(a.shapes):
        shape = Shape(item["shape"])
        probs: list[dict] = []
        try:
            _, lws = spec.dispatch(shape)
            probs += _check_device(limits, list(lws))
        except Exception as e:                          # noqa: BLE001
            probs.append(_problem("error", "dispatch-raised",
                                  f"{type(e).__name__}: {e}"))
        probs += _check_arity(spec, shape, a.source)
        try:
            opts = (spec.build_options(shape) if spec.build_options else "")
            cl.kernels(f'#include "{a.source}"', f"{opts} {spec.defines(shape)}")
        except Exception as e:                          # noqa: BLE001
            # The compiler's own diagnostic is the useful part; keep the tail, where the
            # first real error usually is, rather than the banner at the top.
            tail = "\n".join(str(e).strip().splitlines()[-8:])
            probs.append(_problem("error", "compile-failed", tail))
        out[item["label"]] = probs

    print(RESULT_PREFIX + json.dumps({"limits": {k: str(v) for k, v in limits.items()},
                                      "shapes": out}))


if __name__ == "__main__":
    sys.exit(main())
