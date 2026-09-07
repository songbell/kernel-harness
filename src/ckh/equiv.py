"""Host-side orchestration for `ckh equiv` -- runs the kernel-under-test against its declared
KernelSpec.reference across a shape grid and reports pass/fail, not timing. Mirrors bench.py's
shape deliberately: expand the axes, invoke the runner subprocess, render a table.
"""
from __future__ import annotations

import json

from .bench import shape_label
from .equiv_runner import RESULT_PREFIX
from .platform import Platform


def measure(plat: Platform, spec, shapes: list[dict]) -> dict:
    if spec.reference is None:
        raise SystemExit(
            f"kernel '{spec.name}' has no `reference` set on its KernelSpec -- nothing for "
            f"`ckh equiv` to check against. See kernels/pa_small_q.py (TorchReference) or "
            f"kernels/pa_small_q_vs_baseline.py (KernelReference) for the two supported shapes."
        )
    items = [{"label": shape_label(spec, s), "shape": s} for s in shapes]
    cwd = plat.sandbox / (spec.cwd or "opencl/tests/pageatten")
    r = plat.run_module("ckh.equiv_runner",
                        ["--spec", spec.name, "--shapes", json.dumps(items)], cwd=cwd)
    for line in r.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
    raise SystemExit(f"equiv runner produced no result.\n{tail}")


def render(spec, results: dict) -> str:
    rows, failed = [], []
    w = max((len(k) for k in results), default=6)
    non_vacuous_seen = set()
    for label, v in results.items():
        if "error" in v:
            rows.append(f"{label:<{w}}  ERROR  {v['error']}")
            failed.append(label)
            continue
        mark = "PASS" if v["passed"] else "FAIL"
        if not v["passed"]:
            failed.append(label)
        mad = f"  max_abs_diff={v['max_abs_diff']:.3e}" if v.get("max_abs_diff") is not None else ""
        rows.append(f"{label:<{w}}  {mark}  ({v['kind']}){mad}  {v['detail']}")
        if v.get("non_vacuous"):
            non_vacuous_seen.add(v["non_vacuous"])
    out = "\n".join(rows)
    out += f"\n\n{len(results) - len(failed)}/{len(results)} shapes passed"
    if failed:
        out += f", {len(failed)} FAILED: {', '.join(failed)}"
    if not non_vacuous_seen:
        out += (f"\n\nWARNING: {spec.name}'s reference has no `non_vacuous` note -- there is "
               f"no documented reason to believe this check is capable of failing. A passing "
               f"equiv run with an empty non_vacuous field should not be trusted; see "
               f"kernels/pa_small_q_vs_baseline.py for what a filled-in one looks like.")
    return out
