"""Host-side orchestration: build the config list, invoke the runner, print a table.

The table is the product. If a caller ever has to read the runner's raw output, this module
has failed at its job.
"""
from __future__ import annotations

import itertools
import json

from .platform import Platform
from .runner import RESULT_PREFIX


def expand(axes: dict[str, list]) -> list[dict]:
    """Cartesian product of shape axes, in declaration order."""
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


def measure(plat: Platform, spec, shapes: list[dict], variants: dict[str, dict] | None = None,
            rounds: int | None = None) -> dict:
    """variants maps a label suffix -> jit overrides, for A/B in a single interleaved batch."""
    variants = variants or {"": {}}
    configs = []
    for shape in shapes:
        for vname, ov in variants.items():
            label = shape_label(spec, shape) + (f" | {vname}" if vname else "")
            configs.append({"label": label, "shape": shape, "overrides": ov})

    r = plat.run_module("ckh.runner", [
        "--spec", spec.name,
        "--configs", json.dumps(configs),
        "--rounds", str(rounds or plat.rounds),
    ])
    for line in r.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    # Only on failure does any raw output surface, and only the tail of it.
    tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
    raise SystemExit(f"runner produced no result.\n{tail}")


def shape_label(spec, shape: dict) -> str:
    keys = spec.label_keys or sorted(shape)
    return " ".join(f"{k}={shape[k]}" for k in keys if k in shape)


def render(results: dict, noise_floor_pct: float) -> str:
    rows, unstable = [], []
    w = max((len(k) for k in results), default=6)
    for label, v in results.items():
        if "error" in v:
            rows.append(f"{label:<{w}}  ERROR  {v['error']}")
            continue
        flag = ""
        if v["spread_pct"] > noise_floor_pct:
            flag = "  <- spread exceeds noise floor"
            unstable.append(label)
        rows.append(f"{label:<{w}}  {v['min_ms']:8.3f} ms  spread {v['spread_pct']:5.1f}%"
                    f"  n={v['n']}{flag}")
    out = "\n".join(rows)
    if unstable:
        out += (f"\n\n{len(unstable)} config(s) exceeded the {noise_floor_pct}% noise floor. "
                f"Differences smaller than the spread are NOT resolvable -- fix the rig "
                f"before drawing conclusions.")
    return out
