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
            rounds: int | None = None, sources: dict[str, str] | None = None) -> dict:
    """variants maps a label suffix -> jit overrides; sources maps it -> a kernel source path.

    Both are A/B'd inside one interleaved batch, which is the only comparison that survives a
    drifting rig -- a number from a previous session is not comparable.
    """
    variants = variants or {"": {}}
    configs = []
    for shape in shapes:
        for vname, ov in variants.items():
            label = shape_label(spec, shape) + (f" | {vname}" if vname else "")
            cfg = {"label": label, "shape": shape, "overrides": ov}
            if sources and vname in sources:
                cfg["source"] = sources[vname]
            configs.append(cfg)

    cwd = plat.sandbox / (spec.cwd or "opencl/tests/pageatten")
    r = plat.run_module("ckh.runner", [
        "--spec", spec.name,
        "--configs", json.dumps(configs),
        "--rounds", str(rounds or plat.rounds),
    ], cwd=cwd)
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
        # Device time is the headline; wall time next to it is the only way to see host
        # submission cost, which no amount of kernel work removes.
        wall = f"  wall {v['wall_ms']:7.3f} ms" if v.get("wall_ms") else ""
        rows.append(f"{label:<{w}}  {v['min_ms']:8.3f} ms{wall}  spread {v['spread_pct']:5.1f}%"
                    f"  n={v['n']}{flag}")
    out = "\n".join(rows)
    if unstable:
        out += (f"\n\n{len(unstable)} config(s) exceeded the {noise_floor_pct}% noise floor. "
                f"Differences smaller than the spread are NOT resolvable -- fix the rig "
                f"before drawing conclusions.")
    return out


def pair_by_shape(results: dict, base_name: str, new_name: str) -> dict[str, tuple[dict, dict]]:
    """shape -> (baseline, candidate), for every shape where both sides measured.

    Shared by `render_round` and `ckh trial`'s bench gate so the two cannot drift apart on
    what counts as a comparable pair.
    """
    shapes: dict[str, dict[str, dict]] = {}
    for label, v in results.items():
        if "error" in v:
            continue
        shape, _, variant = label.rpartition(" | ")
        shapes.setdefault(shape, {})[variant] = v
    return {s: (v[base_name], v[new_name]) for s, v in shapes.items()
            if base_name in v and new_name in v}


def round_deltas(results: dict, base_name: str, new_name: str,
                 noise_pct: float) -> dict[str, tuple[float, float]]:
    """shape -> (delta_pct, resolution_pct). The run-to-run spread bounds what a comparison
    can resolve, and the configured noise floor is a lower bound on that."""
    return {s: ((n["min_ms"] / b["min_ms"] - 1.0) * 100.0,
                max(b["spread_pct"], n["spread_pct"], noise_pct))
            for s, (b, n) in pair_by_shape(results, base_name, new_name).items()}


def render_round(results: dict, base_name: str, new_name: str, noise_pct: float) -> tuple[str, int]:
    """Per-shape delta of new vs base, separating "no change" from "cannot tell".

    The full grid is compared, not just the shape being optimized: a round that improves one
    shape and silently breaks another is the failure mode this exists to catch, and it has
    happened twice (a partition tuned at long context regressed short context by 3.3x, and a
    reduce-split change passed every sandbox check while breaking the shipped model).

    A delta smaller than the measured spread is INCONCLUSIVE, not "no regression". Reporting
    it as the latter is how a real 5% regression gets waved through -- observed while testing
    this very function on a drifting host.
    """
    pairs = pair_by_shape(results, base_name, new_name)
    deltas = round_deltas(results, base_name, new_name, noise_pct)

    w = max((len(s) for s in pairs), default=6)
    lines = [f"{'shape':<{w}}  {base_name:>9}  {new_name:>9}  {'delta':>8}  {'noise':>6}"]
    better = worse = unknown = 0
    for shape, (b, n) in pairs.items():
        d, res = deltas[shape]
        if abs(d) <= res:
            tag, unknown = "  ? INCONCLUSIVE", unknown + 1
        elif d > 0:
            tag, worse = "  <- REGRESSION", worse + 1
        else:
            tag, better = "  improved", better + 1
        lines.append(f"{shape:<{w}}  {b['min_ms']:9.3f}  {n['min_ms']:9.3f}  "
                     f"{d:+7.1f}%  {res:5.1f}%{tag}")

    if worse:
        verdict = (f"\n{better} improved, {worse} REGRESSED, {unknown} inconclusive -> "
                   f"NOT shippable: fix or scope the regression first")
    elif unknown:
        verdict = (f"\n{better} improved, {unknown} inconclusive -> VERDICT WITHHELD. The rig "
                   f"cannot resolve those shapes; stabilise it or raise --rounds before "
                   f"believing this round is safe.")
    else:
        verdict = f"\n{better} improved, no regression, all shapes resolved"
    return "\n".join(lines) + verdict, worse
