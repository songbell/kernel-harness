"""In-environment measurement worker. Emits exactly one machine-readable line.

Everything else it prints -- clops banners, compiler chatter, warnings -- stays where it is
produced. The caller reads only the RESULT_PREFIX line, which is what keeps build logs out of
an agent's context. Hand-parsing raw output with ad-hoc regexes was one of the larger token
sinks in the work this replaces.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

RESULT_PREFIX = "CKH_RESULT "


def _load_spec(name: str):
    return importlib.import_module(f"kernels.{name}").SPEC


def _time_one(spec, shape, overrides, loops: int, warmup: int, source: str) -> tuple[float, float]:
    """(device_ms, wall_ms), both per post-warmup enqueue.

    Two numbers because they answer different questions and get confused constantly. The
    device time is what the kernel costs; the wall time additionally carries host submission
    overhead and whatever the queue did not overlap. A kernel that is 3x off its roofline but
    whose wall time barely exceeds its device time has no host-side problem to find, and one
    where they diverge has a dispatch cost that no amount of kernel micro-optimisation touches.
    """
    import time

    from clops import cl

    data = spec.inputs(shape)                      # cached per shape by the spec
    opts = (spec.build_options(shape) if spec.build_options else "")
    kernels = cl.kernels(f'#include "{source}"',
                         f"{opts} {spec.defines(shape, overrides)}")
    gws, lws = spec.dispatch(shape)

    t0 = time.perf_counter()
    for i in range(loops):
        outs = spec.outputs(shape) if spec.outputs else {}
        kernels.enqueue(spec.entry, gws, lws, *spec.args(shape, data, outs))
    lat = cl.finish()
    wall_ms = (time.perf_counter() - t0) * 1e3

    tot = n = 0
    for i, ns in enumerate(lat[:loops]):
        if i >= warmup and float(ns) > 0:
            tot += float(ns)
            n += 1
    if not n:
        raise RuntimeError("no valid profiling events")
    # Wall time covers the warmup enqueues too; there is no per-enqueue host timestamp to
    # exclude them with, so divide by the full loop count and say so rather than pretending
    # the two averages are over the same set.
    return tot * 1e-6 / n, wall_ms / loops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--configs", required=True,
                    help="JSON list of {label, shape, overrides}; interleaved round-robin")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--loops", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    a = ap.parse_args()

    from ckh.kernel import Shape
    from ckh.platform import Platform
    spec = _load_spec(a.spec)
    # Resolve the source here rather than threading it through the config list: the runner is
    # already in the measurement environment, so it is the right place to know the paths.
    default_source = str(Platform.load().sandbox / spec.source)
    configs = json.loads(a.configs)

    samples: dict[str, list[float]] = {c["label"]: [] for c in configs}
    walls: dict[str, list[float]] = {c["label"]: [] for c in configs}
    err: dict[str, str] = {}
    # Round-robin, not config-by-config: sequential ordering charges the second config for
    # the first one's heat, which on an unstable box can invert an A/B outright.
    for _ in range(a.rounds):
        for c in configs:
            if c["label"] in err:
                continue
            try:
                shape = Shape(c["shape"])
                dev, wall = _time_one(spec, shape, c.get("overrides") or {}, a.loops,
                                      a.warmup, c.get("source") or default_source)
                samples[c["label"]].append(dev)
                walls[c["label"]].append(wall)
            except Exception as e:                      # noqa: BLE001 - report, don't abort
                err[c["label"]] = f"{type(e).__name__}: {e}"[:300]

    out = {}
    for label, xs in samples.items():
        if label in err:
            out[label] = {"error": err[label]}
        elif xs:
            out[label] = {"min_ms": min(xs), "max_ms": max(xs), "n": len(xs),
                          "spread_pct": (max(xs) / min(xs) - 1.0) * 100.0,
                          "wall_ms": min(walls[label]) if walls[label] else None}
    print(RESULT_PREFIX + json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
