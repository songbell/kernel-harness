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


def _time_one(spec, shape, overrides, loops: int, warmup: int, source: str) -> float:
    """Mean ms over the post-warmup enqueues of one config."""
    from clops import cl

    data = spec.inputs(shape)                      # cached per shape by the spec
    opts = (spec.build_options(shape) if spec.build_options else "")
    kernels = cl.kernels(f'#include "{source}"',
                         f"{opts} {spec.defines(shape, overrides)}")
    gws, lws = spec.dispatch(shape)

    for i in range(loops):
        kernels.enqueue(spec.entry, gws, lws, *spec.args(shape, data))
    lat = cl.finish()

    tot = n = 0
    for i, ns in enumerate(lat[:loops]):
        if i >= warmup and float(ns) > 0:
            tot += float(ns)
            n += 1
    if not n:
        raise RuntimeError("no valid profiling events")
    return tot * 1e-6 / n


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
    source = str(Platform.load().sandbox / spec.source)
    configs = json.loads(a.configs)

    samples: dict[str, list[float]] = {c["label"]: [] for c in configs}
    err: dict[str, str] = {}
    # Round-robin, not config-by-config: sequential ordering charges the second config for
    # the first one's heat, which on an unstable box can invert an A/B outright.
    for _ in range(a.rounds):
        for c in configs:
            if c["label"] in err:
                continue
            try:
                shape = Shape(c["shape"])
                samples[c["label"]].append(
                    _time_one(spec, shape, c.get("overrides") or {}, a.loops, a.warmup, source))
            except Exception as e:                      # noqa: BLE001 - report, don't abort
                err[c["label"]] = f"{type(e).__name__}: {e}"[:300]

    out = {}
    for label, xs in samples.items():
        if label in err:
            out[label] = {"error": err[label]}
        elif xs:
            out[label] = {"min_ms": min(xs), "max_ms": max(xs), "n": len(xs),
                          "spread_pct": (max(xs) / min(xs) - 1.0) * 100.0}
    print(RESULT_PREFIX + json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
