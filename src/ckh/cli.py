"""`ckh` -- one command per phase, compact output.

Every subcommand answers with a table or a verdict, never a log. Anything verbose stays in
the measurement environment.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from . import bench, ledger
from .platform import Platform


def _spec(name: str):
    mod = importlib.import_module(f"kernels.{name}")
    return mod.SPEC, getattr(mod, "DEFAULT_AXES", {})


def cmd_doctor(a) -> int:
    plat = Platform.load()
    print(f"backend        {plat.backend}")
    print(f"sandbox        {plat.sandbox}  {'ok' if plat.sandbox.exists() else 'MISSING'}")
    print(f"production     {plat.production}  {'ok' if plat.production.exists() else 'MISSING'}")
    print(f"noise floor    {plat.noise_floor_pct}%   rounds {plat.rounds}")
    busy = plat.competing_gpu_work()
    if busy:
        print("\ncompeting GPU work -- measurements will be junk until this stops:")
        for b in busy:
            print(f"  {b}")
        return 1
    print("competing work none")
    return 0


def cmd_bench(a) -> int:
    plat = Platform.load()
    if plat.competing_gpu_work():
        print("refusing to measure: other GPU work is running (see `ckh doctor`)")
        return 1
    spec, default_axes = _spec(a.kernel)
    axes = dict(default_axes)
    for kv in a.axis or []:
        k, _, v = kv.partition("=")
        axes[k] = [int(x) if x.lstrip("-").isdigit() else x for x in v.split(",")]
    shapes = bench.expand(axes)
    variants = json.loads(a.variants) if a.variants else None
    res = bench.measure(plat, spec, shapes, variants, rounds=a.rounds)
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(bench.render(res, plat.noise_floor_pct))
    return 0


def cmd_ledger(a) -> int:
    if a.add:
        f = dict(kv.split("=", 1) for kv in a.add)
        ledger.add(a.kernel, change=f["change"], verdict=f["verdict"], shape=f.get("shape", ""),
                   numbers=f.get("numbers", ""), note=f.get("note", ""),
                   bit_exact={"true": True, "false": False}.get(f.get("bit_exact", "").lower()))
        print("recorded")
        return 0
    print(ledger.render(a.kernel, a.only))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="ckh", description="CM kernel optimization harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="validate the environment before measuring").set_defaults(fn=cmd_doctor)

    b = sub.add_parser("bench", help="interleaved min-of-N timing over a shape grid")
    b.add_argument("kernel")
    b.add_argument("--axis", action="append", metavar="K=v1,v2",
                   help="override a shape axis; repeatable")
    b.add_argument("--variants", help='JSON {label: {DEFINE: value}} for an A/B in one batch')
    b.add_argument("--rounds", type=int)
    b.add_argument("--json", action="store_true", help="machine-readable, for agents")
    b.set_defaults(fn=cmd_bench)

    l = sub.add_parser("ledger", help="prior findings -- read this BEFORE proposing a change")
    l.add_argument("kernel")
    l.add_argument("--only", help="filter by verdict prefix, e.g. rejected")
    l.add_argument("--add", nargs="+", metavar="k=v",
                   help="change=.. verdict=.. shape=.. numbers=.. [bit_exact=..] [note=..]")
    l.set_defaults(fn=cmd_ledger)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
