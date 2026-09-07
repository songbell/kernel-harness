"""In-environment equivalence worker. Mirrors runner.py's shape and its single
CKH_RESULT-line convention, but reports pass/fail against a KernelSpec.reference instead of
timing -- kept as a separate module rather than a mode flag on runner.py because bench and
equiv genuinely execute a kernel differently (bench re-enqueues in a tight loop and only ever
reads back a scalar latency; equiv enqueues once and reads back real buffer contents).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

RESULT_PREFIX = "CKH_RESULT "


def _load_spec(name: str):
    return importlib.import_module(f"kernels.{name}").SPEC


def _defines(jit_fn, shape) -> str:
    return " ".join(f"-D{k}={v}" for k, v in jit_fn(shape).items())


def _run_kernel_once(jit_fn, dispatch_fn, args_fn, outputs_fn, build_options_fn, entry,
                     source, shape, data, compare: list[str]) -> dict:
    """Compile, enqueue once, finish, read back the named output buffers as numpy arrays."""
    from clops import cl

    opts = build_options_fn(shape) if build_options_fn else ""
    kernels = cl.kernels(f'#include "{source}"', f"{opts} {_defines(jit_fn, shape)}")
    gws, lws = dispatch_fn(shape)
    outs = outputs_fn(shape) if outputs_fn else {}
    kernels.enqueue(entry, gws, lws, *args_fn(shape, data, outs))
    cl.finish()
    return {name: outs[name].numpy() for name in compare}


def _eval_one(spec, shape_dict: dict, default_source: str, sandbox) -> dict:
    from ckh.kernel import Shape
    from ckh.reference import KernelReference, TorchReference

    shape = Shape(shape_dict)
    data = spec.inputs(shape)
    readback = _run_kernel_once(spec.jit, spec.dispatch, spec.args, spec.outputs,
                                spec.build_options, spec.entry, default_source, shape, data,
                                spec.compare)

    ref = spec.reference
    if ref is None:
        raise SystemExit(f"kernel '{spec.name}' has no `reference` set")

    if isinstance(ref, TorchReference):
        result = ref.evaluate(shape, data, readback)
        return {"kind": "torch", "passed": result.passed, "detail": result.detail,
               "max_abs_diff": result.max_abs_diff, "non_vacuous": ref.non_vacuous}

    if isinstance(ref, KernelReference):
        r = ref.resolve(spec)
        base_source = str(sandbox / r["source"])
        base_readback = _run_kernel_once(r["jit"], r["dispatch"], r["args"], r["outputs"],
                                         r["build_options"], r["entry"], base_source, shape,
                                         data, spec.compare)
        per_buf = ref.evaluate(readback, base_readback, spec.compare)
        passed = all(v.passed for v in per_buf.values())
        detail = "; ".join(f"{k}: {v.detail}" for k, v in per_buf.items())
        max_abs = max((v.max_abs_diff or 0.0) for v in per_buf.values())
        return {"kind": "kernel", "passed": passed, "detail": detail, "max_abs_diff": max_abs,
               "non_vacuous": ref.non_vacuous}

    raise SystemExit(f"kernel '{spec.name}' has an unrecognised reference type {type(ref)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--shapes", required=True, help="JSON list of {label, shape}")
    a = ap.parse_args()

    from ckh.platform import Platform
    plat = Platform.load()
    spec = _load_spec(a.spec)
    default_source = str(plat.sandbox / spec.source)
    shapes = json.loads(a.shapes)

    out = {}
    for item in shapes:
        try:
            out[item["label"]] = _eval_one(spec, item["shape"], default_source, plat.sandbox)
        except Exception as e:                                # noqa: BLE001 - report, don't abort
            out[item["label"]] = {"error": f"{type(e).__name__}: {e}"[:400]}
    print(RESULT_PREFIX + json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
