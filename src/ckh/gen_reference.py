"""Deterministic generator: kernels/pending/<name>_pytorch.py -> kernels/<name>.py's
inputs()/TorchReference half.

This exists because the common case of this translation is entirely mechanical -- parse
concrete get_inputs()/get_init_inputs() values, template the inputs()/TorchReference
boilerplate, detect matmul/reduction ops for a tolerance heuristic -- and routing it through
an agent (a full subagent LLM invocation) cost real tokens for zero judgment actually being
exercised. Only one case genuinely needs semantic judgment an LLM has and a script doesn't:
shape left completely unspecified, needing operation-aware defaults. That case is refused
here (exit 2) with a pointer at the reference-generator agent, rather than guessed badly.

See kernels/pending/README.md for the input contract and kernels/gemm.py for a generated
worked example.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import re
import sys
from pathlib import Path

# Substrings in Model.forward's source that imply real floating-point summation-order
# rounding, not just elementwise arithmetic -- widen the tolerance when any is present.
_LOOSE_TOL_MARKERS = ("matmul", "mm(", "bmm(", "einsum", "conv", " @ ", "sum(", "mean(",
                     "cumsum", "prod(")
TIGHT_TOL = (1e-4, 1e-4)
LOOSE_TOL = (1e-2, 1e-2)


class Refused(Exception):
    """Raised when this case needs judgment a script shouldn't guess at."""


def _load_pending(name: str):
    mod = importlib.import_module(f"kernels.pending.{name}_pytorch")
    for attr in ("Model", "get_inputs", "get_init_inputs"):
        if not hasattr(mod, attr):
            raise Refused(f"kernels/pending/{name}_pytorch.py has no `{attr}` -- does not "
                          f"match the standard convention (see kernels/pending/README.md). "
                          f"Use the reference-generator agent instead; it can work from a "
                          f"less rigid shape.")
    return mod


def _named_args(fn, values: list):
    """Zip a get_inputs()/get_init_inputs()-style positional list to the callee's own
    parameter names via introspection, so this works for any Model, not just one convention
    of variable names."""
    params = [p for p in inspect.signature(fn).parameters if p != "self"]
    return dict(zip(params, values))


def _tensor_axes(name: str, tensor, known_scalars: dict) -> dict:
    """One axis per tensor dimension, named `<name>_dim<i>` by default, renamed to a known
    scalar's name if that scalar's value happens to equal this dimension's size -- recovers
    readable names (e.g. "input_size" instead of "x_dim1") without needing to parse variable
    names out of source text, which would be far less reliable than this numeric check."""
    axes = {}
    for i, size in enumerate(tensor.shape):
        axis_name = f"{name}_dim{i}"
        for k, v in known_scalars.items():
            if isinstance(v, int) and v == size:
                axis_name = k
                break
        axes[axis_name] = size
    return axes


def _pick_tolerance(model_cls) -> tuple[float, float]:
    try:
        src = inspect.getsource(model_cls.forward)
    except (OSError, TypeError):
        return LOOSE_TOL          # can't inspect it -- assume the riskier default
    hit = next((m for m in _LOOSE_TOL_MARKERS if m in src), None)
    return (LOOSE_TOL, hit) if hit else (TIGHT_TOL, None)


def _shape_mode(mod) -> str:
    if hasattr(mod, "SHAPE_RANGES"):
        return "range"
    return "fixed"          # get_inputs()/get_init_inputs() already returned concrete values


def _range_points(ranges: dict) -> list[dict]:
    """lo / hi only, not a swept middle -- a plain, defensible default a reader can predict
    without needing this script's reasoning explained; ckh bench --axis is where a real sweep
    belongs, not this generator."""
    lo = {k: v[0] for k, v in ranges.items()}
    hi = {k: v[1] for k, v in ranges.items()}
    return [lo, hi] if lo != hi else [lo]


def generate(name: str) -> Path:
    mod = _load_pending(name)
    mode = _shape_mode(mod)

    # get_inputs()/get_init_inputs() existing (checked by _load_pending) doesn't mean they
    # actually produce anything usable -- a placeholder that raises, or returns something
    # that doesn't zip cleanly against Model's real signature, means this case needs judgment
    # the script shouldn't fake its way through. Refuse to the agent rather than crash or
    # (worse) silently generate a file that looks finished but isn't.
    try:
        init_args = mod.get_init_inputs()
        init_named = _named_args(mod.Model.__init__, init_args)
        fwd_args = mod.get_inputs()
        fwd_params = [p for p in inspect.signature(mod.Model.forward).parameters if p != "self"]
        if len(init_named) != len(init_args) or len(fwd_params) != len(fwd_args):
            raise ValueError("get_inputs()/get_init_inputs() returned a different number of "
                             "values than Model's forward/__init__ take positionally")
    except Exception as e:
        raise Refused(
            f"kernels/pending/{name}_pytorch.py's get_inputs()/get_init_inputs() exist but "
            f"couldn't be resolved against Model's real signature ({type(e).__name__}: {e}) "
            f"-- use the reference-generator agent instead."
        ) from e

    if mode == "fixed":
        axes_points = [dict(init_named)]
        for pname, val in zip(fwd_params, fwd_args):
            if hasattr(val, "shape"):
                axes_points[0].update(_tensor_axes(pname, val, init_named))
            else:
                axes_points[0][pname] = val
        axes_note = ("FIXED: kernels/pending/{n}_pytorch.py's get_inputs()/get_init_inputs() "
                    "already return concrete values; used as the one DEFAULT_AXES point, "
                    "unchanged.").format(n=name)
    elif mode == "range":
        axes_points = _range_points(mod.SHAPE_RANGES)
        axes_note = ("RANGE: SHAPE_RANGES gave {r}; sampled lo/hi only ({p}) -- not an "
                    "exhaustive sweep, use `ckh bench --axis` for that.").format(
                        r=mod.SHAPE_RANGES, p=axes_points)
    else:
        raise Refused("unreachable shape mode")

    tol, tol_hit = _pick_tolerance(mod.Model)
    if tol_hit:
        tol_note = (f"matched {tol_hit!r} in Model.forward's source -> real fp "
                   f"summation-order rounding expected.")
    else:
        tol_note = "no matmul/reduction found in Model.forward's source -> elementwise-only."

    out_path = Path(__file__).resolve().parents[2] / "kernels" / f"{name}.py"
    axes_keys = sorted(axes_points[0])

    # -- every piece of generated CODE (not prose) is built as a plain string first, then
    # joined with str.join/concatenation only -- no f-string/.format() touches a chunk that
    # itself contains braces, which is what made the first version of this function corrupt
    # its own placeholders.
    axes_sig = ", ".join(axes_keys)
    inputs_call_args = ", ".join(f"s.{k}" for k in axes_keys)
    # An init arg is either one of the swept axes (reference the parameter) or fixed outside
    # them (embed its literal value) -- true for FIXED mode, where axes_points started as
    # dict(init_named) so every init_named key IS an axis, and equally true for RANGE mode,
    # where axes_points only ever holds SHAPE_RANGES' keys, so an init arg not in SHAPE_RANGES
    # (e.g. a scalar like `scale` in a range-mode spec) is NOT a function parameter and must
    # be embedded as a literal or the generated code references an undefined name.
    init_call = ", ".join(
        f"{k}={k}" if k in axes_keys else f"{k}={v!r}" for k, v in init_named.items()
    )

    fwd_val_by_param = dict(zip(fwd_params, fwd_args))
    input_build_lines = []
    data_dict_entries = ['"model": model']
    for pname in fwd_params:
        val = fwd_val_by_param[pname]
        if hasattr(val, "shape"):
            dims = list(_tensor_axes(pname, val, init_named))
            input_build_lines.append(f"{pname} = torch.rand({', '.join(dims)})")
        # scalar forward args are already in scope under their own name via _inputs_cached's
        # signature (axes_sig includes every axis, scalar or tensor-dim-derived)
        data_dict_entries.append(f'"{pname}": {pname}')
    input_build = "\n    ".join(input_build_lines) if input_build_lines else ""
    data_dict_body = ", ".join(data_dict_entries)
    forward_call_args = ", ".join(f'data["{p}"]' for p in fwd_params)

    default_axes_lines = [
        f'    "{k}": {sorted({p[k] for p in axes_points})!r},' for k in axes_keys
    ]
    default_axes_src = "{\n" + "\n".join(default_axes_lines) + "\n}"

    header = (
        f'"""Generated by ckh.gen_reference from kernels/pending/{name}_pytorch.py.\n'
        f"Reference half only -- jit/dispatch/args/outputs/SPEC need a real CM kernel to "
        f"exist; that's\nkernel-onboarder's job, run after this one.\n\n"
        f"Shape axes: {axes_note}\n"
        f"Tolerance: {tol_note}\n"
        f'"""\n'
    )
    body = (
        "from __future__ import annotations\n\n"
        "import functools\n\n"
        "import torch\n\n"
        "from ckh.kernel import Shape\n"
        "from ckh.reference import TorchReference\n"
        f"from kernels.pending.{name}_pytorch import Model\n\n\n"
        "@functools.lru_cache(maxsize=8)\n"
        f"def _inputs_cached({axes_sig}):\n"
        '    """Deterministic per shape -- covers the Model instance too, not just its '
        "input\n"
        "    tensor(s): if Model.__init__ draws any weights from randomness, a fresh "
        "Model(...) built\n"
        "    inside compute() would re-randomize on every call. Building it once here, "
        "cached\n"
        '    alongside the input tensor(s), is what keeps this deterministic per shape."""\n'
        "    torch.manual_seed(0)\n"
        f"    model = Model({init_call})\n"
        + (f"    {input_build}\n" if input_build else "")
        + f"    return {{{data_dict_body}}}\n\n\n"
        "def inputs(s: Shape) -> dict:\n"
        f"    return _inputs_cached({inputs_call_args})\n\n\n"
        "def _compute(s: Shape, data: dict):\n"
        "    with torch.no_grad():\n"
        f'        return data["model"]({forward_call_args})\n\n\n'
        "REFERENCE = TorchReference(\n"
        "    compute=_compute,\n"
        f"    tol={tol!r},\n"
        "    # TODO(kernel-onboarder): no CM kernel exists yet to break and prove this "
        "check can fail\n"
        f"    # against -- fill this in for real (run a mutation, confirm `ckh equiv "
        f"{name}` reports\n"
        "    # FAIL) once one does. Do not replace this with a plausible-sounding "
        "claim.\n"
        '    non_vacuous="",\n'
        ")\n\n"
        f"DEFAULT_AXES = {default_axes_src}\n\n"
        "# jit/dispatch/args/outputs/SPEC come once a CM kernel exists for this --\n"
        "# see .claude/agents/kernel-onboarder.md\n"
    )

    out_path.write_text(header + body)
    return out_path


def _verify(name: str, sample_point: dict) -> None:
    """No GPU, no clops -- pure PyTorch. Run inline rather than leaving it as a suggested
    manual step, since it costs nothing and an un-runnable inputs()/compute is worse than no
    file at all (it looks finished)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    mod = importlib.import_module(f"kernels.{name}")
    importlib.reload(mod)
    from ckh.kernel import Shape
    s = Shape(sample_point)
    data = mod.inputs(s)
    shapes = {k: getattr(v, "shape", None) for k, v in data.items() if k != "model"}
    out = mod.REFERENCE.compute(s, data)
    print(f"verify: inputs {shapes}, compute() -> {tuple(out.shape)} {out.dtype}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m ckh.gen_reference <name>   "
             "(reads kernels/pending/<name>_pytorch.py)")
        return 2
    name = sys.argv[1]
    try:
        out_path = generate(name)
    except Refused as e:
        print(f"refusing to guess: {e}")
        return 2
    print(f"wrote {out_path}")

    mod = importlib.import_module(f"kernels.pending.{name}_pytorch")
    mode = _shape_mode(mod)
    sample = dict(_named_args(mod.Model.__init__, mod.get_init_inputs()))
    fwd_params = [p for p in inspect.signature(mod.Model.forward).parameters if p != "self"]
    for pname, val in zip(fwd_params, mod.get_inputs()):
        if hasattr(val, "shape"):
            sample.update(_tensor_axes(pname, val, sample))
        else:
            sample[pname] = val
    _verify(name, sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
