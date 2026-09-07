"""Reference-implementation abstraction: what "correct" means for a kernel, expressed once.

Two kinds, both first-class, selected by which one a KernelSpec.reference is:

  TorchReference  -- ground truth computed independently in Python/torch from the same
                     Shape + inputs() dict the kernel itself sees. The strongest kind, because
                     it does not trust any other kernel's output.
  KernelReference -- another compiled kernel, run the same way as the kernel under test, used
                     as ground truth instead. Weaker (it can only prove agreement, not
                     correctness against first principles) but is what most of this project's
                     actual "is the new kernel right" work relied on before this existed
                     (pa_small_q_ov.cm as baseline vs pa_small_q_ov_exp.cm under test).

Both produce an EquivResult per comparison; ckh.equiv is the only thing that should construct
or consume these directly, everything else goes through KernelSpec.reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class EquivResult:
    passed: bool
    detail: str
    max_abs_diff: float | None = None


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):          # torch.Tensor
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _diff(actual: Any, expected: Any, bitexact: bool, tol: tuple[float, float] | None) -> EquivResult:
    a = _to_numpy(actual).astype(np.float64, copy=False)
    e = _to_numpy(expected).astype(np.float64, copy=False)
    if a.shape != e.shape:
        return EquivResult(False, f"shape mismatch: actual={a.shape} expected={e.shape}", None)
    max_abs = float(np.abs(a - e).max()) if a.size else 0.0
    if bitexact:
        ok = np.array_equal(a, e)
        return EquivResult(ok, "bit-exact" if ok else "NOT bit-exact", max_abs)
    atol, rtol = tol or (1e-2, 1e-3)
    ok = bool(np.allclose(a, e, atol=atol, rtol=rtol))
    detail = f"tol atol={atol} rtol={rtol}" + ("" if ok else " EXCEEDED")
    return EquivResult(ok, detail, max_abs)


@dataclass
class TorchReference:
    """Ground truth computed independently from the same inputs the kernel sees.

    For a multi-stage kernel -- pa_small_q's main entry only produces partial
    partition_out/lse; a separate reduce *kernel* combines them -- `combine` must do that
    combination itself, in torch, independent of the reduce kernel. Comparing against
    "the reduce kernel's own reimplementation of itself" would prove nothing about either
    kernel; comparing against an independent torch merge proves both.

    compute(shape, inputs) -> expected           the true answer, from first principles
    combine(shape, kernel_outputs) -> actual      optional; reduces raw kernel buffers
                                                   (named per KernelSpec.compare) into
                                                   something comparable to `expected`.
                                                   Omit if the kernel's own output buffer is
                                                   already directly comparable.
    """
    compute: Callable[[Any, dict], Any]
    combine: Callable[[Any, dict[str, Any]], Any] | None = None
    bitexact: bool = False
    tol: tuple[float, float] = (1e-2, 1e-3)
    # Free-text: why is this comparison known to be capable of failing? Not an executable
    # check -- a documented claim the spec author is answerable for, in the same spirit as
    # the ledger's free-text `note` field. `ckh equiv` warns (does not block) if empty.
    non_vacuous: str = ""

    def evaluate(self, shape: Any, inputs: dict, kernel_outputs: dict[str, Any]) -> EquivResult:
        expected = self.compute(shape, inputs)
        actual = self.combine(shape, kernel_outputs) if self.combine else kernel_outputs
        return _diff(actual, expected, self.bitexact, self.tol)


@dataclass
class KernelReference:
    """Another compiled kernel, run the same way as the kernel under test, as ground truth.

    Any field left None reuses the kernel-under-test's own KernelSpec callable -- covers the
    common case (two variants of the same kernel, differing in one targeted change, like
    equiv_template.py's compile-time-vs-runtime-partition A/B) without forcing a second,
    fully independent descriptor. A baseline with a genuinely different dispatch contract
    (different signature, different TILE_Q convention, ...) should override every field --
    see kernels/pa_small_q_vs_baseline.py, where NONE of the defaults apply.

    bitexact defaults to False (tolerance-based): the common real case is comparing two
    genuinely different implementations (different accumulation order), where exact equality
    is not expected. Set bitexact=True only when comparing two variants of the same
    algorithm, where the accumulation order is provably unchanged.
    """
    source: str            # relative to platform.sandbox, same convention as KernelSpec.source
                            # -- NOT relative to the measurement subprocess's cwd.
    entry: str | None = None
    jit: Callable[[Any], dict] | None = None
    dispatch: Callable[[Any], tuple] | None = None
    args: Callable[[Any, dict, dict], list] | None = None
    outputs: Callable[[Any], dict] | None = None
    build_options: Callable[[Any], str] | None = None
    bitexact: bool = False
    tol: tuple[float, float] | None = (1e-2, 1e-3)
    non_vacuous: str = ""

    def resolve(self, spec: Any) -> dict:
        """Fill in unset fields from the kernel-under-test's own KernelSpec."""
        return dict(
            source=self.source,
            entry=self.entry or spec.entry,
            jit=self.jit or spec.jit,
            dispatch=self.dispatch or spec.dispatch,
            args=self.args or spec.args,
            outputs=self.outputs or spec.outputs,
            build_options=self.build_options or spec.build_options,
        )

    def evaluate(self, actual_outputs: dict[str, Any], baseline_outputs: dict[str, Any],
                 compare: list[str]) -> dict[str, EquivResult]:
        return {name: _diff(actual_outputs[name], baseline_outputs[name], self.bitexact, self.tol)
                for name in compare}


Reference = TorchReference | KernelReference
