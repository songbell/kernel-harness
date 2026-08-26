"""KernelSpec -- everything that is kernel-specific, in one place.

In the work this came from, five things differed per kernel and were re-derived by grepping a
1100-line source roughly fifteen times: the signature, the jit defines, the shape parameters,
the dispatch computation, and input generation. Describing a kernel once makes bench, equiv,
ablate and sweep generic over it, so adopting a new kernel means writing a descriptor rather
than a script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Shape:
    """One concrete measurement point. Free-form so a spec can add its own axes."""
    values: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, k: str) -> Any:      # shape.q_len instead of shape.values["q_len"]
        try:
            return self.values[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def merged(self, **kw) -> "Shape":
        return Shape({**self.values, **kw})

    def label(self, keys: list[str] | None = None) -> str:
        ks = keys or sorted(self.values)
        return " ".join(f"{k}={self.values[k]}" for k in ks if k in self.values)


@dataclass
class KernelSpec:
    name: str
    # Source under development, relative to platform.repos.sandbox.
    source: str
    # The shipped copy, relative to platform.repos.production. Used by the integrator to diff
    # code-only: the two are NOT the same file, and a plugin-only code path was once 17% of
    # the kernel while being entirely absent from the sandbox copy.
    prod_source: str | None
    entry: str

    # shape -> {DEFINE: value}. Anything a change might flip belongs here, not in the source.
    jit: Callable[[Shape], dict[str, Any]]
    # shape -> (gws, lws)
    dispatch: Callable[[Shape], tuple[list[int], list[int]]]
    # shape -> list of kernel arguments, already in signature order. Buffers must come from
    # `inputs` so that both sides of an A/B see identical data.
    args: Callable[[Shape, dict], list[Any]]
    # shape -> dict of host tensors. MUST be deterministic for a given shape: a generator that
    # re-randomises per call once turned a bit-exact change into 0/96 spurious mismatches.
    inputs: Callable[[Shape], dict]
    # Buffers to read back and compare in an equivalence check, by name in the `inputs` dict.
    compare: list[str] = field(default_factory=list)

    build_options: Callable[[Shape], str] | None = None
    # Axes shown in table labels, in order.
    label_keys: list[str] = field(default_factory=list)

    def defines(self, shape: Shape, overrides: dict[str, Any] | None = None) -> str:
        d = dict(self.jit(shape))
        d.update(overrides or {})
        return " ".join(f"-D{k}={v}" for k, v in d.items())
