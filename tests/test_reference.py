"""Pure-Python tests for the reference/equiv logic -- no GPU, no clops, runs anywhere.

The GPU-dependent half of this (does `ckh equiv` actually catch a real kernel bug) was
verified by hand against the real hardware while building this: forcing pa_small_q's torch
reference to use an unweighted mean instead of the logsumexp merge changed max_abs_diff from
2.5e-5 to 0.11 and flipped PASS to FAIL; forcing pa_small_q_vs_baseline's KernelReference to
see a wrong past_lens changed max_abs_diff from 4e-5 to 3e38. Recorded in the ledger. What
this file covers is the part that doesn't need a GPU: the diff/tolerance/rendering logic
itself, so a future change to it doesn't need real hardware to catch an obvious regression.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckh import equiv
from ckh.reference import EquivResult, KernelReference, TorchReference, _diff


def test_diff_bitexact_pass():
    a = np.array([1.0, 2.0, 3.0])
    r = _diff(a, a.copy(), bitexact=True, tol=None)
    assert r.passed
    assert r.max_abs_diff == 0.0


def test_diff_bitexact_fail_on_tiny_difference():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0 + 1e-9])
    r = _diff(a, b, bitexact=True, tol=None)
    assert not r.passed
    assert r.max_abs_diff > 0


def test_diff_tolerance_pass_within_bound():
    a = np.array([1.0, 2.0])
    b = np.array([1.0005, 2.0005])
    r = _diff(a, b, bitexact=False, tol=(1e-2, 1e-2))
    assert r.passed


def test_diff_tolerance_fail_outside_bound():
    a = np.array([1.0, 2.0])
    b = np.array([1.5, 2.0])
    r = _diff(a, b, bitexact=False, tol=(1e-2, 1e-2))
    assert not r.passed
    assert "EXCEEDED" in r.detail


def test_diff_shape_mismatch_fails_without_crashing():
    a = np.zeros((2, 3))
    b = np.zeros((2, 4))
    r = _diff(a, b, bitexact=False, tol=(1e-2, 1e-2))
    assert not r.passed
    assert "shape mismatch" in r.detail
    assert r.max_abs_diff is None


def test_diff_default_tol_used_when_none_given():
    a = np.array([1.0])
    b = np.array([1.0 + 5e-3])
    # Default tol is (1e-2, 1e-3); atol alone covers this.
    r = _diff(a, b, bitexact=False, tol=None)
    assert r.passed


class _Shape:
    """Stand-in for ckh.kernel.Shape -- these tests should not need clops or a real shape."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_torch_reference_without_combine_diffs_raw_output():
    ref = TorchReference(compute=lambda s, inputs: inputs["expected"])
    kernel_outputs = np.array([1.0, 2.0, 3.0])
    result = ref.evaluate(_Shape(), {"expected": kernel_outputs.copy()}, kernel_outputs)
    assert result.passed


def test_torch_reference_with_combine_reduces_first():
    def combine(s, outs):
        return outs["raw"].sum(axis=-1)

    ref = TorchReference(compute=lambda s, inputs: inputs["expected"], combine=combine,
                         tol=(1e-6, 1e-6))
    outs = {"raw": np.array([[1.0, 1.0], [2.0, 2.0]])}
    result = ref.evaluate(_Shape(), {"expected": np.array([2.0, 4.0])}, outs)
    assert result.passed


def test_kernel_reference_resolve_fills_unset_fields_from_spec():
    class _Spec:
        entry = "spec_entry"
        jit = "spec_jit"
        dispatch = "spec_dispatch"
        args = "spec_args"
        outputs = "spec_outputs"
        build_options = "spec_build_options"

    ref = KernelReference(source="baseline.cm")
    resolved = ref.resolve(_Spec())
    assert resolved["source"] == "baseline.cm"
    assert resolved["entry"] == "spec_entry"
    assert resolved["jit"] == "spec_jit"


def test_kernel_reference_resolve_keeps_explicit_overrides():
    class _Spec:
        entry = "spec_entry"
        jit = "spec_jit"
        dispatch = "spec_dispatch"
        args = "spec_args"
        outputs = "spec_outputs"
        build_options = "spec_build_options"

    my_jit = object()
    ref = KernelReference(source="baseline.cm", jit=my_jit)
    resolved = ref.resolve(_Spec())
    assert resolved["jit"] is my_jit
    assert resolved["entry"] == "spec_entry"      # unset field still falls through


def test_kernel_reference_evaluate_aggregates_multiple_buffers():
    ref = KernelReference(source="baseline.cm", tol=(1e-6, 1e-6))
    actual = {"a": np.array([1.0]), "b": np.array([2.0])}
    baseline_ok = {"a": np.array([1.0]), "b": np.array([2.0])}
    results = ref.evaluate(actual, baseline_ok, compare=["a", "b"])
    assert set(results) == {"a", "b"}
    assert all(r.passed for r in results.values())

    baseline_bad = {"a": np.array([1.0]), "b": np.array([999.0])}
    results_bad = ref.evaluate(actual, baseline_bad, compare=["a", "b"])
    assert results_bad["a"].passed
    assert not results_bad["b"].passed


def test_equiv_render_reports_pass_fail_counts():
    class _Spec:
        name = "fake"

    results = {
        "shape1": {"kind": "torch", "passed": True, "detail": "ok", "max_abs_diff": 1e-5,
                  "non_vacuous": "checked by hand"},
        "shape2": {"kind": "torch", "passed": False, "detail": "tol EXCEEDED",
                  "max_abs_diff": 0.5, "non_vacuous": "checked by hand"},
    }
    out = equiv.render(_Spec(), results)
    assert "1/2 shapes passed" in out
    assert "shape2" in out
    assert "FAILED" in out
    assert "WARNING" not in out          # non_vacuous was supplied on at least one shape


def test_equiv_render_warns_when_non_vacuous_is_never_supplied():
    class _Spec:
        name = "fake"

    results = {
        "shape1": {"kind": "torch", "passed": True, "detail": "ok", "max_abs_diff": 1e-5,
                  "non_vacuous": ""},
    }
    out = equiv.render(_Spec(), results)
    assert "WARNING" in out


def test_equiv_render_handles_errors_without_crashing():
    class _Spec:
        name = "fake"

    results = {"shape1": {"error": "RuntimeError: boom"}}
    out = equiv.render(_Spec(), results)
    assert "ERROR" in out
    assert "0/1 shapes passed" in out
