"""Coverage for `ckh validate`'s static half. No GPU, no clops.

The checks worth testing are the ones whose *absence* is silent. An indivisible gws/lws does
not raise -- it dispatches a different grid and every number after it describes something
other than what was intended. So each test here also asserts the check is capable of firing,
not merely that a good spec passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckh.kernel import KernelSpec, Shape  # noqa: E402
from ckh import validate  # noqa: E402


def _spec(**kw) -> KernelSpec:
    base = dict(
        name="t", source="k/t.cm", prod_source=None, entry="cm_t",
        jit=lambda s: {"A": 1},
        dispatch=lambda s: ([16, 4], [4, 1]),
        args=lambda s, d, o: [],
        inputs=lambda s: {},
        label_keys=["n"],
    )
    base.update(kw)
    return KernelSpec(**base)


S = Shape({"n": 1})


def _codes(problems) -> set[str]:
    return {p.code for p in problems}


# -- jit ------------------------------------------------------------------------------

def test_clean_jit_passes():
    assert validate.check_jit(_spec(), S) == []


def test_unset_jit_is_an_error():
    """`ckh kernelgen` leaves values as None on purpose; -DFOO=None is a valid C identifier
    in many contexts, so the compiler is not guaranteed to catch it."""
    probs = validate.check_jit(_spec(jit=lambda s: {"A": 1, "B": None, "C": ""}), S)
    assert _codes(probs) == {"jit-unset"}
    assert len(probs) == 2


def test_non_scalar_jit_is_an_error():
    probs = validate.check_jit(_spec(jit=lambda s: {"A": [1, 2]}), S)
    assert _codes(probs) == {"jit-not-scalar"}


def test_raising_jit_is_reported_not_propagated():
    def boom(s):
        raise ValueError("nope")
    assert _codes(validate.check_jit(_spec(jit=boom), S)) == {"jit-raised"}


# -- dispatch -------------------------------------------------------------------------

def test_clean_dispatch_passes():
    assert validate.check_dispatch(_spec(), S) == []


def test_indivisible_gws_is_an_error():
    probs = validate.check_dispatch(_spec(dispatch=lambda s: ([15, 4], [4, 1])), S)
    assert _codes(probs) == {"dispatch-indivisible"}


def test_rank_mismatch_is_an_error():
    probs = validate.check_dispatch(_spec(dispatch=lambda s: ([16, 4], [4])), S)
    assert _codes(probs) == {"dispatch-rank"}


def test_more_than_three_dimensions_is_an_error():
    probs = validate.check_dispatch(_spec(dispatch=lambda s: ([2] * 4, [1] * 4)), S)
    assert "dispatch-rank" in _codes(probs)


def test_nonpositive_dispatch_is_an_error():
    probs = validate.check_dispatch(_spec(dispatch=lambda s: ([0, 4], [1, 1])), S)
    assert "dispatch-nonpositive" in _codes(probs)


def test_kernelgen_scaffold_is_named_as_such():
    def todo(s):
        raise NotImplementedError("dispatch: translate the host expressions")
    assert _codes(validate.check_dispatch(_spec(dispatch=todo), S)) == {"dispatch-todo"}


def test_all_ones_lws_warns_but_does_not_block():
    probs = validate.check_dispatch(_spec(dispatch=lambda s: ([16, 4], [1, 1])), S)
    assert _codes(probs) == {"lws-all-ones"}
    assert [p.severity for p in probs] == ["warn"]
    _, errors = validate.render({"x": probs}, None)
    assert errors == 0


def test_one_dimensional_all_ones_lws_does_not_warn():
    assert validate.check_dispatch(_spec(dispatch=lambda s: ([16], [1])), S) == []


# -- spec-declared constraints --------------------------------------------------------

def dpas_repeat_count(s: Shape) -> str | None:
    if s.values["chunk"] * s.values["tile_q"] > 8:
        return f"chunk*tile_q = {s.values['chunk'] * s.values['tile_q']} > 8"
    return None


def test_constraint_fires_and_names_itself():
    spec = _spec(constraints=[dpas_repeat_count])
    assert validate.check_constraints(spec, Shape({"chunk": 4, "tile_q": 2})) == []
    probs = validate.check_constraints(spec, Shape({"chunk": 4, "tile_q": 4}))
    assert _codes(probs) == {"constraint-violated"}
    assert "dpas_repeat_count" in probs[0].detail


def test_a_spec_with_no_constraints_is_fine():
    assert validate.check_constraints(_spec(), S) == []


def test_raising_constraint_is_reported():
    def boom(s):
        raise KeyError("missing_axis")
    assert _codes(validate.check_constraints(_spec(constraints=[boom]), S)) \
        == {"constraint-raised"}


# -- rendering ------------------------------------------------------------------------

def test_render_blocks_on_errors_and_says_why():
    probs = validate.check_dispatch(_spec(dispatch=lambda s: ([15, 4], [4, 1])), S)
    table, errors = validate.render({"n=1": probs}, None)
    assert errors == 1
    assert "do NOT measure this yet" in table


def test_render_marks_a_static_only_run():
    table, _ = validate.render({"n=1": []}, None)
    assert "static checks only" in table
    assert "n=1  OK" in table


def test_render_merges_env_problems_into_the_same_shape():
    env = {"shapes": {"n=1": [{"severity": "error", "code": "compile-failed", "detail": "boom"}]}}
    table, errors = validate.render({"n=1": []}, env)
    assert errors == 1
    assert "compile-failed" in table
    assert "static checks only" not in table


def test_static_checks_covers_every_shape():
    spec = _spec()
    out = validate.static_checks(spec, [{"n": 1}, {"n": 2}])
    assert set(out) == {"n=1", "n=2"}


# -- the in-env half's pure helpers ---------------------------------------------------

def test_device_limit_check_fires_on_an_oversized_group():
    from ckh import validate_runner as vr
    limits = {"CL_DEVICE_MAX_WORK_GROUP_SIZE": 512, "CL_DEVICE_MAX_WORK_ITEM_SIZES": [512, 512, 512]}
    assert vr._check_device(limits, [8, 8, 8]) == []
    probs = vr._check_device(limits, [16, 16, 8])
    assert [p["code"] for p in probs] == ["lws-too-large"]
    assert [p["code"] for p in vr._check_device(limits, [1024, 1, 1])] \
        == ["lws-too-large", "lws-dim-too-large"]


def test_device_limit_check_is_silent_when_the_device_says_nothing():
    """An unavailable device query must not manufacture a passing verdict OR a failing one."""
    from ckh import validate_runner as vr
    assert vr._check_device({}, [1024, 1024, 1024]) == []
