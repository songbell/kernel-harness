"""Coverage for the trial loop's logic. No GPU: the three gates are injected.

The two properties worth guarding are the ones that make this different from a plain
generate-and-measure loop:

  1. gate ORDER and short-circuiting -- an invalid candidate must never reach the GPU, and an
     incorrect one must never produce a timing number.
  2. finalize decides on a FRESH interleaved measurement, not on stored per-trial deltas. The
     reference box drifts ~2x within a session, so a tree that ranks by stored numbers picks
     whichever trial ran when the box was coldest.

Test 2 is asserted with a case where the stored deltas and the re-measurement disagree, which
is the only way to show the re-measurement is load-bearing rather than decorative.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckh import trial  # noqa: E402


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, "TRIALS", tmp_path / "trials")
    t = trial.Tree(kernel="k", max_trials=3)
    t.add(None, "baseline", "trial0")
    return t


def _gates(*, errors=0, passed=True, non_vacuous="checked", delta=-5.0, spread=1.0):
    calls = []

    def v(node):
        calls.append("validate")
        return errors, "detail"

    def e(node):
        calls.append("equiv")
        return passed, "detail", non_vacuous

    def b(node, baseline):
        calls.append("bench")
        return delta, spread, {}

    return (v, e, b), calls


# -- gate ordering ---------------------------------------------------------------------

def test_invalid_candidate_never_reaches_the_gpu(tree):
    node = tree.add(0, "bad", "trial1")
    fns, calls = _gates(errors=3)
    trial.run_gates(tree, node, validate_fn=fns[0], equiv_fn=fns[1], bench_fn=fns[2],
                    noise_floor_pct=2.0)
    assert node.status == trial.INVALID
    assert calls == ["validate"]                 # equiv and bench were never called
    assert "bench" not in node.gates


def test_incorrect_candidate_produces_no_timing_number(tree):
    node = tree.add(0, "wrong", "trial1")
    fns, calls = _gates(passed=False)
    trial.run_gates(tree, node, validate_fn=fns[0], equiv_fn=fns[1], bench_fn=fns[2],
                    noise_floor_pct=2.0)
    assert node.status == trial.INCORRECT
    assert calls == ["validate", "equiv"]
    assert "bench" not in node.gates


def test_full_pass_runs_all_three_in_cost_order(tree):
    node = tree.add(0, "good", "trial1")
    fns, calls = _gates()
    trial.run_gates(tree, node, validate_fn=fns[0], equiv_fn=fns[1], bench_fn=fns[2],
                    noise_floor_pct=2.0)
    assert calls == ["validate", "equiv", "bench"]
    assert node.status == trial.IMPROVED


def test_missing_non_vacuous_warns_without_blocking(tree):
    node = tree.add(0, "unproven", "trial1")
    fns, _ = _gates(non_vacuous="")
    trial.run_gates(tree, node, validate_fn=fns[0], equiv_fn=fns[1], bench_fn=fns[2],
                    noise_floor_pct=2.0)
    assert node.status == trial.IMPROVED
    assert "not evidence" in node.gates["equiv"]["warning"]
    assert "WARNING" in trial.render(tree)


# -- classification --------------------------------------------------------------------

def test_delta_below_resolution_is_inconclusive_not_no_change():
    assert trial.classify(-1.0, 2.0) == trial.INCONCLUSIVE
    assert trial.classify(1.5, 2.0) == trial.INCONCLUSIVE
    assert trial.classify(-5.0, 2.0) == trial.IMPROVED
    assert trial.classify(5.0, 2.0) == trial.REGRESSED


def test_noise_floor_is_a_lower_bound_on_resolution(tree):
    """A quiet rig does not license resolving below the configured floor."""
    node = tree.add(0, "tiny", "trial1")
    fns, _ = _gates(delta=-1.0, spread=0.1)
    trial.run_gates(tree, node, validate_fn=fns[0], equiv_fn=fns[1], bench_fn=fns[2],
                    noise_floor_pct=2.0)
    assert node.gates["bench"]["resolution_pct"] == 2.0
    assert node.status == trial.INCONCLUSIVE


# -- tree discipline -------------------------------------------------------------------

def test_cannot_branch_from_a_dead_candidate(tree):
    node = tree.add(0, "bad", "trial1")
    node.status = trial.REGRESSED
    with pytest.raises(trial.TrialError, match="already shown not to work"):
        tree.add(node.id, "child", "trial2")


def test_can_branch_from_inconclusive(tree):
    node = tree.add(0, "meh", "trial1")
    node.status = trial.INCONCLUSIVE
    assert tree.add(node.id, "child", "trial2").parent == node.id


def test_budget_is_enforced_and_excludes_the_baseline(tree):
    for i in range(3):
        tree.add(0, f"t{i}", f"trial{i + 1}")
    assert tree.used == 3
    with pytest.raises(trial.TrialError, match="trial budget"):
        tree.add(0, "one too many", "trial4")


def test_tree_round_trips_through_disk(tree):
    node = tree.add(0, "x", "trial1")
    node.status = trial.IMPROVED
    node.gates = {"bench": {"delta_pct": -3.0, "against": "trial0", "resolution_pct": 2.0}}
    tree.save()
    again = trial.Tree.load("k")
    assert again.get(1).status == trial.IMPROVED
    assert again.get(1).gates["bench"]["delta_pct"] == -3.0


def test_loading_a_missing_tree_says_what_to_run(tree):
    with pytest.raises(trial.TrialError, match="ckh trial init"):
        trial.Tree.load("never_initialised")


# -- selection and finalize ------------------------------------------------------------

def _improved(tree, ident: int, delta: float):
    n = tree.add(0, f"cand{ident}", f"trial{ident}")
    n.status = trial.IMPROVED
    n.gates = {"bench": {"delta_pct": delta, "against": "trial0", "resolution_pct": 2.0}}
    return n


def test_shortlist_orders_by_recorded_delta_and_skips_dead_nodes(tree):
    tree.max_trials = 10
    _improved(tree, 1, -3.0)
    _improved(tree, 2, -9.0)
    bad = tree.add(0, "bad", "trial3")
    bad.status = trial.REGRESSED
    assert [n.id for n in trial.shortlist(tree, 3)] == [2, 1]


def test_finalize_configs_always_include_the_baseline(tree):
    tree.max_trials = 10
    cfg = trial.finalize_configs(tree, [_improved(tree, 1, -3.0)])
    assert cfg == {"trial0": "trial0", "trial1": "trial1"}


def test_finalize_can_contradict_the_recorded_deltas():
    """The point of re-measuring. Both trials recorded a win; fresh interleaved measurement
    says both are slower, and the tool must report that rather than crown one."""
    measured = {
        "trial0": {"min_ms": 1.00, "spread_pct": 0.5},
        "trial1": {"min_ms": 1.08, "spread_pct": 0.5},
        "trial2": {"min_ms": 1.20, "spread_pct": 0.5},
    }
    winner, reason = trial.finalize_verdict(measured, "trial0", 2.0)
    assert winner is None
    assert "slower than the baseline" in reason


def test_finalize_refuses_an_unresolvable_win():
    measured = {"trial0": {"min_ms": 1.000, "spread_pct": 0.5},
                "trial1": {"min_ms": 0.990, "spread_pct": 0.5}}
    winner, reason = trial.finalize_verdict(measured, "trial0", 2.0)
    assert winner is None
    assert "NOT a win" in reason


def test_finalize_crowns_a_resolvable_win():
    measured = {"trial0": {"min_ms": 1.00, "spread_pct": 0.5},
                "trial1": {"min_ms": 0.85, "spread_pct": 0.5}}
    winner, reason = trial.finalize_verdict(measured, "trial0", 2.0)
    assert winner == "trial1"
    assert "-15.0%" in reason


def test_finalize_refuses_when_the_baseline_did_not_remeasure():
    measured = {"trial0": {"error": "compile failed"}, "trial1": {"min_ms": 0.5, "spread_pct": 0.5}}
    winner, reason = trial.finalize_verdict(measured, "trial0", 2.0)
    assert winner is None
    assert "baseline" in reason


def test_render_states_that_stored_deltas_do_not_rank(tree):
    tree.max_trials = 10
    _improved(tree, 1, -3.0)
    out = trial.render(tree)
    assert "they shortlist" in out and "do not rank" in out
    assert "1/10 trials used" in out


# -- per-shape scoring -----------------------------------------------------------------

GRID = {
    "q=6 | base": {"min_ms": 1.00, "spread_pct": 0.5},
    "q=6 | cand": {"min_ms": 0.80, "spread_pct": 0.5},
    "q=16 | base": {"min_ms": 1.00, "spread_pct": 0.5},
    "q=16 | cand": {"min_ms": 1.30, "spread_pct": 0.5},
}


def test_a_candidate_that_helps_one_shape_and_breaks_another_is_a_regression():
    """Scoring the grid by its best row calls this an improvement. It happened twice for real
    -- a partition tuned at long context regressed short context by 3.3x."""
    from ckh import bench

    deltas = bench.round_deltas(GRID, "base", "cand", 2.0)
    assert sorted(round(d, 1) for d, _ in deltas.values()) == [-20.0, 30.0]
    worst = max(deltas.values(), key=lambda d: d[0])
    assert trial.classify(*worst) == trial.REGRESSED
    # The collapse this replaced would have scored it on the best shape:
    best = min(deltas.values(), key=lambda d: d[0])
    assert trial.classify(*best) == trial.IMPROVED


def test_round_deltas_and_render_round_agree_on_pairing():
    """They used to duplicate the pairing logic; a divergence would silently change what the
    trial gate scores relative to what `ckh round` prints."""
    from ckh import bench

    _, worse = bench.render_round(GRID, "base", "cand", 2.0)
    deltas = bench.round_deltas(GRID, "base", "cand", 2.0)
    assert worse == sum(1 for d, res in deltas.values() if d > res)


def test_pairing_drops_shapes_measured_on_only_one_side():
    from ckh import bench

    partial = dict(GRID)
    partial["q=32 | cand"] = {"min_ms": 0.1, "spread_pct": 0.5}   # no baseline for this shape
    assert set(bench.pair_by_shape(partial, "base", "cand")) == {"q=6", "q=16"}


def test_pairing_drops_errored_configs():
    from ckh import bench

    errored = dict(GRID)
    errored["q=6 | cand"] = {"error": "compile failed"}
    assert set(bench.pair_by_shape(errored, "base", "cand")) == {"q=16"}
