"""`ckh trial` -- a branching trial tree, with this project's gates kept in the loop.

The shape is the familiar one: generate a candidate, validate it, benchmark it, use the result
to choose the next candidate, finalize the best. Two things are different here, and both come
from failures recorded in this repo's ledger.

**A stored millisecond is not a comparable millisecond.** The reference box drifts ~2x within
a session; the same config measured 0.52 ms early and 1.59 ms late. So a tree that picks its
winner by sorting recorded numbers picks the trial that ran when the box was coldest. Recorded
deltas here are used ONLY to shortlist. `finalize` re-measures the shortlist and the baseline
together, in one interleaved batch, and the winner is decided on that.

**A trial is not a result until it passes the gates.** Each trial runs validate -> equiv ->
bench, in that order and short-circuiting, so a candidate that cannot compile never reaches
the GPU and a candidate that computes the wrong answer never produces a timing number worth
quoting. Ordering matters for cost too: validate is free, equiv is cheap, bench is the
expensive one.

The tree lives in `trials/<kernel>/tree.json` and every node pins its own source snapshot, so
any node can be re-measured later. That is what makes an honest finalize possible at all.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import bench, results
from .platform import REPO_ROOT

TRIALS = REPO_ROOT / "trials"

# A node's status after the gates have had their say.
OPEN = "open"                # created, not yet gated
INVALID = "invalid"          # validate failed -- no GPU time was spent
INCORRECT = "incorrect"      # equiv failed -- timing would be meaningless
IMPROVED = "improved"
REGRESSED = "regressed"
INCONCLUSIVE = "inconclusive"   # measured, but below the rig's resolution
# Only these are worth branching from; the rest are recorded so they are not re-tried.
BRANCHABLE = (IMPROVED, INCONCLUSIVE)


class TrialError(Exception):
    """Something the caller has to decide, not something to paper over."""


@dataclass
class Node:
    id: int
    parent: int | None
    label: str
    tag: str                       # the snapshot pinning this node's source
    status: str = OPEN
    gates: dict = field(default_factory=dict)
    ts: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(**d)

    def as_dict(self) -> dict:
        return {"id": self.id, "parent": self.parent, "label": self.label, "tag": self.tag,
                "status": self.status, "gates": self.gates, "ts": self.ts}


@dataclass
class Tree:
    kernel: str
    max_trials: int
    nodes: list[Node] = field(default_factory=list)

    # -- persistence ---------------------------------------------------------------
    @staticmethod
    def path(kernel: str) -> Path:
        return TRIALS / kernel / "tree.json"

    @classmethod
    def load(cls, kernel: str) -> "Tree":
        p = cls.path(kernel)
        if not p.exists():
            raise TrialError(f"no trial tree for {kernel}. `ckh trial init {kernel}` first.")
        d = json.loads(p.read_text())
        return cls(kernel=d["kernel"], max_trials=d["max_trials"],
                   nodes=[Node.from_dict(n) for n in d["nodes"]])

    def save(self) -> None:
        p = self.path(self.kernel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"kernel": self.kernel, "max_trials": self.max_trials,
                                 "nodes": [n.as_dict() for n in self.nodes]}, indent=2))

    # -- structure -----------------------------------------------------------------
    def get(self, node_id: int) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise TrialError(f"no trial {node_id} in {self.kernel}'s tree")

    @property
    def root(self) -> Node:
        return self.nodes[0]

    @property
    def used(self) -> int:
        """Root is the baseline, not a trial, so it does not count against the budget."""
        return len(self.nodes) - 1

    def children(self, node_id: int) -> list[Node]:
        return [n for n in self.nodes if n.parent == node_id]

    def add(self, parent: int | None, label: str, tag: str) -> Node:
        if parent is not None:
            p = self.get(parent)
            if p.status not in BRANCHABLE and p.id != self.root.id:
                raise TrialError(
                    f"trial {parent} is {p.status}; branching from it would build on a "
                    f"candidate already shown not to work. Branch from a {'/'.join(BRANCHABLE)} "
                    f"node, or the baseline (0).")
        if self.used >= self.max_trials and parent is not None:
            raise TrialError(
                f"{self.kernel} has used its {self.max_trials}-trial budget. Raise it with "
                f"--max-trials, or `ckh trial finalize` and start a fresh tree -- an "
                f"open-ended loop is how a week goes into a 2%-of-phase kernel.")
        node = Node(id=len(self.nodes), parent=parent, label=label, tag=tag,
                    ts=time.strftime("%Y-%m-%d %H:%M"))
        self.nodes.append(node)
        return node


# -- the loop body ---------------------------------------------------------------------

def classify(delta_pct: float, resolution_pct: float) -> str:
    """A delta smaller than what the rig can resolve is INCONCLUSIVE, never "no change".

    Reporting it as the latter is how a real 5% regression gets waved through -- observed
    while testing `render_round` on a drifting host.
    """
    if abs(delta_pct) <= resolution_pct:
        return INCONCLUSIVE
    return IMPROVED if delta_pct < 0 else REGRESSED


def run_gates(tree: Tree, node: Node, *, validate_fn, equiv_fn, bench_fn,
              noise_floor_pct: float) -> Node:
    """validate -> equiv -> bench, short-circuiting. Each gate's raw verdict is stored.

    The gate callables are injected so this ordering is testable without a GPU; the CLI wires
    them to the real commands.
    """
    errors, detail = validate_fn(node)
    node.gates["validate"] = {"errors": errors, "detail": detail}
    if errors:
        node.status = INVALID
        return node

    passed, eq_detail, non_vacuous = equiv_fn(node)
    node.gates["equiv"] = {"passed": passed, "detail": eq_detail,
                           "non_vacuous": non_vacuous}
    if not passed:
        node.status = INCORRECT
        return node
    if not non_vacuous:
        # Not fatal, but it must not be silent: a passing check with no documented reason to
        # believe it can fail is the failure mode that looks exactly like success.
        node.gates["equiv"]["warning"] = (
            "reference has no non_vacuous note -- this PASS is not evidence")

    baseline = tree.get(node.parent if node.parent is not None else 0)
    delta_pct, spread_pct, raw = bench_fn(node, baseline)
    resolution = max(spread_pct, noise_floor_pct)
    node.gates["bench"] = {"against": baseline.tag, "delta_pct": delta_pct,
                           "resolution_pct": resolution, "results": raw}
    node.status = classify(delta_pct, resolution)
    return node


# -- selection -------------------------------------------------------------------------

def shortlist(tree: Tree, k: int) -> list[Node]:
    """Best-looking candidates by RECORDED delta -- for selection only.

    These numbers came from different moments on a box that drifts, so their ordering is a
    hint about which nodes deserve a fresh measurement, not a ranking. `finalize` is what
    turns the hint into a verdict.
    """
    scored = [n for n in tree.nodes if n.status == IMPROVED]
    scored.sort(key=lambda n: n.gates["bench"]["delta_pct"])
    return scored[:k]


def finalize_configs(tree: Tree, candidates: list[Node]) -> dict[str, str]:
    """label -> snapshot tag, baseline included. One interleaved batch, one moment in time."""
    cfg = {tree.root.tag: tree.root.tag}
    for n in candidates:
        cfg[f"trial{n.id}"] = n.tag
    return cfg


def finalize_verdict(measured: dict, baseline_label: str,
                     noise_floor_pct: float) -> tuple[str | None, str]:
    """Pick the winner from a FRESH interleaved measurement. Returns (winner_label, reason)."""
    ok = {k: v for k, v in measured.items() if "error" not in v}
    if baseline_label not in ok:
        return None, f"the baseline ({baseline_label}) failed to re-measure; nothing to compare against"
    base = ok[baseline_label]["min_ms"]

    ranked = sorted(((k, v) for k, v in ok.items() if k != baseline_label),
                    key=lambda kv: kv[1]["min_ms"])
    if not ranked:
        return None, "no candidate re-measured successfully"

    label, v = ranked[0]
    delta = (v["min_ms"] / base - 1.0) * 100.0
    resolution = max(v["spread_pct"], ok[baseline_label]["spread_pct"], noise_floor_pct)
    if abs(delta) <= resolution:
        return None, (f"best candidate {label} is {delta:+.1f}% vs baseline, inside the "
                      f"{resolution:.1f}% the rig can resolve -- NOT a win. Stabilise the rig "
                      f"or keep the baseline; do not ship an unresolvable difference")
    if delta > 0:
        return None, (f"every candidate re-measured slower than the baseline (best {label} "
                      f"{delta:+.1f}%). The recorded per-trial deltas disagreed with this, "
                      f"which is what re-measuring exists to catch")
    return label, f"{label} is {delta:+.1f}% vs baseline, resolution {resolution:.1f}%"


# -- rendering -------------------------------------------------------------------------

_MARK = {OPEN: "  ", INVALID: "xx", INCORRECT: "!!", IMPROVED: "++",
         REGRESSED: "--", INCONCLUSIVE: "??"}


def render(tree: Tree) -> str:
    lines = [f"{tree.kernel}: {tree.used}/{tree.max_trials} trials used"]

    def walk(node: Node, depth: int) -> None:
        pad = "  " * depth
        g = node.gates.get("bench")
        num = ""
        if g:
            num = f"  {g['delta_pct']:+6.1f}% vs {g['against']} (res {g['resolution_pct']:.1f}%)"
        elif node.status == INVALID:
            num = f"  {node.gates['validate']['errors']} validation error(s)"
        elif node.status == INCORRECT:
            num = f"  {node.gates['equiv']['detail']}"
        lines.append(f"{_MARK.get(node.status, '  ')} {pad}[{node.id}] {node.label:<28}"
                     f"{node.status:<13}{num}")
        if node.gates.get("equiv", {}).get("warning"):
            lines.append(f"        {pad}    WARNING: {node.gates['equiv']['warning']}")
        for c in tree.children(node.id):
            walk(c, depth + 1)

    walk(tree.root, 0)
    lines.append("\nRecorded deltas were measured at different moments on a rig that drifts; "
                 "they shortlist,\nthey do not rank. `ckh trial finalize` re-measures the "
                 "shortlist against the baseline in one\ninterleaved batch and decides there.")
    return "\n".join(lines)


def snapshot_current(plat, spec, kernel: str, node_id: int) -> str:
    """Pin this node's source so it can be re-measured in a later session."""
    tag = f"trial{node_id}"
    results.snapshot(plat, spec, tag)
    return tag


def expand_shapes(spec, default_axes: dict, overrides: list[str] | None) -> list[dict]:
    axes = dict(default_axes)
    for kv in overrides or []:
        k, _, v = kv.partition("=")
        axes[k] = [int(x) if x.lstrip("-").isdigit() else x for x in v.split(",")]
    return bench.expand(axes)
