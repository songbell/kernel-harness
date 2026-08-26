"""Append-only findings store, so a later session does not re-test what was already settled.

This is the single largest token saver in the harness. In the work it came from, two ideas
were re-prototyped and re-measured after having already been rejected -- roughly four
measurement round-trips each -- purely because the earlier verdicts were not queryable.

A record says what was measured, on what, the numbers, and whether it was rejected on
*measurement* or on *policy*. That distinction matters: a measurement rejection can be
revisited when the rig or the kernel changes (a stale "+1 to +3%" verdict turned out to be a
rig artifact and the real effect was -4%), whereas a policy rejection needs the owner, not a
new measurement.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .platform import REPO_ROOT

LEDGER_DIR = REPO_ROOT / "ledger"
VERDICTS = ("adopted", "rejected-measurement", "rejected-policy", "invalid-probe", "open")


def path_for(kernel: str) -> Path:
    LEDGER_DIR.mkdir(exist_ok=True)
    return LEDGER_DIR / f"{kernel}.jsonl"


def add(kernel: str, *, change: str, verdict: str, shape: str, numbers: str,
        note: str = "", bit_exact: bool | None = None) -> None:
    if verdict not in VERDICTS:
        raise SystemExit(f"verdict must be one of {VERDICTS}")
    rec = {"ts": time.strftime("%Y-%m-%d"), "change": change, "verdict": verdict,
           "shape": shape, "numbers": numbers, "bit_exact": bit_exact, "note": note}
    with path_for(kernel).open("a") as f:
        f.write(json.dumps(rec) + "\n")


def load(kernel: str) -> list[dict]:
    p = path_for(kernel)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def render(kernel: str, only: str | None = None) -> str:
    recs = load(kernel)
    if only:
        recs = [r for r in recs if r["verdict"].startswith(only)]
    if not recs:
        return f"(no ledger entries for {kernel}{' with verdict ' + only if only else ''})"
    w = max(len(r["change"]) for r in recs)
    lines = [f"{'change':<{w}}  {'verdict':<21} {'numbers'}"]
    for r in recs:
        be = "" if r.get("bit_exact") is None else ("  [bit-exact]" if r["bit_exact"] else "  [rounds]")
        lines.append(f"{r['change']:<{w}}  {r['verdict']:<21} {r['numbers']}{be}")
        if r.get("note"):
            lines.append(f"{'':<{w}}  {'':<21} {r['note']}")
    return "\n".join(lines)
