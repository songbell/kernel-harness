"""Run store and pinned source snapshots -- the two things iteration needs.

Optimization is not one pass. Round N has to answer "did this help?" and "did it break
anything else?", and neither can be answered by comparing against a number written down in
round N-1: on a rig that drifts, cross-session absolute values are not comparable. The only
valid comparison is the previous *source*, re-measured in the same interleaved run. Hence
snapshots.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .platform import REPO_ROOT, Platform

RESULTS = REPO_ROOT / "results"
SNAPSHOTS = REPO_ROOT / "snapshots"


# -- pinned sources ------------------------------------------------------------------

def snapshot(plat: Platform, spec, tag: str) -> Path:
    """Freeze the current kernel source so a later round can A/B against it."""
    src = plat.sandbox / spec.source
    dst = SNAPSHOTS / spec.name
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / f"{tag}{Path(spec.source).suffix}"
    shutil.copy2(src, out)
    (dst / f"{tag}.meta.json").write_text(json.dumps(
        {"tag": tag, "ts": time.strftime("%Y-%m-%d %H:%M"), "source": str(src)}, indent=2))
    return out


def snapshot_path(spec, tag: str) -> Path:
    suf = Path(spec.source).suffix
    p = SNAPSHOTS / spec.name / f"{tag}{suf}"
    if not p.exists():
        raise SystemExit(f"no snapshot '{tag}' for {spec.name}. have: {snapshots(spec) or '(none)'}")
    return p


def snapshots(spec) -> list[str]:
    d = SNAPSHOTS / spec.name
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob(f"*{Path(spec.source).suffix}"))


# -- run store -----------------------------------------------------------------------

def record(kernel: str, kind: str, payload: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / f"{kernel}.jsonl").open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M"), "kind": kind,
                            **payload}) + "\n")


def history(kernel: str, kind: str | None = None) -> list[dict]:
    p = RESULTS / f"{kernel}.jsonl"
    if not p.exists():
        return []
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in recs if kind is None or r["kind"] == kind]
