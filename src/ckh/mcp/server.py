"""The ckh MCP tool surface -- `ckh doctor|bench|equiv|round|snapshot|ledger` for a model.

Same functions the CLI calls, same discipline: `bench` refuses to measure while other GPU
work is running, results carry the noise-floor verdict, and `ledger` is exposed so an agent
can read what was already settled *before* proposing a change. Output is compact on purpose
-- verbose logs stay in the measurement environment.

Run:  python3 -m ckh.mcp            (stdio, local)
      python3 -m ckh.mcp --http     (Streamable-HTTP, for a shared/remote GPU box)
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

from . import McpServer

REPO_ROOT = Path(__file__).resolve().parents[3]
# `kernels.<name>` descriptors live at the repo root, not inside the package.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ckh import bench, clintercept, equiv, ledger, results   # noqa: E402
from ckh.platform import Platform                      # noqa: E402

server = McpServer("ckh", "0.1.0")

_KERNEL_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _spec(name: str):
    """Resolve a kernel descriptor, refusing anything that is not a plain module name.

    The name reaches `importlib`, so an unconstrained value would be an arbitrary-import
    primitive on a server that may be listening on a socket.
    """
    if not _KERNEL_RE.match(name or ""):
        raise ValueError(f"invalid kernel name: {name!r} (expected [A-Za-z0-9_]+)")
    if not (REPO_ROOT / "kernels" / f"{name}.py").exists():
        raise ValueError(f"no descriptor kernels/{name}.py (see list_kernels)")
    mod = importlib.import_module(f"kernels.{name}")
    return mod.SPEC, getattr(mod, "DEFAULT_AXES", {})


def _axes(name: str, overrides: dict[str, Any] | None) -> dict:
    """Merge `{"q_len": "6,16"}` or `{"q_len": [6, 16]}` onto the descriptor's defaults."""
    _, default_axes = _spec(name)
    axes = dict(default_axes)
    for k, v in (overrides or {}).items():
        vals = v if isinstance(v, list) else str(v).split(",")
        axes[k] = [int(x) if str(x).lstrip("-").isdigit() else x for x in vals]
    return axes


def _guarded_platform() -> Platform:
    plat = Platform.load()
    busy = plat.competing_gpu_work()
    if busy:
        raise RuntimeError(
            "refusing to measure: other GPU work is running -- every number would be "
            f"meaningless. Offenders: {busy[:3]}")
    return plat


@server.tool(
    "doctor",
    "Validate the measurement environment BEFORE trusting any number: config paths, whether "
    "`clops` resolves, and whether competing GPU work is running. Call this first in a session.",
    {"type": "object", "properties": {}},
)
def _doctor(_args: dict[str, Any]) -> Any:
    plat = Platform.load()
    clops_ok, clops_msg = plat.check_clops()
    busy = plat.competing_gpu_work()
    return {
        "backend": plat.backend,
        "sandbox": {"path": str(plat.sandbox), "exists": plat.sandbox.exists()},
        "production": {"path": str(plat.production), "exists": plat.production.exists()},
        "noise_floor_pct": plat.noise_floor_pct,
        "rounds": plat.rounds,
        "clops": {"ok": clops_ok, "detail": clops_msg},
        "competing_gpu_work": busy,
        "ready": bool(clops_ok and not busy),
    }


@server.tool(
    "list_kernels",
    "List the kernel descriptors this harness can measure (kernels/<name>.py) with their "
    "default shape axes.",
    {"type": "object", "properties": {}},
)
def _list_kernels(_args: dict[str, Any]) -> Any:
    out = []
    for p in sorted((REPO_ROOT / "kernels").glob("*.py")):
        if p.stem.startswith("_"):
            continue
        try:
            spec, axes = _spec(p.stem)
            out.append({"kernel": p.stem, "default_axes": axes,
                        "reference": type(getattr(spec, "reference", None)).__name__})
        except Exception as exc:
            out.append({"kernel": p.stem, "error": str(exc)})
    return out


@server.tool(
    "bench",
    "Interleaved min-of-N timing over a shape grid. Configs are interleaved round-by-round "
    "(sequential A-then-B charges B for A's heat) and any config whose spread exceeds the "
    "configured noise floor is flagged -- a difference smaller than the spread is NOT a result.",
    {
        "type": "object",
        "properties": {
            "kernel": {"type": "string", "description": "descriptor name, e.g. pa_small_q"},
            "axes": {"type": "object", "description": 'shape axis overrides, e.g. {"q_len": "6,16"}'},
            "variants": {"type": "object",
                         "description": 'A/B in ONE interleaved batch: {label: {DEFINE: value}}'},
            "rounds": {"type": "integer", "description": "repetitions per config (default from platform.toml)"},
        },
        "required": ["kernel"],
    },
)
def _bench(args: dict[str, Any]) -> Any:
    plat = _guarded_platform()
    spec, _ = _spec(args["kernel"])
    shapes = bench.expand(_axes(args["kernel"], args.get("axes")))
    res = bench.measure(plat, spec, shapes, args.get("variants") or None,
                        rounds=args.get("rounds"))
    return {"table": bench.render(res, plat.noise_floor_pct),
            "noise_floor_pct": plat.noise_floor_pct, "results": res}


@server.tool(
    "equiv",
    "Check the kernel against its declared reference (an independent torch ground truth, or "
    "another compiled kernel as baseline). Returns per-shape pass/fail.",
    {
        "type": "object",
        "properties": {
            "kernel": {"type": "string"},
            "axes": {"type": "object", "description": 'shape axis overrides, e.g. {"q_len": "6"}'},
        },
        "required": ["kernel"],
    },
)
def _equiv(args: dict[str, Any]) -> Any:
    plat = _guarded_platform()
    spec, _ = _spec(args["kernel"])
    shapes = bench.expand(_axes(args["kernel"], args.get("axes")))
    res = equiv.measure(plat, spec, shapes)
    return {"table": equiv.render(spec, res),
            "passed": all(v.get("passed", False) for v in res.values()),
            "results": res}


@server.tool(
    "snapshot",
    "Pin the current kernel source as a named comparison baseline, or list existing snapshots. "
    "A number recorded in an earlier session is not a valid baseline on a rig that drifts -- "
    "the previous SOURCE is what gets re-measured.",
    {
        "type": "object",
        "properties": {
            "kernel": {"type": "string"},
            "tag": {"type": "string", "description": "snapshot name (default: base)"},
            "list": {"type": "boolean", "description": "list snapshots instead of creating one"},
        },
        "required": ["kernel"],
    },
)
def _snapshot(args: dict[str, Any]) -> Any:
    spec, _ = _spec(args["kernel"])
    if args.get("list"):
        return {"snapshots": results.snapshots(spec)}
    tag = args.get("tag") or "base"
    return {"snapshot": tag, "path": str(results.snapshot(Platform.load(), spec, tag))}


@server.tool(
    "round",
    "A/B the working tree against a pinned snapshot in ONE interleaved run over the FULL grid. "
    "The full grid matters: a change that helps one shape and breaks another is what this catches.",
    {
        "type": "object",
        "properties": {
            "kernel": {"type": "string"},
            "against": {"type": "string", "description": "snapshot tag to compare against"},
            "axes": {"type": "object"},
            "rounds": {"type": "integer"},
        },
        "required": ["kernel", "against"],
    },
)
def _round(args: dict[str, Any]) -> Any:
    plat = _guarded_platform()
    spec, _ = _spec(args["kernel"])
    base = str(results.snapshot_path(spec, args["against"]))
    axes = _axes(args["kernel"], args.get("axes"))
    res = bench.measure(plat, spec, bench.expand(axes),
                        variants={args["against"]: {}, "current": {}},
                        sources={args["against"]: base},
                        rounds=args.get("rounds"))
    table, worse = bench.render_round(res, args["against"], "current", plat.noise_floor_pct)
    results.record(args["kernel"], "round",
                   {"against": args["against"], "axes": axes, "regressions": worse, "results": res})
    return {"table": table, "regressions": worse, "results": res}


@server.tool(
    "ledger_query",
    "Prior findings for a kernel -- READ THIS BEFORE PROPOSING A CHANGE. Says what was already "
    "measured and whether it was rejected on measurement (revisitable) or on policy (needs the owner).",
    {
        "type": "object",
        "properties": {
            "kernel": {"type": "string"},
            "only": {"type": "string", "description": "filter by verdict prefix, e.g. rejected"},
        },
        "required": ["kernel"],
    },
)
def _ledger_query(args: dict[str, Any]) -> Any:
    return {"entries": ledger.load(args["kernel"]) if not args.get("only") else
            [r for r in ledger.load(args["kernel"]) if r["verdict"].startswith(args["only"])],
            "table": ledger.render(args["kernel"], args.get("only"))}


@server.tool(
    "ledger_add",
    "Record a finding so a later session does not re-test it. Verdict must be one of: "
    "adopted, rejected-measurement, rejected-policy, invalid-probe, open.",
    {
        "type": "object",
        "properties": {
            "kernel": {"type": "string"},
            "change": {"type": "string", "description": "what was tried"},
            "verdict": {"type": "string", "enum": list(ledger.VERDICTS)},
            "shape": {"type": "string", "description": "what it was measured on"},
            "numbers": {"type": "string", "description": "the measured numbers"},
            "bit_exact": {"type": "boolean"},
            "note": {"type": "string"},
        },
        "required": ["kernel", "change", "verdict"],
    },
)
def _ledger_add(args: dict[str, Any]) -> Any:
    ledger.add(args["kernel"], change=args["change"], verdict=args["verdict"],
               shape=args.get("shape", ""), numbers=args.get("numbers", ""),
               note=args.get("note", ""), bit_exact=args.get("bit_exact"))
    return {"recorded": True}


# `ckh profile run` is deliberately NOT exposed here: it executes an arbitrary user-supplied
# pipeline command, which over a socket would be a remote-execution primitive. Collection
# stays on the CLI; the model gets detection and analysis.

@server.tool(
    "profile_status",
    "Is cl_intercept (cliloader) available for e2e profiling? Detection only -- installing "
    "downloads from the internet and is left to `ckh profile setup --install` on the CLI.",
    {"type": "object",
     "properties": {"prefix": {"type": "string", "description": "install prefix to search"}}},
)
def _profile_status(args: dict[str, Any]) -> Any:
    plat = Platform.load()
    prefix = args.get("prefix") or plat.raw.get("profile", {}).get("install_prefix") \
        or str(REPO_ROOT / "third_party")
    tool = clintercept.locate(prefix, plat.raw.get("profile", {}).get("cliloader"))
    return {"available": tool is not None, "cliloader": str(tool) if tool else None,
            "searched_prefix": prefix,
            "hint": None if tool else "run `ckh profile setup --install` on the measurement box"}


@server.tool(
    "profile_report",
    "Global top-kernel time summary from an existing cl_intercept dump, plus the Amdahl "
    "ceiling: the share of the phase a kernel occupies bounds any e2e win from optimizing "
    "it. Run this BEFORE choosing a kernel -- a kernel 3x off its roofline that is 2% of the "
    "phase is not worth optimizing.",
    {
        "type": "object",
        "properties": {
            "dump_dir": {"type": "string", "description": "directory produced by `ckh profile run`"},
            "kernel": {"type": "string", "description": "regex for the kernel under investigation"},
            "top": {"type": "integer", "description": "global kernel rows (default 5)"},
            "anchor": {"type": "string",
                       "description": "regex marking the generate phase; the prefill/generate "
                                      "split is a decision and this is its knob"},
            "gap_ms": {"type": "number", "description": "silence that separates two cycles"},
            "drop_cycles": {"type": "integer", "description": "warm-up iterations to discard"},
            "dflash": {"type": "boolean", "description": "report the configured dflash pattern"},
            "dflash_main_config": {"type": "string"},
            "dflash_draft_config": {"type": "string"},
        },
        "required": ["dump_dir"],
    },
)
def _profile_report(args: dict[str, Any]) -> Any:
    root = Path(args["dump_dir"])
    traces = sorted(root.rglob("clintercept_trace.json"))
    if not traces:
        raise ValueError(f"no clintercept_trace.json under {root}")
    seg = clintercept.Segmentation(gap_ms=float(args.get("gap_ms", 50.0)),
                                   drop_cycles=int(args.get("drop_cycles", 1)))
    if args.get("anchor"):
        seg.split_kernel = args["anchor"]
    out = []
    for t in traces:
        if args.get("dflash"):
            cfg = Platform.load().raw.get("profile", {}).get("dflash", {})
            main_config = args.get("dflash_main_config") or cfg.get("main_config")
            draft_config = args.get("dflash_draft_config") or cfg.get("draft_config")
            if not main_config or not draft_config:
                raise ValueError("dflash requires profile.dflash main_config and draft_config")
            spec = clintercept.DFlashPatternSpec(
                main_layers=clintercept.load_num_hidden_layers(Path(main_config)),
                draft_layers=clintercept.load_num_hidden_layers(Path(draft_config)),
                main_full_attention_layers=clintercept.load_full_attention_layers(
                    Path(main_config)),
                cm_regex=cfg.get("cm_regex", r"cm_sdpa_vlen"),
                main_regex=cfg.get("main_regex", r"sdpa_micro__generate|paged_attention_opt"),
                draft_regex=cfg.get("draft_regex", r"sdpa_micro__prefill"),
                gap_ms=float(cfg.get("gap_ms", 50.0)))
            res = clintercept.analyze_dflash(t, spec)
            out.append({"table": clintercept.render_dflash(res),
                        "patterns": res["patterns"], "warnings": res["warnings"]})
            continue
        res = clintercept.analyze(t, args.get("kernel", ""), seg)
        out.append({"table": clintercept.render(res, int(args.get("top", 5))),
                    "kernel": res.get("kernel"), "warnings": res["warnings"]})
    return out


def main() -> int:
    """Serve over stdio (local) or Streamable-HTTP (a shared GPU box reached remotely).

    HTTP is selected by ``--http``, ``MCP_TRANSPORT=http`` or ``CKH_HTTP=1``. Unlike a
    knowledge-base server, every tool here compiles and runs kernel code, so the bind
    defaults to loopback; widening it should come with ``MCP_AUTH_TOKEN``.
    """
    import argparse
    import os

    def _clean(name: str, default: str) -> str:
        # Tolerate an inline comment copied from .env.example: "8791  # port" -> "8791".
        return (os.environ.get(name, default) or default).split("#", 1)[0].strip() or default

    ap = argparse.ArgumentParser(prog="ckh-mcp")
    ap.add_argument("--http", action="store_true", help="serve over Streamable-HTTP instead of stdio")
    ap.add_argument("--host", default=_clean("CKH_HTTP_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(_clean("CKH_HTTP_PORT", "8791")))
    args = ap.parse_args()

    use_http = (args.http
                or os.environ.get("MCP_TRANSPORT", "").lower() == "http"
                or os.environ.get("CKH_HTTP", "").lower() in {"1", "true", "yes"})
    if use_http:
        return server.serve_http(args.host, args.port)
    return server.serve_stdio()
