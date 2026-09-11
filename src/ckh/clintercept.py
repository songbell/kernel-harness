"""E2E profiling with cl_intercept: which kernel, in which phase, actually costs time.

Every other phase of this harness measures a kernel that someone chose. This module is how
that choice becomes evidence: it runs the real pipeline under `cliloader`, splits the device
timeline into prefill and generate, and reports each kernel's share of its phase. That share
is the Amdahl ceiling on any end-to-end win, and it is what decides whether optimizing the
kernel is worth doing at all.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VERSION = "3.0.6"
RELEASE_URL = ("https://github.com/intel/opencl-intercept-layer/releases/download/"
               "v{v}/clintercept-{v}-Linux.tar.gz")

# -cdt is the per-queue device timeline (the JSON carrying per-kernel dur) and is required.
# -dv is the aggregate report, a cheap cross-check on it.
DEFAULT_FLAGS = ("-cdt", "-dv")

# Device-side memory traffic, not kernels. Kept out of kernel totals so a kernel's share of
# the phase is never inflated by copies.
MEM_RE = re.compile(r"^clEnqueue", re.I)


class Refused(RuntimeError):
    """Raised when proceeding would produce a number that cannot be trusted."""


@dataclass(frozen=True)
class DFlashPatternSpec:
    """The three attention groups emitted by one dflash iteration."""
    main_layers: int
    draft_layers: int
    main_full_attention_layers: int | None = None
    cm_regex: str = r"cm_sdpa_vlen"
    main_regex: str = r"sdpa_micro__generate|paged_attention_opt"
    draft_regex: str = r"sdpa_micro__prefill"
    gap_ms: float = 50.0


def load_num_hidden_layers(config: Path) -> int:
    """Read layer count from a HF/OpenVINO config, including nested text_config."""
    data = json.loads(config.read_text())
    for obj in (data, data.get("text_config", {}), data.get("dflash_config", {})):
        value = obj.get("num_hidden_layers")
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(f"{config}: no positive num_hidden_layers")


def load_full_attention_layers(config: Path) -> int:
    """Count full-attention layers; linear-attention models expose layer_types."""
    data = json.loads(config.read_text())
    candidates = [data, data.get("text_config", {})]
    for obj in candidates:
        layer_types = obj.get("layer_types")
        if isinstance(layer_types, list):
            count = sum(value == "full_attention" for value in layer_types)
            if count:
                return count
    return load_num_hidden_layers(config)


# -- locate / install ------------------------------------------------------------------

def locate(prefix: str | Path | None = None, explicit: str | None = None) -> Path | None:
    if explicit and os.access(explicit, os.X_OK):
        return Path(explicit)
    env = os.environ.get("CLI")
    if env and os.access(env, os.X_OK):
        return Path(env)
    roots = [Path(prefix)] if prefix else []
    for root in roots:
        for c in sorted(root.glob("clintercept-*/bin/cliloader")) + \
                 sorted(root.glob("clintercept-src/bin/cliloader")):
            if os.access(c, os.X_OK):
                return c
    which = shutil.which("cliloader")
    return Path(which) if which else None


def install(prefix: str | Path, version: str = DEFAULT_VERSION,
            from_source: bool = False) -> Path:
    """Install cl_intercept under `prefix`. The caller is responsible for asking the user
    first -- this downloads from the internet."""
    prefix = Path(prefix)
    prefix.mkdir(parents=True, exist_ok=True)
    if from_source:
        src = prefix / "opencl-intercept-layer"
        if not src.exists():
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/intel/opencl-intercept-layer.git", str(src)],
                           check=True)
        subprocess.run(["cmake", "-S", str(src), "-B", str(src / "build"),
                        "-DCMAKE_BUILD_TYPE=Release",
                        f"-DCMAKE_INSTALL_PREFIX={prefix / 'clintercept-src'}"], check=True)
        subprocess.run(["cmake", "--build", str(src / "build"),
                        "-j", str(os.cpu_count() or 4), "--target", "install"], check=True)
    else:
        url = RELEASE_URL.format(v=version)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed https URL)
                shutil.copyfileobj(resp, tmp)
            tarball = tmp.name
        try:
            with tarfile.open(tarball) as tf:
                tf.extractall(prefix, filter="data")
        finally:
            os.unlink(tarball)
    cli = locate(prefix)
    if not cli:
        raise Refused(f"install finished but no cliloader found under {prefix}")
    return cli


def smoke(cli: Path) -> list[str]:
    """Prove the install can actually trace. `--controls` exits 1 whenever no COMMAND
    follows, so it is judged by its output, not its return code."""
    msgs = []
    r = subprocess.run([str(cli), "--controls"], capture_output=True, text=True)
    if len(r.stdout.splitlines()) < 10:
        raise Refused("cliloader --controls produced no control list")
    msgs.append("--controls OK")
    if shutil.which("clinfo"):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([str(cli), "-cdt", "--dump-dir", d, "clinfo"],
                           capture_output=True, text=True)
            found = any(p.stat().st_size > 0 for p in Path(d).rglob("clintercept_trace.json"))
        msgs.append("trace json produced OK" if found else
                    "WARNING: smoke run produced no trace json")
    else:
        msgs.append("clinfo unavailable, only --controls was verified")
    return msgs


# -- run -------------------------------------------------------------------------------

def run(cli: Path, out_dir: Path, label: str, command: list[str],
        flags: tuple[str, ...] = DEFAULT_FLAGS, repeat: int = 1,
        env: dict[str, str] | None = None, timeout: int = 7200) -> list[dict]:
    """Launch the pipeline under cliloader `repeat` times.

    `command` is run verbatim: this module does not know what the user's pipeline is and
    must not guess. The caller is expected to have refused already if other GPU work is
    running -- see Platform.competing_gpu_work.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in range(1, repeat + 1):
        tag = label if repeat == 1 else f"{label}_r{r}"
        dump = out_dir / tag
        dump.mkdir(parents=True, exist_ok=True)
        log = out_dir / f"{tag}.log"
        e = dict(os.environ)
        e.update({"NEOReadDebugKeys": "1", "EnableCopyWithStagingBuffers": "1"})
        e.update(env or {})
        with log.open("w") as fh:
            rc = subprocess.run([str(cli), *flags, "--dump-dir", str(dump), *command],
                                stdout=fh, stderr=subprocess.STDOUT, env=e,
                                timeout=timeout).returncode
        metrics = _app_metrics(log)
        # The app's own TTFT/TPS lives beside the trace so a kernel budget is never read
        # without the number it is supposed to explain.
        (dump / "app_metrics.txt").write_text(
            f"exit_code={rc}\n" + "".join(f"{k}={v}\n" for k, v in metrics.items()))
        traced = any(p.stat().st_size > 0 for p in dump.rglob("clintercept_trace.json"))
        runs.append({"tag": tag, "dump": str(dump), "log": str(log), "exit_code": rc,
                     "app_metrics": metrics, "trace_present": traced})
    return runs


_METRIC_RE = re.compile(r"^\s*(TTFT|Throughput|Accept length)\s*:\s*(.+)$")


def _app_metrics(log: Path) -> dict[str, str]:
    out = {}
    for line in log.read_text(errors="ignore").splitlines():
        m = _METRIC_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


# -- parse -----------------------------------------------------------------------------

def _load_events(path: Path) -> list[dict]:
    """cliloader writes one JSON array per process and concatenates them; tolerate that
    and any interleaved garbage rather than losing the whole trace to one bad byte."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    dec = json.JSONDecoder()
    events, i, n = [], 0, len(text)
    while i < n:
        while i < n and text[i] not in "[{":
            i += 1
        if i >= n:
            break
        try:
            obj, j = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, list):
            events.extend(x for x in obj if isinstance(x, dict))
        elif isinstance(obj, dict):
            events.append(obj)
        i = j
    return events


def _absolute(events: list[dict]) -> list[tuple[float, float, str]]:
    """`ts` is relative to each process's own start; rebase onto the epoch so events from
    the several processes a pipeline spawns are comparable."""
    base = {ev.get("pid"): float(ev.get("args", {}).get("start_time", 0.0))
            for ev in events
            if ev.get("ph") == "M" and ev.get("name") == "clintercept_start_time"}
    out = [(float(ev.get("ts", 0.0)) + base.get(ev.get("pid"), 0.0),
            float(ev.get("dur", 0.0)), str(ev.get("name", "<unknown>")))
           for ev in events if ev.get("ph") == "X"]
    out.sort()
    return out


def _summarize(evs) -> list[dict]:
    agg = defaultdict(list)
    for _, dur, name in evs:
        agg[name].append(dur)
    rows = []
    for name, durs in agg.items():
        durs.sort()
        c = len(durs)
        rows.append({"kernel": name, "calls": c, "total_ms": sum(durs) / 1000.0,
                     "mean_us": sum(durs) / c, "p50_us": durs[(c - 1) // 2],
                     "p90_us": durs[min(c - 1, int((c - 1) * 0.9))], "max_us": durs[-1]})
    rows.sort(key=lambda r: r["total_ms"], reverse=True)
    return rows


def _span_ms(evs) -> float:
    if not evs:
        return 0.0
    return (max(ts + d for ts, d, _ in evs) - min(ts for ts, _, _ in evs)) / 1000.0


def analyze_dflash(trace: Path, spec: DFlashPatternSpec) -> dict:
    """Find dflash iterations as cm_sdpa_vlen + main attention + draft attention.

    The trace does not carry model ownership, so classification is deliberately regex- and
    count-based. A result is marked incomplete instead of silently assigning events to the
    wrong model when a layer sequence is truncated or interleaved.
    """
    raw = _absolute(_load_events(trace))
    if not raw:
        return {"trace": str(trace), "patterns": [], "warnings": ["no device events"]}
    cm_re = re.compile(spec.cm_regex, re.I)
    main_re = re.compile(spec.main_regex, re.I)
    draft_re = re.compile(spec.draft_regex, re.I)
    attention = [(ts, dur, name) for ts, dur, name in raw
                 if cm_re.search(name) or main_re.search(name) or draft_re.search(name)]
    anchors = [event for event in attention if cm_re.search(event[2])]
    patterns = []
    if anchors:
        main = [event for event in attention if main_re.search(event[2])]
        draft = [event for event in attention if draft_re.search(event[2])]
        main_layers = spec.main_full_attention_layers or spec.main_layers
        patterns.append({
            "index": 1,
            "cm_sdpa_vlen": len(anchors),
            "main_attention": len(main),
            "draft_attention": len(draft),
            "main_sequences": len(main) / main_layers,
            "draft_sequences": len(draft) / spec.draft_layers,
            "complete": (len(main) % main_layers == 0 and
                         len(draft) % spec.draft_layers == 0),
            "start_us": anchors[0][0],
            "end_us": attention[-1][0] + attention[-1][1],
        })
    warnings = []
    if not anchors:
        warnings.append(f"no dflash anchor matches /{spec.cm_regex}/")
    if any(not pattern["complete"] for pattern in patterns):
        warnings.append("one or more dflash patterns have incomplete layer sequences")
    return {"trace": str(trace),
            "main_layers": spec.main_full_attention_layers or spec.main_layers,
            "draft_layers": spec.draft_layers, "patterns": patterns, "warnings": warnings}


def render_dflash(result: dict) -> str:
    lines = [f"trace   {result['trace']}",
             f"dflash  main_layers={result.get('main_layers', '?')} "
             f"draft_layers={result.get('draft_layers', '?')}",
             "pattern cm_sdpa_vlen main_attention draft_attention main_seq draft_seq status"]
    for pattern in result.get("patterns", []):
        status = "complete" if pattern["complete"] else "INCOMPLETE"
        lines.append(f"{pattern['index']:>7} {pattern['cm_sdpa_vlen']:>13} "
                     f"{pattern['main_attention']:>15} {pattern['draft_attention']:>15} "
                     f"{pattern['main_sequences']:>9.2f} {pattern['draft_sequences']:>10.2f} "
                     f"{status}")
    for warning in result.get("warnings", []):
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)


@dataclass
class Segmentation:
    """The prefill/generate split is a decision, not a fact. These are its knobs, and the
    report states which values produced it."""
    split_kernel: str = ""
    pa_regex: str = r"pa_|sdpa|paged_attention"
    gap_ms: float = 50.0
    drop_cycles: int = 1


def analyze(trace: Path, kernel: str = "", seg: Segmentation | None = None) -> dict:
    seg = seg or Segmentation()
    split_re, pa_re = re.compile(seg.split_kernel, re.I), re.compile(seg.pa_regex, re.I)
    kernel_re = re.compile(kernel, re.I) if kernel else None

    raw = _absolute(_load_events(trace))
    if not raw:
        return {"trace": str(trace), "error": "no device events in trace"}

    mem = [e for e in raw if MEM_RE.match(e[2])]
    evs = [e for e in raw if not MEM_RE.match(e[2])]

    first_pa = next((ts for ts, _, name in evs if pa_re.search(name)), None)
    prologue = [e for e in evs if first_pa is not None and e[0] < first_pa]
    body = [e for e in evs if first_pa is None or e[0] >= first_pa]

    windows = _generate_windows(body, split_re, seg.gap_ms * 1000.0) if seg.split_kernel else []
    if seg.drop_cycles and len(windows) > seg.drop_cycles:
        cut = windows[seg.drop_cycles - 1][1]
        body = [e for e in body if e[0] >= cut]
        windows = windows[seg.drop_cycles:]

    gen = [e for e in body if any(lo <= e[0] < hi for lo, hi in windows)]
    pre = [e for e in body if not any(lo <= e[0] < hi for lo, hi in windows)]

    out = {
        "trace": str(trace),
        "app_metrics": _read_metrics(trace.parent / "app_metrics.txt"),
        "global": {"rows": _summarize(evs), "span_ms": _span_ms(evs)},
        "segmentation": {"anchor": seg.split_kernel, "gap_ms": seg.gap_ms,
                         "cycles": len(windows), "dropped_cycles": seg.drop_cycles,
                         "prologue_events": len(prologue),
                         "prologue_ms": sum(d for _, d, _ in prologue) / 1000.0,
                         "mem_ops": len(mem), "mem_ms": sum(d for _, d, _ in mem) / 1000.0},
        "phases": {},
        "warnings": [],
    }
    for name, evl in (("prefill", pre), ("generate", gen)):
        rows = _summarize(evl)
        total = sum(r["total_ms"] for r in rows)
        span = _span_ms(evl)
        out["phases"][name] = {"rows": rows, "total_ms": total, "span_ms": span,
                               "concurrency": total / span if span else 0.0}

    if first_pa is None:
        out["warnings"].append(
            f"no kernel matches /{seg.pa_regex}/, so the model-load prologue could not be "
            f"located and is included in PREFILL")
    for name in ("prefill", "generate"):
        if out["phases"][name]["concurrency"] > 1.5:
            out["warnings"].append(
                f"{name}: queues overlap ({out['phases'][name]['concurrency']:.2f}x), so "
                f"summed device time is not wall time -- shares bound device work, and the "
                f"e2e win may be smaller still")

    if kernel_re:
        if not any(kernel_re.search(n) for _, _, n in raw):
            out["kernel"] = {"regex": kernel, "matched": False,
                             "ran_instead": [r["kernel"] for r in
                                             (out["phases"]["generate"]["rows"] or
                                              out["phases"]["prefill"]["rows"])[:8]]}
            out["warnings"].append(
                "the named kernel was never enqueued. That is a configuration result, not a "
                "performance result -- check the CM path flag and the plugin build")
        else:
            k = {"regex": kernel, "matched": True}
            for name in ("prefill", "generate"):
                ph = out["phases"][name]
                hit = [r for r in ph["rows"] if kernel_re.search(r["kernel"])]
                ms = sum(r["total_ms"] for r in hit)
                k[name] = {"total_ms": ms, "calls": sum(r["calls"] for r in hit),
                           "pct_of_phase": 100.0 * ms / ph["total_ms"] if ph["total_ms"] else 0.0}
            out["kernel"] = k
            leak = sum(r["total_ms"] for r in out["phases"]["prefill"]["rows"]
                       if split_re.search(r["kernel"]))
            if leak > 0.05 * max(out["phases"]["generate"]["total_ms"], 1e-9):
                out["warnings"].append(
                    f"{leak:.1f} ms of the anchor kernel landed in PREFILL -- the "
                    f"segmentation is unreliable, raise gap_ms or pick another anchor")
    return out


def _generate_windows(evs, split_re, gap_us):
    hits = [(ts, ts + dur) for ts, dur, name in evs if split_re.search(name)]
    if not hits:
        return []
    win = [list(hits[0])]
    for start, end in hits[1:]:
        if start - win[-1][1] > gap_us:
            win.append([start, end])
        else:
            win[-1][1] = max(win[-1][1], end)
    return [tuple(w) for w in win]


def _read_metrics(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return dict(l.split("=", 1) for l in path.read_text(errors="ignore").splitlines()
                if "=" in l)


# -- render ----------------------------------------------------------------------------

def render(result: dict, top: int = 15) -> str:
    if "error" in result:
        return f"{result['trace']}: {result['error']}"
    L = [f"trace   {result['trace']}"]
    for k, v in result["app_metrics"].items():
        L.append(f"app     {k}={v}")
    s = result["segmentation"]
    anchor = f"/{s['anchor']}/" if s["anchor"] else "(none; global summary)"
    L.append(f"split   anchor {anchor}  gap>{s['gap_ms']}ms  "
             f"cycles={s['cycles']} (dropped {s['dropped_cycles']})")
    L.append(f"dropped {s['prologue_events']} prologue events ({s['prologue_ms']:.1f} ms), "
             f"{s['mem_ops']} clEnqueue* mem ops ({s['mem_ms']:.1f} ms, excluded)")

    global_phase = result["global"]
    global_rows = global_phase["rows"]
    global_total = sum(r["total_ms"] for r in global_rows)
    L.append(f"\nGLOBAL TOP KERNELS  summed device {global_total:.1f} ms"
             f" over a {global_phase['span_ms']:.1f} ms wall span")
    L.append(f"  {'kernel':<48}{'calls':>8}{'total_ms':>11}{'%total':>8}"
             f"{'mean_us':>10}{'p50':>9}{'p90':>9}")
    for r in global_rows[:top]:
        pct = 100.0 * r["total_ms"] / global_total if global_total else 0.0
        L.append(f"  {r['kernel'][:48]:<48}{r['calls']:>8}{r['total_ms']:>11.2f}"
                 f"{pct:>7.1f}%{r['mean_us']:>10.2f}{r['p50_us']:>9.2f}{r['p90_us']:>9.2f}")

    for name in ("prefill", "generate"):
        ph = result["phases"][name]
        L.append(f"\n{name.upper():<9} summed device {ph['total_ms']:.1f} ms over a "
                 f"{ph['span_ms']:.1f} ms wall span (concurrency {ph['concurrency']:.2f}x)")
        L.append(f"  {'kernel':<48}{'calls':>8}{'total_ms':>11}{'%phase':>8}"
                 f"{'mean_us':>10}{'p50':>9}{'p90':>9}")
        for r in ph["rows"][:top]:
            pct = 100.0 * r["total_ms"] / ph["total_ms"] if ph["total_ms"] else 0.0
            L.append(f"  {r['kernel'][:48]:<48}{r['calls']:>8}{r['total_ms']:>11.2f}"
                     f"{pct:>7.1f}%{r['mean_us']:>10.2f}{r['p50_us']:>9.2f}{r['p90_us']:>9.2f}")

    k = result.get("kernel")
    if k and not k.get("matched"):
        L.append(f"\n/{k['regex']}/  NO MATCH ANYWHERE IN THE TRACE")
        L.append("  kernels that did run: " + ", ".join(k["ran_instead"][:5]))
    elif k:
        L.append(f"\n/{k['regex']}/")
        for name in ("prefill", "generate"):
            e = k[name]
            L.append(f"  {name:<9}{e['total_ms']:9.2f} ms over {e['calls']:6d} calls  "
                     f"= {e['pct_of_phase']:5.1f}% of the phase"
                     f"   -> Amdahl ceiling: {e['pct_of_phase']:.1f}%")
    for w in result["warnings"]:
        L.append(f"\nWARNING: {w}")
    return "\n".join(L)
