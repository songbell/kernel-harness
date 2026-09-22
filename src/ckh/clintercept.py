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
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

DEFAULT_VERSION = "3.0.6"
RELEASE_URL = "https://github.com/intel/opencl-intercept-layer/releases/download/v{v}/clintercept-{v}-{platform}.{extension}"

HARNESS_STAGES = frozenset({
    "profile",
    "kernelgen",
    "validate",
    "kernel-profile",
    "bench",
    "equiv",
    "round",
    "snapshot",
})

# -cdt is the per-queue device timeline (the JSON carrying per-kernel dur) and is required.
# -dv is the aggregate report, a cheap cross-check on it.
DEFAULT_FLAGS = ("-cdt", "-dv")

# Device-side memory traffic, not kernels. Kept out of kernel totals so a kernel's share of
# the phase is never inflated by copies.
MEM_RE = re.compile(r"^clEnqueue", re.I)


def prepare_profile_env(out_dir: Path, env: dict[str, str] | None = None,
                        dump_sources: str | None = None) -> tuple[dict[str, str], Path]:
    """Return a profile env with OV_GPU_DUMP_SOURCES_PATH guaranteed to be set."""
    merged = dict(env or {})
    dump_path = Path(dump_sources or merged.get("OV_GPU_DUMP_SOURCES_PATH")
                     or (out_dir / "ov_gpu_dump_sources"))
    dump_path.mkdir(parents=True, exist_ok=True)
    merged["OV_GPU_DUMP_SOURCES_PATH"] = str(dump_path)
    return merged, dump_path

def model_for_stage(stage: str, models: Mapping[str, str],
                    stage_models: Mapping[str, str]) -> str:
    """Return the configured model path for a named pipeline stage.

    ``models`` maps stable roles such as ``router`` and ``reasoner`` to model paths;
    ``stage_models`` maps CKH workflow stages such as ``profile`` and ``kernelgen`` to
    those names. Both lookups are strict so a typo cannot silently select the wrong model.
    """
    stage = stage.strip()
    if not stage:
        raise ValueError("stage must not be empty")
    if stage not in HARNESS_STAGES:
        raise KeyError(f"unknown CKH stage '{stage}'")
    model_name = stage_models.get(stage)
    if not model_name:
        raise KeyError(f"no model configured for stage '{stage}'")
    model_path = models.get(model_name)
    if not model_path:
        raise KeyError(f"stage '{stage}' references unknown model '{model_name}'")
    return model_path


class Refused(RuntimeError):
    """Raised when proceeding would produce a number that cannot be trusted."""


@dataclass(frozen=True)
class SpeculativePatternSpec:
    """The main and draft attention groups emitted by one speculative-decoding iteration.

    Classification is regex-based only: `main_regex`/`draft_regex` match whatever device
    kernel names the pipeline actually enqueues (OpenVINO's built-in micro-SDPA kernels, a
    CM kernel, or anything else) -- no specific implementation is assumed or required.

    Both regexes are OPTIONAL and default to None: this module must not guess a kernel name
    that happens to work on one pipeline. Without a regex for a side, `analyze_speculative`
    cannot count that side's attention calls or iterations -- instead it reports that side's
    hot-spot kernels (top device time), so the report is still useful and the regex can be
    picked from real evidence rather than guessed up front.
    """
    main_layers: int
    draft_layers: int
    main_full_attention_layers: int | None = None
    main_regex: str | None = None
    draft_regex: str | None = None
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


# -- speculative-decoding inference ------------------------------------------------------

_SPECULATIVE_NAME_RE = re.compile(r"speculat|dflash|draft", re.I)


def infer_speculative_pipeline(pipeline: list[str], max_depth: int = 3) -> dict:
    """Guess whether `pipeline` is a speculative-decoding run and locate its main/draft
    `config.json`, from the argv alone -- no execution, no network.

    This is deliberately conservative: it reports its confidence and every ambiguity it
    found rather than picking a config.json to be helpful. A caller (subagent or human)
    decides whether "heuristic" confidence is good enough to act on; this function never
    is that caller.
    """
    result = {"is_speculative": False, "confidence": "none", "reason": "", "candidates": [],
              "warnings": []}
    if not pipeline:
        result["reason"] = "empty pipeline"
        return result

    exe_hint = bool(_SPECULATIVE_NAME_RE.search(Path(pipeline[0]).stem))
    dirs = [arg for arg in pipeline[1:] if Path(arg).is_dir()]

    if not exe_hint and len(dirs) < 2:
        result["reason"] = ("binary name has no speculative/dflash/draft hint and fewer than "
                            "2 directory-shaped args were found")
        return result

    result["is_speculative"] = True
    result["confidence"] = "high" if exe_hint else "low"
    result["reason"] = (f"binary name matched /{_SPECULATIVE_NAME_RE.pattern}/" if exe_hint
                        else f"{len(dirs)} directory-shaped args, no binary-name hint")
    if len(dirs) < 2:
        result["warnings"].append(
            f"only {len(dirs)} directory-shaped arg(s) found; cannot pair a main and a draft "
            f"model directory")
        return result
    if len(dirs) > 2:
        result["warnings"].append(
            f"{len(dirs)} directory-shaped args found; assuming the FIRST TWO, in argv order, "
            f"are main then draft -- confirm before trusting")

    for role, model_dir in zip(("main", "draft"), dirs[:2]):
        d = Path(model_dir)
        entry = {"role": role, "dir": str(d), "config": None, "config_candidates": []}
        direct = d / "config.json"
        if direct.is_file():
            entry["config"] = str(direct)
        else:
            found = []
            for depth in range(1, max_depth + 1):
                pattern = "/".join(["*"] * depth + ["config.json"])
                found = sorted(d.glob(pattern))
                if found:
                    break
            if len(found) == 1:
                entry["config"] = str(found[0])
            elif len(found) > 1:
                entry["config_candidates"] = [str(f) for f in found]
                result["warnings"].append(
                    f"{role} dir {d} has {len(found)} config.json candidates under it -- "
                    f"refusing to guess which one")
            else:
                result["warnings"].append(f"no config.json found under {role} dir {d} "
                                          f"(searched {max_depth} levels deep)")
        result["candidates"].append(entry)
    result["warnings"].append(
        "main/draft role assignment is POSITIONAL (first dir = main, second = draft) -- this "
        "is the convention for this pipeline's argv, not a property this function can verify")
    return result


# -- locate / install ------------------------------------------------------------------

def _release_info() -> tuple[str, str]:
    if os.name == "nt":
        return "win64", "zip"
    return "Linux", "tar.gz"


def locate(prefix: str | Path | None = None, explicit: str | None = None) -> Path | None:
    if explicit and os.access(explicit, os.X_OK):
        return Path(explicit)
    env = os.environ.get("CLI")
    if env and os.access(env, os.X_OK):
        return Path(env)
    roots = [Path(prefix)] if prefix else []
    for root in roots:
        if os.name == "nt":
            candidates = sorted(root.glob("clintercept-*/bin/cliloader.exe")) + \
                         sorted(root.glob("clintercept-*/cliloader.exe")) + \
                         sorted(root.glob("clintercept-*/Release/cliloader.exe")) + \
                         sorted(root.glob("clintercept-src/**/cliloader.exe"))
        else:
            candidates = sorted(root.glob("clintercept-*/bin/cliloader")) + \
                         sorted(root.glob("clintercept-src/bin/cliloader"))
        for c in candidates:
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
        release_platform, extension = _release_info()
        url = RELEASE_URL.format(v=version, platform=release_platform, extension=extension)
        with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as tmp:
            with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed https URL)
                shutil.copyfileobj(resp, tmp)
            tarball = tmp.name
        try:
            if extension == "zip":
                with zipfile.ZipFile(tarball) as zf:
                    zf.extractall(prefix)
            else:
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
    env: dict[str, str] | None = None, timeout: int = 7200,
    dump_sources: str | None = None) -> list[dict]:
    """Launch the pipeline under cliloader `repeat` times.

    `command` is run verbatim: this module does not know what the user's pipeline is and
    must not guess. The caller is expected to have refused already if other GPU work is
    running -- see Platform.competing_gpu_work.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_env, dump_sources_path = prepare_profile_env(out_dir, env, dump_sources)
    (out_dir / "profile_run.json").write_text(json.dumps({
        "dump_sources": str(dump_sources_path),
        "command": command,
        "repeat": repeat,
    }, indent=2))
    runs = []
    for r in range(1, repeat + 1):
        tag = label if repeat == 1 else f"{label}_r{r}"
        dump = out_dir / tag
        dump.mkdir(parents=True, exist_ok=True)
        log = out_dir / f"{tag}.log"
        e = dict(os.environ)
        e.update({"NEOReadDebugKeys": "1", "EnableCopyWithStagingBuffers": "1"})
        e.update(profile_env)
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
                 "app_metrics": metrics, "trace_present": traced,
                 "dump_sources": str(dump_sources_path)})
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


def _classify_by_call_count(rows: list[dict], main_layers: int, draft_layers: int
                            ) -> tuple[list[dict], list[dict], list[dict]]:
    """Split hot-spot rows into main/draft/unclassified using call-count divisibility only.

    No kernel name is ever inspected here. Main and draft models are almost always
    different sizes, so their compiled kernel variants get different call counts per
    iteration -- a kernel whose total `calls` divides evenly by exactly one side's layer
    count is (heuristically) that side's. A count divisible by BOTH, or by NEITHER, is
    reported unclassified rather than guessed.
    """
    main_rows, draft_rows, other_rows = [], [], []
    for row in rows:
        calls = row["calls"]
        is_main = bool(main_layers) and calls % main_layers == 0
        is_draft = bool(draft_layers) and calls % draft_layers == 0
        if is_main and not is_draft:
            main_rows.append(row)
        elif is_draft and not is_main:
            draft_rows.append(row)
        else:
            other_rows.append(row)
    return main_rows, draft_rows, other_rows


_PREFILL_FAMILY_RE = re.compile(r"sdpa_micro__prefill|paged_attention|cm_pa", re.I)
_GENERATE_FAMILY_RE = re.compile(r"sdpa_micro__generate|paged_attention|cm_pa", re.I)


def _speculative_phase_name_sets(rows: list[dict], main_layers: int, draft_layers: int) -> dict[str, set[str]]:
    names = {
        "main_prefill": set(),
        "draft_generate": set(),
        "main_generate": set(),
    }
    for row in rows:
        kernel = row["kernel"]
        calls = row["calls"]
        main_div = bool(main_layers) and calls % main_layers == 0
        draft_div = bool(draft_layers) and calls % draft_layers == 0
        if _PREFILL_FAMILY_RE.search(kernel):
            if main_div and not draft_div:
                names["main_prefill"].add(kernel)
            elif draft_div and not main_div:
                names["draft_generate"].add(kernel)
        if _GENERATE_FAMILY_RE.search(kernel) and main_div:
            names["main_generate"].add(kernel)
    return names


def _bucket_for_event(name: str, phase_names: dict[str, set[str]]) -> str | None:
    for bucket in ("main_prefill", "draft_generate", "main_generate"):
        if name in phase_names[bucket]:
            return bucket
    return None


def _attention_windows(evs, phase_names: dict[str, set[str]]) -> list[dict]:
    windows = []
    attention = []
    for ts, dur, name in evs:
        bucket = _bucket_for_event(name, phase_names)
        if bucket is None:
            continue
        attention.append((ts, dur, name, bucket))
    for ts, dur, name, bucket in attention:
        end = ts + dur
        if windows and windows[-1]["bucket"] == bucket:
            windows[-1]["end_us"] = max(windows[-1]["end_us"], end)
            windows[-1]["attention_calls"] += 1
        else:
            side, phase = bucket.split("_", 1)
            windows.append({
                "bucket": bucket,
                "side": side,
                "phase": phase,
                "start_us": ts,
                "end_us": end,
                "attention_calls": 1,
            })
    for index, window in enumerate(windows, 1):
        window["index"] = index
        center = window["start_us"] + (window["end_us"] - window["start_us"]) / 2.0
        window["center_us"] = center
    return windows


def _classify_events_by_windows(evs, windows: list[dict]) -> dict[str, list[tuple[float, float, str]]]:
    buckets = {
        "main_prefill": [],
        "main_generate": [],
        "draft_prefill": [],
        "draft_generate": [],
    }
    if not windows:
        return buckets
    centers = [w["center_us"] for w in windows]
    boundaries = [(centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1)]
    for event in evs:
        midpoint = event[0] + event[1] / 2.0
        index = 0
        while index < len(boundaries) and midpoint >= boundaries[index]:
            index += 1
        window = windows[index]
        buckets[f"{window['side']}_{window['phase']}"] .append(event)
    return buckets


def _intervals_between_windows(windows: list[dict]) -> list[dict]:
    intervals = []
    for left, right in zip(windows, windows[1:]):
        intervals.append({
            "from": f"{left['side']}_{left['phase']}",
            "to": f"{right['side']}_{right['phase']}",
            "gap_ms": max(0.0, (right["start_us"] - left["end_us"]) / 1000.0),
            "start_us": left["end_us"],
            "end_us": right["start_us"],
        })
    return intervals


def _summarize_intervals(intervals: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for interval in intervals:
        groups[(interval["from"], interval["to"])].append(interval["gap_ms"])
    rows = []
    for (left, right), values in groups.items():
        values.sort()
        count = len(values)
        rows.append({
            "from": left,
            "to": right,
            "count": count,
            "mean_gap_ms": sum(values) / count,
            "p50_gap_ms": values[(count - 1) // 2],
            "max_gap_ms": values[-1],
        })
    rows.sort(key=lambda row: (row["from"], row["to"]))
    return rows


def _generation_count(windows: list[dict]) -> int:
    """One main-generate window == one decode step; this is the denominator for
    per-generation kernel cost, not a raw event count."""
    return sum(1 for w in windows if w["bucket"] == "main_generate")


def _draft_span_ms(windows: list[dict]) -> float | None:
    """Average draft-window duration, counted only for draft windows sandwiched between
    two main-generate windows -- excludes the leftover draft activity right after prefill,
    which is not a steady-state speculative iteration."""
    spans = [(w["end_us"] - w["start_us"]) / 1000.0
             for prev, w, nxt in zip(windows, windows[1:], windows[2:])
             if w["bucket"] == "draft_generate"
             and prev["bucket"] == "main_generate" and nxt["bucket"] == "main_generate"]
    return sum(spans) / len(spans) if spans else None


def analyze_speculative(trace: Path, spec: SpeculativePatternSpec, top: int = 10) -> dict:
    """Count main/draft attention kernel calls and infer iteration counts from layer counts.

    Classification is regex-based only, over whatever kernel names the pipeline actually
    enqueues -- it never assumes a specific kernel implementation (CM or otherwise) is
    present. A result is marked incomplete instead of silently assigning events to the wrong
    model when a layer sequence is truncated or interleaved.

    Iteration counting (the `patterns` entry) requires BOTH `main_regex` and `draft_regex`.
    Without a name for a side, this falls back to splitting ALL hot-spot kernels into
    main/draft by call-count divisibility against `main_layers`/`draft_layers` (see
    `_classify_by_call_count`) -- no kernel name needed at all. That split degrades to a
    single unsplit hot-spot table only when the layer counts themselves cannot distinguish
    the two models (missing, zero, or equal).
    """
    raw = [e for e in _absolute(_load_events(trace)) if not MEM_RE.match(e[2])]
    if not raw:
        return {"trace": str(trace), "patterns": [], "warnings": ["no device events"]}
    main_re = re.compile(spec.main_regex, re.I) if spec.main_regex else None
    draft_re = re.compile(spec.draft_regex, re.I) if spec.draft_regex else None
    main = [event for event in raw if main_re and main_re.search(event[2])]
    draft = [event for event in raw if draft_re and draft_re.search(event[2])]
    main_layers = spec.main_full_attention_layers or spec.main_layers
    draft_layers = spec.draft_layers
    out = {"trace": str(trace), "main_layers": main_layers,
           "draft_layers": draft_layers, "patterns": [], "warnings": []}

    if main_re and draft_re:
        phase_names = _speculative_phase_name_sets(_summarize(raw), main_layers, draft_layers)
        windows = _attention_windows(raw, phase_names)
        out["attention_windows"] = windows
        classified = _classify_events_by_windows(raw, windows)
        out["phase_hotspots"] = {name: _summarize(events)[:top] for name, events in classified.items()}
        generation_count = _generation_count(windows)
        out["generation_count"] = generation_count
        out["draft_span_ms"] = _draft_span_ms(windows)
        for bucket in ("main_generate", "draft_generate"):
            if generation_count:
                for row in out["phase_hotspots"].get(bucket, []):
                    row["ms_per_generation"] = row["total_ms"] / generation_count
        intervals = _intervals_between_windows(windows)
        out["intervals"] = intervals
        out["interval_summary"] = _summarize_intervals(intervals)
        if main or draft:
            attention = sorted(main + draft)
            out["patterns"].append({
                "index": 1,
                "main_attention": len(main),
                "draft_attention": len(draft),
                "main_sequences": len(main) / main_layers if main_layers else 0.0,
                "draft_sequences": len(draft) / draft_layers if draft_layers else 0.0,
                "complete": (bool(main_layers) and len(main) % main_layers == 0 and
                             bool(draft_layers) and len(draft) % draft_layers == 0),
                "start_us": attention[0][0],
                "end_us": attention[-1][0] + attention[-1][1],
            })
        else:
            out["warnings"].append(f"no main (/{spec.main_regex}/) or draft "
                                    f"(/{spec.draft_regex}/) attention kernel matched this trace")
        if any(not p["complete"] for p in out["patterns"]):
            out["warnings"].append("one or more speculative patterns have incomplete layer "
                                    "sequences")
        return out

    # At least one regex is missing: the pool that regex would have claimed is "remaining".
    remaining = [event for event in raw if not (main_re and main_re.search(event[2]))
                 and not (draft_re and draft_re.search(event[2]))]
    if spec.main_regex:
        out["warnings"].append(f"main_regex configured (/{spec.main_regex}/) -- main side is "
                                f"an exact count, not call-count classification")
    if spec.draft_regex:
        out["warnings"].append(f"draft_regex configured (/{spec.draft_regex}/) -- draft side "
                                f"is an exact count, not call-count classification")

    can_classify = (bool(main_layers) and bool(draft_layers) and main_layers != draft_layers)
    if not can_classify:
        out["hotspots"] = _summarize(remaining)[:top]
        out["warnings"].append(
            "cannot separate main vs draft by call count (main_layers/draft_layers missing, "
            "zero, or equal) -- showing combined hot-spot kernels only")
        return out

    rows = _summarize(remaining)
    main_rows, draft_rows, other_rows = _classify_by_call_count(rows, main_layers, draft_layers)
    if not spec.main_regex:
        out["main_hotspots"] = main_rows[:top]
        out["warnings"].append(
            f"no main_regex configured -- main hot-spots inferred from call counts divisible "
            f"by main_layers={main_layers} only")
    if not spec.draft_regex:
        out["draft_hotspots"] = draft_rows[:top]
        out["warnings"].append(
            f"no draft_regex configured -- draft hot-spots inferred from call counts "
            f"divisible by draft_layers={draft_layers} only")
    if other_rows:
        out["unclassified_hotspots"] = other_rows[:top]
        out["warnings"].append(
            f"{len(other_rows)} kernel(s) could not be assigned to either model by call "
            f"count (divisible by both or neither layer count)")
    return out


def _render_hotspot_table(title: str, rows: list[dict]) -> list[str]:
    per_generation = bool(rows) and "ms_per_generation" in rows[0]
    header = "  kernel  calls  total_ms  mean_us  p50_us  p90_us"
    if per_generation:
        header += "  ms_per_gen"
    lines = [title, header]
    for r in rows:
        line = (f"  {r['kernel'][:60]:<60} {r['calls']:>6} {r['total_ms']:>9.2f} "
                f"{r['mean_us']:>8.2f} {r['p50_us']:>7.2f} {r['p90_us']:>7.2f}")
        if per_generation:
            line += f" {r['ms_per_generation']:>10.3f}"
        lines.append(line)
    return lines


def render_speculative(result: dict) -> str:
    lines = [f"trace       {result['trace']}",
             f"speculative main_layers={result.get('main_layers', '?')} "
             f"draft_layers={result.get('draft_layers', '?')}"]
    if result.get("patterns"):
        lines.append("pattern main_attention draft_attention main_seq draft_seq status")
        for pattern in result["patterns"]:
            status = "complete" if pattern["complete"] else "INCOMPLETE"
            lines.append(f"{pattern['index']:>7} "
                         f"{pattern['main_attention']:>15} {pattern['draft_attention']:>15} "
                         f"{pattern['main_sequences']:>9.2f} {pattern['draft_sequences']:>10.2f} "
                         f"{status}")
    if result.get("attention_windows"):
        lines.append("execution windows side phase attention_calls span_ms")
        for window in result["attention_windows"]:
            span_ms = (window["end_us"] - window["start_us"]) / 1000.0
            lines.append(f"{window['index']:>7} {window['side']:<5} {window['phase']:<8} "
                         f"{window['attention_calls']:>15} {span_ms:>7.2f}")
    if "generation_count" in result:
        lines.append(f"generations   {result['generation_count']}")
    if result.get("draft_span_ms") is not None:
        lines.append(f"draft span    {result['draft_span_ms']:.2f} ms "
                     f"(avg draft time between two main generations)")
    if result.get("phase_hotspots"):
        for bucket in ("main_prefill", "main_generate", "draft_prefill", "draft_generate"):
            rows = result["phase_hotspots"].get(bucket, [])
            title = bucket.replace("_", " ") + " hot-spot kernels"
            lines += _render_hotspot_table(title, rows)
    if result.get("interval_summary"):
        lines.append("intervals between adjacent main/draft windows")
        lines.append("  from          to            count  mean_gap_ms  p50_gap_ms  max_gap_ms")
        for row in result["interval_summary"]:
            lines.append(f"  {row['from']:<13}{row['to']:<13}{row['count']:>5}"
                         f"{row['mean_gap_ms']:>13.2f}{row['p50_gap_ms']:>12.2f}"
                         f"{row['max_gap_ms']:>12.2f}")
    if "hotspots" in result:
        lines += _render_hotspot_table("hot-spot kernels (no main/draft split)",
                                       result["hotspots"])
    if "main_hotspots" in result:
        lines += _render_hotspot_table("main hot-spot kernels", result["main_hotspots"])
    if "draft_hotspots" in result:
        lines += _render_hotspot_table("draft hot-spot kernels", result["draft_hotspots"])
    if "unclassified_hotspots" in result:
        lines += _render_hotspot_table("unclassified hot-spot kernels (neither model, "
                                       "by call count)", result["unclassified_hotspots"])
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
                         "first_event_us": evs[0][0] if evs else None,
                         "first_body_us": body[0][0] if body else None,
                         "prologue_events": len(prologue),
                         "prologue_ms": sum(d for _, d, _ in prologue) / 1000.0,
                         "mem_ops": len(mem), "mem_ms": sum(d for _, d, _ in mem) / 1000.0},
        "phases": {},
        "warnings": [],
    }
    out["segmentation"]["windows"] = [
        {"start_us": lo, "end_us": hi, "label": "generate"} for lo, hi in windows
    ]
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


def _fmt_timeline_time(offset_us: float) -> str:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    stamp = base + timedelta(microseconds=offset_us)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def _timeline_mermaid(result: dict) -> str:
    lines = [
        "gantt",
        f"title {Path(result['trace']).parent.name} runtime timeline",
        "dateFormat YYYY-MM-DDTHH:mm:ss.SSS",
        "axisFormat %H:%M:%S.%L",
    ]
    if result.get("attention_windows"):
        lines.append("section speculative")
        for window in result["attention_windows"]:
            start = _fmt_timeline_time(window["start_us"])
            end = _fmt_timeline_time(window["end_us"])
            label = f"{window['side']} {window['phase']} #{window['index']}"
            lines.append(f"{label} : {start}, {end}")
        return "\n".join(lines) + "\n"

    seg = result.get("segmentation", {})
    windows = seg.get("windows", [])
    if windows:
        if seg.get("prologue_events") and seg.get("first_event_us") is not None and seg.get("first_body_us") is not None:
            lines.append("section prologue")
            lines.append(
                f"model load / compile : {_fmt_timeline_time(seg['first_event_us'])}, {_fmt_timeline_time(seg['first_body_us'])}"
            )
        if seg.get("first_body_us") is not None and windows[0]["start_us"] > seg["first_body_us"]:
            lines.append("section prefill")
            lines.append(
                f"prefill : {_fmt_timeline_time(seg['first_body_us'])}, {_fmt_timeline_time(windows[0]['start_us'])}"
            )
        lines.append("section generate")
        for index, window in enumerate(windows, 1):
            lines.append(
                f"generate #{index} : {_fmt_timeline_time(window['start_us'])}, {_fmt_timeline_time(window['end_us'])}"
            )
        return "\n".join(lines) + "\n"

    lines.append("section global")
    lines.append("global summary only : milestone, 2025-01-01T00:00:00.000, 0ms")
    return "\n".join(lines) + "\n"


def write_report_artifacts(trace: Path, result: dict, rendered: str) -> dict[str, str]:
    summary = trace.parent / "profile_summary.json"
    report = trace.parent / "profile_report.txt"
    timeline = trace.parent / "profile_timeline.mmd"
    summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report.write_text(rendered + "\n", encoding="utf-8")
    timeline.write_text(_timeline_mermaid(result), encoding="utf-8")
    return {
        "summary": str(summary),
        "report": str(report),
        "timeline": str(timeline),
    }


def load_profile_summary(path: Path) -> dict:
    """Load one saved profile summary from a summary file or a profile output directory."""
    target = path
    if path.is_dir():
        direct = path / "profile_summary.json"
        if direct.exists():
            target = direct
        else:
            hits = sorted(path.rglob("profile_summary.json"))
            if not hits:
                raise FileNotFoundError(f"no profile_summary.json under {path}")
            target = hits[0]
    return json.loads(target.read_text(encoding="utf-8"))


def pick_hot_kernel(summary: dict, pick: int = 1, bucket: str = "") -> dict:
    """Pick one kernel row from a saved profile summary.

    Default priority follows the usual optimization path: steady-state generate before global
    totals, and main-generate before draft on speculative reports.
    """
    if pick < 1:
        raise ValueError("pick must be >= 1")

    candidates: list[tuple[str, list[dict]]] = []
    if bucket:
        if bucket in {"main_prefill", "main_generate", "draft_prefill", "draft_generate"}:
            rows = summary.get("phase_hotspots", {}).get(bucket, [])
            candidates.append((bucket, rows))
        elif bucket in {"prefill", "generate"}:
            rows = summary.get("phases", {}).get(bucket, {}).get("rows", [])
            candidates.append((bucket, rows))
        elif bucket in {"global", "hotspots", "main_hotspots", "draft_hotspots",
                        "unclassified_hotspots"}:
            if bucket == "global":
                rows = summary.get("global", {}).get("rows", [])
            else:
                rows = summary.get(bucket, [])
            candidates.append((bucket, rows))
        else:
            raise ValueError(f"unknown bucket '{bucket}'")
    else:
        ordered = [
            ("main_generate", summary.get("phase_hotspots", {}).get("main_generate", [])),
            ("generate", summary.get("phases", {}).get("generate", {}).get("rows", [])),
            ("main_hotspots", summary.get("main_hotspots", [])),
            ("hotspots", summary.get("hotspots", [])),
            ("global", summary.get("global", {}).get("rows", [])),
        ]
        candidates.extend((name, rows) for name, rows in ordered if rows)

    for source_bucket, rows in candidates:
        if len(rows) >= pick:
            row = dict(rows[pick - 1])
            row["bucket"] = source_bucket
            return row

    available = {name: len(rows) for name, rows in candidates}
    raise ValueError(f"pick={pick} exceeds available rows: {available or {'none': 0}}")


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
