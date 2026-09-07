"""`ckh` -- one command per phase, compact output.

Every subcommand answers with a table or a verdict, never a log. Anything verbose stays in
the measurement environment.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shlex
import sys
from pathlib import Path

from . import bench, equiv, ledger, results
from .platform import REPO_ROOT, Platform


def _spec(name: str):
    try:
        mod = importlib.import_module(f"kernels.{name}")
    except ModuleNotFoundError as e:
        if e.name != f"kernels.{name}":
            raise                       # the spec exists but its own imports are broken
        have = sorted(p.stem for p in (REPO_ROOT / "kernels").glob("*.py")
                      if not p.stem.startswith("_"))
        raise SystemExit(f"no kernel spec '{name}'. have: {', '.join(have) or '(none)'}") from e
    return mod.SPEC, getattr(mod, "DEFAULT_AXES", {})


def cmd_doctor(a) -> int:
    plat = Platform.load()
    print(f"backend        {plat.backend}")
    print(f"sandbox        {plat.sandbox}  {'ok' if plat.sandbox.exists() else 'MISSING'}")
    print(f"production     {plat.production}  {'ok' if plat.production.exists() else 'MISSING'}")
    print(f"noise floor    {plat.noise_floor_pct}%   rounds {plat.rounds}")

    ok = True
    clops_ok, clops_msg = plat.check_clops()
    print(clops_msg)
    ok = ok and clops_ok

    busy = plat.competing_gpu_work()
    if busy:
        print("\ncompeting GPU work -- measurements will be junk until this stops:")
        for b in busy:
            print(f"  {b}")
        ok = False
    else:
        print("competing work none")
    return 0 if ok else 1


def _axes(a, default_axes: dict) -> dict:
    axes = dict(default_axes)
    for kv in a.axis or []:
        k, _, v = kv.partition("=")
        axes[k] = [int(x) if x.lstrip("-").isdigit() else x for x in v.split(",")]
    return axes


def cmd_validate(a) -> int:
    """Everything that can be wrong before a measurement is worth taking. Compiles, never
    enqueues -- so it costs a compile, not a bench round, and it runs while the GPU is busy."""
    from . import validate
    plat = Platform.load()
    spec, default_axes = _spec(a.kernel)
    shapes = bench.expand(_axes(a, default_axes))
    if not shapes:
        print("no shapes: the spec has no DEFAULT_AXES and none were passed with --axis")
        return 1

    static = validate.static_checks(spec, shapes)
    env = None
    if not a.static_only:
        # Deliberately NOT gated on competing_gpu_work: this compiles and never enqueues, so
        # it is the one command that stays useful while the GPU is busy.
        env = validate.env_checks(plat, spec, shapes)
    table, errors = validate.render(static, env)
    print(table)
    return 1 if errors else 0


def cmd_bench(a) -> int:
    plat = Platform.load()
    if plat.competing_gpu_work():
        print("refusing to measure: other GPU work is running (see `ckh doctor`)")
        return 1
    spec, default_axes = _spec(a.kernel)
    axes = _axes(a, default_axes)
    shapes = bench.expand(axes)

    if a.verify:
        # Correctness first, and a failure stops the run. A timing number from a kernel that
        # computes the wrong answer is not a slower-or-faster fact, it is noise with units --
        # and it is the number most likely to be quoted later, because it looks like a result.
        res = equiv.measure(plat, spec, shapes)
        print(equiv.render(spec, res))
        if any(not v.get("passed", False) for v in res.values()):
            print("\nrefusing to time a kernel that fails its own reference check")
            return 1
        print()

    variants = json.loads(a.variants) if a.variants else None
    res = bench.measure(plat, spec, shapes, variants, rounds=a.rounds)
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(bench.render(res, plat.noise_floor_pct))
    return 0


def cmd_gen_reference(a) -> int:
    from . import gen_reference
    try:
        out_path = gen_reference.generate(a.name)
    except gen_reference.Refused as e:
        print(f"refusing to guess: {e}")
        print("this case needs judgment a script shouldn't -- use the reference-generator "
             "agent instead (.claude/agents/reference-generator.md)")
        return 2
    print(f"wrote {out_path}")
    return 0


def _pick_candidate(hits: list, needle: str, no_prompt: bool):
    """Exactly one match is the only case that needs no human. Same isatty guard as
    `ckh profile run`: an unattended caller must fail, not block on stdin forever."""
    if len(hits) == 1:
        return hits[0]
    if not hits:
        print(f"no .cm under the plugin's impls/cm matches '{needle}'. "
              f"Pass --source <path> if you already know the file.")
        return None
    print(f"'{needle}' matches {len(hits)} kernels:")
    for i, p in enumerate(hits, 1):
        print(f"  {i}. {p.name}")
    if no_prompt or not sys.stdin.isatty():
        print("ambiguous, and nobody is here to disambiguate -- pass --source <path>")
        return None
    try:
        choice = input("pick> ").strip()
    except EOFError:
        return None
    return hits[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(hits) else None


def cmd_kernelgen(a) -> int:
    """Port a plugin kernel into the sandbox. OPTIONAL -- skip it if you already have one.

    Runs after `ckh profile` has named a kernel worth the effort. Deliberately generates
    something INCOMPLETE: the parts that can only be guessed are left as failing TODOs rather
    than plausible values, because a spec that runs on invented data hides a wrong kernel
    instead of exposing it.
    """
    from . import kernelgen
    plat = Platform.load()
    try:
        if a.source:
            src_path = Path(a.source)
            if not src_path.exists():
                print(f"{src_path} not found")
                return 1
        else:
            hits = kernelgen.candidates(plat.production, a.kernel)
            src_path = _pick_candidate(hits, a.kernel, a.yes or a.no_prompt)
            if src_path is None:
                return 1

        cmd = kernelgen.cm_dir(plat.production)
        src = kernelgen.parse_cm(src_path, [cmd, cmd / "include"])
        src.resolve_entry(a.kernel)
        generator = a.generator or kernelgen.guess_generator(src_path.stem)
        hints = kernelgen.scrape_host(cmd, generator)
        name = a.name or src_path.stem
        dest = a.dest or plat.kernelgen_dest

        print(f"source     {src_path}")
        print(f"entry      {src.entry}  ({len(src.params)} params)")
        print(f"includes   {len(src.includes)} resolved"
              + (f", MISSING {src.missing_includes}" if src.missing_includes else ""))
        print(f"needs -D   {' '.join(sorted(src.required(hints))) or '(none)'}")
        host_line = (f"{hints.file.name}::{generator}" if hints.found
                     else f"no match for {generator} -- pass --generator")
        print(f"host       {host_line}")
        print(f"-> sandbox {plat.sandbox / dest / (name + '.cm')}")
        print(f"-> spec    kernels/{name}.py")

        if not a.yes:
            if a.no_prompt or not sys.stdin.isatty():
                print("\nnot confirmed (no tty) -- re-run with --yes to write these files")
                return 1
            if input("\nport it? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted, nothing written")
                return 1

        manifest = kernelgen.port(plat.sandbox, dest, src, name, overwrite=a.overwrite)
        spec = kernelgen.emit_spec(REPO_ROOT, name, dest, src, hints, manifest)
        test = kernelgen.emit_test(plat.sandbox, dest, name, src, hints)
    except kernelgen.Refused as e:
        print(f"refusing: {e}")
        return 2

    print(f"\nwrote {manifest['sandbox_kernel']}")
    print(f"wrote {test}")
    print(f"wrote {spec}")
    print(f"\nnext: fill JIT in {test.name} and run `pytest -q {test.name}` in the sandbox -- "
          f"a compile is the first thing a port can actually fail at.\n"
          f"then: .claude/agents/kernel-onboarder.md for dispatch/inputs/args/reference.")
    return 0


def cmd_equiv(a) -> int:
    plat = Platform.load()
    if plat.competing_gpu_work():
        print("refusing to check: other GPU work is running (see `ckh doctor`)")
        return 1
    spec, default_axes = _spec(a.kernel)
    axes = dict(default_axes)
    for kv in a.axis or []:
        k, _, v = kv.partition("=")
        axes[k] = [int(x) if x.lstrip("-").isdigit() else x for x in v.split(",")]
    shapes = bench.expand(axes)
    res = equiv.measure(plat, spec, shapes)
    print(equiv.render(spec, res))
    return 1 if any(not v.get("passed", False) for v in res.values()) else 0


def cmd_snapshot(a) -> int:
    """Freeze the current source so later rounds can A/B against it."""
    plat = Platform.load()
    spec, _ = _spec(a.kernel)
    if a.list:
        print("\n".join(results.snapshots(spec)) or "(none)")
        return 0
    out = results.snapshot(plat, spec, a.tag)
    print(f"snapshot {a.tag} -> {out}")
    return 0


def cmd_round(a) -> int:
    """A/B the working tree against a pinned snapshot, in one interleaved run, on the FULL grid.

    Comparing against a number recorded in a previous session is invalid on a rig that drifts,
    so the previous *source* is re-measured alongside the new one. And the whole grid is
    measured, not just the shape under optimization -- a round that helps one shape and breaks
    another is the mistake this catches.
    """
    plat = Platform.load()
    if plat.competing_gpu_work():
        print("refusing to measure: other GPU work is running (see `ckh doctor`)")
        return 1
    spec, default_axes = _spec(a.kernel)
    base = str(results.snapshot_path(spec, a.against))
    axes = dict(default_axes)
    for kv in a.axis or []:
        k, _, v = kv.partition("=")
        axes[k] = [int(x) if x.lstrip("-").isdigit() else x for x in v.split(",")]
    shapes = bench.expand(axes)

    res = bench.measure(plat, spec, shapes,
                        variants={a.against: {}, "current": {}},
                        sources={a.against: base},   # "current" falls back to the spec source
                        rounds=a.rounds)
    table, worse = bench.render_round(res, a.against, "current", plat.noise_floor_pct)
    print(table)
    results.record(a.kernel, "round", {"against": a.against, "axes": axes,
                                       "regressions": worse, "results": res})
    return 1 if worse else 0


def cmd_log(a) -> int:
    recs = results.history(a.kernel, "round")
    if not recs:
        print("(no rounds recorded)")
        return 0
    for r in recs:
        print(f"{r['ts']}  vs {r['against']:<10} regressions={r['regressions']}  "
              f"shapes={len(r.get('results', {})) // 2}")
    return 0


def _prompt_pipeline(no_prompt: bool) -> list[str]:
    """Ask for the pipeline command, but only when a human is there to answer.

    Without the isatty guard an unattended run (CI, a systemd unit, an agent) would block
    forever on stdin instead of failing, which looks exactly like a hung pipeline.
    """
    if no_prompt or not sys.stdin.isatty():
        return []
    print("No pipeline command configured. Enter the e2e command to profile, or blank to abort.")
    print("  e.g. python benchmark.py -d GPU -m /models/foo -n 3")
    try:
        line = input("pipeline> ").strip()
    except EOFError:
        return []
    if not line:
        return []
    cmd = shlex.split(line)
    print("to make this unattended next time, add to platform.toml under [profile]:")
    print(f"  pipeline = [{', '.join(json.dumps(c) for c in cmd)}]")
    return cmd


def cmd_profile(a) -> int:
    """Where the e2e time actually goes -- run BEFORE choosing a kernel to optimize."""
    from . import clintercept as cli_layer
    plat = Platform.load()
    cfg = plat.raw.get("profile", {})
    prefix = a.prefix or cfg.get("install_prefix") or str(REPO_ROOT / "third_party")

    tool = cli_layer.locate(prefix, a.cli or cfg.get("cliloader"))
    if a.action == "setup":
        if tool is None:
            if not a.install:
                print(f"cl_intercept NOT found under {prefix} or on PATH.")
                print("Re-run with --install (downloads the prebuilt release) or "
                      "--install --from-source.")
                return 2
            print(f"installing cl_intercept into {prefix} ...")
            tool = cli_layer.install(prefix, from_source=a.from_source)
        print(f"cliloader {tool}")
        for m in cli_layer.smoke(tool):
            print(f"smoke     {m}")
        return 0

    if tool is None:
        print("cl_intercept not installed -- run `ckh profile setup --install` first")
        return 2

    if a.action == "run":
        if plat.competing_gpu_work():
            print("refusing to profile: other GPU work is running (see `ckh doctor`)")
            return 1
        # argparse.REMAINDER keeps the separating "--"; passing it on makes cliloader print
        # its help instead of running anything.
        command = a.command[1:] if a.command and a.command[0] == "--" else a.command
        source = "argv"
        if not command:
            command = cfg.get("pipeline") or []
            source = "platform.toml [profile].pipeline"
        if not command:
            command = _prompt_pipeline(a.no_prompt)
            source = "prompt"
        if not command:
            print("no pipeline command. Put it after `--`, or set [profile].pipeline in "
                  "platform.toml for unattended runs.")
            return 1
        print(f"pipeline from {source}: {shlex.join(command)}")
        runs = cli_layer.run(tool, Path(a.out_dir), a.label, command, repeat=a.repeat,
                             env=cfg.get("pipeline_env") or None)
        for r in runs:
            print(f"{r['tag']:<16} exit={r['exit_code']}  trace="
                  f"{'yes' if r['trace_present'] else 'MISSING'}  {r['app_metrics']}")
        print(f"\nnext: ckh profile report --dump-dir {a.out_dir} --kernel <name>")
        return 0 if all(r["exit_code"] == 0 and r["trace_present"] for r in runs) else 1

    seg = cli_layer.Segmentation(gap_ms=a.gap_ms, drop_cycles=a.drop_cycles)
    if a.anchor:
        seg.split_kernel = a.anchor
    traces = sorted(Path(a.dump_dir).rglob("clintercept_trace.json"))
    if not traces:
        print(f"no clintercept_trace.json under {a.dump_dir}")
        return 1
    shares = []
    for t in traces:
        res = cli_layer.analyze(t, a.kernel, seg)
        print(cli_layer.render(res, a.top))
        print()
        k = res.get("kernel")
        if k and k.get("matched"):
            shares.append(k["generate"]["total_ms"])
    if len(shares) > 1 and max(shares) > 0:
        spread = 100.0 * (max(shares) - min(shares)) / max(shares)
        print(f"run-to-run spread of /{a.kernel}/ in generate: {spread:.1f}% over "
              f"{len(shares)} runs -> effects below this are NOT resolvable")
    return 0


def cmd_kernel_profile(a) -> int:
    """Where the kernel spends itself, from the compiler's own output.

    Hardware counters would be better; on the reference box they were unavailable entirely
    (see prof_results/PROFILING.md), and this is what replaced them. It compiles and never
    enqueues, so like `validate` it does not need an idle GPU.
    """
    from . import asm_runner, asmprofile
    plat = Platform.load()

    if a.asm:
        path = Path(a.asm)
        if not path.exists():
            print(f"{path} not found")
            return 1
    else:
        spec, default_axes = _spec(a.kernel)
        shapes = bench.expand(_axes(a, default_axes))
        if not shapes:
            print("no shapes: pass --axis, or profile a dump directly with --asm")
            return 1
        if len(shapes) > 1:
            # A single dump describes one set of -D values. Silently profiling the first of
            # several would attribute one shape's code to all of them.
            print(f"{len(shapes)} shapes selected; an assembly dump describes exactly one "
                  f"set of jit values. Narrow it with --axis.")
            return 1
        cwd = plat.sandbox / (spec.cwd or "opencl/tests/pageatten")
        r = plat.run_module("ckh.asm_runner", [
            "--spec", spec.name,
            "--shape", json.dumps(shapes[0]),
            "--source", str(plat.sandbox / spec.source),
        ], cwd=cwd)
        res = next((json.loads(l[len(asm_runner.RESULT_PREFIX):])
                    for l in r.stdout.splitlines()
                    if l.startswith(asm_runner.RESULT_PREFIX)), None)
        if res is None:
            tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
            print(f"asm runner produced no result.\n{tail}")
            return 1
        if "error" in res:
            print(res["error"])
            return 1
        path = Path(res["asm"])
        print(f"dump       {path}  (shape {bench.shape_label(spec, shapes[0])})\n")

    print(asmprofile.render(asmprofile.parse(path, width=a.width), top=a.top))
    return 0


def _trial_gates(plat, spec, shapes: list[dict]):
    """Bind the three gates to the real commands. Each returns only what the tree records --
    the verbose half stays in the measurement environment, as everywhere else."""
    from . import validate as validate_mod

    def validate_fn(node):
        static = validate_mod.static_checks(spec, shapes)
        env = validate_mod.env_checks(plat, spec, shapes)
        table, errors = validate_mod.render(static, env)
        return errors, table if errors else "clean"

    def equiv_fn(node):
        res = equiv.measure(plat, spec, shapes)
        passed = all(v.get("passed", False) for v in res.values())
        nv = any(v.get("non_vacuous") for v in res.values())
        failed = [k for k, v in res.items() if not v.get("passed", False)]
        return passed, ("all shapes pass" if passed else f"FAILED: {', '.join(failed)}"), nv

    def bench_fn(node, baseline):
        # The baseline's SOURCE is re-measured beside the candidate, never its stored number.
        base_path = str(results.snapshot_path(spec, baseline.tag))
        cand_path = str(results.snapshot_path(spec, node.tag))
        res = bench.measure(plat, spec, shapes,
                            variants={baseline.tag: {}, node.tag: {}},
                            sources={baseline.tag: base_path, node.tag: cand_path})
        table, worse = bench.render_round(res, baseline.tag, node.tag, plat.noise_floor_pct)
        # Scored on the WORST shape, not the average. A candidate that helps one shape and
        # breaks another is a regression, and averaging is exactly what hides it -- the same
        # reason `render_round` reports per shape rather than collapsing the grid.
        per_shape = bench.round_deltas(res, baseline.tag, node.tag, plat.noise_floor_pct)
        if not per_shape:
            raise SystemExit(f"bench produced no comparable pair:\n"
                             f"{bench.render(res, plat.noise_floor_pct)}")
        worst = max(per_shape.values(), key=lambda d: d[0])
        return worst[0], worst[1], {"table": table, "regressions": worse, "results": res}

    return validate_fn, equiv_fn, bench_fn


def cmd_trial(a) -> int:
    from . import trial
    plat = Platform.load()
    spec, default_axes = _spec(a.kernel)

    try:
        if a.action == "init":
            if trial.Tree.path(a.kernel).exists() and not a.force:
                print(f"{a.kernel} already has a trial tree. --force to start over "
                      f"(the old tree is overwritten, its snapshots are not).")
                return 1
            tree = trial.Tree(kernel=a.kernel, max_trials=a.max_trials)
            tag = "trial0"
            results.snapshot(plat, spec, tag)
            tree.add(None, "baseline", tag)
            tree.save()
            print(f"trial tree for {a.kernel}: baseline pinned as {tag}, "
                  f"budget {a.max_trials} trials")
            return 0

        tree = trial.Tree.load(a.kernel)

        if a.action == "status":
            print(trial.render(tree))
            return 0

        if a.action == "new":
            node = tree.add(a.from_id, a.label or "(unlabelled)", tag="")
            node.tag = trial.snapshot_current(plat, spec, a.kernel, node.id)
            tree.save()
            print(f"trial {node.id} created from {a.from_id}, source pinned as {node.tag}")
            print(f"next: ckh trial run {a.kernel} {node.id}")
            return 0

        shapes = trial.expand_shapes(spec, default_axes, a.axis)

        if a.action == "run":
            node = tree.get(a.id)
            v, e, b = _trial_gates(plat, spec, shapes)
            trial.run_gates(tree, node, validate_fn=v, equiv_fn=e, bench_fn=b,
                            noise_floor_pct=plat.noise_floor_pct)
            tree.save()
            print(trial.render(tree))
            table = node.gates.get("bench", {}).get("results", {}).get("table")
            if table:
                # The per-shape table is the part that shows a candidate helping one shape
                # and breaking another; the tree line only carries the worst.
                print(f"\n{table}")
            # A negative is a finding too; the ledger is what stops it being re-prototyped.
            print(f"\nrecord it: ckh ledger {a.kernel} --add change='{node.label}' "
                  f"verdict=<adopted|rejected-measurement|rejected-policy> "
                  f"numbers='{node.gates.get('bench', {}).get('delta_pct', 'n/a')}'")
            return 0 if node.status in trial.BRANCHABLE else 1

        if a.action == "finalize":
            cands = trial.shortlist(tree, a.top)
            if not cands:
                print("no trial is marked improved -- nothing to finalize. The baseline "
                      "stands.")
                return 1
            sources = trial.finalize_configs(tree, cands)
            print(f"re-measuring {len(sources)} sources in ONE interleaved batch: "
                  f"{', '.join(sources)}\n")
            res = bench.measure(
                plat, spec, shapes,
                variants={label: {} for label in sources},
                sources={label: str(results.snapshot_path(spec, tag))
                         for label, tag in sources.items()},
                rounds=a.rounds)
            print(bench.render(res, plat.noise_floor_pct))
            # Score each candidate by its WORST shape relative to the baseline, per shape.
            # Taking each source's best shape would let a candidate that wins one row and
            # loses three be crowned -- the same averaging mistake `render_round` avoids.
            by_shape: dict[str, dict] = {}
            for label, v in res.items():
                if "error" in v:
                    continue
                shape, _, variant = label.rpartition(" | ")
                by_shape.setdefault(shape, {})[variant] = v
            worst: dict = {}
            for shape, variants in by_shape.items():
                base = variants.get(tree.root.tag)
                if base is None:
                    continue
                for name, v in variants.items():
                    ratio = v["min_ms"] / base["min_ms"]
                    prev = worst.get(name)
                    if prev is None or ratio > prev["_ratio"]:
                        worst[name] = {"min_ms": v["min_ms"], "spread_pct": v["spread_pct"],
                                       "_ratio": ratio, "_base_ms": base["min_ms"],
                                       "_shape": shape}
            if tree.root.tag in worst:
                # Normalise so finalize_verdict's baseline comparison stays a like-for-like
                # ratio even though each entry came from a different (worst) shape.
                for name, v in worst.items():
                    v["min_ms"] = v["_ratio"]
                worst[tree.root.tag]["min_ms"] = 1.0
            winner, reason = trial.finalize_verdict(worst, tree.root.tag, plat.noise_floor_pct)
            print(f"\nscored on each candidate's WORST shape vs the baseline")
            for name, v in sorted(worst.items(), key=lambda kv: kv[1]["_ratio"]):
                print(f"  {name:<12} worst shape: {v['_shape']}  "
                      f"{(v['_ratio'] - 1) * 100:+6.1f}%")
            print(f"\n{reason}")
            if winner is None:
                return 1
            print(f"WINNER: {winner}  (snapshot {sources[winner]})")
            return 0
    except trial.TrialError as e:
        print(f"refusing: {e}")
        return 2
    return 1


def cmd_ledger(a) -> int:
    if a.add:
        f = dict(kv.split("=", 1) for kv in a.add)
        ledger.add(a.kernel, change=f["change"], verdict=f["verdict"], shape=f.get("shape", ""),
                   numbers=f.get("numbers", ""), note=f.get("note", ""),
                   bit_exact={"true": True, "false": False}.get(f.get("bit_exact", "").lower()))
        print("recorded")
        return 0
    print(ledger.render(a.kernel, a.only))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="ckh", description="CM kernel optimization harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="validate the environment before measuring").set_defaults(fn=cmd_doctor)

    b = sub.add_parser("bench", help="interleaved min-of-N timing over a shape grid")
    b.add_argument("kernel")
    b.add_argument("--axis", action="append", metavar="K=v1,v2",
                   help="override a shape axis; repeatable")
    b.add_argument("--variants", help='JSON {label: {DEFINE: value}} for an A/B in one batch')
    b.add_argument("--rounds", type=int)
    b.add_argument("--verify", action="store_true",
                   help="run `equiv` first and refuse to report timings if it fails")
    b.add_argument("--json", action="store_true", help="machine-readable, for agents")
    b.set_defaults(fn=cmd_bench)

    va = sub.add_parser("validate",
                        help="static + compile checks before any GPU time is spent")
    va.add_argument("kernel")
    va.add_argument("--axis", action="append", metavar="K=v1,v2")
    va.add_argument("--static-only", action="store_true",
                    help="skip the compile and device query; pure spec arithmetic")
    va.set_defaults(fn=cmd_validate)

    gr = sub.add_parser("gen-reference",
                        help="generate kernels/<name>.py's reference half from "
                             "kernels/pending/<name>_pytorch.py")
    gr.add_argument("name")
    gr.set_defaults(fn=cmd_gen_reference)

    eq = sub.add_parser("equiv", help="check the kernel against its declared reference")
    eq.add_argument("kernel")
    eq.add_argument("--axis", action="append", metavar="K=v1,v2",
                    help="override a shape axis; repeatable")
    eq.set_defaults(fn=cmd_equiv)

    kg = sub.add_parser("kernelgen",
                        help="OPTIONAL: port a plugin kernel into the sandbox + scaffold "
                             "its spec and a compile test")
    kg.add_argument("kernel", help="profiled kernel name, e.g. cm_pa_small_q")
    kg.add_argument("--source", help="skip the search: the .cm to port, by path")
    kg.add_argument("--name", help="sandbox kernel/spec name (default: the .cm's stem)")
    kg.add_argument("--dest", help="override [kernelgen].dest, relative to repos.sandbox")
    kg.add_argument("--generator", help="host generator class to scrape, when the name "
                                        "doesn't follow <Stem>Generator")
    kg.add_argument("--overwrite", action="store_true",
                    help="replace an existing sandbox kernel of the same name")
    kg.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    kg.add_argument("--no-prompt", action="store_true",
                    help="never ask anything; fail instead of blocking")
    kg.set_defaults(fn=cmd_kernelgen)

    sn = sub.add_parser("snapshot", help="pin the current source as a comparison baseline")
    sn.add_argument("kernel")
    sn.add_argument("tag", nargs="?", default="base")
    sn.add_argument("--list", action="store_true")
    sn.set_defaults(fn=cmd_snapshot)

    rd = sub.add_parser("round", help="A/B working tree vs a snapshot, same run, full grid")
    rd.add_argument("kernel")
    rd.add_argument("--against", required=True, metavar="TAG")
    rd.add_argument("--axis", action="append", metavar="K=v1,v2")
    rd.add_argument("--rounds", type=int)
    rd.set_defaults(fn=cmd_round)

    lg = sub.add_parser("log", help="round history for a kernel")
    lg.add_argument("kernel")
    lg.set_defaults(fn=cmd_log)

    pr = sub.add_parser("profile", help="e2e cl_intercept profiling: which kernel, which phase")
    pr.set_defaults(fn=cmd_profile)
    pact = pr.add_subparsers(dest="action", required=True)

    ps = pact.add_parser("setup", help="find cl_intercept, or install it")
    ps.add_argument("--install", action="store_true", help="download and install if missing")
    ps.add_argument("--from-source", action="store_true", help="build instead of using a release")

    prun = pact.add_parser("run", help="run a pipeline under cliloader (command after --)")
    prun.add_argument("--out-dir", default="results/profile")
    prun.add_argument("--label", default="run")
    prun.add_argument("--repeat", type=int, default=2,
                      help="cl_intercept adds its own overhead; one run cannot show it is stable")
    prun.add_argument("--no-prompt", action="store_true",
                      help="never ask for the pipeline command; fail instead")
    prun.add_argument("command", nargs=argparse.REMAINDER,
                      help="the pipeline command, verbatim, after --")

    prep = pact.add_parser("report", help="per-phase kernel budget + Amdahl ceiling")
    prep.add_argument("--dump-dir", required=True)
    prep.add_argument("--kernel", default="", help="regex for the kernel under investigation")
    prep.add_argument("--top", type=int, default=15)
    prep.add_argument("--anchor", default="", help="regex marking the generate phase")
    prep.add_argument("--gap-ms", type=float, default=50.0)
    prep.add_argument("--drop-cycles", type=int, default=1)

    for p in (ps, prun, prep):
        p.add_argument("--cli", help="path to cliloader")
        p.add_argument("--prefix", help="install prefix to search")

    kp = sub.add_parser("kernel-profile",
                        help="where the kernel spends itself, from the IGC assembly dump")
    kp.add_argument("kernel", nargs="?", default="",
                    help="spec to compile and dump; omit when using --asm")
    kp.add_argument("--axis", action="append", metavar="K=v1,v2",
                    help="a dump describes ONE set of jit values, so this must select one shape")
    kp.add_argument("--asm", help="skip the compile: analyse an existing .asm dump")
    kp.add_argument("--width", type=int, default=16,
                    help="ALU lanes; an exec-N op costs ceil(N/width) cycles")
    kp.add_argument("--top", type=int, default=12)
    kp.set_defaults(fn=cmd_kernel_profile)

    tr = sub.add_parser("trial", help="branching trial tree: validate -> equiv -> bench, "
                                      "with an honest re-measured finalize")
    tr.set_defaults(fn=cmd_trial)
    tact = tr.add_subparsers(dest="action", required=True)

    ti = tact.add_parser("init", help="pin the current source as the baseline")
    ti.add_argument("kernel")
    ti.add_argument("--max-trials", type=int, default=10,
                    help="budget; an open-ended loop is how a week goes into a 2% kernel")
    ti.add_argument("--force", action="store_true", help="overwrite an existing tree")

    tn = tact.add_parser("new", help="pin the current source as a new trial")
    tn.add_argument("kernel")
    tn.add_argument("--from", dest="from_id", type=int, default=0,
                    metavar="ID", help="parent trial; 0 is the baseline")
    tn.add_argument("--label", help="what this candidate changes")

    trn = tact.add_parser("run", help="run the gates on a trial, in cost order")
    trn.add_argument("kernel")
    trn.add_argument("id", type=int)

    tst = tact.add_parser("status", help="the tree so far")
    tst.add_argument("kernel")

    tf = tact.add_parser("finalize", help="re-measure the shortlist and the baseline together")
    tf.add_argument("kernel")
    tf.add_argument("--top", type=int, default=3)
    tf.add_argument("--rounds", type=int)

    for p in (ti, tn, trn, tst, tf):
        p.add_argument("--axis", action="append", metavar="K=v1,v2")

    l = sub.add_parser("ledger", help="prior findings -- read this BEFORE proposing a change")
    l.add_argument("kernel")
    l.add_argument("--only", help="filter by verdict prefix, e.g. rejected")
    l.add_argument("--add", nargs="+", metavar="k=v",
                   help="change=.. verdict=.. shape=.. numbers=.. [bit_exact=..] [note=..]")
    l.set_defaults(fn=cmd_ledger)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
