"""`ckh kernelgen` -- port a production kernel source into the sandbox so it can be iterated on.

`ckh profile` names the kernel worth optimizing, but that name is an OpenCL kernel name from a
device timeline; the thing you actually edit is a kernel source file in the plugin tree, and
it will not compile outside it (it pulls plugin-private headers and every one of its shape
constants arrives as a `-D` from host C++). Turning that into "a kernel I can rebuild in
seconds" was previously a manual half-day of grepping, and it was redone from scratch for
each kernel.

This module does the mechanical part of that and refuses the rest:

  DERIVED (trustworthy)   entry signature, transitive include closure, which macros the source
                          references without defining.
  SCRAPED (a GUESS)       jit constant *values* and gws/lws, regex-lifted from the generator
                          class in the host .cpp. Real C++ -- branches on runtime state,
                          reads env vars, calls helpers. Every scraped value is emitted
                          commented with its originating expression and marked GUESS; nothing
                          silently becomes a number.
  REFUSED                 input tensors. There is no honest way to invent them, so the
                          generated test skips its launch case rather than fabricating data
                          that would make a wrong kernel look fine.

The step is OPTIONAL: if a sandbox kernel already exists, skip it and write `kernels/<n>.py`
by hand (or with `kernel-onboarder`) against the path you already have.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

CM_SUBDIR = "src/plugins/intel_gpu/src/graph/impls/cm"
CL_SUBDIR = "src/plugins/intel_gpu/src/kernel_selector/cl_kernels"
SOURCE_EXTS = (".cm", ".cl")

# `#if`/`#define` bodies are full of these; they are not kernel shape constants.
_NOT_A_DEFINE = {"defined", "sizeof", "true", "false", "if", "else", "elif", "endif"}
# `#define KV_ELEMENT_TYPE uint8_t` puts a type in a macro body. It is not a missing -D.
_TYPE_TOKENS = {
    "char", "short", "int", "long", "float", "double", "void", "bool", "half", "uchar",
    "ushort", "uint", "ulong", "signed", "unsigned", "const", "static", "inline",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
}


class Refused(Exception):
    """This case needs judgment a regex shouldn't fake."""


def runtime_key(name: str) -> str:
    """Normalise a runtime kernel name for source lookup.

    OpenVINO runtime names often carry a generated numeric suffix and a shape-adaptation tag
    (`__sa`). Those identify one instantiated entry point, but the owning source is keyed by
    the stable base name.
    """
    key = runtime_symbol(name).lower()
    key = re.sub(r"^cm_", "", key)
    key = re.sub(r"__sa$", "", key)
    key = re.sub(r"_[0-9]{6,}$", "", key)
    return key


def runtime_symbol(name: str) -> str:
    """Strip profiler-added launch metadata and keep only the runtime entry name."""
    token = (name or "").strip().split()[0] if (name or "").strip() else ""
    return token.rstrip(":")


# -- 1. what the source itself says (derived, trustworthy) ------------------------------

@dataclass
class Param:
    ctype: str
    name: str
    guard: str = ""          # the #if condition this parameter sits under, if any


@dataclass
class CmSource:
    path: Path
    entry: str
    params: list[Param]
    body: str = ""                 # this file plus its include closure, comments stripped
    includes: list[Path] = field(default_factory=list)
    missing_includes: list[str] = field(default_factory=list)
    defined: set[str] = field(default_factory=set)
    referenced: set[str] = field(default_factory=set)

    def resolve_entry(self, profiled: str) -> None:
        """Plugin kernels are named by a macro (`void KERNEL_NAME(...)`) that the host fills
        in per instantiation, so the source alone cannot say what the kernel is called. The name
        `ckh profile` reported IS that value -- adopt it rather than emitting a spec whose
        entry point is the literal string "KERNEL_NAME"."""
        if self.entry == "KERNEL_NAME" and profiled:
            self.entry = profiled.strip()

    @property
    def undefined(self) -> set[str]:
        """Macros used in preprocessor context but never defined -- must arrive as `-D`."""
        return self.referenced - self.defined - _NOT_A_DEFINE - _TYPE_TOKENS

    def required(self, hints: "HostHints") -> set[str]:
        """The full `-D` list, from two independent directions.

        `undefined` alone misses every constant used only in ordinary code (HEAD_SIZE,
        SCALE_FACTOR): those never appear in an `#if`, so no preprocessor scan can see them.
        The host's jit names cover exactly those -- but the host also sets constants for
        sibling kernels, so each is admitted only if this source actually mentions it.
        Neither half is sufficient; the union is.
        """
        used = {m for m in hints.jit
                if re.search(rf"\b{re.escape(m)}\b", self.body)}
        return (self.undefined | used) - self.defined


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _scan_macros(text: str) -> tuple[set[str], set[str]]:
    """(defined, referenced). `referenced` is deliberately restricted to preprocessor
    context -- `#if` conditions and `#define` bodies -- rather than every identifier in the
    file. A whole-file identifier sweep drowns the real shape constants in CM builtins and
    local variables, and a list nobody reads is the same as no list."""
    defined = set(re.findall(r"^\s*#\s*define\s+([A-Za-z_]\w*)", text, re.M))
    referenced: set[str] = set()
    for cond in re.findall(r"^\s*#\s*(?:if|elif)\s+([^\n]*)", text, re.M):
        referenced |= set(re.findall(r"[A-Za-z_]\w*", cond))
    for name in re.findall(r"^\s*#\s*(?:ifdef|ifndef)\s+([A-Za-z_]\w*)", text, re.M):
        referenced.add(name)
    for body in re.findall(r"^\s*#\s*define\s+[A-Za-z_]\w*(?:\([^)]*\))?[ \t]+([^\n]*)",
                           text, re.M):
        referenced |= set(re.findall(r"[A-Za-z_]\w*", body))
    return defined, referenced


def _parse_entry(text: str) -> tuple[str, list[Param]]:
    # Dumped OpenVINO CL sources often name the runtime kernel in a macro define and then
    # declare it as `KERNEL(foo)(...)`. Prefer that over the first helper `void` function in
    # the file, or the parser will latch onto utilities like block_store().
    dumped_kernel = re.search(
        r'^\s*#\s*define\s+KERNEL\(name\)\s+__kernel\s+void\s+([A-Za-z_]\w*)',
        text,
        re.M,
    )
    dumped_decl = re.search(r'^\s*KERNEL\([A-Za-z_]\w*\)\s*\((.*?)\)\s*\{', text, re.S | re.M)
    if dumped_kernel and dumped_decl:
        entry, blob = dumped_kernel.group(1), dumped_decl.group(1)
    else:
        patterns = [
            r'extern\s+"C"\s+_GENX_MAIN_\s+void\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*\{',
            r'(?:(?:__kernel|kernel)\s+)?void\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*\{',
        ]
        m = None
        for pattern in patterns:
            m = re.search(pattern, text, re.S)
            if m:
                break
        if not m:
            raise Refused("no recognized kernel entry (`_GENX_MAIN_ void` or `__kernel void`) "
                          "found -- this source uses a convention this parser doesn't know")
        entry, blob = m.group(1), m.group(2)

    params, guard = [], ""
    for raw in blob.split("\n"):
        line = raw.strip()
        if line.startswith("#if"):
            guard = line.split(None, 1)[1] if " " in line else line
            continue
        if line.startswith(("#endif", "#else", "#elif")):
            guard = ""
            continue
        line = line.rstrip(",").strip()
        if not line:
            continue
        # `half* query [[type("svmptr_t")]]` -> ctype `half*`, name `query`
        line = re.sub(r"\[\[.*?\]\]", "", line).strip()
        line = re.sub(r"__attribute__\s*\(\(.*?\)\)", "", line).strip()
        p = re.match(r"^(.*?[\s*])([A-Za-z_]\w*)$", line)
        if p:
            params.append(Param(ctype=p.group(1).strip(), name=p.group(2), guard=guard))
    if not params:
        raise Refused(f"entry `{entry}` parsed with zero parameters -- refusing to emit a "
                      f"descriptor whose args() would be empty")
    return entry, params


def parse_source(path: Path, include_dirs: list[Path]) -> CmSource:
    """Parse a kernel source entry signature and walk the include closure."""
    text = _strip_comments(path.read_text(errors="replace"))
    entry, params = _parse_entry(text)
    src = CmSource(path=path, entry=entry, params=params, body=text)
    src.defined, src.referenced = _scan_macros(text)

    seen: set[Path] = {path.resolve()}
    queue = [(path, text)]
    while queue:
        owner, body = queue.pop()
        for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"', body, re.M):
            hit = next((d / inc for d in [owner.parent, *include_dirs] if (d / inc).exists()),
                       None)
            if hit is None:
                src.missing_includes.append(inc)
                continue
            hit = hit.resolve()
            if hit in seen:
                continue
            seen.add(hit)
            src.includes.append(hit)
            sub = _strip_comments(hit.read_text(errors="replace"))
            d, r = _scan_macros(sub)
            src.defined |= d
            src.referenced |= r
            src.body += "\n" + sub
            queue.append((hit, sub))
    return src


def parse_cm(path: Path, include_dirs: list[Path]) -> CmSource:
    return parse_source(path, include_dirs)


# -- 2. what the host .cpp hints at (scraped, a GUESS) ---------------------------------

@dataclass
class HostHints:
    file: Path | None = None
    generator: str = ""
    jit: dict[str, str] = field(default_factory=dict)      # name -> the C++ expression
    gws: str = ""
    lws: str = ""
    args: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.jit or self.gws or self.args)


def _generator_block(text: str, cls: str, method: str) -> str:
    m = re.search(rf"{re.escape(cls)}::{re.escape(method)}\b", text)
    if not m:
        return ""
    # Brace-match from the method's opening `{` so a sibling generator's body is never
    # attributed to this kernel -- these files hold half a dozen generators side by side.
    start = text.find("{", m.end())
    if start < 0:
        return ""
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return ""


def scrape_host(source_root: Path, generator: str) -> HostHints:
    """Regex-lift jit constants, dispatch and argument order out of a generator class.

    Everything here is a GUESS by construction: `get_jit_constants` is C++ that branches on
    descriptor state and reads env vars, so the expression at a call site is frequently not
    the value that reaches the kernel at runtime. Emitted as commented expressions, never as
    bare numbers.
    """
    hints = HostHints(generator=generator)
    for cpp in sorted(source_root.rglob("*.cpp")):
        text = _strip_comments(cpp.read_text(errors="replace"))
        if f"{generator}::" not in text:
            continue
        hints.file = cpp
        jit_body = _generator_block(text, generator, "get_jit_constants")
        for name, expr in re.findall(r'jit\.make\(\s*"([^"]+)"\s*,\s*(.+?)\s*\)\s*;', jit_body):
            hints.jit[name] = expr
        for name, expr in re.findall(
                r'make_jit_constant\(\s*"([^"]+)"\s*,\s*(.+?)\s*\)\s*\)\s*;', jit_body):
            hints.jit.setdefault(name, expr)

        disp = _generator_block(text, generator, "get_dispatch_data_func")
        g = re.search(r"wgs\.global\s*=\s*\{(.*?)\}\s*;", disp, re.S)
        l = re.search(r"wgs\.local\s*=\s*\{(.*?)\}\s*;", disp, re.S)
        hints.gws = " ".join(g.group(1).split()) if g else ""
        hints.lws = " ".join(l.group(1).split()) if l else ""

        argb = _generator_block(text, generator, "get_arguments_desc")
        hints.args = [" ".join(a.split())
                      for a in re.findall(r"args\.push_back\(\s*\{(.*?)\}\s*\)\s*;", argb, re.S)]
        break
    return hints


def guess_generator(stem: str) -> str:
    """`pa_small_q` -> `PagedAttentionGeneratorSmallQ` is not derivable; this only covers the
    plain `Foo` -> `FooGenerator` shape. Callers pass --generator when it doesn't fit."""
    return "".join(p.capitalize() for p in stem.split("_")) + "Generator"


# -- 3. candidate discovery ------------------------------------------------------------

def cm_dir(production: Path) -> Path:
    d = production / CM_SUBDIR
    if not d.exists():
        raise Refused(f"{d} does not exist -- is repos.production really an openvino "
                      f"checkout with the intel_gpu CM impls?")
    return d


def source_dirs(production: Path) -> list[Path]:
    candidates = [production / CM_SUBDIR, production / CL_SUBDIR]
    hits = [p for p in candidates if p.exists()]
    if hits:
        return hits
    raise Refused(
        "no known kernel source directories exist under repos.production; expected at least "
        f"one of {[str(p) for p in candidates]}"
    )


def include_dirs_for(src_path: Path, production: Path) -> list[Path]:
    dirs = [src_path.parent]
    for root in source_dirs(production):
        if root not in dirs:
            dirs.append(root)
        inc = root / "include"
        if inc.exists() and inc not in dirs:
            dirs.append(inc)
    return dirs


def candidates(production: Path, needle: str) -> list[Path]:
    """Match a profiled OpenCL kernel name against known kernel source files.

    The profiled name (`cm_pa_small_q`) is the plugin's entry point, not the filename
    (`pa_small_q.cm`), so a plain equality test finds nothing. Substring both ways.
    """
    key = runtime_key(needle)
    hits = []
    seen: set[Path] = set()
    for root in source_dirs(production):
        for ext in SOURCE_EXTS:
            for p in sorted(root.glob(f"*{ext}")):
                if p in seen:
                    continue
                seen.add(p)
                stem = p.stem.lower()
                if key == stem or key in stem or stem in key:
                    hits.append(p)
    return hits


def dump_candidates(dump_root: Path, needle: str) -> list[Path]:
    """Match a profiled runtime kernel name against OV_GPU_DUMP_SOURCES_PATH outputs.

    A dumped source came from the runtime instance that actually ran, so it is the strongest
    source anchor available when the profiled name is a generated entry rather than a file stem.
    """
    if not dump_root.exists():
        return []
    key = runtime_key(needle)
    hits: list[Path] = []
    seen: set[Path] = set()
    for ext in SOURCE_EXTS:
        for p in sorted(dump_root.rglob(f"*{ext}")):
            if p in seen:
                continue
            seen.add(p)
            stem = runtime_key(p.stem)
            if key == stem or key in stem or stem in key:
                hits.append(p)
                continue
            try:
                src = parse_source(p, [p.parent])
            except Exception:
                continue
            entry = runtime_key(src.entry)
            if key == entry or key in entry or entry in key:
                hits.append(p)
    return hits


def configured_dump_sources(raw: dict, override: str | None = None) -> Path | None:
    """Resolve the source-dump directory from an explicit override or profile env."""
    if override:
        return Path(override)
    profile = raw.get("profile", {}) if isinstance(raw, dict) else {}
    run_meta = profile.get("last_run_meta") if isinstance(profile, dict) else None
    if isinstance(run_meta, dict) and run_meta.get("dump_sources"):
        return Path(run_meta["dump_sources"])
    env = profile.get("pipeline_env", {}) if isinstance(profile, dict) else {}
    path = env.get("OV_GPU_DUMP_SOURCES_PATH") if isinstance(env, dict) else None
    return Path(path) if path else None


def dump_sources_from_profile_dir(profile_dir: Path) -> Path | None:
    meta = profile_dir / "profile_run.json"
    if not meta.exists():
        return None
    try:
        data = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return None
    dump_sources = data.get("dump_sources")
    return Path(dump_sources) if dump_sources else None


# -- 4. porting ------------------------------------------------------------------------

def port(sandbox: Path, dest_rel: str, src: CmSource, name: str,
         overwrite: bool = False) -> dict:
    """Copy the kernel and its include closure into the sandbox, with provenance.

    Provenance matters later: `integrator` has to diff the sandbox kernel against the shipped
    one code-only, and a copy with no record of where it came from turns that into guesswork.
    """
    dest = sandbox / dest_rel
    dest.mkdir(parents=True, exist_ok=True)
    source_ext = src.path.suffix or ".cl"
    kernel_dst = dest / f"{name}{source_ext}"
    if kernel_dst.exists() and not overwrite:
        raise Refused(f"{kernel_dst} already exists. Re-run with --overwrite to replace it, "
                      f"or pick another --name -- clobbering a kernel you have already been "
                      f"editing is not something this should do quietly.")

    inc_dst = dest / f"{name}_include"
    copied = []
    if src.includes:
        inc_dst.mkdir(exist_ok=True)
        for inc in src.includes:
            shutil.copy2(inc, inc_dst / inc.name)
            copied.append(inc.name)

    banner = (f"// PORTED by `ckh kernelgen` from {src.path}\n"
              f"// This is a COPY. The shipped kernel is the one above; they will diverge.\n"
              f"// Ledger findings belong in BOTH -- see .github/agents/ckh-integrator.agent.md\n")
    kernel_dst.write_text(banner + src.path.read_text(errors="replace"))

    manifest = {
        "name": name,
        "origin": str(src.path),
        "source_ext": source_ext,
        "entry": src.entry,
        "sandbox_kernel": str(kernel_dst),
        "include_dir": str(inc_dst) if copied else "",
        "includes": copied,
        "missing_includes": src.missing_includes,
        "undefined_macros": sorted(src.undefined),        "params": [{"ctype": p.ctype, "name": p.name, "guard": p.guard} for p in src.params],
    }
    (dest / f"{name}.kernelgen.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def normalize_axes(axes: dict[str, object] | None) -> dict[str, list[object]]:
    """Normalise axis overrides into value lists for emitted scaffolds.

    The preparation step records the user's current focus so the generated files start from
    the profiled shape instead of a blank slate.
    """
    out: dict[str, list[object]] = {}
    for key, value in (axes or {}).items():
        vals = value if isinstance(value, list) else str(value).split(",")
        out[key] = [int(x) if str(x).lstrip("-").isdigit() else x for x in vals]
    return out


# -- 5. emission -----------------------------------------------------------------------

def _jit_lines(src: CmSource, hints: HostHints) -> list[str]:
    """One line per macro the kernel needs, value left as a TODO with the host expression
    alongside it. Without every one of these the kernel does not compile, so the list being
    complete matters more than it being short."""
    lines = []
    for macro in sorted(src.required(hints)):
        expr = hints.jit.get(macro)
        note = f"  # GUESS from host: {expr}" if expr else "  # not found in host .cpp"
        lines.append(f'        # "{macro}": TODO,{note}')
    extra = sorted(set(hints.jit) - src.required(hints))
    if extra:
        lines.append("        # -- host sets these but this source never mentions them "
                     "(a sibling kernel's):")
        lines += [f'        #   "{m}" = {hints.jit[m]}' for m in extra]
    return lines


def emit_spec(repo_root: Path, name: str, dest_rel: str, src: CmSource, hints: HostHints,
              manifest: dict, focus_axes: dict[str, list[object]] | None = None) -> Path:
    out = repo_root / "kernels" / f"{name}.py"
    origin_text = json.dumps(manifest["origin"])
    args_lines = [
        f'        # {p.ctype} {p.name}' + (f'   [only when {p.guard}]' if p.guard else '')
        for p in src.params
    ]
    host_args = ("\n".join(f"    #   {a}" for a in hints.args)
                 if hints.args else "    #   (not found)")

    body = f'''"""Generated by `ckh kernelgen` from {origin_text}.

DERIVED and trustworthy: the entry signature below, and the macro list in jit() -- both read
straight out of the .cm.

GUESS: every value marked `# GUESS from host`. Lifted by regex from
{hints.file.name if hints.file else "(no host .cpp matched)"}::{hints.generator}, which is C++ that branches on
runtime descriptor state. Treat them as leads, not facts.

MISSING: inputs(). There is no honest way to generate it -- see the module docstring in
src/ckh/kernelgen.py. Until it is written, `ckh bench`/`ckh equiv` on this spec will fail
loudly, which is the intended behaviour; a spec that runs on fabricated data would make a
wrong kernel look correct.

Next: .github/agents/ckh-kernel-onboarder.agent.md picks up from here.
"""
from __future__ import annotations

from ckh.kernel import KernelSpec, Shape

SOURCE = "{dest_rel}/{name}{manifest.get('source_ext', '.cl')}"
ENTRY = "{src.entry}"


def jit(s: Shape) -> dict:
    """Macros the source references but never defines -- without these it will not compile."""
    return {{
        "KERNEL_NAME": ENTRY,
{chr(10).join(_jit_lines(src, hints))}
    }}


def dispatch(s: Shape):
    """(gws, lws).

    host gws = {hints.gws or "(not found)"}
    host lws = {hints.lws or "(not found)"}
    Those are C++ expressions over runtime params, not numbers. Resolve them against this
    spec's Shape axes yourself; a wrong dispatch does not fail loudly, it measures garbage.
    """
    raise NotImplementedError("dispatch: translate the host expressions above into Shape terms")


def inputs(s: Shape) -> dict:
    """MUST be deterministic per shape -- a generator that re-randomises between the two
    sides of an A/B once produced 0/96 spurious mismatches that looked exactly like a kernel
    bug. Cache it (see kernels/pa_small_q.py)."""
    raise NotImplementedError("inputs: no honest way to generate this -- see kernel-onboarder")


def outputs(s: Shape) -> dict:
    raise NotImplementedError("outputs: one entry per buffer `compare` should read back")


def args(s: Shape, data: dict, out: dict) -> list:
    """Signature order, straight from the source entry point:
{chr(10).join(args_lines)}

    host argument descriptors:
{host_args}
    """
    raise NotImplementedError("args: fill in, in the signature order listed above")


SPEC = KernelSpec(
    name="{name}",
    source=SOURCE,
    prod_source="{Path(manifest["origin"]).name}",
    entry=ENTRY,
    jit=jit,
    dispatch=dispatch,
    args=args,
    inputs=inputs,
    outputs=outputs,
    compare=[],          # buffer names to diff in `ckh equiv`
    reference=None,      # TorchReference or KernelReference -- see kernel-onboarder step 2
    cwd="{dest_rel}",
    label_keys=[],
)

DEFAULT_AXES: dict = {focus_axes or {}!r}
'''
    out.write_text(body)
    return out


def emit_test(sandbox: Path, dest_rel: str, name: str, src: CmSource, hints: HostHints) -> Path:
    """A sandbox pytest whose FIRST case is real work: does the port compile at all?

    That question is answerable with nothing but the macro list, and it is where a port
    actually fails -- a plugin-private header that did not come along, a macro the host
    supplies that nobody wrote down. The launch case is skipped, explicitly, rather than run
    against invented buffers.
    """
    out = sandbox / dest_rel / f"test_{name}.py"
    body = f'''#!/usr/bin/env python3
"""Generated by `ckh kernelgen`: can the ported {name}{src.path.suffix} be built here at all?

Run:  pytest -q test_{name}.py
"""
import pytest

from {name}_wrapper import ENTRY, Runner, build, missing_jit


def test_compiles():
    missing = missing_jit()
    assert not missing, f"fill in JIT values first: {{missing}}"
    build()


@pytest.mark.skip(reason="needs real input buffers -- ckh kernelgen refuses to invent them; "
                         "see .github/agents/ckh-kernel-onboarder.agent.md")
def test_runs():
    runner = Runner(gws=GWS, lws=LWS)  # noqa: F821
    runner.enqueue(*ARGS)  # noqa: F821
    assert ENTRY == "{src.entry}"
'''
    out.write_text(body)
    return out


def emit_wrapper(sandbox: Path, dest_rel: str, name: str, src: CmSource, hints: HostHints,
                 focus_axes: dict[str, list[object]] | None = None) -> Path:
    out = sandbox / dest_rel / f"{name}_wrapper.py"
    defines = "\n".join(f'    "{m}": None,' + (f"  # GUESS from host: {hints.jit[m]}"
                                               if m in hints.jit else "")
                        for m in sorted(src.required(hints)))
    args_note = "\n".join(
        f"#   {p.ctype} {p.name}" + (f"   [only when {p.guard}]" if p.guard else "")
        for p in src.params
    ) or "#   (no parameters parsed)"
    body = f'''#!/usr/bin/env python3
"""Generated by `ckh kernelgen`: aboutSHW wrapper scaffold for {name}{src.path.suffix}.

This is the sandbox-side anchor the correctness and perf scaffolds import. It deliberately
stops before inventing host inputs or dispatch policy.
"""
from __future__ import annotations

from pathlib import Path

from clops import cl

HERE = Path(__file__).resolve().parent
INCLUDE_DIR = HERE / "{name}_include"
SOURCE = HERE / "{name}{src.path.suffix}"
ENTRY = "{src.entry}"
FOCUS_AXES = {focus_axes or {}!r}

# Every macro the kernel references but never defines. Fill in the values before the first
# real compile; this wrapper keeps them in one place so correctness and perf use the same
# build.
JIT = {{
    "KERNEL_NAME": ENTRY,
{defines}
}}

# Host generator breadcrumbs, copied verbatim from the scrape step.
# host gws = {hints.gws or "(not found)"}
# host lws = {hints.lws or "(not found)"}
# signature order:
{args_note}


def missing_jit() -> list[str]:
    return [k for k, v in JIT.items() if v is None or v == ""]


def build(extra_flags: str = ""):
    missing = missing_jit()
    if missing:
        raise ValueError(f"fill in JIT values first: {{missing}}")
    opts = " ".join([
        "-cmc",
        f"-I{{HERE}}",
        f"-I{{INCLUDE_DIR}}",
        *(f"-D{{k}}={{v}}" for k, v in JIT.items()),
        extra_flags,
    ]).strip()
    return cl.kernels(f'#include "{name}{src.path.suffix}"', opts)


class Runner:
    def __init__(self, gws=None, lws=None, extra_flags: str = ""):
        self.gws = gws
        self.lws = lws
        self.extra_flags = extra_flags

    def create_kernels(self):
        return build(self.extra_flags)

    def enqueue(self, *kernel_args):
        if self.gws is None or self.lws is None:
            raise NotImplementedError(
                "fill gws/lws from the owning host generator before launching the wrapper"
            )
        kernels = self.create_kernels()
        kernels.enqueue(ENTRY, self.gws, self.lws, *kernel_args)
        return cl.finish()
'''
    out.write_text(body)
    return out


def emit_correctness_test(sandbox: Path, dest_rel: str, name: str,
                          focus_axes: dict[str, list[object]] | None = None) -> Path:
    out = sandbox / dest_rel / f"test_{name}_correctness.py"
    axes = focus_axes or {}
    body = f'''#!/usr/bin/env python3
"""Generated by `ckh kernelgen`: correctness scaffold for {name}_wrapper.

Shared-input A/B and a non-vacuity proof are mandatory. This file points at the exact places
that still need human judgment instead of fabricating them.
"""
from __future__ import annotations

import pytest

from {name}_wrapper import FOCUS_AXES, Runner

TARGET_Q_LENS = tuple(FOCUS_AXES.get("q_len", {axes.get("q_len", [6, 16])!r}))
TARGET_PAST_LENS = tuple(FOCUS_AXES.get("past_len", {axes.get("past_len", [15360])!r}))


def build_case(q_len: int, past_len: int):
    raise NotImplementedError(
        "reuse the owning aboutSHW/OpenVINO input builder here; kernelgen refuses to invent it"
    )


def reference(case):
    raise NotImplementedError(
        "fill with an independent reference, not the kernel under test or a value derived from it"
    )


@pytest.mark.skip(reason="fill build_case/reference/dispatch/args first; shared inputs and "
                         "a non-vacuity proof are mandatory")
@pytest.mark.parametrize("q_len", TARGET_Q_LENS)
@pytest.mark.parametrize("past_len", TARGET_PAST_LENS)
def test_{name}_correctness(q_len: int, past_len: int):
    case = build_case(q_len=q_len, past_len=past_len)
    expected = reference(case)
    runner = Runner(gws=GWS, lws=LWS)  # noqa: F821
    actual = runner.enqueue(*ARGS)  # noqa: F821
    assert expected is not None and actual is not None


@pytest.mark.skip(reason="prove the test can fail before trusting a pass")
def test_{name}_non_vacuous():
    raise NotImplementedError("mutate one causal input and show the result changes")
'''
    out.write_text(body)
    return out


def emit_perf_test(sandbox: Path, dest_rel: str, name: str,
                   focus_axes: dict[str, list[object]] | None = None) -> Path:
    out = sandbox / dest_rel / f"test_{name}_perf.py"
    axes = focus_axes or {}
    body = f'''#!/usr/bin/env python3
"""Generated by `ckh kernelgen`: perf scaffold for {name}_wrapper.

Run only after correctness passes. Keep perf guarded by an env var so normal test runs do not
spend GPU time.
"""
from __future__ import annotations

import os

import pytest

from {name}_wrapper import FOCUS_AXES, Runner

TARGET_Q_LENS = tuple(FOCUS_AXES.get("q_len", {axes.get("q_len", [6, 16])!r}))
TARGET_PAST_LENS = tuple(FOCUS_AXES.get("past_len", {axes.get("past_len", [15360])!r}))


def build_case(q_len: int, past_len: int):
    raise NotImplementedError("reuse the real input builder; do not invent perf-only inputs")


def time_case(runner: Runner, case):
    raise NotImplementedError("fill with the owning aboutSHW timing loop for this kernel family")


@pytest.mark.skip(reason="fill build_case/time_case/dispatch/args first")
@pytest.mark.parametrize("q_len", TARGET_Q_LENS)
@pytest.mark.parametrize("past_len", TARGET_PAST_LENS)
def test_{name}_perf(q_len: int, past_len: int):
    if os.environ.get("RUN_CKH_PERF", "0") != "1":
        pytest.skip("Set RUN_CKH_PERF=1 to enable perf scaffolds")
    case = build_case(q_len=q_len, past_len=past_len)
    runner = Runner(gws=GWS, lws=LWS)  # noqa: F821
    result = time_case(runner, case)
    assert result is not None
'''
    out.write_text(body)
    return out


def prepare(repo_root: Path, sandbox: Path, dest_rel: str, src: CmSource, name: str,
            hints: HostHints, overwrite: bool = False,
            focus_axes: dict[str, list[object]] | None = None) -> dict[str, str]:
    manifest = port(sandbox, dest_rel, src, name, overwrite=overwrite)
    focus_axes = normalize_axes(focus_axes)
    wrapper = emit_wrapper(sandbox, dest_rel, name, src, hints, focus_axes)
    compile_test = emit_test(sandbox, dest_rel, name, src, hints)
    correctness_test = emit_correctness_test(sandbox, dest_rel, name, focus_axes)
    perf_test = emit_perf_test(sandbox, dest_rel, name, focus_axes)
    spec = emit_spec(repo_root, name, dest_rel, src, hints, manifest, focus_axes)
    return {
        "sandbox_kernel": manifest["sandbox_kernel"],
        "manifest": str(sandbox / dest_rel / f"{name}.kernelgen.json"),
        "wrapper": str(wrapper),
        "compile_test": str(compile_test),
        "correctness_test": str(correctness_test),
        "perf_test": str(perf_test),
        "spec": str(spec),
        "entry": src.entry,
        "origin": manifest["origin"],
    }
