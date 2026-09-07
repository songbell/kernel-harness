"""`ckh kernelgen` -- port a production CM kernel into the sandbox so it can be iterated on.

`ckh profile` names the kernel worth optimizing, but that name is an OpenCL kernel name from a
device timeline; the thing you actually edit is a `.cm` file in the plugin tree, and it will
not compile outside it (it pulls plugin-private headers and every one of its shape constants
arrives as a `-D` from host C++). Turning that into "a kernel I can rebuild in seconds" was
previously a manual half-day of grepping, and it was redone from scratch for each kernel.

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


# -- 1. what the .cm itself says (derived, trustworthy) --------------------------------

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
        in per instantiation, so the .cm alone cannot say what the kernel is called. The name
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
    m = re.search(r"extern\s+\"C\"\s+_GENX_MAIN_\s+void\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*\{",
                  text, re.S)
    if not m:
        raise Refused("no `extern \"C\" _GENX_MAIN_ void <entry>(...)` found -- this file is "
                      "not a CM entry point, or uses a convention this parser doesn't know")
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
        p = re.match(r"^(.*?[\s*])([A-Za-z_]\w*)$", line)
        if p:
            params.append(Param(ctype=p.group(1).strip(), name=p.group(2), guard=guard))
    if not params:
        raise Refused(f"entry `{entry}` parsed with zero parameters -- refusing to emit a "
                      f"descriptor whose args() would be empty")
    return entry, params


def parse_cm(path: Path, include_dirs: list[Path]) -> CmSource:
    """Parse the entry signature and walk the include closure for macro definitions."""
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


def scrape_host(cm_dir: Path, generator: str) -> HostHints:
    """Regex-lift jit constants, dispatch and argument order out of a generator class.

    Everything here is a GUESS by construction: `get_jit_constants` is C++ that branches on
    descriptor state and reads env vars, so the expression at a call site is frequently not
    the value that reaches the kernel at runtime. Emitted as commented expressions, never as
    bare numbers.
    """
    hints = HostHints(generator=generator)
    for cpp in sorted(cm_dir.glob("*.cpp")):
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


def candidates(production: Path, needle: str) -> list[Path]:
    """Match a profiled OpenCL kernel name against .cm files.

    The profiled name (`cm_pa_small_q`) is the plugin's entry point, not the filename
    (`pa_small_q.cm`), so a plain equality test finds nothing. Substring both ways.
    """
    key = re.sub(r"^cm_", "", needle.strip().lower())
    hits = []
    for p in sorted(cm_dir(production).glob("*.cm")):
        stem = p.stem.lower()
        if key == stem or key in stem or stem in key:
            hits.append(p)
    return hits


# -- 4. porting ------------------------------------------------------------------------

def port(sandbox: Path, dest_rel: str, src: CmSource, name: str,
         overwrite: bool = False) -> dict:
    """Copy the kernel and its include closure into the sandbox, with provenance.

    Provenance matters later: `integrator` has to diff the sandbox kernel against the shipped
    one code-only, and a copy with no record of where it came from turns that into guesswork.
    """
    dest = sandbox / dest_rel
    dest.mkdir(parents=True, exist_ok=True)
    kernel_dst = dest / f"{name}.cm"
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
              f"// Ledger findings belong in BOTH -- see .claude/agents/integrator.md\n")
    kernel_dst.write_text(banner + src.path.read_text(errors="replace"))

    manifest = {
        "name": name,
        "origin": str(src.path),
        "entry": src.entry,
        "sandbox_kernel": str(kernel_dst),
        "include_dir": str(inc_dst) if copied else "",
        "includes": copied,
        "missing_includes": src.missing_includes,
        "undefined_macros": sorted(src.undefined),        "params": [{"ctype": p.ctype, "name": p.name, "guard": p.guard} for p in src.params],
    }
    (dest / f"{name}.kernelgen.json").write_text(json.dumps(manifest, indent=2))
    return manifest


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
              manifest: dict) -> Path:
    out = repo_root / "kernels" / f"{name}.py"
    args_lines = [
        f'        # {p.ctype} {p.name}' + (f'   [only when {p.guard}]' if p.guard else '')
        for p in src.params
    ]
    host_args = ("\n".join(f"    #   {a}" for a in hints.args)
                 if hints.args else "    #   (not found)")

    body = f'''"""Generated by `ckh kernelgen` from {manifest["origin"]}.

DERIVED and trustworthy: the entry signature below, and the macro list in jit() -- both read
straight out of the .cm.

GUESS: every value marked `# GUESS from host`. Lifted by regex from
{hints.file.name if hints.file else "(no host .cpp matched)"}::{hints.generator}, which is C++ that branches on
runtime descriptor state. Treat them as leads, not facts.

MISSING: inputs(). There is no honest way to generate it -- see the module docstring in
src/ckh/kernelgen.py. Until it is written, `ckh bench`/`ckh equiv` on this spec will fail
loudly, which is the intended behaviour; a spec that runs on fabricated data would make a
wrong kernel look correct.

Next: .claude/agents/kernel-onboarder.md picks up from here.
"""
from __future__ import annotations

from ckh.kernel import KernelSpec, Shape

SOURCE = "{dest_rel}/{name}.cm"
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
    """Signature order, straight from the .cm entry point:
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

DEFAULT_AXES: dict = {{}}
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
    defines = "\n".join(f'    "{m}": None,' + (f"  # GUESS from host: {hints.jit[m]}"
                                               if m in hints.jit else "")
                        for m in sorted(src.required(hints)))
    body = f'''#!/usr/bin/env python3
"""Generated by `ckh kernelgen`: can the ported {name}.cm be built here at all?

Run:  pytest -q test_{name}.py
"""
import os

import pytest

from clops import cl

HERE = os.path.dirname(os.path.realpath(__file__))
INCLUDE_DIR = os.path.join(HERE, "{name}_include")

# Every macro the kernel references but never defines. Fill in the values -- the compile
# cannot succeed until they are all present, and that failure is the point of this file.
JIT = {{
    "KERNEL_NAME": "{src.entry}",
{defines}
}}


def _build():
    opts = " ".join([
        "-cmc", f"-I{{HERE}}", f"-I{{INCLUDE_DIR}}",
        *(f"-D{{k}}={{v}}" for k, v in JIT.items()),
    ])
    return cl.kernels('#include "{name}.cm"', opts)


def test_compiles():
    missing = [k for k, v in JIT.items() if v is None]
    assert not missing, f"fill in JIT values first: {{missing}}"
    _build()


@pytest.mark.skip(reason="needs real input buffers -- ckh kernelgen refuses to invent them; "
                         "see .claude/agents/kernel-onboarder.md")
def test_runs():
    # host gws = {hints.gws or "(not found)"}
    # host lws = {hints.lws or "(not found)"}
    kernels = _build()
    kernels.enqueue("{src.entry}", GWS, LWS, *ARGS)  # noqa: F821
    cl.finish()
'''
    out.write_text(body)
    return out
