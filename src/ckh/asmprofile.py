"""`ckh kernel-profile` -- what the compiler actually emitted, weighted by pipe cycles.

Hardware counters would be the obvious way to answer "where does this kernel spend itself".
They were not available on the reference box -- see `prof_results/PROFILING.md`, where unitrace
could not be made to run at all -- so the analysis that produced this project's results was
built on the IGC assembly dump instead. This module generalizes that over any KernelSpec.

Two design decisions carried over from the one-off script, both of which changed conclusions:

**Cycles, not instructions.** At a 16-wide ALU an exec-32 op costs 2 cycles and an exec-1 op
costs 1, so widening an already-vector loop buys nothing while deleting scalar code pays.
A change that *cut instruction count and raised cycle count* measured +1.0% -- slower. Ranking
by instruction count would have recommended it.

**Executed, not static.** Blocks the shape branches over (a causal-mask path, a lazy-rescale
body) are in the static histogram and not in the run. A static ranking misattributes them.

What this module does NOT do is decide anything. Every finding it prints is a hypothesis with
its evidence attached, and this codebase's own history is four confident hypotheses in a row
that were wrong. `ckh bench` remains the only thing that settles one.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from pathlib import Path

# Xe ALU width in fp32 lanes. An exec-N op issues over ceil(N / WIDTH) cycles.
DEFAULT_WIDTH = 16
# RepeatCount-8 DPAS occupies the systolic array this long.
DPAS_CYCLES = 8

_INST = re.compile(r"^\s*(?:\(([^)]*)\)\s*)?([a-z][a-z0-9_.]*)\s*\((\d+)\|")
_BRANCH = re.compile(r"^\s*(?:\(([^)]*)\)\s*)?(jmpi|goto|while|call|ret|brc|brd|join)\b")
_LABEL = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*$")
# Any trailing identifier; the labels dict lookup is what validates it. Matching only the
# `BB_<n>` form silently made branches to any other label shape untargetable, which does not
# error -- it under-counts branched-over blocks and quietly restores the misattribution this
# analysis exists to remove.
_TARGET = re.compile(r"\s([A-Za-z_]\w*)\s*(?://.*)?$")

# Opcode -> the pipe it contends for. Keeps the rollup meaningful across kernels that share
# no source structure at all, which per-region decomposition cannot do.
CATEGORIES = {
    "systolic": ("dpas",),
    "memory": ("send",),
    "transcendental": ("math",),
    "move": ("mov", "sel", "csel"),
    "control": ("jmpi", "goto", "while", "call", "ret", "brc", "brd", "join", "cmp",
                "and", "or", "sync"),
}


def categorize(op: str) -> str:
    for cat, prefixes in CATEGORIES.items():
        if any(op.startswith(p) for p in prefixes):
            return cat
    return "alu"


@dataclass
class Inst:
    idx: int
    line: int
    op: str
    exec_size: int
    text: str

    def cycles(self, width: int) -> int:
        if "dpas" in self.op:
            return DPAS_CYCLES
        if self.op.startswith("send"):
            return 1                    # issue slot only; latency is hidden by thread parallelism
        return max(1, -(-self.exec_size // width))


@dataclass
class Finding:
    code: str
    evidence: str
    hypothesis: str
    scope: str = "loop"


@dataclass
class AsmProfile:
    path: Path
    kernel: str = ""
    platform: str = ""
    num_grf: int = 0                    # the budget the kernel was compiled for
    grf_used: int = 0                   # what the register allocator actually needed
    inst_count: int = 0
    spills: int = 0
    insts: list[Inst] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)
    width: int = DEFAULT_WIDTH

    @property
    def total_cycles(self) -> int:
        return sum(i.cycles(self.width) for i in self.insts)


def _header(text: str, key: str) -> str:
    m = re.search(rf"^//\.{re.escape(key)}\s+(.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def parse(path: Path, width: int = DEFAULT_WIDTH) -> AsmProfile:
    text = path.read_text(errors="replace")
    p = AsmProfile(path=path, width=width)
    p.kernel = _header(text, "kernel")
    p.platform = _header(text, "platform")
    p.inst_count = int(_header(text, "instCount") or 0)
    grf = _header(text, "GRF count")
    p.grf_used = int(grf) if grf.isdigit() else 0
    m = re.search(r"numGRF=(\d+)", text)
    p.num_grf = int(m.group(1)) if m else 0
    # A spill is not reported in the header; it shows up as declares the RA had to invent.
    p.spills = len(re.findall(r"^//\.declare\s+\S*[Ss]pill", text, re.M))

    for i, line in enumerate(text.split("\n")):
        lab = _LABEL.match(line)
        if lab:
            p.labels[lab.group(1)] = len(p.insts)
            continue
        m = _INST.match(line)
        if m:
            p.insts.append(Inst(len(p.insts), i + 1, m.group(2), int(m.group(3)), line))
            continue
        # Branches carry no `(exec|...)` field, so the instruction regex misses them; without
        # this they are absent from the histogram AND from loop detection.
        b = _BRANCH.match(line)
        if b:
            p.insts.append(Inst(len(p.insts), i + 1, b.group(2), 1, line))
    return p


# -- structure ------------------------------------------------------------------------

def branch_target(p: AsmProfile, inst: Inst) -> int | None:
    if inst.op not in ("jmpi", "goto", "while", "brc", "brd"):
        return None
    m = _TARGET.search(inst.text)
    return p.labels.get(m.group(1)) if m else None


def back_edges(p: AsmProfile) -> list[tuple[int, int]]:
    return sorted({(t, i.idx) for i in p.insts
                   for t in [branch_target(p, i)] if t is not None and t < i.idx})


def main_loop(p: AsmProfile) -> tuple[int, int] | None:
    """The largest back-edge span. Not "the hot loop" -- that would need a trip count this
    cannot know -- but for a kernel whose cost is a loop over KV, it is usually the same
    region. `render` prints how many back-edges it chose between, because when the choice is
    not obvious the scoping is the least trustworthy thing on the page."""
    edges = back_edges(p)
    return max(edges, key=lambda e: e[1] - e[0]) if edges else None


def skipped_blocks(p: AsmProfile, lo: int, hi: int, min_size: int = 20) -> set[int]:
    """Instructions inside forward jumps that skip a substantial block.

    These are the shape-dependent paths -- a causal mask, a rescale guard -- and counting them
    is how a static histogram ends up attributing a quarter of a kernel to code the operating
    point never runs. `min_size` exists because short forward jumps are ordinary control flow,
    not optional blocks.
    """
    out: set[int] = set()
    for i in p.insts[lo:hi]:
        t = branch_target(p, i)
        if t is not None and t > i.idx and t - i.idx >= min_size:
            out |= set(range(i.idx + 1, t))
    return out


def histogram(p: AsmProfile, subset: range | None = None,
              exclude: set[int] | None = None) -> tuple[dict, dict]:
    """(by_opcode, by_category), each name -> {"n": count, "cycles": pipe cycles}."""
    exclude = exclude or set()
    insts = p.insts[subset.start:subset.stop] if subset else p.insts
    ops: dict[str, dict] = collections.defaultdict(lambda: {"n": 0, "cycles": 0})
    cats: dict[str, dict] = collections.defaultdict(lambda: {"n": 0, "cycles": 0})
    for i in insts:
        if i.idx in exclude:
            continue
        c = i.cycles(p.width)
        for bucket, key in ((ops, i.op), (cats, categorize(i.op))):
            bucket[key]["n"] += 1
            bucket[key]["cycles"] += c
    return dict(ops), dict(cats)


# -- findings -------------------------------------------------------------------------

def findings(p: AsmProfile, ops: dict, cats: dict) -> list[Finding]:
    """Hypotheses with their evidence. Never conclusions -- see the module docstring.

    Each threshold is a share of measured pipe cycles in the analysed region, not a count, and
    each is stated in the evidence so a reader can disagree with the number rather than having
    to reverse-engineer it.
    """
    out: list[Finding] = []
    total = sum(v["cycles"] for v in cats.values()) or 1

    def share(cat: str) -> float:
        return 100.0 * cats.get(cat, {}).get("cycles", 0) / total

    if p.spills:
        out.append(Finding(
            "spill",
            f"{p.spills} spill/fill declare(s); GRF count {p.grf_used} of {p.num_grf}",
            "register pressure is forcing scratch traffic. Raising "
            "-Qxcm_register_file_size costs resident threads, so this is a trade, not a fix"))
    elif p.num_grf and p.grf_used >= p.num_grf * 0.95:
        out.append(Finding(
            "grf-tight",
            f"GRF count {p.grf_used} of {p.num_grf} ({100 * p.grf_used / p.num_grf:.0f}%)",
            "no spill yet, but no slack for the scheduler to batch loads either. "
            "A/B the register file size; it has flipped sign across rounds in this codebase"))

    if share("move") >= 20:
        out.append(Finding(
            "move-bound",
            f"move ops are {share('move'):.0f}% of pipe cycles "
            f"({cats['move']['n']} instructions)",
            "register marshalling, not arithmetic, dominates. Look for layout changes that "
            "let the data land where the consumer wants it"))

    madm = ops.get("madm", {}).get("n", 0)
    if madm:
        out.append(Finding(
            "ieee-divide",
            f"{madm} madm -- the IEEE divide/sqrt macro-sequence",
            "a correctly-rounded divide expands to a long sequence. If the divisor is a "
            "softmax denominator, a reciprocal approximation may be acceptable -- that is an "
            "ACCURACY decision and needs explicit sign-off, not a silent swap"))

    if share("systolic") and share("systolic") < 10:
        out.append(Finding(
            "dpas-starved",
            f"systolic ops are only {share('systolic'):.0f}% of pipe cycles",
            "the matrix engine is idle most of the time; the surrounding code is the cost. "
            "Confirm against a measured roofline before restructuring the DPAS itself"))

    if share("transcendental") >= 15:
        out.append(Finding(
            "transcendental-heavy",
            f"math ops are {share('transcendental'):.0f}% of pipe cycles",
            "exp/log/rsqrt at this share is worth checking for redundant evaluation across "
            "an unrolled body"))

    return out


# -- rendering ------------------------------------------------------------------------

def render(p: AsmProfile, top: int = 12) -> str:
    edges = back_edges(p)
    loop = main_loop(p)
    _, whole_cats = histogram(p)
    if loop:
        lo, hi = loop
        skipped = skipped_blocks(p, lo, hi)
        ops, cats = histogram(p, range(lo, hi + 1), skipped)
        scope = (f"main loop [{lo}:{hi}], {hi - lo + 1} instructions, "
                 f"{len(skipped)} branched over at this shape "
                 f"(largest of {len(edges)} back-edges)")
    else:
        skipped = set()
        ops, cats = histogram(p)
        scope = "whole kernel (no back-edge found -- nothing to scope to)"

    total = sum(v["cycles"] for v in cats.values()) or 1
    whole_total = sum(v["cycles"] for v in whole_cats.values()) or 1
    coverage = 100.0 * len(p.insts) / p.inst_count if p.inst_count else 100.0
    lines = [
        f"kernel     {p.kernel}   platform {p.platform}",
        f"registers  GRF count {p.grf_used} of {p.num_grf}"
        + (f"   SPILLS: {p.spills}" if p.spills else "   no spill"),
        f"size       {p.inst_count} instructions (header), {len(p.insts)} parsed "
        f"({coverage:.1f}% covered)",
        f"scope      {scope}",
        f"cycle model exec-N costs ceil(N/{p.width}); dpas {DPAS_CYCLES}; send 1 (issue slot)",
    ]
    if coverage < 95:
        lines.append(f"WARNING    {p.inst_count - len(p.insts)} instructions were not "
                     f"recognised; every share below is off by an unknown amount")
    if len(edges) > 1:
        lines.append("NOTE       more than one loop nest exists; the largest span was scoped "
                     "to, which is a guess about where the time goes, not a measurement")

    # Loop-scoped and whole-kernel side by side: a term that is large in one and small in the
    # other is the tell that the scoping picked the wrong region.
    lines += ["", f"{'category':<16}{'instrs':>8}{'cycles':>9}{'% loop':>9}{'% kernel':>10}"]
    for cat, v in sorted(cats.items(), key=lambda kv: -kv[1]["cycles"]):
        whole = 100.0 * whole_cats.get(cat, {}).get("cycles", 0) / whole_total
        lines.append(f"{cat:<16}{v['n']:>8}{v['cycles']:>9}"
                     f"{100 * v['cycles'] / total:>8.1f}%{whole:>9.1f}%")

    lines += ["", f"{'opcode':<16}{'instrs':>8}{'cycles':>9}{'% loop':>9}"]
    for op, v in sorted(ops.items(), key=lambda kv: -kv[1]["cycles"])[:top]:
        lines.append(f"{op:<16}{v['n']:>8}{v['cycles']:>9}{100 * v['cycles'] / total:>8.1f}%")

    fs = findings(p, ops, cats)
    if loop:
        # A cost outside the loop is still a cost. The IEEE-divide sequence in this project's
        # own reference kernel sits entirely outside the KV loop; reporting only the scoped
        # histogram would have dropped the single largest opcode finding in its profile.
        seen = {f.code for f in fs}
        whole_ops, _ = histogram(p)
        for f in findings(p, whole_ops, whole_cats):
            if f.code not in seen:
                f.scope = "whole kernel"
                fs.append(f)
    lines.append("")
    if not fs:
        lines.append("no findings above threshold.")
    for f in fs:
        lines.append(f"[{f.code}]  ({f.scope}) {f.evidence}")
        lines.append(f"           -> {f.hypothesis}")
    lines.append(
        "\nThese are HYPOTHESES, ranked by static cycle estimate. Four confident ones in a "
        "row were wrong in this codebase.\nNothing here is a result until `ckh bench` "
        "resolves it above the noise floor -- and `ckh ledger --add` records it either way.")
    return "\n".join(lines)
