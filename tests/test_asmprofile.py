"""Coverage for `ckh kernel-profile`'s assembly analysis. No GPU, no IGC -- synthetic dumps.

The cycle model is the part worth guarding. Ranking by instruction count instead of pipe
cycles once recommended a change that measured +1.0% -- slower -- because it cut instructions
and raised cycles. So the tests below assert the two rankings actually differ on a case
constructed to separate them; a test that only checked "the histogram has entries" would have
passed for the wrong model too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckh import asmprofile as ap  # noqa: E402

HEADER = """//.kernel cm_test
//.platform XE3
//.thread_config numGRF=256, numAcc=8, numSWSB=32
//.instCount {n}
//.GRF count {grf}
"""


def _dump(tmp_path: Path, body: str, n: int = 0, grf: int = 100, extra: str = "") -> Path:
    p = tmp_path / "cm_test.asm"
    p.write_text(HEADER.format(n=n or len(body.strip().split("\n")), grf=grf) + extra + body)
    return p


def test_header_facts_are_read(tmp_path):
    p = ap.parse(_dump(tmp_path, "mov (16|M0) r1.0<1>:f r2.0<1;1,0>:f\n", n=1, grf=243))
    assert (p.kernel, p.platform) == ("cm_test", "XE3")
    assert (p.grf_used, p.num_grf, p.inst_count) == (243, 256, 1)


def test_spill_declares_are_counted(tmp_path):
    extra = "//.declare Spill_0 (99)  rf=r size=64\n//.declare V_spill_1 (98)  rf=r size=64\n"
    p = ap.parse(_dump(tmp_path, "mov (16|M0) r1.0<1>:f r2.0<1;1,0>:f\n", extra=extra))
    assert p.spills == 2


# -- the cycle model -------------------------------------------------------------------

def test_exec_size_drives_cycles_not_instruction_count():
    wide = ap.Inst(0, 1, "mov", 32, "")
    narrow = ap.Inst(1, 2, "mov", 1, "")
    assert wide.cycles(16) == 2
    assert narrow.cycles(16) == 1
    # Two exec-1 ops are cheaper than one exec-32 op, which is the whole point.
    assert narrow.cycles(16) * 2 == wide.cycles(16)


def test_dpas_and_send_are_special_cased():
    assert ap.Inst(0, 1, "dpas.8x8", 8, "").cycles(16) == ap.DPAS_CYCLES
    assert ap.Inst(0, 1, "send.ugm", 16, "").cycles(16) == 1


def test_ranking_by_cycles_differs_from_ranking_by_count(tmp_path):
    """Constructed so the two orders disagree: many narrow `add`s vs fewer wide `mul`s."""
    body = "".join(f"add (1|M0) r{i}.0<1>:d r1.0<0;1,0>:d\n" for i in range(10))
    body += "".join(f"mul (32|M0) r{i}.0<1>:f r2.0<1;1,0>:f\n" for i in range(6))
    p = ap.parse(_dump(tmp_path, body))
    ops, _ = ap.histogram(p)
    assert ops["add"]["n"] > ops["mul"]["n"]
    assert ops["mul"]["cycles"] > ops["add"]["cycles"]


def test_width_is_configurable(tmp_path):
    body = "mov (32|M0) r1.0<1>:f r2.0<1;1,0>:f\n"
    assert ap.parse(_dump(tmp_path, body), width=32).total_cycles == 1
    assert ap.parse(_dump(tmp_path, body), width=8).total_cycles == 4


# -- structure -------------------------------------------------------------------------

LOOP = """
BB_0:
mov (16|M0) r1.0<1>:f r2.0<1;1,0>:f
BB_1:
add (16|M0) r3.0<1>:f r4.0<1;1,0>:f
mul (16|M0) r5.0<1>:f r6.0<1;1,0>:f
jmpi BB_1
mov (16|M0) r7.0<1>:f r8.0<1;1,0>:f
"""


def test_back_edge_is_found(tmp_path):
    p = ap.parse(_dump(tmp_path, LOOP))
    assert ap.back_edges(p) == [(1, 3)]
    assert ap.main_loop(p) == (1, 3)


def test_no_back_edge_returns_none(tmp_path):
    p = ap.parse(_dump(tmp_path, "mov (16|M0) r1.0<1>:f r2.0<1;1,0>:f\n"))
    assert ap.main_loop(p) is None
    assert "no back-edge found" in ap.render(p)


def test_scoping_to_the_loop_excludes_code_outside_it(tmp_path):
    p = ap.parse(_dump(tmp_path, LOOP))
    lo, hi = ap.main_loop(p)
    scoped, _ = ap.histogram(p, range(lo, hi + 1))
    whole, _ = ap.histogram(p)
    assert whole["mov"]["n"] == 2 and "mov" not in scoped


def test_branched_over_blocks_are_excluded(tmp_path):
    """A static histogram counts code the shape never runs; that misattribution is the
    reason this function exists."""
    body = "jmpi BB_END\n"
    body += "".join(f"mul (16|M0) r{i}.0<1>:f r2.0<1;1,0>:f\n" for i in range(25))
    body += "BB_END:\nadd (16|M0) r1.0<1>:f r2.0<1;1,0>:f\n"
    p = ap.parse(_dump(tmp_path, body))
    skipped = ap.skipped_blocks(p, 0, len(p.insts))
    assert len(skipped) == 25
    ops, _ = ap.histogram(p, None, skipped)
    assert "mul" not in ops and ops["add"]["n"] == 1


def test_short_forward_jumps_are_ordinary_control_flow(tmp_path):
    body = "jmpi BB_END\nmul (16|M0) r1.0<1>:f r2.0<1;1,0>:f\nBB_END:\n"
    p = ap.parse(_dump(tmp_path, body))
    assert ap.skipped_blocks(p, 0, len(p.insts)) == set()


def test_branches_are_parsed_despite_having_no_exec_field(tmp_path):
    p = ap.parse(_dump(tmp_path, "jmpi BB_0\ngoto BB_0\njoin BB_0\n"))
    assert [i.op for i in p.insts] == ["jmpi", "goto", "join"]
    _, cats = ap.histogram(p)
    assert cats["control"]["n"] == 3


# -- findings --------------------------------------------------------------------------

def _findings(tmp_path, body, **kw):
    p = ap.parse(_dump(tmp_path, body, **kw))
    ops, cats = ap.histogram(p)
    return p, {f.code: f for f in ap.findings(p, ops, cats)}


def test_move_bound_fires_and_stays_silent(tmp_path):
    heavy = "".join(f"mov (16|M0) r{i}.0<1>:f r2.0<1;1,0>:f\n" for i in range(8))
    assert "move-bound" in _findings(tmp_path, heavy)[1]
    light = "mov (16|M0) r1.0<1>:f r2.0<1;1,0>:f\n" + \
        "".join(f"add (16|M0) r{i}.0<1>:f r2.0<1;1,0>:f\n" for i in range(20))
    assert "move-bound" not in _findings(tmp_path, light)[1]


def test_ieee_divide_is_reported_as_an_accuracy_decision(tmp_path):
    _, f = _findings(tmp_path, "madm (16|M0) r1.0<1>:f r2.0<1;1,0>:f r3.0<1;1,0>:f r4.0:f\n")
    assert "ACCURACY" in f["ieee-divide"].hypothesis


def test_spill_and_grf_tight_are_mutually_exclusive(tmp_path):
    body = "mov (16|M0) r1.0<1>:f r2.0<1;1,0>:f\n"
    _, spilled = _findings(tmp_path, body, grf=250,
                           )  # no spill declares -> grf-tight
    assert "grf-tight" in spilled and "spill" not in spilled
    p = ap.parse(_dump(tmp_path, body, grf=250, extra="//.declare Spill_0 (9) rf=r\n"))
    ops, cats = ap.histogram(p)
    codes = {f.code for f in ap.findings(p, ops, cats)}
    assert "spill" in codes and "grf-tight" not in codes


def test_render_never_presents_a_finding_as_a_conclusion(tmp_path):
    heavy = "".join(f"mov (16|M0) r{i}.0<1>:f r2.0<1;1,0>:f\n" for i in range(8))
    out = ap.render(ap.parse(_dump(tmp_path, heavy)))
    assert "HYPOTHESES" in out
    assert "is a result until `ckh bench`" in out


def test_render_warns_when_coverage_is_poor(tmp_path):
    p = ap.parse(_dump(tmp_path, "mov (16|M0) r1.0<1>:f r2.0<1;1,0>:f\n", n=100))
    out = ap.render(p)
    assert "1.0% covered" in out
    assert "off by an unknown amount" in out


def test_render_flags_an_ambiguous_loop_choice(tmp_path):
    body = ("BB_0:\nadd (16|M0) r1.0<1>:f r2.0<1;1,0>:f\njmpi BB_0\n"
            "BB_1:\nmul (16|M0) r3.0<1>:f r4.0<1;1,0>:f\njmpi BB_1\n")
    out = ap.render(ap.parse(_dump(tmp_path, body)))
    assert "more than one loop nest" in out
    assert "not a measurement" in out
