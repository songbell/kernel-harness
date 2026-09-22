import json
from pathlib import Path

import pytest

from ckh.clintercept import (
    SpeculativePatternSpec,
    analyze_speculative,
    infer_speculative_pipeline,
    load_full_attention_layers,
    load_num_hidden_layers,
    render_speculative,
)


def _trace(tmp_path: Path, names: list[str]) -> Path:
    events = [{"ph": "M", "name": "clintercept_start_time", "pid": 1,
               "args": {"start_time": 0}}]
    events.extend({"ph": "X", "pid": 1, "ts": index * 10, "dur": 1, "name": name}
                  for index, name in enumerate(names))
    path = tmp_path / "clintercept_trace.json"
    path.write_text(json.dumps(events))
    return path


def test_speculative_pattern_uses_model_layer_counts(tmp_path):
    main = tmp_path / "main.json"
    main.write_text(json.dumps({"text_config": {
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention",
                         "linear_attention", "full_attention"],
    }}))
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps({"num_hidden_layers": 2}))
    trace = _trace(tmp_path, ["sdpa_micro__generate"] + ["sdpa_micro__prefill"] * 2)

    result = analyze_speculative(
        trace,
        SpeculativePatternSpec(load_num_hidden_layers(main), load_num_hidden_layers(draft),
                               main_full_attention_layers=load_full_attention_layers(main),
                               main_regex=r"sdpa_micro__generate",
                               draft_regex=r"sdpa_micro__prefill"),
    )

    assert len(result["patterns"]) == 1
    pattern = result["patterns"][0]
    assert pattern["main_attention"] == 1
    assert pattern["draft_attention"] == 2
    assert pattern["complete"] is True
    assert not result["warnings"]


def test_speculative_pattern_does_not_require_a_cm_kernel(tmp_path):
    """No cm_sdpa_vlen anywhere in the trace -- OV's own micro-SDPA kernels are enough."""
    trace = _trace(tmp_path, ["sdpa_micro__generate"] * 2 + ["sdpa_micro__prefill"] * 4)

    result = analyze_speculative(
        trace,
        SpeculativePatternSpec(2, 4, main_regex=r"sdpa_micro__generate",
                               draft_regex=r"sdpa_micro__prefill"))

    assert len(result["patterns"]) == 1
    assert result["patterns"][0]["complete"] is True
    assert not result["warnings"]


def test_speculative_report_marks_incomplete_layer_sequence(tmp_path):
    trace = _trace(tmp_path, ["sdpa_micro__generate"] + ["sdpa_micro__prefill"] * 2)
    result = analyze_speculative(
        trace,
        SpeculativePatternSpec(3, 2, main_regex=r"sdpa_micro__generate",
                               draft_regex=r"sdpa_micro__prefill"))

    assert result["patterns"][0]["complete"] is False
    assert "incomplete" in result["warnings"][0]
    assert "INCOMPLETE" in render_speculative(result)


def test_speculative_report_splits_phase_hotspots_and_intervals(tmp_path):
    names = [
        "sdpa_micro__prefill_mainsig__sa", "prefill_helper_kernel",
        "sdpa_micro__prefill_mainsig__sa", "prefill_helper_kernel",
        "sdpa_micro__prefill_mainsig__sa", "prefill_helper_kernel",
        "sdpa_micro__prefill_draftsig__sa", "draft_helper_kernel",
        "sdpa_micro__prefill_draftsig__sa", "draft_helper_kernel",
        "sdpa_micro__generate_mainsig__sa", "main_generate_helper",
        "sdpa_micro__generate_mainsig__sa", "main_generate_helper",
        "sdpa_micro__generate_mainsig__sa", "main_generate_helper",
    ]
    trace = _trace(tmp_path, names)

    result = analyze_speculative(
        trace,
        SpeculativePatternSpec(3, 2, main_regex=r"sdpa_micro__generate",
                               draft_regex=r"sdpa_micro__prefill", gap_ms=0.001),
        top=5,
    )

    assert {row["kernel"] for row in result["phase_hotspots"]["main_prefill"]} >= {
        "sdpa_micro__prefill_mainsig__sa", "prefill_helper_kernel"
    }
    assert {row["kernel"] for row in result["phase_hotspots"]["draft_generate"]} >= {
        "sdpa_micro__prefill_draftsig__sa", "draft_helper_kernel"
    }
    assert {row["kernel"] for row in result["phase_hotspots"]["main_generate"]} >= {
        "sdpa_micro__generate_mainsig__sa", "main_generate_helper"
    }
    assert any(row["from"] == "main_prefill" and row["to"] == "draft_generate"
               for row in result["interval_summary"])
    assert any(row["from"] == "draft_generate" and row["to"] == "main_generate"
               for row in result["interval_summary"])
    rendered = render_speculative(result)
    assert "main prefill hot-spot kernels" in rendered
    assert "draft generate hot-spot kernels" in rendered
    assert "intervals between adjacent main/draft windows" in rendered


def test_speculative_report_adds_per_generation_cost_and_draft_span(tmp_path):
    events = [{"ph": "M", "name": "clintercept_start_time", "pid": 1, "args": {"start_time": 0}}]

    def add(ts, name):
        events.append({"ph": "X", "pid": 1, "ts": ts, "dur": 1, "name": name})

    for i in range(3):
        add(i * 10, "sdpa_micro__prefill_mainsig__sa")           # main prefill, 3 calls
    for i in range(2):
        add(30 + i * 10, "sdpa_micro__prefill_draftsig__sa")     # draft (after prefill, excluded from span)
    for i in range(3):
        add(50 + i * 10, "sdpa_micro__generate_mainsig__sa")     # main generate #1
    for i in range(2):
        add(80 + i * 10, "sdpa_micro__prefill_draftsig__sa")     # draft between two main generations
    for i in range(3):
        add(100 + i * 10, "sdpa_micro__generate_mainsig__sa")    # main generate #2

    trace = tmp_path / "clintercept_trace.json"
    trace.write_text(json.dumps(events))

    result = analyze_speculative(
        trace,
        SpeculativePatternSpec(3, 2, main_regex=r"sdpa_micro__generate",
                               draft_regex=r"sdpa_micro__prefill", gap_ms=0.001),
        top=5,
    )

    assert result["generation_count"] == 2
    assert result["draft_span_ms"] == pytest.approx(0.011)

    main_generate_row = next(row for row in result["phase_hotspots"]["main_generate"]
                             if row["kernel"] == "sdpa_micro__generate_mainsig__sa")
    assert main_generate_row["ms_per_generation"] == pytest.approx(
        main_generate_row["total_ms"] / result["generation_count"])

    rendered = render_speculative(result)
    assert "generations   2" in rendered
    assert "draft span    0.01 ms" in rendered
    assert "ms_per_gen" in rendered


def test_speculative_report_warns_when_no_kernel_matches(tmp_path):
    trace = _trace(tmp_path, ["some_unrelated_kernel"])
    result = analyze_speculative(
        trace,
        SpeculativePatternSpec(3, 2, main_regex=r"sdpa_micro__generate",
                               draft_regex=r"sdpa_micro__prefill"))

    assert result["patterns"] == []
    assert "no main" in result["warnings"][0]


def test_speculative_pattern_regexes_are_optional_by_default(tmp_path):
    """With no regex at all, a spec still constructs, and both sides default to None."""
    spec = SpeculativePatternSpec(3, 2)
    assert spec.main_regex is None
    assert spec.draft_regex is None


def test_speculative_pattern_classifies_by_call_count_when_no_regex(tmp_path):
    """No main_regex/draft_regex at all: main/draft layers differ, so call-count alone
    separates the two models -- no kernel name is ever inspected."""
    trace = _trace(tmp_path, ["kernel_a"] * 3 + ["kernel_b"] * 2)
    result = analyze_speculative(trace, SpeculativePatternSpec(3, 2))

    assert result["patterns"] == []
    assert result["main_hotspots"][0]["kernel"] == "kernel_a"
    assert result["draft_hotspots"][0]["kernel"] == "kernel_b"
    assert "unclassified_hotspots" not in result
    assert any("no main_regex configured" in w for w in result["warnings"])
    rendered = render_speculative(result)
    assert "main hot-spot kernels" in rendered
    assert "draft hot-spot kernels" in rendered


def test_speculative_pattern_falls_back_to_flat_hotspots_when_layers_equal(tmp_path):
    """main_layers == draft_layers: call count cannot distinguish the two models, so this
    degrades to one combined hot-spot table instead of a wrong split."""
    trace = _trace(tmp_path, ["kernel_a"] * 3 + ["kernel_b"] * 3)
    result = analyze_speculative(trace, SpeculativePatternSpec(3, 3))

    assert result["patterns"] == []
    assert "main_hotspots" not in result
    assert "draft_hotspots" not in result
    assert {r["kernel"] for r in result["hotspots"]} == {"kernel_a", "kernel_b"}
    assert any("cannot separate main vs draft by call count" in w for w in result["warnings"])


def test_speculative_pattern_one_sided_regex_uses_call_count_for_the_other_side(tmp_path):
    """Only main_regex configured: the draft side still tries call-count classification on
    whatever main_regex left unmatched, rather than dumping it all as "draft"."""
    trace = _trace(tmp_path, ["sdpa_micro__generate"] * 2 + ["other_kernel"] * 3)
    result = analyze_speculative(
        trace, SpeculativePatternSpec(2, 4, main_regex=r"sdpa_micro__generate"))

    assert result["patterns"] == []
    assert "main_hotspots" not in result
    # 3 calls divides neither main_layers=2 nor draft_layers=4 -> unclassified, not draft.
    assert result["draft_hotspots"] == []
    assert result["unclassified_hotspots"][0]["kernel"] == "other_kernel"
    assert any("no draft_regex configured" in w for w in result["warnings"])


def test_infer_speculative_pipeline_high_confidence_from_binary_name(tmp_path):
    main_dir, draft_dir = tmp_path / "main", tmp_path / "draft"
    main_dir.mkdir()
    draft_dir.mkdir()
    (main_dir / "config.json").write_text("{}")
    (draft_dir / "config.json").write_text("{}")

    result = infer_speculative_pipeline(
        ["speculative_decoding_lm.exe", str(main_dir), str(draft_dir), "prompt.txt", "false"])

    assert result["is_speculative"] is True
    assert result["confidence"] == "high"
    assert result["candidates"][0]["role"] == "main"
    assert result["candidates"][0]["config"] == str(main_dir / "config.json")
    assert result["candidates"][1]["role"] == "draft"
    assert result["candidates"][1]["config"] == str(draft_dir / "config.json")


def test_infer_speculative_pipeline_finds_nested_config(tmp_path):
    main_dir, draft_dir = tmp_path / "main", tmp_path / "draft"
    (main_dir / "OV_FP16").mkdir(parents=True)
    draft_dir.mkdir()
    (main_dir / "OV_FP16" / "config.json").write_text("{}")
    (draft_dir / "config.json").write_text("{}")

    result = infer_speculative_pipeline(
        ["speculative_decoding_lm.exe", str(main_dir), str(draft_dir)])

    assert result["candidates"][0]["config"] == str(main_dir / "OV_FP16" / "config.json")


def test_infer_speculative_pipeline_refuses_to_guess_ambiguous_config(tmp_path):
    main_dir, draft_dir = tmp_path / "main", tmp_path / "draft"
    (main_dir / "a").mkdir(parents=True)
    (main_dir / "b").mkdir(parents=True)
    draft_dir.mkdir()
    (main_dir / "a" / "config.json").write_text("{}")
    (main_dir / "b" / "config.json").write_text("{}")
    (draft_dir / "config.json").write_text("{}")

    result = infer_speculative_pipeline(
        ["speculative_decoding_lm.exe", str(main_dir), str(draft_dir)])

    assert result["candidates"][0]["config"] is None
    assert len(result["candidates"][0]["config_candidates"]) == 2
    assert any("refusing to guess" in w for w in result["warnings"])


def test_infer_speculative_pipeline_low_confidence_without_binary_hint(tmp_path):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    result = infer_speculative_pipeline(["benchmark_vlm.exe", "-m", str(dir_a), "-i", str(dir_b)])

    assert result["is_speculative"] is True
    assert result["confidence"] == "low"


def test_infer_speculative_pipeline_not_speculative(tmp_path):
    only_dir = tmp_path / "only"
    only_dir.mkdir()

    result = infer_speculative_pipeline(["benchmark_vlm.exe", "-m", str(only_dir)])

    assert result["is_speculative"] is False
