import json
from pathlib import Path

from ckh.clintercept import (
    DFlashPatternSpec,
    analyze_dflash,
    load_full_attention_layers,
    load_num_hidden_layers,
    render_dflash,
)


def _trace(tmp_path: Path) -> Path:
    events = [{"ph": "M", "name": "clintercept_start_time", "pid": 1,
               "args": {"start_time": 0}}]
    names = (
        ["cm_sdpa_vlen"] * 2
        + ["sdpa_micro__generate"]
        + ["sdpa_micro__prefill"] * 2
    )
    events.extend({"ph": "X", "pid": 1, "ts": index * 10, "dur": 1, "name": name}
                  for index, name in enumerate(names))
    path = tmp_path / "clintercept_trace.json"
    path.write_text(json.dumps(events))
    return path


def test_dflash_pattern_uses_model_layer_counts(tmp_path):
    main = tmp_path / "main.json"
    main.write_text(json.dumps({"text_config": {
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention",
                         "linear_attention", "full_attention"],
    }}))
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps({"num_hidden_layers": 2}))

    result = analyze_dflash(
        _trace(tmp_path),
        DFlashPatternSpec(load_num_hidden_layers(main), load_num_hidden_layers(draft),
                          main_full_attention_layers=load_full_attention_layers(main)),
    )

    assert len(result["patterns"]) == 1
    pattern = result["patterns"][0]
    assert pattern["cm_sdpa_vlen"] == 2
    assert pattern["main_attention"] == 1
    assert pattern["draft_attention"] == 2
    assert pattern["complete"] is True
    assert not result["warnings"]


def test_dflash_report_marks_incomplete_layer_sequence(tmp_path):
    result = analyze_dflash(_trace(tmp_path), DFlashPatternSpec(3, 2))

    assert result["patterns"][0]["complete"] is False
    assert "incomplete" in result["warnings"][0]
    assert "INCOMPLETE" in render_dflash(result)
