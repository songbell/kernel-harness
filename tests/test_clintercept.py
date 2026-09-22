import os
from pathlib import Path

import pytest

from ckh import clintercept


def test_model_for_stage_selects_named_model():
    assert clintercept.model_for_stage(
        "profile", {"router": "router-model", "reasoner": "reasoner-model"},
        {"profile": "router", "kernelgen": "reasoner"}) == "router-model"


def test_model_for_stage_rejects_unknown_route():
    with pytest.raises(KeyError, match="unknown model"):
        clintercept.model_for_stage("kernelgen", {"router": "router-model"},
                        {"kernelgen": "reasoner"})


def test_model_for_stage_rejects_non_harness_stage():
    with pytest.raises(KeyError, match="unknown CKH stage"):
        clintercept.model_for_stage("draft", {"router": "router-model"},
                        {"draft": "router"})


def test_locate_finds_windows_cliloader(tmp_path, monkeypatch):
    monkeypatch.setattr(clintercept.os, "name", "nt")
    linux_loader = tmp_path / "clintercept-3.0.6-Linux" / "bin" / "cliloader"
    linux_loader.parent.mkdir(parents=True)
    linux_loader.write_text("")
    os.chmod(linux_loader, 0o755)
    loader = tmp_path / "clintercept-3.0.6-win64" / "Release" / "cliloader.exe"
    loader.parent.mkdir(parents=True)
    loader.write_text("")
    os.chmod(loader, 0o755)

    assert clintercept.locate(tmp_path) == loader


def _trace(tmp_path: Path) -> Path:
    trace = tmp_path / "clintercept_trace.json"
    trace.write_text(
        "["
        '{"ph":"M","name":"clintercept_start_time","pid":1,"args":{"start_time":0}},'
        '{"ph":"X","pid":1,"ts":0,"dur":100,"name":"first_kernel"},'
        '{"ph":"X","pid":1,"ts":200,"dur":300,"name":"second_kernel"},'
        '{"ph":"X","pid":1,"ts":600,"dur":50,"name":"first_kernel"}'
        "]"
    )
    return trace


def test_default_analysis_has_global_kernel_summary_without_small_q_anchor(tmp_path):
    result = clintercept.analyze(_trace(tmp_path))

    assert result["segmentation"]["anchor"] == ""
    assert [row["kernel"] for row in result["global"]["rows"]] == [
        "second_kernel", "first_kernel"
    ]
    assert result["global"]["rows"][0]["total_ms"] == 0.3


def test_render_shows_global_top_rows(tmp_path):
    result = clintercept.analyze(_trace(tmp_path))

    output = clintercept.render(result, top=1)

    assert "GLOBAL TOP KERNELS" in output
    global_section = output.split("GLOBAL TOP KERNELS", 1)[1].split("PREFILL", 1)[0]
    assert "second_kernel" in global_section
    assert "first_kernel" not in global_section


def test_write_report_artifacts_persists_summary_report_and_timeline(tmp_path):
    trace = _trace(tmp_path)
    result = clintercept.analyze(trace)
    rendered = clintercept.render(result, top=2)

    paths = clintercept.write_report_artifacts(trace, result, rendered)

    summary = Path(paths["summary"])
    report = Path(paths["report"])
    timeline = Path(paths["timeline"])

    assert summary.exists()
    assert report.exists()
    assert timeline.exists()
    assert '"trace":' in summary.read_text(encoding="utf-8")
    assert "GLOBAL TOP KERNELS" in report.read_text(encoding="utf-8")
    assert "gantt" in timeline.read_text(encoding="utf-8")


def test_prepare_profile_env_injects_dump_sources_path(tmp_path):
    env, dump = clintercept.prepare_profile_env(tmp_path)
    assert dump == tmp_path / "ov_gpu_dump_sources"
    assert env["OV_GPU_DUMP_SOURCES_PATH"] == str(dump)
    assert dump.exists()


def test_load_profile_summary_from_directory(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "profile_summary.json").write_text('{"global": {"rows": []}}', encoding="utf-8")
    summary = clintercept.load_profile_summary(run_dir)
    assert summary["global"]["rows"] == []


def test_pick_hot_kernel_prefers_generate_then_global():
    summary = {
        "phases": {"generate": {"rows": [{"kernel": "g0", "total_ms": 1.0}]}, "prefill": {"rows": []}},
        "global": {"rows": [{"kernel": "all0", "total_ms": 5.0}]},
    }
    selected = clintercept.pick_hot_kernel(summary)
    assert selected["kernel"] == "g0"
    assert selected["bucket"] == "generate"


def test_pick_hot_kernel_uses_speculative_main_generate_bucket():
    summary = {
        "phase_hotspots": {
            "main_generate": [{"kernel": "sdpa_micro__generate_x", "total_ms": 3.0}],
            "draft_generate": [{"kernel": "draft_kernel", "total_ms": 2.0}],
        }
    }
    selected = clintercept.pick_hot_kernel(summary)
    assert selected["kernel"] == "sdpa_micro__generate_x"
    assert selected["bucket"] == "main_generate"
