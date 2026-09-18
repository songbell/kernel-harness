import os
from pathlib import Path

from ckh import clintercept


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
