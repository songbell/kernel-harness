"""Protocol-level coverage of the MCP layer. No GPU, no clops, no platform.toml needed.

These tests exist because a broken MCP server fails *silently* from the model's side: the
tool list simply comes back empty or the client drops the connection, and the failure is
indistinguishable from "the model chose not to call anything". So the handshake, the tool
schemas and the two refusals that protect the server are asserted here rather than
discovered in a chat session.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ckh.mcp import McpServer  # noqa: E402
from ckh.mcp import server as ckh_server  # noqa: E402


def call(srv, method, params=None, msg_id=1):
    return srv._handle({"jsonrpc": "2.0", "id": msg_id, "method": method,
                        **({"params": params} if params else {})})


def test_initialize_advertises_tools_capability():
    r = call(ckh_server.server, "initialize")
    assert r["result"]["serverInfo"]["name"] == "ckh"
    assert "tools" in r["result"]["capabilities"]


def test_tools_list_exposes_the_harness_surface():
    tools = {t["name"]: t for t in call(ckh_server.server, "tools/list")["result"]["tools"]}
    assert {"doctor", "bench", "equiv", "round", "snapshot",
            "ledger_query", "ledger_add", "list_kernels",
            "profile_run", "profile_status", "profile_report", "kernel_prepare"} <= set(tools)
    for t in tools.values():
        assert t["description"].strip(), f"{t['name']} has no description for the model to read"
        assert t["inputSchema"]["type"] == "object"


def test_notifications_get_no_response():
    assert ckh_server.server._handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_tool_is_a_jsonrpc_error():
    r = call(ckh_server.server, "tools/call", {"name": "nope", "arguments": {}})
    assert r["error"]["code"] == -32601


def test_tool_exception_is_content_not_transport_error():
    """A failing measurement must reach the model as readable text, not kill the session."""
    srv = McpServer("t")

    @srv.tool("boom", "always fails", {"type": "object", "properties": {}})
    def _boom(_a):
        raise RuntimeError("rig is on fire")

    r = call(srv, "tools/call", {"name": "boom", "arguments": {}})
    assert r["result"]["isError"] is True
    assert "rig is on fire" in r["result"]["content"][0]["text"]


@pytest.mark.parametrize("name", ["../../etc/passwd", "os.path", "a-b", "", "kernels.x"])
def test_kernel_name_must_be_a_plain_module_name(name):
    """The name reaches importlib on a server that may be listening on a socket."""
    with pytest.raises(ValueError):
        ckh_server._spec(name)


def test_unknown_but_wellformed_kernel_is_refused_before_import():
    with pytest.raises(ValueError, match="no descriptor"):
        ckh_server._spec("definitely_not_a_kernel_here")


def test_axes_accept_both_csv_and_list(monkeypatch):
    monkeypatch.setattr(ckh_server, "_spec", lambda n: (None, {"q_len": [1], "kv": [8]}))
    assert ckh_server._axes("k", {"q_len": "6,16"}) == {"q_len": [6, 16], "kv": [8]}
    assert ckh_server._axes("k", {"q_len": [6]}) == {"q_len": [6], "kv": [8]}
    assert ckh_server._axes("k", None) == {"q_len": [1], "kv": [8]}


def test_measurement_tools_refuse_while_other_gpu_work_runs(monkeypatch):
    """The whole point of the harness: numbers taken next to competing work are junk."""
    class _Plat:
        def competing_gpu_work(self):
            return ["pid 1 benchmark_app"]

    monkeypatch.setattr(ckh_server.Platform, "load", classmethod(lambda cls: _Plat()))
    with pytest.raises(RuntimeError, match="refusing to measure"):
        ckh_server._guarded_platform()


def test_result_is_serializable_text():
    r = call(ckh_server.server, "tools/call", {"name": "list_kernels", "arguments": {}})
    json.loads(r["result"]["content"][0]["text"])


def test_profile_run_refuses_when_disabled(monkeypatch):
    monkeypatch.setattr(ckh_server, "_PROFILE_RUN_ALLOWED", False)
    with pytest.raises(RuntimeError, match="disabled"):
        ckh_server._profile_run({"command": ["app"], "out_dir": "results/profile_test"})


def test_profile_run_uses_platform_pipeline_when_command_is_omitted(monkeypatch, tmp_path):
    class _Plat:
        raw = {"profile": {"pipeline": ["demo-app", "--flag"], "install_prefix": "prefix"}}

    monkeypatch.setattr(ckh_server, "_guarded_platform", lambda: _Plat())
    monkeypatch.setattr(ckh_server.clintercept, "locate", lambda prefix, pinned: tmp_path / "cliloader")
    monkeypatch.setattr(
        ckh_server.clintercept,
        "run",
        lambda tool, out_dir, label, command, repeat, env, timeout, dump_sources=None: [
            {"exit_code": 0, "trace_present": True, "command": command,
             "dump_sources": str(tmp_path / "ov_gpu_dump_sources")}
        ],
    )

    result = ckh_server._profile_run({"out_dir": "results/profile_test"})

    assert result["ok"] is True
    assert result["runs"][0]["command"] == ["demo-app", "--flag"]
    assert result["dump_sources"] == str(tmp_path / "ov_gpu_dump_sources")


def test_kernel_prepare_ports_and_emits_scaffolds(monkeypatch, tmp_path):
    sandbox = tmp_path / "sandbox"
    production = tmp_path / "prod"
    production_cm = production / ckh_server.kernelgen.CM_SUBDIR
    production_cm.mkdir(parents=True)
    source = production_cm / "foo.cm"
    source.write_text("extern \"C\" _GENX_MAIN_ void KERNEL_NAME(int x) {}")

    _Plat = type(
        "_Plat",
        (),
        {"sandbox": sandbox, "production": production, "kernelgen_dest": "opencl/tests"},
    )

    class _Src:
        def __init__(self, path):
            self.path = path
            self.entry = "KERNEL_NAME"
            self.params = []
            self.includes = []
            self.missing_includes = []
            self.undefined = set()

        def resolve_entry(self, profiled):
            self.entry = profiled

        def required(self, hints):
            return set()

    monkeypatch.setattr(ckh_server.Platform, "load", classmethod(lambda cls: _Plat()))
    monkeypatch.setattr(ckh_server.kernelgen, "parse_source", lambda path, include_dirs: _Src(path))
    monkeypatch.setattr(ckh_server.kernelgen, "include_dirs_for", lambda path, production: [path.parent])
    monkeypatch.setattr(
        ckh_server.kernelgen,
        "scrape_host",
        lambda source_root, generator: ckh_server.kernelgen.HostHints(generator=generator),
    )

    result = ckh_server._kernel_prepare({
        "kernel": "cm_foo",
        "source": str(source),
        "axes": {"q_len": "6,16", "past_len": "15360"},
    })

    generated = result["generated"]
    assert Path(generated["sandbox_kernel"]).is_file()
    assert Path(generated["wrapper"]).is_file()
    assert Path(generated["compile_test"]).is_file()
    assert Path(generated["correctness_test"]).is_file()
    assert Path(generated["perf_test"]).is_file()
    assert Path(generated["spec"]).is_file()


def test_kernel_prepare_prefers_dumped_runtime_source(monkeypatch, tmp_path):
    sandbox = tmp_path / "sandbox"
    production = tmp_path / "prod"
    dump_root = tmp_path / "ovdump"
    dump_root.mkdir()
    dumped = dump_root / "sdpa_micro__generate_123456789__sa.cl"
    dumped.write_text("__kernel void sdpa_micro__generate_123456789__sa(int x) {}")

    _Plat = type(
        "_Plat",
        (),
        {
            "sandbox": sandbox,
            "production": production,
            "kernelgen_dest": "opencl/tests",
            "raw": {"profile": {"pipeline_env": {"OV_GPU_DUMP_SOURCES_PATH": str(dump_root)}}},
        },
    )

    class _Src:
        def __init__(self, path):
            self.path = path
            self.entry = "sdpa_micro__generate_123456789__sa"
            self.params = []
            self.includes = []
            self.missing_includes = []
            self.undefined = set()

        def resolve_entry(self, profiled):
            self.entry = profiled

        def required(self, hints):
            return set()

    monkeypatch.setattr(ckh_server.Platform, "load", classmethod(lambda cls: _Plat()))
    monkeypatch.setattr(ckh_server.kernelgen, "parse_source", lambda path, include_dirs: _Src(path))
    monkeypatch.setattr(ckh_server.kernelgen, "include_dirs_for", lambda path, production: [path.parent])
    monkeypatch.setattr(
        ckh_server.kernelgen,
        "scrape_host",
        lambda source_root, generator: ckh_server.kernelgen.HostHints(generator=generator),
    )
    monkeypatch.setattr(ckh_server.kernelgen, "candidates", lambda production, kernel: [])

    result = ckh_server._kernel_prepare({"kernel": "sdpa_micro__generate"})

    assert Path(result["source"]) == dumped


def test_kernel_prepare_reads_dump_source_from_profile_dir(monkeypatch, tmp_path):
    sandbox = tmp_path / "sandbox"
    production = tmp_path / "prod"
    profile_dir = tmp_path / "profile"
    dump_root = tmp_path / "ovdump"
    profile_dir.mkdir()
    dump_root.mkdir()
    dumped = dump_root / "foo_123456789__sa.cl"
    dumped.write_text("__kernel void foo_123456789__sa(int x) {}")
    (profile_dir / "profile_run.json").write_text(json.dumps({"dump_sources": str(dump_root)}))

    _Plat = type(
        "_Plat",
        (),
        {"sandbox": sandbox, "production": production, "kernelgen_dest": "opencl/tests", "raw": {}},
    )

    class _Src:
        def __init__(self, path):
            self.path = path
            self.entry = "foo_123456789__sa"
            self.params = []
            self.includes = []
            self.missing_includes = []
            self.undefined = set()

        def resolve_entry(self, profiled):
            self.entry = profiled

        def required(self, hints):
            return set()

    monkeypatch.setattr(ckh_server.Platform, "load", classmethod(lambda cls: _Plat()))
    monkeypatch.setattr(ckh_server.kernelgen, "parse_source", lambda path, include_dirs: _Src(path))
    monkeypatch.setattr(ckh_server.kernelgen, "include_dirs_for", lambda path, production: [path.parent])
    monkeypatch.setattr(
        ckh_server.kernelgen,
        "scrape_host",
        lambda source_root, generator: ckh_server.kernelgen.HostHints(generator=generator),
    )
    monkeypatch.setattr(ckh_server.kernelgen, "candidates", lambda production, kernel: [])

    result = ckh_server._kernel_prepare({"kernel": "foo", "profile_dump_dir": str(profile_dir)})

    assert Path(result["source"]) == dumped


def test_kernel_prepare_falls_back_to_last_profile_metadata(monkeypatch, tmp_path):
    sandbox = tmp_path / "sandbox"
    production = tmp_path / "prod"
    profile_dir = tmp_path / "profile"
    dump_root = tmp_path / "ovdump"
    profile_dir.mkdir()
    dump_root.mkdir()
    dumped = dump_root / "bar_123456789__sa.cl"
    dumped.write_text("__kernel void bar_123456789__sa(int x) {}")

    monkeypatch.setattr(ckh_server.results, "last_profile",
                        lambda: {"out_dir": str(profile_dir), "dump_sources": str(dump_root)})
    monkeypatch.setattr(ckh_server.kernelgen, "dump_sources_from_profile_dir", lambda p: dump_root)

    _Plat = type(
        "_Plat",
        (),
        {"sandbox": sandbox, "production": production, "kernelgen_dest": "opencl/tests", "raw": {}},
    )

    class _Src:
        def __init__(self, path):
            self.path = path
            self.entry = "bar_123456789__sa"
            self.params = []
            self.includes = []
            self.missing_includes = []
            self.undefined = set()

        def resolve_entry(self, profiled):
            self.entry = profiled

        def required(self, hints):
            return set()

    monkeypatch.setattr(ckh_server.Platform, "load", classmethod(lambda cls: _Plat()))
    monkeypatch.setattr(ckh_server.kernelgen, "parse_source", lambda path, include_dirs: _Src(path))
    monkeypatch.setattr(ckh_server.kernelgen, "include_dirs_for", lambda path, production: [path.parent])
    monkeypatch.setattr(
        ckh_server.kernelgen,
        "scrape_host",
        lambda source_root, generator: ckh_server.kernelgen.HostHints(generator=generator),
    )
    monkeypatch.setattr(ckh_server.kernelgen, "candidates", lambda production, kernel: [])

    result = ckh_server._kernel_prepare({"kernel": "bar"})

    assert Path(result["source"]) == dumped


def test_profile_prepare_selects_from_last_profile_and_prepares(monkeypatch, tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "profile_summary.json").write_text(json.dumps({
        "phase_hotspots": {"main_generate": [{"kernel": "sdpa_micro__generate_pick", "total_ms": 1.0}]}
    }))

    monkeypatch.setattr(ckh_server.results, "last_profile", lambda: {"out_dir": str(profile_dir)})
    monkeypatch.setattr(
        ckh_server,
        "_kernel_prepare",
        lambda args: {"source": "picked.cl", "generated": {"spec": "kernels/picked.py"}, "kernel": args["kernel"]},
    )

    result = ckh_server._profile_prepare({})

    assert result["selected"]["kernel"] == "sdpa_micro__generate_pick"
    assert result["kernel"] == "sdpa_micro__generate_pick"
