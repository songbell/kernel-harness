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
            "ledger_query", "ledger_add", "list_kernels"} <= set(tools)
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
