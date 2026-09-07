"""A tiny, dependency-free Model Context Protocol server.

Hand-rolled rather than pulling in the `mcp` SDK: the measurement box is often offline and
the harness has no runtime dependencies today, which is a property worth keeping. Implements
just enough of MCP for GitHub Copilot (VS Code Chat / Copilot CLI) to discover and call
tools: JSON-RPC 2.0 over stdio or Streamable-HTTP, `initialize`, `tools/list`, `tools/call`.

Structure mirrors the OpenVINO engineering harness's `mcp_servers/_mcp` so the two are
recognisably the same thing to anyone who has read one of them.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

PROTOCOL_VERSION = "2024-11-05"


@dataclass
class _Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]


class McpServer:
    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, _Tool] = {}

    def tool(self, name: str, description: str, input_schema: dict[str, Any]):
        """Decorator: register a tool handler ``fn(arguments: dict) -> Any``."""
        def deco(fn: Callable[[dict[str, Any]], Any]):
            self._tools[name] = _Tool(name, description, input_schema, fn)
            return fn
        return deco

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    # -- transport ---------------------------------------------------------------------

    def serve_stdio(self) -> int:
        """stdout is the protocol channel: anything a tool prints would corrupt it.

        Measurement code underneath is chatty (clops prints a build/GPU banner on import),
        so stdout is redirected to stderr for the whole session and only framed JSON-RPC
        goes to the real stdout.
        """
        import contextlib

        real_stdout = sys.stdout
        with contextlib.redirect_stdout(sys.stderr):
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response = self._handle(msg)
                if response is not None:
                    real_stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                    real_stdout.flush()
        return 0

    def serve_http(self, host: str = "127.0.0.1", port: int = 8791,
                   auth_token: str | None = None) -> int:
        """Streamable-HTTP: ``POST /mcp`` JSON-RPC, ``GET /healthz``.

        This is how a remote GPU box serves its harness to a laptop's VS Code. Unlike a
        knowledge-base server, every tool here *executes kernel code on the GPU*, so the
        default bind is loopback and a bearer token (``MCP_AUTH_TOKEN``) is strongly
        advised whenever the bind is widened.
        """
        import contextlib
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        srv = self
        token = auth_token if auth_token is not None else os.environ.get("MCP_AUTH_TOKEN")

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code: int, payload: Any = None) -> None:
                body = b"" if payload is None else (
                    payload if isinstance(payload, bytes)
                    else json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                self.send_response(code)
                if body:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _authorized(self) -> bool:
                return not token or self.headers.get("Authorization", "") == f"Bearer {token}"

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/") in ("/healthz", "/health"):
                    self._send(200, {"status": "ok", "server": srv.name,
                                     "version": srv.version, "tools": srv.tool_names})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                if not self._authorized():
                    self._send(401, {"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32001, "message": "unauthorized"}})
                    return
                if self.path.rstrip("/") not in ("/mcp", ""):
                    self._send(404, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    msg = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    self._send(400, {"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "parse error"}})
                    return
                # Tool bodies run measurement code that prints; keep it off this socket.
                with contextlib.redirect_stdout(sys.stderr):
                    if isinstance(msg, list):
                        out = [r for r in (srv._handle(m) for m in msg) if r is not None]
                        self._send(200, out)
                        return
                    resp = srv._handle(msg)
                if resp is None:
                    self._send(202)
                    return
                self._send(200, resp)

            def log_message(self, *_a):
                return

        httpd = ThreadingHTTPServer((host, port), _Handler)
        sys.stderr.write(f"[{self.name}] MCP Streamable-HTTP on http://{host}:{port}/mcp"
                         f"{' (auth: bearer)' if token else ' (auth: NONE)'}\n")
        sys.stderr.flush()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return 0

    # -- dispatch ----------------------------------------------------------------------

    def _handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method = msg.get("method")
        msg_id = msg.get("id")

        if method and method.startswith("notifications/"):
            return None

        if method == "initialize":
            return self._result(msg_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": self.name, "version": self.version},
            })

        if method == "tools/list":
            return self._result(msg_id, {
                "tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                    for t in self._tools.values()
                ]
            })

        if method == "tools/call":
            params = msg.get("params", {}) or {}
            name = params.get("name")
            arguments = params.get("arguments", {}) or {}
            tool = self._tools.get(name)
            if tool is None:
                return self._error(msg_id, -32601, f"unknown tool: {name}")
            try:
                result = tool.handler(arguments)
                return self._result(msg_id, {
                    "content": [{"type": "text", "text": _as_text(result)}],
                    "isError": False,
                })
            except Exception as exc:  # tool errors are content, not transport errors
                return self._result(msg_id, {
                    "content": [{"type": "text", "text": f"error: {exc}"}],
                    "isError": True,
                })

        if method == "ping":
            return self._result(msg_id, {})

        return self._error(msg_id, -32601, f"method not found: {method}")

    @staticmethod
    def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _as_text(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
