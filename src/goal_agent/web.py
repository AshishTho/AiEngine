"""Authenticated, loopback-only browser interface for one local user.

The service owns execution, persistence, cancellation and filesystem confinement.
This module only adapts that service to a small HTTP API. It deliberately has no
deployment mode: use a production server with real user isolation for hosting.
"""

from __future__ import annotations

import ipaddress
import json
import re
import secrets
import socket
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any, Protocol
from urllib.parse import parse_qs, unquote, urlsplit


class AgentService(Protocol):
    def list_runs(self) -> list[dict[str, Any]]: ...
    def get_run(self, run_id: str) -> dict[str, Any]: ...
    def start(self, goal: str, plan_only: bool = False, demo: bool = False) -> str: ...
    def execute(self, run_id: str) -> str: ...
    def resume(self, run_id: str) -> str: ...
    def cancel(self, run_id: str) -> bool: ...
    def artifact(self, run_id: str, path: str) -> tuple[bytes, str]: ...


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_MAX_BODY = 32_768


class AgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], service: AgentService) -> None:
        self.service = service
        self.auth_token = secrets.token_urlsafe(32)
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, _Handler)
        host, port = self.server_address[:2]
        authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        self.authority = authority
        self.origin = f"http://{authority}"
        self.entry_url = f"{self.origin}/#token={self.auth_token}"


class _Handler(BaseHTTPRequestHandler):
    server: AgentHTTPServer
    server_version = "AiEngine"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        # Goals, artifact names and bearer tokens should not enter access logs.
        pass

    def _respond(self, status: int, body: bytes, content_type: str,
                 headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._respond(status, json.dumps(payload, ensure_ascii=True).encode("utf-8"),
                      "application/json; charset=utf-8")

    def _check_request(self, authenticated: bool, write: bool = False) -> bool:
        # An exact Host check also prevents DNS rebinding to this local service.
        if self.headers.get("Host") != self.server.authority:
            self._json(HTTPStatus.FORBIDDEN, {"error": "Unrecognized request host."})
            return False
        origin = self.headers.get("Origin")
        if (write and not origin) or (origin and origin != self.server.origin):
            self._json(HTTPStatus.FORBIDDEN, {"error": "A same-origin request is required."})
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._json(HTTPStatus.FORBIDDEN, {"error": "Cross-site requests are disabled."})
            return False
        if authenticated:
            expected = f"Bearer {self.server.auth_token}".encode("utf-8")
            provided = self.headers.get("Authorization", "").encode("utf-8")
            if not secrets.compare_digest(expected, provided):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "Open the launch URL printed in your terminal to sign in."})
                return False
        return True

    def _body(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked request bodies are not supported.")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("The request body must be application/json.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid content length.") from exc
        if not 0 < length <= _MAX_BODY:
            raise ValueError("Request body must be between 1 and 32768 bytes.")
        # Bound slow local clients rather than leaving request threads occupied.
        self.connection.settimeout(10)
        try:
            value = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeError) as exc:
            raise ValueError("The request body contains invalid JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError("The request body must be a JSON object.")
        return value

    def _route(self) -> tuple[list[str], dict[str, list[str]]]:
        url = urlsplit(self.path)
        if url.scheme or url.netloc or url.fragment:
            raise ValueError("Invalid request URL.")
        parts = [unquote(part) for part in url.path.strip("/").split("/")]
        if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
            if not _RUN_ID.fullmatch(parts[2]):
                raise ValueError("Invalid run identifier.")
        return parts, parse_qs(url.query, keep_blank_values=True, max_num_fields=10)

    def _safe_artifact_path(self, query: dict[str, list[str]]) -> str:
        if set(query) != {"path"} or len(query["path"]) != 1:
            raise ValueError("Exactly one artifact path is required.")
        path = query["path"][0]
        parts = path.replace("\\", "/").split("/")
        if (not path or len(path) > 2048 or any(ord(c) < 32 for c in path)
                or ":" in path or any(part in ("", ".", "..") for part in parts)):
            raise ValueError("Artifact paths must be relative workspace paths.")
        return "/".join(parts)

    def do_GET(self) -> None:
        try:
            route, query = self._route()
            public = route == [""] and not query
            if not self._check_request(authenticated=not public):
                return
            if public:
                nonce = secrets.token_urlsafe(24)
                html = files("goal_agent").joinpath("static/index.html").read_text(encoding="utf-8")
                html = html.replace("__CSP_NONCE__", nonce)
                policy = (f"default-src 'none'; script-src 'nonce-{nonce}'; "
                          f"style-src 'nonce-{nonce}'; connect-src 'self'; "
                          "base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
                self._respond(200, html.encode("utf-8"), "text/html; charset=utf-8",
                              {"Content-Security-Policy": policy})
            elif route == ["api", "runs"] and not query:
                self._json(200, {"runs": self.server.service.list_runs()})
            elif len(route) == 3 and route[:2] == ["api", "runs"] and not query:
                self._json(200, self.server.service.get_run(route[2]))
            elif len(route) == 4 and route[:2] == ["api", "runs"] and route[3] == "artifacts":
                path = self._safe_artifact_path(query)
                data, _content_type = self.server.service.artifact(route[2], path)
                filename = re.sub(r"[^A-Za-z0-9._-]", "_", path.rsplit("/", 1)[-1]) or "artifact"
                # Always force a download, even if an agent created HTML or SVG.
                self._respond(200, data, "application/octet-stream", {
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Security-Policy": "sandbox; default-src 'none'",
                })
            else:
                self._json(404, {"error": "Endpoint not found."})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self._error(exc)

    def do_POST(self) -> None:
        try:
            if not self._check_request(authenticated=True, write=True):
                return
            route, query = self._route()
            if query:
                raise ValueError("Query parameters are not accepted for this action.")
            body = self._body()
            if route == ["api", "runs"]:
                if set(body) - {"goal", "plan_only", "demo"}:
                    raise ValueError("Unrecognized start options.")
                goal = body.get("goal")
                if not isinstance(goal, str) or not 1 <= len(goal.strip()) <= 12_000:
                    raise ValueError("Enter a goal between 1 and 12000 characters.")
                plan_only, demo = body.get("plan_only", False), body.get("demo", False)
                if type(plan_only) is not bool or type(demo) is not bool:
                    raise ValueError("plan_only and demo must be booleans.")
                run_id = self.server.service.start(goal.strip(), plan_only=plan_only, demo=demo)
                self._json(202, {"run_id": run_id})
            elif (len(route) == 4 and route[:2] == ["api", "runs"]
                  and route[3] in ("execute", "resume", "cancel")):
                if body:
                    raise ValueError("This action takes an empty JSON object.")
                action = route[3]
                if action == "cancel":
                    self._json(202, {"run_id": route[2], "cancelled": self.server.service.cancel(route[2])})
                else:
                    run_id = getattr(self.server.service, action)(route[2])
                    self._json(202, {"run_id": run_id})
            else:
                self._json(404, {"error": "Endpoint not found."})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self._error(exc)

    def _error(self, exc: Exception) -> None:
        if isinstance(exc, (KeyError, FileNotFoundError)):
            status, message = 404, "Run or artifact not found."
        elif isinstance(exc, (ValueError, TypeError)):
            status, message = 400, str(exc)
        elif isinstance(exc, PermissionError):
            status, message = 403, "This artifact is not accessible."
        elif isinstance(exc, RuntimeError):
            status, message = 409, str(exc)
        elif isinstance(exc, TimeoutError):
            status, message = 408, "Request timed out."
        else:
            status, message = 500, "The local service encountered an unexpected error. Check the run log."
        self._json(status, {"error": message})

    def do_OPTIONS(self) -> None:
        self._json(405, {"error": "Cross-origin API access is disabled."})


def create_server(service: AgentService, host: str = "127.0.0.1", port: int = 8765) -> AgentHTTPServer:
    """Construct a server; port 0 selects an unused port for integration tests."""
    if host == "localhost":
        host = "127.0.0.1"
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if not local:
        raise ValueError("The local UI must bind to a loopback address (127.0.0.1 or ::1).")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535.")
    return AgentHTTPServer((host, port), service)


def serve(service: AgentService, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the local UI until interrupted. Keep the printed launch URL private."""
    server = create_server(service, host, port)
    print(f"AiEngine local UI: {server.entry_url}", flush=True)
    print("Open this URL in your browser. Press Ctrl+C to stop the UI.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
