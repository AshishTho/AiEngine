"""Real local HTTP tests for browser authorization and service integration."""

from __future__ import annotations

import http.client
import json
import threading
import unittest
from urllib.parse import quote

from goal_agent.web import create_server


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.runs = {"run-1": {"run_id": "run-1", "goal": "Write a report", "status": "planned"}}

    def list_runs(self):
        self.calls.append(("list",))
        return list(self.runs.values())

    def get_run(self, run_id):
        self.calls.append(("get", run_id))
        return self.runs[run_id]

    def start(self, goal, plan_only=False, demo=False):
        self.calls.append(("start", goal, plan_only, demo))
        self.runs["run-2"] = {"run_id": "run-2", "goal": goal, "status": "planning"}
        return "run-2"

    def execute(self, run_id):
        self.calls.append(("execute", run_id))
        self.runs[run_id]["status"] = "running"
        return run_id

    def resume(self, run_id):
        self.calls.append(("resume", run_id))
        if run_id == "run-1":
            raise RuntimeError("Run is already active.")
        return run_id

    def cancel(self, run_id):
        self.calls.append(("cancel", run_id))
        self.runs[run_id]["status"] = "cancelled"
        return True

    def artifact(self, run_id, path):
        self.calls.append(("artifact", run_id, path))
        self.runs[run_id]
        if path != "reports/report.html":
            raise FileNotFoundError(path)
        return b"<script>alert('untrusted output')</script>", "text/html"


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(FakeService(), port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def setUp(self):
        self.service = FakeService()
        self.server.service = self.service

    def request(self, method, path, body=None, *, authenticated=True, origin=True, extra=None):
        host, port = self.server.server_address[:2]
        connection = http.client.HTTPConnection(host, port, timeout=3)
        headers = {}
        if authenticated:
            headers["Authorization"] = "Bearer " + self.server.auth_token
        if origin:
            headers["Origin"] = self.server.origin
        if body is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(body).encode("utf-8")
        headers.update(extra or {})
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_launch_url_uses_fragment_and_page_never_contains_token(self):
        self.assertIn("/#token=", self.server.entry_url)
        self.assertGreaterEqual(len(self.server.auth_token), 40)
        status, headers, body = self.request("GET", "/", authenticated=False, origin=False)
        self.assertEqual(status, 200)
        self.assertIn(b"AiEngine", body)
        self.assertNotIn(self.server.auth_token.encode(), body)
        self.assertNotIn(b"__CSP_NONCE__", body)
        self.assertIn("nonce-", headers["Content-Security-Policy"])
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_all_api_reads_and_downloads_require_authentication(self):
        for path in ("/api/runs", "/api/runs/run-1", "/api/runs/run-1/artifacts?path=reports/report.html"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path, authenticated=False)[0], 401)
        self.assertEqual(self.service.calls, [])

    def test_wrong_token_cannot_read_or_mutate(self):
        extra = {"Authorization": "Bearer wrong-token"}
        self.assertEqual(self.request("GET", "/api/runs", extra=extra)[0], 401)
        self.assertEqual(self.request("POST", "/api/runs", {"goal": "hello"}, extra=extra)[0], 401)
        self.assertEqual(self.service.calls, [])

    def test_writes_require_same_origin_and_known_host(self):
        scenarios = [
            {"origin": False},
            {"extra": {"Origin": "https://evil.example"}},
            {"extra": {"Host": "evil.example"}},
            {"extra": {"Sec-Fetch-Site": "cross-site"}},
        ]
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                self.assertEqual(self.request("POST", "/api/runs", {"goal": "hello"}, **scenario)[0], 403)
        self.assertEqual(self.service.calls, [])

    def test_cross_origin_reads_and_rebinding_are_rejected(self):
        self.assertEqual(self.request("GET", "/api/runs", extra={"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.request("GET", "/", authenticated=False, extra={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request("OPTIONS", "/api/runs", authenticated=False)[0], 405)

    def test_history_plan_preview_execution_and_cancellation(self):
        status, _, body = self.request("GET", "/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["runs"][0]["run_id"], "run-1")
        status, _, body = self.request("POST", "/api/runs", {"goal": "  A cited report  ", "plan_only": True, "demo": True})
        self.assertEqual(status, 202)
        self.assertEqual(json.loads(body)["run_id"], "run-2")
        self.assertIn(("start", "A cited report", True, True), self.service.calls)
        self.assertEqual(self.request("POST", "/api/runs/run-2/execute", {})[0], 202)
        self.assertEqual(json.loads(self.request("GET", "/api/runs/run-2")[2])["status"], "running")
        self.assertEqual(json.loads(self.request("POST", "/api/runs/run-2/cancel", {})[2])["cancelled"], True)
        self.assertEqual(self.request("POST", "/api/runs/run-2/resume", {})[0], 202)

    def test_action_conflicts_and_missing_runs_have_actionable_status(self):
        self.assertEqual(self.request("POST", "/api/runs/run-1/resume", {})[0], 409)
        self.assertEqual(self.request("GET", "/api/runs/missing")[0], 404)
        self.assertEqual(self.request("POST", "/api/runs/missing/execute", {})[0], 404)

    def test_start_validation_prevents_invalid_options_reaching_service(self):
        for body in ({}, {"goal": " "}, {"goal": 5}, {"goal": "x", "demo": "false"},
                     {"goal": "x", "plan_only": 1}, {"goal": "x", "workspace": "C:/"},
                     {"goal": "x" * 12001}, ["x"]):
            with self.subTest(body=str(body)[:80]):
                self.assertEqual(self.request("POST", "/api/runs", body)[0], 400)
        self.assertEqual(self.service.calls, [])

    def test_oversized_and_non_json_payloads_are_rejected(self):
        self.assertEqual(self.request("POST", "/api/runs", {"goal": "x" * 40000})[0], 400)
        self.assertEqual(self.request("POST", "/api/runs", {"goal": "x"}, extra={"Content-Type": "text/plain"})[0], 400)
        self.assertEqual(self.request("POST", "/api/runs/run-1/execute", {"goal": "extra"})[0], 400)
        self.assertEqual(self.service.calls, [])

    def test_artifact_download_forces_attachment_even_for_html(self):
        status, headers, body = self.request("GET", "/api/runs/run-1/artifacts?path=reports%2Freport.html")
        self.assertEqual(status, 200)
        self.assertIn(b"<script>", body)
        self.assertEqual(headers["Content-Type"], "application/octet-stream")
        self.assertEqual(headers["Content-Disposition"], 'attachment; filename="report.html"')
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("sandbox", headers["Content-Security-Policy"])

    def test_traversal_paths_never_reach_service(self):
        for path in ("../secrets", "reports/../../secrets", "/etc/passwd", "C:\\secret", "\\\\server\\share",
                     "reports/./report.html", "reports\x00/report.html", "reports//report.html"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", "/api/runs/run-1/artifacts?path=" + quote(path, safe=""))[0], 400)
        self.assertEqual(self.request("GET", "/api/runs/run-1/artifacts?path=a&path=b")[0], 400)
        self.assertEqual(self.request("GET", "/api/runs/%2E%2E")[0], 400)
        self.assertEqual(self.service.calls, [])

    def test_unknown_routes_do_not_invoke_service(self):
        self.assertEqual(self.request("GET", "/api/runs/run-1/unknown")[0], 404)
        self.assertEqual(self.request("POST", "/api/runs/run-1/unknown", {})[0], 404)
        self.assertEqual(self.request("GET", "/api/runs?unexpected=1")[0], 404)
        self.assertEqual(self.service.calls, [])

    def test_public_interfaces_are_rejected(self):
        for host in ("0.0.0.0", "::", "192.168.1.1", "example.com"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "loopback"):
                create_server(self.service, host=host, port=0)


if __name__ == "__main__":
    unittest.main()
