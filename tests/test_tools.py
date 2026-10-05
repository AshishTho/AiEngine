"""Tool boundary tests; no credentials or network access required."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from goal_agent.tools import ToolRegistry


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "workspace"
        self.tools = ToolRegistry(self.workspace, "test-secret-key", max_file_bytes=100)

    def execute(self, name: str, **arguments: object) -> dict:
        return self.tools.execute(name, json.dumps(arguments))

    def test_create_nested_utf8_file_and_refuse_overwrite(self) -> None:
        result = self.execute("create_file", path="notes/report.txt", content="caf\u00e9\n")
        self.assertEqual(result, {"ok": True, "result": {
            "path": "notes/report.txt", "bytes_written": 6,
            "sha256": hashlib.sha256("caf\u00e9\n".encode()).hexdigest(),
        }})
        self.assertEqual((self.workspace / "notes/report.txt").read_bytes(), b"caf\xc3\xa9\n")
        result = self.execute("create_file", path="notes/report.txt", content="replacement")
        self.assertFalse(result["ok"])
        self.assertEqual((self.workspace / "notes/report.txt").read_bytes(), b"caf\xc3\xa9\n")

    def test_reject_nonportable_and_escaping_paths(self) -> None:
        paths = [
            "../escape.txt", "notes/../../escape.txt", "notes\\..\\escape.txt", "/tmp/escape.txt",
            "C:\\escape.txt", "C:escape.txt", "\\\\server\\share\\escape.txt", "\\escape.txt",
            "report.txt:stream", "CON", "nul.txt", "COM1.log", "lpt9", "COM\u00b9.txt",
            "dir/NUL/data.txt", "dir./file.txt", "dir /file.txt", "trailing.", "trailing ",
            "a//b", "./file", "", "file\x00.txt", "star*.txt", "question?.txt",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertFalse(self.execute("create_file", path=path, content="test")["ok"])
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_byte_limit_counts_encoded_utf8(self) -> None:
        result = self.execute("create_file", path="large.txt", content="\u00e9" * 51)
        self.assertFalse(result["ok"])
        self.assertFalse((self.workspace / "large.txt").exists())

    def test_invalid_utf8_is_a_tool_error(self) -> None:
        self.assertFalse(self.execute("create_file", path="invalid.txt", content="\ud800")["ok"])
        self.assertFalse(self.execute("create_file", path="\ud800.txt", content="test")["ok"])
        self.assertFalse(self.execute("web_search", query="\ud800", max_results=1)["ok"])

    def test_reject_symlink_to_outside(self) -> None:
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        link = self.workspace / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("This OS/user cannot create symbolic links.")
        self.assertFalse(self.execute("create_file", path="link/escape.txt", content="test")["ok"])
        self.assertFalse((outside / "escape.txt").exists())

    def test_reject_existing_file_symlink(self) -> None:
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("original", encoding="utf-8")
        try:
            (self.workspace / "link.txt").symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("This OS/user cannot create symbolic links.")
        self.assertFalse(self.execute("create_file", path="link.txt", content="test")["ok"])
        self.assertEqual(outside.read_text(encoding="utf-8"), "original")

    def test_reject_malformed_missing_extra_duplicate_and_wrong_type_arguments(self) -> None:
        cases = ["{", "[]", '{"path":"a"}', '{"path":"a","content":"b","extra":1}',
                 '{"path":"a","path":"b","content":"c"}', '{"path":null,"content":"b"}',
                 '{"path":"a","content":42}', '{"path":"a","content":NaN}']
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assertFalse(self.tools.execute("create_file", arguments)["ok"])
        self.assertFalse(self.tools.execute("shell", '{}')["ok"])

    def test_schemas_are_strict_and_copied(self) -> None:
        for schema in self.tools.schemas:
            self.assertTrue(schema["strict"])
            parameters = schema["parameters"]
            self.assertFalse(parameters["additionalProperties"])
            self.assertEqual(set(parameters["properties"]), set(parameters["required"]))
        first = self.tools.schemas
        first[0]["name"] = "changed"
        self.assertEqual(self.tools.schemas[0]["name"], "web_search")

    @patch("goal_agent.tools.httpx.post")
    def test_search_auth_limits_and_citations(self, post) -> None:
        request = httpx.Request("POST", "https://api.tavily.com/search")
        post.return_value = httpx.Response(200, request=request, json={"results": [
            {"title": "Source", "url": "https://example.com/article", "content": "a" * 2000},
            {"title": "Other", "url": "https://example.com/other", "content": "b"},
        ]})
        result = self.execute("web_search", query=" reference ", max_results=1)
        self.assertTrue(result["ok"])
        sources = result["result"]["sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["url"], "https://example.com/article")
        self.assertEqual(len(sources[0]["snippet"]), 1500)
        arguments = post.call_args.kwargs
        self.assertEqual(arguments["headers"], {"Authorization": "Bearer test-secret-key"})
        self.assertEqual(arguments["json"]["query"], "reference")
        self.assertEqual(arguments["json"]["max_results"], 1)
        self.assertFalse(arguments["follow_redirects"])
        self.assertEqual(arguments["timeout"].read, 20.0)

    @patch("goal_agent.tools.httpx.post")
    def test_search_argument_bounds_do_not_call_network(self, post) -> None:
        for query, maximum in [("", 1), (" ", 1), ("a" * 501, 1), ("ok", 0),
                               ("ok", 6), ("ok", True), ("ok", 1.0), (42, 1)]:
            with self.subTest(query=query, maximum=maximum):
                self.assertFalse(self.execute("web_search", query=query, max_results=maximum)["ok"])
        post.assert_not_called()

    @patch("goal_agent.tools.httpx.post")
    def test_missing_search_key_does_not_call_network(self, post) -> None:
        tools = ToolRegistry(self.workspace, None)
        result = tools.execute("web_search", '{"query":"test","max_results":1}')
        self.assertFalse(result["ok"])
        self.assertIn("TAVILY_API_KEY", result["error"])
        post.assert_not_called()

    @patch("goal_agent.tools.httpx.post")
    def test_search_error_does_not_expose_secret_or_body(self, post) -> None:
        request = httpx.Request("POST", "https://api.tavily.com/search")
        post.return_value = httpx.Response(401, request=request, text="test-secret-key private response")
        result = self.execute("web_search", query="test", max_results=1)
        self.assertEqual(result, {
            "ok": False, "error": "Search service returned HTTP 401.",
            "code": "authentication_failed", "retryable": False, "attempts": 1,
        })
        self.assertEqual(post.call_count, 1)
        post.reset_mock()
        post.side_effect = httpx.ReadTimeout("test-secret-key")
        result = self.execute("web_search", query="test", max_results=1)
        self.assertEqual(result, {
            "ok": False, "error": "Search request timed out.",
            "code": "timeout", "retryable": True, "attempts": 2,
        })
        self.assertEqual(post.call_count, 2)

    @patch("goal_agent.tools.httpx.post")
    def test_malformed_search_response_is_handled(self, post) -> None:
        request = httpx.Request("POST", "https://api.tavily.com/search")
        for body in [[], {}, {"results": "wrong"}]:
            post.return_value = httpx.Response(200, request=request, json=body)
            self.assertFalse(self.execute("web_search", query="test", max_results=1)["ok"])
        post.return_value = httpx.Response(200, request=request, text="private non-JSON body")
        self.assertFalse(self.execute("web_search", query="test", max_results=1)["ok"])

    @patch("goal_agent.tools.httpx.post")
    def test_invalid_search_entries_are_ignored(self, post) -> None:
        request = httpx.Request("POST", "https://api.tavily.com/search")
        post.return_value = httpx.Response(200, request=request, json={"results": [
            None, {}, {"url": "file:///etc/passwd"}, {"url": "https://[malformed"},
            {"title": None, "url": "https://example.com", "content": None},
        ]})
        result = self.execute("web_search", query="test", max_results=5)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "no_results")
        self.assertFalse(result["retryable"])

    def test_read_file_returns_matching_content_and_hash(self) -> None:
        created = self.execute("create_file", path="notes/report.md", content="caf\u00e9\n")
        result = self.execute("read_file", path="notes\\report.md")
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], {
            "path": "notes/report.md", "content": "caf\u00e9\n", "bytes_read": 6,
            "sha256": created["result"]["sha256"],
        })

    def test_read_file_rejects_missing_invalid_oversized_and_escaping_files(self) -> None:
        (self.workspace / "binary.txt").write_bytes(b"\xff")
        (self.workspace / "huge.txt").write_bytes(b"a" * 101)
        (self.workspace / "directory").mkdir()
        for path, code in [
            ("missing.txt", "file_not_found"), ("binary.txt", "invalid_file"),
            ("huge.txt", "file_too_large"), ("directory", "invalid_file"),
            ("../outside.txt", "invalid_arguments"), ("C:\\outside.txt", "invalid_arguments"),
        ]:
            with self.subTest(path=path):
                result = self.execute("read_file", path=path)
                self.assertFalse(result["ok"])
                self.assertEqual(result["code"], code)

    def test_read_file_rejects_file_and_directory_symlinks(self) -> None:
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (outside / "private.txt").write_text("private", encoding="utf-8")
        try:
            (self.workspace / "link").symlink_to(outside, target_is_directory=True)
            (self.workspace / "link.txt").symlink_to(outside / "private.txt")
        except (OSError, NotImplementedError):
            self.skipTest("This OS/user cannot create symbolic links.")
        for path in ("link/private.txt", "link.txt"):
            result = self.execute("read_file", path=path)
            self.assertFalse(result["ok"])
            self.assertIn("Symbolic", result["error"])

    @patch("goal_agent.tools.httpx.post")
    def test_cancel_and_expired_deadline_prevent_tools(self, post) -> None:
        self.tools.configure_runtime(cancel=lambda: True)
        result = self.execute("create_file", path="cancelled.txt", content="not written")
        self.assertEqual(result["code"], "cancelled")
        self.assertFalse((self.workspace / "cancelled.txt").exists())
        self.tools.configure_runtime(deadline=0.0)
        result = self.execute("web_search", query="test", max_results=1)
        self.assertEqual(result["code"], "deadline_exceeded")
        post.assert_not_called()

    @patch("goal_agent.tools.time.monotonic", return_value=100.0)
    @patch("goal_agent.tools.httpx.post")
    def test_request_timeout_is_capped_by_remaining_budget(self, post, monotonic) -> None:
        request = httpx.Request("POST", "https://api.tavily.com/search")
        post.return_value = httpx.Response(200, request=request, json={"results": [
            {"url": "https://example.com", "content": "evidence"},
        ]})
        self.tools.configure_runtime(deadline=102.0)
        self.assertTrue(self.execute("web_search", query="test", max_results=1)["ok"])
        self.assertEqual(post.call_args.kwargs["timeout"].read, 2.0)
        self.assertEqual(post.call_args.kwargs["timeout"].connect, 2.0)

    @patch("goal_agent.tools.time.sleep")
    @patch("goal_agent.tools.httpx.post")
    def test_transient_failure_retries_but_cancel_stops_backoff(self, post, sleep) -> None:
        request = httpx.Request("POST", "https://api.tavily.com/search")
        post.side_effect = [httpx.Response(429, request=request), httpx.Response(
            200, request=request, json={"results": [{"url": "https://example.com", "content": "evidence"}]},
        )]
        result = self.execute("web_search", query="test", max_results=1)
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["attempts"], 2)
        self.assertEqual(post.call_count, 2)
        post.reset_mock()
        post.side_effect = httpx.ReadTimeout("private")
        cancelled = False

        def cancel_during_backoff(_seconds):
            nonlocal cancelled
            cancelled = True

        sleep.side_effect = cancel_during_backoff
        self.tools.configure_runtime(cancel=lambda: cancelled)
        result = self.execute("web_search", query="test", max_results=1)
        self.assertEqual(result["code"], "cancelled")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
