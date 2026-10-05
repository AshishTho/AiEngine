"""Tool boundary tests; no credentials or network access required."""

from __future__ import annotations

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
        self.assertEqual(result, {"ok": True, "result": {"path": "notes/report.txt", "bytes_written": 6}})
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
        self.assertEqual(result, {"ok": False, "error": "Search service returned HTTP 401."})
        post.side_effect = httpx.ReadTimeout("test-secret-key")
        result = self.execute("web_search", query="test", max_results=1)
        self.assertEqual(result, {"ok": False, "error": "Search request timed out."})

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
        self.assertEqual(result["result"]["sources"], [{"title": "", "url": "https://example.com", "snippet": ""}])


if __name__ == "__main__":
    unittest.main()
