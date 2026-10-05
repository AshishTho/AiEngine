"""Public-page extraction and retry contract tests, with no live HTTP requests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from goal_agent.tools import ToolRegistry


class FetchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tools = ToolRegistry(Path(self.temp.name), "private-key")
        self.request = httpx.Request("POST", "https://api.tavily.com/extract")

    def fetch(self, url: object = "https://example.com/article") -> dict:
        return self.tools.execute("fetch_page", json.dumps({"url": url}))

    @patch("goal_agent.tools.httpx.post")
    def test_public_page_uses_fixed_provider_and_bounded_markdown(self, post) -> None:
        post.return_value = httpx.Response(200, request=self.request, json={"results": [{
            "url": "https://example.com/article", "raw_content": "# Evidence\n" + "a" * 21000,
        }]})
        result = self.fetch()
        self.assertTrue(result["ok"])
        page = result["result"]
        self.assertEqual(len(page["content"]), 20000)
        self.assertTrue(page["truncated"])
        self.assertEqual(page["sources"][0]["url"], "https://example.com/article")
        self.assertEqual(page["requested_url"], "https://example.com/article")
        self.assertEqual(post.call_args.args, ("https://api.tavily.com/extract",))
        arguments = post.call_args.kwargs
        self.assertEqual(arguments["headers"], {"Authorization": "Bearer private-key"})
        self.assertEqual(arguments["json"]["urls"], ["https://example.com/article"])
        self.assertEqual(arguments["json"]["format"], "markdown")
        self.assertFalse(arguments["follow_redirects"])

    @patch("goal_agent.tools.httpx.post")
    def test_private_malformed_or_credentialed_urls_never_reach_provider(self, post) -> None:
        urls = [
            "file:///etc/passwd", "ftp://example.com", "http://localhost", "http://localhost.",
            "http://service.local", "http://service.internal", "http://singlelabel", "http://127.0.0.1",
            "http://10.1.2.3", "https://192.168.1.1", "http://169.254.169.254/latest/meta-data",
            "http://172.16.0.1", "http://100.64.0.1", "http://[::1]", "http://[fc00::1]",
            "http://[fe80::1%25eth0]", "http://224.0.0.1", "http://0.0.0.0", "http://2130706433",
            "http://0177.0.0.1", "http://0x7f.0.0.1", "https://user:secret@example.com",
            "https://example.com\\@localhost", "https://[malformed", "https://example.com:bad",
            "https://example.com:0", "https://example.com\n", "https://%6cocalhost", "https://..",
            "https://bad_host.example", "https://example.com/\ud800", "", None, 42,
        ]
        for url in urls:
            with self.subTest(url=url):
                result = self.fetch(url)
                self.assertFalse(result["ok"])
                self.assertEqual(result["code"], "invalid_url")
        post.assert_not_called()

    @patch("goal_agent.tools.httpx.post")
    def test_empty_failed_or_private_extraction_is_not_success(self, post) -> None:
        for results in [[], [None], [{"url": "https://example.com", "raw_content": "  "}],
                        [{"url": "http://localhost", "raw_content": "private"}]]:
            post.return_value = httpx.Response(200, request=self.request, json={
                "results": results, "failed_results": [{"error": "private-key provider error"}],
            })
            result = self.fetch()
            self.assertFalse(result["ok"])
            self.assertEqual(result["code"], "no_results")
            self.assertFalse(result["retryable"])
            self.assertNotIn("private-key", str(result))

    @patch("goal_agent.tools.time.sleep")
    @patch("goal_agent.tools.httpx.post")
    def test_transient_errors_have_two_attempts_and_permanent_errors_one(self, post, sleep) -> None:
        for status in (400, 401, 403, 404, 429, 500, 503):
            with self.subTest(status=status):
                post.reset_mock()
                post.return_value = httpx.Response(status, request=self.request, text="private-key")
                result = self.fetch()
                transient = status == 429 or status >= 500
                self.assertFalse(result["ok"])
                self.assertEqual(result["retryable"], transient)
                self.assertEqual(result["attempts"], 2 if transient else 1)
                self.assertEqual(post.call_count, 2 if transient else 1)
                self.assertNotIn("private-key", str(result))

    @patch("goal_agent.tools.httpx.post")
    def test_missing_key_and_malformed_response_are_typed_errors(self, post) -> None:
        self.tools = ToolRegistry(Path(self.temp.name), None)
        self.assertEqual(self.fetch()["code"], "missing_credentials")
        post.assert_not_called()
        self.tools = ToolRegistry(Path(self.temp.name), "private-key")
        for body in [{}, [], {"results": 1}]:
            post.return_value = httpx.Response(200, request=self.request, json=body)
            self.assertEqual(self.fetch()["code"], "invalid_response")
        post.return_value = httpx.Response(200, request=self.request, text="private-key")
        self.assertEqual(self.fetch()["code"], "invalid_response")


if __name__ == "__main__":
    unittest.main()
