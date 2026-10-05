"""Independent evaluation oracles and deterministic production-loop fixtures."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goal_agent.evals import grade_case, load_cases, run_evaluations


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_offline_suite_runs_real_files_without_keys_or_network(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "httpx.Client.send", side_effect=AssertionError("Unexpected network call")
        ), patch("goal_agent.evals.OpenAI", side_effect=AssertionError("Unexpected model client")):
            report = run_evaluations(self.folder)
        self.assertEqual(report["mode"], "offline")
        self.assertGreaterEqual(report["total"], 15)
        self.assertEqual(report["failed"], 0, json.dumps([
            {"id": case["id"], "checks": case["checks"], "status": case["actual_status"],
             "error": case["error"]} for case in report["cases"] if not case["passed"]
        ], indent=2))
        self.assertEqual(report["passed"], report["total"])
        self.assertEqual(report["metrics"]["total_tokens"], 0)
        self.assertEqual(report["metrics"]["tool_policy_violations"], 0)
        self.assertGreater(report["metrics"]["tool_policy_attempts_rejected"], 0)
        by_id = {case["id"]: case for case in report["cases"]}
        self.assertEqual(by_id["missing_search_key"]["actual_status"], "failed")
        self.assertEqual(by_id["unsupported_goal"]["actual_status"], "blocked")
        self.assertFalse(by_id["fabricated_citation"]["artifact_outcome_valid"])
        self.assertFalse((Path(report["suite_path"]) / "path_traversal_rejected" / "escape.txt").exists())
        run_path = Path(by_id["cited_report"]["run_path"])
        self.assertTrue((run_path.parent / "artifacts" / "report.md").is_file())
        saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["passed"], report["passed"])
        self.assertEqual(len([case for case in load_cases() if case.get("live")]), 3)

    def test_reruns_preserve_previous_results(self):
        cases = self.folder / "small-cases.json"
        cases.write_text(json.dumps({"cases": [load_cases()[0]]}), encoding="utf-8")
        first = run_evaluations(self.folder, cases_path=cases)
        second = run_evaluations(self.folder, cases_path=cases)
        self.assertNotEqual(first["suite_path"], second["suite_path"])
        self.assertTrue(Path(first["report_path"]).is_file())
        self.assertEqual(first["failed"], 0)
        self.assertEqual(second["failed"], 0)

    def test_artifact_oracle_does_not_trust_runner_completion_claim(self):
        workspace = self.folder / "artifacts"
        workspace.mkdir()
        (workspace / "report.md").write_text("# Summary\nMissing content\n", encoding="utf-8")
        case = {
            "id": "oracle", "goal": "Create a cited report",
            "expected": {"status": "completed", "artifacts": [{
                "path": "report.md", "required_sections": ["Summary", "Sources"], "min_sources": 1,
            }]},
        }
        grade = grade_case(case, {"status": "completed", "verification": {"passed": True}}, workspace)
        self.assertFalse(grade["passed"])
        self.assertFalse(grade["checks"]["artifact_outcome"])

    def test_citation_oracle_distinguishes_retrieved_and_fabricated_urls(self):
        url = "https://docs.python.org/3/library/pathlib.html"
        workspace = self.folder / "artifacts"
        workspace.mkdir()
        artifact = workspace / "report.md"
        artifact.write_text(f"# Summary\n[Source]({url})\n", encoding="utf-8")
        case = {"id": "oracle", "goal": "Write research", "expected": {
            "status": "completed", "artifacts": [{"path": "report.md", "min_sources": 1}],
        }}
        state = {"status": "completed", "steps": [{"tool": "web_search", "events": [{
            "tool": "web_search", "ok": True, "result": {"sources": [{"url": url}]},
        }]}]}
        good = grade_case(case, state, workspace)
        self.assertTrue(good["passed"])
        self.assertEqual(good["citations_with_retrieved_provenance"], 1)
        artifact.write_text(artifact.read_text(encoding="utf-8") +
                            "[Invented](https://fabricated.invalid/source)\n", encoding="utf-8")
        bad = grade_case(case, state, workspace)
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["citations_checked"], 2)
        self.assertEqual(bad["citations_with_retrieved_provenance"], 1)

    def test_live_mode_requires_both_keys_before_creating_run_or_client(self):
        for environment, missing in (({}, "OPENAI_API_KEY"),
                                     ({"OPENAI_API_KEY": "placeholder"}, "TAVILY_API_KEY")):
            with self.subTest(missing=missing), patch.dict(os.environ, environment, clear=True), patch(
                "goal_agent.evals.OpenAI", side_effect=AssertionError("Created live client too early")
            ):
                with self.assertRaisesRegex(ValueError, missing):
                    run_evaluations(self.folder / "live", live=True)
        self.assertFalse((self.folder / "live").exists())

    def test_fixture_ids_cannot_escape_suite_directory(self):
        case = load_cases()[0]
        case["id"] = "../outside"
        path = self.folder / "unsafe.json"
        path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identifier"):
            run_evaluations(self.folder, cases_path=path)


if __name__ == "__main__":
    unittest.main()
