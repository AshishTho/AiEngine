"""CLI contracts exercise actual saved plans and file generation without keys."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goal_agent.cli import main


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main(args)
        return code, output.getvalue()

    def test_saved_demo_plan_executes_without_replanning_and_resumes(self):
        code, output = self.invoke(["--demo", "--plan-only", "--runs-dir", str(self.root / "plans")])
        self.assertEqual(code, 0, output)
        plan = next((self.root / "plans").glob("*/plan.json"))
        code, output = self.invoke(["--execute-plan", str(plan), "--runs-dir", str(self.root / "executed")])
        self.assertEqual(code, 0, output)
        record = next((self.root / "executed").glob("*/run.json"))
        state = json.loads(record.read_text())
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["verification"]["passed"])
        self.assertEqual(state["metrics"]["model_requests"], 4)
        code, output = self.invoke(["--resume", str(record)])
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(record.read_text())["tool_calls"], 2)

    def test_plan_only_run_can_be_resumed_in_place(self):
        code, _ = self.invoke(["--demo", "--plan-only", "--runs-dir", str(self.root)])
        self.assertEqual(code, 0)
        record = next(self.root.glob("*/run.json"))
        code, output = self.invoke(["--resume", str(record.parent)])
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(record.read_text())["status"], "completed")

    def test_missing_key_fails_without_network(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch("goal_agent.cli.load_dotenv"):
            code, output = self.invoke(["Create note.md", "--runs-dir", str(self.root)])
        self.assertEqual(code, 1)
        self.assertIn("OPENAI_API_KEY", output)
        self.assertFalse(list(self.root.glob("*/run.json")))

    def test_invalid_mode_combinations_are_rejected(self):
        for args in (["--live"], ["goal", "--demo"], ["--resume", "x", "--plan-only"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as caught:
                self.invoke(args)
            self.assertEqual(caught.exception.code, 2)

    def test_limits_are_preserved_and_can_be_explicitly_raised_on_resume(self):
        code, _ = self.invoke(["--demo", "--plan-only", "--runs-dir", str(self.root), "--max-tool-calls", "1"])
        self.assertEqual(code, 0)
        record = next(self.root.glob("*/run.json"))
        code, _ = self.invoke(["--resume", str(record)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(record.read_text())["status"], "budget_exceeded")
        code, output = self.invoke(["--resume", str(record), "--max-tool-calls", "3"])
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(record.read_text())["tool_calls"], 2)


if __name__ == "__main__":
    unittest.main()
