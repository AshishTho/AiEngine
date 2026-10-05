"""Offline tests of planning, sequential execution, and bounded tool use."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path

from goal_agent.models import ArtifactExpectation, Limits, Plan, Step
from goal_agent.runner import Agent
from goal_agent.tools import ToolError, ToolRegistry


def function_call(name: str, arguments: dict | str, call_id: str) -> dict:
    return {
        "type": "function_call",
        "name": name,
        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
        "call_id": call_id,
        "id": f"fc_{call_id}",
        "status": "completed",
    }


def finish(call_id: str, status: str = "completed", summary: str = "Done") -> dict:
    return function_call("finish_step", {"status": status, "summary": summary}, call_id)


class ScriptedBackend:
    """Return predetermined responses and retain what the runner sent."""

    def __init__(self, plan: Plan, responses: list[list[dict]]) -> None:
        self.result = plan
        self.responses = deque(responses)
        self.plan_requests: list[dict] = []
        self.histories: list[list[dict]] = []

    def plan(self, goal: str, max_steps: int, tools: list[dict]) -> Plan:
        self.plan_requests.append({"goal": goal, "max_steps": max_steps, "tools": tools})
        return self.result

    def respond(self, history: list[dict], tools: list[dict]) -> list[dict]:
        self.histories.append(copy.deepcopy(history))
        if not self.responses:
            raise AssertionError("Agent requested an unexpected extra response")
        return self.responses.popleft()


class FakeTools:
    """A deterministic registry with the production JSON-in/result-out contract."""

    schemas = [
        {
            "type": "function",
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "create_file",
            "description": "Create a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function", "name": "read_file", "description": "Read a created file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"], "additionalProperties": False},
        },
    ]

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.files: dict[str, str] = {}

    def execute(self, name: str, args_json: str) -> dict:
        if name not in {"web_search", "create_file", "read_file"}:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        try:
            arguments = json.loads(args_json)
        except (ValueError, TypeError):
            return {"ok": False, "error": "Invalid JSON arguments"}
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "Arguments must be an object"}
        self.calls.append((name, arguments))
        if name == "web_search":
            if arguments.get("query") == "fail":
                return {"ok": False, "error": "Search service temporarily unavailable"}
            return {
                "ok": True,
                "result": {
                    "query": arguments.get("query"),
                    "sources": [{"title": "Example source", "url": "https://example.com/source", "snippet": "Source evidence: blue widgets."}],
                },
            }
        try:
            path = "/".join(ToolRegistry._path_parts(arguments.get("path")))
        except ToolError as exc:
            return {"ok": False, "error": str(exc)}
        if name == "create_file":
            if path in self.files:
                return {"ok": False, "error": "File already exists.", "code": "file_exists"}
            content = arguments.get("content")
            if not isinstance(content, str):
                return {"ok": False, "error": "content must be a string."}
            self.files[path] = content
            encoded = content.encode("utf-8")
            return {"ok": True, "result": {"path": path, "bytes_written": len(encoded),
                                             "sha256": hashlib.sha256(encoded).hexdigest()}}
        if path not in self.files:
            return {"ok": False, "error": "File does not exist.", "code": "file_not_found"}
        encoded = self.files[path].encode("utf-8")
        return {"ok": True, "result": {"path": path, "content": self.files[path],
                                         "bytes_read": len(encoded),
                                         "sha256": hashlib.sha256(encoded).hexdigest()}}


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "run"
        self.tools = FakeTools()

    def agent(self, plan: Plan, responses: list[list[dict]], **limits: int) -> tuple[Agent, ScriptedBackend]:
        backend = ScriptedBackend(plan, responses)
        agent = Agent(backend=backend, tools=self.tools, run_dir=self.run_dir, limits=Limits(**limits))
        return agent, backend

    def test_search_then_file_preserves_tool_results_in_context(self) -> None:
        plan = Plan(
            steps=[Step(description="Research widgets", tool="web_search"),
                   Step(description="Write a report", tool="create_file")],
            expected_artifacts=[ArtifactExpectation(path="report.md", required_sections=["Summary"], min_sources=1)],
            minimum_sources=1,
        )
        content = "# Summary\n\nBlue widgets: https://example.com/source"
        agent, backend = self.agent(plan, [
            [function_call("web_search", {"query": "widgets"}, "search")],
            [finish("research_done", summary="Found blue widgets")],
            [function_call("create_file", {"path": "report.md", "content": content}, "write")],
            [finish("write_done")],
        ])

        state = agent.run("Research widgets and write a report")

        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["tool_calls"], 2)
        self.assertEqual([name for name, _ in self.tools.calls],
                         ["web_search", "read_file", "create_file", "read_file"])
        self.assertEqual(self.tools.files["report.md"], content)
        self.assertTrue(state["verification"]["passed"])
        self.assertTrue(all(check["passed"] for check in state["verification"]["checks"]))
        self.assertEqual(state["verification"]["source_urls"], ["https://example.com/source"])
        self.assertEqual(state["verification"]["artifacts"][0]["sha256"],
                         hashlib.sha256(content.encode("utf-8")).hexdigest())
        self.assertEqual(len(backend.histories), 4)
        self.assertIn("Source evidence: blue widgets.", json.dumps(backend.histories[2]))
        self.assertIn("https://example.com/source", json.dumps(backend.histories[2]))
        saved = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(len(saved["steps"]), 2)
        self.assertEqual([step["status"] for step in saved["steps"]], ["completed", "completed"])

    def test_plan_only_does_not_execute_or_request_execution(self) -> None:
        plan = Plan(steps=[Step(description="Research widgets", tool="web_search")])
        agent, backend = self.agent(plan, [])

        state = agent.run("Research widgets", plan_only=True)

        self.assertEqual(state["status"], "planned")
        self.assertEqual(self.tools.calls, [])
        self.assertEqual(backend.histories, [])
        self.assertEqual(len(backend.plan_requests), 1)
        self.assertTrue((self.run_dir / "run.json").is_file())

    def test_tool_step_cannot_complete_before_required_tool_succeeds(self) -> None:
        plan = Plan(steps=[Step(description="Research widgets", tool="web_search")])
        agent, backend = self.agent(plan, [
            [finish("premature")],
            [function_call("web_search", {"query": "widgets"}, "search")],
            [finish("done")],
        ])

        state = agent.run("Research widgets")

        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(backend.histories), 3)
        self.assertEqual([name for name, _ in self.tools.calls], ["web_search"])

    def test_failed_tool_result_does_not_count_as_success(self) -> None:
        plan = Plan(steps=[Step(description="Research widgets", tool="web_search")])
        agent, backend = self.agent(plan, [
            [function_call("web_search", {"query": "fail"}, "bad_search")],
            [finish("premature")],
            [function_call("web_search", {"query": "widgets"}, "retry")],
            [finish("done")],
        ])

        state = agent.run("Research widgets")

        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(backend.histories), 4)
        self.assertEqual(len(self.tools.calls), 2)

    def test_explicit_failure_stops_dependent_steps(self) -> None:
        plan = Plan(
            steps=[Step(description="Assess the request", tool="reason"),
                   Step(description="Write a report", tool="create_file")],
            expected_artifacts=[ArtifactExpectation(path="report.md")],
        )
        agent, backend = self.agent(plan, [[finish("failed", status="failed", summary="Insufficient information")]])

        state = agent.run("Assess and write a report")

        self.assertEqual(state["status"], "failed")
        self.assertTrue(state.get("error"))
        self.assertEqual(len(backend.histories), 1)
        self.assertEqual(self.tools.calls, [])
        self.assertEqual(state["steps"][1]["status"], "pending")

    def test_turn_limit_stops_unproductive_execution(self) -> None:
        plan = Plan(steps=[Step(description="Think about widgets", tool="reason")])
        message = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Still thinking", "annotations": []}]}
        agent, backend = self.agent(plan, [[message]], max_turns_per_step=1)

        state = agent.run("Think about widgets")

        self.assertEqual(state["status"], "failed")
        self.assertTrue(state.get("error"))
        self.assertEqual(len(backend.histories), 1)

    def test_tool_budget_prevents_an_extra_execution(self) -> None:
        plan = Plan(steps=[Step(description="Research widgets", tool="web_search")])
        agent, _ = self.agent(plan, [
            [function_call("web_search", {"query": "first"}, "first")],
            [function_call("web_search", {"query": "second"}, "second")],
        ], max_tool_calls=1)

        state = agent.run("Research widgets")

        self.assertEqual(state["status"], "budget_exceeded")
        self.assertEqual(state["tool_calls"], 1)
        self.assertEqual(len(self.tools.calls), 1)
        self.assertTrue(state.get("error"))

    def test_unknown_and_malformed_calls_can_be_corrected(self) -> None:
        plan = Plan(steps=[Step(description="Research widgets", tool="web_search")])
        agent, backend = self.agent(plan, [
            [function_call("unknown_tool", {}, "unknown")],
            [function_call("web_search", "{invalid-json", "malformed")],
            [function_call("web_search", {"query": "widgets"}, "valid")],
            [finish("done")],
        ])

        state = agent.run("Research widgets")

        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["tool_calls"], 3)
        self.assertEqual(len(backend.histories), 4)
        self.assertEqual(len(self.tools.calls), 1)
        self.assertIn("malformed", json.dumps(backend.histories[2]))

    def test_multiple_calls_in_one_turn_stop_before_any_tool_execution(self) -> None:
        plan = Plan(steps=[Step(description="Research widgets", tool="web_search")])
        agent, _ = self.agent(plan, [[
            function_call("web_search", {"query": "first"}, "first"),
            function_call("web_search", {"query": "second"}, "second"),
        ]])

        state = agent.run("Research widgets")

        self.assertEqual(state["status"], "failed")
        self.assertEqual(self.tools.calls, [])
        self.assertTrue(state.get("error"))

    def test_tool_for_later_step_is_rejected(self) -> None:
        plan = Plan(steps=[Step(description="Research widgets", tool="web_search")])
        agent, _ = self.agent(plan, [
            [function_call("create_file", {"path": "report.md", "content": "Premature report"}, "disallowed")],
            [function_call("web_search", {"query": "widgets"}, "allowed")],
            [finish("done")],
        ])

        state = agent.run("Research widgets")

        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["tool_calls"], 2)
        self.assertEqual([name for name, _ in self.tools.calls], ["web_search"])

    def test_oversized_plan_is_rejected_before_execution(self) -> None:
        plan = Plan(steps=[Step(description="First step", tool="reason"), Step(description="Second step", tool="reason")])
        agent, backend = self.agent(plan, [], max_steps=1)

        state = agent.run("Think about widgets")

        self.assertEqual(state["status"], "failed")
        self.assertTrue(state.get("error"))
        self.assertEqual(backend.histories, [])
        self.assertEqual(self.tools.calls, [])

    def test_malformed_completion_can_be_corrected(self) -> None:
        plan = Plan(steps=[Step(description="Think about widgets", tool="reason")])
        agent, backend = self.agent(plan, [
            [function_call("finish_step", {"status": "completed"}, "missing_summary")],
            [finish("corrected")],
        ])

        state = agent.run("Think about widgets")

        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["tool_calls"], 0)
        self.assertEqual(len(backend.histories), 2)
        self.assertEqual(self.tools.calls, [])


if __name__ == "__main__":
    unittest.main()
