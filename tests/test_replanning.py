"""Bounded plan repair preserves completed work and output requirements."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path

from goal_agent.models import ArtifactExpectation, Limits, Plan, Step
from goal_agent.runner import Agent
from goal_agent.tools import ToolRegistry


def call(name: str, arguments: dict, call_id: str) -> list[dict]:
    return [{"type": "function_call", "name": name, "arguments": json.dumps(arguments),
             "call_id": call_id, "id": f"fc_{call_id}", "status": "completed"}]


def finish(call_id: str, *, failed: bool = False) -> list[dict]:
    return call("finish_step", {"status": "failed" if failed else "completed",
                               "summary": "Step blocked." if failed else "Step complete."}, call_id)


class RepairBackend:
    model = "offline-replanning-fixture"

    def __init__(self, plan: Plan, responses: list[list[dict]], repairs: list[Plan]) -> None:
        self.initial = plan
        self.responses = deque(responses)
        self.repairs = deque(repairs)
        self.repair_calls = []
        self.histories = []

    def plan(self, goal: str, max_steps: int, tools: list[dict]) -> Plan:
        return self.initial

    def respond(self, history: list[dict], tools: list[dict]) -> list[dict]:
        self.histories.append(copy.deepcopy(history))
        if not self.responses:
            raise AssertionError("Unexpected execution model request")
        return self.responses.popleft()

    def repair(self, goal: str, plan: Plan, state: dict) -> Plan:
        self.repair_calls.append({"goal": goal, "plan": plan.model_dump(), "state": copy.deepcopy(state)})
        if not self.repairs:
            raise AssertionError("Unexpected repair request")
        return self.repairs.popleft()


class ResearchTools(ToolRegistry):
    """Real file tools; deterministic research results never reach the network."""

    def __init__(self, workspace: Path, search_results: list[dict] | None = None) -> None:
        super().__init__(workspace, None)
        self.search_results = deque(search_results or [])
        self.writes = []
        self.searches = []

    def execute(self, name: str, arguments_json: str) -> dict:
        if name == "web_search" and self.search_results:
            self.searches.append(json.loads(arguments_json))
            return self.search_results.popleft()
        if name == "create_file":
            self.writes.append(json.loads(arguments_json)["path"])
        return super().execute(name, arguments_json)


class ReplanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "run"
        self.tools = ResearchTools(self.run_dir / "artifacts")

    def execute(self, backend: RepairBackend, **limits) -> dict:
        return Agent(backend, self.tools, self.run_dir, limits=Limits(**limits)).run("Complete the requested work")

    def test_repair_replaces_only_remaining_steps_and_keeps_completed_file(self) -> None:
        expected = [ArtifactExpectation(path="first.txt"), ArtifactExpectation(path="second.txt")]
        plan = Plan(steps=[Step(description="Save first file", tool="create_file"),
                           Step(description="Determine next action", tool="reason")], expected_artifacts=expected)
        repaired = Plan(steps=[Step(description="Save second file", tool="create_file")], expected_artifacts=expected)
        backend = RepairBackend(plan, [
            call("create_file", {"path": "first.txt", "content": "First."}, "first"), finish("first_done"),
            finish("blocked", failed=True),
            call("create_file", {"path": "second.txt", "content": "Second."}, "second"), finish("second_done"),
        ], [repaired])
        state = self.execute(backend)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(self.tools.writes, ["first.txt", "second.txt"])
        self.assertEqual(state["replans"], 1)
        self.assertEqual(state["tool_calls"], 2)
        self.assertEqual([s["number"] for s in state["steps"]], [1, 2])
        self.assertEqual(state["steps"][0]["status"], "completed")
        self.assertEqual(state["steps"][1]["description"], "Save second file")
        failed_snapshot = state["recovery_history"][0]["replaced_steps"][0]
        self.assertEqual(failed_snapshot["status"], "failed")
        self.assertEqual([e["call_id"] for e in failed_snapshot["events"]], ["blocked"])
        self.assertEqual(len(backend.repair_calls), 1)
        self.assertEqual(backend.repair_calls[0]["goal"], "Complete the requested work")
        self.assertEqual(backend.repair_calls[0]["plan"], plan.model_dump())
        self.assertEqual(backend.repair_calls[0]["state"]["steps"][0]["status"], "completed")
        self.assertIn("recovery_plan", json.dumps(backend.histories[-1]))
        self.assertEqual((self.tools.workspace / "first.txt").read_text(encoding="utf-8"), "First.")

    def test_failed_research_can_repair_and_finish_with_real_evidence(self) -> None:
        plan = Plan(steps=[Step(description="Research original query", tool="web_search")], minimum_sources=1)
        repaired = Plan(steps=[Step(description="Research alternate query", tool="web_search")], minimum_sources=1)
        self.tools.search_results = deque([
            {"ok": False, "error": "Temporary search failure.", "code": "service_error", "retryable": True, "attempts": 2},
            {"ok": True, "result": {"sources": [{"url": "https://example.com/evidence", "snippet": "Source evidence."}]}},
        ])
        backend = RepairBackend(plan, [
            call("web_search", {"query": "original", "max_results": 1}, "original"), finish("blocked", failed=True),
            call("web_search", {"query": "alternative", "max_results": 1}, "alternative"), finish("research_done"),
        ], [repaired])
        state = self.execute(backend)
        self.assertEqual(state["status"], "completed")
        self.assertEqual([q["query"] for q in self.tools.searches], ["original", "alternative"])
        self.assertEqual(state["verification"]["source_urls"], ["https://example.com/evidence"])
        self.assertEqual(state["replans"], 1)

    def test_default_repair_budget_allows_only_one_repair(self) -> None:
        plan = Plan(steps=[Step(description="Find a feasible approach", tool="reason")])
        backend = RepairBackend(plan, [finish("first_failure", failed=True), finish("second_failure", failed=True)], [plan])
        state = self.execute(backend)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["replans"], 1)
        self.assertEqual(len(backend.repair_calls), 1)
        self.assertEqual(len(backend.histories), 2)

    def test_zero_repair_budget_stops_without_a_repair_request(self) -> None:
        plan = Plan(steps=[Step(description="Find a feasible approach", tool="reason")])
        backend = RepairBackend(plan, [finish("blocked", failed=True)], [])
        state = self.execute(backend, max_replans=0)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["replans"], 0)
        self.assertEqual(backend.repair_calls, [])

    def test_repair_cannot_change_expected_artifacts(self) -> None:
        plan = Plan(steps=[Step(description="Choose report structure", tool="reason")],
                    expected_artifacts=[ArtifactExpectation(path="report.md", required_sections=["Findings"])])
        repaired = Plan(steps=[Step(description="Skip report", tool="reason")], expected_artifacts=[])
        backend = RepairBackend(plan, [finish("blocked", failed=True)], [repaired])
        state = self.execute(backend)
        self.assertEqual(state["status"], "failed")
        self.assertIn("changed output requirements", state["error"])
        self.assertEqual(len(backend.histories), 1)
        self.assertEqual(state["plan"]["expected_artifacts"], plan.model_dump()["expected_artifacts"])

    def test_repair_cannot_lower_minimum_sources(self) -> None:
        plan = Plan(steps=[Step(description="Assess needed evidence", tool="reason")], minimum_sources=2)
        repaired = Plan(steps=[Step(description="Skip research", tool="reason")], minimum_sources=0)
        backend = RepairBackend(plan, [finish("blocked", failed=True)], [repaired])
        state = self.execute(backend)
        self.assertEqual(state["status"], "failed")
        self.assertIn("changed output requirements", state["error"])
        self.assertEqual(len(backend.histories), 1)

    def test_repair_cannot_exceed_total_step_limit(self) -> None:
        plan = Plan(steps=[Step(description="Assess request", tool="reason")])
        repaired = Plan(steps=[Step(description="First replacement", tool="reason"),
                               Step(description="Second replacement", tool="reason")])
        backend = RepairBackend(plan, [finish("blocked", failed=True)], [repaired])
        state = self.execute(backend, max_steps=1)
        self.assertEqual(state["status"], "failed")
        self.assertIn("step limit", state["error"])
        self.assertEqual(len(backend.histories), 1)

    def test_missing_credentials_do_not_trigger_repair(self) -> None:
        plan = Plan(steps=[Step(description="Research example", tool="web_search")])
        backend = RepairBackend(plan, [call("web_search", {"query": "example", "max_results": 1}, "search")], [])
        state = self.execute(backend)
        self.assertEqual(state["status"], "failed")
        self.assertIn("TAVILY_API_KEY", state["error"])
        self.assertEqual(backend.repair_calls, [])
        self.assertEqual(state["replans"], 0)

    def test_provider_authentication_failure_does_not_trigger_repair(self) -> None:
        plan = Plan(steps=[Step(description="Research example", tool="web_search")])
        self.tools.search_results.append({"ok": False, "error": "Search service returned HTTP 401.",
                                         "code": "authentication_failed", "retryable": False, "attempts": 1})
        backend = RepairBackend(plan, [call("web_search", {"query": "example", "max_results": 1}, "search")], [])
        state = self.execute(backend)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(backend.repair_calls, [])
        self.assertEqual(len(backend.histories), 1)
        self.assertEqual(state["replans"], 0)

    def test_repair_cannot_introduce_file_creation_without_expected_artifacts(self) -> None:
        plan = Plan(steps=[Step(description="Assess request", tool="reason")])
        repaired = Plan(steps=[Step(description="Write an undeclared file", tool="create_file")])
        backend = RepairBackend(plan, [finish("blocked", failed=True)], [repaired])
        state = self.execute(backend)
        self.assertEqual(state["status"], "failed")
        self.assertIn("expected_artifacts", state["error"])
        self.assertEqual(len(backend.histories), 1)
        self.assertEqual(self.tools.writes, [])
        self.assertEqual(state["steps"][0]["tool"], "reason")

    def recovered_research_report(self, content: str) -> dict:
        expected = [ArtifactExpectation(path="report.md")]
        plan = Plan(steps=[Step(description="Prepare report from existing knowledge", tool="reason")],
                    expected_artifacts=expected, minimum_sources=0)
        repaired = Plan(steps=[Step(description="Find fresh evidence", tool="web_search"),
                               Step(description="Save report", tool="create_file")],
                        expected_artifacts=expected, minimum_sources=0)
        self.tools.search_results.append({"ok": True, "result": {"sources": [
            {"url": "https://example.com/retrieved", "snippet": "Actually retrieved evidence."},
        ]}})
        backend = RepairBackend(plan, [
            finish("need_research", failed=True),
            call("web_search", {"query": "fresh evidence", "max_results": 1}, "search"), finish("searched"),
            call("create_file", {"path": "report.md", "content": content}, "write"), finish("written"),
        ], [repaired])
        state = self.execute(backend)
        self.assertEqual(state["plan"]["minimum_sources"], 0)
        self.assertEqual(state["plan"]["steps"][0]["tool"], "reason")
        self.assertEqual(state["replans"], 1)
        return state

    def test_research_introduced_by_repair_rejects_fabricated_only_citation(self) -> None:
        state = self.recovered_research_report("# Report\nUnsupported claim: https://example.com/fabricated")
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stop_reason"], "verification_failed")
        checks = {check["name"]: check["passed"] for check in state["verification"]["checks"]}
        self.assertTrue(checks["research_evidence"])
        self.assertFalse(checks["citations:report.md"])
        self.assertFalse(checks["citation_provenance:report.md"])

    def test_research_introduced_by_repair_requires_a_citation(self) -> None:
        state = self.recovered_research_report("# Report\nA research report without any citations.")
        self.assertEqual(state["status"], "failed")
        checks = {check["name"]: check["passed"] for check in state["verification"]["checks"]}
        self.assertFalse(checks["citations:report.md"])

    def test_research_introduced_by_repair_accepts_retrieved_citation(self) -> None:
        state = self.recovered_research_report("# Report\nEvidence: https://example.com/retrieved")
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["verification"]["passed"])
        self.assertEqual(state["verification"]["source_urls"], ["https://example.com/retrieved"])

    def test_research_retained_only_in_event_history_still_requires_citation_provenance(self) -> None:
        expected = [ArtifactExpectation(path="report.md")]
        plan = Plan(steps=[Step(description="Prepare report", tool="reason")], expected_artifacts=expected)
        research_repair = Plan(steps=[Step(description="Find evidence", tool="web_search")], expected_artifacts=expected)
        writing_repair = Plan(steps=[Step(description="Save report", tool="create_file")], expected_artifacts=expected)
        self.tools.search_results.append({"ok": True, "result": {"sources": [
            {"url": "https://example.com/retrieved", "snippet": "Actually retrieved evidence."},
        ]}})
        backend = RepairBackend(plan, [
            finish("need_research", failed=True),
            call("web_search", {"query": "fresh evidence", "max_results": 1}, "search"),
            finish("need_file", failed=True),
            call("create_file", {"path": "report.md", "content": "Claim: https://example.com/fabricated"}, "write"),
            finish("written"),
        ], [research_repair, writing_repair])
        state = self.execute(backend, max_replans=2)
        self.assertEqual(state["replans"], 2)
        self.assertEqual([step["tool"] for step in state["steps"]], ["create_file"])
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stop_reason"], "verification_failed")
        checks = {check["name"]: check["passed"] for check in state["verification"]["checks"]}
        self.assertTrue(checks["research_evidence"])
        self.assertFalse(checks["citations:report.md"])
        self.assertFalse(checks["citation_provenance:report.md"])

    def test_repair_preflight_estimates_full_goal_plan_and_step_payload(self) -> None:
        plan = Plan(steps=[Step(description="Assess request", tool="reason")])
        repaired = Plan(steps=[Step(description="Complete revised approach", tool="reason")])
        backend = RepairBackend(plan, [finish("blocked", failed=True), finish("done")], [repaired])
        estimated_payloads = []

        def estimate_request(payload):
            estimated_payloads.append(copy.deepcopy(payload))
            return (0, 0)

        backend.estimate_request = estimate_request
        state = self.execute(backend)
        self.assertEqual(state["status"], "completed")
        repair_payloads = [payload for payload in estimated_payloads
                           if isinstance(payload, dict) and set(payload) == {"goal", "plan", "steps"}]
        self.assertEqual(len(repair_payloads), 1)
        repair_call = backend.repair_calls[0]
        self.assertEqual(repair_payloads[0], {
            "goal": repair_call["goal"], "plan": repair_call["plan"],
            "steps": repair_call["state"]["steps"],
        })


if __name__ == "__main__":
    unittest.main()
