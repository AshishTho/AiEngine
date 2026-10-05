"""Offline recovery and run-budget checks against real workspace file tools."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

from goal_agent.models import AgentError, ArtifactExpectation, Limits, Plan, Pricing, Step
from goal_agent.runner import Agent
from goal_agent.storage import run_lock
from goal_agent.tools import ToolRegistry


def call(name: str, arguments: dict, call_id: str) -> dict:
    return {"type": "function_call", "name": name,
            "arguments": json.dumps(arguments), "call_id": call_id,
            "id": f"fc_{call_id}", "status": "completed"}


def write(path: str, content: str, call_id: str = "write") -> list[dict]:
    return [call("create_file", {"path": path, "content": content}, call_id)]


def finish(call_id: str = "done") -> list[dict]:
    return [call("finish_step", {"status": "completed", "summary": "Saved requested file."}, call_id)]


def file_plan(*paths: str) -> Plan:
    return Plan(
        steps=[Step(description=f"Save {path}", tool="create_file") for path in paths],
        expected_artifacts=[ArtifactExpectation(path=path) for path in paths],
    )


class RecoveryBackend:
    model = "offline-recovery-fixture"

    def __init__(self, plan: Plan, responses: list, *, usage: int = 0) -> None:
        self.result = plan
        self.responses = deque(responses)
        self.plan_calls = 0
        self.histories = []
        self.usage = usage
        self.last_usage = None

    def configure_runtime(self, deadline=None, cancel=None) -> None:
        self.deadline, self.cancel = deadline, cancel

    def estimate_request(self, payload) -> tuple[int, int]:
        return (0, 0)

    def record_usage(self) -> None:
        self.last_usage = {"input_tokens": self.usage, "output_tokens": 0, "total_tokens": self.usage}

    def plan(self, goal: str, max_steps: int, tools: list[dict]) -> Plan:
        self.plan_calls += 1
        self.record_usage()
        return self.result

    def respond(self, history: list[dict], tools: list[dict]) -> list[dict]:
        self.histories.append(copy.deepcopy(history))
        self.record_usage()
        if not self.responses:
            raise AssertionError("Unexpected model response request")
        output = self.responses.popleft()
        if isinstance(output, BaseException):
            raise output
        return output


class RecordingTools(ToolRegistry):
    def __init__(self, workspace: Path, *, interrupt_after_create: bool = False) -> None:
        super().__init__(workspace, None)
        self.calls = []
        self.interrupt_after_create = interrupt_after_create

    def execute(self, name: str, arguments_json: str) -> dict:
        self.calls.append((name, arguments_json))
        result = super().execute(name, arguments_json)
        if name == "create_file" and result.get("ok") and self.interrupt_after_create:
            self.interrupt_after_create = False
            raise KeyboardInterrupt()
        return result


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "run"
        self.workspace = self.run_dir / "artifacts"
        self.tools = RecordingTools(self.workspace)

    def agent(self, backend: RecoveryBackend, **kwargs) -> Agent:
        return Agent(backend, self.tools, self.run_dir, **kwargs)

    def saved(self) -> dict:
        return json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))

    def test_saved_plan_executes_without_planning_again(self) -> None:
        plan = file_plan("report.md")
        backend = RecoveryBackend(plan, [write("report.md", "# Report\nDone."), finish()])
        state = self.agent(backend).run("Save report.md", plan=plan)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(backend.plan_calls, 0)
        self.assertEqual(state["plan"], plan.model_dump())
        self.assertEqual((self.workspace / "report.md").read_text(encoding="utf-8"), "# Report\nDone.")

    def test_resume_planned_run_uses_its_saved_plan(self) -> None:
        plan = file_plan("report.md")
        planning = RecoveryBackend(plan, [])
        state = self.agent(planning).run("Save report.md", plan_only=True)
        self.assertEqual(state["status"], "planned")
        execution = RecoveryBackend(plan, [write("report.md", "Saved once."), finish()])
        state = self.agent(execution).resume()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(execution.plan_calls, 0)
        self.assertEqual(state["tool_calls"], 1)

    def test_interrupt_immediately_after_write_recovers_without_rewriting(self) -> None:
        plan = file_plan("report.md")
        self.tools.interrupt_after_create = True
        backend = RecoveryBackend(plan, [write("report.md", "Persisted before interruption.")])
        state = self.agent(backend).run("Save report.md")
        self.assertEqual(state["status"], "interrupted")
        self.assertIsNotNone(self.saved()["pending_call"])
        original_bytes = (self.workspace / "report.md").read_bytes()
        resumed_backend = RecoveryBackend(plan, [finish()])
        state = self.agent(resumed_backend).resume()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["tool_calls"], 1)
        self.assertEqual([name for name, _ in self.tools.calls].count("create_file"), 1)
        self.assertEqual((self.workspace / "report.md").read_bytes(), original_bytes)
        events = [event for event in state["steps"][0]["events"] if event["tool"] == "create_file"]
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["result"]["sha256"], hashlib.sha256(original_bytes).hexdigest())
        outputs = [item for item in resumed_backend.histories[0]
                   if item.get("type") == "function_call_output" and item.get("call_id") == "write"]
        self.assertEqual(len(outputs), 1)
        self.assertIsNone(self.saved()["pending_call"])

    def test_completed_step_is_not_repeated_after_later_interruption(self) -> None:
        plan = file_plan("first.txt", "second.txt")
        initial = RecoveryBackend(plan, [write("first.txt", "First.", "first"), finish("first_done"), KeyboardInterrupt()])
        state = self.agent(initial).run("Save first.txt and second.txt")
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["steps"][0]["status"], "completed")
        initial_hash = hashlib.sha256((self.workspace / "first.txt").read_bytes()).hexdigest()
        resumed = RecoveryBackend(plan, [write("second.txt", "Second.", "second"), finish("second_done")])
        state = self.agent(resumed).resume()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["tool_calls"], 2)
        self.assertEqual([name for name, _ in self.tools.calls].count("create_file"), 2)
        self.assertEqual(hashlib.sha256((self.workspace / "first.txt").read_bytes()).hexdigest(), initial_hash)
        self.assertEqual(resumed.plan_calls, 0)

    def test_tampered_completed_artifact_blocks_resume(self) -> None:
        plan = file_plan("first.txt", "second.txt")
        initial = RecoveryBackend(plan, [write("first.txt", "Original.", "first"), finish("first_done"), KeyboardInterrupt()])
        state = self.agent(initial).run("Save first.txt and second.txt")
        self.assertEqual(state["status"], "interrupted")
        (self.workspace / "first.txt").write_text("Changed outside the run.", encoding="utf-8")
        resumed = RecoveryBackend(plan, [])
        try:
            state = self.agent(resumed).resume()
        except (AgentError, ValueError) as exc:
            self.assertTrue(str(exc))
        else:
            self.assertNotEqual(state["status"], "completed")
            self.assertTrue(state.get("error"))
        self.assertEqual(resumed.histories, [])
        self.assertEqual(resumed.plan_calls, 0)
        self.assertFalse((self.workspace / "second.txt").exists())
        self.assertEqual((self.workspace / "first.txt").read_text(encoding="utf-8"), "Changed outside the run.")

    def test_conflicting_pending_write_is_not_accepted_as_recovered(self) -> None:
        plan = file_plan("report.md")
        self.tools.interrupt_after_create = True
        initial = RecoveryBackend(plan, [write("report.md", "Expected bytes.")])
        state = self.agent(initial).run("Save report.md")
        self.assertEqual(state["status"], "interrupted")
        (self.workspace / "report.md").write_text("Different bytes.", encoding="utf-8")
        resumed = RecoveryBackend(plan, [])
        try:
            state = self.agent(resumed).resume()
        except (AgentError, ValueError) as exc:
            self.assertTrue(str(exc))
        else:
            self.assertNotEqual(state["status"], "completed")
            self.assertTrue(state.get("error"))
        self.assertEqual(resumed.histories, [])
        self.assertEqual((self.workspace / "report.md").read_text(encoding="utf-8"), "Different bytes.")

    def test_actual_token_usage_stops_before_file_execution(self) -> None:
        plan = file_plan("report.md")
        backend = RecoveryBackend(plan, [], usage=101)
        state = self.agent(backend, limits=Limits(max_tokens=100)).run("Save report.md")
        self.assertNotEqual(state["status"], "completed")
        self.assertTrue(state.get("error"))
        self.assertEqual(backend.plan_calls, 1)
        self.assertEqual(backend.histories, [])
        self.assertEqual(self.tools.calls, [])
        self.assertFalse((self.workspace / "report.md").exists())
        self.assertEqual(state["status"], "budget_exceeded")
        self.assertEqual(state["metrics"]["total_tokens"], 101)

    def test_token_reservation_prevents_billable_request(self) -> None:
        plan = file_plan("report.md")
        backend = RecoveryBackend(plan, [])
        backend.estimate_request = lambda payload: (60, 50)
        state = self.agent(backend, limits=Limits(max_tokens=100)).run("Save report.md")
        self.assertNotEqual(state["status"], "completed")
        self.assertEqual(backend.plan_calls, 0)
        self.assertEqual(backend.histories, [])
        self.assertEqual(self.tools.calls, [])

    def test_interrupted_pending_write_without_file_executes_once_on_resume(self) -> None:
        plan = file_plan("report.md")
        backend = RecoveryBackend(plan, [write("report.md", "Created on resume.")])
        original_execute = self.tools.execute

        def interrupt_before_write(name, arguments):
            if name == "create_file":
                raise KeyboardInterrupt()
            return original_execute(name, arguments)

        with patch.object(self.tools, "execute", side_effect=interrupt_before_write):
            state = self.agent(backend).run("Save report.md")
        self.assertEqual(state["status"], "interrupted")
        self.assertFalse((self.workspace / "report.md").exists())
        self.assertIsNotNone(self.saved()["pending_call"])
        resumed = RecoveryBackend(plan, [finish()])
        state = self.agent(resumed).resume()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["tool_calls"], 1)
        self.assertEqual([name for name, _ in self.tools.calls].count("create_file"), 1)
        self.assertEqual((self.workspace / "report.md").read_text(encoding="utf-8"), "Created on resume.")

    def test_cancel_before_planning_avoids_model_and_tools(self) -> None:
        backend = RecoveryBackend(file_plan("report.md"), [])
        state = self.agent(backend, cancel=lambda: True).run("Save report.md")
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["stop_reason"], "cancelled")
        self.assertEqual(backend.plan_calls, 0)
        self.assertEqual(self.tools.calls, [])

    def test_cancel_after_file_write_records_it_and_resume_does_not_repeat(self) -> None:
        plan = file_plan("report.md")
        backend = RecoveryBackend(plan, [write("report.md", "Saved before cancellation.")])
        cancelled = False
        original_execute = self.tools.execute

        def cancel_after_write(name, arguments):
            nonlocal cancelled
            result = original_execute(name, arguments)
            if name == "create_file" and result.get("ok"):
                cancelled = True
            return result

        with patch.object(self.tools, "execute", side_effect=cancel_after_write):
            state = self.agent(backend, cancel=lambda: cancelled).run("Save report.md")
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["tool_calls"], 1)
        self.assertIsNone(state["pending_call"])
        self.assertTrue(state["steps"][0]["events"][0]["ok"])
        state = self.agent(RecoveryBackend(plan, [finish()])).resume()
        self.assertEqual(state["status"], "completed")
        self.assertEqual([name for name, _ in self.tools.calls].count("create_file"), 1)

    def test_elapsed_time_limit_stops_after_slow_model_before_tools(self) -> None:
        plan = file_plan("report.md")
        backend = RecoveryBackend(plan, [])
        now = 100.0
        original_plan = backend.plan

        def slow_plan(*args):
            nonlocal now
            result = original_plan(*args)
            now += 2.0
            return result

        backend.plan = slow_plan
        with patch("goal_agent.runner.time.monotonic", side_effect=lambda: now):
            state = self.agent(backend, limits=Limits(max_seconds=1.0)).run("Save report.md")
        self.assertEqual(state["status"], "budget_exceeded")
        self.assertIn("Elapsed", state["error"])
        self.assertEqual(self.tools.calls, [])
        self.assertGreaterEqual(state["metrics"]["elapsed_seconds"], 2.0)

    def test_resume_retains_consumed_tool_budget(self) -> None:
        plan = file_plan("first.txt", "second.txt")
        backend = RecoveryBackend(plan, [write("first.txt", "First.", "first"), finish("first_done"), KeyboardInterrupt()])
        limits = Limits(max_tool_calls=1)
        state = self.agent(backend, limits=limits).run("Save first.txt and second.txt")
        self.assertEqual(state["status"], "interrupted")
        resumed = RecoveryBackend(plan, [write("second.txt", "Second.", "second")])
        state = self.agent(resumed, limits=limits).resume()
        self.assertEqual(state["status"], "budget_exceeded")
        self.assertEqual(state["tool_calls"], 1)
        self.assertFalse((self.workspace / "second.txt").exists())

    def test_cost_reservation_prevents_billable_request(self) -> None:
        backend = RecoveryBackend(file_plan("report.md"), [])
        backend.estimate_request = lambda payload: (1000, 1000)
        state = self.agent(backend, limits=Limits(max_cost_usd=0.001),
                           pricing=Pricing(input_per_million=1.0, output_per_million=2.0)).run("Save report.md")
        self.assertEqual(state["status"], "budget_exceeded")
        self.assertEqual(backend.plan_calls, 0)
        self.assertEqual(self.tools.calls, [])

    def test_resume_cancellation_is_not_reported_as_artifact_tampering(self) -> None:
        plan = file_plan("first.txt", "second.txt")
        backend = RecoveryBackend(plan, [write("first.txt", "Original.", "first"), finish("first_done"), KeyboardInterrupt()])
        state = self.agent(backend).run("Save first.txt and second.txt")
        self.assertEqual(state["status"], "interrupted")
        state = self.agent(RecoveryBackend(plan, []), cancel=lambda: True).resume()
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["stop_reason"], "cancelled")

    def test_observed_cost_limit_stops_when_reservation_underestimated_usage(self) -> None:
        plan = file_plan("report.md")
        backend = RecoveryBackend(plan, [], usage=2000)
        state = self.agent(backend, limits=Limits(max_cost_usd=0.001),
                           pricing=Pricing(input_per_million=1.0, output_per_million=2.0)).run("Save report.md")
        self.assertEqual(state["status"], "budget_exceeded")
        self.assertEqual(state["metrics"]["estimated_cost_usd"], 0.002)
        self.assertEqual(backend.histories, [])
        self.assertEqual(self.tools.calls, [])

    def test_final_response_cannot_exceed_cost_cap_and_claim_success(self) -> None:
        plan = file_plan("report.md")
        backend = RecoveryBackend(plan, [write("report.md", "Saved file."), finish()])
        requests = 0

        def usage_only_on_final_response():
            nonlocal requests
            requests += 1
            tokens = 2000 if requests == 3 else 0
            backend.last_usage = {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens}

        backend.record_usage = usage_only_on_final_response
        state = self.agent(backend, limits=Limits(max_cost_usd=0.001),
                           pricing=Pricing(input_per_million=1.0, output_per_million=2.0)).run("Save report.md")
        self.assertEqual(state["status"], "budget_exceeded")
        self.assertEqual(state["metrics"]["estimated_cost_usd"], 0.002)
        self.assertTrue((self.workspace / "report.md").exists())

    def test_resume_expired_budget_is_not_reported_as_artifact_tampering(self) -> None:
        plan = file_plan("first.txt", "second.txt")
        backend = RecoveryBackend(plan, [write("first.txt", "Original.", "first"), finish("first_done"), KeyboardInterrupt()])
        state = self.agent(backend).run("Save first.txt and second.txt")
        self.assertEqual(state["status"], "interrupted")
        state["metrics"]["elapsed_seconds"] = 5.0
        (self.run_dir / "run.json").write_text(json.dumps(state), encoding="utf-8")
        state = self.agent(RecoveryBackend(plan, []), limits=Limits(max_seconds=1.0)).resume()
        self.assertEqual(state["status"], "budget_exceeded")
        self.assertEqual(state["stop_reason"], "budget_exceeded")

    def test_execution_lock_blocks_concurrent_run_without_writing_checkpoint(self) -> None:
        backend = RecoveryBackend(file_plan("report.md"), [])
        with run_lock(self.run_dir):
            with self.assertRaisesRegex(AgentError, "already executing"):
                self.agent(backend).run("Save report.md")
        self.assertEqual(backend.plan_calls, 0)
        self.assertFalse((self.run_dir / "run.json").exists())
        # A released lock must permit the next run.
        state = self.agent(backend).run("Save report.md", plan_only=True)
        self.assertEqual(state["status"], "planned")

    def test_interrupted_request_reservation_is_charged_once_across_resumes(self) -> None:
        plan = file_plan("report.md")
        state = self.agent(RecoveryBackend(plan, [])).run("Save report.md", plan_only=True)
        state["pending_request"] = {"operation": "respond", "reserved_tokens": 101, "reserved_cost": 0.0}
        (self.run_dir / "run.json").write_text(json.dumps(state), encoding="utf-8")
        for _ in range(2):
            resumed = RecoveryBackend(plan, [])
            state = self.agent(resumed, limits=Limits(max_tokens=100)).resume()
            self.assertEqual(state["status"], "budget_exceeded")
            self.assertEqual(state["metrics"]["unconfirmed_reserved_tokens"], 101)
            self.assertFalse(state["metrics"]["usage_complete"])
            self.assertIsNone(state["pending_request"])
            self.assertEqual(resumed.histories, [])
        self.assertFalse((self.workspace / "report.md").exists())

    def test_missing_search_credentials_stop_without_an_extra_model_turn(self) -> None:
        plan = Plan(steps=[Step(description="Find source evidence", tool="web_search")], minimum_sources=1)
        backend = RecoveryBackend(plan, [[call("web_search", {"query": "example", "max_results": 1}, "search")]])
        state = self.agent(backend).run("Research an example")
        self.assertEqual(state["status"], "failed")
        self.assertIn("TAVILY_API_KEY", state["error"])
        self.assertEqual(len(backend.histories), 1)
        self.assertEqual(state["steps"][0]["events"][0]["code"], "missing_credentials")


if __name__ == "__main__":
    unittest.main()
