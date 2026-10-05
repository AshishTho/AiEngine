"""Verified, resumable sequential execution with durable tool-call journals."""

import hashlib
import copy
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from .llm import Backend
from .models import AgentError, BudgetExceeded, Cancelled, Limits, Plan, Pricing, StepCompletion
from .storage import read_json, run_lock, write_json
from .tools import ToolRegistry
from .verification import verify

FINISH_SCHEMA = {
    "type": "function", "name": "finish_step", "strict": True,
    "description": "Finish only the current step, reporting success or a blocker.",
    "parameters": {"type": "object", "properties": {
        "status": {"type": "string", "enum": ["completed", "failed"]},
        "summary": {"type": "string"}}, "required": ["status", "summary"],
        "additionalProperties": False},
}


class StepFailed(AgentError):
    pass


class Agent:
    def __init__(self, backend: Backend, tools: ToolRegistry, run_dir: Path,
                 limits: Limits = Limits(), on_event: Callable[[str], None] | None = None,
                 cancel: Callable[[], bool] | None = None, pricing: Pricing = Pricing()):
        self.backend, self.tools = backend, tools
        self.run_dir, self.limits, self.pricing = Path(run_dir), limits, pricing
        self.on_event = on_event or (lambda message: None)
        self.cancel = cancel or (lambda: False)
        self.state: dict = {}
        self._started, self._prior_elapsed = time.monotonic(), 0.0
        if limits.max_cost_usd is not None and pricing.input_per_million is None:
            raise ValueError("A cost cap requires both configured token prices.")

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state["metrics"]["elapsed_seconds"] = round(
            self._prior_elapsed + time.monotonic() - self._started, 4)
        write_json(self.run_dir / "run.json", self.state)

    def _configure(self) -> None:
        self._started = time.monotonic()
        self._prior_elapsed = self.state["metrics"]["elapsed_seconds"]
        deadline = self._started + self.limits.max_seconds - self._prior_elapsed
        for component in (self.backend, self.tools):
            if hasattr(component, "configure_runtime"):
                component.configure_runtime(deadline=deadline, cancel=self.cancel)

    def _guard(self) -> None:
        if self.cancel():
            raise Cancelled("Cancelled by the user.")
        if self._prior_elapsed + time.monotonic() - self._started >= self.limits.max_seconds:
            raise BudgetExceeded("Elapsed-time budget exhausted.")

    def _model_call(self, method, payload, *args):
        self._guard()
        metrics = self.state["metrics"]
        estimated_input, reserved_output = (self.backend.estimate_request(payload)
                                           if hasattr(self.backend, "estimate_request") else (0, 0))
        reserved = estimated_input + reserved_output
        if metrics["total_tokens"] + metrics["unconfirmed_reserved_tokens"] + reserved > self.limits.max_tokens:
            raise BudgetExceeded("Token budget cannot accommodate the next model request.")
        cost = self.pricing.estimate(estimated_input, reserved_output)
        if self.limits.max_cost_usd is not None and (
            (metrics["estimated_cost_usd"] or 0) + metrics["unconfirmed_cost_usd"]
            + (cost or 0) > self.limits.max_cost_usd
        ):
            raise BudgetExceeded("Estimated model-cost budget cannot accommodate the next request.")
        self.state["pending_request"] = {"operation": method.__name__,
                                         "reserved_tokens": reserved, "reserved_cost": cost or 0}
        self._save()
        start = time.monotonic()
        if hasattr(self.backend, "last_usage"):
            self.backend.last_usage = None
        try:
            result = method(*args)
        finally:
            duration = time.monotonic() - start
            metrics["model_requests"] += 1
            metrics["model_seconds"] += duration
            usage = getattr(self.backend, "last_usage", None)
            if usage is not None:
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    metrics[key] += int(usage.get(key, 0))
            elif hasattr(self.backend, "estimate_request"):
                metrics["usage_complete"] = False
                metrics["unconfirmed_reserved_tokens"] += reserved
                metrics["unconfirmed_cost_usd"] += cost or 0
            metrics["estimated_cost_usd"] = self.pricing.estimate(metrics["input_tokens"], metrics["output_tokens"])
            metrics["requests"].append({"operation": method.__name__, "seconds": duration,
                                        "usage": usage, "reserved_tokens": reserved})
            self.state["pending_request"] = None
            self._save()
        self._guard()
        if metrics["total_tokens"] > self.limits.max_tokens:
            raise BudgetExceeded("Observed token usage exceeded the token budget.")
        if self.limits.max_cost_usd is not None and (
            (metrics["estimated_cost_usd"] or 0) + metrics["unconfirmed_cost_usd"] > self.limits.max_cost_usd
        ):
            raise BudgetExceeded("Observed model-cost estimate exceeded the configured budget.")
        return result

    def _validate_plan(self, plan: Plan) -> None:
        if len(plan.steps) > self.limits.max_steps:
            raise AgentError("Plan exceeds the configured step limit.")
        if any(step.tool == "create_file" for step in plan.steps) and not plan.expected_artifacts:
            raise AgentError("File creation plans must declare expected_artifacts.")
        paths = ["/".join(ToolRegistry._path_parts(a.path)).casefold() for a in plan.expected_artifacts]
        if len(paths) != len(set(paths)):
            raise AgentError("Expected artifact paths must be unique.")

    def _set_plan(self, plan: Plan) -> None:
        self._validate_plan(plan)
        self.state["plan"] = plan.model_dump()
        self.state["steps"] = [self._step_record(step.model_dump(), n)
                               for n, step in enumerate(plan.steps, 1)]
        self.state["history"] = [{"role": "user", "content": json.dumps({
            "goal": self.state["goal"], "plan": plan.model_dump()})}]
        write_json(self.run_dir / "plan.json", {
            "schema_version": 2, "goal": self.state["goal"], "plan": plan.model_dump(),
            "demo": self.state["demo"], "model": self.state["model"],
        })

    @staticmethod
    def _step_record(step: dict, number: int) -> dict:
        return {"number": number, **step, "status": "pending", "events": [], "turns": 0, "started": False}

    def run(self, goal: str, plan_only: bool = False, plan: Plan | None = None) -> dict:
        if not goal.strip() or len(goal) > 12000:
            raise ValueError("Goal must contain 1 to 12,000 characters.")
        with run_lock(self.run_dir):
            if (self.run_dir / "run.json").exists():
                raise ValueError("Run exists. Use --resume or choose a new directory.")
            self.state = {
                "schema_version": 2, "run_id": self.run_dir.name, "goal": goal,
                "status": "planning", "plan": None, "steps": [], "history": [],
                "model": getattr(self.backend, "model", type(self.backend).__name__),
                "demo": bool(getattr(self.backend, "is_demo", False)),
                "limits": asdict(self.limits), "pricing": asdict(self.pricing),
                "tool_calls": 0, "replans": 0, "recovery_history": [],
                "pending_call": None, "pending_request": None, "verification": None, "artifacts": [],
                "metrics": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                            "model_requests": 0, "model_seconds": 0.0, "tool_seconds": 0.0,
                            "elapsed_seconds": 0.0, "estimated_cost_usd": self.pricing.estimate(0, 0),
                            "unconfirmed_reserved_tokens": 0, "unconfirmed_cost_usd": 0.0,
                            "usage_complete": True, "requests": []},
            }
            self._configure()
            self._save()
            return self._drive(plan_only=plan_only, supplied_plan=plan)

    def resume(self) -> dict:
        with run_lock(self.run_dir):
            self.state = read_json(self.run_dir / "run.json")
            if self.state.get("schema_version") != 2:
                raise AgentError("Only version 2 run records can be resumed.")
            self._configure()
            self.state["limits"] = asdict(self.limits)
            self.state["pricing"] = asdict(self.pricing)
            if self.state.get("pending_request"):
                reservation = self.state["pending_request"]
                self.state["metrics"]["unconfirmed_reserved_tokens"] += reservation["reserved_tokens"]
                self.state["metrics"]["unconfirmed_cost_usd"] += reservation["reserved_cost"]
                self.state["metrics"]["usage_complete"] = False
                self.state["pending_request"] = None
            self.state.pop("error", None)
            self.state.pop("stop_reason", None)
            try:
                self._guard()
            except (Cancelled, BudgetExceeded) as exc:
                status = "cancelled" if isinstance(exc, Cancelled) else "budget_exceeded"
                self.state.update(status=status, stop_reason=status, error=str(exc))
                self._save()
                return self.state
            for step in self.state["steps"]:
                for event in step["events"]:
                    if event.get("tool") == "create_file" and event.get("ok"):
                        expected = event["result"]
                        actual = self.tools.execute("read_file", json.dumps({"path": expected["path"]}))
                        if actual.get("code") in {"cancelled", "deadline_exceeded"}:
                            status = "cancelled" if actual["code"] == "cancelled" else "budget_exceeded"
                            self.state.update(status=status, stop_reason=status, error=actual["error"])
                            self._save()
                            return self.state
                        if not actual.get("ok") or actual["result"].get("sha256") != expected.get("sha256"):
                            self.state.update(status="blocked", stop_reason="artifact_changed",
                                              error="A checkpoint artifact was changed or removed; start a new run.")
                            self._save()
                            return self.state
            for step in self.state["steps"]:
                if step["status"] != "completed":
                    step["status"] = "pending"
            return self._drive()

    def _drive(self, plan_only=False, supplied_plan=None) -> dict:
        try:
            self._guard()
            if self.state["plan"] is None:
                self.on_event("Planning goal...")
                plan = supplied_plan or self._model_call(
                    self.backend.plan, {"goal": self.state["goal"], "tools": self.tools.schemas},
                    self.state["goal"], self.limits.max_steps, self.tools.schemas)
                self._set_plan(plan)
            plan = Plan.model_validate(self.state["plan"])
            self._validate_plan(plan)
            if not plan.supported:
                self.state.update(status="blocked", stop_reason="unsupported_goal",
                                  error=plan.unsupported_reason or "The goal needs clarification or unsupported tools.")
                return self.state
            if plan_only:
                self.state.update(status="planned", stop_reason="plan_saved")
                return self.state
            self.state["status"] = "running"
            self._save()
            index = 0
            while index < len(self.state["steps"]):
                step = self.state["steps"][index]
                if step["status"] != "completed":
                    try:
                        self._execute_step(step)
                    except StepFailed as exc:
                        if self._repair(plan, index, str(exc)):
                            continue
                        raise
                index += 1
            self._guard()
            self.state["status"] = "verifying"
            self._save()
            verification = verify(plan, self.state, self.tools)
            self._guard()
            self.state["verification"], self.state["artifacts"] = verification, verification["artifacts"]
            if verification["passed"]:
                self.state.update(status="completed", stop_reason="verified")
            else:
                self.state.update(status="failed", stop_reason="verification_failed",
                                  error="Expected outputs did not pass verification. Inspect the recorded checks.")
        except Cancelled as exc:
            self.state.update(status="cancelled", stop_reason="cancelled", error=str(exc))
        except BudgetExceeded as exc:
            self.state.update(status="budget_exceeded", stop_reason="budget_exceeded", error=str(exc))
        except (AgentError, ValueError, ValidationError) as exc:
            message = str(exc) if isinstance(exc, AgentError) else "Invalid plan or run data."
            self.state.update(status="failed", stop_reason="execution_failed", error=message)
        except KeyboardInterrupt:
            self.state.update(status="interrupted", stop_reason="interrupted", error="Interrupted by the user.")
        except Exception as exc:
            self.state.update(status="failed", stop_reason="unexpected_error", error=f"Unexpected {type(exc).__name__}.")
        finally:
            for step in self.state["steps"]:
                if step["status"] == "running":
                    step["status"] = "failed"
            self._save()
        return self.state

    def _repair(self, plan: Plan, index: int, error: str) -> bool:
        step = self.state["steps"][index]
        permanent = {"missing_credentials", "authentication", "authentication_failed", "permission_denied", "cancelled"}
        if (self.state["replans"] >= self.limits.max_replans or not hasattr(self.backend, "repair")
                or any(e.get("code") in permanent for e in step["events"])):
            return False
        self.state["replans"] += 1
        self.state["recovery_history"].append({"error": error, "replaced_steps": copy.deepcopy(self.state["steps"][index:])})
        repair_payload = {"goal": self.state["goal"], "plan": plan.model_dump(), "steps": self.state["steps"]}
        repaired = self._model_call(self.backend.repair, repair_payload, self.state["goal"], plan, self.state)
        if not repaired.supported:
            return False
        self._validate_plan(repaired)
        if (repaired.expected_artifacts != plan.expected_artifacts or repaired.minimum_sources != plan.minimum_sources
                or index + len(repaired.steps) > self.limits.max_steps):
            raise AgentError("Recovery plan changed output requirements or exceeded the step limit.")
        replacement = [self._step_record(s.model_dump(), index + n) for n, s in enumerate(repaired.steps, 1)]
        replacement[0]["events"] = step["events"]
        self.state["steps"] = self.state["steps"][:index] + replacement
        self.state["history"].append({"role": "user", "content": json.dumps({
            "recovery_plan": [s.model_dump() for s in repaired.steps],
            "instruction": "Execute revised remaining steps without repeating completed work."})})
        self.state["pending_call"] = None
        self._save()
        return True

    def _execute_step(self, step: dict) -> None:
        self._guard()
        step["status"] = "running"
        history = self.state["history"]
        if not step["started"]:
            history.append({"role": "user", "content": json.dumps({
                "current_step": step["number"], "description": step["description"],
                "required_tool": step["tool"], "expected_artifacts": self.state["plan"]["expected_artifacts"],
                "instruction": "Execute this step now, then call finish_step."})})
            step["started"] = True
        schemas = [t for t in self.tools.schemas if t["name"] == step["tool"]]
        allowed = {t["name"] for t in schemas}
        schemas.append(FINISH_SCHEMA)
        self.on_event(f"Executing step {step['number']}: {step['description']}")
        self._save()
        while True:
            self._guard()
            pending = self.state.get("pending_call")
            if pending is None:
                if step["turns"] >= self.limits.max_turns_per_step:
                    raise StepFailed(f"Step {step['number']} exceeded its model-turn limit.")
                step["turns"] += 1
                output = self._model_call(self.backend.respond, {"history": history, "tools": schemas}, history, schemas)
                history.extend(output)
                calls = [item for item in output if item.get("type") == "function_call"]
                if not calls:
                    history.append({"role": "user", "content": "Use a tool or finish_step."})
                    self._save()
                    continue
                if len(calls) != 1:
                    raise AgentError("Expected one function call per turn; execution stopped.")
                call = calls[0]
                if not all(isinstance(call.get(key), str) for key in ("name", "arguments", "call_id")):
                    raise AgentError("Malformed function-call envelope.")
                pending = {"call": call, "step": step["number"], "counted": False}
                self.state["pending_call"] = pending
                self._save()
            call = pending["call"]
            name, completion = call["name"], None
            if name == "finish_step":
                try:
                    completion = StepCompletion.model_validate_json(call["arguments"])
                    succeeded = any(e.get("tool") == step["tool"] and e.get("ok") for e in step["events"])
                    if completion.status == "completed" and step["tool"] != "reason" and not succeeded:
                        completion = None
                        result = {"ok": False, "error": "The required tool has not succeeded."}
                    else:
                        result = {"ok": True, "result": completion.model_dump()}
                except (ValidationError, TypeError):
                    result = {"ok": False, "error": "Invalid finish_step arguments."}
            else:
                if not pending["counted"]:
                    if self.state["tool_calls"] >= self.limits.max_tool_calls:
                        raise BudgetExceeded("Tool-call budget exhausted.")
                    self.state["tool_calls"] += 1
                    pending["counted"] = True
                    self._save()
                start = time.monotonic()
                result = (self._execute_or_reconcile(call) if name in allowed else
                          {"ok": False, "error": "Tool is not allowed for this step.", "code": "tool_not_allowed"})
                self.state["metrics"]["tool_seconds"] += time.monotonic() - start
                self.on_event(f"  {name}: {'ok' if result.get('ok') else 'failed'}")
            step["events"].append({"tool": name, "call_id": call["call_id"], **result})
            history.append({"type": "function_call_output", "call_id": call["call_id"],
                            "output": json.dumps(result, ensure_ascii=True)})
            self.state["pending_call"] = None
            if completion is not None:
                step.update(status=completion.status, summary=completion.summary)
            self._save()
            self._guard()
            if result.get("code") == "cancelled":
                raise Cancelled("Cancelled by the user.")
            if result.get("code") == "deadline_exceeded":
                raise BudgetExceeded("Elapsed-time budget exhausted.")
            if completion is not None:
                if completion.status == "failed":
                    raise StepFailed(f"Step {step['number']} failed: {completion.summary}")
                return
            if result.get("code") in {"missing_credentials", "authentication", "authentication_failed", "permission_denied"}:
                raise AgentError(result["error"])

    def _execute_or_reconcile(self, call: dict) -> dict:
        pending = self.state["pending_call"]
        if call["name"] == "create_file":
            try:
                args = json.loads(call["arguments"])
                if set(args) == {"path", "content"} and isinstance(args["content"], str):
                    ToolRegistry._path_parts(args["path"])
                    content = args["content"].encode("utf-8")
                    digest = hashlib.sha256(content).hexdigest()
                    if pending.get("write_started"):
                        existing = self.tools.execute("read_file", json.dumps({"path": args["path"]}))
                        if existing.get("ok"):
                            if existing["result"].get("sha256") != digest:
                                raise AgentError("Interrupted file write conflicts with an existing artifact.")
                            return {"ok": True, "result": {"path": args["path"], "sha256": digest,
                                    "bytes_written": len(content), "recovered": True}}
                    else:
                        existing = self.tools.execute("read_file", json.dumps({"path": args["path"]}))
                        if not existing.get("ok"):
                            pending["write_started"] = True
                            self._save()
            except (ValueError, TypeError, KeyError, UnicodeError):
                pass
        return self.tools.execute(call["name"], call["arguments"])
