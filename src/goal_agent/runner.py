"""Sequential plan-and-execute orchestration with explicit stop conditions."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from .llm import Backend
from .models import AgentError, Limits, StepCompletion
from .tools import ToolRegistry


FINISH_SCHEMA = {
    "type": "function", "name": "finish_step", "strict": True,
    "description": "Finish only the current step, reporting success or a blocker.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed", "failed"]},
            "summary": {"type": "string"},
        },
        "required": ["status", "summary"], "additionalProperties": False,
    },
}


class Agent:
    def __init__(self, backend: Backend, tools: ToolRegistry, run_dir: Path,
                 limits: Limits = Limits(),
                 on_event: Callable[[str], None] | None = None):
        self.backend, self.tools = backend, tools
        self.run_dir, self.limits = Path(run_dir), limits
        self.on_event = on_event or (lambda message: None)
        self.state: dict = {}

    def _save(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        pending = self.run_dir / "run.json.tmp"
        pending.write_text(json.dumps(self.state, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        pending.replace(self.run_dir / "run.json")

    def run(self, goal: str, plan_only: bool = False) -> dict:
        if not goal.strip() or len(goal) > 12000:
            raise ValueError("Goal must contain 1 to 12,000 characters.")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if (self.run_dir / "run.json").exists():
            raise ValueError("Choose a new run directory; an existing run cannot be overwritten.")
        self.state = {
            "goal": goal, "status": "planning", "plan": None,
            "steps": [], "tool_calls": 0,
        }
        self._save()
        try:
            self.on_event("Planning goal...")
            plan = self.backend.plan(goal, self.limits.max_steps, self.tools.schemas)
            if len(plan.steps) > self.limits.max_steps:
                raise AgentError("Plan exceeds the configured step limit.")
            self.state["plan"] = plan.model_dump()
            self.state["steps"] = [
                {"number": n, **step.model_dump(), "status": "pending", "events": []}
                for n, step in enumerate(plan.steps, start=1)
            ]
            for step in self.state["steps"]:
                self.on_event(f"{step['number']}. [{step['tool']}] {step['description']}")
            self.state["status"] = "planned" if plan_only else "running"
            self._save()
            if plan_only:
                return self.state

            history = [{"role": "user", "content": json.dumps({
                "goal": goal, "plan": plan.model_dump(),
            })}]
            for step in self.state["steps"]:
                self._execute_step(step, history)
            self.state["status"] = "completed"
        except AgentError as exc:
            self.state.update(status="failed", error=str(exc))
        except KeyboardInterrupt:
            self.state.update(status="interrupted", error="Interrupted by the user.")
        except Exception as exc:
            # Do not leak SDK response bodies, credentials, or filesystem paths.
            self.state.update(status="failed", error=f"Unexpected {type(exc).__name__}.")
        finally:
            for step in self.state["steps"]:
                if step["status"] == "running":
                    step["status"] = "failed"
            self._save()
        return self.state

    def _execute_step(self, step: dict, history: list[dict]) -> None:
        step["status"] = "running"
        successful_tools: set[str] = set()
        history.append({"role": "user", "content": json.dumps({
            "current_step": step["number"], "description": step["description"],
            "required_tool": step["tool"],
            "instruction": "Execute this step now, then call finish_step.",
        })})
        # Expose only this step's action and the completion control tool.
        schemas = [tool for tool in self.tools.schemas if tool["name"] == step["tool"]]
        allowed_names = {tool["name"] for tool in schemas}
        schemas = [*schemas, FINISH_SCHEMA]
        self.on_event(f"Executing step {step['number']}...")
        self._save()
        for _ in range(self.limits.max_turns_per_step):
            output = self.backend.respond(history, schemas)
            # Replay every output item, including reasoning and tool-call IDs.
            history.extend(output)
            calls = [item for item in output if item.get("type") == "function_call"]
            if not calls:
                history.append({"role": "user", "content": "Use a tool or finish_step."})
                continue
            if len(calls) != 1:
                raise AgentError("Expected one function call per turn; execution stopped.")
            call = calls[0]
            name = call["name"]
            completion = None
            if name == "finish_step":
                try:
                    completion = StepCompletion.model_validate_json(call["arguments"])
                    if (completion.status == "completed" and step["tool"] != "reason"
                            and step["tool"] not in successful_tools):
                        completion = None
                        result = {"ok": False, "error": "The required tool has not succeeded."}
                    else:
                        result = {"ok": True, "result": completion.model_dump()}
                except (ValidationError, TypeError):
                    result = {"ok": False, "error": "Invalid finish_step arguments."}
            else:
                if self.state["tool_calls"] >= self.limits.max_tool_calls:
                    raise AgentError("Tool-call budget exhausted.")
                self.state["tool_calls"] += 1
                if name not in allowed_names:
                    result = {"ok": False, "error": "Tool is not allowed for this step."}
                else:
                    result = self.tools.execute(name, call["arguments"])
                if result.get("ok"):
                    successful_tools.add(name)
                self.on_event(f"  {name}: {'ok' if result.get('ok') else 'failed'}")
            # Log observations without model-generated arguments (may contain large files).
            step["events"].append({"tool": name, "call_id": call["call_id"], **result})
            history.append({"type": "function_call_output", "call_id": call["call_id"],
                            "output": json.dumps(result, ensure_ascii=False)})
            self._save()
            if completion is not None:
                step.update(status=completion.status, summary=completion.summary)
                self._save()
                if completion.status == "failed":
                    raise AgentError(f"Step {step['number']} failed: {completion.summary}")
                return
        raise AgentError(f"Step {step['number']} exceeded its model-turn limit.")
