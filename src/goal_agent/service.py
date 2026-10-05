"""Shared CLI/UI run service. The web service is local and single-user."""

import json
import os
import re
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

from .demo import DEMO_GOAL, DemoBackend, DemoTools
from .llm import OpenAIBackend
from .models import AgentError, Limits, Plan, Pricing
from .runner import Agent
from .storage import read_json
from .tools import ToolRegistry


def pricing_from_env() -> Pricing:
    def number(name):
        value = os.getenv(name, "").strip()
        return float(value) if value else None
    return Pricing(number("INPUT_COST_PER_MILLION"), number("OUTPUT_COST_PER_MILLION"))


def new_run_dir(root: Path) -> Path:
    return root.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])


def run_once(run_dir: Path, goal: str = "", *, demo=False, plan_only=False,
             plan: Plan | None = None, resume=False, model: str | None = None,
             limits: Limits = Limits(), pricing: Pricing | None = None,
             cancel=None, on_event=None) -> dict:
    client = None
    try:
        if demo:
            backend = DemoBackend()
            registry = DemoTools(run_dir / "artifacts", None)
        else:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise AgentError("Set OPENAI_API_KEY locally in .env or your environment, or use --demo.")
            # One retry layer lives inside the SDK; no additional blind LLM retries.
            client = OpenAI(api_key=api_key, timeout=45.0, max_retries=2)
            backend = OpenAIBackend(client, model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini")
            registry = ToolRegistry(run_dir / "artifacts", os.getenv("TAVILY_API_KEY"))
        agent = Agent(backend, registry, run_dir, limits, on_event=on_event,
                      cancel=cancel, pricing=pricing if pricing is not None else pricing_from_env())
        return agent.resume() if resume else agent.run(DEMO_GOAL if demo else goal, plan_only, plan)
    finally:
        if client:
            client.close()


class RunService:
    """One authenticated local user, isolated runs, bounded concurrent jobs."""

    def __init__(self, root: Path, model=None, limits: Limits = Limits(),
                 max_active_runs=1, max_runs=1000):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model, self.limits = model, limits
        self.max_active_runs, self.max_runs = max_active_runs, max_runs
        self._lock = threading.RLock()
        self._jobs: dict[str, threading.Event] = {}
        self._pending: dict[str, dict] = {}

    def _path(self, run_id: str) -> Path:
        if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}", run_id):
            raise ValueError("Invalid run ID.")
        path = self.root / run_id
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError("Run path escapes the configured directory.")
        for candidate in (path, path / "run.json", path / "artifacts"):
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Run files and directories must not use symlinks or reparse points.")
        return path

    def get_run(self, run_id: str) -> dict:
        with self._lock:
            path = self._path(run_id)
            if not (path / "run.json").exists():
                if run_id in self._pending:
                    return dict(self._pending[run_id])
                raise FileNotFoundError("Run not found.")
            state = read_json(path / "run.json")
            if (state.get("schema_version") != 2 or state.get("run_id") != run_id
                    or not isinstance(state.get("goal"), str)
                    or not isinstance(state.get("status"), str)
                    or not isinstance(state.get("steps"), list)
                    or not isinstance(state.get("artifacts"), list)
                    or not isinstance(state.get("limits"), dict)
                    or not isinstance(state.get("pricing"), dict)):
                raise ValueError("Invalid or unsupported saved run record.")
            for artifact in state["artifacts"]:
                if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                    raise ValueError("Invalid saved artifact record.")
            for step in state["steps"]:
                if not isinstance(step, dict) or not isinstance(step.get("events"), list):
                    raise ValueError("Invalid saved step record.")
                for event in step["events"]:
                    if not isinstance(event, dict):
                        raise ValueError("Invalid saved tool event.")
                    if event.get("ok") and event.get("tool") == "create_file":
                        result = event.get("result")
                        if not isinstance(result, dict) or not isinstance(result.get("path"), str):
                            raise ValueError("Invalid saved file creation result.")
            public = {key: value for key, value in state.items()
                      if key not in {"history", "pending_call", "pending_request", "recovery_history"}}
            if run_id not in self._jobs and public["status"] in {"running", "planning", "verifying"}:
                public["status"] = "interrupted"
            pending = self._pending.get(run_id, {})
            if run_id not in self._jobs and pending.get("status") == "failed" and pending.get("error"):
                public.update(status="failed", error=pending["error"])
            artifacts = {a["path"]: a for a in public.get("artifacts", [])}
            for step in state.get("steps", []):
                for event in step.get("events", []):
                    if event.get("ok") and event.get("tool") == "create_file":
                        artifacts[event["result"]["path"]] = event["result"]
            public["artifacts"] = list(artifacts.values())
            return public

    def list_runs(self) -> list[dict]:
        with self._lock:
            ids = {p.parent.name for p in self.root.glob("*/run.json")} | set(self._pending)
        runs = []
        for run_id in sorted(ids, reverse=True):
            try:
                state = self.get_run(run_id)
            except (AgentError, ValueError, OSError):
                continue
            runs.append({k: state.get(k) for k in ("run_id", "goal", "status", "updated_at", "demo", "metrics")})
        return runs

    def _launch(self, run_dir: Path, **kwargs) -> str:
        with self._lock:
            if len(self._jobs) >= self.max_active_runs:
                raise ValueError("Local concurrency limit reached; wait or cancel the active run.")
            if run_dir.name in self._jobs:
                raise ValueError("Run is already active.")
            flag = threading.Event()
            self._jobs[run_dir.name] = flag
            self._pending[run_dir.name] = {"run_id": run_dir.name, "goal": kwargs.get("goal", ""),
                                           "status": "queued", "demo": kwargs.get("demo", False)}

        def work():
            try:
                result = run_once(run_dir, cancel=flag.is_set, **kwargs)
                with self._lock:
                    self._pending[run_dir.name] = result
            except Exception as exc:
                with self._lock:
                    self._pending[run_dir.name].update(status="failed", error=(
                        str(exc) if isinstance(exc, (AgentError, ValueError)) else f"Unexpected {type(exc).__name__}."))
            finally:
                with self._lock:
                    self._jobs.pop(run_dir.name, None)

        threading.Thread(target=work, name=f"run-{run_dir.name}", daemon=True).start()
        return run_dir.name

    def start(self, goal: str, plan_only=False, demo=False) -> str:
        if not demo and (not isinstance(goal, str) or not goal.strip() or len(goal) > 12000):
            raise ValueError("Provide a goal of 1 to 12,000 characters.")
        if not demo and not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("Configure OPENAI_API_KEY locally, or select Offline demo.")
        with self._lock:
            if len(self.list_runs()) >= self.max_runs:
                raise ValueError("Local saved-run limit reached. Archive older runs before starting more.")
            # Queue registration belongs to the same critical section as the
            # saved-run quota check, including when simultaneous POSTs arrive.
            return self._launch(new_run_dir(self.root), goal=DEMO_GOAL if demo else goal,
                                plan_only=plan_only, demo=demo, model=self.model, limits=self.limits)

    def execute(self, run_id: str) -> str:
        state = self.get_run(run_id)
        if state["status"] != "planned":
            raise ValueError("Only a saved plan can be executed. Use Resume for interrupted runs.")
        return self._continue(run_id, state)

    def resume(self, run_id: str) -> str:
        state = self.get_run(run_id)
        if state["status"] in {"queued", "running", "planning", "verifying"}:
            raise ValueError("Run is already active.")
        return self._continue(run_id, state)

    def _continue(self, run_id: str, state: dict) -> str:
        if not state.get("demo") and not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("Configure OPENAI_API_KEY locally before resuming this run.")
        return self._launch(self._path(run_id), goal=state["goal"], demo=state.get("demo", False),
                            resume=True, model=state.get("model") or self.model,
                            limits=Limits(**state["limits"]), pricing=Pricing(**state["pricing"]))

    def cancel(self, run_id: str) -> bool:
        self._path(run_id)
        with self._lock:
            flag = self._jobs.get(run_id)
            if flag:
                flag.set()
            return flag is not None

    def artifact(self, run_id: str, path: str) -> tuple[bytes, str]:
        state = self.get_run(run_id)
        if path not in {a["path"] for a in state.get("artifacts", [])}:
            raise FileNotFoundError("Artifact not listed for this run.")
        registry = ToolRegistry(self._path(run_id) / "artifacts", None)
        result = registry.execute("read_file", json.dumps({"path": path}))
        if not result.get("ok"):
            raise FileNotFoundError("Artifact unavailable.")
        expected = next(a for a in state["artifacts"] if a["path"] == path)
        if result["result"]["sha256"] != expected.get("sha256"):
            raise ValueError("Artifact changed since it was recorded; download refused. Start a new run.")
        return result["result"]["content"].encode("utf-8"), "application/octet-stream"
