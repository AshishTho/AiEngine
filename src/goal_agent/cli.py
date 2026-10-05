"""CLI entry point: goal-agent \"your plain-English goal\"."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI

from .demo import DEMO_GOAL, DemoBackend, DemoTools
from .llm import OpenAIBackend
from .models import Limits
from .runner import Agent
from .tools import ToolRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan a goal, search the web, and create files sequentially.")
    parser.add_argument("goal", nargs="?", help="A high-level goal in plain English")
    parser.add_argument("--demo", action="store_true", help="Run a fixed demo with no API calls")
    parser.add_argument("--plan-only", action="store_true", help="Generate a plan without executing tools")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Parent folder for new runs")
    parser.add_argument("--model", help="Override OPENAI_MODEL (default: gpt-4.1-mini)")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-turns-per-step", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=24)
    args = parser.parse_args(argv)

    if args.demo and args.goal:
        parser.error("--demo uses a fixed example; omit the goal argument.")
    goal = DEMO_GOAL if args.demo else args.goal
    if not goal or not goal.strip() or len(goal) > 12000:
        parser.error("Provide a goal of 1 to 12,000 characters, or use --demo.")
    try:
        limits = Limits(args.max_steps, args.max_turns_per_step, args.max_tool_calls)
    except ValueError as exc:
        parser.error(str(exc))

    # Load only this project's/current directory's .env, without replacing exported values.
    load_dotenv(Path.cwd() / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not args.demo and not api_key:
        parser.error("Set OPENAI_API_KEY in .env or your environment. Try --demo without a key.")
    model = (args.model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()
    if not model:
        parser.error("The model name must not be blank.")

    run_dir = args.runs_dir.resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    client = None
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        if args.demo:
            print("OFFLINE DEMO: fixed decisions and search fixture; file creation is real.")
            backend = DemoBackend()
            registry = DemoTools(run_dir / "artifacts", None)
        else:
            # SDK retries transient connection/429/5xx failures at most twice.
            client = OpenAI(api_key=api_key, timeout=45.0, max_retries=2)
            backend = OpenAIBackend(client, model)
            registry = ToolRegistry(run_dir / "artifacts", os.getenv("TAVILY_API_KEY"))
        print(f"Run directory: {run_dir}")
        state = Agent(backend, registry, run_dir, limits, on_event=print).run(
            goal, plan_only=args.plan_only)
        print(f"\nStatus: {state['status']}")
        if state.get("error"):
            print(state["error"], file=sys.stderr)
        for step in state["steps"]:
            if step.get("summary"):
                print(f"{step['number']}. {step['summary']}")
        print(f"Run record: {run_dir / 'run.json'}")
        print(f"Artifacts: {run_dir / 'artifacts'}")
        return 0 if state["status"] in {"completed", "planned"} else 1
    except OSError:
        print("Cannot create or update the run directory. Check its path and permissions.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    finally:
        if client is not None:
            client.close()
