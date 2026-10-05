"""CLI for verified goals, saved plans, resumable runs, evaluations, and the UI."""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .demo import DEMO_GOAL
from .models import AgentError, Limits, Plan, Pricing
from .service import RunService, new_run_dir, pricing_from_env, run_once
from .storage import read_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan, execute, verify, and resume research/file tasks.")
    parser.add_argument("goal", nargs="?", help="A high-level goal in plain English")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Run the fixed offline demo")
    mode.add_argument("--execute-plan", type=Path, help="Execute a saved plan.json exactly, in a new run")
    mode.add_argument("--resume", type=Path, help="Resume a run directory or run.json")
    mode.add_argument("--ui", "--serve", action="store_true", help="Start the authenticated local web interface")
    mode.add_argument("--eval", action="store_true", help="Run the repeatable evaluation suite")
    parser.add_argument("--plan-only", action="store_true", help="Save a plan without executing action tools")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--model", help="Override OPENAI_MODEL for a new run")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-turns-per-step", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-replans", type=int)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--live", action="store_true", help="Enable paid API smoke tests with --eval")
    parser.add_argument("--eval-output", type=Path, default=Path("evaluations"))
    args = parser.parse_args(argv)
    if args.live and not args.eval:
        parser.error("--live is only valid with --eval.")
    if args.goal and any((args.demo, args.execute_plan, args.resume, args.ui, args.eval)):
        parser.error("The selected mode supplies its own goal; omit the positional goal.")
    if args.plan_only and any((args.resume, args.execute_plan, args.ui, args.eval)):
        parser.error("--plan-only is only valid with a new goal or --demo.")
    load_dotenv(Path.cwd() / ".env", override=False)
    try:
        if args.eval:
            from .evals import run_evaluations
            report = run_evaluations(args.eval_output, live=args.live, model=args.model)
            print(f"Evaluations: {report['passed']}/{report['total']} passed")
            print(f"Report: {report['report_path']}")
            return 0 if report["failed"] == 0 else 1
        saved = None
        run_dir = None
        if args.resume:
            run_dir = args.resume.resolve()
            if run_dir.is_file():
                run_dir = run_dir.parent
            saved = read_json(run_dir / "run.json")
            if args.model and args.model != saved.get("model"):
                parser.error("Resuming preserves the model. Start a new run to change it.")
        config = saved.get("limits", {}) if saved else asdict(Limits())
        config = {**config, **{name: getattr(args, name) for name in asdict(Limits())
                              if getattr(args, name) is not None}}
        limits = Limits(**config)
        pricing = Pricing(**saved["pricing"]) if saved else pricing_from_env()
        if args.ui:
            from .web import serve
            serve(RunService(args.runs_dir, model=args.model, limits=limits), port=args.port)
            return 0
        supplied_plan = None
        demo = args.demo
        goal = DEMO_GOAL if demo else args.goal
        model = args.model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
        if args.execute_plan:
            path = args.execute_plan
            document = read_json(path / "plan.json" if path.is_dir() else path)
            if document.get("schema_version") != 2:
                raise AgentError("Expected a version 2 saved plan.")
            supplied_plan = Plan.model_validate(document["plan"])
            goal, demo = document["goal"], document.get("demo", False)
            model = args.model or document.get("model") or model
        if saved:
            goal, demo, model = saved["goal"], saved.get("demo", False), saved["model"]
        if not goal or not goal.strip() or len(goal) > 12000:
            parser.error("Provide a goal of 1 to 12,000 characters, or use --demo / --ui.")
        run_dir = run_dir or new_run_dir(args.runs_dir)
        if demo:
            print("OFFLINE DEMO: scripted decisions and source fixture; file creation is real.")
        print(f"Run directory: {run_dir}")
        state = run_once(run_dir, goal, demo=demo, plan_only=args.plan_only,
                         plan=supplied_plan, resume=bool(saved), model=model,
                         limits=limits, pricing=pricing, on_event=print)
        print(f"\nStatus: {state['status']} ({state.get('stop_reason', '')})")
        if state.get("error"):
            print(state["error"], file=sys.stderr)
        if state["status"] == "planned":
            for n, step in enumerate(state["plan"]["steps"], 1):
                print(f"{n}. [{step['tool']}] {step['description']}")
            print(f"Saved plan: {run_dir / 'plan.json'}")
        for step in state["steps"]:
            if step.get("summary"):
                print(f"{step['number']}. {step['summary']}")
        print(f"Tokens: {state['metrics']['total_tokens']} observed; "
              f"elapsed: {state['metrics']['elapsed_seconds']:.1f}s")
        cost = state["metrics"]["estimated_cost_usd"]
        print(f"Model cost estimate: ${cost:.6f}" if cost is not None else "Cost estimate: pricing not configured")
        print(f"Run record: {run_dir / 'run.json'}")
        print(f"Artifacts: {run_dir / 'artifacts'}")
        return 0 if state["status"] in {"completed", "planned"} else 1
    except (AgentError, ValueError, KeyError, OSError) as exc:
        # ValidationError inherits ValueError; do not print model inputs or secrets.
        message = str(exc) if isinstance(exc, AgentError) else "Invalid configuration or unavailable run file. Check paths, options, and .env."
        print(message, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
