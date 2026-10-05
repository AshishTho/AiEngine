"""Repeatable functional evaluations with independent artifact/status grading.

Offline mode executes scripted decisions through the production runner and tool
registry. It does not measure an LLM's reasoning or prompt-injection resistance.
Live mode is opt-in, uses real APIs, and runs only cases marked ``live``.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
from openai import OpenAI

from .llm import OpenAIBackend
from .models import AgentError, Limits, Plan, Pricing
from .runner import Agent
from .tools import ToolRegistry


DEFAULT_CASES = Path(__file__).with_name("evaluation_cases.json")
_URL = re.compile(r"https?://[^\s<>\[\]\"')]+")
_ID = re.compile(r"[a-z][a-z0-9_-]{0,79}\Z")


class ScriptedBackend:
    """Replay data fixtures without contacting an LLM or inventing observations."""

    model = "offline-scripted-fixture"

    def __init__(self, plan: Plan, responses: list[dict | list[dict]]) -> None:
        self.result = plan
        self.responses = deque(copy.deepcopy(responses))
        self.histories: list[list[dict]] = []
        self.turn = 0

    def plan(self, goal: str, max_steps: int, tools: list[dict]) -> Plan:
        return self.result.model_copy(deep=True)

    def respond(self, history: list[dict], tools: list[dict]) -> list[dict]:
        self.histories.append(copy.deepcopy(history))
        if not self.responses:
            raise AgentError("Evaluation script exhausted before execution finished.")
        self.turn += 1
        actions = self.responses.popleft()
        if isinstance(actions, dict):
            actions = [actions]
        return [{
            "type": "function_call", "name": action["name"],
            "call_id": f"fixture_{self.turn}_{index}",
            "arguments": (action["arguments"] if isinstance(action["arguments"], str)
                          else json.dumps(action["arguments"])),
        } for index, action in enumerate(actions)]


def load_cases(path: Path | None = None) -> list[dict]:
    """Load local fixture data; identifiers cannot escape the evaluation folder."""
    payload = json.loads((path or DEFAULT_CASES).read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation cases must contain a nonempty cases list.")
    seen = set()
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str)
                or not _ID.fullmatch(case["id"])):
            raise ValueError("Each evaluation case needs a safe, unique identifier.")
        if case["id"] in seen:
            raise ValueError("Evaluation case identifiers must be unique.")
        seen.add(case["id"])
        if not isinstance(case.get("goal"), str) or not case["goal"].strip():
            raise ValueError("Each evaluation case needs a nonempty goal.")
        Plan.model_validate(case["plan"])
        expected = case.get("expected", {})
        if expected.get("status") not in {
            "completed", "planned", "failed", "blocked", "cancelled", "budget_exceeded",
        }:
            raise ValueError("Each evaluation case needs an explicit expected status.")
        for artifact in expected.get("artifacts", []):
            ToolRegistry._path_parts(artifact["path"])
        for forbidden in expected.get("forbidden_files", []):
            ToolRegistry._path_parts(forbidden)
        for artifact in case.get("initial_files", []):
            ToolRegistry._path_parts(artifact["path"])
    return cases


def _search_response(case: dict):
    """Replace only the HTTP boundary, retaining real search argument validation."""
    def respond(url: str, **kwargs: Any) -> httpx.Response:
        if url != "https://api.tavily.com/search":
            raise AssertionError("An offline evaluation attempted an unexpected endpoint.")
        return httpx.Response(
            case.get("search_http_status", 200),
            json={"results": copy.deepcopy(case.get("sources", []))},
            request=httpx.Request("POST", url),
        )
    return respond


def _events(state: dict) -> list[dict]:
    return [event for step in state.get("steps", []) for event in step.get("events", [])]


def _source_key(url: str) -> str:
    # A fragment addresses content within the same retrieved page.
    try:
        parts = urlsplit(url.rstrip(".,;!?"))
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                           parts.path.rstrip("/"), parts.query, ""))
    except ValueError:
        return url


def _retrieved_urls(state: dict) -> set[str]:
    urls: set[str] = set()
    for event in _events(state):
        if not event.get("ok"):
            continue
        result = event.get("result", {})
        if event.get("tool") == "web_search":
            urls.update(_source_key(source["url"]) for source in result.get("sources", [])
                        if isinstance(source, dict) and isinstance(source.get("url"), str))
        elif event.get("tool") == "fetch_page" and isinstance(result.get("url"), str):
            urls.add(_source_key(result["url"]))
    return urls


def _headings(content: str) -> set[str]:
    return {re.sub(r"\s+#+\s*$", "", line).strip().casefold()
            for line in re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", content, re.MULTILINE)}


def grade_case(case: dict, state: dict, workspace: Path) -> dict:
    """Grade actual files and source events, independently of runner verification.

    Citation provenance means a URL occurred in retrieved evidence. It does not
    establish that a source supports the surrounding factual claim.
    """
    expected = case["expected"]
    sources = _retrieved_urls(state)
    artifact_checks: list[dict] = []
    citations_checked = 0
    citations_grounded = 0
    for artifact in expected.get("artifacts", []):
        path = workspace.joinpath(*ToolRegistry._path_parts(artifact["path"]))
        exists = path.is_file() and not path.is_symlink()
        inside_workspace = path.resolve().is_relative_to(workspace.resolve())
        content = path.read_text(encoding="utf-8") if exists and inside_workspace else ""
        citations = {_source_key(url) for url in _URL.findall(content)}
        grounded = citations & sources
        citations_checked += len(citations)
        citations_grounded += len(grounded)
        required = {section.strip().lstrip("#").strip().casefold()
                    for section in artifact.get("required_sections", [])}
        checks = {
            "file_exists": exists and inside_workspace,
            "nonempty": bool(content.strip()),
            "required_sections": required <= _headings(content),
            "minimum_citations": len(grounded) >= artifact.get("min_sources", 0),
            "citation_provenance": citations <= sources,
            "required_text": all(text in content for text in artifact.get("contains", [])),
        }
        artifact_checks.append({"path": artifact["path"], "passed": all(checks.values()),
                                "checks": checks, "citations": len(citations),
                                "grounded_citations": len(grounded)})
    artifacts_valid = all(item["passed"] for item in artifact_checks)
    actual_events = _events(state)
    rejected = sum(1 for event in actual_events
                   if event.get("tool") != "finish_step" and not event.get("ok"))
    # A disallowed call that is rejected is an attempted violation, not a bypass.
    policy_attempts = sum(1 for event in actual_events
                          if "not allowed" in str(event.get("error", "")).lower())
    policy_violations = sum(1 for step in state.get("steps", [])
                            for event in step.get("events", [])
                            if event.get("ok") and event.get("tool") != "finish_step"
                            and event.get("tool") != step.get("tool"))
    checks = {
        "status": state.get("status") == expected["status"],
        "artifact_outcome": artifacts_valid == expected.get("artifacts_valid", True),
        "minimum_sources": len(sources) >= expected.get("minimum_sources", 0),
        "forbidden_files_absent": not any(
            workspace.joinpath(*ToolRegistry._path_parts(name)).exists()
            for name in expected.get("forbidden_files", [])),
        "required_tool_rejections": rejected >= expected.get("min_rejected_calls", 0),
        "no_policy_bypass": policy_violations == 0,
    }
    if "stop_reason" in expected:
        checks["stop_reason"] = state.get("stop_reason") == expected["stop_reason"]
    return {
        "id": case["id"], "goal": case["goal"], "passed": all(checks.values()),
        "expected_status": expected["status"], "actual_status": state.get("status"),
        "checks": checks, "artifacts": artifact_checks,
        "artifact_outcome_valid": artifacts_valid,
        "retrieved_source_count": len(sources), "citations_checked": citations_checked,
        "citations_with_retrieved_provenance": citations_grounded,
        "tool_policy_attempts_rejected": policy_attempts,
        "tool_policy_violations": policy_violations,
        "metrics": state.get("metrics", {}),
        "verification": state.get("verification"),
        "error": state.get("error"),
    }


def _numeric_metric(metrics: dict, *names: str) -> float:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def run_evaluations(output_dir: Path, live: bool = False, model: str | None = None,
                    cases_path: Path | None = None) -> dict:
    """Run the suite and save report.json plus full per-case run records.

    Offline mode never calls an API and does not require credentials. ``live``
    explicitly enables paid OpenAI/Tavily requests for the three smoke cases.
    A fresh child folder is used every time, preserving all previous reports.
    """
    cases = load_cases(cases_path)
    if live:
        cases = [case for case in cases if case.get("live")]
        if not cases:
            raise ValueError("No evaluation cases are marked live.")
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("Live evaluations require OPENAI_API_KEY.")
        if not os.getenv("TAVILY_API_KEY", "").strip():
            raise ValueError("Live evaluations require TAVILY_API_KEY.")
    selected_model = (model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()
    if not selected_model:
        raise ValueError("The model name must not be blank.")
    if live:
        input_price = os.getenv("INPUT_COST_PER_MILLION", "").strip()
        output_price = os.getenv("OUTPUT_COST_PER_MILLION", "").strip()
        pricing = Pricing(float(input_price) if input_price else None,
                          float(output_price) if output_price else None)
    else:
        pricing = Pricing(0.0, 0.0)
    suite_dir = Path(output_dir).resolve() / (
        datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
    suite_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    results = []
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=45.0, max_retries=2) if live else None
    try:
        for case in cases:
            run_dir = suite_dir / case["id"]
            registry = ToolRegistry(
                run_dir / "artifacts",
                os.getenv("TAVILY_API_KEY") if live else
                (None if case.get("missing_search_key") else "offline-fixture-key"),
            )
            for artifact in case.get("initial_files", []):
                seeded = registry.execute("create_file", json.dumps(artifact))
                if not seeded.get("ok"):
                    raise ValueError(f"Cannot create initial file for case {case['id']}.")
            backend = (OpenAIBackend(client, selected_model) if live else ScriptedBackend(
                Plan.model_validate(case["plan"]), case.get("responses", [])))
            # Mock only the search HTTP boundary. Other network calls fail closed.
            search_context = (nullcontext() if live else patch(
                "goal_agent.tools.httpx.post", side_effect=_search_response(case)))
            network_context = (nullcontext() if live else patch(
                "httpx.Client.send", side_effect=AssertionError("Offline evaluations forbid network requests.")))
            case_started = perf_counter()
            with search_context, network_context:
                state = Agent(backend, registry, run_dir, Limits(**case.get("limits", {})),
                              pricing=pricing).run(case["goal"])
            result = grade_case(case, state, registry.workspace)
            result["elapsed_seconds"] = round(perf_counter() - case_started, 6)
            result["run_path"] = str(run_dir / "run.json")
            results.append(result)
    finally:
        if client is not None:
            client.close()
    passed = sum(result["passed"] for result in results)
    positive_cases = [result for result in results if result["expected_status"] == "completed"]
    citations = sum(result["citations_checked"] for result in results)
    grounded = sum(result["citations_with_retrieved_provenance"] for result in results)
    completed_results = [result for result in results if result["actual_status"] == "completed"]
    accepted_citations = sum(result["citations_checked"] for result in completed_results)
    accepted_grounded = sum(result["citations_with_retrieved_provenance"] for result in completed_results)
    costs = [result["metrics"].get("estimated_cost_usd") for result in results]
    estimated_cost = (sum(costs) if all(isinstance(cost, (int, float)) for cost in costs)
                      else None) if live else 0.0
    report = {
        "mode": "live" if live else "offline",
        "model": selected_model if live else ScriptedBackend.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": ("Live API smoke tests with independent file/status/provenance checks."
                        if live else "Deterministic functional fixtures; not a live-model quality benchmark."),
        "limitations": [
            "Source provenance does not establish factual accuracy or citation entailment.",
            "Offline injection fixtures test tool containment, not model resistance to malicious text.",
            "Smoke cases do not measure quality across arbitrary goals.",
        ],
        "total": len(results), "passed": passed, "failed": len(results) - passed,
        "metrics": {
            "case_pass_rate": passed / len(results),
            "task_success_rate": (sum(result["passed"] for result in positive_cases) / len(positive_cases)
                                  if positive_cases else None),
            "task_success_eligible_cases": len(positive_cases),
            "citation_provenance_rate": grounded / citations if citations else None,
            "completed_output_citation_provenance_rate": (
                accepted_grounded / accepted_citations if accepted_citations else None),
            "citations_checked": citations, "citations_with_retrieved_provenance": grounded,
            "citation_metric_note": "All observed artifacts include deliberately invalid negative fixtures; the completed-output rate excludes stopped runs.",
            "tool_policy_attempts_rejected": sum(result["tool_policy_attempts_rejected"] for result in results),
            "tool_policy_violations": sum(result["tool_policy_violations"] for result in results),
            "total_tokens": int(sum(_numeric_metric(result["metrics"], "total_tokens", "tokens")
                                    for result in results)),
            "input_tokens": int(sum(_numeric_metric(result["metrics"], "input_tokens")
                                    for result in results)),
            "output_tokens": int(sum(_numeric_metric(result["metrics"], "output_tokens")
                                     for result in results)),
            "usage_complete": all(result["metrics"].get("usage_complete", not live) for result in results),
            "unconfirmed_reserved_tokens": int(sum(
                _numeric_metric(result["metrics"], "unconfirmed_reserved_tokens") for result in results)),
            "model_requests": (int(sum(_numeric_metric(result["metrics"], "model_requests")
                                        for result in results)) if live else 0),
            "backend_requests": int(sum(_numeric_metric(result["metrics"], "model_requests")
                                        for result in results)),
            "backend_seconds": sum(_numeric_metric(result["metrics"], "model_seconds")
                                   for result in results),
            "tool_seconds": sum(_numeric_metric(result["metrics"], "tool_seconds")
                                for result in results),
            "elapsed_seconds": round(perf_counter() - started, 6),
            "estimated_cost_usd": estimated_cost,
            "usage_note": ("Observed provider usage; cost uses configured prices and excludes Tavily."
                           if live else "No model requests: token usage and API cost are zero."),
        },
        "cases": results, "suite_path": str(suite_dir), "report_path": str(suite_dir / "report.json"),
    }
    (suite_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
