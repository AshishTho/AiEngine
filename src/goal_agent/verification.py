"""Deterministic artifact, section, and citation-provenance checks.

These checks establish source provenance and required structure, not factual
entailment or research completeness. Evals and human review address those gaps.
"""

import json
import re
from urllib.parse import urlsplit, urlunsplit

from .models import Plan


def canonical_url(url: str) -> str:
    parts = urlsplit(url.rstrip(".,;!?"))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path.rstrip("/"), parts.query, ""))


def collect_sources(state: dict) -> set[str]:
    urls = set()
    for step in state.get("steps", []):
        for event in step.get("events", []):
            if not event.get("ok") or event.get("tool") not in {"web_search", "fetch_page"}:
                continue
            result = event.get("result", {})
            for source in result.get("sources", []):
                if source.get("url") and (source.get("snippet") or result.get("content")):
                    urls.add(canonical_url(source["url"]))
    return urls


def verify(plan: Plan, state: dict, tools) -> dict:
    checks = []
    artifacts = []
    sources = collect_sources(state)

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("supported_goal", plan.supported, plan.unsupported_reason or "Goal is supported.")
    research_tools = {"web_search", "fetch_page"}
    research = (any(step.tool in research_tools for step in plan.steps)
                or any(step.get("tool") in research_tools for step in state.get("steps", []))
                or any(event.get("tool") in research_tools for step in state.get("steps", [])
                       for event in step.get("events", [])))
    needed = max(plan.minimum_sources, int(research))
    check("research_evidence", len(sources) >= needed,
          f"Retrieved {len(sources)} distinct usable sources; required {needed}.")
    for expected in plan.expected_artifacts:
        observation = tools.execute("read_file", json.dumps({"path": expected.path}))
        present = observation.get("ok", False)
        check(f"file:{expected.path}", present, "File exists and is readable." if present
              else observation.get("error", "File missing."))
        if not present:
            continue
        artifact = observation["result"]
        content = artifact["content"]
        artifacts.append({k: artifact[k] for k in ("path", "sha256", "bytes_read") if k in artifact})
        check(f"nonempty:{expected.path}", bool(content.strip()), "Artifact must contain text.")
        headings = {re.sub(r"\s+#+\s*$", "", line).strip().casefold()
                    for line in re.findall(r"^\s{0,3}#{1,6}\s+(.+)$", content, re.MULTILINE)}
        for section in expected.required_sections:
            check(f"section:{expected.path}:{section}", section.strip().casefold() in headings,
                  f"Required Markdown heading: {section}")
        cited = {canonical_url(url) for url in re.findall(r'https?://[^\s<>"\]\)]+', content)}
        grounded = cited & sources
        min_citations = max(expected.min_sources, int(research))
        check(f"citations:{expected.path}", len(grounded) >= min_citations,
              f"Found {len(grounded)} retrieved source URLs; required {min_citations}.")
        if research or expected.min_sources:
            check(f"citation_provenance:{expected.path}", not cited - sources,
                  "All cited URLs were retrieved." if not cited - sources
                  else "Artifact contains URLs that were not retrieved in this run.")
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "artifacts": artifacts, "source_urls": sorted(sources),
            "scope": "Checks structure and citation provenance; does not prove factual correctness."}
