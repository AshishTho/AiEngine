"""A fixed, clearly labeled offline demonstration of the real execution loop."""

import json

from .models import Plan, Step
from .tools import ToolRegistry


DEMO_GOAL = "Research Python's pathlib module and save a short note as pathlib-note.md."


class DemoBackend:
    def __init__(self):
        self.turn = 0

    def plan(self, goal: str, max_steps: int, tools: list[dict]) -> Plan:
        return Plan(steps=[
            Step(description="Find the Python pathlib documentation.", tool="web_search"),
            Step(description="Save a brief note with a source link.", tool="create_file"),
        ])

    def respond(self, history: list[dict], tools: list[dict]) -> list[dict]:
        # Consume the preceding search observation, just as an LLM would.
        source_url = "https://docs.python.org/3/library/pathlib.html"
        for message in history:
            if message.get("type") == "function_call_output":
                observation = json.loads(message["output"])
                sources = observation.get("result", {}).get("sources", [])
                if sources:
                    source_url = sources[0]["url"]
        script = [
            ("web_search", {"query": "Python pathlib documentation", "max_results": 1}),
            ("finish_step", {"status": "completed", "summary": "Found the demo source."}),
            ("create_file", {"path": "pathlib-note.md", "content": (
                "# pathlib note\n\n"
                "This file was produced by the offline demo using canned search data.\n\n"
                "Python's pathlib module represents filesystem paths as objects. "
                "Path objects support joining paths and reading or writing files.\n\n"
                f"Source: [Python documentation]({source_url})\n"
            )}),
            ("finish_step", {"status": "completed", "summary": "Created pathlib-note.md."}),
        ]
        name, arguments = script[self.turn]
        self.turn += 1
        return [{"type": "function_call", "name": name,
                 "call_id": f"demo_{self.turn}", "arguments": json.dumps(arguments)}]


class DemoTools(ToolRegistry):
    def execute(self, name: str, arguments_json: str) -> dict:
        if name == "web_search":
            return {"ok": True, "result": {
                "query": "Python pathlib documentation", "sources": [{
                    "title": "pathlib - Object-oriented filesystem paths",
                    "url": "https://docs.python.org/3/library/pathlib.html",
                    "snippet": "OFFLINE FIXTURE: pathlib provides object-oriented filesystem paths.",
                }],
            }}
        return super().execute(name, arguments_json)
