"""Exercise the real OpenAI SDK against an in-memory HTTP transport only."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx
from openai import OpenAI

from goal_agent.llm import OpenAIBackend
from goal_agent.models import AgentError, Plan, Pricing
from goal_agent.runner import Agent, FINISH_SCHEMA
from goal_agent.tools import ToolRegistry


def response_body(output: list[dict], status: str = "completed", usage: dict | None = None) -> dict:
    """A minimal Responses API response that the actual SDK deserializes."""
    return {
        "id": "resp_offline",
        "object": "response",
        "created_at": 1750000000,
        "model": "offline-model",
        "status": status,
        "error": None,
        "incomplete_details": ({"reason": "max_output_tokens"}
                               if status == "incomplete" else None),
        "instructions": None,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "temperature": 1.0,
        "top_p": 1.0,
        "metadata": {},
        "usage": usage,
    }


def text_message(text: str) -> dict:
    return {
        "type": "message", "id": "msg_offline", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class OpenAIBackendTests(unittest.TestCase):
    def backend(self, bodies: list[dict], status_code: int = 200):
        self.requests: list[dict] = []

        def handle(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/v1/responses")
            self.requests.append(json.loads(request.content))
            self.assertTrue(bodies, "Unexpected extra SDK request")
            return httpx.Response(status_code, json=bodies.pop(0))

        client = OpenAI(
            api_key="offline-placeholder",
            base_url="https://offline.invalid/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
        self.addCleanup(client.close)
        return OpenAIBackend(client, "offline-model", max_output_tokens=1234)

    def test_plan_uses_strict_structured_output_and_parses_validated_plan(self):
        document = {
            "steps": [{"description": "Think through the answer", "tool": "reason"}],
            "supported": True, "unsupported_reason": "",
            "expected_artifacts": [], "minimum_sources": 0,
        }
        backend = self.backend([response_body([text_message(json.dumps(document))])])

        plan = backend.plan("Explain a concept", 3, [FINISH_SCHEMA])

        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.model_dump(), document)
        request = self.requests[0]
        self.assertEqual(request["model"], "offline-model")
        self.assertEqual(request["max_output_tokens"], 1234)
        self.assertFalse(request["store"])
        self.assertEqual(request["input"], "Explain a concept")
        self.assertIn("1 to 3", request["instructions"])
        output_format = request["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        schema = output_format["schema"]
        self.assertNotIn('"default":', json.dumps(schema))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), {
            "steps", "supported", "unsupported_reason", "expected_artifacts", "minimum_sources",
        })
        self.assertFalse(schema["$defs"]["Step"]["additionalProperties"])
        self.assertEqual(set(schema["$defs"]["Step"]["required"]), {"description", "tool"})
        self.assertEqual(set(schema["$defs"]["Step"]["properties"]["tool"]["enum"]),
                         {"web_search", "fetch_page", "create_file", "read_file", "reason"})
        self.assertFalse(schema["$defs"]["ArtifactExpectation"]["additionalProperties"])
        self.assertEqual(set(schema["$defs"]["ArtifactExpectation"]["required"]),
                         {"path", "required_sections", "min_sources"})

    def test_executor_preserves_reasoning_function_calls_and_tool_results(self):
        reasoning = {
            "type": "reasoning", "id": "rs_offline", "summary": [],
            "encrypted_content": "opaque-reasoning-state",
        }
        call = {
            "type": "function_call", "id": "fc_offline", "call_id": "call_offline",
            "name": "finish_step", "arguments": '{"status":"completed","summary":"Done"}',
            "status": "completed",
        }
        message = text_message("Current step is complete.")
        backend = self.backend([
            response_body([reasoning, call, message]),
            response_body([call]),
        ])
        history = [{"role": "user", "content": "Complete the current step"}]

        output = backend.respond(history, [FINISH_SCHEMA])
        self.assertEqual(output, [reasoning, call, message])
        tool_result = {
            "type": "function_call_output", "call_id": "call_offline",
            "output": '{"ok": true}',
        }
        continued_history = history + output + [tool_result]
        backend.respond(continued_history, [FINISH_SCHEMA])

        for request in self.requests:
            self.assertFalse(request["store"])
            self.assertFalse(request["parallel_tool_calls"])
            self.assertEqual(request["tool_choice"], "required")
            self.assertEqual(request["include"], ["reasoning.encrypted_content"])
            self.assertEqual(request["tools"], [FINISH_SCHEMA])
        self.assertEqual(self.requests[1]["input"], continued_history)

    def test_invalid_structured_plan_is_rejected_without_echoing_output(self):
        secret = "private-response-body-do-not-print"
        backend = self.backend([response_body([text_message(json.dumps({
            "steps": [{"description": secret, "tool": "run_shell"}],
        }))])])

        with self.assertRaises(AgentError) as caught:
            backend.plan("Do something", 3, [])

        self.assertIn("valid structured response", str(caught.exception))
        self.assertNotIn(secret, str(caught.exception))

    def test_missing_plan_is_rejected(self):
        backend = self.backend([response_body([])])
        with self.assertRaisesRegex(AgentError, "valid plan"):
            backend.plan("Do something", 3, [])

    def test_incomplete_response_does_not_execute_partial_calls(self):
        backend = self.backend([response_body([], status="incomplete")])
        with self.assertRaisesRegex(AgentError, "incomplete"):
            backend.respond([], [FINISH_SCHEMA])

    def test_usage_is_captured_per_response_and_missing_usage_does_not_reuse_previous(self):
        usage = {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17,
                 "input_tokens_details": {"cached_tokens": 0},
                 "output_tokens_details": {"reasoning_tokens": 0}}
        backend = self.backend([response_body([], usage=usage), response_body([])])
        backend.respond([], [FINISH_SCHEMA])
        self.assertEqual(backend.last_usage, {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17})
        backend.respond([], [FINISH_SCHEMA])
        self.assertIsNone(backend.last_usage)

    def test_incomplete_response_usage_is_retained_in_run_cost_without_executing_partial_call(self):
        plan = Plan.model_validate({"steps": [{"description": "Explain the concept", "tool": "reason"}]})
        partial_call = {"type": "function_call", "id": "fc_partial", "call_id": "partial",
                        "name": "finish_step", "arguments": '{"status":"completed","summary":"Done"}',
                        "status": "completed"}
        plan_usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}}
        partial_usage = {"input_tokens": 7, "output_tokens": 9, "total_tokens": 16,
                         "input_tokens_details": {"cached_tokens": 0},
                         "output_tokens_details": {"reasoning_tokens": 0}}
        backend = self.backend([
            response_body([text_message(plan.model_dump_json())], usage=plan_usage),
            response_body([partial_call], status="incomplete", usage=partial_usage),
        ])
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "run"
            state = Agent(backend, ToolRegistry(run_dir / "artifacts", None), run_dir,
                          pricing=Pricing(1.0, 2.0)).run("Explain the concept")
        self.assertEqual(state["status"], "failed")
        self.assertIn("incomplete", state["error"])
        self.assertEqual(state["steps"][0]["events"], [])
        self.assertEqual(state["tool_calls"], 0)
        self.assertEqual(state["metrics"]["model_requests"], 2)
        self.assertEqual(state["metrics"]["input_tokens"], 17)
        self.assertEqual(state["metrics"]["output_tokens"], 29)
        self.assertEqual(state["metrics"]["total_tokens"], 46)
        self.assertTrue(state["metrics"]["usage_complete"])
        self.assertAlmostEqual(state["metrics"]["estimated_cost_usd"], 75 / 1_000_000)

    def test_refusal_is_rejected_for_both_planning_and_execution(self):
        refusal = {
            "type": "message", "id": "msg_refusal", "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "private refusal detail"}],
        }
        for operation in ("plan", "respond"):
            with self.subTest(operation=operation):
                backend = self.backend([response_body([refusal])])
                with self.assertRaisesRegex(AgentError, "declined this request"):
                    if operation == "plan":
                        backend.plan("Goal", 3, [])
                    else:
                        backend.respond([], [FINISH_SCHEMA])

    def test_http_errors_do_not_expose_provider_response_or_credentials(self):
        for status in (401, 429, 500):
            with self.subTest(status=status):
                backend = self.backend([{
                    "error": {
                        "message": "Secret credential: offline-placeholder",
                        "type": "provider_error", "code": "secret-error-code",
                    },
                }], status_code=status)
                with self.assertRaises(AgentError) as caught:
                    backend.respond([], [FINISH_SCHEMA])
                message = str(caught.exception)
                self.assertIn(f"HTTP {status}", message)
                self.assertNotIn("offline-placeholder", message)
                self.assertNotIn("secret-error-code", message)
                self.assertEqual(len(self.requests), 1)

    def test_transport_timeout_is_sanitized(self):
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("private transport detail", request=request)

        client = OpenAI(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(timeout)),
        )
        self.addCleanup(client.close)
        backend = OpenAIBackend(client, "offline-model")
        with self.assertRaisesRegex(AgentError, "connection failed or timed out") as caught:
            backend.respond([], [FINISH_SCHEMA])
        self.assertNotIn("private transport detail", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
