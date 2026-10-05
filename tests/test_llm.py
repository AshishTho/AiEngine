"""Exercise the real OpenAI SDK against an in-memory HTTP transport only."""

from __future__ import annotations

import json
import unittest

import httpx
from openai import OpenAI

from goal_agent.llm import OpenAIBackend
from goal_agent.models import AgentError, Plan
from goal_agent.runner import FINISH_SCHEMA


def response_body(output: list[dict], status: str = "completed") -> dict:
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
        document = {"steps": [{"description": "Think through the answer", "tool": "reason"}]}
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
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["required"], ["steps"])
        self.assertFalse(schema["$defs"]["Step"]["additionalProperties"])
        self.assertEqual(set(schema["$defs"]["Step"]["required"]), {"description", "tool"})

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
