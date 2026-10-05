"""Integration tests for the local UI service using the real offline agent."""

from __future__ import annotations

import json
import http.client
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from goal_agent.demo import DEMO_GOAL
from goal_agent.models import AgentError
from goal_agent.service import RunService, new_run_dir
from goal_agent.storage import run_lock
from goal_agent.web import create_server


class RunServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "runs"
        self.service = RunService(self.root)
        self.pricing = patch.dict(os.environ, {"INPUT_COST_PER_MILLION": "", "OUTPUT_COST_PER_MILLION": ""})
        self.pricing.start()
        self.addCleanup(self.pricing.stop)

    def settled(self, run_id, service=None):
        service = service or self.service
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with service._lock:
                active = run_id in service._jobs
            if not active:
                return service.get_run(run_id)
            time.sleep(0.01)
        self.fail(f"Local run {run_id} did not finish within ten seconds.")

    def demo(self, **kwargs):
        run_id = self.service.start("This text is replaced by the fixed offline goal.", demo=True, **kwargs)
        return run_id, self.settled(run_id)

    def test_demo_start_verifies_and_exposes_only_safe_state(self):
        run_id, state = self.demo()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["goal"], DEMO_GOAL)
        self.assertEqual(state["run_id"], run_id)
        self.assertTrue(state["verification"]["passed"])
        self.assertEqual(len(state["artifacts"]), 1)
        self.assertIn("metrics", state)
        for private in ("history", "pending_call", "pending_request", "recovery_history"):
            self.assertNotIn(private, state)
        content, content_type = self.service.artifact(run_id, "pathlib-note.md")
        self.assertIn(b"# pathlib note", content)
        self.assertIn(b"https://docs.python.org/3/library/pathlib.html", content)
        self.assertEqual(content_type, "application/octet-stream")
        persisted = RunService(self.root).get_run(run_id)
        self.assertEqual(persisted["status"], "completed")
        self.assertEqual(self.service.list_runs()[0]["run_id"], run_id)

    def test_execute_uses_exact_saved_plan_without_replanning(self):
        run_id, planned = self.demo(plan_only=True)
        self.assertEqual(planned["status"], "planned")
        self.assertEqual(planned["metrics"]["model_requests"], 1)
        self.assertFalse((self.root / run_id / "artifacts" / "pathlib-note.md").exists())
        saved_plan = (self.root / run_id / "plan.json").read_bytes()
        self.assertEqual(self.service.execute(run_id), run_id)
        completed = self.settled(run_id)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["plan"], planned["plan"])
        self.assertEqual((self.root / run_id / "plan.json").read_bytes(), saved_plan)
        self.assertEqual(completed["metrics"]["model_requests"], 5)
        with self.assertRaises(ValueError):
            self.service.execute(run_id)

    def test_cancel_after_file_write_and_resume_without_recreating_file(self):
        # The actual runner's event hook lets cancellation happen at a precise
        # checkpoint, with no mocked model, tools, filesystem or execution loop.
        run_dir = new_run_dir(self.root)
        cancellations = []

        def on_event(message):
            if "create_file: ok" in message:
                cancellations.append(self.service.cancel(run_dir.name))

        run_id = self.service._launch(run_dir, goal=DEMO_GOAL, demo=True, on_event=on_event)
        cancelled = self.settled(run_id)
        self.assertEqual(cancellations, [True])
        self.assertEqual(cancelled["status"], "cancelled")
        output = run_dir / "artifacts" / "pathlib-note.md"
        before = output.read_bytes(), output.stat().st_mtime_ns
        self.assertIn("pathlib-note.md", {a["path"] for a in cancelled["artifacts"]})
        self.assertFalse(self.service.cancel(run_id))
        restarted = RunService(self.root)
        self.assertEqual(restarted.resume(run_id), run_id)
        completed = self.settled(run_id, restarted)
        self.assertEqual(completed["status"], "completed")
        self.assertTrue(completed["verification"]["passed"])
        self.assertEqual((output.read_bytes(), output.stat().st_mtime_ns), before)
        writes = [event for step in completed["steps"] for event in step["events"]
                  if event.get("tool") == "create_file" and event.get("ok")]
        self.assertEqual(len(writes), 1)

    def test_completed_resume_does_not_repeat_model_calls_or_file_creation(self):
        run_id, original = self.demo()
        output = self.root / run_id / "artifacts" / "pathlib-note.md"
        modified = output.stat().st_mtime_ns
        self.service.resume(run_id)
        resumed = self.settled(run_id)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["metrics"]["model_requests"], original["metrics"]["model_requests"])
        self.assertEqual(output.stat().st_mtime_ns, modified)

    def test_runs_use_separate_workspaces(self):
        first, _ = self.demo()
        second, _ = self.demo()
        self.assertNotEqual(first, second)
        first_file = self.root / first / "artifacts" / "pathlib-note.md"
        second_file = self.root / second / "artifacts" / "pathlib-note.md"
        self.assertNotEqual(first_file.resolve(), second_file.resolve())
        first_file.write_text("Changed first run only.", encoding="utf-8")
        self.assertIn("# pathlib note", second_file.read_text(encoding="utf-8"))
        self.assertEqual({r["run_id"] for r in self.service.list_runs()}, {first, second})

    def test_recorded_artifact_digest_prevents_stale_verified_download(self):
        run_id, _ = self.demo()
        output = self.root / run_id / "artifacts" / "pathlib-note.md"
        output.write_text("The artifact was replaced after verification.", encoding="utf-8")
        with self.assertRaises((ValueError, FileNotFoundError, AgentError)):
            self.service.artifact(run_id, "pathlib-note.md")

    def test_unlisted_files_and_path_attacks_are_not_downloadable(self):
        run_id, _ = self.demo()
        (self.root / run_id / "artifacts" / ".env").write_text("FAKE_TEST_KEY=fixture", encoding="utf-8")
        for path in (".env", "run.json", "plan.json", "../run.json", "/etc/passwd", "C:\\fake-secret", "\\\\host\\share"):
            with self.subTest(path=path), self.assertRaises((ValueError, FileNotFoundError)):
                self.service.artifact(run_id, path)
        for invalid_id in ("..", "../run", "C:\\", "20260101T000000Z-deadbeef/..", ""):
            with self.subTest(run_id=invalid_id), self.assertRaises(ValueError):
                self.service.get_run(invalid_id)

    def test_missing_key_is_rejected_before_a_nonresumable_run_is_created(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaises((AgentError, ValueError)):
                self.service.start("Create a short local note.")
        self.assertEqual(self.service.list_runs(), [])

    def test_saved_run_quota_survives_service_restart(self):
        limited = RunService(self.root, max_runs=1)
        run_id = limited.start("fixed demo", demo=True)
        self.settled(run_id, limited)
        with self.assertRaisesRegex(ValueError, "saved-run limit"):
            RunService(self.root, max_runs=1).start("another", demo=True)

    def test_active_run_limit_and_cooperative_cancel(self):
        entered, release = threading.Event(), threading.Event()

        def wait_for_cancel(run_dir, **kwargs):
            entered.set()
            release.wait(timeout=5)
            return {"run_id": run_dir.name, "goal": kwargs["goal"],
                    "status": "cancelled" if kwargs["cancel"]() else "completed"}

        with patch("goal_agent.service.run_once", side_effect=wait_for_cancel):
            run_id = self.service.start("fixed demo", demo=True)
            try:
                self.assertTrue(entered.wait(timeout=3))
                self.assertEqual(self.service.get_run(run_id)["status"], "queued")
                with self.assertRaisesRegex(ValueError, "concurrency limit"):
                    self.service.start("second run", demo=True)
                with self.assertRaisesRegex(ValueError, "already active"):
                    self.service.resume(run_id)
                self.assertTrue(self.service.cancel(run_id))
            finally:
                release.set()
            self.assertEqual(self.settled(run_id)["status"], "cancelled")

    def test_simultaneous_starts_enforce_saved_run_quota(self):
        class SchedulingService(RunService):
            def _launch(inner, run_dir, **kwargs):
                # Expose the scheduling gap between the quota check and queue
                # registration. A correct check holds its lock across both.
                time.sleep(0.025)
                return super()._launch(run_dir, **kwargs)

        service = SchedulingService(self.root, max_active_runs=2, max_runs=1)
        barrier, release = threading.Barrier(3), threading.Event()
        accepted, rejected = [], []

        def slow_run(run_dir, **kwargs):
            release.wait(timeout=5)
            return {"run_id": run_dir.name, "goal": kwargs["goal"], "status": "completed"}

        def start():
            barrier.wait(timeout=3)
            try:
                accepted.append(service.start("demo", demo=True))
            except ValueError as exc:
                rejected.append(str(exc))

        with patch("goal_agent.service.run_once", side_effect=slow_run):
            threads = [threading.Thread(target=start) for _ in range(2)]
            try:
                for thread in threads:
                    thread.start()
                barrier.wait(timeout=3)
                for thread in threads:
                    thread.join(timeout=3)
                self.assertFalse(any(thread.is_alive() for thread in threads))
            finally:
                release.set()
            for run_id in accepted:
                self.settled(run_id, service)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn("saved-run limit", rejected[0])

    def test_execution_lock_failure_is_visible_even_with_existing_record(self):
        run_id, _ = self.demo(plan_only=True)
        with run_lock(self.root / run_id):
            self.service.execute(run_id)
            state = self.settled(run_id)
        self.assertEqual(state["status"], "failed")
        self.assertIn("already executing", state["error"])
        self.service.resume(run_id)
        self.assertEqual(self.settled(run_id)["status"], "completed")

    def test_malformed_saved_record_does_not_break_healthy_run_history(self):
        run_id, _ = self.demo()
        bad_dir = self.root / "20260101T000000Z-deadbeef"
        bad_dir.mkdir()
        (bad_dir / "run.json").write_text('{"unexpected":"fixture"}', encoding="utf-8")
        self.assertEqual([run["run_id"] for run in self.service.list_runs()], [run_id])

    def test_orphaned_active_run_is_shown_as_interrupted(self):
        run_id, _ = self.demo(plan_only=True)
        path = self.root / run_id / "run.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["status"] = "running"
        path.write_text(json.dumps(record), encoding="utf-8")
        reopened = RunService(self.root)
        self.assertEqual(reopened.get_run(run_id)["status"], "interrupted")
        reopened.resume(run_id)
        self.assertEqual(self.settled(run_id, reopened)["status"], "completed")

    def test_live_polling_does_not_interrupt_atomic_checkpoint_writes(self):
        run_id = self.service.start("fixed demo", demo=True)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with self.service._lock:
                active = run_id in self.service._jobs
            state = self.service.get_run(run_id)
            if not active:
                break
            time.sleep(0.001)
        else:
            self.fail("The polled demo did not complete in time.")
        self.assertEqual(state["status"], "completed", self.service._pending.get(run_id))
        self.assertTrue(state["verification"]["passed"])

    def test_authenticated_http_plan_execute_and_artifact_with_real_service(self):
        server = create_server(self.service, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, path, body=None):
            connection = http.client.HTTPConnection(*server.server_address[:2], timeout=3)
            headers = {"Authorization": "Bearer " + server.auth_token, "Origin": server.origin}
            if body is not None:
                headers["Content-Type"] = "application/json"
                body = json.dumps(body).encode("utf-8")
            try:
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        try:
            status, body = request("POST", "/api/runs", {"goal": "Run the offline demo", "demo": True, "plan_only": True})
            self.assertEqual(status, 202)
            run_id = json.loads(body)["run_id"]
            self.assertEqual(self.settled(run_id)["status"], "planned")
            self.assertEqual(json.loads(request("GET", f"/api/runs/{run_id}")[1])["status"], "planned")
            self.assertEqual(request("POST", f"/api/runs/{run_id}/execute", {})[0], 202)
            self.assertEqual(self.settled(run_id)["status"], "completed")
            final = json.loads(request("GET", f"/api/runs/{run_id}")[1])
            self.assertTrue(final["verification"]["passed"])
            status, artifact = request("GET", f"/api/runs/{run_id}/artifacts?path=pathlib-note.md")
            self.assertEqual(status, 200)
            self.assertIn(b"# pathlib note", artifact)
            self.assertEqual(json.loads(request("GET", "/api/runs")[1])["runs"][0]["run_id"], run_id)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def symlink(self, link: Path, target: Path, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Creating symlinks is unavailable on this platform: {exc}")

    def test_run_directory_symlink_cannot_escape_root(self):
        external = self.base / "external"
        external.mkdir()
        run_id = "20260101T000000Z-deadbeef"
        self.symlink(self.root / run_id, external, directory=True)
        with self.assertRaises((ValueError, OSError, AgentError)):
            self.service.get_run(run_id)

    def test_run_record_symlink_cannot_read_external_json(self):
        run_id, _ = self.demo()
        path = self.root / run_id / "run.json"
        external = self.base / "outside-record.json"
        external.write_bytes(path.read_bytes())
        path.unlink()
        self.symlink(path, external)
        with self.assertRaises((ValueError, OSError, AgentError)):
            self.service.get_run(run_id)

    def test_artifact_root_symlink_cannot_expose_another_workspace(self):
        run_id, _ = self.demo()
        path = self.root / run_id / "artifacts"
        external = self.base / "outside-artifacts"
        path.rename(external)
        # Identical bytes ensure this tests path confinement, not the SHA guard.
        self.symlink(path, external, directory=True)
        with self.assertRaises((ValueError, OSError, AgentError)):
            self.service.artifact(run_id, "pathlib-note.md")


if __name__ == "__main__":
    unittest.main()
