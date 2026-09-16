"""Independent API/concurrency regressions; only synthetic, in-process models."""

import asyncio
import threading
from types import SimpleNamespace
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.translation.backend import TranslationResult


class ControlledBackend:
    def __init__(self):
        self.calls = []
        self.counts = {}
        self.started = {}
        self.held = set()
        self.gates = {}
        self.fixed_result = None
        self.closed = False

    def hold_first(self, source):
        self.held.add(source)
        self.started[source] = threading.Event()

    async def translate(self, source, source_language, target_language,
                        source_context=(), translation_context=(), glossary=None, **kwargs):
        number = self.counts[source] = self.counts.get(source, 0) + 1
        self.calls.append((source, tuple(source_context), tuple(translation_context), number))
        if source in self.held and number == 1:
            gate = self.gates[source] = asyncio.Event()
            self.started[source].set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                # Exercise the application-side late-result fence too.
                await gate.wait()
        text = self.fixed_result if self.fixed_result is not None else f"translated-{number}: {source}"
        return TranslationResult(text, 1)

    def release(self, source):
        self.gates[source].set()

    def release_all(self):
        for gate in self.gates.values():
            gate.set()

    async def close(self):
        self.closed = True


class ApiReviewTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None, streaming_api_key="review-test-" + "x" * 32,
                                 translation_commit_delay_ms=0, translation_debounce_ms=0)
        self.backend = ControlledBackend()
        self.app = create_app(self.settings, backend=self.backend)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.admin = {"Authorization": "Bearer " + self.settings.streaming_api_key.get_secret_value()}

    def tearDown(self):
        self.client.portal.call(self.backend.release_all)
        self.client.__exit__(None, None, None)

    def create(self):
        response = self.client.post("/sessions", headers=self.admin, json={})
        self.assertEqual(response.status_code, 201)
        data = response.json()
        return data, {"Authorization": "Bearer " + data["session_token"]}

    def send(self, websocket, sequence, text, utterance):
        websocket.send_json({"type": "asr_final", "sequence": sequence, "text": text,
                             "utterance_id": utterance})

    def until(self, websocket, predicate):
        for _ in range(20):
            event = websocket.receive_json()
            if predicate(event):
                return event
        self.fail("Expected synthetic pipeline event was not produced")

    def stored(self, session_id):
        return self.client.portal.call(self.app.state.manager.sessions[session_id].snapshot)

    def test_binary_browser_auth_frame_is_rejected_with_protocol_close_not_server_error(self):
        data, _ = self.create()
        with self.client.websocket_connect(data["websocket_path"]) as websocket:
            websocket.send_bytes(b"synthetic-invalid-auth")
            close = websocket.receive()
            self.assertEqual(close["type"], "websocket.close")
            self.assertEqual(close["code"], 1008)

    def test_snapshot_overflow_closes_explicitly_and_preserves_completed_result(self):
        # Narrow this deployment guard after validated construction to exercise
        # overflow with a small synthetic result, not a huge memory allocation.
        self.settings.max_snapshot_bytes = 1024
        self.backend.fixed_result = "번역" * 200
        data, headers = self.create()
        with self.client.websocket_connect(data["websocket_path"], headers=headers) as websocket:
            websocket.receive_json()
            self.send(websocket, 0, "Please stop.", "first")
            event = self.until(websocket, lambda item: item.get("code") == "SNAPSHOT_LIMIT")
            self.assertFalse(event["recoverable"])
            self.assertEqual(websocket.receive()["code"], 1009)
        retained = self.stored(data["session_id"])
        self.assertEqual(retained["full_text"], self.backend.fixed_result)
        self.assertEqual(retained["segments"][0]["status"], "complete")
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_reset_during_resistant_model_discards_old_generation_and_keeps_sequence(self):
        old, new = "We arrived yesterday.", "We left today."
        self.backend.hold_first(old)
        data, headers = self.create()
        with self.client.websocket_connect(data["websocket_path"], headers=headers) as websocket:
            initial = websocket.receive_json()
            self.send(websocket, 0, old, "old")
            self.assertTrue(self.backend.started[old].wait(2))
            websocket.send_json({"type": "reset_context"})
            reset = self.until(websocket, lambda item: item.get("generation_id", -1) > initial["generation_id"])
            self.assertEqual(reset["full_text"], "")
            self.assertEqual(reset["last_sequence"], 0)
            self.send(websocket, 1, new, "new")
            self.client.portal.call(self.backend.release, old)
            final = self.until(websocket, lambda item: item.get("final") is True)
            self.assertIn(new, final["committed"])
            self.assertNotIn(old, final["full_text"])
            self.assertEqual([call[0] for call in self.backend.calls], [old, new])

    def test_disconnect_reconnect_keeps_completed_history_and_retries_only_pending_work(self):
        complete, pending = "We arrived yesterday.", "We left today."
        self.backend.hold_first(pending)
        data, headers = self.create()
        with self.client.websocket_connect(data["websocket_path"], headers=headers) as websocket:
            initial = websocket.receive_json()
            self.send(websocket, 0, complete, "first")
            previous = self.until(websocket, lambda item: item.get("final") is True)["committed"]
            self.send(websocket, 1, pending, "second")
            self.assertTrue(self.backend.started[pending].wait(2))
        retained = self.stored(data["session_id"])
        self.assertEqual(retained["committed"], previous)
        self.assertGreater(retained["generation_id"], initial["generation_id"])
        with self.client.websocket_connect(data["websocket_path"], headers=headers) as websocket:
            restored = websocket.receive_json()
            self.assertEqual(restored["committed"], previous)
            self.client.portal.call(self.backend.release, pending)
            final = self.until(websocket, lambda item: item.get("final") is True)
            self.assertTrue(final["committed"].startswith(previous))
            self.assertIn("translated-2: " + pending, final["committed"])
            self.assertNotIn("translated-1: " + pending, final["full_text"])
        self.assertEqual(self.backend.counts[complete], 1)
        self.assertEqual(self.backend.counts[pending], 2)

    def test_callback_already_waiting_at_disconnect_cannot_publish_after_detach(self):
        source = "We arrived yesterday."
        self.backend.hold_first(source)
        data, headers = self.create()
        session = self.app.state.manager.sessions[data["session_id"]]
        with self.client.websocket_connect(data["websocket_path"], headers=headers) as websocket:
            websocket.receive_json()
            self.send(websocket, 0, source, "first")
            self.assertTrue(self.backend.started[source].wait(2))
            before = self.stored(data["session_id"])
            segment = before["segments"][0]
            late_job = SimpleNamespace(generation_id=before["generation_id"],
                                       segment_id=segment["segment_id"], source_revision=segment["source_revision"])
        # Models/scheduler cannot revoke a callback already waiting on the state
        # lock. Replay that arrival after detach; the state's generation fences it.
        self.client.portal.call(session._result, late_job, TranslationResult("STALE-CALLBACK", 1), None)
        after = self.stored(data["session_id"])
        self.assertEqual(after["full_text"], "")
        self.assertEqual(after["segments"][0]["status"], "ready")
        self.assertGreater(after["generation_id"], before["generation_id"])

    def test_same_session_second_request_waits_and_receives_first_translation_context(self):
        first, second = "We arrived yesterday.", "We left today."
        self.backend.hold_first(first)
        data, headers = self.create()
        with self.client.websocket_connect(data["websocket_path"], headers=headers) as websocket:
            websocket.receive_json()
            self.send(websocket, 0, first, "first")
            self.assertTrue(self.backend.started[first].wait(2))
            self.send(websocket, 1, second, "second")
            # A ping round-trip proves the second ASR event was processed while
            # model one is still outstanding; no second request may start yet.
            websocket.send_json({"type": "ping"})
            self.until(websocket, lambda item: item.get("type") == "pong")
            self.assertEqual([call[0] for call in self.backend.calls], [first])
            self.client.portal.call(self.backend.release, first)
            self.until(websocket, lambda item: item.get("final") is True)
        self.assertEqual([call[0] for call in self.backend.calls], [first, second])
        self.assertEqual(self.backend.calls[1][1], (first,))
        self.assertEqual(self.backend.calls[1][2], ("translated-1: " + first,))

    def test_overflow_marker_remains_first_and_retained_state_is_not_discarded(self):
        self.settings.outgoing_queue_size = 2
        data, _ = self.create()
        session = self.app.state.manager.sessions[data["session_id"]]

        async def overflow():
            queue = await session.attach()
            queue.get_nowait()
            for index in range(15):
                session.emit({"type": "synthetic", "index": index})
            self.assertLessEqual(queue.qsize(), 2)
            self.assertIsNone(queue.get_nowait())
            await session.detach()

        self.client.portal.call(overflow)
        self.assertIn(data["session_id"], self.app.state.manager.sessions)


if __name__ == "__main__":
    unittest.main()
