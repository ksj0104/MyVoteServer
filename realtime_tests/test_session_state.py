import asyncio
import time
import unittest
from app.core.config import Settings
from app.core.models import ASREvent, SessionConfig
from app.metrics.collector import Metrics
from app.session.manager import SessionManager
from app.translation.backend import TranslationError, TranslationResult
from app.translation.scheduler import TranslationScheduler


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.delay = 0
        self.fail = False

    async def translate(self, source, source_language, target_language, source_context, translation_context, glossary, **kwargs):
        self.calls.append((source, source_context, translation_context, dict(glossary)))
        await asyncio.sleep(self.delay)
        if self.fail:
            raise TranslationError("timeout")
        return TranslationResult("번역:" + source.strip(), 1)

    async def close(self):
        pass


async def eventually(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = Settings(_env_file=None, streaming_api_key="x" * 32, translation_commit_delay_ms=10,
                                 translation_debounce_ms=10)
        self.backend = FakeBackend()
        self.metrics = Metrics()
        self.scheduler = TranslationScheduler(self.backend, workers=4, max_pending=32, metrics=self.metrics)
        await self.scheduler.start()
        self.manager = SessionManager(self.settings, self.scheduler, self.metrics)
        self.session, self.token = await self.manager.create(SessionConfig())
        self.queue = await self.session.attach()

    async def asyncTearDown(self):
        await self.manager.close()
        await self.scheduler.close()

    async def send(self, sequence, text, *, final=True, utterance_id="a", speaker_id=None):
        await self.session.ingest(ASREvent(type="asr_final" if final else "asr_partial", sequence=sequence,
                                          text=text, utterance_id=utterance_id, speaker_id=speaker_id))

    async def test_final_passes_pipeline_and_commits(self):
        await self.send(0, "I want to go to the hospital today.")
        await eventually(lambda: self.session.snapshot()["final"])
        state = self.session.snapshot()
        self.assertEqual(len(self.backend.calls), 1)
        self.assertIn("hospital", state["committed"])
        self.assertEqual(state["draft"], "")

    async def test_incomplete_final_is_source_only_not_forced(self):
        await self.send(1, "I think that")
        await eventually(lambda: self.session.utterances["a"].incomplete_reported)
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.session.snapshot()["committed_source"], "I think that")
        self.assertFalse(self.session.snapshot()["final"])

    async def test_correction_keeps_prefix_and_retranslates_dependent_suffix(self):
        await self.send(1, "We arrived yesterday.", utterance_id="a")
        await self.send(2, "I went to the hospital yesterday.", utterance_id="b")
        await self.send(3, "It was very busy.", utterance_id="c")
        await eventually(lambda: self.session.snapshot()["final"])
        first = self.session.segments[0]
        await self.send(4, "I went to the hotel yesterday.", utterance_id="b")
        await eventually(lambda: self.session.snapshot()["final"] and len(self.backend.calls) >= 5)
        self.assertIs(self.session.segments[0], first)
        self.assertEqual([s.utterance_id for s in self.session.segments], ["a", "b", "c"])
        self.assertNotIn("hospital", self.session.snapshot()["full_text"])
        self.assertIn("hotel", self.backend.calls[-1][1][-1])

    async def test_reset_rejects_old_results_and_preserves_sequence(self):
        self.backend.delay = 0.2
        await self.send(1, "I went to the hospital yesterday.")
        await eventually(lambda: len(self.backend.calls) == 1)
        old_segment = self.session.segments[0]
        old_generation = self.session.generation_id
        await self.session.reset()
        from types import SimpleNamespace
        await self.session._result(SimpleNamespace(segment_id=old_segment.segment_id,
                                   generation_id=old_generation, source_revision=old_segment.source_revision),
                                   TranslationResult("OLD", 1), None)
        self.assertEqual(self.session.snapshot()["full_text"], "")
        self.assertEqual(self.session.last_sequence, 1)
        with self.assertRaisesRegex(ValueError, "STALE_SEQUENCE"):
            await self.send(1, "Please stop.")

    async def test_partial_conflict_waits_for_explicit_final(self):
        await self.send(1, "I went to the hospital yesterday.")
        await eventually(lambda: self.session.snapshot()["final"])
        with self.assertRaisesRegex(ValueError, "FINAL_REQUIRES_FINAL"):
            await self.send(2, "I went to the hotel yesterday.", final=False)
        self.assertIn("hospital", self.session.snapshot()["committed"])

    async def test_rolling_context_and_glossary_are_session_isolated(self):
        await self.session.set_glossary({"CUDA": "쿠다", "unrelated": "다른 말"})
        await self.send(1, "We compared the two computers.")
        await self.send(2, "CUDA makes this computer faster.", utterance_id="b")
        await eventually(lambda: self.session.snapshot()["final"])
        current = self.backend.calls[-1]
        self.assertIn("We compared", current[1][0])
        self.assertEqual(current[3], {"CUDA": "쿠다"})
        other, _ = await self.manager.create(SessionConfig())
        self.assertEqual(other.snapshot()["full_text"], "")
        self.assertEqual(other.glossary, {})

    async def test_backpressure_retains_finals_in_source_order(self):
        self.backend.delay = 0.02
        self.settings.max_pending_per_session = 2
        for index in range(8):
            await self.send(index, f"The computer works very well today.", utterance_id=f"u{index}")
        await eventually(lambda: self.session.snapshot()["final"])
        self.assertEqual(len(self.backend.calls), 8)
        self.assertEqual([s.utterance_id for s in self.session.segments], [f"u{i}" for i in range(8)])

    async def test_disconnect_and_reconnect_restores_snapshot(self):
        await self.send(1, "Please stop.")
        await eventually(lambda: self.session.snapshot()["final"])
        previous = self.session.snapshot()["committed"]
        await self.session.detach()
        self.queue = await self.session.attach()
        state = await self.queue.get()
        self.assertEqual(state["committed"], previous)
        self.assertEqual(state["last_sequence"], 1)

    async def test_failed_final_can_be_retried_with_a_new_sequence(self):
        self.backend.fail = True
        await self.send(1, "Please stop.")
        await eventually(lambda: self.session.segments and self.session.segments[0].status == "failed")
        self.backend.fail = False
        await self.send(2, "Please stop.")
        await eventually(lambda: self.session.snapshot()["final"])
        self.assertEqual(len(self.backend.calls), 2)

    async def test_limits_auth_cleanup_and_speaker_identity(self):
        self.assertIsNone(self.manager.authorized(self.session.session_id, "wrong-token"))
        self.assertIs(self.manager.authorized(self.session.session_id, self.token), self.session)
        await self.send(1, "Please stop.", speaker_id="speaker:1")
        with self.assertRaisesRegex(ValueError, "SPEAKER_MISMATCH"):
            await self.send(2, "Please stop.", speaker_id="speaker:2")
        with self.assertRaisesRegex(ValueError, "SOURCE_LIMIT"):
            await self.send(3, "a" * 8001)
        self.session.last_activity = time.monotonic() - 1801
        await self.manager.cleanup()
        self.assertTrue(self.session.closed)
        self.assertEqual(self.manager.sessions, {})

    async def test_twenty_partials_at_50ms_do_not_create_twenty_jobs(self):
        for index in range(20):
            await self.send(index, "I want to go to the hospital today.", final=False)
            await asyncio.sleep(0.05)
        await self.send(20, "I want to go to the hospital today.")
        await eventually(lambda: self.session.snapshot()["final"])
        self.assertEqual(len(self.backend.calls), 1)

    async def test_open_partial_token_is_not_committed(self):
        for index in range(5):
            await self.send(index, "I want to go to the hosp", final=False)
        u = self.session.utterances["a"]
        self.assertNotIn("hosp", u.text[:u.stable_end])
        self.assertEqual(self.backend.calls, [])

    async def test_long_complete_sentence_is_not_silently_retired_at_24_tokens(self):
        source = ("The computer works very well when our team runs the complete translation "
                  "pipeline with several independent speakers and a carefully controlled context "
                  "window during the entire demonstration today.")
        await self.send(1, source)
        await eventually(lambda: self.session.snapshot()["final"])
        self.assertEqual(self.backend.calls[0][0], source)

    async def test_unknown_source_language_is_an_explicit_configuration_error(self):
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED_SOURCE_LANGUAGE"):
            await self.manager.create(SessionConfig(source_language="fr"))
