"""Transcription inactivity checkpoints; no live models, network, or server."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
install = runpy.run_path(str(ROOT / "scripts/semantic_transcript_idle.py"))["install_transcript_idle"]


class Builder:
    def __init__(self):
        self._segment_id = "retained"
        self.reasons = []

    def flush(self, *, reason):
        self.reasons.append(reason)
        return [SimpleNamespace(reason=reason, final=True, original_audio=True)]

    def reset(self):
        self._segment_id = None


class BaseSession:
    def __init__(self):
        self._closed = self._finishing = False
        self.builder = Builder()
        self._pending = {}
        self.counts = {}
        self.queued = []
        self.events = []
        self.checkpoint = asyncio.Event()
        self.processing_entered = asyncio.Event()
        self.processing_release = None
        self.fail_emit = False

    async def feed(self, frame, speech_probability):
        self.builder._segment_id = "retained"

    async def _emit(self, kind, segment_id=None, **data):
        if self.fail_emit:
            raise ValueError("synthetic sink failure")
        self.events.append((kind, segment_id, data))

    async def _enqueue(self, window):
        self.queued.append(window)
        self.checkpoint.set()

    async def _process_window(self, window, *, queued_at=None):
        self.processing_entered.set()
        if self.processing_release is not None:
            await self.processing_release.wait()

    async def finish(self):
        self._finishing = True

    async def close(self):
        self._closed = True


class TranscriptIdleTests(unittest.IsolatedAsyncioTestCase):
    async def make(self, interval=.03):
        module = SimpleNamespace(StreamingSession=BaseSession)
        metadata = install(module, inactivity_flush_s=interval)
        session = module.StreamingSession()
        self.addAsyncCleanup(session.close)
        await session.feed(None, 1)
        return module, session, metadata

    async def test_both_inputs_idle_queues_normal_final_once(self):
        _, session, _ = await self.make()
        await session._emit("transcript.updated", "segment", text="provisional words", stable_text="")
        await asyncio.wait_for(session.checkpoint.wait(), .5)
        self.assertEqual(session.builder.reasons, ["transcript_inactivity"])
        self.assertEqual(len(session.queued), 1)
        self.assertTrue(session.queued[0].final)
        self.assertTrue(session.queued[0].original_audio)
        self.assertIsNone(session.builder._segment_id)
        self.assertEqual(session.counts["transcript_idle_checkpoints"], 1)

    async def test_same_text_empty_heartbeat_and_preview_retirement_do_not_reset_clock(self):
        _, session, _ = await self.make()
        await session._emit("transcript.updated", "segment", text="same words")
        original = session._transcript_idle_text_at
        await session._emit("transcript.updated", "segment", text="same words", stable_text="same words")
        await session._emit("transcript.updated", "segment", text=" ")
        await session._emit("transcript.preview", "segment", source_text="retired")
        self.assertEqual(session._transcript_idle_text_at, original)
        await asyncio.wait_for(session.checkpoint.wait(), .5)

    async def test_new_pcm_postpones_checkpoint_even_without_text_change(self):
        _, session, _ = await self.make()
        await session._emit("transcript.updated", "segment", text="source")
        await asyncio.sleep(.02)
        await session.feed(None, 1)
        await asyncio.sleep(.015)
        self.assertFalse(session.queued)
        await asyncio.wait_for(session.checkpoint.wait(), .5)

    async def test_changed_text_postpones_checkpoint_even_without_new_pcm(self):
        _, session, _ = await self.make()
        await session._emit("transcript.updated", "segment", text="source")
        await asyncio.sleep(.02)
        await session._emit("transcript.updated", "segment", text="source changed")
        await asyncio.sleep(.015)
        self.assertFalse(session.queued)
        await asyncio.wait_for(session.checkpoint.wait(), .5)

    async def test_inflight_asr_is_never_interrupted_and_completion_rechecks_idle(self):
        _, session, _ = await self.make()
        session.processing_release = asyncio.Event()
        processing = asyncio.create_task(session._process_window(None))
        await session.processing_entered.wait()
        try:
            await session._emit("transcript.updated", "segment", text="pending")
            await asyncio.sleep(.05)
            self.assertFalse(session.queued)
            self.assertFalse(processing.done())
            session.processing_release.set()
            await processing
            await asyncio.wait_for(session.checkpoint.wait(), .5)
        finally:
            session.processing_release.set()
            await processing

    async def test_queued_asr_blocks_checkpoint_until_processing_finishes(self):
        _, session, _ = await self.make()
        session._pending["queued"] = object()
        await session._emit("transcript.updated", "segment", text="pending")
        await asyncio.sleep(.05)
        self.assertFalse(session.queued)
        session._pending.clear()
        await session._process_window(None)
        await asyncio.wait_for(session.checkpoint.wait(), .5)

    async def test_close_finish_and_gap_cancel_old_timer(self):
        for action in ("close", "finish", "gap"):
            with self.subTest(action=action):
                _, session, _ = await self.make()
                await session._emit("transcript.updated", "segment", text="pending")
                if action == "gap":
                    await session._emit("audio.gap", reason="synthetic")
                else:
                    await getattr(session, action)()
                await asyncio.sleep(.04)
                self.assertFalse(session.queued)
                self.assertIsNone(session._transcript_idle_task)

    async def test_failed_transcript_emit_has_no_observation(self):
        _, session, _ = await self.make()
        session.fail_emit = True
        with self.assertRaises(ValueError):
            await session._emit("transcript.updated", "segment", text="undelivered")
        self.assertIsNone(session._transcript_idle_text_at)
        self.assertIsNone(session._transcript_idle_task)

    async def test_installer_disabled_validation_idempotence_and_bounded_digest_history(self):
        disabled = SimpleNamespace(StreamingSession=BaseSession)
        install(disabled, inactivity_flush_s=0)
        self.assertIs(disabled.StreamingSession, BaseSession)
        for value in (True, -1, float("nan"), float("inf"), 121):
            with self.subTest(value=value), self.assertRaises(ValueError):
                install(disabled, inactivity_flush_s=value)
        module, session, metadata = await self.make()
        self.assertEqual(install(module, inactivity_flush_s=.03), metadata)
        with self.assertRaises(ValueError):
            install(module, inactivity_flush_s=.04)
        for index in range(40):
            await session._emit("transcript.updated", str(index), text="private source")
        self.assertEqual(len(session._transcript_idle_signatures), 32)
        self.assertNotIn("private source", repr(session._transcript_idle_signatures) + repr(metadata))

    async def test_verified_session_provisional_only_resolves_through_original_asr_finalization(self):
        try:
            import httpx  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Verified engine integration uses MyVote-Mac-Demo/.venv/bin/python")
        runpy.run_path(str(ROOT / "tests/verified_engine.py"))["load_verified_engine"]()
        from myvote_engine import audio, pipeline, semantic_pipeline, semantic_routing
        from myvote_engine.asr import ASRHypothesis, TimedWord
        from myvote_engine.translation import Chunk

        # Other test modules may have installed the non-quality adapter globally.
        # Use that known base and an isolated module namespace, never replace it.
        semantic_base = semantic_pipeline.SemanticCaptionPipeline
        if getattr(semantic_base, "_myvote_natural_boundaries_v1", False):
            semantic_base = next(cls for cls in semantic_base.__mro__
                                 if "_myvote_natural_boundaries_v1" not in cls.__dict__
                                 and cls.__name__ == "SemanticCaptionPipeline")
        stream_base = pipeline.StreamingSession
        while stream_base.__name__ in ("BoundaryStreamingSession", "TranscriptIdleSession"):
            stream_base = stream_base.__mro__[1]
        sm = SimpleNamespace(SemanticCaptionPipeline=semantic_base, _Preview=semantic_pipeline._Preview,
                             _Source=semantic_pipeline._Source, SemanticPendingUnit=semantic_pipeline.SemanticPendingUnit)
        pm = SimpleNamespace(StreamingSession=stream_base)
        runpy.run_path(str(ROOT / "scripts/semantic_boundary_pipeline.py"))["install_pipeline"](
            sm, pm, audio, quality_first=True)
        install(pm, inactivity_flush_s=.03)
        runpy.run_path(str(ROOT / "scripts/semantic_boundary_routing.py"))["install_routing"](semantic_routing)
        quality = SimpleNamespace(SemanticTranslationCoordinator=semantic_routing.SemanticTranslationCoordinator,
                                  SemanticTranslationRouter=semantic_routing.SemanticTranslationRouter,
                                  SemanticTranslationConfig=semantic_routing.SemanticTranslationConfig)
        runpy.run_path(str(ROOT / "scripts/semantic_quality_policy.py"))["install_quality_policy"](
            quality, min_request_interval_s=.001, inactivity_flush_s=.03)

        class FakeASR:
            calls = 0

            def transcribe_pcm(self, samples, *, sample_rate, window_start_ns):
                self.calls += 1
                end = window_start_ns + len(samples) * 1_000_000_000 // sample_rate
                return ASRHypothesis(window_start_ns, end,
                    (TimedWord("A pending fragment", window_start_ns, end - 1),), "en")

        class Provider:
            semantic_mode = "orchestrated"

            def __init__(self):
                self.requests = []

            async def semantic_stream(self, request):
                self.requests.append(request)
                yield Chunk(json.dumps({"action": "commit", "through_id": request.units[-1].unit_id,
                                        "text": "Synthetic residual result"}), completed=True, finish_reason="stop")

        events, provider, asr = [], Provider(), FakeASR()
        completed = asyncio.Event()

        async def sink(event):
            events.append(event)
            if event.kind == "translation.completed":
                completed.set()

        with patch.object(semantic_pipeline, "SemanticCaptionPipeline", sm.SemanticCaptionPipeline), \
                patch.object(semantic_pipeline, "SemanticTranslationRouter", quality.SemanticTranslationRouter), \
                patch.object(semantic_routing, "SemanticTranslationCoordinator", quality.SemanticTranslationCoordinator):
            session = pm.StreamingSession("idle-final", asr, provider, sink=sink,
                builder=audio.SpeechWindowBuilder(step_s=.032, min_window_s=.032),
                config=pipeline.PipelineConfig(source_language="en", target_language="ko", semantic_translation=True))
            try:
                frame = audio.AudioFrame("track", "epoch", 0, 0, (.1,) * 512)
                await session.feed(frame, 1)
                await asyncio.wait_for(completed.wait(), 1)
                updates = [event for event in events if event.kind == "transcript.updated"]
                self.assertEqual(updates[0].data["stable_text"], "")
                self.assertEqual(updates[-1].data["reason"], "flush")
                self.assertEqual(updates[-1].data["stable_text"], "A pending fragment")
                self.assertEqual(asr.calls, 1)  # Upstream cached-finalization path.
                self.assertEqual(session.counts["asr_cached_finalizations"], 1)
                self.assertEqual(session.counts["transcript_idle_checkpoints"], 1)
                self.assertTrue(provider.requests[0].force_flush)
                self.assertEqual("".join(unit.text for unit in provider.requests[0].units), "A pending fragment")
                self.assertFalse([event for event in events if event.kind == "translation.failed"])
                # New PCM after the checkpoint uses the reset builder's next
                # segment, without replaying or changing ownership of old PCM.
                completed.clear()
                await session.feed(audio.AudioFrame("track", "epoch", 1, frame.end_time_ns,
                                                     (.1,) * 512), 1)
                await asyncio.wait_for(completed.wait(), 1)
                sources = [event for event in events if event.kind == "caption.source"]
                self.assertEqual(len(sources), 2)
                self.assertNotEqual(sources[0].segment_id, sources[1].segment_id)
                self.assertEqual([event.data["start_ns"] for event in sources], [0, frame.end_time_ns])
                self.assertEqual(asr.calls, 2)
                self.assertEqual(session.counts["transcript_idle_checkpoints"], 2)
            finally:
                await session.close()


if __name__ == "__main__":
    unittest.main()
