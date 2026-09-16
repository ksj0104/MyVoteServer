"""Verified-engine semantic boundary tests; no GPU, network or server process."""

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
import json
from pathlib import Path
import runpy
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

try:
    import httpx  # noqa: F401 -- needed by the verified translation module
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        "Pipeline tests require httpx; use MyVote-Mac-Demo/.venv/bin/python"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "tests/verified_engine.py"))["load_verified_engine"]()

from myvote_engine import audio, pipeline as streaming, semantic_pipeline as semantic
from myvote_engine import semantic_routing as routing
from myvote_engine.asr import ASRHypothesis, TimedWord, TranscriptUpdate
from myvote_engine.translation import Chunk

runpy.run_path(str(ROOT / "scripts/semantic_boundary_routing.py"))["install_routing"](routing)
semantic.SemanticTranslationRouter = routing.SemanticTranslationRouter
install_pipeline = runpy.run_path(
    str(ROOT / "scripts/semantic_boundary_pipeline.py")
)["install_pipeline"]
install_pipeline(semantic, streaming, audio)
BASE_SESSION = streaming.StreamingSession.__mro__[1]


@dataclass(frozen=True)
class GeneratedWindow(audio.SpeechWindow):
    semantic_generation: int = 0


class CapturedRouter:
    def __init__(self, *args, **kwargs):
        self.counts = {}
        self.pending_units = []
        self.calls = []
        self.soft_boundaries = set()
        self.hard_boundaries = set()

    async def append(self, unit):
        self.pending_units.append(unit)
        self.calls.append(("append", unit.scope))

    def request_boundary(self, scope):
        self.calls.append(("boundary", scope))
        units = [unit for unit in self.pending_units if unit.scope == scope]
        if units:
            self.soft_boundaries.add(units[-1].unit_id)

    def note_transcription(self, scope):
        self.calls.append(("transcription", scope))

    def resume_boundary(self, scope, unit_id, *, allow_after_hold=False):
        units = [unit for unit in self.pending_units if unit.scope == scope]
        if (not units or units[-1].unit_id != unit_id or unit_id not in self.soft_boundaries
                or unit_id in self.hard_boundaries):
            return False
        self.soft_boundaries.remove(unit_id)
        self.calls.append(("resume", scope, unit_id))
        return True

    def request_flush(self, scope=None):
        if scope is None:
            self.calls.append(("flush", tuple(unit.unit_id for unit in self.pending_units)))
        else:
            self.calls.append(("flush_scope", scope))
        self.hard_boundaries.update(unit.unit_id for unit in self.pending_units
                                    if scope is None or unit.scope == scope)


class AcousticMapper:
    def __init__(self):
        self.speakers = {}
        self.calls = []
        self.adverse = {}
        self.retention_floor_ns = 0

    def confirmed_speaker_for(self, **scope):
        self.calls.append(scope)
        return self.speakers.get(scope["start_ns"])

    def _decision(self, reference):
        speaker = self.speakers.get(reference.start_ns)
        reason = self.adverse.get(reference.start_ns)
        return SimpleNamespace(speaker_id=None if reason else speaker,
                               reason=reason or ("dominant_confirmed_speaker" if speaker else "insufficient_coverage"),
                               overlap_ns=0, ambiguous_ns=0, conflict_ns=0)


class CapturedSemantic:
    def __init__(self):
        self._boundary_generation = 0
        self.breaks = []

    def break_continuity(self, generation):
        self.breaks.append(generation)
        self._boundary_generation = generation


class SemanticBoundaryPipelineTests(unittest.IsolatedAsyncioTestCase):
    def make_pipeline(self, *, quality_first=False):
        session = SimpleNamespace(
            config=SimpleNamespace(source_language="en", target_language="ko",
                                   translation_budget_ms=2500),
            counts={}, _closed=False, _speaker_mapper=AcousticMapper(),
            _recent_words=deque(), _emit=AsyncMock(),
        )
        with patch.object(semantic, "SemanticTranslationRouter", CapturedRouter), \
                patch.object(semantic.SemanticCaptionPipeline, "_myvote_endpoint_resume", quality_first):
            pipe = semantic.SemanticCaptionPipeline(session, object())
        return pipe, session, pipe.coordinator

    def make_stream(self, *, semantic_instance=None, max_pending=2):
        # Bypass real VAD/model/session construction, but exercise real enqueue.
        session = object.__new__(streaming.StreamingSession)
        session._boundary_capture_generation = 0
        session._boundary_processing = False
        session._pending = OrderedDict()
        session._pending_enqueued_at = {}
        session._states = {}
        session._available = asyncio.Event()
        session._asr_scheduler = None
        session.config = SimpleNamespace(max_asr_pending=max_pending)
        session.counts = {"asr_skipped": 0, "asr_errors": 0}
        session._semantic = semantic_instance or CapturedSemantic()
        return session

    def window(self, segment="one", *, start=0, end=1000, emit=None,
               final=False, reason="partial", generation=0, track="track", epoch="epoch"):
        return GeneratedWindow(segment, track, epoch, start, end, (.1, .2),
                               final, reason, start if emit is None else emit,
                               semantic_generation=generation)

    def word(self, text, index):
        return TimedWord(text, index * 100, (index + 1) * 100)

    async def accept(self, pipe, window, words, *, state=None, provisional=()):
        if state is None:
            state = SimpleNamespace(semantic_state=None, pending_word_timings=deque())
        submitted = state.semantic_state.submitted if state.semantic_state else 0
        state.pending_word_timings.extend(
            SimpleNamespace(ready_at=time.monotonic()) for _ in words[submitted:])
        update = TranscriptUpdate(window.segment_id, len(words) + 1, "stable",
                                  tuple(words), tuple(provisional), window.reason)
        await pipe.accept(window, state, update, "en", final=window.final,
                          timing=SimpleNamespace())
        return state

    def assert_no_pending_hard_flush(self, router):
        self.assertFalse([call for call in router.calls if call[0] == "flush" and call[1]])

    async def test_quality_endpoint_resumes_same_speaker_after_provisional_only_update(self):
        pipe, session, router = self.make_pipeline(quality_first=True)
        first = TimedWord("The shipment weighs about twenty", 0, 900_000_000)
        second = TimedWord(" kilograms.", 1_410_000_000, 1_900_000_000)
        session._speaker_mapper.speakers.update({first.start_time_ns: "A", second.start_time_ns: "A"})
        await self.accept(pipe, self.window("one", end=1_000_000_000,
                                           final=True, reason="endpoint"), (first,))
        previous = router.pending_units[-1]
        self.assertIn(previous.unit_id, router.soft_boundaries)
        window = self.window("two", start=1_400_000_000, end=2_400_000_000)
        state = await self.accept(pipe, window, (), provisional=(second,))
        self.assertIn(previous.unit_id, router.soft_boundaries)
        await self.accept(pipe, window, (second,), state=state)
        self.assertIn(("resume", previous.scope, previous.unit_id), router.calls)
        self.assertNotIn(previous.unit_id, router.soft_boundaries)
        self.assertEqual(router.pending_units[-1].scope, previous.scope)
        self.assertEqual(state.semantic_state._boundary_turn, "one")
        self.assertIs(router.pending_units[0], previous)
        self.assertEqual("".join(unit.text for unit in router.pending_units),
                         "The shipment weighs about twenty kilograms.")

    async def test_provisional_activity_only_refreshes_changed_confirmed_current_lane(self):
        pipe, session, router = self.make_pipeline(quality_first=True)
        first, tail = self.word("The shipment", 0), self.word(" weighs", 1)
        session._speaker_mapper.speakers.update({0: "A", 100: "A", 101: "A"})
        window = self.window()
        state = await self.accept(pipe, window, (first,))
        scope = router.pending_units[0].scope
        router.calls.clear()
        await self.accept(pipe, window, (first,), state=state, provisional=(tail,))
        self.assertEqual(router.calls, [("transcription", scope)])
        # Repeated text and an ASR timestamp-only revision are not new input.
        await self.accept(pipe, window, (first,), state=state, provisional=(tail,))
        await self.accept(pipe, window, (first,), state=state,
                          provisional=(replace(tail, start_time_ns=101),))
        await self.accept(pipe, window, (first,), state=state)
        self.assertEqual(router.calls, [("transcription", scope)])
        self.assertEqual(len(router.pending_units), 1)

    async def test_other_or_overlap_provisional_cannot_keep_old_speaker_alive(self):
        for label, reason in (("B", None), ("A", "explicit_overlap_evidence")):
            with self.subTest(label=label, reason=reason):
                pipe, session, router = self.make_pipeline(quality_first=True)
                first, tail = self.word("The shipment", 0), self.word(" weighs", 1)
                session._speaker_mapper.speakers.update({0: "A", 100: label})
                if reason:
                    session._speaker_mapper.adverse[100] = reason
                window = self.window()
                state = await self.accept(pipe, window, (first,))
                router.calls.clear()
                await self.accept(pipe, window, (first,), state=state, provisional=(tail,))
                self.assertFalse(any(call[0] == "transcription" for call in router.calls))
                self.assertEqual(len(router.pending_units), 1)

    async def test_benign_unknown_or_no_mapper_activity_keeps_same_anonymous_turn_alive(self):
        for mapper_present in (True, False):
            with self.subTest(mapper_present=mapper_present):
                pipe, session, router = self.make_pipeline(quality_first=True)
                if not mapper_present:
                    session._speaker_mapper = None
                first, tail = self.word("The shipment", 0), self.word(" weighs", 1)
                window = self.window()
                state = await self.accept(pipe, window, (first,))
                original = router.pending_units[0]
                router.calls.clear()
                await self.accept(pipe, window, (first,), state=state, provisional=(tail,))
                self.assertEqual(router.calls, [("transcription", original.scope)])
                self.assertEqual(router.pending_units, [original])
                self.assertIsNone(pipe._boundary_speaker)
                self.assertIsNone(pipe._boundary_anchor)

    async def test_provisional_endpoint_activity_does_not_resume_or_promote_source(self):
        pipe, session, router = self.make_pipeline(quality_first=True)
        first = TimedWord("The shipment weighs twenty", 0, 900_000_000)
        tail = TimedWord(" kilograms", 1_410_000_000, 1_900_000_000)
        session._speaker_mapper.speakers.update({0: "A", tail.start_time_ns: "A"})
        await self.accept(pipe, self.window("first", end=1_000_000_000,
                                           final=True, reason="endpoint"), (first,))
        original = router.pending_units[0]
        router.calls.clear()
        await self.accept(pipe, self.window("second", start=1_400_000_000, end=2_400_000_000),
                          (), provisional=(tail,))
        self.assertEqual(router.calls, [("transcription", original.scope)])
        self.assertEqual(router.pending_units, [original])
        self.assertIn(original.unit_id, router.soft_boundaries)

    async def test_transcript_inactivity_final_only_hard_flushes_its_own_lane(self):
        pipe, session, router = self.make_pipeline(quality_first=True)
        first, second = self.word("A incomplete", 0), self.word("B ongoing", 1)
        session._speaker_mapper.speakers.update({0: "A", 100: "B"})
        a_state = await self.accept(pipe, self.window("a"), (first,))
        await self.accept(pipe, self.window("b"), (second,))
        a_unit, b_unit = router.pending_units
        router.calls.clear()
        await self.accept(pipe, self.window("a", final=True, reason="transcript_inactivity"),
                          (first,), state=a_state)
        self.assertEqual(router.calls, [("flush_scope", a_unit.scope)])
        self.assertIn(a_unit.unit_id, router.hard_boundaries)
        self.assertNotIn(b_unit.unit_id, router.hard_boundaries)

    async def test_quality_endpoint_allows_late_confirmation_only_for_all_pending_words(self):
        pipe, session, router = self.make_pipeline(quality_first=True)
        first = TimedWord("The shipment weighs", 0, 400_000_000)
        middle = TimedWord(" about twenty", 450_000_000, 900_000_000)
        second = TimedWord(" kilograms.", 1_410_000_000, 1_900_000_000)
        await self.accept(pipe, self.window("one", end=1_000_000_000,
                                           final=True, reason="endpoint"), (first, middle))
        original_scope = router.pending_units[-1].scope
        session._speaker_mapper.speakers.update(
            {word.start_time_ns: "A" for word in (first, middle, second)})
        await self.accept(pipe, self.window("two", start=1_400_000_000, end=2_400_000_000), (second,))
        self.assertEqual(router.pending_units[-1].scope, original_scope)
        self.assertTrue(any(call[0] == "resume" for call in router.calls))
        self.assertEqual(pipe._boundary_speaker, "A")

    async def test_quality_endpoint_rejects_changed_speaker_gap_and_hard_cut(self):
        cases = (
            ("changed", "A", "B", 1_410_000_000, 0, False),
            ("long_silence", "A", "A", 2_400_000_001, 0, False),
            ("overlapping_time", "A", "A", 850_000_000, 0, False),
            ("generation_changed", "A", "A", 1_410_000_000, 1, False),
            ("hard_marker", "A", "A", 1_410_000_000, 0, True),
        )
        for name, first_speaker, second_speaker, second_start, generation, hard in cases:
            with self.subTest(name=name):
                pipe, session, router = self.make_pipeline(quality_first=True)
                first = TimedWord("The shipment weighs about twenty", 0, 900_000_000)
                second = TimedWord(" kilograms.", second_start, second_start + 100_000_000)
                session._speaker_mapper.speakers.update({0: first_speaker, second_start: second_speaker})
                await self.accept(pipe, self.window("one", end=1_000_000_000,
                                                   final=True, reason="endpoint"), (first,))
                previous = router.pending_units[-1]
                if hard:
                    router.request_flush()
                await self.accept(pipe, self.window("two", start=second_start,
                                                   end=second_start + 1_000_000_000,
                                                   generation=generation), (second,))
                self.assertFalse(any(call[0] == "resume" for call in router.calls))
                self.assertEqual(router.pending_units[-1].payload.preview._boundary_turn, "two")
                self.assertIn(previous.unit_id, router.soft_boundaries)

    async def test_quality_benign_unknown_endpoints_resume_without_assigning_a_speaker(self):
        pipe, session, router = self.make_pipeline(quality_first=True)
        first = TimedWord("The shipment weighs about twenty", 0, 900_000_000)
        second = TimedWord(" kilograms.", 1_410_000_000, 1_900_000_000)
        await self.accept(pipe, self.window("one", end=1_000_000_000,
                                           final=True, reason="endpoint"), (first,))
        previous = router.pending_units[-1]
        await self.accept(pipe, self.window("two", start=1_400_000_000, end=2_400_000_000), (second,))
        self.assertNotIn(previous.unit_id, router.soft_boundaries)
        self.assertEqual(router.pending_units[-1].scope, previous.scope)
        self.assertIn(":continuity:", previous.lane_id)
        self.assertNotIn("speaker:", previous.lane_id)
        self.assertIsNone(pipe._boundary_speaker)
        self.assertIsNone(pipe._boundary_anchor)
        self.assertEqual(session._speaker_mapper.speakers, {})

    async def test_quality_anchor_survives_unknown_but_never_overrides_an_actual_second_speaker(self):
        pipe, session, router = self.make_pipeline(quality_first=True)
        labels = ("A", None, None, "A", None, "B", "B", None, "A")
        session._speaker_mapper.speakers.update({i * 100: speaker for i, speaker in enumerate(labels)})
        await self.accept(pipe, self.window(), tuple(self.word(" unit", i) for i in range(len(labels))))
        scopes = [unit.scope for unit in router.pending_units]
        self.assertEqual(len(set(scopes[:5])), 1)
        self.assertEqual(len(set(scopes[5:8])), 1)
        self.assertEqual(len(set(scopes)), 3)
        self.assertNotEqual(scopes[0], scopes[-1])
        self.assertEqual(len(router.soft_boundaries), 2)
        self.assertEqual(pipe._boundary_anchor, "A")
        self.assertTrue(all(":continuity:" in scope[-1] and "speaker:" not in scope[-1] for scope in scopes))

    async def test_quality_adverse_evidence_stays_between_separate_translation_runs(self):
        for reason in ("explicit_overlap_evidence", "ambiguous_identity_evidence",
                       "conflicting_evidence", "multiple_speakers", "unexpected_reason"):
            with self.subTest(reason=reason):
                pipe, session, router = self.make_pipeline(quality_first=True)
                session._speaker_mapper.speakers.update({i * 100: "A" for i in range(4)})
                session._speaker_mapper.adverse.update({100: reason, 200: reason})
                await self.accept(pipe, self.window(), tuple(self.word(" unit", i) for i in range(4)))
                scopes = [unit.scope for unit in router.pending_units]
                self.assertEqual(len(set(scopes)), 3)
                self.assertEqual(scopes[1], scopes[2])
                self.assertEqual(len(router.soft_boundaries), 2)

    async def test_quality_endpoint_rechecks_adverse_pending_and_all_new_sources(self):
        for location in ("old", "new_tail"):
            with self.subTest(location=location):
                pipe, session, router = self.make_pipeline(quality_first=True)
                first = TimedWord("prefix", 0, 900_000_000)
                second = TimedWord(" middle", 1_410_000_000, 1_600_000_000)
                third = TimedWord(" tail.", 1_700_000_000, 1_900_000_000)
                session._speaker_mapper.speakers.update({word.start_time_ns: "A" for word in (first, second, third)})
                await self.accept(pipe, self.window("one", end=1_000_000_000, final=True, reason="endpoint"), (first,))
                previous = router.pending_units[-1]
                session._speaker_mapper.adverse[0 if location == "old" else third.start_time_ns] = "explicit_overlap_evidence"
                await self.accept(pipe, self.window("two", start=1_400_000_000, end=2_400_000_000), (second, third))
                self.assertIn(previous.unit_id, router.soft_boundaries)
                self.assertFalse(any(call[0] == "resume" for call in router.calls))

    async def test_quality_unknown_continuity_never_crosses_audio_generation(self):
        pipe, _, router = self.make_pipeline(quality_first=True)
        await self.accept(pipe, self.window("one", final=True, reason="endpoint"), (self.word("prefix", 0),))
        previous = router.pending_units[-1]
        await self.accept(pipe, self.window("two", start=1000, end=2000, generation=1), (self.word(" tail", 11),))
        self.assertNotEqual(router.pending_units[-1].scope, previous.scope)
        self.assertIn(previous.unit_id, router.hard_boundaries)

    async def test_quality_evidence_cache_reduces_once_per_interval_and_refreshes_next_accept(self):
        pipe, session, router = self.make_pipeline(quality_first=True)
        words = tuple(self.word(" unit", i) for i in range(12))
        mapper = session._speaker_mapper
        with patch.object(mapper, "_decision", wraps=mapper._decision) as decide:
            state = await self.accept(pipe, self.window(), words)
            self.assertEqual(decide.call_count, 12)
            mapper.adverse[0] = "conflicting_evidence"
            await self.accept(pipe, self.window(), words + (self.word(" next", 12),), state=state)
            self.assertNotEqual(router.pending_units[-1].scope, router.pending_units[0].scope)
            self.assertLessEqual(decide.call_count, 25)

    async def test_quality_actual_mapper_long_source_and_unknown_endpoints_keep_identity_unassigned(self):
        from myvote_engine.speaker_captions import SpeakerCaptionReducer
        from myvote_engine.speakers import Assignment

        quality = SimpleNamespace(SemanticTranslationCoordinator=routing.SemanticTranslationCoordinator,
                                  SemanticTranslationRouter=routing.SemanticTranslationRouter,
                                  SemanticTranslationConfig=routing.SemanticTranslationConfig)
        runpy.run_path(str(ROOT / "scripts/semantic_quality_policy.py"))["install_quality_policy"](quality)
        pieces = ("The", " shipment", " contains", " several", " carefully", " packed",
                  " boxes", " and", " weighs", " exactly", " twenty", " kilograms.")
        expected = "".join(pieces)

        for scenario, split, adverse, expected_complete in (
                ("jitter", False, False, True), ("unknown_endpoints", True, False, True),
                ("overlap", False, True, False), ("audio_gap", True, False, False)):
            with self.subTest(scenario=scenario):
                words = tuple(TimedWord(part,
                    index * 150_000_000 if not split or index < 6 else 1_200_000_000 + (index - 6) * 150_000_000,
                    (index * 150_000_000 if not split or index < 6 else 1_200_000_000 + (index - 6) * 150_000_000) + 130_000_000)
                    for index, part in enumerate(pieces))

                class FakeASR:
                    def transcribe_pcm(self, samples, *, sample_rate, window_start_ns):
                        chosen = words if not split else words[:6] if window_start_ns == 0 else words[6:]
                        end = 2_000_000_000 if not split else 1_000_000_000 if window_start_ns == 0 else 2_300_000_000
                        return ASRHypothesis(window_start_ns, end, chosen, "en")

                class Provider:
                    semantic_mode = "orchestrated"

                    def __init__(self):
                        self.requests = []

                    async def semantic_stream(self, request):
                        self.requests.append(request)
                        packet = ({"action": "commit", "through_id": request.units[-1].unit_id,
                                   "text": "Synthetic complete result"}
                                  if "".join(unit.text for unit in request.units) == expected else {"action": "wait"})
                        yield Chunk(json.dumps(packet), completed=True, finish_reason="stop")

                events, provider = [], Provider()

                async def sink(event):
                    events.append(event)

                with patch.object(semantic.SemanticCaptionPipeline, "_myvote_endpoint_resume", True), \
                        patch.object(semantic, "SemanticTranslationRouter", quality.SemanticTranslationRouter), \
                        patch.object(routing, "SemanticTranslationCoordinator", quality.SemanticTranslationCoordinator):
                    session = streaming.StreamingSession("synthetic-continuity", FakeASR(), provider, sink=sink,
                        config=streaming.PipelineConfig(source_language="en", target_language="ko", semantic_translation=True))
                    mapper = session._speaker_mapper = SpeakerCaptionReducer(session.store)
                    if not split:
                        mapper.observe(Assignment("known", "track", 0, 2_000_000_000, "existing", speaker_id="A"), capture_epoch="epoch")
                        mapper.observe(Assignment("uncertain", "track", words[4].start_time_ns, words[7].end_time_ns,
                            "unknown", reason="predicted_overlap" if adverse else "insufficient_clean_speech"), capture_epoch="epoch")
                    before_assignments = mapper.retained_assignments
                    session._semantic.coordinator.config = replace(session._semantic.coordinator.config,
                                                                     min_request_interval_s=.001)
                    try:
                        await session._process_window(self.window("one", end=1_000_000_000 if split else 2_000_000_000,
                                                                  final=True, reason="endpoint"))
                        if split:
                            await session._process_window(self.window("two", start=1_100_000_000, end=2_300_000_000,
                                generation=1 if scenario == "audio_gap" else 0, final=True, reason="endpoint"))
                        await session.finish(timeout_s=1)
                        sources = [event for event in events if event.kind == "caption.source"]
                        completed = [event for event in events if event.kind == "translation.completed"]
                        self.assertEqual(session.counts["asr_errors"], 0)
                        self.assertEqual(len(completed), 1 if expected_complete else 0)
                        self.assertTrue(all(not request.force_flush for request in provider.requests))
                        if expected_complete:
                            self.assertEqual([event.data["text"] for event in sources], [expected])
                            self.assertTrue(any(len(request.units) == 12 for request in provider.requests))
                            self.assertIsNone(session.store.get_segment(sources[0].segment_id).speaker_id)
                        else:
                            self.assertFalse(any(len(request.units) == 12 for request in provider.requests))
                            self.assertGreaterEqual(len(sources), 2)
                        self.assertEqual(mapper.retained_assignments, before_assignments)
                        self.assertEqual(mapper.retained_candidate_resolutions, 0)
                    finally:
                        await session.close()

    async def test_quality_incomplete_a_does_not_block_complete_b_or_returning_a(self):
        from myvote_engine.speaker_captions import SpeakerCaptionReducer
        from myvote_engine.speakers import Assignment

        quality = SimpleNamespace(SemanticTranslationCoordinator=routing.SemanticTranslationCoordinator,
                                  SemanticTranslationRouter=routing.SemanticTranslationRouter,
                                  SemanticTranslationConfig=routing.SemanticTranslationConfig)
        runpy.run_path(str(ROOT / "scripts/semantic_quality_policy.py"))["install_quality_policy"](
            quality, min_request_interval_s=.001)
        parts = (("The", " shipment", " weighs"),
                 ("The", " meeting", " has", " ended."),
                 ("The", " report", " is", " ready."))
        texts = tuple("".join(items) for items in parts)
        groups = tuple(tuple(TimedWord(text, stage * 1_100_000_000 + index * 150_000_000,
                                       stage * 1_100_000_000 + index * 150_000_000 + 130_000_000)
                             for index, text in enumerate(items)) for stage, items in enumerate(parts))

        class FakeASR:
            def transcribe_pcm(self, samples, *, sample_rate, window_start_ns):
                return ASRHypothesis(window_start_ns, window_start_ns + 1_000_000_000,
                                     groups[window_start_ns // 1_100_000_000], "en")

        class Provider:
            semantic_mode = "orchestrated"

            def __init__(self):
                self.requests = []

            async def semantic_stream(self, request):
                self.requests.append(request)
                source = "".join(unit.text for unit in request.units)
                packet = ({"action": "commit", "through_id": request.units[-1].unit_id,
                           "text": "Synthetic completed result"}
                          if source in texts[1:] else {"action": "wait"})
                yield Chunk(json.dumps(packet), completed=True, finish_reason="stop")

        async def until(predicate):
            async with asyncio.timeout(1):
                while not predicate():
                    await asyncio.sleep(.001)

        events, provider = [], Provider()

        async def sink(event):
            events.append(event)

        with patch.object(semantic.SemanticCaptionPipeline, "_myvote_endpoint_resume", True), \
                patch.object(semantic, "SemanticTranslationRouter", quality.SemanticTranslationRouter), \
                patch.object(routing, "SemanticTranslationCoordinator", quality.SemanticTranslationCoordinator):
            session = streaming.StreamingSession("multispeaker-turns", FakeASR(), provider, sink=sink,
                config=streaming.PipelineConfig(source_language="en", target_language="ko", semantic_translation=True))
            mapper = session._speaker_mapper = SpeakerCaptionReducer(session.store)
            for stage, speaker in enumerate(("A", "B", "A")):
                start = stage * 1_100_000_000
                mapper.observe(Assignment("known-" + str(stage), "track", start, start + 1_000_000_000,
                                          "existing", speaker_id=speaker), capture_epoch="epoch")
            try:
                await session._process_window(self.window("a-first", end=1_000_000_000,
                                                          final=True, reason="endpoint"))
                await until(lambda: session._semantic.coordinator.counts["waits"] >= 1)
                first_units = session._semantic.coordinator.pending_units
                self.assertEqual("".join(unit.text for unit in first_units), texts[0])
                for stage, segment in ((1, "b-complete"), (2, "a-return")):
                    start = stage * 1_100_000_000
                    await session._process_window(self.window(segment, start=start,
                        end=start + 1_000_000_000, final=True, reason="endpoint"))
                    await until(lambda: len([event for event in events
                                             if event.kind == "translation.completed"]) >= stage)
                    self.assertEqual(session._semantic.coordinator.pending_units, first_units)
                captions = [event for event in events if event.kind == "caption.source"]
                self.assertEqual([event.data["text"] for event in captions], list(texts[1:]))
                self.assertEqual([session.store.get_segment(event.segment_id).speaker_id
                                  for event in captions], ["B", "A"])
                self.assertFalse([event for event in events if event.kind == "translation.failed"])
                self.assertEqual(len(session._semantic.coordinator._lanes), 3)
                self.assertTrue(all(not request.force_flush for request in provider.requests))
                by_source = {"".join(unit.text for unit in request.units): request
                             for request in provider.requests}
                self.assertEqual(set(by_source), set(texts))
                self.assertEqual(by_source[texts[1]].context, ())
                # Context is labelled shared dialogue, not an unfinished word
                # buffer and not generated translation. A return is a new turn.
                self.assertTrue(any(texts[1] in entry for entry in by_source[texts[2]].context))
                self.assertFalse(any(texts[0] in entry for entry in by_source[texts[2]].context))
            finally:
                await session.close()

    async def test_default_policy_keeps_known_speaker_endpoint_sealed(self):
        pipe, session, router = self.make_pipeline()
        first = TimedWord("The shipment weighs about twenty", 0, 900_000_000)
        second = TimedWord(" kilograms.", 1_410_000_000, 1_900_000_000)
        session._speaker_mapper.speakers.update({0: "A", second.start_time_ns: "A"})
        await self.accept(pipe, self.window("one", end=1_000_000_000,
                                           final=True, reason="endpoint"), (first,))
        previous = router.pending_units[-1]
        await self.accept(pipe, self.window("two", start=1_400_000_000, end=2_400_000_000), (second,))
        self.assertFalse(any(call[0] == "resume" for call in router.calls))
        self.assertIn(previous.unit_id, router.soft_boundaries)

    async def test_pipeline_installer_rejects_a_conflicting_quality_mode(self):
        with self.assertRaisesRegex(ValueError, "different endpoint policy"):
            install_pipeline(semantic, streaming, audio, quality_first=True)

    async def test_quality_real_session_resumes_endpoint_during_inflight_wait_without_source_loss(self):
        quality_routing = SimpleNamespace(
            SemanticTranslationCoordinator=routing.SemanticTranslationCoordinator,
            SemanticTranslationRouter=routing.SemanticTranslationRouter,
            SemanticTranslationConfig=routing.SemanticTranslationConfig,
        )
        runpy.run_path(str(ROOT / "scripts/semantic_quality_policy.py"))[
            "install_quality_policy"
        ](quality_routing)
        second_start = 1_040_000_000
        first_words = tuple(TimedWord(text, index * 150_000_000, index * 150_000_000 + 130_000_000)
                            for index, text in enumerate(("The", " shipment", " weighs", " about", " twenty")))
        last_word = TimedWord(" kilograms.", 1_050_000_000, 1_400_000_000)
        expected = "The shipment weighs about twenty kilograms."

        class FakeASR:
            def transcribe_pcm(self, samples, *, sample_rate, window_start_ns):
                words = first_words if window_start_ns == 0 else (last_word,)
                return ASRHypothesis(window_start_ns, window_start_ns + 1_000_000_000, words, "en")

        class ConfirmedMapper:
            def confirmed_speaker_for(self, **scope):
                return "A"

            def register_caption(self, *args, **kwargs):
                return ()

        class GeneratedProvider:
            semantic_mode = "orchestrated"

            def __init__(self):
                self.requests = []
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def semantic_stream(self, request):
                self.requests.append(request)
                source = "".join(unit.text for unit in request.units)
                if source == expected:
                    packet = {"action": "commit", "through_id": request.units[-1].unit_id,
                              "text": "배송물은 약 20킬로그램입니다."}
                else:
                    self.entered.set()
                    await self.release.wait()
                    packet = {"action": "wait"}
                yield Chunk(json.dumps(packet, ensure_ascii=False), completed=True, finish_reason="stop")

        for resume_enabled in (False, True):
            with self.subTest(resume_enabled=resume_enabled):
                events, translated = [], asyncio.Event()

                async def sink(event):
                    events.append(event)
                    if event.kind == "translation.completed":
                        translated.set()

                provider = GeneratedProvider()
                with patch.object(semantic.SemanticCaptionPipeline, "_myvote_endpoint_resume", resume_enabled), \
                        patch.object(semantic, "SemanticTranslationRouter", quality_routing.SemanticTranslationRouter), \
                        patch.object(routing, "SemanticTranslationCoordinator", quality_routing.SemanticTranslationCoordinator):
                    session = streaming.StreamingSession(
                        "quality-endpoint-reproduction", FakeASR(), provider, sink=sink,
                        config=streaming.PipelineConfig(source_language="en", target_language="ko",
                                                       semantic_translation=True))
                    session._speaker_mapper = ConfirmedMapper()
                    session._semantic.coordinator.config = replace(
                        session._semantic.coordinator.config, min_request_interval_s=.001)
                    try:
                        await session._process_window(self.window("first", end=1_000_000_000,
                                                                  final=True, reason="endpoint"))
                        await asyncio.wait_for(provider.entered.wait(), 1)
                        previous_units = session._semantic.coordinator.pending_units
                        await session._process_window(self.window(
                            "second", start=second_start, end=second_start + 1_000_000_000,
                            final=True, reason="endpoint"))
                        self.assertEqual(session._semantic.coordinator.pending_units[:len(previous_units)],
                                         previous_units)
                        provider.release.set()
                        if resume_enabled:
                            await asyncio.wait_for(translated.wait(), 1)
                        await session.finish(timeout_s=2)
                    finally:
                        provider.release.set()
                        await session.close()
                sources = [event for event in events if event.kind == "caption.source"]
                results = [event for event in events if event.kind == "translation.completed"]
                failures = [event for event in events if event.kind == "translation.failed"]
                self.assertEqual(session.counts["asr_errors"], 0)
                self.assertTrue(all(not request.force_flush for request in provider.requests))
                if resume_enabled:
                    self.assertEqual([event.data["text"] for event in sources], [expected])
                    self.assertEqual([event.data["text"] for event in results], ["배송물은 약 20킬로그램입니다."])
                    self.assertEqual(failures, [])
                    self.assertEqual(len(provider.requests[-1].units), 6)
                    self.assertEqual(len({unit.unit_id for unit in provider.requests[-1].units}), 6)
                else:
                    self.assertEqual(results, [])
                    self.assertEqual([event.data["error"] for event in failures],
                                     ["semantic_incomplete_source", "semantic_incomplete_source"])
                    self.assertEqual([event.data["text"] for event in sources],
                                     ["The shipment weighs about twenty", "kilograms."])

    async def test_max_window_keeps_last_stable_word_and_final_preview_without_force(self):
        pipe, session, router = self.make_pipeline()
        words = (self.word("We ", 0), self.word("finished.", 1))
        provisional = (self.word(" hallucinated", 2),)
        state = await self.accept(pipe, self.window(), words[:1], provisional=provisional)
        router.calls.clear()
        await self.accept(pipe, self.window(final=True, reason="max_window"),
                          words, state=state, provisional=provisional)
        self.assertEqual([unit.payload.word for unit in router.pending_units], list(words))
        self.assertEqual(list(session._recent_words), list(words))
        self.assertTrue(all(unit.payload.window.samples == () for unit in router.pending_units))
        preview = session._emit.await_args.kwargs
        self.assertEqual(preview["source_text"], "We finished.")
        self.assertEqual(preview["stable_text"], "We finished.")
        self.assertEqual((preview["start_ns"], preview["end_ns"]), (0, 200))
        self.assertEqual(state.semantic_state.update.provisional_words, ())
        self.assertFalse(state.pending_word_timings)
        self.assertEqual(router.calls, [("append", router.pending_units[-1].scope)])

    async def test_only_exact_continuation_reuses_unknown_lane(self):
        for emit, start, track, epoch, reuse in (
            (1000, 800, "track", "epoch", True),
            (1001, 800, "track", "epoch", False),
            (1000, 1001, "track", "epoch", False),
            (1000, 800, "other", "epoch", False),
            (1000, 800, "track", "other", False),
        ):
            with self.subTest(emit=emit, start=start, track=track, epoch=epoch):
                pipe, _, router = self.make_pipeline()
                await self.accept(pipe, self.window(final=True, reason="max_window"),
                                  (self.word("First ", 0),))
                first = router.pending_units[0].scope
                await self.accept(pipe, self.window("two", start=start, emit=emit,
                                                   end=2000, track=track, epoch=epoch),
                                  (self.word("next", 11),))
                second = router.pending_units[1].scope
                self.assertEqual(first == second, reuse)
                self.assertEqual([c for c in router.calls if c[0] == "boundary"],
                                 [] if reuse else [("boundary", first)])
                self.assert_no_pending_hard_flush(router)

    async def test_endpoint_without_fresh_words_soft_flushes_only_its_preview_scope(self):
        pipe, session, router = self.make_pipeline()
        session._speaker_mapper.speakers.update({0: "A", 100: "B"})
        words = (self.word("A", 0),)
        state = await self.accept(pipe, self.window("a"), words)
        first_scope = router.pending_units[0].scope
        await self.accept(pipe, self.window("b"), (self.word("B", 1),))
        other_scope = router.pending_units[1].scope
        router.calls.clear()
        await self.accept(pipe, self.window("a", final=True, reason="endpoint"),
                          words, state=state)
        self.assertNotEqual(first_scope, other_scope)
        self.assertEqual(router.calls, [("boundary", first_scope)])
        self.assertIsNone(pipe._boundary_continuation)

    async def test_unknown_lane_promotion_requires_all_pending_acoustic_evidence(self):
        pipe, session, router = self.make_pipeline()
        words = (self.word("one ", 0), self.word("two ", 1))
        state = await self.accept(pipe, self.window(), words)
        original_scope = router.pending_units[0].scope
        session._speaker_mapper.speakers.update({0: "A", 100: "A", 200: "A"})
        router.calls.clear()
        await self.accept(pipe, self.window(), words + (self.word("three", 2),), state=state)
        self.assertTrue(all(unit.scope == original_scope for unit in router.pending_units))
        self.assertEqual(pipe._boundary_speaker, "A")
        self.assertEqual(router.calls, [("append", original_scope)])
        confirmed_starts = [call["start_ns"] for call in session._speaker_mapper.calls]
        self.assertEqual(confirmed_starts[-3:], [200, 0, 100])

    async def test_partial_or_conflicting_evidence_cannot_promote_unknown_lane(self):
        for earlier_speaker in (None, "B"):
            with self.subTest(earlier_speaker=earlier_speaker):
                pipe, session, router = self.make_pipeline()
                words = (self.word("one ", 0), self.word("two ", 1))
                state = await self.accept(pipe, self.window(), words)
                first = router.pending_units[0].scope
                session._speaker_mapper.speakers.update({0: "A", 100: earlier_speaker, 200: "A"})
                router.calls.clear()
                await self.accept(pipe, self.window(), words + (self.word("three", 2),), state=state)
                self.assertNotEqual(router.pending_units[-1].scope, first)
                self.assertEqual(router.calls[0], ("boundary", first))
                self.assert_no_pending_hard_flush(router)

    async def test_promoted_unknown_lane_never_absorbs_overlap_or_changed_speaker(self):
        for next_speaker in (None, "B"):
            with self.subTest(next_speaker=next_speaker):
                pipe, session, router = self.make_pipeline()
                words = (self.word("unknown ", 0),)
                state = await self.accept(pipe, self.window(), words)
                opaque_scope = router.pending_units[0].scope
                session._speaker_mapper.speakers.update({0: "A", 100: "A"})
                words += (self.word("confirmed ", 1),)
                await self.accept(pipe, self.window(), words, state=state)
                self.assertEqual(router.pending_units[-1].scope, opaque_scope)
                session._speaker_mapper.speakers[200] = next_speaker
                router.calls.clear()
                await self.accept(pipe, self.window(), words + (self.word("uncertain", 2),), state=state)
                new_scope = router.pending_units[-1].scope
                self.assertNotEqual(new_scope, opaque_scope)
                if next_speaker is None:
                    self.assertTrue(new_scope[-1].endswith(":r3"))
                self.assertEqual(router.calls[0], ("boundary", opaque_scope))
                self.assert_no_pending_hard_flush(router)

    async def test_a_b_a_soft_boundaries_precede_each_new_turn(self):
        pipe, session, router = self.make_pipeline()
        session._speaker_mapper.speakers.update({0: "A", 100: "B", 200: "A"})
        await self.accept(pipe, self.window(), tuple(self.word(text, index)
                         for index, text in enumerate(("A ", "B ", "A again"))))
        scopes = [unit.scope for unit in router.pending_units]
        self.assertEqual(scopes[0], scopes[2])
        self.assertNotEqual(scopes[0], scopes[1])
        self.assertEqual(router.calls[1:], [
            ("append", scopes[0]), ("boundary", scopes[0]),
            ("append", scopes[1]), ("boundary", scopes[1]), ("append", scopes[2]),
        ])
        self.assert_no_pending_hard_flush(router)

    async def test_gap_generation_prevents_even_exact_max_window_continuation(self):
        pipe, _, router = self.make_pipeline()
        await self.accept(pipe, self.window(final=True, reason="max_window"),
                          (self.word("before", 0),))
        first = router.pending_units[0].scope
        router.calls.clear()
        await self.accept(pipe, self.window("two", start=800, emit=1000, end=2000,
                                           generation=1), (self.word("after", 11),))
        second = router.pending_units[-1].scope
        self.assertIn("boundary-g0:unassigned:one", first[-1])
        self.assertIn("boundary-g1:unassigned:two", second[-1])
        self.assertEqual(router.calls, [("flush", ("u1",)), ("append", second)])

    async def test_idle_audio_gap_breaks_immediately_and_preserves_event(self):
        session = self.make_stream()
        with patch.object(BASE_SESSION, "_emit", new_callable=AsyncMock) as emit:
            await session._emit("audio.gap", track_id="track", reason="lost")
        self.assertEqual(session._boundary_capture_generation, 1)
        self.assertEqual(session._semantic.breaks, [1])
        emit.assert_awaited_once_with("audio.gap", None, track_id="track", reason="lost")

    async def test_active_asr_finishes_old_generation_before_new_gap_window(self):
        pipe, _, router = self.make_pipeline()
        pipe._boundary_generation = 0
        session = self.make_stream(semantic_instance=pipe)
        await session._enqueue(self.window("old"))
        _, active = session._pending.popitem(last=False)

        async def process(instance, window, *, queued_at=None):
            self.assertTrue(instance._boundary_processing)
            if window.segment_id == "old":
                await instance._emit("audio.gap", reason="lost")
                self.assertEqual(pipe._boundary_generation, 0)
                await instance._enqueue(self.window("new", start=1000, end=2000))
                self.assertEqual(window.semantic_generation, 0)
            await self.accept(pipe, window, (self.word(window.segment_id, 0),))

        with patch.object(BASE_SESSION, "_emit", new_callable=AsyncMock), \
                patch.object(BASE_SESSION, "_process_window", process):
            await session._process_window(active, queued_at=12.5)
            self.assertFalse(session._boundary_processing)
            self.assertEqual(pipe._boundary_generation, 0)
            self.assertEqual(session._pending["new"].semantic_generation, 1)
            self.assertEqual([call[0] for call in router.calls], ["append"])
            _, new = session._pending.popitem(last=False)
            await session._process_window(new)
        self.assertEqual([call[0] for call in router.calls], ["append", "flush", "append"])
        self.assertEqual(pipe._boundary_generation, 1)
        self.assertNotEqual(router.pending_units[0].scope, router.pending_units[1].scope)

    async def test_overload_retags_survivors_and_new_window_without_merging_old_gaps(self):
        session = self.make_stream(max_pending=3)
        with patch.object(BASE_SESSION, "_emit", new_callable=AsyncMock):
            await session._enqueue(self.window("dropped"))
            await session._enqueue(self.window("before-gap"))
            active = session._pending["dropped"]
            await session._emit("audio.gap", reason="lost")
            await session._enqueue(self.window("after-gap"))
            self.assertEqual([w.semantic_generation for w in session._pending.values()], [0, 0, 1])
            await session._enqueue(self.window("new"))
        self.assertEqual(list(session._pending), ["before-gap", "after-gap", "new"])
        self.assertEqual([w.semantic_generation for w in session._pending.values()], [2, 3, 3])
        self.assertEqual(session._boundary_capture_generation, 3)
        self.assertEqual(active.semantic_generation, 0)
        self.assertEqual(session.counts["asr_skipped"], 1)
        self.assertEqual(session._semantic.breaks, [])
        self.assertEqual(session._pending["new"].samples, (.1, .2))

    async def test_final_asr_error_breaks_continuity_after_processing_finishes(self):
        session = self.make_stream()

        async def failed_process(instance, window, *, queued_at=None):
            self.assertTrue(instance._boundary_processing)
            instance.counts["asr_errors"] += 1

        with patch.object(BASE_SESSION, "_process_window", failed_process):
            await session._process_window(self.window(final=True, reason="endpoint"))
        self.assertFalse(session._boundary_processing)
        self.assertEqual(session._boundary_capture_generation, 1)
        self.assertEqual(session._semantic.breaks, [1])

    async def test_real_session_preserves_source_across_max_window_endpoint_and_finish(self):
        first_words = (TimedWord("The", 0, 250_000_000),
                       TimedWord(" meeting ", 300_000_000, 950_000_000))
        continuation_words = (
            # Retained rolling context crosses the boundary; it must not repeat.
            TimedWord(" meeting ", 300_000_000, 1_050_000_000),
            TimedWord("ended.", 1_050_000_000, 1_400_000_000),
            TimedWord(" We", 1_400_000_000, 1_550_000_000),
            TimedWord(" went", 1_550_000_000, 1_750_000_000),
            TimedWord(" home.", 1_750_000_000, 1_950_000_000),
        )
        hypotheses = {
            0: ASRHypothesis(0, 1_000_000_000, first_words, "en"),
            300_000_000: ASRHypothesis(300_000_000, 2_000_000_000,
                                     continuation_words, "en"),
        }
        expected_source = "The meeting ended. We went home."
        expected_result = "회의가 끝났습니다. 우리는 집에 갔습니다."

        class FakeASR:
            def __init__(self):
                self.calls = []

            def transcribe_pcm(self, samples, *, sample_rate, window_start_ns):
                self.calls.append((sample_rate, window_start_ns))
                return hypotheses[window_start_ns]

        class SemanticProvider:
            def __init__(self):
                self.requests = []
                self.waited = asyncio.Event()

            async def semantic_stream(self, request):
                self.requests.append(request)
                source = "".join(unit.text for unit in request.units).strip()
                if source == expected_source:
                    packet = {"action": "commit", "through_id": request.units[-1].unit_id,
                              "text": expected_result}
                else:
                    packet = {"action": "wait"}
                    self.waited.set()
                yield Chunk(json.dumps(packet), completed=True, finish_reason="stop")

        events, translated = [], asyncio.Event()

        async def sink(event):
            events.append(event)
            if event.kind == "translation.completed":
                translated.set()

        asr, provider = FakeASR(), SemanticProvider()
        session = streaming.StreamingSession(
            "integration", asr, provider, sink=sink,
            config=streaming.PipelineConfig(source_language="en", target_language="ko",
                                            semantic_translation=True))
        self.addAsyncCleanup(session.close)
        # Only shorten request pacing; retain production hold/timeout/age limits.
        session._semantic.coordinator.config = replace(
            session._semantic.coordinator.config, min_request_interval_s=.001)
        await session.start()
        first = self.window("first", end=1_000_000_000, final=True, reason="max_window")
        await session._process_window(first)
        await asyncio.wait_for(provider.waited.wait(), 2)
        self.assertFalse(provider.requests[0].force_flush)
        self.assertEqual("".join(unit.text for unit in provider.requests[0].units).strip(),
                         "The meeting")
        first_lane = session._semantic._last_route
        second = self.window("second", start=300_000_000, emit=1_000_000_000,
                             end=2_000_000_000)
        await session._process_window(second)
        self.assertFalse(any(event.kind == "caption.source" for event in events))
        await session._process_window(replace(second, final=True, reason="endpoint"))
        self.assertEqual(session._semantic._last_route, first_lane)
        await asyncio.wait_for(translated.wait(), 2)
        await session.finish(timeout_s=2)
        event_count = len(events)
        await session.close()  # Closing an already finished session is harmless.

        sources = [event for event in events if event.kind == "caption.source"]
        results = [event for event in events if event.kind == "translation.completed"]
        self.assertEqual([event.data["text"] for event in sources], [expected_source])
        self.assertEqual([event.data["text"] for event in results], [expected_result])
        self.assertEqual(sources[0].segment_id, results[0].segment_id)
        self.assertEqual((sources[0].data["start_ns"], sources[0].data["end_ns"]),
                         (0, 1_950_000_000))
        self.assertFalse(sources[0].data["semantic_force_flush"])
        self.assertTrue(all(not request.force_flush for request in provider.requests))
        self.assertEqual(len(provider.requests[-1].units), 6)
        self.assertEqual(len({unit.unit_id for unit in provider.requests[-1].units}), 6)
        self.assertEqual(asr.calls, [(16000, 0), (16000, 300_000_000)])
        self.assertEqual(session.counts["asr_cached_finalizations"], 1)
        self.assertEqual(session.counts["boundary_duplicates"], 1)
        self.assertEqual(session.counts["asr_errors"], 0)
        self.assertEqual(session.counts["translation_failed"], 0)
        self.assertEqual(session.counts["translation_completed"], 1)
        self.assertFalse([event for event in events if event.kind.endswith(".failed")])
        self.assertEqual(len(events), event_count)
        self.assertTrue(session._closed)
        self.assertTrue(session._asr_task.done())
        self.assertFalse(session._states)
        self.assertFalse(session._pending)


if __name__ == "__main__":
    unittest.main()
