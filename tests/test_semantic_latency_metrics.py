"""No-model tests for the observational first-preview-to-server-event clock."""

from dataclasses import dataclass
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest


SUPPORT = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                            "scripts/semantic_latency_metrics.py"))
install = SUPPORT["install_latency_metrics"]


@dataclass(frozen=True)
class Word:
    text: str
    start_time_ns: int
    end_time_ns: int


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


def preview(stable=(), provisional=()):
    return SimpleNamespace(update=SimpleNamespace(stable_words=stable, provisional_words=provisional),
                           retired=set(), revision=0, last_payload=None)


def result_for(item, word, *, ready_at=12.0):
    return SimpleNamespace(units=(SimpleNamespace(payload=SimpleNamespace(
        preview=item, word=word, commit=SimpleNamespace(ready_at=ready_at))),))


class BasePipeline:
    """Mirror upstream no-op, pre-emit revision mutation and source lifecycle."""
    def __init__(self):
        self.closed = self.fail_preview = self.fail_source = False
        self.timings = {}
        self.emitted = []
        self.sources = []

    async def _preview(self, item):
        if self.closed:
            return
        stable = tuple(word for index, word in enumerate(item.update.stable_words)
                       if index not in item.retired)
        text = "".join(word.text for word in stable + item.update.provisional_words).strip()[-4000:]
        stable_text = "".join(word.text for word in stable).strip()[-4000:]
        payload = (text, stable_text)
        if item.last_payload == payload:
            return
        item.last_payload = payload
        item.revision += 1
        if self.fail_preview:
            raise ValueError("synthetic sink failure")
        self.emitted.append(payload)

    async def _source(self, result):
        if self.closed:
            return None
        if self.fail_source:
            raise ValueError("synthetic source failure")
        self.sources.append(result)
        key = "caption-" + str(len(self.sources))
        self.timings[key] = {"semantic_total_ms": 123.0}
        return key

    def metrics(self, key):
        value = self.timings.get(key)
        return None if value is None else dict(value)


class LatencyMetricsTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        module = SimpleNamespace(SemanticCaptionPipeline=BasePipeline)
        metadata = install(module)
        pipe = module.SemanticCaptionPipeline()
        clock = Clock()
        pipe._latency_clock = clock
        return module, pipe, clock, metadata

    async def test_later_words_do_not_reset_oldest_word(self):
        _, pipe, clock, _ = self.make()
        first, last = Word("The shipment", 0, 100), Word(" arrived.", 101, 200)
        item = preview(provisional=(first,))
        await pipe._preview(item)
        clock.now = 11.5
        item.update = SimpleNamespace(stable_words=(first, last), provisional_words=())
        await pipe._preview(item)
        result = result_for(item, first, ready_at=11.5)
        key = await pipe._source(result)
        clock.now = 12.1
        metrics = pipe.metrics(key)
        self.assertAlmostEqual(metrics["semantic_first_word_age_ms"], 2100)
        self.assertTrue(metrics["semantic_latency_exceeded"])
        self.assertFalse(metrics["semantic_latency_anchor_fallback"])
        self.assertIs(pipe.sources[0], result)
        self.assertEqual(result.units[0].payload.commit.ready_at, 11.5)
        self.assertEqual(metrics["semantic_total_ms"], 123.0)

    async def test_provisional_to_canonical_stable_and_repeat_keep_first_clock(self):
        _, pipe, clock, _ = self.make()
        provisional = Word(" Hello", 0, 100)
        stable = Word(" HELLO", 0, 100)
        item = preview(provisional=(provisional,))
        await pipe._preview(item)
        clock.now = 11
        await pipe._preview(item)
        self.assertEqual(len(pipe.emitted), 1)
        item.update = SimpleNamespace(stable_words=(stable,), provisional_words=())
        await pipe._preview(item)
        key = await pipe._source(result_for(item, stable, ready_at=11))
        clock.now = 12
        self.assertEqual(pipe.metrics(key)["semantic_first_word_age_ms"], 2000)
        self.assertFalse(pipe.metrics(key)["semantic_latency_exceeded"])

    async def test_second_caption_uses_its_own_first_appearance_after_retirement(self):
        _, pipe, clock, _ = self.make()
        first, second = Word("First.", 0, 100), Word(" Second.", 101, 200)
        item = preview(stable=(first,))
        await pipe._preview(item)
        clock.now = 11
        item.update = SimpleNamespace(stable_words=(first, second), provisional_words=())
        await pipe._preview(item)
        first_key = await pipe._source(result_for(item, first))
        item.retired.add(0)
        await pipe._preview(item)
        clock.now = 13
        second_key = await pipe._source(result_for(item, second))
        self.assertEqual(pipe.metrics(first_key)["semantic_first_word_age_ms"], 3000)
        self.assertEqual(pipe.metrics(second_key)["semantic_first_word_age_ms"], 2000)
        self.assertEqual(len(item._latency_first_seen), 1)

    async def test_failed_emit_does_not_fabricate_observation_or_change_lifecycle(self):
        _, pipe, clock, _ = self.make()
        word = Word("Ready.", 0, 100)
        item = preview(stable=(word,))
        pipe.fail_preview = True
        with self.assertRaisesRegex(ValueError, "sink failure"):
            await pipe._preview(item)
        self.assertFalse(hasattr(item, "_latency_first_seen"))
        # Upstream already mutated last_payload. Its repeated no-op must not
        # turn that unsuccessful emission into a successful observation.
        pipe.fail_preview = False
        await pipe._preview(item)
        key = await pipe._source(result_for(item, word, ready_at=11))
        clock.now = 12
        self.assertTrue(pipe.metrics(key)["semantic_latency_anchor_fallback"])
        self.assertEqual(pipe.metrics(key)["semantic_first_word_age_ms"], 1000)

    async def test_timestamp_revision_never_guesses_ownership(self):
        _, pipe, clock, _ = self.make()
        old, changed = Word(" go", 0, 100), Word(" go", 1, 101)
        item = preview(provisional=(old,))
        await pipe._preview(item)
        key = await pipe._source(result_for(item, changed, ready_at=11))
        clock.now = 12
        self.assertTrue(pipe.metrics(key)["semantic_latency_anchor_fallback"])
        self.assertEqual(pipe.metrics(key)["semantic_first_word_age_ms"], 1000)

    async def test_clipped_word_is_not_claimed_visible(self):
        _, pipe, clock, _ = self.make()
        first, huge = Word("Hidden.", 0, 100), Word(" " + "x" * 4001, 101, 200)
        item = preview(stable=(first, huge))
        await pipe._preview(item)
        self.assertEqual(len(item._latency_first_seen), 0)
        key = await pipe._source(result_for(item, first, ready_at=11))
        clock.now = 12
        self.assertTrue(pipe.metrics(key)["semantic_latency_anchor_fallback"])

    async def test_histories_are_bounded_and_preserve_visible_head(self):
        _, pipe, clock, _ = self.make()
        words = tuple(Word(" x", index * 2, index * 2 + 1) for index in range(300))
        item = preview(stable=words)
        await pipe._preview(item)
        self.assertEqual(len(item._latency_first_seen), 256)
        for _ in range(300):
            key = await pipe._source(result_for(item, words[0]))
        self.assertEqual(len(pipe._latency_anchors), 256)
        clock.now = 11
        self.assertEqual(pipe.metrics(key)["semantic_first_word_age_ms"], 1000)
        self.assertFalse(pipe.metrics(key)["semantic_latency_anchor_fallback"])

    async def test_closed_or_failed_source_does_not_register_caption_anchor(self):
        _, pipe, _, _ = self.make()
        word = Word("Ready.", 0, 100)
        item = preview(stable=(word,))
        pipe.closed = True
        await pipe._preview(item)
        self.assertFalse(hasattr(item, "_latency_first_seen"))
        self.assertIsNone(await pipe._source(result_for(item, word)))
        pipe.closed, pipe.fail_source = False, True
        with self.assertRaisesRegex(ValueError, "source failure"):
            await pipe._source(result_for(item, word))
        self.assertFalse(pipe._latency_anchors)
        self.assertIsNone(pipe.metrics("missing"))

    async def test_metadata_and_metrics_expose_no_text_or_raw_clock(self):
        module, pipe, clock, metadata = self.make()
        self.assertEqual(install(module), metadata)
        with self.assertRaises(ValueError):
            install(module, target_latency_s=3)
        self.assertEqual(metadata["semantic_latency_scope"],
                         "first_source_preview_to_server_translation_event")
        word = Word("private source", 0, 100)
        item = preview(stable=(word,))
        await pipe._preview(item)
        key = await pipe._source(result_for(item, word))
        clock.now = 11
        metrics = pipe.metrics(key)
        self.assertEqual(set(metrics), {"semantic_total_ms", "semantic_first_word_age_ms",
                         "semantic_latency_target_ms", "semantic_latency_exceeded",
                         "semantic_latency_anchor_fallback"})
        self.assertNotIn(word.text, repr(metadata) + repr(metrics) + repr(item._latency_first_seen))

    async def test_invalid_targets_are_rejected(self):
        for value in (True, 0, -1, float("inf"), float("nan"), 121):
            with self.subTest(value=value), self.assertRaises(ValueError):
                install(SimpleNamespace(SemanticCaptionPipeline=BasePipeline), target_latency_s=value)

    async def test_verified_pipeline_commit_emits_metric_without_changing_source(self):
        try:
            import httpx  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Verified engine integration uses MyVote-Mac-Demo/.venv/bin/python")
        root = Path(__file__).resolve().parents[1]
        runpy.run_path(str(root / "tests/verified_engine.py"))["load_verified_engine"]()
        from myvote_engine import semantic_pipeline
        from myvote_engine.captions import CaptionStore

        module = SimpleNamespace(SemanticCaptionPipeline=semantic_pipeline.SemanticCaptionPipeline)
        install(module)
        events = []

        async def emit(kind, segment_id=None, **data):
            events.append((kind, segment_id, data))

        async def on_translation(event):
            events.append(("translation." + event.kind, event.segment_id,
                           pipe.metrics(event.segment_id)))

        session = SimpleNamespace(
            config=SimpleNamespace(source_language="en", target_language="ko", translation_budget_ms=2500),
            counts={"clauses": 0, "asr_errors": 0}, _closed=False,
            ingress_timeline=None, _caption_times={}, _ingress_anchors={}, _word_trace=None,
            _context_refinement=None, _speaker_mapper=None, _overlap=None,
            _context=[], _correction_context=[], store=CaptionStore("latency-integration"),
            _translator=SimpleNamespace(generation=0), _emit=emit, _on_translation=on_translation)
        # No append/inference is performed. The real coordinator is idle.
        pipe = module.SemanticCaptionPipeline(session, SimpleNamespace(semantic_mode="orchestrated"))
        clock = Clock()
        pipe._latency_clock = clock
        word = Word("The report is ready.", 0, 100)
        window = SimpleNamespace(segment_id="source", track_id="track", capture_epoch="epoch",
                                 emit_start_ns=0, window_start_ns=0, window_end_ns=100)
        item = semantic_pipeline._Preview(window, SimpleNamespace(
            stable_words=(), provisional_words=(word,)), "en")
        await pipe._preview(item)
        clock.now = 11
        item.update = SimpleNamespace(stable_words=(word,), provisional_words=())
        await pipe._preview(item)
        commit = SimpleNamespace(ready_at=11, caption_metrics=lambda **kwargs: {})
        source = semantic_pipeline._Source(item, 0, word, commit, None, window, 1, "agreement")
        unit = SimpleNamespace(payload=source, source_language="en")
        result = SimpleNamespace(units=(unit,), buffer_wait_ms=100, model_ms=200, total_ms=300,
                                 stage_metrics={}, request_id="request", force_flush=False,
                                 text="보고서가 준비됐습니다.")
        clock.now = 12.1
        await pipe._commit(result)
        completed = [event for event in events if event[0] == "translation.completed"]
        self.assertEqual(len(completed), 1)
        metrics = completed[0][2]
        self.assertAlmostEqual(metrics["semantic_first_word_age_ms"], 2100)
        self.assertFalse(metrics["semantic_latency_anchor_fallback"])
        self.assertEqual(metrics["semantic_total_ms"], 300)
        caption = session.store.get_segment(completed[0][1])
        self.assertEqual(caption.source_text, word.text)
        self.assertEqual(commit.ready_at, 11)
        self.assertEqual(item.retired, {0})
        self.assertFalse(item._latency_first_seen)


if __name__ == "__main__":
    unittest.main()
