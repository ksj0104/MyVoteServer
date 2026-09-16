"""Meaning-ready dispatch and per-speaker wall-clock inactivity, without a model."""

import asyncio
from dataclasses import replace
from pathlib import Path
import runpy
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

support = runpy.run_path(str(Path(__file__).with_name("test_semantic_quality_policy.py")))
semantic, routing = support["semantic"], support["routing"]
GeneratedProvider = support["GeneratedProvider"]
install = support["install_quality_policy"]


class InactivityPolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = SimpleNamespace(
            SemanticTranslationCoordinator=semantic.SemanticTranslationCoordinator,
            SemanticTranslationRouter=support["original_router"](),
            SemanticTranslationConfig=semantic.SemanticTranslationConfig)
        support["install_routing"](self.module)
        self.metadata = install(self.module, min_request_interval_s=.005, max_hold_s=.01,
            request_timeout_s=.3, max_total_age_s=1, inactivity_flush_s=.08)
        self.commits, self.failures = [], []
        self.published = asyncio.Event()

    def unit(self, identifier, lane="A"):
        return semantic.SemanticPendingUnit(identifier, " " + identifier, "en", "track", "epoch",
            time.monotonic(), payload=object(), lane_id="boundary-g1:speaker:" + lane)

    async def committed(self, result):
        self.commits.append(result)
        self.published.set()

    async def failed(self, result):
        self.failures.append(result)
        self.published.set()

    def coordinator(self, provider):
        obj = self.module.SemanticTranslationCoordinator(provider, self.committed, self.failed,
            config=semantic.SemanticTranslationConfig(min_request_interval_s=.005,
                max_hold_s=.01, request_timeout_s=.3, max_total_age_s=1))
        self.addAsyncCleanup(obj.close, flush=False)
        return obj

    async def test_ready_source_is_not_delayed_by_inactivity_or_initial_debounce(self):
        provider = GeneratedProvider([{"action": "commit", "through_id": "ready", "text": "완성"}])
        obj = self.coordinator(provider)
        obj.config = replace(obj.config, min_request_interval_s=.25)
        await obj.append(self.unit("ready"))
        await asyncio.wait_for(self.published.wait(), .06)
        self.assertFalse(provider.requests[0].force_flush)
        self.assertEqual(len(self.commits), 1)

    async def test_wait_then_idle_translates_exact_snapshot_not_source_only_failure(self):
        provider = GeneratedProvider([{"action": "wait"},
            {"action": "commit", "through_id": "unfinished", "text": "만약"}])
        obj = self.coordinator(provider)
        unit = self.unit("unfinished")
        await obj.append(unit)
        await asyncio.wait_for(provider.entered.wait(), .05)
        await asyncio.sleep(.04)
        self.assertEqual(len(provider.requests), 1)  # max_hold does not re-query
        self.assertEqual(obj.pending_units, (unit,))
        await asyncio.wait_for(self.published.wait(), .2)
        self.assertEqual([r.force_flush for r in provider.requests], [False, True])
        self.assertIs(self.commits[0].units[0], unit)
        self.assertTrue(self.commits[0].force_flush)
        self.assertEqual(self.failures, [])
        self.assertEqual(obj.counts["inactivity_flushes"], 1)

    async def test_actual_provisional_change_extends_only_its_lane_deadline(self):
        provider = GeneratedProvider([{"action": "wait"},
            {"action": "commit", "through_id": "head", "text": "잔여"}])
        obj = self.coordinator(provider)
        await obj.append(self.unit("head"))
        await asyncio.wait_for(provider.entered.wait(), .05)
        await asyncio.sleep(.05)
        self.assertTrue(obj.note_transcription())
        await asyncio.sleep(.05)
        self.assertEqual(len(provider.requests), 1)
        await asyncio.wait_for(self.published.wait(), .15)
        self.assertEqual(len(self.commits), 1)

    async def test_new_source_before_deadline_is_checked_without_forcing(self):
        provider = GeneratedProvider([{"action": "wait"},
            {"action": "commit", "through_id": "tail", "text": "완성 문장"}])
        obj = self.coordinator(provider)
        head, tail = self.unit("head"), self.unit("tail")
        await obj.append(head)
        await asyncio.wait_for(provider.entered.wait(), .05)
        await asyncio.sleep(.015)
        await obj.append(tail)
        await asyncio.wait_for(self.published.wait(), .05)
        self.assertEqual([r.force_flush for r in provider.requests], [False, False])
        self.assertEqual(self.commits[0].units, (head, tail))

    async def test_ready_prefix_then_unfinished_suffix_has_its_own_idle_flush(self):
        provider = GeneratedProvider([
            {"action": "commit", "through_id": "head", "text": "완성"},
            {"action": "wait"},
            {"action": "commit", "through_id": "tail", "text": "내일은"}])
        obj = self.coordinator(provider)
        head, tail = self.unit("head"), self.unit("tail")
        await obj.append(head)
        await obj.append(tail)
        await asyncio.sleep(.16)
        self.assertEqual([r.force_flush for r in provider.requests], [False, False, True])
        self.assertEqual([r.units for r in self.commits], [(head,), (tail,)])
        self.assertEqual(self.failures, [])

    async def test_forced_wait_is_invalid_not_another_unbounded_wait(self):
        provider = GeneratedProvider([{"action": "wait"}, {"action": "wait"}])
        obj = self.coordinator(provider)
        await obj.append(self.unit("head"))
        await asyncio.wait_for(self.published.wait(), .2)
        self.assertEqual(self.commits, [])
        self.assertEqual(self.failures[0].reason, "invalid_model_response")
        self.assertEqual(obj.pending_units, ())

    async def test_model_wait_finishing_after_idle_deadline_immediately_drains(self):
        provider = GeneratedProvider([{"action": "wait"},
            {"action": "commit", "through_id": "head", "text": "잔여"}], pause=True)
        obj = self.coordinator(provider)
        await obj.append(self.unit("head"))
        await asyncio.wait_for(provider.entered.wait(), .05)
        await asyncio.sleep(.1)
        provider.release.set()
        await asyncio.wait_for(self.published.wait(), .05)
        self.assertEqual([r.force_flush for r in provider.requests], [False, True])
        self.assertEqual(self.failures, [])

    async def test_forced_snapshot_does_not_consume_words_added_during_inference(self):
        provider = GeneratedProvider([
            {"action": "commit", "through_id": "old", "text": "이전 구간"},
            {"action": "commit", "through_id": "new", "text": "다음 구간"}], pause=True)
        obj = self.coordinator(provider)
        old, new = self.unit("old"), self.unit("new")
        await obj.append(old)
        obj.request_flush()
        await asyncio.wait_for(provider.entered.wait(), .05)
        await obj.append(new)
        provider.release.set()
        await asyncio.sleep(.04)
        self.assertEqual([r.units for r in self.commits], [(old,), (new,)])
        self.assertEqual([r.force_flush for r in provider.requests], [True, False])
        self.assertEqual(self.failures, [])

    async def test_close_drain_translates_residual_without_waiting_for_silence(self):
        provider = GeneratedProvider([{"action": "commit", "through_id": "head", "text": "잔여"}])
        obj = self.coordinator(provider)
        await obj.append(self.unit("head"))
        await asyncio.wait_for(obj.flush(), .05)
        self.assertTrue(provider.requests[0].force_flush)

    async def test_reset_cancels_timer_and_rejects_stale_result(self):
        provider = GeneratedProvider([{"action": "commit", "through_id": "head", "text": "낡은 결과"}], pause=True)
        obj = self.coordinator(provider)
        await obj.append(self.unit("head"))
        await asyncio.wait_for(provider.entered.wait(), .05)
        await obj.reset()
        provider.release.set()
        await asyncio.sleep(.12)
        self.assertIsNone(obj._last_transcription_at)
        self.assertEqual(self.commits, [])
        self.assertEqual(obj.pending_units, ())

    async def test_other_speaker_is_not_head_blocked_and_does_not_reset_old_timer(self):
        class PerLaneProvider:
            def __init__(self):
                self.requests = []

            async def semantic_stream(inner, request):
                import json
                inner.requests.append(request)
                first = request.units[0].unit_id
                packet = ({"action": "wait"} if first == "A" and not request.force_flush else
                          {"action": "commit", "through_id": request.units[-1].unit_id, "text": first})
                yield support["Chunk"](json.dumps(packet), completed=True, finish_reason="stop")

        provider = PerLaneProvider()
        router = self.module.SemanticTranslationRouter(provider, self.committed, self.failed)
        self.addAsyncCleanup(router.close, flush=False)
        a, b = self.unit("A"), self.unit("B", "B")
        with patch.object(routing, "SemanticTranslationCoordinator", self.module.SemanticTranslationCoordinator):
            await router.append(a)
            await asyncio.sleep(.02)
            old_clock = router._lanes[a.scope]._last_transcription_at
            await router.append(b)
        self.assertEqual(router._lanes[a.scope]._last_transcription_at, old_clock)
        await asyncio.wait_for(self.published.wait(), .04)
        self.assertEqual(self.commits[0].units, (b,))
        await asyncio.sleep(.1)
        self.assertEqual([r.units for r in self.commits], [(b,), (a,)])
        self.assertEqual(self.failures, [])

    async def test_configuration_metadata_and_invalid_values(self):
        self.assertEqual(self.metadata["semantic_inactivity_flush_ms"], 80)
        self.assertEqual(self.metadata["semantic_quality_policy"], "complete-prefix-inactivity-v3")
        for value in (True, -1, float("nan"), float("inf"), 45, "2"):
            module = SimpleNamespace(SemanticTranslationCoordinator=semantic.SemanticTranslationCoordinator,
                SemanticTranslationRouter=support["original_router"](),
                SemanticTranslationConfig=semantic.SemanticTranslationConfig)
            with self.subTest(value=value), self.assertRaises(ValueError):
                install(module, inactivity_flush_s=value)


if __name__ == "__main__":
    unittest.main()
