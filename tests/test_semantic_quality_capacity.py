"""Bounded request projection over verified code; no model/network calls."""

import asyncio
import json
from pathlib import Path
import runpy
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import httpx  # noqa: F401 -- the verified translation module dependency
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Use MyVote-Mac-Demo/.venv/bin/python for capacity tests") from exc


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "tests/verified_engine.py"))["load_verified_engine"]()

from myvote_engine import semantic_routing as routing
from myvote_engine import semantic_translation as semantic
from myvote_engine.translation import Chunk

install_routing = runpy.run_path(str(ROOT / "scripts/semantic_boundary_routing.py"))["install_routing"]
capacity_support = runpy.run_path(str(ROOT / "scripts/semantic_quality_capacity.py"))
install_capacity = capacity_support["install_quality_capacity"]
install_policy = runpy.run_path(str(ROOT / "scripts/semantic_quality_policy.py"))["install_quality_policy"]


def original_router():
    cls = routing.SemanticTranslationRouter
    while True:
        previous = next((cls.__dict__[name] for name in (
            "_myvote_quality_original", "_myvote_capacity_original", "_myvote_boundary_original"
        ) if name in cls.__dict__), None)
        if previous is None:
            return cls
        cls = previous


class GeneratedProvider:
    def __init__(self, respond=None, *, pause=False, ignore_cancel=False):
        self.respond = respond
        self.requests = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.ignore_cancel = ignore_cancel
        if not pause:
            self.release.set()

    async def semantic_stream(self, request):
        self.requests.append(request)
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            if not self.ignore_cancel:
                raise
            await self.release.wait()
        packet = self.respond(request) if self.respond else {
            "action": "commit", "through_id": request.units[-1].unit_id,
            "text": "완결된 번역입니다.",
        }
        yield Chunk(json.dumps(packet), completed=True, finish_reason="stop",
                    usage={"stage_ms": {"selection": 2.0, "translation": 3.0}})


class SemanticQualityCapacityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = SimpleNamespace(
            SemanticTranslationCoordinator=semantic.SemanticTranslationCoordinator,
            SemanticTranslationRouter=original_router(),
            SemanticTranslationConfig=semantic.SemanticTranslationConfig,
        )
        install_routing(self.module)
        self.soft_coordinator = self.module.SemanticTranslationCoordinator
        self.soft_router = self.module.SemanticTranslationRouter
        install_capacity(self.module)
        self.capacity_coordinator = self.module.SemanticTranslationCoordinator
        install_policy(self.module)
        # Exercise real query/publication methods without timing a scheduler.
        worker = patch.object(self.module.SemanticTranslationCoordinator, "_ensure_worker", lambda instance: None)
        worker.start()
        self.addCleanup(worker.stop)
        self.commits, self.failures = [], []

    def units(self, count, *, start=1, lane="boundary-g1:speaker:A", text=None):
        return tuple(semantic.SemanticPendingUnit(
            f"u{index}", text if text is not None else (" end." if index == start + count - 1 else " word"),
            "en", "track", "epoch", time.monotonic(),
            payload=SimpleNamespace(word=object(), timing=object(), index=index), lane_id=lane,
        ) for index in range(start, start + count))

    def make(self, provider=None, *, router=False, **kwargs):
        async def committed(result):
            self.commits.append(result)

        async def failed(result):
            self.failures.append(result)

        provider = provider or GeneratedProvider()
        cls = self.module.SemanticTranslationRouter if router else self.module.SemanticTranslationCoordinator
        kwargs.setdefault("config", semantic.SemanticTranslationConfig(
            min_request_interval_s=.001, max_hold_s=1,
            request_timeout_s=1, max_total_age_s=5,
        ))
        instance = cls(provider, committed, failed, **kwargs)
        self.addAsyncCleanup(instance.close, flush=False)
        return instance, provider

    async def append_all(self, instance, units):
        for unit in units:
            await instance.append(unit)

    async def test_installer_is_scoped_idempotent_and_preserves_config_validation(self):
        global_coordinator = routing.SemanticTranslationCoordinator
        metadata = install_capacity(self.module)
        self.assertEqual(metadata, {
            "semantic_quality_capacity": "original-words-request-blocks-v1",
            "semantic_pending_unit_capacity": 256, "semantic_request_unit_limit": 64,
        })
        metadata["semantic_pending_unit_capacity"] = 0
        self.assertEqual(install_capacity(self.module)["semantic_pending_unit_capacity"], 256)
        self.assertIs(routing.SemanticTranslationCoordinator, global_coordinator)
        self.assertIs(self.capacity_coordinator._myvote_capacity_original, self.soft_coordinator)
        for method in ("_loop", "_query", "_collect", "_failure", "_head", "reset"):
            self.assertIs(getattr(self.capacity_coordinator, method), getattr(self.soft_coordinator, method))
        with self.assertRaises(ValueError):
            semantic.SemanticTranslationConfig(max_units=65)
        router, _ = self.make(router=True)
        self.assertEqual(router.config.max_units, 64)
        self.assertEqual(router.max_total_units, 256)
        self.assertEqual(router.config.request_timeout_s, 20)
        for invalid in (True, 0, 257, 1.5):
            with self.assertRaises(ValueError):
                self.module.SemanticTranslationRouter(None, None, None, max_total_units=invalid)
            with self.assertRaises(ValueError):
                self.module.SemanticTranslationCoordinator(None, None, None, max_pending_units=invalid)

    async def test_65_original_words_keep_ending_and_timing_payload_identity(self):
        coordinator, provider = self.make()
        units = self.units(65)
        await self.append_all(coordinator, units)
        await coordinator._query(*coordinator._head())
        self.assertEqual(self.failures, [])
        self.assertEqual(len(self.commits), 1)
        self.assertEqual(self.commits[0].units, units)
        for original, consumed in zip(units, self.commits[0].units):
            self.assertIs(consumed, original)
            self.assertIs(consumed.payload, original.payload)
            self.assertIs(consumed.payload.word, original.payload.word)
            self.assertIs(consumed.payload.timing, original.payload.timing)
        request = provider.requests[0]
        self.assertLessEqual(len(request.units), 64)
        self.assertEqual(request.units[-1].unit_id, "u65")
        self.assertEqual("".join(unit.text for unit in request.units), "".join(unit.text for unit in units))
        self.assertFalse(request.force_flush)
        self.assertEqual(self.commits[0].stage_metrics, {"selection": 2.0, "translation": 3.0})
        self.assertEqual(coordinator.counts["committed_units"], 65)

    async def test_router_150_words_constructs_quality_child_and_commits_all(self):
        router, provider = self.make(router=True)
        units = self.units(150)
        await self.append_all(router, units)
        self.assertEqual(router.pending_units, units)
        lane = router._lanes[units[0].scope]
        self.assertIsInstance(lane, self.module.SemanticTranslationCoordinator)
        self.assertEqual(lane.max_pending_units, 256)
        await lane._query(*lane._head())
        self.assertEqual(self.failures, [])
        self.assertEqual(self.commits[0].units, units)
        self.assertEqual(len(provider.requests[0].units), 50)
        self.assertEqual(provider.requests[0].units[-1].unit_id, "u150")
        self.assertEqual(router.pending_units, ())

    async def test_partial_group_commit_maps_only_original_endpoint_prefix(self):
        provider = GeneratedProvider(lambda request: {
            "action": "commit", "through_id": request.units[2].unit_id, "text": "prefix",
        })
        coordinator, _ = self.make(provider)
        units = self.units(150)
        await self.append_all(coordinator, units)
        await coordinator._query(*coordinator._head())
        self.assertEqual(self.commits[0].units, units[:9])
        self.assertEqual(coordinator.pending_units, units[9:])
        self.assertEqual(coordinator._context, ["".join(unit.text for unit in units[:9]).strip()])

    async def test_inside_group_and_foreign_ids_fail_without_any_translation(self):
        for identifier in ("u1", "foreign"):
            with self.subTest(identifier=identifier):
                self.commits.clear()
                self.failures.clear()
                coordinator, _ = self.make(GeneratedProvider(lambda request, identifier=identifier: {
                    "action": "commit", "through_id": identifier, "text": "must not publish",
                }))
                units = self.units(65)
                await self.append_all(coordinator, units)
                await coordinator._query(*coordinator._head())
                self.assertEqual(self.commits, [])
                self.assertEqual(self.failures[0].reason, "invalid_model_response")
                self.assertEqual(self.failures[0].units, units)

    async def test_append_during_request_is_not_consumed_by_older_snapshot(self):
        provider = GeneratedProvider(pause=True)
        coordinator, _ = self.make(provider)
        units, later = self.units(65), self.units(1, start=66)
        await self.append_all(coordinator, units)
        task = asyncio.create_task(coordinator._query(*coordinator._head()))
        await asyncio.wait_for(provider.entered.wait(), 1)
        await self.append_all(coordinator, later)
        provider.release.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(self.commits[0].units, units)
        self.assertEqual(coordinator.pending_units, later)
        self.assertNotIn(later[0].unit_id, [unit.unit_id for unit in provider.requests[0].units])

    async def test_reset_rejects_cancellation_resistant_old_snapshot(self):
        provider = GeneratedProvider(pause=True, ignore_cancel=True)
        coordinator, _ = self.make(provider)
        units = self.units(65)
        await self.append_all(coordinator, units)
        task = asyncio.create_task(coordinator._query(*coordinator._head()))
        await asyncio.wait_for(provider.entered.wait(), 1)
        await coordinator.reset()
        later = self.units(1, start=1000)
        await self.append_all(coordinator, later)
        provider.release.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(self.commits, [])
        self.assertEqual(self.failures[0].reason, "reset")
        self.assertEqual(self.failures[0].units, units)
        self.assertEqual(coordinator.pending_units, later)
        self.assertEqual(coordinator.counts["stale"], 1)

    async def test_hard_and_soft_boundaries_never_group_across_source_turn(self):
        for hard in (False, True):
            with self.subTest(hard=hard):
                self.commits.clear()
                coordinator, provider = self.make()
                first, second = self.units(65), self.units(20, start=66)
                await self.append_all(coordinator, first)
                coordinator.request_flush() if hard else coordinator.request_boundary()
                await self.append_all(coordinator, second)
                self.assertEqual(coordinator._head(), (first, hard))
                await coordinator._query(*coordinator._head())
                self.assertEqual(self.commits[0].units, first)
                self.assertEqual(coordinator.pending_units, second)
                self.assertEqual(provider.requests[0].units[-1].unit_id, "u65")
                self.assertFalse(provider.requests[0].force_flush)

    async def test_capacity_257_is_bounded_for_router_and_standalone_lane(self):
        for router in (False, True):
            with self.subTest(router=router):
                self.failures.clear()
                instance, _ = self.make(router=router)
                units = self.units(257, text=" w")
                await self.append_all(instance, units)
                self.assertEqual(instance.pending_units, units[:256])
                self.assertEqual(len(self.failures), 1)
                self.assertEqual(self.failures[0].reason, "buffer_capacity")
                self.assertIs(self.failures[0].units[0], units[-1])
                lane = instance._lanes[units[0].scope] if router else instance
                self.assertIn(units[255].unit_id, lane._boundaries)

    async def test_global_capacity_and_duplicate_checks_span_all_lanes(self):
        router, _ = self.make(router=True)
        first, second = self.units(128), self.units(128, start=129, lane="boundary-g1:speaker:B")
        await self.append_all(router, first + second)
        extra = self.units(1, start=257, lane="boundary-g1:speaker:C")[0]
        await router.append(extra)
        self.assertEqual(len(router.pending_units), 256)
        self.assertEqual(self.failures[0].units, (extra,))
        with self.assertRaises(ValueError):
            await router.append(self.units(1, start=1, lane="boundary-g1:speaker:C")[0])

    async def test_4000_character_cap_is_not_extended_by_word_capacity(self):
        for router in (False, True):
            with self.subTest(router=router):
                self.failures.clear()
                instance, _ = self.make(router=router,
                    config=semantic.SemanticTranslationConfig(max_source_chars=12000))
                first = self.units(1, text="x" * 4000)[0]
                extra = self.units(1, start=2, text="y")[0]
                await instance.append(first)
                await instance.append(extra)
                self.assertEqual(instance.pending_units, (first,))
                self.assertEqual(self.failures[0].reason, "buffer_capacity")

    async def test_small_queries_delegate_and_keep_original_single_word_ids(self):
        coordinator, provider = self.make()
        units = self.units(64)
        await self.append_all(coordinator, units)
        calls = []

        async def delegated(instance, original, forced):
            calls.append((instance, original, forced))
            return await semantic.SemanticTranslationCoordinator._query_active(instance, original, forced)

        with patch.object(self.soft_coordinator, "_query_active", delegated):
            await coordinator._query(*coordinator._head())
        self.assertEqual(calls, [(coordinator, units, False)])
        self.assertEqual([unit.unit_id for unit in provider.requests[0].units], [unit.unit_id for unit in units])
        self.assertEqual(self.commits[0].units, units)

    async def test_projection_preserves_exact_multilingual_spacing_and_bound(self):
        original = tuple(semantic.SemanticPendingUnit(
            f"mixed{index}", (" 회의" if index % 2 else "가"), "ko", "track", "epoch",
            time.monotonic(), payload=object(),
        ) for index in range(256))
        projected = capacity_support["project_request_units"](original, semantic.SemanticUnit)
        self.assertEqual(len(projected), 64)
        self.assertEqual("".join(unit.text for unit in projected), "".join(unit.text for unit in original))
        self.assertEqual([unit.unit_id for unit in projected], [original[index].unit_id for index in range(3, 256, 4)])

    async def test_grouped_timeout_keeps_original_failure_and_late_result_cannot_publish(self):
        provider = GeneratedProvider(pause=True, ignore_cancel=True)
        coordinator, _ = self.make(provider, config=semantic.SemanticTranslationConfig(
            min_request_interval_s=.001, max_hold_s=1, request_timeout_s=.01, max_total_age_s=5))
        units = self.units(65)
        await self.append_all(coordinator, units)
        await coordinator._query(*coordinator._head())
        self.assertEqual(self.commits, [])
        self.assertEqual(self.failures[0].reason, "model_timeout")
        self.assertEqual(self.failures[0].units, units)
        physical = coordinator._request_task
        self.assertIsNotNone(physical)
        provider.release.set()
        await asyncio.wait_for(physical, 1)
        self.assertEqual(self.commits, [])


if __name__ == "__main__":
    unittest.main()
