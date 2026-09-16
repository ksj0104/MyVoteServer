"""Real verified coordinator/router tests with in-process generated responses.

No model, network, audio file or server process is used. Run with the demo venv.
"""

import asyncio
import json
from pathlib import Path
import runpy
import time
import unittest
from types import SimpleNamespace

try:
    import httpx  # noqa: F401 -- required by the verified translation module
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        "Semantic routing tests require httpx; use MyVote-Mac-Demo/.venv/bin/python"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
UPDATE = ROOT / "MyVote-Speaker-Update"
support = runpy.run_path(str(ROOT / "tests/verified_engine.py"))
manifest = support["load_verified_engine"]()

from myvote_engine import semantic_routing as routing
from myvote_engine import semantic_translation as semantic
from myvote_engine.translation import Chunk

routing_support = runpy.run_path(str(ROOT / "scripts/semantic_boundary_routing.py"))
install_routing = routing_support["install_routing"]
install_routing(routing)


class RecordingProvider:
    def __init__(self, respond=None):
        self.requests = []
        self.respond = respond

    async def semantic_stream(self, request):
        self.requests.append(request)
        packet = (self.respond(request) if self.respond is not None else {
            "action": "commit", "through_id": request.units[-1].unit_id,
            "text": "generated result",
        })
        yield Chunk(json.dumps(packet), completed=True, finish_reason="stop")


class SemanticBoundaryRoutingTests(unittest.IsolatedAsyncioTestCase):
    def make(self, *, router=False, provider=None, config=None, **kwargs):
        self.commits, self.failures = [], []
        self.published = asyncio.Event()

        async def committed(result):
            self.commits.append(result)
            self.published.set()

        async def failed(result):
            self.failures.append(result)
            self.published.set()

        provider = provider or RecordingProvider()
        cls = (routing.SemanticTranslationRouter if router
               else routing.SemanticTranslationCoordinator)
        instance = cls(provider, committed, failed, config=config, **kwargs)
        self.addAsyncCleanup(instance.close, flush=False)
        return instance, provider

    def unit(self, identifier, *, lane="speaker:A", age=0, text=None):
        return semantic.SemanticPendingUnit(
            identifier, text or " " + identifier, "en", "track", "epoch",
            time.monotonic() - age, payload=object(), lane_id=lane,
        )

    def fast_config(self, **overrides):
        values = dict(min_request_interval_s=.001, max_hold_s=.2,
                      request_timeout_s=.1, max_total_age_s=1)
        values.update(overrides)
        return semantic.SemanticTranslationConfig(**values)

    async def wait_count(self, count):
        async with asyncio.timeout(1):
            while len(self.commits) + len(self.failures) < count:
                self.published.clear()
                await self.published.wait()

    async def test_install_is_idempotent_and_preserves_base_methods_and_defaults(self):
        coordinator_type = routing.SemanticTranslationCoordinator
        router_type = routing.SemanticTranslationRouter
        metadata = install_routing(routing)
        metadata["semantic_boundary_routing"] = "mutated by caller"
        self.assertEqual(install_routing(routing),
                         {"semantic_boundary_routing": "soft-lane-v1"})
        self.assertIs(routing.SemanticTranslationCoordinator, coordinator_type)
        self.assertIs(routing.SemanticTranslationRouter, router_type)
        self.assertIs(coordinator_type._myvote_boundary_original,
                      semantic.SemanticTranslationCoordinator)
        for name in ("_loop", "_query_active", "_collect", "_failure"):
            self.assertIs(getattr(coordinator_type, name),
                          getattr(semantic.SemanticTranslationCoordinator, name))
        coordinator, _ = self.make()
        self.assertEqual(coordinator.config, semantic.SemanticTranslationConfig())
        self.assertEqual(coordinator.config.max_hold_s, 4)
        self.assertEqual(coordinator.config.max_total_age_s, 8)

    async def test_soft_boundary_allows_shorter_prefix_without_swallowing_next_turn(self):
        provider = RecordingProvider(lambda request: {
            "action": "commit", "through_id": request.units[0].unit_id,
            "text": "generated prefix",
        })
        coordinator, _ = self.make(provider=provider, config=self.fast_config())
        units = tuple(self.unit(identifier) for identifier in ("ready", "tail", "next"))
        await coordinator.append(units[0])
        await coordinator.append(units[1])
        self.assertTrue(coordinator.request_boundary())
        await coordinator.append(units[2])
        await self.wait_count(3)
        self.assertEqual([tuple(unit.unit_id for unit in request.units)
                          for request in provider.requests],
                         [("ready", "tail"), ("tail",), ("next",)])
        self.assertTrue(all(not request.force_flush for request in provider.requests))
        self.assertEqual(tuple(unit for result in self.commits for unit in result.units), units)
        self.assertEqual(coordinator._soft_boundaries, set())

    async def test_router_boundary_and_scoped_hard_flush_affect_only_requested_lane(self):
        router, _ = self.make(router=True)
        a, b = self.unit("a"), self.unit("b", lane="speaker:B")
        await router.append(a)
        await router.append(b, boundary=True)
        lane_a, lane_b = router._lanes[a.scope], router._lanes[b.scope]
        self.assertEqual(lane_a._soft_boundaries, set())
        self.assertEqual(lane_a._boundaries, set())
        self.assertEqual(lane_b._head(), ((b,), False))
        self.assertTrue(router.request_flush(b.scope))
        self.assertEqual(lane_b._head(), ((b,), True))
        self.assertEqual(lane_a._boundaries, set())
        self.assertFalse(router.request_boundary(("missing",)))
        self.assertFalse(router.request_flush(("missing",)))

    async def test_a_b_a_turns_do_not_merge_across_a_soft_boundary(self):
        router, provider = self.make(router=True, config=self.fast_config())
        a1, b1, a2 = (self.unit("a1"), self.unit("b1", lane="speaker:B"), self.unit("a2"))
        await router.append(a1)
        router.request_boundary(a1.scope)
        await router.append(b1, boundary=True)
        await router.append(a2)
        await self.wait_count(3)
        self.assertCountEqual([tuple(unit.unit_id for unit in request.units)
                               for request in provider.requests], [("a1",), ("b1",), ("a2",)])
        self.assertTrue(all(not request.force_flush for request in provider.requests))
        self.assertEqual(self.failures, [])

    async def test_hard_stop_promotes_all_soft_boundaries_and_drains_each_turn(self):
        router, provider = self.make(router=True, config=self.fast_config())
        for identifier in ("a1", "a2"):
            await router.append(self.unit(identifier), boundary=True)
        await router.append(self.unit("a3"))
        await router.append(self.unit("b1", lane="speaker:B"), boundary=True)
        await asyncio.wait_for(router.flush(), timeout=1)
        self.assertEqual(len(self.commits), 4)
        self.assertTrue(all(request.force_flush for request in provider.requests))
        self.assertCountEqual([tuple(unit.unit_id for unit in request.units)
                               for request in provider.requests],
                              [("a1",), ("a2",), ("a3",), ("b1",)])
        self.assertEqual(router.pending_units, ())
        self.assertTrue(all(not lane._soft_boundaries for lane in router._lanes.values()))

    async def test_quality_resume_after_hold_requires_opt_in_and_respects_total_deadline(self):
        quality = SimpleNamespace(SemanticTranslationCoordinator=routing.SemanticTranslationCoordinator,
                                  SemanticTranslationRouter=routing.SemanticTranslationRouter,
                                  SemanticTranslationConfig=routing.SemanticTranslationConfig)
        runpy.run_path(str(ROOT / "scripts/semantic_quality_policy.py"))["install_quality_policy"](quality)

        async def ignored(result):
            pass

        config = self.fast_config(max_hold_s=.05, max_total_age_s=1)
        for enabled, age, hard, expected in ((False, .1, False, False), (True, .1, False, True),
                                             (True, 2, False, False), (True, .1, True, False)):
            with self.subTest(enabled=enabled, age=age, hard=hard):
                coordinator = quality.SemanticTranslationCoordinator(RecordingProvider(), ignored, ignored, config=config)
                coordinator._ensure_worker = lambda: None
                self.addAsyncCleanup(coordinator.close, flush=False)
                unit = self.unit("older", age=age)
                await coordinator.append(unit)
                coordinator.request_boundary()
                if hard:
                    coordinator.request_flush()
                self.assertEqual(coordinator.resume_boundary(unit.unit_id, allow_after_hold=enabled), expected)
        legacy, _ = self.make(config=config)
        legacy._ensure_worker = lambda: None
        unit = self.unit("legacy", age=.1)
        await legacy.append(unit)
        legacy.request_boundary()
        self.assertFalse(legacy.resume_boundary(unit.unit_id, allow_after_hold=True))

    async def test_exact_pending_boundary_can_resume_and_include_new_source(self):
        router, _ = self.make(router=True)
        first, continuation = self.unit("first"), self.unit("continuation")
        await router.append(first, boundary=True)
        self.assertFalse(router.resume_boundary(first.scope, "not-the-marker"))
        self.assertTrue(router.resume_boundary(first.scope, first.unit_id))
        await router.append(continuation)
        self.assertEqual(router._lanes[first.scope]._head(), ((first, continuation), False))
        self.assertIs(router.pending_units[0], first)

    async def test_resume_cannot_remove_hard_drain_or_an_earlier_turn_boundary(self):
        for hard in (False, True):
            with self.subTest(hard=hard):
                router, _ = self.make(router=True)
                first, next_turn = self.unit("first"), self.unit("next-turn")
                await router.append(first, boundary=True)
                if hard:
                    router.request_flush()
                else:
                    await router.append(next_turn)
                self.assertFalse(router.resume_boundary(first.scope, first.unit_id))
                self.assertIn(first.unit_id, router._lanes[first.scope]._soft_boundaries)

    async def test_resume_cannot_reopen_expired_or_consumed_source(self):
        router, _ = self.make(router=True)
        old = self.unit("old", age=5)
        await router.append(old, boundary=True)
        self.assertFalse(router.resume_boundary(old.scope, old.unit_id))
        router._lanes[old.scope]._remove_prefix(1)
        self.assertFalse(router.resume_boundary(old.scope, old.unit_id))
        self.assertFalse(router.resume_boundary(("missing",), old.unit_id))

    async def test_resume_is_disabled_while_closing(self):
        router, _ = self.make(router=True)
        unit = self.unit("source")
        await router.append(unit, boundary=True)
        lane = router._lanes[unit.scope]
        lane._closing = True
        self.assertFalse(router.resume_boundary(unit.scope, unit.unit_id))
        lane._closing = False
        router._closing = True
        self.assertFalse(router.resume_boundary(unit.scope, unit.unit_id))

    async def test_later_hard_marker_does_not_force_an_earlier_soft_head(self):
        coordinator, _ = self.make()
        first, second = self.unit("first"), self.unit("second")
        await coordinator.append(first)
        coordinator.request_boundary()
        await coordinator.append(second, boundary=True)
        self.assertEqual(coordinator._head(), ((first,), False))
        coordinator.request_flush()
        self.assertEqual(coordinator._head(), ((first,), True))

    async def test_max_hold_still_forces_a_waiting_natural_boundary(self):
        provider = RecordingProvider(lambda request: ({
            "action": "commit", "through_id": request.units[-1].unit_id, "text": "forced result",
        } if request.force_flush else {"action": "wait"}))
        coordinator, _ = self.make(provider=provider,
                                   config=self.fast_config(max_hold_s=.015))
        await coordinator.append(self.unit("pending"))
        coordinator.request_boundary()
        await self.wait_count(1)
        self.assertGreaterEqual(len(provider.requests), 2)
        self.assertFalse(provider.requests[0].force_flush)
        self.assertTrue(provider.requests[-1].force_flush)
        self.assertEqual(len(self.commits), 1)
        self.assertEqual(self.failures, [])

    async def test_expired_soft_source_preserves_original_payload_without_inference(self):
        coordinator, provider = self.make(
            config=self.fast_config(max_hold_s=.01, max_total_age_s=.02))
        unit = self.unit("old", age=.03, text=" 원문 그대로")
        await coordinator.append(unit)
        coordinator.request_boundary()
        await self.wait_count(1)
        self.assertEqual(provider.requests, [])
        self.assertEqual(self.commits, [])
        self.assertEqual(self.failures[0].reason, "source_deadline")
        self.assertIs(self.failures[0].units[0], unit)
        self.assertEqual(coordinator._soft_boundaries, set())

    async def test_timeout_preserves_soft_source_and_uses_upstream_cancellation(self):
        cancelled = asyncio.Event()

        class SlowProvider:
            async def semantic_stream(self, request):
                try:
                    await asyncio.Future()
                    yield Chunk("unreachable")
                finally:
                    cancelled.set()

        coordinator, _ = self.make(provider=SlowProvider(),
                                   config=self.fast_config(request_timeout_s=.01))
        unit = self.unit("timeout")
        await coordinator.append(unit)
        coordinator.request_boundary()
        await self.wait_count(1)
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        self.assertEqual(self.failures[0].reason, "model_timeout")
        self.assertIs(self.failures[0].units[0], unit)
        self.assertEqual(coordinator.counts["timeouts"], 1)
        self.assertEqual(coordinator._soft_boundaries, set())

    async def test_capacity_rejection_retains_upstream_global_hard_flush_and_source(self):
        router, _ = self.make(router=True, max_total_units=1)
        first, rejected = self.unit("first"), self.unit("rejected", lane="speaker:B")
        await router.append(first, boundary=True)
        await router.append(rejected)
        self.assertEqual(router._lanes[first.scope]._head(), ((first,), True))
        self.assertEqual(self.failures[0].reason, "buffer_capacity")
        self.assertIs(self.failures[0].units[0], rejected)

    async def test_reset_clears_soft_markers_and_preserves_reset_failure(self):
        coordinator, _ = self.make()
        unit = self.unit("reset")
        await coordinator.append(unit)
        coordinator.request_boundary()
        await coordinator.reset()
        self.assertEqual(coordinator._soft_boundaries, set())
        self.assertEqual(coordinator.pending_units, ())
        self.assertEqual(self.failures[0].reason, "reset")
        self.assertIs(self.failures[0].units[0], unit)

    async def test_dialogue_context_is_partitioned_by_capture_and_boundary_generation(self):
        router, _ = self.make(router=True)
        current = self.unit("current", lane="boundary-g2:speaker:A").scope
        same = self.unit("same", lane="boundary-g2:speaker:B").scope
        old = self.unit("old", lane="boundary-g1:speaker:A").scope
        generic = self.unit("legacy").scope
        other_track = ("other-track", *current[1:])
        other_epoch = (current[0], "other-epoch", *current[2:])
        router._dialogue = [
            (current, "older matching context"),
            (same, "matching context one"),
            (current, "matching context two"),
            (old, "late old-generation commit"),
            (generic, "legacy context"),
            (other_track, "other track"),
            (other_epoch, "other epoch"),
        ]
        self.assertEqual(router._context_for(current),
                         ("matching context one", "matching context two"))
        self.assertEqual(router._context_for(old), ("late old-generation commit",))
        self.assertEqual(router._context_for(generic), ("legacy context",))


if __name__ == "__main__":
    unittest.main()
