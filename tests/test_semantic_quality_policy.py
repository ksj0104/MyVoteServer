"""Quality-policy regression tests over verified code, with no model or network."""

import asyncio
import json
from pathlib import Path
import runpy
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import httpx  # noqa: F401 -- verified translation module dependency
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        "Quality policy tests require httpx; use MyVote-Mac-Demo/.venv/bin/python"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "tests/verified_engine.py"))["load_verified_engine"]()

from myvote_engine import semantic_routing as routing
from myvote_engine import semantic_translation as semantic
from myvote_engine import orchestrated_translation as orchestration
from myvote_engine.translation import Chunk, SemanticUnit, ProviderError

install_routing = runpy.run_path(
    str(ROOT / "scripts/semantic_boundary_routing.py")
)["install_routing"]
install_quality_policy = runpy.run_path(
    str(ROOT / "scripts/semantic_quality_policy.py")
)["install_quality_policy"]


def original_router():
    cls = routing.SemanticTranslationRouter
    while True:
        original = cls.__dict__.get("_myvote_quality_original")
        if original is None:
            original = cls.__dict__.get("_myvote_boundary_original")
        if original is None:
            return cls
        cls = original


class GeneratedProvider:
    def __init__(self, packets, *, pause=False):
        self.packets = list(packets)
        self.requests = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        if not pause:
            self.release.set()

    async def semantic_stream(self, request):
        self.requests.append(request)
        self.entered.set()
        await self.release.wait()
        if not self.packets:
            raise AssertionError("Unexpected generated-model request")
        yield Chunk(json.dumps(self.packets.pop(0)), completed=True, finish_reason="stop")


class SemanticQualityPolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Do not mutate the shared engine module: other test modules verify the
        # legacy and boundary-only policies in the same discovery process.
        self.module = SimpleNamespace(
            SemanticTranslationCoordinator=semantic.SemanticTranslationCoordinator,
            SemanticTranslationRouter=original_router(),
            SemanticTranslationConfig=semantic.SemanticTranslationConfig,
        )
        install_routing(self.module)
        self.soft_coordinator = self.module.SemanticTranslationCoordinator
        install_quality_policy(self.module)
        self.commits, self.failures, self.active_at_failure = [], [], []
        self.published = asyncio.Event()

    def unit(self, identifier, text=None):
        return semantic.SemanticPendingUnit(
            identifier, text or " " + identifier, "en", "track", "epoch",
            time.monotonic(), payload=object(), lane_id="boundary-g1:speaker:A",
        )

    def make(self, provider, *, scheduled=False, config=None):
        instance = None

        async def committed(result):
            self.commits.append(result)
            self.published.set()

        async def failed(result):
            self.failures.append(result)
            self.active_at_failure.append(instance.unsettled_units)
            self.published.set()

        instance = self.module.SemanticTranslationCoordinator(
            provider, committed, failed,
            config=config or semantic.SemanticTranslationConfig(
                min_request_interval_s=.001, max_hold_s=.3,
                request_timeout_s=.5, max_total_age_s=1,
            ),
        )
        if not scheduled:
            # Isolate _query's snapshot/publication contract from its scheduler.
            instance._ensure_worker = lambda: None
        self.addAsyncCleanup(instance.close, flush=False)
        return instance

    def http_provider(self, replies):
        requests, pending = [], list(replies)

        def handler(request):
            requests.append((request.url.path, json.loads(request.content)))
            self.assertTrue(pending, "Unexpected mocked inference request")
            expected_path, text = pending.pop(0)
            self.assertEqual(request.url.path, expected_path)
            field = ({"delta": {"content": text}}
                     if expected_path == "/v1/chat/completions" else {"text": text})
            packets = [
                {"choices": [{"index": 0, **field, "finish_reason": None}]},
                {"choices": [{"index": 0, "finish_reason": "stop"}]},
            ]
            body = "".join("data: " + json.dumps(packet, ensure_ascii=False) + "\n\n"
                           for packet in packets) + "data: [DONE]\n\n"
            return httpx.Response(200, content=body.encode("utf-8"),
                                  headers={"content-type": "text/event-stream"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
        self.addAsyncCleanup(client.aclose)
        selection_module = SimpleNamespace(
            _SelectionProvider=orchestration._SelectionProvider,
            SemanticTranslationRequest=orchestration.SemanticTranslationRequest,
        )
        runpy.run_path(str(ROOT / "scripts/gemma_json_compat.py"))[
            "install_selection_schema"
        ](selection_module)
        with patch.object(orchestration, "_SelectionProvider", selection_module._SelectionProvider):
            provider = orchestration.OrchestratedTranslationProvider(
                client, "http://mock.invalid", "mock-orchestrator", ("mock-translator",),
            )
        self.addAsyncCleanup(provider.aclose)
        return provider, requests, pending

    def assert_natural_selector_payload(self, payload, identifiers):
        self.assertEqual(payload["model"], "mock-orchestrator")
        source = json.loads(payload["messages"][1]["content"])
        self.assertFalse(source["force_flush"])
        self.assertEqual([unit["unit_id"] for unit in source["units"]], identifiers)
        branches = payload["response_format"]["json_schema"]["schema"]["anyOf"]
        choices = {branch["properties"]["action"]["enum"][0]: branch for branch in branches}
        self.assertEqual(set(choices), {"wait", "commit"})
        self.assertEqual(choices["wait"]["required"], ["action"])
        self.assertEqual(choices["commit"]["properties"]["through_id"]["enum"], identifiers)

    async def test_install_is_idempotent_returns_fresh_metadata_and_preserves_base(self):
        coordinator = self.module.SemanticTranslationCoordinator
        router = self.module.SemanticTranslationRouter
        actual_global = routing.SemanticTranslationCoordinator
        metadata = install_quality_policy(self.module)
        self.assertEqual(metadata["semantic_quality_policy"], "complete-thought-bounded-wait-v2")
        self.assertEqual(metadata["semantic_max_hold_ms"], 12000)
        self.assertEqual(metadata["semantic_total_budget_ms"], 45000)
        self.assertEqual(metadata["semantic_request_timeout_ms"], 20000)
        metadata["semantic_max_hold_ms"] = -1
        self.assertEqual(install_quality_policy(self.module)["semantic_max_hold_ms"], 12000)
        self.assertIs(self.module.SemanticTranslationCoordinator, coordinator)
        self.assertIs(self.module.SemanticTranslationRouter, router)
        self.assertIs(coordinator._myvote_quality_original, self.soft_coordinator)
        self.assertIs(routing.SemanticTranslationCoordinator, actual_global)
        for name in ("_query", "_failure", "request_boundary"):
            self.assertIs(getattr(coordinator, name), getattr(self.soft_coordinator, name))

    async def test_fast_input_cadence_does_not_repeat_unchanged_wait_forever(self):
        provider = GeneratedProvider([{"action": "wait"}, {"action": "wait"},
            {"action": "commit", "through_id": "tail", "text": "완성 번역"}])
        coordinator = self.make(provider, scheduled=True, config=semantic.SemanticTranslationConfig(
            min_request_interval_s=.01, max_hold_s=.025,
            request_timeout_s=.1, max_total_age_s=.5))
        first, tail = self.unit("first", "We need to"), self.unit("tail", " finish the report.")
        await coordinator.append(first)
        await asyncio.sleep(.12)
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(coordinator.pending_units, (first,))
        self.assertEqual(self.failures, [])
        # New text wakes the scheduler even after its one idle recheck.
        await coordinator.append(tail)
        await asyncio.wait_for(self.published.wait(), .25)
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(self.commits[0].units, (first, tail))

    async def test_custom_pacing_is_applied_to_router_and_metadata(self):
        module = SimpleNamespace(SemanticTranslationCoordinator=semantic.SemanticTranslationCoordinator,
            SemanticTranslationRouter=original_router(), SemanticTranslationConfig=semantic.SemanticTranslationConfig)
        install_routing(module)
        metadata = install_quality_policy(module, min_request_interval_s=.25, max_hold_s=.75)
        self.assertEqual(metadata["semantic_min_request_interval_ms"], 250)
        self.assertEqual(metadata["semantic_max_hold_ms"], 750)
        router = module.SemanticTranslationRouter(GeneratedProvider([]), None, None)
        self.addAsyncCleanup(router.close, flush=False)
        self.assertEqual(router.config.min_request_interval_s, .25)
        self.assertEqual(router.config.max_hold_s, .75)

    async def test_router_changes_only_hold_settings_and_constructs_quality_child(self):
        config = semantic.SemanticTranslationConfig(
            min_request_interval_s=.2, max_hold_s=2, max_total_age_s=3,
            request_timeout_s=2.7, max_units=9, max_source_chars=777,
        )

        async def sink(result):
            pass

        router = self.module.SemanticTranslationRouter(
            GeneratedProvider([]), sink, sink, config=config,
            target_language="ja", max_lanes=3, max_total_units=9,
        )
        self.addAsyncCleanup(router.close, flush=False)
        self.assertEqual((router.config.min_request_interval_s, router.config.max_hold_s,
                          router.config.max_total_age_s), (1.5, 12, 45))
        self.assertEqual(router.config.request_timeout_s, 20)
        self.assertEqual(router.config.max_units, 9)
        self.assertEqual(router.config.max_source_chars, 777)
        self.assertEqual(config.max_hold_s, 2)
        self.assertEqual((router.max_lanes, router.max_total_units, router.target_language), (3, 9, "ja"))
        unit = self.unit("configured")
        # The original verified router creates children through its module
        # globals. Scope that patch to this assertion instead of all tests.
        with patch.object(routing, "SemanticTranslationCoordinator",
                          self.module.SemanticTranslationCoordinator):
            await router.append(unit)
        self.assertIsInstance(router._lanes[unit.scope], self.module.SemanticTranslationCoordinator)
        self.assertEqual(router._lanes[unit.scope].config, router.config)

    async def test_ordinary_wait_retains_original_source_without_translation(self):
        provider = GeneratedProvider([{"action": "wait"}])
        coordinator = self.make(provider)
        unit = self.unit("pending", "The trip took roughly")
        await coordinator.append(unit)
        await coordinator._query((unit,), False)
        self.assertEqual(coordinator.pending_units, (unit,))
        self.assertEqual(self.commits, [])
        self.assertEqual(self.failures, [])
        self.assertEqual(coordinator.counts["waits"], 1)
        self.assertFalse(provider.requests[0].force_flush)

    async def test_terminal_wait_retires_only_source_and_keeps_it_owned_until_callback(self):
        provider = GeneratedProvider([{"action": "wait"}])
        coordinator = self.make(provider)
        unit = self.unit("unfinished", "회의를 시작하려면")
        await coordinator.append(unit)
        coordinator.request_boundary()
        coordinator.request_flush()
        await coordinator._query(*coordinator._head())
        self.assertFalse(provider.requests[0].force_flush)
        self.assertEqual(self.commits, [])
        self.assertEqual(len(self.failures), 1)
        self.assertEqual(self.failures[0].reason, "incomplete_source")
        self.assertTrue(self.failures[0].force_flush)
        self.assertIs(self.failures[0].units[0], unit)
        self.assertEqual(self.active_at_failure, [(unit,)])
        self.assertEqual(coordinator.pending_units, ())
        self.assertEqual(coordinator.unsettled_units, ())
        self.assertEqual(coordinator._soft_boundaries, set())
        self.assertEqual(coordinator.counts["invalid"], 0)

    async def test_forced_partial_commit_then_wait_preserves_untranslated_tail(self):
        provider = GeneratedProvider([
            {"action": "commit", "through_id": "ready", "text": "회의가 끝났습니다."},
            {"action": "wait"},
        ])
        coordinator = self.make(provider)
        ready = self.unit("ready", "The meeting ended.")
        tail = self.unit("tail", " Tomorrow we will")
        await coordinator.append(ready)
        await coordinator.append(tail)
        coordinator.request_flush()
        await coordinator._query(*coordinator._head())
        self.assertEqual(self.commits[0].units, (ready,))
        self.assertEqual(coordinator.pending_units, (tail,))
        self.assertTrue(coordinator._head()[1])
        await coordinator._query(*coordinator._head())
        self.assertTrue(all(not request.force_flush for request in provider.requests))
        self.assertEqual([tuple(unit.unit_id for unit in request.units)
                          for request in provider.requests], [("ready", "tail"), ("tail",)])
        self.assertEqual(self.failures[0].units, (tail,))
        self.assertEqual(self.failures[0].reason, "incomplete_source")
        self.assertEqual(coordinator.pending_units, ())

    async def test_terminal_wait_does_not_consume_source_appended_after_request_snapshot(self):
        provider = GeneratedProvider([{"action": "wait"}], pause=True)
        coordinator = self.make(provider)
        old, later = self.unit("old"), self.unit("later")
        await coordinator.append(old)
        coordinator.request_flush()
        query = asyncio.create_task(coordinator._query((old,), True))
        await asyncio.wait_for(provider.entered.wait(), timeout=1)
        await coordinator.append(later)
        provider.release.set()
        await asyncio.wait_for(query, timeout=1)
        self.assertEqual(self.failures[0].units, (old,))
        self.assertEqual(coordinator.pending_units, (later,))
        self.assertIs(coordinator.pending_units[0].payload, later.payload)

    async def test_hold_recheck_wait_keeps_prefix_for_later_sentence_ending(self):
        provider = GeneratedProvider([
            {"action": "wait"},
            {"action": "commit", "through_id": "ending", "text": "완성된 문장입니다."},
        ])
        coordinator = self.make(provider)
        prefix = self.unit("prefix", "After reviewing the detailed proposal, we decided to")
        ending = self.unit("ending", " proceed with the revised plan.")
        await coordinator.append(prefix)
        coordinator.request_boundary()  # A VAD pause is not a hard stop.
        await coordinator._query((prefix,), True)  # max_hold elapsed
        self.assertEqual(coordinator.pending_units, (prefix,))
        self.assertEqual(self.failures, [])
        self.assertTrue(coordinator.resume_boundary(prefix.unit_id))
        await coordinator.append(ending)
        await coordinator._query((prefix, ending), False)
        self.assertEqual(self.commits[0].units, (prefix, ending))
        self.assertTrue(all(not request.force_flush for request in provider.requests))

    async def test_waits_are_still_bounded_by_absolute_source_lifetime(self):
        class WaitingProvider:
            async def semantic_stream(self, request):
                yield Chunk('{"action":"wait"}', completed=True, finish_reason="stop")

        coordinator = self.make(WaitingProvider(), scheduled=True,
            config=semantic.SemanticTranslationConfig(
                min_request_interval_s=.01, max_hold_s=.02,
                request_timeout_s=.05, max_total_age_s=.09))
        await coordinator.append(self.unit("never-complete"))
        await asyncio.wait_for(self.published.wait(), timeout=1)
        self.assertEqual(self.failures[0].reason, "source_deadline")
        self.assertEqual(coordinator.pending_units, ())
        self.assertGreater(coordinator.counts["waits"], 1)

    async def test_one_transient_timeout_retries_same_snapshot_inside_same_deadline(self):
        requests = []

        class TransientProvider:
            async def semantic_stream(self, request):
                requests.append(request)
                if len(requests) == 1:
                    raise ProviderError("timeout", "synthetic transient stage timeout")
                yield Chunk('{"action":"commit","through_id":"whole","text":"완성 번역"}',
                            completed=True, finish_reason="stop")

        coordinator = self.make(TransientProvider())
        unit = self.unit("whole", "This is a complete sentence.")
        await coordinator.append(unit)
        await coordinator._query((unit,), False)
        self.assertEqual(len(self.commits), 1)
        self.assertEqual(self.failures, [])
        self.assertEqual(coordinator.counts["quality_retries"], 1)
        self.assertEqual(requests[0].units, requests[1].units)
        self.assertEqual(requests[0].request_id, requests[1].request_id)
        self.assertEqual(requests[0].deadline_monotonic, requests[1].deadline_monotonic)
        self.assertLess(requests[1].budget_ms, requests[0].budget_ms)

    async def test_repeated_timeout_is_not_retried_forever(self):
        requests = []

        class UnavailableProvider:
            async def semantic_stream(self, request):
                requests.append(request)
                raise ProviderError("timeout", "synthetic timeout")
                yield

        coordinator = self.make(UnavailableProvider())
        unit = self.unit("whole")
        await coordinator.append(unit)
        await coordinator._query((unit,), False)
        self.assertEqual(len(requests), 2)
        self.assertEqual(self.failures[0].reason, "model_timeout")

    async def test_generation_change_prevents_stale_terminal_retirement(self):
        provider = GeneratedProvider([{"action": "wait"}], pause=True)
        coordinator = self.make(provider)
        unit = self.unit("stale")
        await coordinator.append(unit)
        query = asyncio.create_task(coordinator._query((unit,), True))
        await asyncio.wait_for(provider.entered.wait(), timeout=1)
        coordinator._generation += 1
        provider.release.set()
        await asyncio.wait_for(query, timeout=1)
        self.assertEqual(self.commits, [])
        self.assertEqual(self.failures, [])
        self.assertEqual(coordinator.pending_units, (unit,))
        self.assertEqual(coordinator.counts["stale"], 1)

    async def test_actual_worker_flush_completes_without_repeated_forced_waits(self):
        provider = GeneratedProvider([
            {"action": "commit", "through_id": "ready", "text": "generated complete sentence"},
            {"action": "wait"},
        ])
        coordinator = self.make(provider, scheduled=True)
        ready, tail = self.unit("ready"), self.unit("tail")
        await coordinator.append(ready)
        await coordinator.append(tail)
        await asyncio.wait_for(coordinator.flush(), timeout=1)
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(self.commits[0].units, (ready,))
        self.assertEqual(self.failures[0].units, (tail,))
        self.assertEqual(self.failures[0].reason, "incomplete_source")
        self.assertEqual(coordinator.unsettled_units, ())

    async def test_original_force_true_parser_still_rejects_wait_and_partial_commit(self):
        units = (SemanticUnit("u1", "Complete."), SemanticUnit("u2", " unfinished"))
        for packet in ({"action": "wait"},
                       {"action": "commit", "through_id": "u1", "text": "result"}):
            with self.subTest(packet=packet):
                with self.assertRaises(semantic.InvalidSemanticResponse):
                    semantic.parse_semantic_response(json.dumps(packet), units, force_flush=True)

    async def test_http_forced_drain_allows_wait_and_never_calls_translation_api(self):
        provider, requests, pending = self.http_provider([
            ("/v1/chat/completions", '{"action":"wait"}'),
        ])
        coordinator = self.make(provider, scheduled=True)
        unit = self.unit("unfinished", "The repair will take roughly")
        await coordinator.append(unit)
        await asyncio.wait_for(coordinator.flush(), timeout=1)
        self.assertEqual([path for path, _ in requests], ["/v1/chat/completions"])
        self.assert_natural_selector_payload(requests[0][1], ["unfinished"])
        self.assertEqual(self.commits, [])
        self.assertEqual(len(self.failures), 1)
        self.assertEqual(self.failures[0].reason, "incomplete_source")
        self.assertIs(self.failures[0].units[0], unit)
        self.assertIs(self.failures[0].units[0].payload, unit.payload)
        self.assertEqual(self.failures[0].units[0].text, "The repair will take roughly")
        self.assertEqual(pending, [])
        self.assertEqual(coordinator.unsettled_units, ())

    async def test_http_terminal_prefix_translation_excludes_unfinished_tail_and_context(self):
        provider, requests, pending = self.http_provider([
            ("/v1/chat/completions", '{"action":"commit","through_id":"ready"}'),
            ("/v1/completions", "회의가 끝났습니다."),
            ("/v1/chat/completions", '{"action":"wait"}'),
        ])
        coordinator = self.make(provider, scheduled=True, config=semantic.SemanticTranslationConfig(
            min_request_interval_s=.001, max_hold_s=1,
            request_timeout_s=3.5, max_total_age_s=5,
        ))
        ready = self.unit("ready", "The meeting ended.")
        tail = self.unit("tail", " Tomorrow we will")
        coordinator._context_scope = ready.scope
        coordinator._context = ["PRIVATE_CONTEXT"]
        await coordinator.append(ready)
        await coordinator.append(tail)
        await asyncio.wait_for(coordinator.flush(), timeout=1)
        self.assertEqual([path for path, _ in requests],
                         ["/v1/chat/completions", "/v1/completions", "/v1/chat/completions"])
        self.assert_natural_selector_payload(requests[0][1], ["ready", "tail"])
        self.assert_natural_selector_payload(requests[2][1], ["tail"])
        translator = requests[1][1]
        self.assertEqual(translator["model"], "mock-translator")
        self.assertTrue(translator["prompt"].endswith(
            "\n\n\nThe meeting ended.<end_of_turn>\n<start_of_turn>model\n"))
        self.assertNotIn("Tomorrow we will", translator["prompt"])
        self.assertNotIn("PRIVATE_CONTEXT", translator["prompt"])
        self.assertNotIn("response_format", translator)
        self.assertEqual(len(self.commits), 1)
        self.assertEqual(self.commits[0].text, "회의가 끝났습니다.")
        self.assertEqual(self.commits[0].units, (ready,))
        self.assertEqual(len(self.failures), 1)
        self.assertEqual(self.failures[0].reason, "incomplete_source")
        self.assertEqual(self.failures[0].units, (tail,))
        self.assertIs(self.failures[0].units[0].payload, tail.payload)
        self.assertEqual((coordinator.counts["committed_units"], coordinator.counts["retired_units"]), (1, 1))
        self.assertEqual(pending, [])
        self.assertEqual(coordinator.unsettled_units, ())


if __name__ == "__main__":
    unittest.main()
