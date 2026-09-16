"""Quality inference regression tests: verified parsers and mock HTTP only."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import runpy
import time
from types import SimpleNamespace
import unittest

try:
    import httpx
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Use MyVote-Mac-Demo/.venv/bin/python for provider tests") from exc

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "tests/verified_engine.py"))["load_verified_engine"]()
from myvote_engine import orchestrated_translation as engine
from myvote_engine.translation import SemanticTranslationRequest, SemanticUnit, ProviderError
from scripts.semantic_quality_provider import RESIDUAL_SOURCE_INSTRUCTION, install_quality_provider


def response(text, *, chat=True, finish="stop"):
    field = {"delta": {"content": text}} if chat else {"text": text}
    packets = [{"choices": [{"index": 0, **field, "finish_reason": None}]},
               {"choices": [{"index": 0, "finish_reason": finish}]}]
    body = "".join("data: " + json.dumps(packet) + "\n\n" for packet in packets)
    return httpx.Response(200, content=(body + "data: [DONE]\n\n").encode(),
                          headers={"content-type": "text/event-stream"})


class QualityProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = SimpleNamespace(**vars(engine))
        # Keep this test independent of other installer tests in discovery.
        while getattr(self.module.OrchestratedTranslationProvider,
                      "_myvote_quality_provider_v1", False):
            self.module.OrchestratedTranslationProvider = (
                self.module.OrchestratedTranslationProvider._myvote_quality_provider_original)
        self.requests = []

    def make(self, handler=None, **settings):
        install_quality_provider(self.module, **settings)

        async def tracked(request):
            payload = json.loads(request.content)
            self.requests.append((request.url.path, payload))
            if handler is not None:
                return await handler(request, payload)
            if "prompt" in payload:
                return response("Translated.", chat=False)
            data = json.loads(payload["messages"][1]["content"])
            if "units" in data:
                return response(json.dumps({"action": "commit", "through_id": data["units"][-1]["unit_id"]}))
            return response("Whole completed translation.")

        client = httpx.AsyncClient(transport=httpx.MockTransport(tracked), trust_env=False)
        self.addAsyncCleanup(client.aclose)
        provider = self.module.OrchestratedTranslationProvider(
            client, "http://mock.invalid", "fake-gemma", ("fake-translategemma",))
        self.addAsyncCleanup(provider.aclose)
        return provider

    def request(self, source="The meeting ended.", language="en", **kwargs):
        return SemanticTranslationRequest("synthetic", (SemanticUnit("u1", source),), language,
                                          "en" if language == "ko" else "ko",
                                          budget_ms=20000, **kwargs)

    async def collect(self, provider, request=None):
        chunks = [chunk async for chunk in provider.semantic_stream(request or self.request())]
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].completed)
        return json.loads(chunks[0].text), chunks[0]

    async def test_install_validates_and_is_idempotent(self):
        metadata = install_quality_provider(self.module)
        installed = self.module.OrchestratedTranslationProvider
        self.assertEqual(metadata["semantic_request_timeout_ms"], 20000)
        self.assertEqual(metadata["semantic_translation_budget_ms"], 15000)
        self.assertEqual(metadata["semantic_orchestrator_budget_ms"], 5000)
        self.assertEqual(install_quality_provider(self.module), metadata)
        self.assertIs(self.module.OrchestratedTranslationProvider, installed)
        with self.assertRaises(ValueError):
            install_quality_provider(self.module, selection_timeout_s=4)
        for name in ("selection_timeout_s", "translation_timeout_s", "request_timeout_s"):
            for value in (0, -1, True, "5", float("nan"), float("inf"), 121):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    install_quality_provider(self.module, **{name: value})
        for value in (True, 511, 4097, 2048.0):
            with self.assertRaises(ValueError):
                install_quality_provider(self.module, fallback_max_tokens=value)
        for value in (True, 1, 65, 8.0, "8"):
            with self.assertRaises(ValueError):
                install_quality_provider(self.module, whole_source_recheck_min_units=value)
        for value in (0, 1, "true", None):
            with self.assertRaises(ValueError):
                install_quality_provider(self.module, residual_flush=value)
            with self.assertRaises(ValueError):
                install_quality_provider(self.module, early_prefix_recheck=value)
        self.assertFalse(metadata["semantic_residual_flush"])

    def fragmented_request(self, *, unfinished=False):
        pieces = (("We", " will", " deploy", " only", " if", " all", " the", " tests")
                  if unfinished else
                  ("The", " meeting", " ended.", " We", " all", " went", " back", " home."))
        return SemanticTranslationRequest(
            "fragmented", tuple(SemanticUnit(f"u{index}", text)
                                for index, text in enumerate(pieces, 1)),
            "en", "ko", context=("Earlier source.",), budget_ms=20000)

    async def test_first_wait_whole_source_commit_preserves_exact_source_and_last_id(self):
        async def handler(request, payload):
            if "prompt" in payload:
                return response("완성된 번역.", chat=False)
            data = json.loads(payload["messages"][1]["content"])
            if len(data["units"]) > 1:
                return response('{"action":"wait"}')
            return response('{"action":"commit","through_id":"u8"}')
        provider = self.make(handler)
        request = self.fragmented_request()
        result, chunk = await self.collect(provider, request)
        self.assertEqual(result, {"action": "commit", "through_id": "u8", "text": "완성된 번역."})
        first = json.loads(self.requests[0][1]["messages"][1]["content"])
        second = json.loads(self.requests[1][1]["messages"][1]["content"])
        self.assertEqual({key: value for key, value in first.items() if key != "units"},
                         {key: value for key, value in second.items() if key != "units"})
        self.assertEqual(second["units"], [{"unit_id": "u8", "text": "".join(u.text for u in request.units)}])
        self.assertFalse(second["force_flush"])
        self.assertEqual(self.requests[0][1]["messages"][0], self.requests[1][1]["messages"][0])
        self.assertIn(second["units"][0]["text"], self.requests[2][1]["prompt"])
        self.assertEqual(chunk.usage["quality_selection_attempts"], 2)

    async def test_incomplete_whole_source_wait_never_forces_or_translates(self):
        async def handler(request, payload):
            self.assertNotIn("prompt", payload)
            data = json.loads(payload["messages"][1]["content"])
            self.assertFalse(data["force_flush"])
            return response('{"action":"wait"}')
        provider = self.make(handler)
        result, chunk = await self.collect(provider, self.fragmented_request(unfinished=True))
        self.assertEqual(result, {"action": "wait"})
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(chunk.usage["quality_selection_attempts"], 2)

    async def test_collapsed_recheck_rejects_id_valid_only_in_original_units(self):
        async def handler(request, payload):
            data = json.loads(payload["messages"][1]["content"])
            return response('{"action":"wait"}' if len(data["units"]) > 1
                            else '{"action":"commit","through_id":"u4"}')
        provider = self.make(handler)
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider, self.fragmented_request())
        self.assertEqual(caught.exception.category, "invalid_output")
        self.assertEqual(len(self.requests), 2)

    async def test_first_pass_complete_prefix_commit_never_rechecks_or_extends(self):
        async def handler(request, payload):
            if "prompt" in payload:
                return response("회의가 끝났어요.", chat=False)
            return response('{"action":"commit","through_id":"u3"}')
        provider = self.make(handler)
        result, chunk = await self.collect(provider, self.fragmented_request())
        self.assertEqual(result["through_id"], "u3")
        self.assertEqual(len(self.requests), 2)
        self.assertIn("The meeting ended.<end_of_turn>", self.requests[1][1]["prompt"])
        self.assertEqual(chunk.usage["quality_selection_attempts"], 1)

    async def test_recheck_shares_original_selection_deadline(self):
        async def handler(request, payload):
            await asyncio.sleep(.03)
            data = json.loads(payload["messages"][1]["content"])
            return response('{"action":"wait"}' if len(data["units"]) > 1
                            else '{"action":"commit","through_id":"u8"}')
        provider = self.make(handler, selection_timeout_s=.05,
                             translation_timeout_s=.1, request_timeout_s=.2)
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider, self.fragmented_request())
        self.assertEqual(caught.exception.category, "timeout")
        self.assertEqual(len(self.requests), 2)
        self.assertFalse(provider._active_tasks)

    async def test_clause_projection_preserves_exact_concat_and_entire_tail(self):
        async def handler(request, payload):
            if "prompt" in payload:
                return response("첫 문장.", chat=False)
            data = json.loads(payload["messages"][1]["content"])
            return response('{"action":"wait"}' if len(data["units"]) == 8
                            else '{"action":"commit","through_id":"u3"}')
        provider = self.make(handler, early_prefix_recheck=True)
        request = self.fragmented_request()
        packet, chunk = await self.collect(provider, request)
        self.assertEqual(packet["through_id"], "u3")
        first = json.loads(self.requests[0][1]["messages"][1]["content"])
        projected = json.loads(self.requests[1][1]["messages"][1]["content"])
        self.assertEqual(projected["units"], [
            {"unit_id": "u3", "text": "The meeting ended."},
            {"unit_id": "u8", "text": " We all went back home."}])
        self.assertEqual("".join(item["text"] for item in projected["units"]),
                         "".join(item["text"] for item in first["units"]))
        self.assertEqual({key: value for key, value in projected.items() if key != "units"},
                         {key: value for key, value in first.items() if key != "units"})
        self.assertEqual(chunk.usage["quality_selection_attempts"], 2)
        self.assertEqual(len(self.requests), 3)

    async def test_projected_commit_cannot_select_an_interior_original_id(self):
        async def handler(request, payload):
            data = json.loads(payload["messages"][1]["content"])
            return response('{"action":"wait"}' if len(data["units"]) == 8
                            else '{"action":"commit","through_id":"u2"}')
        provider = self.make(handler, early_prefix_recheck=True)
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider, self.fragmented_request())
        self.assertEqual(caught.exception.category, "invalid_output")
        self.assertEqual(len(self.requests), 2)

    async def test_projected_attached_but_wait_never_becomes_translation_permission(self):
        async def handler(request, payload):
            self.assertNotIn("prompt", payload)
            data = json.loads(payload["messages"][1]["content"])
            self.assertFalse(data["force_flush"])
            return response('{"action":"wait"}')
        provider = self.make(handler, early_prefix_recheck=True)
        request = SemanticTranslationRequest("contrast", tuple(
            SemanticUnit(f"u{i}", piece) for i, piece in enumerate(
                ("The", " report", " is", " ready,", " but"), 1)), "en", "ko", budget_ms=20000)
        packet, chunk = await self.collect(provider, request)
        self.assertEqual(packet, {"action": "wait"})
        self.assertEqual(chunk.usage["quality_selection_attempts"], 2)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(json.loads(self.requests[1][1]["messages"][1]["content"])["units"], [
            {"unit_id": "u4", "text": "The report is ready,"}, {"unit_id": "u5", "text": " but"}])

    async def test_projection_never_changes_first_nonwait_result(self):
        provider = self.make(early_prefix_recheck=True)
        packet, chunk = await self.collect(provider, self.fragmented_request())
        self.assertEqual(packet["through_id"], "u8")
        self.assertEqual(chunk.usage["quality_selection_attempts"], 1)
        self.assertEqual(len(self.requests), 2)

    async def test_projection_wait_can_still_use_original_whole_source_fallback(self):
        async def handler(request, payload):
            if "prompt" in payload:
                return response("전체 번역.", chat=False)
            data = json.loads(payload["messages"][1]["content"])
            return response('{"action":"commit","through_id":"u8"}' if len(data["units"]) == 1
                            else '{"action":"wait"}')
        provider = self.make(handler, early_prefix_recheck=True)
        packet, chunk = await self.collect(provider, self.fragmented_request())
        self.assertEqual(packet["through_id"], "u8")
        self.assertEqual(chunk.usage["quality_selection_attempts"], 3)
        self.assertEqual([len(json.loads(payload["messages"][1]["content"])["units"])
                          for _, payload in self.requests[:3]], [8, 2, 1])
        self.assertEqual(len(self.requests), 4)

    async def test_single_projected_block_does_not_repeat_whole_source_query(self):
        async def handler(request, payload):
            return response('{"action":"wait"}')
        provider = self.make(handler, early_prefix_recheck=True)
        request = replace(self.fragmented_request(), units=tuple(
            SemanticUnit(f"u{i}", text) for i, text in enumerate(
                ("The", " team", " will", " finish", " all", " these", " checks", " tomorrow."), 1)))
        packet, chunk = await self.collect(provider, request)
        self.assertEqual(packet, {"action": "wait"})
        self.assertEqual(chunk.usage["quality_selection_attempts"], 2)
        self.assertEqual(len(self.requests), 2)

    async def test_projection_noop_for_already_grouped_units_or_no_punctuation(self):
        async def handler(request, payload):
            return response('{"action":"wait"}')
        provider = self.make(handler, early_prefix_recheck=True)
        for pieces in (("The meeting ended,", " and tomorrow we will"),
                       ("The", " meeting", " ended", " and", " tomorrow", " we", " will")):
            self.requests.clear()
            request = SemanticTranslationRequest("unchanged", tuple(
                SemanticUnit(f"u{i}", piece) for i, piece in enumerate(pieces, 1)),
                "en", "ko", budget_ms=20000)
            _, chunk = await self.collect(provider, request)
            self.assertEqual(chunk.usage["quality_selection_attempts"], 1)
            self.assertEqual(len(self.requests), 1)

    async def test_projection_never_splits_inside_original_unit_and_preserves_cjk_spacing(self):
        async def handler(request, payload):
            return response('{"action":"wait"}')
        provider = self.make(handler, early_prefix_recheck=True)
        pieces = ("  회의가", " 끝났어요。 다음", " 이야기는", " 계속됩니다！  ", " 아직")
        request = SemanticTranslationRequest("cjk", tuple(
            SemanticUnit(f"u{i}", piece) for i, piece in enumerate(pieces, 1)), "ko", "en", budget_ms=20000)
        await self.collect(provider, request)
        projected = json.loads(self.requests[1][1]["messages"][1]["content"])["units"]
        self.assertEqual(projected, [{"unit_id": "u4", "text": "".join(pieces[:4])},
                                     {"unit_id": "u5", "text": pieces[4]}])
        self.assertEqual("".join(unit["text"] for unit in projected), "".join(pieces))

    async def test_three_selection_attempts_share_one_original_stage_deadline(self):
        async def handler(request, payload):
            await asyncio.sleep(.025)
            data = json.loads(payload["messages"][1]["content"])
            return response('{"action":"commit","through_id":"u8"}' if len(data["units"]) == 1
                            else '{"action":"wait"}')
        provider = self.make(handler, early_prefix_recheck=True, selection_timeout_s=.06,
                             translation_timeout_s=.1, request_timeout_s=.2)
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider, self.fragmented_request())
        self.assertEqual(caught.exception.category, "timeout")
        self.assertEqual(len(self.requests), 3)
        self.assertFalse(provider._active_tasks)

    async def test_selection_over_one_second_succeeds_and_short_source_stays_tg(self):
        async def handler(request, payload):
            if "prompt" in payload:
                return response("회의가 끝났어요.", chat=False)
            await asyncio.sleep(1.05)
            return response('{"action":"commit","through_id":"u1"}')
        provider = self.make(handler)
        result, chunk = await self.collect(provider)
        self.assertEqual(result["text"], "회의가 끝났어요.")
        self.assertGreater(chunk.usage["stage_ms"]["selection"], 1000)
        self.assertEqual([path for path, _ in self.requests],
                         ["/v1/chat/completions", "/v1/completions"])
        self.assertFalse(provider._active_tasks)
        self.assertEqual(provider._slots.qsize(), 4)

    async def test_long_korean_falls_back_with_exact_complete_source_and_bounded_output(self):
        source = "회의에서 모든 항목을 검토한 다음 최종 결정을 내렸습니다. " * 18
        provider = self.make()
        result, _ = await self.collect(provider, self.request(source, "ko"))
        self.assertEqual(result["through_id"], "u1")
        self.assertEqual([path for path, _ in self.requests], ["/v1/chat/completions"] * 2)
        payload = self.requests[1][1]
        self.assertEqual(payload["model"], "fake-gemma")
        self.assertEqual(payload["max_tokens"], 2048)
        data = json.loads(payload["messages"][1]["content"])
        self.assertEqual(data["source_text"], source)
        self.assertNotIn("units", data)
        self.assertEqual(provider._orchestrator.max_tokens, 256)
        self.assertEqual(provider._slots.qsize(), 4)

    async def test_invalid_source_or_language_never_falls_back_even_above_character_cap(self):
        for source, language, expected in (
                ("<start_of_turn>bad", "en", "invalid_source"),
                ("a" * 4100 + "<start_of_turn>", "en", "invalid_source"),
                ("Hello.", "xx", "unsupported_language"),
                ("a" * 4100, "xx", "unsupported_language")):
            with self.subTest(expected=expected, length=len(source)):
                self.requests.clear()
                provider = self.make()
                with self.assertRaises(ProviderError) as caught:
                    await self.collect(provider, self.request(source, language))
                self.assertEqual(caught.exception.category, expected)
                self.assertEqual(len(self.requests), 1)

    async def test_wait_never_starts_translation_and_forced_wait_stays_invalid(self):
        async def handler(request, payload):
            return response('{"action":"wait"}')
        provider = self.make(handler)
        result, _ = await self.collect(provider)
        self.assertEqual(result, {"action": "wait"})
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider, self.request(force_flush=True))
        self.assertEqual(caught.exception.category, "invalid_output")
        self.assertEqual(len(self.requests), 2)

    async def test_shorter_caller_absolute_deadline_and_budget_are_not_extended(self):
        async def handler(request, payload):
            await asyncio.sleep(.1)
            return response('{"action":"commit","through_id":"u1"}')
        provider = self.make(handler)
        for request in (self.request(deadline_monotonic=time.monotonic() + .02),
                        replace(self.request(), budget_ms=20)):
            with self.assertRaises(ProviderError) as caught:
                await self.collect(provider, request)
            self.assertEqual(caught.exception.category, "timeout")
            self.assertFalse(provider._active_tasks)

    async def test_expired_deadline_performs_no_http(self):
        provider = self.make()
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider, self.request(deadline_monotonic=time.monotonic() - 1))
        self.assertEqual(caught.exception.category, "timeout")
        self.assertEqual(self.requests, [])

    async def test_translation_stage_timeout_releases_slot(self):
        async def handler(request, payload):
            if "prompt" not in payload:
                return response('{"action":"commit","through_id":"u1"}')
            await asyncio.sleep(.1)
            return response("Late.", chat=False)
        provider = self.make(handler, selection_timeout_s=.05,
                             translation_timeout_s=.02, request_timeout_s=.1)
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider)
        self.assertEqual(caught.exception.category, "timeout")
        self.assertEqual(provider._slots.qsize(), 4)
        self.assertFalse(provider._active_tasks)

    async def test_cancellation_resistant_transport_cannot_publish_expired_selection(self):
        async def handler(request, payload):
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            return response('{"action":"commit","through_id":"u1"}')
        provider = self.make(handler, selection_timeout_s=.02,
                             translation_timeout_s=.05, request_timeout_s=.1)
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider)
        self.assertEqual(caught.exception.category, "timeout")
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(provider._active_tasks)

    async def test_close_cancels_resistant_transport_and_prevents_new_admission(self):
        entered = asyncio.Event()
        async def handler(request, payload):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            return response('{"action":"commit","through_id":"u1"}')
        provider = self.make(handler)
        task = asyncio.create_task(self.collect(provider))
        await asyncio.wait_for(entered.wait(), 1)
        await provider.aclose()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(provider._active_tasks)
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider)
        self.assertEqual(caught.exception.category, "closed")

    async def test_fallback_truncation_is_not_published_as_success(self):
        async def handler(request, payload):
            data = json.loads(payload["messages"][1]["content"])
            if "units" in data:
                return response('{"action":"commit","through_id":"u1"}')
            return response("Unfinished translation", finish="length")
        provider = self.make(handler)
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider, self.request("가" * 600, "ko"))
        self.assertEqual(caught.exception.category, "incomplete")

    async def test_fallback_shares_selector_gate(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def handler(request, payload):
            data = json.loads(payload["messages"][1]["content"])
            if "units" in data:
                return response('{"action":"commit","through_id":"u1"}')
            entered.set()
            await release.wait()
            return response("Complete.")
        provider = self.make(handler)
        first = asyncio.create_task(self.collect(provider, self.request("가" * 600, "ko")))
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(self.collect(provider, self.request("가" * 600, "ko")))
        await asyncio.sleep(.01)
        self.assertEqual(len(self.requests), 2)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 1)
        self.assertEqual(len(self.requests), 4)

    async def test_explicit_residual_flush_bypasses_selection_and_commits_exact_whole_source(self):
        from myvote_engine.semantic_translation import parse_semantic_response
        provider = self.make(residual_flush=True)
        request = replace(self.fragmented_request(unfinished=True), force_flush=True)
        packet, chunk = await self.collect(provider, request)
        parsed = parse_semantic_response(chunk.text, request.units, force_flush=True)
        self.assertEqual(parsed.through_id, "u8")
        self.assertEqual(packet["action"], "commit")
        self.assertEqual(len(self.requests), 1)
        path, payload = self.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(payload["model"], "fake-gemma")
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertNotIn("response_format", payload)
        data = json.loads(payload["messages"][1]["content"])
        self.assertEqual(data["source_text"], "".join(unit.text for unit in request.units))
        self.assertNotIn("units", data)
        self.assertNotIn(RESIDUAL_SOURCE_INSTRUCTION, data["source_text"])
        self.assertIn(RESIDUAL_SOURCE_INSTRUCTION, payload["messages"][0]["content"])
        self.assertTrue(chunk.usage["quality_residual_flush"])
        self.assertEqual(chunk.usage["quality_selection_attempts"], 0)
        self.assertEqual(chunk.usage["stage_ms"]["selection"], 0)
        self.assertFalse(provider._active_tasks)
        self.assertEqual(provider._slots.qsize(), 4)

    async def test_residual_enabled_never_changes_unforced_wait_policy(self):
        async def handler(request, payload):
            self.assertIn("units", json.loads(payload["messages"][1]["content"]))
            self.assertNotIn(RESIDUAL_SOURCE_INSTRUCTION, payload["messages"][0]["content"])
            return response('{"action":"wait"}')
        provider = self.make(handler, residual_flush=True)
        packet, chunk = await self.collect(provider, self.fragmented_request(unfinished=True))
        self.assertEqual(packet, {"action": "wait"})
        self.assertEqual(len(self.requests), 2)
        self.assertNotIn("quality_residual_flush", chunk.usage)

    async def test_residual_disabled_keeps_legacy_forced_selector_contract(self):
        provider = self.make(residual_flush=False)
        packet, chunk = await self.collect(provider, self.request(force_flush=True))
        self.assertEqual(packet["through_id"], "u1")
        self.assertEqual(len(self.requests), 2)
        self.assertIn("units", json.loads(self.requests[0][1]["messages"][1]["content"]))
        self.assertIn("prompt", self.requests[1][1])
        self.assertNotIn("quality_residual_flush", chunk.usage)

    async def test_residual_source_and_language_safety_run_before_any_http(self):
        provider = self.make(residual_flush=True)
        for source, language, expected in (
                ("<start_of_turn>bad", "en", "invalid_source"),
                ("a" * 4100 + "<start_of_turn>", "en", "invalid_source"),
                ("a" * 4100, "xx", "unsupported_language")):
            with self.subTest(expected=expected), self.assertRaises(ProviderError) as caught:
                await self.collect(provider, self.request(source, language, force_flush=True))
            self.assertEqual(caught.exception.category, expected)
        self.assertEqual(self.requests, [])

    async def test_residual_same_language_keeps_correction_and_context_contract(self):
        provider = self.make(residual_flush=True)
        request = SemanticTranslationRequest(
            "correction", (SemanticUnit("final-id", "제가 원한 것은 돈이 아니라"),),
            "ko", "ko", context=("앞선 발화입니다.",), force_flush=True, budget_ms=20000)
        packet, _ = await self.collect(provider, request)
        self.assertEqual(packet["through_id"], "final-id")
        payload = self.requests[0][1]
        self.assertIn("same-language ASR correction", payload["messages"][0]["content"])
        self.assertEqual(json.loads(payload["messages"][1]["content"]), {
            "context": ["앞선 발화입니다."], "source_text": "제가 원한 것은 돈이 아니라"})

    async def test_residual_still_uses_outer_primary_admission(self):
        from myvote_engine.priority_translation import AdmittedTranslationProvider, TranslationAdmission
        provider = self.make(residual_flush=True)
        admission = TranslationAdmission()
        admitted = AdmittedTranslationProvider(provider, admission, secondary=True)
        async with admission.acquire():
            task = asyncio.create_task(self.collect(admitted, self.request(force_flush=True)))
            await asyncio.sleep(.01)
            self.assertEqual(self.requests, [])
        await asyncio.wait_for(task, 1)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(admission.primary_calls, 2)
        self.assertEqual(admission.secondary_calls, 0)

    async def test_residual_gate_queue_counts_against_caller_deadline(self):
        provider = self.make(residual_flush=True)
        async with provider._orchestrator_gate:
            with self.assertRaises(ProviderError) as caught:
                await self.collect(provider, replace(self.request(force_flush=True), budget_ms=20))
        self.assertEqual(caught.exception.category, "timeout")
        self.assertEqual(self.requests, [])
        self.assertFalse(provider._active_tasks)

    async def test_residual_close_rejects_cancellation_resistant_late_result(self):
        entered = asyncio.Event()
        async def handler(request, payload):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            return response("Late fragment")
        provider = self.make(handler, residual_flush=True)
        task = asyncio.create_task(self.collect(provider, self.request(force_flush=True)))
        await asyncio.wait_for(entered.wait(), 1)
        await provider.aclose()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(provider._active_tasks)

    async def test_residual_truncated_output_is_never_published(self):
        async def handler(request, payload):
            return response("Unfinished", finish="length")
        provider = self.make(handler, residual_flush=True)
        with self.assertRaises(ProviderError) as caught:
            await self.collect(provider, self.request(force_flush=True))
        self.assertEqual(caught.exception.category, "incomplete")


if __name__ == "__main__":
    unittest.main()
