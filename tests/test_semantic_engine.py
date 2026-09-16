"""Verified semantic engine contracts using mocked HTTP and generated text only.

Run with MyVote-Mac-Demo/.venv/bin/python. No LM Studio connection, audio,
model loading, or server process is used by this module.
"""

import asyncio
import json
from pathlib import Path
import runpy
import time
import unittest

try:
    import httpx
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        "Semantic engine tests require httpx; use MyVote-Mac-Demo/.venv/bin/python"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
UPDATE = ROOT / "MyVote-Speaker-Update"
support = runpy.run_path(str(ROOT / "tests/verified_engine.py"))
manifest = support["load_verified_engine"]()

from myvote_engine import translation

compat = runpy.run_path(str(ROOT / "scripts/gemma_json_compat.py"))
compat["install_context_review_schema"](translation)

from myvote_engine import orchestrated_translation as orchestration

compat["install_selection_schema"](orchestration)

from myvote_engine.context_refinement import parse_context_corrections
from myvote_engine import semantic_translation as semantic


def _response(text, *, chat):
    field = {"delta": {"content": text}} if chat else {"text": text}
    packets = [
        {"choices": [{"index": 0, **field, "finish_reason": None}]},
        {"choices": [{"index": 0, "finish_reason": "stop"}]},
    ]
    body = "".join("data: " + json.dumps(packet, ensure_ascii=False) + "\n\n"
                   for packet in packets) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode("utf-8"),
                          headers={"content-type": "text/event-stream"})


async def _collect(stream):
    text = ""
    terminal = None
    async for chunk in stream:
        text += chunk.text
        if chunk.completed:
            terminal = chunk
    if terminal is None or terminal.finish_reason != "stop":
        raise AssertionError("Expected a completed stop result")
    return text, terminal


class SemanticEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_latest_boundary_prompt_changes_only_real_selection_system_message(self):
        prompt_support = runpy.run_path(str(ROOT / "scripts/gemma_boundary_prompt.py"))
        document = ROOT / "gemma4-semantic-boundary-prompt_latest.md"
        original_builder = orchestration.selection_messages
        provider, _, _ = self.provider([])
        request = self.request(force=True)
        baseline = provider._orchestrator._payload(request)
        correction = translation.TranslationRequest("same", 1, "안녕 하세요.", "ko", "ko")
        correction_payload = provider._orchestrator._payload(correction)
        source = translation.TranslationRequest("translate", 1, "Hello.", "en", "ko")
        translation_prompt = orchestration.translategemma_prompt(source)
        try:
            metadata = prompt_support["install_boundary_prompt"](orchestration, document)
            actual = provider._orchestrator._payload(request)
            self.assertEqual(actual["messages"][0]["content"], prompt_support["load_boundary_prompt"](document))
            self.assertEqual(metadata["semantic_boundary_prompt_sha256"],
                             "2e04a944d0de4b642acc5b9f5ce15c0e7b35b0bcffe22133c25822881b925fa8")
            self.assertEqual(actual["messages"][1], baseline["messages"][1])
            self.assertEqual({key: value for key, value in actual.items() if key != "messages"},
                             {key: value for key, value in baseline.items() if key != "messages"})
            self.assertEqual(actual["temperature"], 0)
            self.assertEqual(actual["max_tokens"], 128)
            self.assertEqual(provider._orchestrator._payload(correction), correction_payload)
            self.assertEqual(orchestration.translategemma_prompt(source), translation_prompt)
        finally:
            orchestration.selection_messages = original_builder

    def provider(self, replies):
        pending = list(replies)
        requests = []

        def handler(request):
            self.assertTrue(pending, "Unexpected inference request")
            expected_path, result = pending.pop(0)
            self.assertEqual(request.url.path, expected_path)
            self.assertEqual(request.method, "POST")
            requests.append((request.url.path, json.loads(request.content)))
            return _response(result, chat=expected_path == "/v1/chat/completions")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
        self.addAsyncCleanup(client.aclose)
        provider = orchestration.OrchestratedTranslationProvider(
            client, "http://mock.invalid", "configured-orchestrator",
            ("configured-translator",), translation_concurrency=4,
        )
        self.addAsyncCleanup(provider.aclose)
        return provider, requests, pending

    def request(self, *, language="en", force=False):
        return translation.SemanticTranslationRequest(
            "request-1",
            (translation.SemanticUnit("u1", "Hello."),
             translation.SemanticUnit("u2", " PRIVATE_SUFFIX")),
            language, "ko", context=("PRIVATE_CONTEXT",), force_flush=force,
            budget_ms=3500,
        )

    async def test_verified_imports_and_both_compatibility_layers_reach_real_selector(self):
        self.assertEqual(manifest["release"], "semantic-translation-2026-09-16")
        self.assertEqual(Path(translation.__file__).resolve(),
                         UPDATE / "src/myvote_engine/translation.py")
        self.assertTrue(issubclass(orchestration._SelectionProvider, translation.LMStudioProvider))
        provider, requests, _ = self.provider([])
        target = translation.ContextReviewTarget("s1", 1, 1, "원문", "결과", "ko", "ko")
        review = translation.ContextReviewRequest("review-1", (target,))
        self.assertEqual(
            provider._orchestrator._payload(review)["response_format"]["json_schema"]["name"],
            "myvote_context_review",
        )
        self.assertEqual(
            provider._orchestrator._payload(self.request())["response_format"]["json_schema"]["name"],
            "myvote_semantic_selection",
        )
        generic = translation.LMStudioProvider(provider._orchestrator.client,
                                               "http://mock.invalid", "generic-model")
        self.assertNotIn("response_format", generic._payload(self.request()))
        self.assertEqual(requests, [])

    async def test_selected_prefix_only_reaches_translation_completion_api(self):
        provider, requests, pending = self.provider([
            ("/v1/chat/completions", '{"action":"commit","through_id":"u1"}'),
            ("/v1/completions", "안녕하세요."),
        ])
        request = self.request()
        text, terminal = await _collect(provider.semantic_stream(request))
        result = semantic.parse_semantic_response(text, request.units)
        self.assertEqual((result.action, result.through_id, result.text),
                         ("commit", "u1", "안녕하세요."))
        selector, translator = (row[1] for row in requests)
        self.assertEqual(selector["model"], "configured-orchestrator")
        self.assertEqual(translator["model"], "configured-translator")
        self.assertIn("Hello.", translator["prompt"])
        self.assertNotIn("PRIVATE_SUFFIX", translator["prompt"])
        self.assertNotIn("PRIVATE_CONTEXT", translator["prompt"])
        self.assertTrue(translator["prompt"].startswith("<bos><start_of_turn>user\n"))
        self.assertTrue(translator["prompt"].endswith("<end_of_turn>\n<start_of_turn>model\n"))
        self.assertNotIn("messages", translator)
        self.assertNotIn("response_format", translator)
        self.assertEqual(translator["max_tokens"], 512)
        self.assertEqual(set(terminal.usage["stage_ms"]), {"selection", "translation"})
        self.assertEqual(pending, [])

    async def test_same_language_routes_selection_and_correction_to_orchestrator(self):
        provider, requests, pending = self.provider([
            ("/v1/chat/completions", '{"action":"commit","through_id":"u2"}'),
            ("/v1/chat/completions", "안녕하세요."),
        ])
        request = translation.SemanticTranslationRequest(
            "same-language", (translation.SemanticUnit("u1", "안녕"),
                              translation.SemanticUnit("u2", " 하세요.")),
            "ko", "ko", context=("이전 원문",), force_flush=True, budget_ms=3500,
        )
        text, _ = await _collect(provider.semantic_stream(request))
        result = semantic.parse_semantic_response(text, request.units, force_flush=True)
        self.assertEqual(result.text, "안녕하세요.")
        self.assertEqual([body["model"] for _, body in requests],
                         ["configured-orchestrator", "configured-orchestrator"])
        correction = requests[1][1]
        self.assertNotIn("response_format", correction)
        self.assertEqual(json.loads(correction["messages"][1]["content"]),
                         {"context": ["이전 원문"], "source_text": "안녕 하세요."})
        self.assertEqual(pending, [])

    async def test_context_review_keeps_schema_and_strict_revision_parser(self):
        packet = {"corrections": [{"segment_id": "s1", "source_revision": 1,
                                    "result_revision": 2, "text": "회의는 내일입니다."}]}
        provider, requests, pending = self.provider([
            ("/v1/chat/completions", json.dumps(packet, ensure_ascii=False)),
        ])
        target = translation.ContextReviewTarget(
            "s1", 1, 2, "회의는 내일 입니다.", "회의는 내일 입니다.", "ko", "ko",
        )
        review = translation.ContextReviewRequest("review-2", (target,))
        text, _ = await _collect(provider.stream(review))
        corrections = parse_context_corrections(text, review.targets)
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0].result_revision, 2)
        self.assertEqual(requests[0][1]["response_format"]["json_schema"]["name"],
                         "myvote_context_review")
        self.assertEqual(requests[0][1]["model"], "configured-orchestrator")
        self.assertEqual(pending, [])

    async def test_wait_performs_no_translation_request(self):
        provider, requests, pending = self.provider([
            ("/v1/chat/completions", '{"action":"wait"}'),
        ])
        request = self.request()
        text, _ = await _collect(provider.semantic_stream(request))
        self.assertEqual(semantic.parse_semantic_response(text, request.units).action, "wait")
        self.assertEqual(len(requests), 1)
        self.assertEqual(pending, [])

    async def test_invalid_selected_id_never_reaches_translator(self):
        provider, requests, pending = self.provider([
            ("/v1/chat/completions", '{"action":"commit","through_id":"invented-id"}'),
        ])
        with self.assertRaises(translation.ProviderError) as raised:
            await _collect(provider.semantic_stream(self.request()))
        self.assertEqual(raised.exception.category, "invalid_output")
        self.assertEqual(len(requests), 1)
        self.assertEqual(pending, [])

    async def test_final_parser_keeps_forced_prefix_and_plain_json_requirements(self):
        units = self.request().units
        cases = [
            ('```json\n{"action":"wait"}\n```', False),
            ('{"action":"wait"}', True),
            ('{"action":"commit","through_id":"u1","text":"결과"}', True),
            ('{"action":"commit","through_id":"invented-id","text":"결과"}', False),
            ('{"action":"commit","through_id":"u2"}', False),
            ('{"action":"wait","action":"wait"}', False),
        ]
        for packet, forced in cases:
            with self.subTest(packet=packet, forced=forced):
                with self.assertRaises(semantic.InvalidSemanticResponse):
                    semantic.parse_semantic_response(packet, units, force_flush=forced)

    async def test_timeout_returns_original_source_to_failure_handler(self):
        cancelled = asyncio.Event()

        class SlowProvider:
            async def semantic_stream(self, request):
                try:
                    await asyncio.Future()
                    yield translation.Chunk(text="unreachable")
                finally:
                    cancelled.set()

        commits, failures = [], []

        async def on_commit(value):
            commits.append(value)

        async def on_failure(value):
            failures.append(value)

        coordinator = semantic.SemanticTranslationCoordinator(
            SlowProvider(), on_commit, on_failure,
            config=semantic.SemanticTranslationConfig(
                min_request_interval_s=.001, max_hold_s=.1,
                request_timeout_s=.02, max_total_age_s=1,
            ),
        )
        original_payload = object()
        unit = semantic.SemanticPendingUnit(
            "pending-1", "원문 보존", "ko", "track-1", "epoch-1",
            time.monotonic(), payload=original_payload,
        )
        try:
            await coordinator.append(unit, boundary=True)
            await asyncio.wait_for(coordinator.flush(), timeout=1)
        finally:
            await coordinator.close(flush=False)
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        self.assertEqual(commits, [])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].reason, "model_timeout")
        self.assertEqual(failures[0].units, (unit,))
        self.assertIs(failures[0].units[0].payload, original_payload)
        self.assertEqual(failures[0].units[0].text, "원문 보존")
        self.assertEqual(coordinator.pending_units, ())
        self.assertEqual(coordinator.counts["timeouts"], 1)


if __name__ == "__main__":
    unittest.main()
