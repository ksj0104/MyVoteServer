"""Complete overlap children and task-local source-only failure protection."""

import asyncio
import json
from pathlib import Path
import runpy
import time
from types import SimpleNamespace
import unittest

try:
    import httpx  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Use MyVote-Mac-Demo/.venv/bin/python") from exc

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "tests/verified_engine.py"))["load_verified_engine"]()
from myvote_engine import overlap_pipeline
from myvote_engine.captions import CaptionSegment
from myvote_engine.priority_translation import AdmittedTranslationProvider, TranslationAdmission
from myvote_engine.semantic_translation import InvalidSemanticResponse
from myvote_engine.translation import Chunk, ProviderError, SemanticTranslationRequest

install = runpy.run_path(str(ROOT / "scripts/semantic_quality_output.py"))["install_quality_output"]


class Provider:
    def __init__(self, packet):
        self.packet = packet
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield Chunk(json.dumps(self.packet), completed=True, finish_reason="stop")


class QualityOutputTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        semantic = SimpleNamespace(SemanticCaptionPipeline=type("Base", (), {}))
        overlap = SimpleNamespace(OverlapCaptionCoordinator=overlap_pipeline.OverlapCaptionCoordinator)
        metadata = install(semantic, overlap)
        self.assertEqual(install(semantic, overlap), metadata)
        coordinator = object.__new__(overlap.OverlapCaptionCoordinator)
        coordinator.session = SimpleNamespace(config=SimpleNamespace(source_language="en"))
        coordinator.counts = {"overlap_translations_completed": 0}
        child = CaptionSegment("child", "track", 100, 200, "The report is ready.",
                               source_revision=1, target_language="ko", source_language="en")
        return coordinator, child

    async def test_complete_child_uses_whole_semantic_source_and_secondary_admission(self):
        coordinator, child = self.make()
        base = Provider({"action": "commit", "through_id": "overlap-complete-source", "text": "보고서가 준비됐습니다."})
        admission = TranslationAdmission()
        provider = AdmittedTranslationProvider(base, admission, secondary=True)
        deadline = time.monotonic() + 10
        result = await coordinator._translate(child, "en", provider, deadline, 7)
        request, = base.requests
        self.assertIsInstance(request, SemanticTranslationRequest)
        self.assertEqual(request.request_id, child.segment_id)
        self.assertEqual(request.units[0].text, child.source_text)
        self.assertEqual(len(request.units), 1)
        self.assertEqual(request.context, ())
        self.assertFalse(request.force_flush)
        self.assertEqual(request.deadline_monotonic, deadline)
        self.assertEqual(result.translation.text, "보고서가 준비됐습니다.")
        self.assertEqual(result.translation.provider_generation, 7)
        self.assertEqual(result.source_text, child.source_text)
        self.assertEqual((admission.secondary_calls, admission.primary_calls), (1, 0))
        self.assertEqual(coordinator.counts["overlap_translations_completed"], 1)

    async def test_incomplete_child_has_no_translation_result(self):
        coordinator, child = self.make()
        provider = Provider({"action": "wait"})
        with self.assertRaises(ProviderError) as caught:
            await coordinator._translate(child, "en", provider, time.monotonic() + 10, 0)
        self.assertEqual(caught.exception.category, "incomplete_source")
        self.assertEqual(coordinator.counts["overlap_translations_completed"], 0)

    async def test_unknown_child_id_cannot_authorize_a_translation(self):
        coordinator, child = self.make()
        provider = Provider({"action": "commit", "through_id": "invented", "text": "bad"})
        with self.assertRaises(InvalidSemanticResponse):
            await coordinator._translate(child, "en", provider, time.monotonic() + 10, 0)
        self.assertEqual(coordinator.counts["overlap_translations_completed"], 0)

    async def test_source_only_scope_skips_registration_without_affecting_other_tasks(self):
        recorded = []
        entered, release = asyncio.Event(), asyncio.Event()

        class BaseOverlap:
            async def register_caption(self, caption, words, anchor, language):
                recorded.append(caption)

        class BasePipeline:
            async def _failure(self, result):
                await self.overlap.register_caption("failed", (), None, "en")
                entered.set()
                await release.wait()
                raise ValueError("sink failure")

        semantic = SimpleNamespace(SemanticCaptionPipeline=BasePipeline)
        overlap = SimpleNamespace(OverlapCaptionCoordinator=BaseOverlap)
        install(semantic, overlap)
        pipe = semantic.SemanticCaptionPipeline()
        pipe.overlap = overlap.OverlapCaptionCoordinator()
        task = asyncio.create_task(pipe._failure(None))
        await asyncio.wait_for(entered.wait(), 1)
        await pipe.overlap.register_caption("concurrent valid", (), None, "en")
        release.set()
        with self.assertRaisesRegex(ValueError, "sink failure"):
            await task
        await pipe.overlap.register_caption("later valid", (), None, "en")
        self.assertEqual(recorded, ["concurrent valid", "later valid"])


if __name__ == "__main__":
    unittest.main()
