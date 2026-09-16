import unittest
from pydantic import ValidationError
from app.context.glossary import matching_entries
from app.context.manager import ContextEntry, ContextManager
from app.core.config import Settings
from app.core.models import ASREvent, ASRToken, GlossaryUpdate


class ContextAndConfigurationTests(unittest.TestCase):
    def test_paired_context_is_bounded_and_keeps_latest_pairs(self):
        context = ContextManager(3, 1500)
        context.replace([ContextEntry(str(i), f"source{i}", f"target{i}") for i in range(6)])
        source, target = context.snapshot()
        self.assertEqual(source, ["source3", "source4", "source5"])
        self.assertEqual(target, ["target3", "target4", "target5"])
        context = ContextManager(3, 40)
        context.replace([ContextEntry("1", "a", "b"), ContextEntry("2", "c", "d")])
        self.assertEqual(context.snapshot(), (["c"], ["d"]))
        context = ContextManager(0, 1500)
        context.replace([ContextEntry("1", "a", "b")])
        self.assertEqual(context.snapshot(), ([], []))

    def test_glossary_matches_terms_not_arbitrary_substrings_and_obeys_budget(self):
        self.assertEqual(matching_entries("CUDA and a hotel", {"CUDA": "쿠다", "hot": "뜨거운", "hotel": "호텔"}),
                         {"hotel": "호텔", "CUDA": "쿠다"})
        self.assertEqual(matching_entries("CUDA", {"CUDA": "쿠다"}, budget=1), {})
        self.assertEqual(matching_entries("한국어 번역", {"한국어": "Korean"}), {"한국어": "Korean"})
        self.assertEqual(matching_entries("CUDA를 쓰면 좋아요", {"CUDA": "쿠다"}), {"CUDA": "쿠다"})

    def test_deployment_limits_and_urls_are_validated(self):
        for overrides in ({"translation_workers": 0}, {"max_pending_jobs": -1},
                          {"min_segment_tokens": 30, "max_segment_tokens": 24},
                          {"translation_base_url": "file:///tmp/private"},
                          {"translation_base_url": "http://user:password@localhost:1234"},
                          {"stability_agreement_weight": 0, "stability_age_weight": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                Settings(_env_file=None, **overrides)

    def test_wire_rejects_nonfinite_values_and_invalid_token_times(self):
        with self.assertRaises(ValidationError):
            ASREvent(type="asr_partial", sequence=1, text="test", timestamp=float("nan"))
        with self.assertRaises(ValidationError):
            ASREvent(type="asr_partial", sequence=True, text="test")
        with self.assertRaises(ValidationError):
            ASRToken(text="test", start=2, end=1)
        with self.assertRaises(ValidationError):
            GlossaryUpdate(entries={"x": ""})
