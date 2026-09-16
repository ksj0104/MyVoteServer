"""Compatibility transport contracts without network, model, or engine imports."""

from types import SimpleNamespace
import unittest

from scripts.gemma_json_compat import install_context_review_schema, install_selection_schema


class TranslationRequest:
    def __init__(self, source_language, target_language):
        self.source_language = source_language
        self.target_language = target_language


class ContextReviewRequest:
    output_tokens = 512


class SemanticTranslationRequest:
    def __init__(self, *, force_flush=False):
        self.units = (SimpleNamespace(unit_id="u1"), SimpleNamespace(unit_id="u2"))
        self.force_flush = force_flush


class FakeLMStudioProvider:
    def __init__(self, model, *, max_tokens=128, options=None):
        if options is not None and set(options) - {"temperature", "seed"}:
            raise ValueError("Unexpected provider option")
        self.model = model
        self.max_tokens = max_tokens
        self.options = dict(options or {})
        self.last_payload = None

    def _payload(self, request):
        self.last_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": type(request).__name__}],
            "stream": True,
            "max_tokens": (request.output_tokens if isinstance(request, ContextReviewRequest)
                           else self.max_tokens),
            "temperature": 0,
            **self.options,
        }
        return self.last_payload


class GemmaJsonCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.module = SimpleNamespace(
            LMStudioProvider=FakeLMStudioProvider,
            ContextReviewRequest=ContextReviewRequest,
        )
        self.provider_type = install_context_review_schema(self.module)
        self.provider = self.provider_type("configured-model", options={"seed": 7})

    def test_translation_and_initial_correction_payloads_are_unchanged(self):
        baseline = FakeLMStudioProvider("configured-model", options={"seed": 7})
        for languages in (("en", "ko"), ("ko", "ko")):
            with self.subTest(languages=languages):
                request = TranslationRequest(*languages)
                actual = self.provider._payload(request)
                self.assertEqual(actual, baseline._payload(request))
                self.assertIs(actual, self.provider.last_payload)
                self.assertNotIn("response_format", actual)

    def test_only_context_review_adds_the_expected_schema(self):
        request = ContextReviewRequest()
        actual = self.provider._payload(request)
        baseline = FakeLMStudioProvider("configured-model", options={"seed": 7})._payload(request)
        response_format = actual.pop("response_format")
        self.assertEqual(actual, baseline)
        self.assertNotIn("response_format", self.provider.last_payload)
        self.assertEqual(response_format["type"], "json_schema")
        definition = response_format["json_schema"]
        self.assertEqual(definition["name"], "myvote_context_review")
        self.assertIs(definition["strict"], True)
        schema = definition["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["corrections"])
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(set(schema["properties"]), {"corrections"})
        corrections = schema["properties"]["corrections"]
        self.assertEqual(corrections["type"], "array")
        self.assertEqual(corrections["maxItems"], 2)
        self.assertNotIn("minItems", corrections)  # An unchanged review may return [].
        item = corrections["items"]
        self.assertEqual(item["type"], "object")
        self.assertIs(item["additionalProperties"], False)
        expected_types = {
            "segment_id": "string", "source_revision": "integer",
            "result_revision": "integer", "text": "string",
        }
        self.assertEqual(set(item["required"]), set(expected_types))
        self.assertEqual(item["properties"], {
            name: {"type": value} for name, value in expected_types.items()
        })

    def test_provider_option_validation_is_inherited(self):
        self.assertIs(self.provider_type.__init__, FakeLMStudioProvider.__init__)
        with self.assertRaisesRegex(ValueError, "Unexpected provider option"):
            self.provider_type("configured-model", options={"response_format": {}})

    def test_schema_mutation_cannot_affect_subsequent_requests_or_instances(self):
        first = self.provider._payload(ContextReviewRequest())["response_format"]
        first["json_schema"]["schema"]["properties"]["corrections"]["items"]["required"].clear()
        first["json_schema"]["schema"]["additionalProperties"] = True
        for provider in (self.provider, self.provider_type("another-configured-model")):
            with self.subTest(model=provider.model):
                second = provider._payload(ContextReviewRequest())["response_format"]
                self.assertIsNot(second, first)
                schema = second["json_schema"]["schema"]
                self.assertIs(schema["additionalProperties"], False)
                self.assertEqual(len(schema["properties"]["corrections"]["items"]["required"]), 4)
        ordinary = self.provider._payload(TranslationRequest("en", "ko"))
        self.assertNotIn("response_format", ordinary)

    def test_installation_is_idempotent_and_existing_instances_are_unchanged(self):
        original = FakeLMStudioProvider("existing-model")
        installed_again = install_context_review_schema(self.module)
        self.assertIs(installed_again, self.provider_type)
        self.assertIs(self.module.LMStudioProvider, self.provider_type)
        self.assertEqual(self.provider_type.__bases__, (FakeLMStudioProvider,))
        self.assertNotIn("response_format", original._payload(ContextReviewRequest()))


class SelectionCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.translation = SimpleNamespace(
            LMStudioProvider=FakeLMStudioProvider, ContextReviewRequest=ContextReviewRequest,
        )
        review_provider = install_context_review_schema(self.translation)

        class Selector(review_provider):
            def _payload(self, request):
                payload = super()._payload(request)
                if isinstance(request, SemanticTranslationRequest):
                    payload["messages"] = [{"role": "system", "content": "selection only"}]
                    payload["max_tokens"] = 128
                return payload

        self.original_selector = Selector
        self.orchestration = SimpleNamespace(
            _SelectionProvider=Selector, SemanticTranslationRequest=SemanticTranslationRequest,
        )
        self.provider_type = install_selection_schema(self.orchestration)
        self.provider = self.provider_type("orchestrator")

    def test_normal_selection_can_wait_or_select_only_supplied_ids(self):
        result = self.provider._payload(SemanticTranslationRequest())
        self.assertEqual(result["messages"][0]["content"], "selection only")
        self.assertEqual(result["max_tokens"], 128)
        schema = result["response_format"]["json_schema"]["schema"]
        wait, commit = schema["anyOf"]
        self.assertEqual(wait["properties"], {"action": {"type": "string", "enum": ["wait"]}})
        self.assertEqual(wait["required"], ["action"])
        self.assertEqual(commit["properties"]["through_id"]["enum"], ["u1", "u2"])
        self.assertEqual(commit["required"], ["action", "through_id"])
        for branch in (wait, commit):
            self.assertIs(branch["additionalProperties"], False)
            self.assertNotIn("text", branch["properties"])

    def test_force_flush_requires_commit_through_last_unit(self):
        schema = self.provider._payload(SemanticTranslationRequest(force_flush=True))[
            "response_format"]["json_schema"]["schema"]
        self.assertNotIn("anyOf", schema)
        self.assertEqual(schema["properties"]["action"]["enum"], ["commit"])
        self.assertEqual(schema["properties"]["through_id"]["enum"], ["u2"])

    def test_context_review_schema_survives_selector_patch(self):
        result = self.provider._payload(ContextReviewRequest())
        self.assertEqual(result["response_format"]["json_schema"]["name"], "myvote_context_review")
        self.assertEqual(result["max_tokens"], 512)

    def test_generic_semantic_and_plain_requests_are_unchanged(self):
        generic = self.translation.LMStudioProvider("generic")
        self.assertNotIn("response_format", generic._payload(SemanticTranslationRequest()))
        for languages in (("en", "ko"), ("ko", "ko")):
            self.assertNotIn("response_format", self.provider._payload(TranslationRequest(*languages)))

    def test_selection_schema_is_fresh_and_installation_is_idempotent(self):
        first = self.provider._payload(SemanticTranslationRequest())["response_format"]
        first["json_schema"]["schema"]["anyOf"][1]["properties"]["through_id"]["enum"].clear()
        second = self.provider._payload(SemanticTranslationRequest())["response_format"]
        self.assertEqual(second["json_schema"]["schema"]["anyOf"][1]["properties"]["through_id"]["enum"],
                         ["u1", "u2"])
        self.assertIs(install_selection_schema(self.orchestration), self.provider_type)
        self.assertIs(self.provider_type.__init__, FakeLMStudioProvider.__init__)
        self.assertNotIn("response_format", self.original_selector("existing")._payload(SemanticTranslationRequest()))


if __name__ == "__main__":
    unittest.main()
