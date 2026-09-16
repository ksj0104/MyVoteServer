"""Opt-in JSON schema transport for verified context reviews and selection.

Install after loading the verified translation module and before constructing a
gateway provider. The upstream parser and ordinary translation payloads remain
unchanged. Structured output API: https://lmstudio.ai/docs/developer/openai-compat/structured-output
"""


def _context_review_response_format():
    """Build a fresh schema so one request cannot alter a later request."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "myvote_context_review",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "corrections": {
                        "type": "array",
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "segment_id": {"type": "string"},
                                "source_revision": {"type": "integer"},
                                "result_revision": {"type": "integer"},
                                "text": {"type": "string"},
                            },
                            "required": [
                                "segment_id", "source_revision", "result_revision", "text",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["corrections"],
                "additionalProperties": False,
            },
        },
    }


def install_context_review_schema(translation_module):
    """Install once, preserving the provider's constructor and stream parser.

    The caller explicitly enables compatibility for its configured LM Studio
    model. No model name is inferred here. Existing provider instances are not
    changed. Return the installed class, including on repeated calls.
    """
    provider_type = translation_module.LMStudioProvider
    if getattr(provider_type, "_myvote_context_review_json_schema_v1", False):
        return provider_type
    review_type = translation_module.ContextReviewRequest

    class ContextReviewJsonSchemaProvider(provider_type):
        _myvote_context_review_json_schema_v1 = True

        def _payload(self, request):
            payload = super()._payload(request)
            if isinstance(request, review_type):
                payload = dict(payload)
                payload["response_format"] = _context_review_response_format()
            return payload

    translation_module.LMStudioProvider = ContextReviewJsonSchemaProvider
    return ContextReviewJsonSchemaProvider


def _selection_response_format(request):
    """Constrain only boundary decisions; never ask TranslateGemma for JSON."""
    identifiers = ([request.units[-1].unit_id] if request.force_flush
                   else [unit.unit_id for unit in request.units])
    commit = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["commit"]},
            "through_id": {"type": "string", "enum": identifiers},
        },
        "required": ["action", "through_id"],
        "additionalProperties": False,
    }
    wait = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["wait"]}},
        "required": ["action"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {
        "name": "myvote_semantic_selection", "strict": True,
        "schema": commit if request.force_flush else {"anyOf": [wait, commit]},
    }}


def install_selection_schema(orchestrated_module):
    """Patch only the dedicated selector, not generic semantic translations.

    Install the review schema before importing orchestrated_translation, then
    call this before constructing its provider. All upstream parsers are kept.
    """
    provider_type = orchestrated_module._SelectionProvider
    if getattr(provider_type, "_myvote_selection_json_schema_v1", False):
        return provider_type
    request_type = orchestrated_module.SemanticTranslationRequest

    class SelectionJsonSchemaProvider(provider_type):
        _myvote_selection_json_schema_v1 = True

        def _payload(self, request):
            payload = super()._payload(request)
            if isinstance(request, request_type):
                payload = dict(payload)
                payload["response_format"] = _selection_response_format(request)
            return payload

    orchestrated_module._SelectionProvider = SelectionJsonSchemaProvider
    return SelectionJsonSchemaProvider
