"""Opt-in bounded quality inference over the unchanged verified provider.

The caller owns the HTTP client, source lifecycle and stale-result authorization.
Only semantic requests change: longer stage budgets and an explicit local-model
fallback for a completed prefix that cannot fit TranslateGemma's admission cap.
An optional residual-flush profile publishes the literal remaining source only
on an explicitly forced request; this is NOT a claim of semantic completion.
"""

import asyncio
from dataclasses import replace
import json
import math
import time


QUALITY_SELECTION_TIMEOUT_S = 5.0
QUALITY_TRANSLATION_TIMEOUT_S = 15.0
QUALITY_REQUEST_TIMEOUT_S = 20.0
QUALITY_FALLBACK_MAX_TOKENS = 2048
QUALITY_WHOLE_SOURCE_RECHECK_MIN_UNITS = 8
CLAUSE_ENDPOINT_PUNCTUATION = ".?!,;:。！？；：，"

RESIDUAL_SOURCE_INSTRUCTION = """RESIDUAL SOURCE PUBLICATION
An explicit input/session boundary requests publication of the remaining source. This boundary does NOT mean the words form a complete thought.
Render only the literal supplied source_text according to the requested translation or same-language correction operation. It may end mid-sentence, condition, number, negation, contrast, or unfinished phrase. Preserve that incompleteness in the output: do not add a missing subject, predicate, complement, amount, unit, consequence, reason, or inferred continuation. Do not answer a question contained in source_text. Context may disambiguate existing words but must never supply missing speech.
An unfinished translation fragment is correct here. Never manufacture a grammatically complete target sentence by guessing what a test, reviewer, manager or result will do. In particular, a missing predicate must remain missing: do not infer passing, succeeding, approving, finishing or being acceptable. If target-language word order would otherwise hide the missing source, an ellipsis (…) may mark the absent speech. That ellipsis is an incompleteness marker, not permission to add words or meaning. Do not add explanations or labels. Return only the rendition of the existing source.
Examples of faithful unfinished renditions, not phrases to copy into unrelated input:
English source 'only if the tests' -> Korean '검사가 …인 경우에만'. Do NOT supply '통과하면', '성공하면' or '완료되면'; the source contains no such predicate.
English source 'The shipment weighs about twenty' -> Korean '화물의 무게는 약 20…'. Do NOT invent kilograms, pounds or any other unit.
Korean source '담당자가 확인하면' -> English 'If the person in charge confirms…'. Do NOT invent the missing consequence or what is confirmed.
Korean source '제가 원한 것은 돈이 아니라' -> English 'What I wanted was not money, but…'. Do NOT invent the missing alternative."""


def install_quality_provider(
        orchestrated_module, *, selection_timeout_s=QUALITY_SELECTION_TIMEOUT_S,
        translation_timeout_s=QUALITY_TRANSLATION_TIMEOUT_S,
        request_timeout_s=QUALITY_REQUEST_TIMEOUT_S,
        fallback_max_tokens=QUALITY_FALLBACK_MAX_TOKENS,
        whole_source_recheck_min_units=QUALITY_WHOLE_SOURCE_RECHECK_MIN_UNITS,
        residual_flush=False, early_prefix_recheck=False):
    """Install once before constructing providers; return fresh safe metadata."""
    values = (selection_timeout_s, translation_timeout_s, request_timeout_s)
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not 0 < value <= 120 for value in values):
        raise ValueError("Quality inference timeouts must be finite seconds in (0, 120]")
    if request_timeout_s < max(selection_timeout_s, translation_timeout_s):
        raise ValueError("Quality request timeout must cover either inference stage")
    if type(fallback_max_tokens) is not int or not 512 <= fallback_max_tokens <= 4096:
        raise ValueError("Quality fallback output tokens must be in [512, 4096]")
    if (type(whole_source_recheck_min_units) is not int
            or not 2 <= whole_source_recheck_min_units <= 64):
        raise ValueError("Whole-source recheck threshold must be in [2, 64]")
    if type(residual_flush) is not bool:
        raise ValueError("Residual flush must be true or false")
    if type(early_prefix_recheck) is not bool:
        raise ValueError("Early-prefix recheck must be true or false")
    signature = (*values, fallback_max_tokens, whole_source_recheck_min_units,
                 residual_flush, early_prefix_recheck)
    metadata = {
        "semantic_quality_provider": "bounded-complete-prefix-v1",
        "semantic_orchestrator_budget_ms": int(selection_timeout_s * 1000),
        "semantic_translation_budget_ms": int(translation_timeout_s * 1000),
        "semantic_request_timeout_ms": int(request_timeout_s * 1000),
        "semantic_source_limit_fallback": "orchestrator_translation",
        "semantic_fallback_max_tokens": fallback_max_tokens,
        "semantic_whole_source_recheck": "wait-only-exact-whole-snapshot-v1",
        "semantic_whole_source_recheck_min_units": whole_source_recheck_min_units,
        "semantic_residual_flush": residual_flush,
    }
    if residual_flush:
        metadata["semantic_residual_flush_policy"] = "explicit-boundary-exact-source-v1"
    if early_prefix_recheck:
        metadata["semantic_boundary_projection_recheck"] = "wait-only-original-clause-endpoints-v1"
    original = orchestrated_module.OrchestratedTranslationProvider
    if getattr(original, "_myvote_quality_provider_v1", False):
        if original._myvote_quality_provider_values != signature:
            raise ValueError("Quality provider already installed with different settings")
        return metadata
    module = orchestrated_module

    def clause_projection(units):
        """Group for readability only; punctuation never authorizes a commit."""
        groups, current = [], []
        found_endpoint = False
        for unit in units:
            current.append(unit)
            if unit.text.rstrip().endswith(tuple(CLAUSE_ENDPOINT_PUNCTUATION)):
                groups.append(type(unit)(unit.unit_id, "".join(item.text for item in current)))
                current = []
                found_endpoint = True
        if current:
            # Every remaining word still reaches the selector. No tail hiding,
            # internal-unit splitting, inserted spacing or manufactured ID.
            last = current[-1]
            groups.append(type(last)(last.unit_id, "".join(item.text for item in current)))
        return tuple(groups) if found_endpoint else units

    def validate_source(request):
        # Gemma's larger input window does not widen source/language authority.
        module._language(request.source_language)
        module._language(request.target_language)
        if any(token in request.text for token in module._CONTROL_TOKENS):
            raise module.ProviderError("invalid_source", "Source contains reserved model turn tokens")
        try:
            request.text.encode("utf-8")
        except UnicodeEncodeError as invalid:
            raise module.ProviderError("invalid_source", "Source is not valid Unicode") from invalid

    def check_stage(timeout, deadline):
        # A transport suppressing cancellation never makes an expired result
        # valid. Preserve external cancellation instead of yielding a commit.
        if timeout.expired() or time.monotonic() >= deadline:
            raise module.ProviderError("timeout", "Quality inference stage expired")
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError

    class ResidualSourceProvider(module.LMStudioProvider):
        def _payload(self, request):
            if not isinstance(request, module.TranslationRequest):
                raise ValueError("Residual publication requires a literal TranslationRequest")
            payload = super()._payload(request)
            messages = payload["messages"]
            if (len(messages) != 2 or messages[0].get("role") != "system"
                    or messages[1].get("role") != "user"):
                raise ValueError("Unexpected residual translation message contract")
            # Source/context remain their exact JSON data. Never append a
            # policy instruction inside the transcript or modify model state.
            return {**payload, "messages": [
                {**messages[0], "content": messages[0]["content"] + "\n\n" + RESIDUAL_SOURCE_INSTRUCTION},
                messages[1],
            ]}

    class QualityOrchestratedProvider(original):
        _myvote_quality_provider_v1 = True
        _myvote_quality_provider_values = signature
        _myvote_quality_provider_original = original

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            selector = self._orchestrator
            # Do not mutate the selector's output limit while other requests
            # share it. This sibling uses the same client, model and gate.
            self._quality_fallback = module.LMStudioProvider(
                selector.client, selector.base_url, self.orchestrator_model,
                max_tokens=fallback_max_tokens)
            self._quality_residual = ResidualSourceProvider(
                selector.client, selector.base_url, self.orchestrator_model,
                max_tokens=fallback_max_tokens) if residual_flush else None

        async def _quality_source_stream(self, request):
            fallback = False
            if request.operation == "translation":
                try:
                    module.translategemma_prompt(request, output_tokens=self.max_tokens)
                except module.ProviderError as exc:
                    if exc.category != "source_limit":
                        raise
                    # The upstream character cap precedes these validations.
                    # A size fallback must never bypass them for oversized text.
                    validate_source(request)
                    fallback = True
            if fallback:
                async with self._orchestrator_gate:
                    async with module._closing_stream(self._quality_fallback.stream(request)) as stream:
                        async for chunk in stream:
                            yield chunk
            else:
                async with module._closing_stream(self._source_stream(request)) as stream:
                    async for chunk in stream:
                        yield chunk

        async def _quality_residual_result(self, request, deadline):
            """Inside normal semantic admission: exact forced source, no selector."""
            started = time.monotonic()
            result_deadline = min(deadline, started + translation_timeout_s)
            remaining = (result_deadline - time.monotonic()) * 1000
            if remaining <= 0:
                raise module.ProviderError("timeout", "Residual translation deadline expired")
            source = "".join(unit.text for unit in request.units)
            translating = module.TranslationRequest(
                request.request_id, 1, source, request.source_language, request.target_language,
                request.context if request.operation == "transcript_correction" else (),
                min(request.budget_ms, remaining), result_deadline)
            validate_source(translating)
            result_timeout = asyncio.timeout_at(result_deadline)
            async with result_timeout:
                async with self._orchestrator_gate:
                    result, usage = await module._collect(
                        self._quality_residual.stream(translating), limit=24000)
            check_stage(result_timeout, result_deadline)
            through = module._selection(json.dumps({"action": "commit",
                "through_id": request.units[-1].unit_id}), request)
            return module.Chunk(
                text=json.dumps({"action": "commit", "through_id": through, "text": result},
                                ensure_ascii=False), completed=True, finish_reason="stop",
                usage={"result": usage, "quality_selection_attempts": 0,
                       "quality_residual_flush": True,
                       "stage_ms": {"selection": 0.0,
                                    "translation": (time.monotonic() - started) * 1000}})

        async def semantic_stream(self, request):
            if not isinstance(request, module.SemanticTranslationRequest):
                raise ValueError("A SemanticTranslationRequest is required")
            now = time.monotonic()
            deadline = now + request_timeout_s
            if request.deadline_monotonic is not None:
                deadline = min(deadline, request.deadline_monotonic)
            bounded = replace(request, budget_ms=min(request.budget_ms, request_timeout_s * 1000),
                              deadline_monotonic=deadline)
            # Reuse closed admission, active-task accounting, cancellation and
            # conversion of stage/whole-request TimeoutError into ProviderError.
            async with self._scope(bounded) as deadline:
                if residual_flush and bounded.force_flush:
                    # Only the caller may authorize a silence/explicit drain.
                    # WAIT, length, elapsed time and punctuation never enter
                    # this branch by themselves. Outer admission is unchanged.
                    yield await self._quality_residual_result(bounded, deadline)
                    return
                started = time.monotonic()
                selection_deadline = min(deadline, started + selection_timeout_s)
                selection_timeout = asyncio.timeout_at(selection_deadline)
                selection_attempts = 1
                whole_source_already_checked = False
                async with selection_timeout:
                    async with self._orchestrator_gate:
                        decision, selection_usage = await module._collect(
                            self._orchestrator.stream(bounded), limit=8192)
                        check_stage(selection_timeout, selection_deadline)
                        through = module._selection(decision, bounded)
                        if through is None and not bounded.force_flush and early_prefix_recheck:
                            projected_units = clause_projection(bounded.units)
                            if projected_units != bounded.units:
                                projected = replace(bounded, units=projected_units)
                                retry, retry_usage = await module._collect(
                                    self._orchestrator.stream(projected), limit=8192)
                                check_stage(selection_timeout, selection_deadline)
                                through = module._selection(retry, projected)
                                if through is not None:
                                    through = module._selection(retry, bounded)
                                selection_attempts += 1
                                whole_source_already_checked = len(projected_units) == 1
                                selection_usage = {"initial": selection_usage,
                                                   "clause_projection": retry_usage}
                        if (through is None and not bounded.force_flush
                                and not whole_source_already_checked
                                and len(bounded.units) >= whole_source_recheck_min_units):
                            # A performance threshold, never proof of completion.
                            # Ask the same selector to judge exactly the same
                            # source continuously; a second WAIT remains WAIT.
                            # One block can authorize only the entire snapshot.
                            unit_type = type(bounded.units[0])
                            collapsed = replace(bounded, units=(unit_type(
                                bounded.units[-1].unit_id,
                                "".join(unit.text for unit in bounded.units)),))
                            retry, retry_usage = await module._collect(
                                self._orchestrator.stream(collapsed), limit=8192)
                            check_stage(selection_timeout, selection_deadline)
                            through = module._selection(retry, collapsed)
                            if through is not None:
                                # Keep both authorization contracts explicit;
                                # no model-supplied source or replacement IDs.
                                through = module._selection(retry, bounded)
                            selection_attempts += 1
                            selection_usage = {"initial": selection_usage,
                                               "whole_source_recheck": retry_usage}
                check_stage(selection_timeout, selection_deadline)
                selected_at = time.monotonic()
                if through is None:
                    yield module.Chunk(text='{"action":"wait"}', completed=True,
                                       finish_reason="stop", usage={"orchestrator": selection_usage,
                                                                   "quality_selection_attempts": selection_attempts})
                    return
                count = next(index for index, unit in enumerate(bounded.units, 1)
                             if unit.unit_id == through)
                source = "".join(unit.text for unit in bounded.units[:count])
                result_deadline = min(deadline, selected_at + translation_timeout_s)
                remaining = (result_deadline - time.monotonic()) * 1000
                if remaining <= 0:
                    raise module.ProviderError("timeout", "Quality translation deadline expired")
                translating = module.TranslationRequest(
                    bounded.request_id, 1, source, bounded.source_language, bounded.target_language,
                    bounded.context if bounded.operation == "transcript_correction" else (),
                    min(bounded.budget_ms, remaining), result_deadline)
                result_timeout = asyncio.timeout_at(result_deadline)
                async with result_timeout:
                    result, translation_usage = await module._collect(
                        self._quality_source_stream(translating), limit=24000)
                check_stage(result_timeout, result_deadline)
                yield module.Chunk(
                    text=json.dumps({"action": "commit", "through_id": through, "text": result},
                                    ensure_ascii=False), completed=True, finish_reason="stop",
                    usage={"orchestrator": selection_usage, "result": translation_usage,
                           "quality_selection_attempts": selection_attempts,
                           "stage_ms": {"selection": (selected_at - started) * 1000,
                                        "translation": (time.monotonic() - selected_at) * 1000}})

    orchestrated_module.OrchestratedTranslationProvider = QualityOrchestratedProvider
    return metadata
