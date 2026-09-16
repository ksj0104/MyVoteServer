"""Keep source-only failures and overlap corrections inside the quality policy."""

from contextvars import ContextVar


def install_quality_output(semantic_module, overlap_module):
    from myvote_engine import orchestrated_translation as orchestration
    from myvote_engine.semantic_translation import parse_semantic_response
    from myvote_engine.translation import Chunk, ProviderError, SemanticTranslationRequest, SemanticUnit

    pipeline_type = semantic_module.SemanticCaptionPipeline
    overlap_type = overlap_module.OverlapCaptionCoordinator
    marker = "_myvote_quality_output_v1"
    metadata = {"semantic_quality_output": "source-only-and-complete-overlap-v1"}
    installed = (getattr(pipeline_type, marker, False), getattr(overlap_type, marker, False))
    if all(installed):
        return metadata
    if any(installed):
        raise RuntimeError("Partially installed quality output guards")
    # A per-task scope, not a shared session flag: other captions may publish
    # concurrently while the failed caption's source events are being emitted.
    source_only = ContextVar("myvote_quality_source_only", default=False)

    class QualityCaptionPipeline(pipeline_type):
        _myvote_quality_output_v1 = True

        async def _failure(self, result):
            token = source_only.set(True)
            try:
                return await super()._failure(result)
            finally:
                source_only.reset(token)

    class CompleteChildProvider:
        def __init__(self, provider):
            self.provider = provider

        async def stream(self, request):
            # One indivisible replacement child: either its WHOLE meaning is
            # complete or the original atomic group stays in place. Do not
            # alter through_id after translating a different source span.
            semantic_request = SemanticTranslationRequest(
                request.segment_id,
                (SemanticUnit("overlap-complete-source", request.text),),
                request.source_language, request.target_language,
                context=(), force_flush=False, budget_ms=request.budget_ms,
                deadline_monotonic=request.deadline_monotonic,
            )
            # stream(), not semantic_stream(), preserves the upstream wrapper's
            # SECONDARY admission. The orchestrated provider dispatches by type.
            text, usage = await orchestration._collect(
                self.provider.stream(semantic_request), limit=32768)
            decision = parse_semantic_response(text, semantic_request.units)
            if decision.action != "commit":
                raise ProviderError("incomplete_source", "Overlap child is not a complete thought")
            yield Chunk(decision.text, completed=True, finish_reason="stop", usage=usage)

    class QualityOverlapCoordinator(overlap_type):
        _myvote_quality_output_v1 = True

        async def register_caption(self, caption, words, anchor, language):
            if source_only.get():
                return
            return await super().register_caption(caption, words, anchor, language)

        async def _translate(self, child, language, provider, deadline, generation):
            return await super()._translate(
                child, language, CompleteChildProvider(provider), deadline, generation)

    semantic_module.SemanticCaptionPipeline = QualityCaptionPipeline
    overlap_module.OverlapCaptionCoordinator = QualityOverlapCoordinator
    return metadata
