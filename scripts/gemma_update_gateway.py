"""Verified engine plus opt-in review/selection JSON schemas before startup."""

import json
import os
from pathlib import Path
import runpy


def main():
    scripts = Path(__file__).resolve().parent
    update = Path(os.environ["MYVOTE_VERIFIED_UPDATE_DIR"]).resolve(strict=True)
    support = runpy.run_path(str(scripts / "start_update_with_compat.py"))
    launcher, manifest, _ = support["load_update"](update)
    launcher.install_engine_importer(update, manifest=manifest)
    from myvote_engine import translation
    compat = runpy.run_path(str(scripts / "gemma_json_compat.py"))
    compat["install_context_review_schema"](translation)
    # Import after the review patch: the dedicated selector inherits it too.
    from myvote_engine import orchestrated_translation
    compat["install_selection_schema"](orchestrated_translation)
    prompt_metadata = {}
    refined = os.environ.get("MYVOTE_BOUNDARY_REFINEMENT") == "1"
    quality_first = os.environ.get("MYVOTE_SEMANTIC_QUALITY_FIRST") == "1"
    low_latency = os.environ.get("MYVOTE_SEMANTIC_LOW_LATENCY") == "1"
    if low_latency and not quality_first:
        raise ValueError("Low latency requires quality-first safeguards")
    if quality_first and (not refined or not os.environ.get("MYVOTE_SEMANTIC_BOUNDARY_PROMPT")):
        raise ValueError("Quality-first requires boundary refinement and an explicit prompt")
    quality_timings = {}
    inactivity_s = 0
    if quality_first:
        quality_timings = json.loads(os.environ.get("MYVOTE_SEMANTIC_QUALITY_TIMINGS", "{}"))
        provider_support = runpy.run_path(str(scripts / "semantic_quality_provider.py"))
        selection_s = quality_timings.get("semantic_selection_timeout_s", provider_support["QUALITY_SELECTION_TIMEOUT_S"])
        translation_s = quality_timings.get("semantic_translation_timeout_s", provider_support["QUALITY_TRANSLATION_TIMEOUT_S"])
        request_s = selection_s + translation_s
        inactivity_s = quality_timings.get("semantic_inactivity_flush_s", 0)
        prompt_metadata.update(provider_support["install_quality_provider"](
            orchestrated_translation, selection_timeout_s=selection_s,
            translation_timeout_s=translation_s, request_timeout_s=request_s,
            residual_flush=bool(inactivity_s), early_prefix_recheck=low_latency))
    if document := os.environ.get("MYVOTE_SEMANTIC_BOUNDARY_PROMPT"):
        prompt_support = runpy.run_path(str(scripts / "gemma_boundary_prompt.py"))
        prompt_metadata.update(prompt_support["install_boundary_prompt"](
            orchestrated_translation, Path(document), refine_ready_prefix=refined,
            quality_first=quality_first, low_latency=low_latency))
    if refined:
        from myvote_engine import semantic_routing
        routing_support = runpy.run_path(str(scripts / "semantic_boundary_routing.py"))
        prompt_metadata.update(routing_support["install_routing"](semantic_routing))
        if quality_first:
            capacity_support = runpy.run_path(str(scripts / "semantic_quality_capacity.py"))
            prompt_metadata.update(capacity_support["install_quality_capacity"](semantic_routing))
            quality_support = runpy.run_path(str(scripts / "semantic_quality_policy.py"))
            prompt_metadata.update(quality_support["install_quality_policy"](
                semantic_routing, request_timeout_s=request_s,
                inactivity_flush_s=inactivity_s,
                min_request_interval_s=quality_timings.get("semantic_min_request_interval_s", quality_support["QUALITY_MIN_REQUEST_INTERVAL_S"]),
                max_hold_s=quality_timings.get("semantic_max_hold_s", quality_support["QUALITY_MAX_HOLD_S"]),
                max_total_age_s=quality_timings.get("semantic_total_age_s", quality_support["QUALITY_MAX_TOTAL_AGE_S"])))
        # These modules and their consumers must see the refined router/session.
        from myvote_engine import semantic_pipeline, pipeline, audio
        semantic_pipeline.SemanticTranslationRouter = semantic_routing.SemanticTranslationRouter
        pipeline_support = runpy.run_path(str(scripts / "semantic_boundary_pipeline.py"))
        prompt_metadata.update(pipeline_support["install_pipeline"](
            semantic_pipeline, pipeline, audio, quality_first=quality_first))
        if quality_first:
            from myvote_engine import overlap_pipeline
            output_support = runpy.run_path(str(scripts / "semantic_quality_output.py"))
            prompt_metadata.update(output_support["install_quality_output"](semantic_pipeline, overlap_pipeline))
            if inactivity_s:
                idle_support = runpy.run_path(str(scripts / "semantic_transcript_idle.py"))
                prompt_metadata.update(idle_support["install_transcript_idle"](
                    pipeline, inactivity_flush_s=inactivity_s))
            if low_latency and not inactivity_s:
                latency_support = runpy.run_path(str(scripts / "semantic_latency_metrics.py"))
                prompt_metadata.update(latency_support["install_latency_metrics"](
                    semantic_pipeline, target_latency_s=quality_timings.get("semantic_latency_target_s", 2)))
    # All imports of LMStudioProvider must see the opt-in subclass.
    from myvote_engine import gateway_server
    if quality_first:
        quality_gateway = runpy.run_path(str(scripts / "semantic_quality_gateway.py"))
        prompt_metadata.update(quality_gateway["install_gateway_metadata"](
            gateway_server,
            min_request_interval_s=prompt_metadata["semantic_min_request_interval_ms"] / 1000,
            max_hold_s=prompt_metadata["semantic_max_hold_ms"] / 1000,
            max_total_age_s=prompt_metadata["semantic_total_budget_ms"] / 1000,
            request_timeout_s=request_s, selection_timeout_s=selection_s,
            translation_timeout_s=translation_s,
            low_latency=low_latency,
            latency_target_s=quality_timings.get("semantic_latency_target_s", 2),
            inactivity_flush_s=inactivity_s,
        ))
    print(json.dumps({"kind": "gateway.compatibility",
                      "engine_revision": manifest["release"],
                      "context_review_json_schema": True,
                      "semantic_selection_json_schema": True,
                      **prompt_metadata,
                      "update_sources_modified": False}), flush=True)
    return gateway_server.main()


if __name__ == "__main__":
    raise SystemExit(main())
