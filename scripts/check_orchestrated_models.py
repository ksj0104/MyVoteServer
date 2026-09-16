"""Explicit local-model smoke test using synthetic text, not Windows audio.

Run with MyVote-Mac-Demo/.venv/bin/python -I scripts/check_orchestrated_models.py.
This sends bounded requests to the configured loopback API. It neither starts
the audio server nor changes models, certificates, or upstream engine files.
"""

import argparse
import asyncio
import json
from pathlib import Path
import runpy
import time
import tomllib


ROOT = Path(__file__).resolve().parents[1]


QUALITY_CASES = (
    # A short fragment is ready only when an explicit contextual question makes
    # it a complete answer. Length and punctuation alone do not authorize it.
    ("en_noun_phrase_wait", ("The", " project", " manager"), (), None, False, "en"),
    ("en_noun_phrase_answer", ("The", " project", " manager"),
     ("Who approved the change?",), 3, False, "en"),
    ("ko_noun_phrase_wait", ("개발", " 팀장님"), (), None, False, "ko"),
    ("ko_noun_phrase_answer", ("개발", " 팀장님"),
     ("누가 변경을 승인했나요?",), 2, False, "ko"),
    ("en_attached_contrast_wait", ("The", " report", " is", " ready,", " but"),
     (), None, False, "en"),
    ("ko_connective_wait", ("보고서는", " 준비됐지만"), (), None, False, "ko"),
    ("ko_connective_complete", ("보고서는", " 준비됐지만", " 검토는", " 끝나지", " 않았어요."),
     (), 5, False, "ko"),
    ("en_attached_condition_wait", ("We", " will", " deploy", " only", " if", " the", " tests"),
     (), None, False, "en"),
    ("en_attached_condition_complete", ("We", " will", " deploy", " only", " if", " the", " tests", " pass."),
     (), 8, False, "en"),
    ("ko_condition_without_main_clause", ("검사가", " 끝나면"), (), None, False, "ko"),
    ("en_quantity_missing_unit", ("The", " shipment", " weighs", " about", " twenty"),
     (), None, False, "en"),
    ("en_quantity_with_unit", ("The", " shipment", " weighs", " about", " twenty", " kilograms."),
     (), 6, False, "en"),
    ("ko_quantity_missing_currency", ("가격은", " 백이십만"), (), None, False, "ko"),
    ("ko_quantity_with_currency", ("가격은", " 백이십만", " 원이에요."), (), 3, False, "ko"),
    ("en_negation_missing_complement", ("It", " is", " not", " because"),
     (), None, False, "en"),
    ("en_negation_complete_complement", ("It", " is", " not", " because", " of", " the", " cost."),
     (), 7, False, "en"),
    ("ko_negation_missing_alternative", ("제가", " 원한", " 것은", " 돈이", " 아니라"),
     (), None, False, "ko"),
    ("ko_negation_complete_alternative", ("제가", " 원한", " 것은", " 돈이", " 아니라", " 시간이에요."),
     (), 6, False, "ko"),
    ("en_one_unit_complete_answer", ("Yes.",), ("Did it work?",), 1, False, "en"),
    ("ko_one_unit_complete_answer", ("아직이요.",), ("작업이 끝났나요?",), 1, False, "ko"),
)


async def collect(stream):
    text, usage, completed = "", {}, False
    first_ms = None
    started = time.monotonic()
    try:
        async for chunk in stream:
            if chunk.text and first_ms is None:
                first_ms = round((time.monotonic() - started) * 1000, 1)
            text += chunk.text
            if len(text) > 32768:
                raise ValueError("Test output exceeds limit")
            if chunk.completed:
                if completed or chunk.finish_reason != "stop":
                    raise ValueError("Unexpected terminal result")
                completed, usage = True, chunk.usage
    finally:
        await stream.aclose()
    if not completed or not text.strip():
        raise ValueError("Incomplete model result")
    return {"text": text, "first_ms": first_ms,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1), "usage": usage}


async def run(benchmark, boundaries=False, quality=False):
    config = tomllib.loads((ROOT / "server.toml").read_text())
    llm = config["lmstudio"]
    if llm.get("translation_profile") != "translategemma" or llm["base_url"] != "http://127.0.0.1:1234":
        raise ValueError("This check requires the dedicated profile and local API")
    update = (ROOT / config["speaker_update"]["bundle_dir"]).resolve(strict=True)
    support = runpy.run_path(str(ROOT / "scripts/start_update_with_compat.py"))
    launcher, manifest, _ = support["load_update"](update)
    launcher.install_engine_importer(update, manifest=manifest)
    from myvote_engine import translation as tr
    if llm.get("context_review_json_schema", False):
        compat = runpy.run_path(str(ROOT / "scripts/gemma_json_compat.py"))
        compat["install_context_review_schema"](tr)
    from myvote_engine import orchestrated_translation as ot
    if llm.get("context_review_json_schema", False):
        compat["install_selection_schema"](ot)
    prompt_metadata = {}
    if document := llm.get("semantic_boundary_prompt"):
        prompt_support = runpy.run_path(str(ROOT / "scripts/gemma_boundary_prompt.py"))
        prompt_metadata = prompt_support["install_boundary_prompt"](
            ot, ROOT / document, refine_ready_prefix=llm.get("semantic_boundary_refinement", False),
            quality_first=llm.get("semantic_quality_first", False),
            low_latency=llm.get("semantic_low_latency", False))
    if llm.get("semantic_quality_first", False):
        provider_support = runpy.run_path(str(ROOT / "scripts/semantic_quality_provider.py"))
        selection_s, translation_s = llm.get("semantic_selection_timeout_s", 5), llm.get("semantic_translation_timeout_s", 15)
        prompt_metadata.update(provider_support["install_quality_provider"](
            ot, selection_timeout_s=selection_s, translation_timeout_s=translation_s,
            request_timeout_s=selection_s + translation_s,
            residual_flush=bool(llm.get("semantic_inactivity_flush_s", 0)),
            early_prefix_recheck=llm.get("semantic_low_latency", False)))
    from myvote_engine.semantic_translation import parse_semantic_response
    from myvote_engine.context_refinement import parse_context_corrections
    import httpx

    print(json.dumps({"kind": "model_check.start", "release": manifest["release"],
                      **prompt_metadata,
                      "scope": "synthetic_local_model_requests_not_windows_e2e"}), flush=True)
    failures = []

    async def check(name, stream, validator=None, *, timeout_s=20):
        result = None
        try:
            async with asyncio.timeout(timeout_s):
                result = await collect(stream)
            if validator is not None:
                validator(result["text"])
            print(json.dumps({"check": name, "ok": True, **result}, ensure_ascii=False), flush=True)
            return result
        except Exception as exc:
            failures.append(name)
            print(json.dumps({"check": name, "ok": False, **(result or {}), "error": type(exc).__name__,
                              "category": getattr(exc, "category", None), "detail": str(exc)},
                             ensure_ascii=False), flush=True)
            return None

    async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
        provider = ot.OrchestratedTranslationProvider(
            client, llm["base_url"], llm["orchestrator_model_id"],
            tuple(llm.get("translation_model_ids") or [llm["model_id"]]),
            translation_concurrency=llm.get("translation_workers", 4),
        )
        units = (tr.SemanticUnit("u1", "Please close "), tr.SemanticUnit("u2", "the window."))
        forced = tr.SemanticTranslationRequest("check-forced", units, "en", "ko", force_flush=True, budget_ms=3500)
        # Compile/exercise both selection schema shapes outside the live 1s
        # selection deadline. Actual semantic checks below keep all deadlines.
        await check("warmup_selection_commit", provider._orchestrator.stream(forced),
                    lambda text: ot._selection(text, forced))
        pending = tr.SemanticTranslationRequest("check-wait", (tr.SemanticUnit("w1", "If"),), "en", "ko", budget_ms=3500)
        await check("warmup_selection_open", provider._orchestrator.stream(pending),
                    lambda text: ot._selection(text, pending))
        await check("warmup_translation", provider.stream(tr.TranslationRequest(
            "warmup", 1, "Ready.", "en", "ko", budget_ms=15000)))
        await check("semantic_en_ko", provider.semantic_stream(forced),
                    lambda text: parse_semantic_response(text, units, force_flush=True))
        await check("semantic_open", provider.semantic_stream(pending),
                    lambda text: parse_semantic_response(text, pending.units))
        korean_units = (tr.SemanticUnit("k1", "안녕 하세요. "), tr.SemanticUnit("k2", "오늘 회의를 시작 하겠습니다."))
        korean = tr.SemanticTranslationRequest("check-ko", korean_units, "ko", "ko", force_flush=True, budget_ms=3500)
        await check("semantic_ko_ko", provider.semantic_stream(korean),
                    lambda text: parse_semantic_response(text, korean_units, force_flush=True))
        targets = (tr.ContextReviewTarget("bank", 1, 1, "He sat on the bank.", "그는 은행에 앉았다.", "en", "ko"),)
        review = tr.ContextReviewRequest("check-review", targets,
            (tr.ContextClause("river", 1, "The river flowed beside him.", "en"),), budget_ms=4000)
        await check("context_review", provider.revision_stream(review),
                    lambda text: parse_context_corrections(text, targets))

        if boundaries or quality:
            # Hand-written development cases, not the separate 18-case report
            # in the prompt document. Expected end is one-based; None = wait.
            # Quality-first permits fragments only as explicit question replies:
            # an unfinished contextual phrase alone no longer makes one ready.
            context_completion_end = None if llm.get("semantic_quality_first", False) else 2
            cases = (
                ("complete_then_tail", ("The", " meeting", " ended.", " Tomorrow", " we", " will"), (), 3, False),
                ("filler_then_complete", ("Well,", " the", " repair", " is", " finished."), (), 5, False),
                ("short_reply", ("Not", " quite."), ("Is the report ready?",), 2, False),
                ("duration_complete", ("Around", " ninety", " minutes."), ("How long did the repair take?",), 3, False),
                ("number_reply", ("Roughly", " twelve."), ("How many people attended?",), 2, False),
                ("duration_incomplete", ("The", " journey", " lasted", " roughly", " twelve"), (), None, False),
                ("complete_question_preposition", ("What", " was", " that", " for?"), (), 4, False),
                ("missing_complement", ("The", " meeting", " begins", " at"), (), None, False),
                ("unfinished_condition", ("It", " succeeds", " only", " if"), (), None, False),
                ("filler_then_unfinished", ("Yeah,", " because", " the", " report", " is"), (), None, False),
                ("negation_complete", ("It", " is", " not", " ready."), (), 4, False),
                ("prefix_before_question", ("We", " finished", " the", " report.", " How", " did", " you"), (), 4, False),
                ("context_completion", ("twelve", " minutes."), ("The journey lasted roughly",), context_completion_end, False),
                ("forced_incomplete", ("If", " it", " rains"), (), 3, True),
                ("forced_all", ("The", " meeting", " ended.", " Tomorrow", " we", " will"), (), 6, True),
                ("two_complete_thoughts", ("The", " meeting", " ended.", " We", " went", " home."), (), 6, False),
                ("two_complete_then_tail", ("The meeting", " ended.", " We went", " home.", " Tomorrow", " we will"), (), 4, False),
                ("ready_then_condition", ("We", " are ready.", " It", " works", " only if"), (), 2, False),
            )
            cases = [(*case, "en") for case in cases] + [
                ("ko_two_complete", ("회의가", " 끝났어요.", " 우리는", " 집에", " 갔어요."), (), 5, False, "ko"),
                ("ko_two_then_tail", ("회의가", " 끝났어요.", " 우리는", " 집에", " 갔어요.", " 내일은", " 회의를"), (), 5, False, "ko"),
                ("ko_negation", ("준비가", " 되지", " 않았어요."), (), 3, False, "ko"),
                ("ko_ready_then_condition", ("준비는", " 끝났어요.", " 내일", " 비가", " 오면"), (), 2, False, "ko"),
                ("ko_amount_incomplete", ("십이만",), ("금액과 통화를 함께 알려주세요.",), None, False, "ko"),
                ("ko_amount_complete", ("십이만", " 원입니다."), ("금액과 통화를 함께 알려주세요.",), 2, False, "ko"),
                ("ko_conditional_wait", ("비가", " 오면"), (), None, False, "ko"),
                ("ko_conditional_forced", ("비가", " 오면"), (), 2, True, "ko"),
            ]
            if quality:
                cases = []
            if quality or llm.get("semantic_quality_first", False):
                cases.extend(QUALITY_CASES)
            if llm.get("semantic_low_latency", False):
                earliest = {"two_complete_thoughts": 3, "two_complete_then_tail": 2,
                            "ko_two_complete": 2, "ko_two_then_tail": 2}
                cases = [(name, fragments, context, earliest.get(name, end), force, language)
                         for name, fragments, context, end, force, language in cases]
                cases.extend((
                    ("en_complete_without_stop", ("The", " report", " is", " ready"), (), 4, False, "en"),
                    ("en_independent_additive_tail", ("The", " meeting", " ended,", " and", " tomorrow", " we", " will"), (), 3, False, "en"),
                    ("ko_independent_additive_tail", ("회의가", " 끝났고", " 내일은", " 회의를"), (), 2, False, "ko"),
                    ("ko_auxiliary_predicate", ("저는", " 집에", " 가고", " 싶어요."), (), 4, False, "ko"),
                ))
            for name, fragments, context, expected_end, force, source_language in cases:
                case_units = tuple(tr.SemanticUnit(f"{name}-unit-{index}", fragment)
                                   for index, fragment in enumerate(fragments, 1))
                request = tr.SemanticTranslationRequest(name, case_units, source_language, "ko",
                    context=context, force_flush=force, budget_ms=3500)
                expected = None if expected_end is None else case_units[expected_end - 1].unit_id

                def validate(text, request=request, expected=expected):
                    actual = ot._selection(text, request)
                    if actual != expected:
                        raise ValueError(f"Boundary mismatch: expected {expected!r}, got {actual!r}; response={text}")

                await check("boundary_" + name, provider._orchestrator.stream(request),
                            validate, timeout_s=1)

        if benchmark:
            sources = ("Please close the window.", "The meeting starts at nine.",
                       "We will send the report tomorrow.", "Thank you for your help.")
            # Same four independent inputs for every concurrency; no audio or
            # ASR contention. This does not establish production p95 latency.
            for workers in (1, 2, 4):
                bench = ot.OrchestratedTranslationProvider(
                    client, llm["base_url"], llm["orchestrator_model_id"], (provider.translation_models[0],),
                    translation_concurrency=workers,
                )
                started = time.monotonic()
                results = await asyncio.gather(*(check(f"translation_workers_{workers}_item_{index}",
                    bench.stream(tr.TranslationRequest(f"bench-{workers}-{index}", 1, source,
                                                       "en", "ko", budget_ms=2500)))
                    for index, source in enumerate(sources)))
                print(json.dumps({"benchmark_workers": workers, "requests": len(sources),
                                  "completed": sum(result is not None for result in results),
                                  "wall_ms": round((time.monotonic() - started) * 1000, 1)}), flush=True)
                await bench.aclose()
        await provider.aclose()
    print(json.dumps({"kind": "model_check.finished", "failures": failures}), flush=True)
    return bool(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="Compare four synthetic translation requests at concurrency 1/2/4")
    boundary_mode = parser.add_mutually_exclusive_group()
    boundary_mode.add_argument("--boundaries", action="store_true",
        help="Check 26 boundary cases, plus 20 quality and 4 early-complete cases for the configured profile")
    boundary_mode.add_argument("--quality", action="store_true",
        help="Run the basic diagnostics and only the 20 quality boundary cases")
    args = parser.parse_args()
    return asyncio.run(run(args.benchmark, args.boundaries, args.quality))


if __name__ == "__main__":
    raise SystemExit(main())
