"""Measure first untranslated preview word -> server translation event.

Use --run only while the existing audio server has no connected clients. This
uses synthetic ASR hypotheses, the real StreamingSession/stabilizer/coordinator
and configured local models. It does not measure microphone/Whisper time before
the first preview, gRPC delivery or Windows rendering, and is not a latency SLA.
No finish/hard flush is called to make a benchmark pass. Incomplete source waits
for more transcription, then the configured inactivity policy publishes its
literal residue. The no-PCM case exercises the normal ASR idle checkpoint too.
Measurements have no two-second pass/fail criterion: that is an input-idle timer,
not a promise that the translation finishes within two seconds of the first word.

Example: MyVote-Mac-Demo/.venv/bin/python -I scripts/check_translation_latency.py --run
Use --mock to check the harness without any model/network requests.
"""

import argparse
import asyncio
import json
import math
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import time
import tomllib


ROOT = Path(__file__).resolve().parents[1]
# Schedule entries are (wall-clock seconds after first hypothesis, word count).
# The final entry is a normal ASR endpoint, not a semantic hard flush.
SCENARIOS = {
    "incremental_400ms": ("The meeting starts at nine tomorrow.", ((0, 1), (.4, 6)), 6),
    "incremental_800ms": ("The meeting starts at nine tomorrow.", ((0, 1), (.4, 3), (.8, 6)), 6),
    "independent_clause_tail": ("The meeting ended, and tomorrow we will",
                                ((0, 1), (.4, 3), (.8, 7)), 3),
    "complete_first_batch": ("Please send the updated report today.", ((0, 6),), 6),
    "incomplete_exception": ("The shipment weighs about twenty", ((0, 1), (.4, 5)), None),
    "inactivity_residual": ("The shipment weighs about twenty", ((0, 1), (.4, 5)), 5),
    "no_pcm_idle_residual": ("The shipment weighs about twenty", ((0, 5),), 5),
}


def support(name):
    return runpy.run_path(str(ROOT / "scripts" / name))


def install(config):
    """Use the gateway's installer order and the CURRENT configured cadence."""
    llm = config["lmstudio"]
    if (llm.get("base_url") != "http://127.0.0.1:1234"
            or llm.get("translation_profile") != "translategemma"
            or not all(llm.get(key) is True for key in (
                "context_review_json_schema", "semantic_boundary_refinement", "semantic_quality_first"))):
        raise ValueError("This benchmark requires the local configured quality-first profile")
    update = (ROOT / config["speaker_update"]["bundle_dir"]).resolve(strict=True)
    launcher, manifest, _ = support("start_update_with_compat.py")["load_update"](update)
    launcher.install_engine_importer(update, manifest=manifest)
    from myvote_engine import translation
    compat = support("gemma_json_compat.py")
    compat["install_context_review_schema"](translation)
    from myvote_engine import orchestrated_translation as provider_module, semantic_routing as routing
    compat["install_selection_schema"](provider_module)
    metadata = support("gemma_boundary_prompt.py")["install_boundary_prompt"](
        provider_module, ROOT / llm["semantic_boundary_prompt"], refine_ready_prefix=True,
        quality_first=True, low_latency=llm.get("semantic_low_latency", False))
    provider_support = support("semantic_quality_provider.py")
    selection_s = llm.get("semantic_selection_timeout_s", provider_support["QUALITY_SELECTION_TIMEOUT_S"])
    translation_s = llm.get("semantic_translation_timeout_s", provider_support["QUALITY_TRANSLATION_TIMEOUT_S"])
    inactivity_s = llm.get("semantic_inactivity_flush_s", 0)
    metadata.update(provider_support["install_quality_provider"](
        provider_module, selection_timeout_s=selection_s, translation_timeout_s=translation_s,
        request_timeout_s=selection_s + translation_s, residual_flush=bool(inactivity_s),
        early_prefix_recheck=llm.get("semantic_low_latency", False)))
    support("semantic_boundary_routing.py")["install_routing"](routing)
    support("semantic_quality_capacity.py")["install_quality_capacity"](routing)
    quality = support("semantic_quality_policy.py")
    metadata.update(quality["install_quality_policy"](
        routing, request_timeout_s=selection_s + translation_s, inactivity_flush_s=inactivity_s,
        min_request_interval_s=llm.get("semantic_min_request_interval_s", quality["QUALITY_MIN_REQUEST_INTERVAL_S"]),
        max_hold_s=llm.get("semantic_max_hold_s", quality["QUALITY_MAX_HOLD_S"]),
        max_total_age_s=llm.get("semantic_total_age_s", quality["QUALITY_MAX_TOTAL_AGE_S"])))
    from myvote_engine import semantic_pipeline, pipeline, audio, overlap_pipeline
    semantic_pipeline.SemanticTranslationRouter = routing.SemanticTranslationRouter
    support("semantic_boundary_pipeline.py")["install_pipeline"](
        semantic_pipeline, pipeline, audio, quality_first=True)
    support("semantic_quality_output.py")["install_quality_output"](semantic_pipeline, overlap_pipeline)
    if inactivity_s:
        metadata.update(support("semantic_transcript_idle.py")["install_transcript_idle"](
            pipeline, inactivity_flush_s=inactivity_s))
    # The benchmark sink observes its first preview directly. Do not install
    # the retired latency-target instrumentation or change production behavior.
    return provider_module, pipeline, audio, metadata


def reject_active_audio_clients(port):
    executable = shutil.which("lsof")
    if executable is None:
        raise RuntimeError("lsof is required to check for active audio clients before real inference")
    result = subprocess.run([executable, "-nP", f"-iTCP:{port}", "-sTCP:ESTABLISHED", "-t"],
                            capture_output=True, text=True, timeout=5, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError("Cannot check active audio clients")
    if result.stdout.strip():
        raise RuntimeError("Disconnect audio clients before sharing the local model for this benchmark")


def mock_response(request):
    """Fixture decisions only, not another production boundary algorithm."""
    import httpx
    packet = json.loads(request.content)
    chat = "messages" in packet
    result = "Synthetic completed translation."
    if chat:
        data = json.loads(packet["messages"][1]["content"])
        if "units" in data:
            result = '{"action":"wait"}'
            source = ""
            complete = {"The meeting starts at nine tomorrow.", "The meeting ended,",
                        "Please send the updated report today."}
            for unit in data["units"]:
                source += unit["text"]
                if source in complete:
                    result = json.dumps({"action": "commit", "through_id": unit["unit_id"]})
                    break
    field = {"delta": {"content": result}} if chat else {"text": result}
    packets = [{"choices": [{"index": 0, **field, "finish_reason": None}]},
               {"choices": [{"index": 0, "finish_reason": "stop"}]}]
    body = "".join("data: " + json.dumps(packet) + "\n\n" for packet in packets) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


async def sample(name, provider, pipeline, audio, *, observation_s, inactivity_s):
    from myvote_engine.asr import ASRHypothesis, TimedWord
    source, schedule, expected_words = SCENARIOS[name]
    pieces = re.findall(r"\s*\S+", source)
    words = tuple(TimedWord(piece, index * 100_000_000, (index + 1) * 100_000_000)
                  for index, piece in enumerate(pieces))

    class SyntheticASR:
        def transcribe_pcm(self, samples, *, sample_rate, window_start_ns):
            if name == "no_pcm_idle_residual":
                return ASRHypothesis(window_start_ns,
                                     window_start_ns + len(samples) * audio.SAMPLE_NS, words, "en")
            count = int(samples[0])
            return ASRHypothesis(0, count * 100_000_000, words[:count], "en")

    first_preview_at = None
    first_translation_at = None
    last_transcription_at, last_transcription_text = None, None
    dispatches = {}
    translated = asyncio.Event()
    preview_seen = asyncio.Event()
    captions, results, failures = {}, [], []

    async def sink(event):
        nonlocal first_preview_at, first_translation_at, last_transcription_at, last_transcription_text
        now = time.monotonic()
        if event.kind == "transcript.updated":
            text = event.data.get("text", "").strip()
            if text and text != last_transcription_text:
                last_transcription_at, last_transcription_text = now, text
        if event.kind == "transcript.preview" and event.data.get("source_text", "").strip():
            if first_preview_at is None:
                first_preview_at = now
            preview_seen.set()
        elif event.kind == "caption.source":
            captions[event.segment_id] = event.data
        elif event.kind == "translation.completed":
            if first_translation_at is None:
                first_translation_at = now
            results.append((event.segment_id, event.data))
            translated.set()
        elif event.kind in ("translation.failed", "asr.failed", "pipeline.failed"):
            failures.append(event.data.get("error", event.kind))

    class ObservedProvider:
        """Observe after existing admission; never alter requests or results."""
        def __init__(self, actual):
            self.provider = actual

        def __getattr__(self, name):
            return getattr(self.provider, name)

        async def semantic_stream(self, request):
            dispatches[request.request_id] = (time.monotonic(), last_transcription_at,
                                              request.force_flush)
            stream = self.provider.semantic_stream(request)
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                await stream.aclose()

    session = pipeline.StreamingSession(
        "latency-synthetic-" + name, SyntheticASR(), ObservedProvider(provider), sink=sink,
        config=pipeline.PipelineConfig(source_language="en", target_language="ko",
                                       semantic_translation=True))
    started = time.monotonic()
    try:
        if name == "no_pcm_idle_residual":
            # An ordinary open speech window. No stop, gap, final flag or
            # manual builder flush: the actual idle adapter must finalize it.
            await session.feed(audio.AudioFrame("synthetic-track", "synthetic-epoch", 0, 0,
                                                (.1,) * 8000), .9)
            await asyncio.wait_for(preview_seen.wait(), 1)
        else:
            for index, (offset, count) in enumerate(schedule):
                await asyncio.sleep(max(0, started + offset - time.monotonic()))
                await session._process_window(audio.SpeechWindow(
                    name, "synthetic-track", "synthetic-epoch", 0, count * 100_000_000,
                    (float(count),), index == len(schedule) - 1, "endpoint", 0))
        observed_for = observation_s
        if name == "incomplete_exception" and inactivity_s:
            # Observe WAIT before the input-idle timer, not after a newly
            # authorized residual translation should have been dispatched.
            observed_for = min(observed_for, schedule[-1][0] + inactivity_s * .5)
        remaining = max(0, started + observed_for - time.monotonic())
        if not translated.is_set() and remaining:
            try:
                await asyncio.wait_for(translated.wait(), timeout=remaining)
            except TimeoutError:
                pass
        pending = session._semantic.coordinator.pending_units
        first_result = results[0][1] if results else {}
        caption = captions.get(results[0][0], {}) if results else {}
        dispatch = dispatches.get(caption.get("semantic_request_id"))
        measured = (None if first_preview_at is None or first_translation_at is None
                    else (first_translation_at - first_preview_at) * 1000)
        expected_source = None if expected_words is None else "".join(pieces[:expected_words]).strip()
        safe = (not results and bool(pending) if expected_words is None else
                bool(results) and caption.get("text") == expected_source)
        ok = safe and not failures and session.counts["asr_errors"] == 0
        if results:
            # Strict prefix publication starts at the first synthetic word.
            # Its first preview is the anchor, never last-word arrival time.
            ok = ok and measured is not None and caption.get("start_ns") == words[0].start_time_ns
        residual_expected = name in ("inactivity_residual", "no_pcm_idle_residual")
        if residual_expected:
            ok = ok and caption.get("semantic_force_flush") is True
        if name == "no_pcm_idle_residual":
            ok = ok and session.counts.get("transcript_idle_checkpoints", 0) >= 1
        report = {
            "case": name, "ok": bool(ok),
            "outcome": "translated" if results else "waiting_for_complete_meaning" if expected_words is None
                       else "no_translation_within_observation_window",
            "first_word_anchor": "server_transcript_preview",
            "first_word_preview_to_translation_ms": None if measured is None else round(measured, 1),
            "measurement": "synthetic_sink_first_preview_to_first_translation_event",
            "inactivity_flush_ms": inactivity_s * 1000,
            "residual_flush_expected": residual_expected,
            "residual_flush_observed": caption.get("semantic_force_flush") is True,
            "dispatch_after_last_transcription_ms": (None if not dispatch or dispatch[1] is None
                else round((dispatch[0] - dispatch[1]) * 1000, 1)),
            "translation_after_dispatch_ms": (None if not dispatch or first_translation_at is None
                else round((first_translation_at - dispatch[0]) * 1000, 1)),
            "transcript_idle_checkpoints": session.counts.get("transcript_idle_checkpoints", 0),
            "source_words": len(words), "expected_first_commit_words": expected_words,
            "first_commit_source_words": len(caption.get("text", "").split()),
            "completed_events": len(results), "pending_source_words": len(pending),
            "failures": failures, "semantic_requests": session._semantic.coordinator.counts["requests"],
            "observation_elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            "observation_window_ms": observed_for * 1000,
            "input_schedule": [{"at_ms": offset * 1000, "cumulative_words": count}
                               for offset, count in schedule],
        }
        for key in ("semantic_buffer_wait_ms", "semantic_selection_ms", "semantic_translation_ms", "semantic_model_ms"):
            if key in first_result:
                report[key] = round(first_result[key], 1)
        return report
    finally:
        # Do not finish()/flush() before observing normal scheduled results.
        # Stop leftover synthetic work before the next sample shares the model.
        await session.close()


async def main(args):
    config = tomllib.loads((ROOT / "server.toml").read_text())
    if args.run:
        reject_active_audio_clients(config["network"]["port"])
    provider_module, pipeline, audio, metadata = install(config)
    llm = config["lmstudio"]
    inactivity_s = llm.get("semantic_inactivity_flush_s", 0)
    if (not inactivity_s and any("residual" in name for name in args.case or SCENARIOS)):
        raise ValueError("Residual cases require semantic_inactivity_flush_s in server.toml")
    import httpx
    inference_calls = 0

    async def bounded_request(request):
        nonlocal inference_calls
        if inference_calls >= args.max_inference_requests:
            raise RuntimeError("Synthetic benchmark inference limit reached before transport")
        inference_calls += 1

    print(json.dumps({"kind": "translation_latency.start", "mock": args.mock,
                      "scope": "synthetic_asr_to_server_events_not_windows_rendering",
                      "manual_hard_flush_used": False, "synthetic_measurement_only": True,
                      "has_latency_sla": False, **metadata}), flush=True)
    reports = []
    async with httpx.AsyncClient(timeout=30, trust_env=False,
            transport=httpx.MockTransport(mock_response) if args.mock else None,
            event_hooks={"request": [bounded_request]}) as client:
        provider = provider_module.OrchestratedTranslationProvider(
            client, llm["base_url"], llm["orchestrator_model_id"],
            tuple(llm.get("translation_model_ids") or [llm["model_id"]]),
            translation_concurrency=llm.get("translation_workers", 4))
        try:
            for name in args.case or SCENARIOS:
                if inference_calls >= args.max_inference_requests:
                    break
                before = inference_calls
                report = await sample(name, provider, pipeline, audio,
                                      observation_s=args.observation_seconds,
                                      inactivity_s=inactivity_s)
                report.update(kind="translation_latency.sample", mock=args.mock,
                              inference_calls=inference_calls - before)
                reports.append(report)
                print(json.dumps(report), flush=True)
        finally:
            await provider.aclose()
    ages = [report["first_word_preview_to_translation_ms"] for report in reports
            if report["first_word_preview_to_translation_ms"] is not None]
    complete = len(reports) == len(args.case or SCENARIOS)
    print(json.dumps({"kind": "translation_latency.summary", "mock": args.mock,
                      "samples": len(reports), "all_samples_ran": complete,
                      "all_semantic_expectations_passed": complete and all(item["ok"] for item in reports),
                      "translated_samples": len(ages),
                      "max_first_word_age_ms": max(ages) if ages else None,
                      "inference_calls": inference_calls,
                      "claim": "bounded_synthetic_measurement_not_a_latency_guarantee"}), flush=True)
    return 0 if complete and all(item["ok"] for item in reports) else 1


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Send bounded requests to the configured local models")
    mode.add_argument("--mock", action="store_true", help="Validate the real scheduling harness with fixture HTTP")
    parser.add_argument("--case", action="append", choices=SCENARIOS)
    parser.add_argument("--observation-seconds", type=float, default=6)
    parser.add_argument("--max-inference-requests", type=int, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.observation_seconds) or not 1 <= args.observation_seconds <= 30:
        parser.error("observation-seconds must be in [1, 30]")
    if not 1 <= args.max_inference_requests <= 100:
        parser.error("max-inference-requests must be in [1, 100]")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(arguments())))
