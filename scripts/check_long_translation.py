"""Opt-in synthetic ASR-to-caption smoke test against the local loaded models.

No microphone, private transcript, client certificate or real ASR is used.
Run while the audio server has no active clients: this shares its local GPU.
This is an explicit-session-drain check, not a natural inactivity benchmark.
With the configured residual profile, finish() must publish unfinished source
too; that is not a claim of semantic completion. Use --mock for no model calls.
"""

import argparse
import asyncio
import json
from pathlib import Path
import re
import runpy
import time
import tomllib


ROOT = Path(__file__).resolve().parents[1]
LONG_SOURCE = (
    "Although the engineering team initially expected the migration to take only a few hours "
    "and had already prepared a detailed announcement for all of our customers, we decided "
    "to postpone the public launch until every customer record had been checked against "
    "the original database and all of the remaining discrepancies had been resolved by the "
    "people responsible for those accounts, because preserving the accuracy and completeness "
    "of the information was ultimately more important than meeting the deadline that we had "
    "announced at the beginning of the project."
)


def support(name):
    return runpy.run_path(str(ROOT / "scripts" / name))


async def main(*, mock=False):
    config = tomllib.loads((ROOT / "server.toml").read_text())
    llm = config["lmstudio"]
    if llm["base_url"] != "http://127.0.0.1:1234" or not llm.get("semantic_quality_first"):
        raise ValueError("This check requires the local quality-first configuration")
    # One installer path keeps both checks aligned with configured prompt,
    # pacing, residual publication and normal ASR inactivity finalization.
    benchmark = support("check_translation_latency.py")
    if not mock:
        benchmark["reject_active_audio_clients"](config["network"]["port"])
    provider_module, pipeline, audio, metadata = benchmark["install"](config)
    residual_enabled = bool(llm.get("semantic_inactivity_flush_s", 0))
    from myvote_engine.asr import ASRHypothesis, TimedWord
    from myvote_engine.speaker_captions import SpeakerCaptionReducer
    from myvote_engine.speakers import Assignment
    import httpx

    print(json.dumps({"kind": "long_translation.start", "mock": mock,
                      "scope": "explicit_session_drain", **metadata}), flush=True)
    async with httpx.AsyncClient(timeout=30, trust_env=False,
            transport=httpx.MockTransport(benchmark["mock_response"]) if mock else None) as client:
        provider = provider_module.OrchestratedTranslationProvider(
            client, llm["base_url"], llm["orchestrator_model_id"],
            tuple(llm.get("translation_model_ids") or [llm["model_id"]]),
            translation_concurrency=llm.get("translation_workers", 4))
        try:
            for name, source, should_translate in (
                ("complete_long", LONG_SOURCE, True),
                ("unfinished_long", LONG_SOURCE.split("because", 1)[0] + "because", False),
            ):
                words = tuple(TimedWord(piece, index * 100_000_000, (index + 1) * 100_000_000)
                              for index, piece in enumerate(re.findall(r"\S+\s*", source)))
                end = len(words) * 100_000_000

                class SyntheticASR:
                    def transcribe_pcm(self, samples, *, sample_rate, window_start_ns):
                        return ASRHypothesis(0, end, words, "en")

                events = []

                async def sink(event):
                    events.append(event)

                session = pipeline.StreamingSession(name, SyntheticASR(), provider, sink=sink,
                    config=pipeline.PipelineConfig(source_language="en", target_language="ko", semantic_translation=True))
                mapper = session._speaker_mapper = SpeakerCaptionReducer(session.store)
                mapper.observe(Assignment("known", "track", 0, end, "existing", speaker_id="A"), capture_epoch="epoch")
                mapper.observe(Assignment("uncertain", "track", words[25].start_time_ns, words[40].end_time_ns,
                    "unknown", reason="insufficient_clean_speech"), capture_epoch="epoch")
                started = time.monotonic()
                try:
                    await session._process_window(audio.SpeechWindow(
                        name, "track", "epoch", 0, end, (.1, .2), True, "endpoint", 0))
                    await session.finish(timeout_s=30)
                    captions = [event for event in events if event.kind == "caption.source"]
                    completed = [event for event in events if event.kind == "translation.completed"]
                    failed = [event.data.get("error") for event in events if event.kind == "translation.failed"]
                    preserved = len(captions) == 1 and captions[0].data["text"] == source
                    unnamed = bool(captions) and session.store.get_segment(captions[0].segment_id).speaker_id is None
                    expect_translation = should_translate or residual_enabled
                    expected = (len(completed) == 1 and not failed if expect_translation
                                else not completed and failed == ["semantic_incomplete_source"])
                    ok = bool(expected and preserved and unnamed and session.counts["asr_errors"] == 0)
                    print(json.dumps({"case": name, "ok": ok, "source_words": len(words),
                        "source_chars": len(source), "whole_source_preserved": preserved,
                        "speaker_unassigned": unnamed, "completed": len(completed), "failures": failed,
                        "mock": mock, "semantic_source_complete": should_translate,
                        "expected_translation_after_finish": expect_translation,
                        "residual_flush_enabled": residual_enabled,
                        "measurement": "explicit_finish_not_natural_inactivity",
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                        "scope": "synthetic_asr_mock_models" if mock else
                                 "synthetic_asr_real_models_not_windows_e2e"}), flush=True)
                    if not ok:
                        raise RuntimeError("Long-source smoke test failed: " + name)
                finally:
                    await session.close()
        finally:
            await provider.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Use fixture HTTP instead of the local models")
    asyncio.run(main(mock=parser.parse_args().mock))
