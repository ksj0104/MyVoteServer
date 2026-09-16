"""Replay PCM16/16kHz/mono WAV through actual VAD, ASR and translation engines.

Default mode supplies frames at their audio playback times. Reported audio-to-result
latency excludes Windows capture, LAN and UI rendering. Fast mode has no valid live
latency metric. Models are prepared separately and loaded only from local paths.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import sys
import time
import wave

import httpx

from .asr import MlxWhisperWindowTranscriber
from .audio import iter_wav_frames, SileroOnnxVad, SpeechWindowBuilder
from .benchmark import percentile
from .captions import export_srt, export_webvtt, export_json
from .pipeline import PipelineConfig, PipelineEvent, StreamingSession
from .translation import LlamaCppProvider, OllamaProvider, LMStudioProvider, TranslationRequest


async def run_replay(args: argparse.Namespace, *, transcriber=None, vad=None, provider=None, speaker_analyzer=None) -> dict:
    """Dependency injection is for tests; CLI never offers fabricated engines."""
    if not math.isfinite(args.drain_timeout_s) or args.drain_timeout_s <= 0:
        raise ValueError("Drain timeout must be finite and positive")
    builder = SpeechWindowBuilder(step_s=args.asr_step_s)
    config = PipelineConfig(
        source_language=args.source_language, target_language=args.target_language,
        clause_target_s=args.clause_s, translation_budget_ms=args.deadline_ms,
    )
    injected = any(engine is not None for engine in (transcriber, vad, provider, speaker_analyzer))
    segmentation_model = getattr(args, "speaker_segmentation_model", None)
    embedding_model = getattr(args, "speaker_embedding_model", None)
    if bool(segmentation_model) != bool(embedding_model):
        raise ValueError("Provide both speaker segmentation and embedding model paths")
    profile_path = getattr(args, "speaker_profile", None)
    if profile_path and not segmentation_model:
        raise ValueError("A speaker profile requires both speaker model paths")
    if segmentation_model and speaker_analyzer is None:
        from .speaker_stream import LocalSpeakerAnalyzer
        speaker_analyzer = LocalSpeakerAnalyzer.from_local_models(segmentation_model, embedding_model, profile_path=profile_path)
    with wave.open(str(args.input), "rb") as audio:
        if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("Replay requires uncompressed PCM16 mono 16 kHz WAV")
        duration_s = audio.getnframes() / 16000
        if not audio.getnframes():
            raise ValueError("Replay input is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_latencies: dict[str, float] = {}
    complete_latencies: list[float] = []
    warmup = {"enabled": not args.no_warmup, "vad": "not_started",
              "asr": "not_started", "translation": "not_started"}
    async with AsyncExitStack() as stack:
        if provider is None:
            client = await stack.enter_async_context(httpx.AsyncClient(
                timeout=httpx.Timeout(None, connect=3), follow_redirects=False, trust_env=False,
            ))
            provider_type = {"llamacpp": LlamaCppProvider, "ollama": OllamaProvider, "lmstudio": LMStudioProvider}[args.backend]
            provider = provider_type(client, args.endpoint, args.model, max_tokens=args.max_tokens)
        transcriber = transcriber or MlxWhisperWindowTranscriber(
            args.asr_model, language=args.source_language,
        )
        vad = vad or SileroOnnxVad(args.vad_model)
        await asyncio.to_thread(vad.preload)
        warmup["vad"] = "loaded"
        if not args.no_warmup:
            await asyncio.to_thread(transcriber.transcribe_pcm, (0.0,) * 16000,
                                    sample_rate=16000, window_start_ns=0)
            warmup["asr"] = "completed"
            try:
                completed = False
                async with asyncio.timeout(60):
                    async for chunk in provider.stream(TranslationRequest(
                        "warmup", 1, "The meeting will start soon.", "en",
                        args.target_language, budget_ms=60000,
                    )):
                        completed = completed or chunk.completed
                warmup["translation"] = "completed" if completed else "incomplete"
            except Exception as exc:
                # Translation readiness must not prevent source captions.
                warmup["translation"] = type(exc).__name__
        else:
            warmup["asr"] = warmup["translation"] = "skipped"
        vad.reset()
        with (args.output_dir / "events.jsonl").open("w", encoding="utf-8") as events:
            async def sink(event: PipelineEvent):
                events.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
                events.flush()
                latency = event.data.get("audio_to_event_ms")
                if latency is None or not event.segment_id:
                    return
                if event.kind == "translation.preview" and event.data.get("text", "").strip():
                    first_latencies.setdefault(event.segment_id, latency)
                if event.kind == "translation.completed":
                    complete_latencies.append(latency)

            origin = time.monotonic()
            session = StreamingSession(
                "wav-replay", transcriber, provider,
                builder=builder, config=config, sink=sink,
                source_origin_monotonic=None if args.fast else origin,
                speaker_analyzer=speaker_analyzer,
            )
            failure = None
            cleanup_failure = None
            try:
                for frame in iter_wav_frames(args.input):
                    end_ns = frame.start_time_ns + len(frame.samples) * 1_000_000_000 // 16000
                    if not args.fast:
                        delay = origin + end_ns / 1e9 - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                    probability = await asyncio.to_thread(vad.probability, frame)
                    await session.feed(frame, probability)
                await session.finish(timeout_s=args.drain_timeout_s)
            except Exception as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)}
            finally:
                try:
                    await session.close()
                except Exception as exc:
                    cleanup_failure = {"type": type(exc).__name__, "message": str(exc)}
            wall_s = time.monotonic() - origin
        snapshot = session.store.snapshot()
        (args.output_dir / "captions.srt").write_text(
            export_srt(snapshot, mode="bilingual"), encoding="utf-8",
        )
        (args.output_dir / "captions.vtt").write_text(
            export_webvtt(snapshot, mode="bilingual"), encoding="utf-8",
        )
        (args.output_dir / "session.json").write_text(export_json(snapshot), encoding="utf-8")
        total = session.counts["clauses"]
        summary = {
            "scope": "fast_processing_only" if args.fast else "paced_wav_audio_to_translation_result",
            "engine_mode": "injected_test_engines" if injected else "configured_local_engines",
            "input": str(args.input), "audio_duration_s": duration_s, "wall_duration_s": wall_s,
            "platform": platform.platform(), "machine": platform.machine(),
            "backend": args.backend, "model": args.model, "asr_model": str(args.asr_model),
            "vad_model": str(args.vad_model), "warmup": warmup,
            "source_language": args.source_language, "target_language": args.target_language,
            "config": asdict(session.config), "counts": dict(session.counts),
            "completion_rate": session.counts["translation_completed"] / total if total else None,
            "first_nonwhite_text_p95_ms": percentile(list(first_latencies.values()), .95),
            "completed_only_p50_ms": percentile(complete_latencies, .50),
            "completed_only_p95_ms": percentile(complete_latencies, .95),
            "completed_latency_sample_count": len(complete_latencies), "failure": failure,
            "cleanup_failure": cleanup_failure,
            "speaker_status": session.speaker_status,
            "limitations": [
                "WAV replay excludes live Windows capture, network and subtitle rendering",
                "First non-whitespace text may not yet be a readable clause",
                "Accuracy, language coverage, source omissions and speaker quality need labeled audio",
                "CPU/GPU memory, model hashes and runtime versions are not automatically measured",
                "Cancelling the ASR Python await does not terminate native model inference",
                "asyncio.run may still wait for native ASR threads during interpreter shutdown",
            ],
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vad-model", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--backend", choices=("llamacpp", "ollama", "lmstudio"), default="llamacpp")
    parser.add_argument("--endpoint", help="Server origin; default port 1234 for LM Studio, otherwise 8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-language")
    parser.add_argument("--target-language", default="ko")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/replay"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--deadline-ms", type=float, default=2500)
    parser.add_argument("--asr-step-s", type=float, default=.5)
    parser.add_argument("--clause-s", type=float, default=2)
    parser.add_argument("--drain-timeout-s", type=float, default=60)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--speaker-segmentation-model", type=Path)
    parser.add_argument("--speaker-embedding-model", type=Path)
    parser.add_argument("--speaker-profile", type=Path)
    args = parser.parse_args()
    if args.endpoint is None:
        args.endpoint = "http://127.0.0.1:1234" if args.backend == "lmstudio" else "http://127.0.0.1:8080"
    try:
        result = asyncio.run(run_replay(args))
    except Exception as exc:
        print(f"Replay setup failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failure"] or result["cleanup_failure"] or any(result["counts"].get(key, 0) for key in
                                ("asr_errors", "asr_skipped", "translation_failed", "audio_gaps", "speaker_errors", "speaker_skipped")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
