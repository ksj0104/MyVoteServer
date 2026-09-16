"""Stream WAV or Windows loopback audio to a selected mTLS inference gateway.

Source production, network sending, and result consumption run independently.
An exhausted client queue fails visibly instead of growing the live delay without
limit. Partial captions and the event journal are saved after errors or cancellation.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import uuid
import wave

from .audio import AudioFrame, iter_wav_frames
from .capture import CaptureFormat, CaptureStopped, NativeCaptureProcess
from .captions import export_json, export_srt, export_webvtt
from .client_latency import ClientLatencyTracker
from .dsp import NativePcmChunk, StreamingNormalizer
from .gateway_client import GatewayClient, GatewayProtocolError
from .remote_captions import RemoteCaptionProjection


@dataclass(frozen=True)
class InputGap:
    track_id: str
    reason: str
    next_start_ns: int | None


async def wav_inputs(path: Path, *, origin_ns: int | None = None):
    origin = time.perf_counter_ns() if origin_ns is None else origin_ns
    for frame in iter_wav_frames(path):
        # Event-loop timers may wake early at their clock's resolution. The
        # recorded source and receipt clock is perf_counter_ns, so await again
        # until that same clock reaches the frame end; never send future PCM.
        while (delay := (origin + frame.end_time_ns - time.perf_counter_ns()) / 1e9) > 0:
            await asyncio.sleep(delay)
        yield frame


async def _capture_records(source, stop_event):
    """Request the owned helper's terminal, continuing to consume its tail."""
    queue = asyncio.Queue(maxsize=1)

    async def read_records():
        # Keep the helper's asyncio.timeout context on one long-lived task.
        # A new task per anext() would orphan that context's cancellation target.
        try:
            async with aclosing(source.records()) as records:
                async for record in records:
                    await queue.put((True, record))
        except Exception as exc:
            await queue.put((False, exc))
        else:
            await queue.put((False, None))

    reader = asyncio.create_task(read_records())
    pending = None
    stopping = asyncio.create_task(stop_event.wait()) if stop_event is not None else None
    deadline = None
    try:
        while True:
            pending = asyncio.create_task(queue.get())
            waiting = {pending} | ({stopping} if stopping is not None else set())
            timeout = None if deadline is None else max(0, deadline - asyncio.get_running_loop().time())
            done, _ = await asyncio.wait(waiting, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise TimeoutError("capture_stop_timeout")
            if stopping is not None and stopping in done:
                await source.request_stop()
                stopping = None
                deadline = asyncio.get_running_loop().time() + 5
            if not pending.done():
                remaining = max(0, deadline - asyncio.get_running_loop().time())
                done, _ = await asyncio.wait({pending}, timeout=remaining)
                if not done:
                    raise TimeoutError("capture_stop_timeout")
            success, record = pending.result()
            pending = None
            if not success:
                if record is not None:
                    raise record
                return
            yield record
    finally:
        for task in (pending, stopping, reader):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (pending, stopping, reader) if task is not None), return_exceptions=True)


async def _until_stopped(source, stop_event):
    """Cooperatively close an injected/WAV source; already queued frames drain."""
    if stop_event is None:
        async for item in source:
            yield item
        return
    queue = asyncio.Queue(maxsize=1)

    async def read_source():
        try:
            async with aclosing(source):
                async for item in source:
                    await queue.put((True, item))
        except Exception as exc:
            await queue.put((False, exc))
        else:
            await queue.put((False, None))

    reader = asyncio.create_task(read_source())
    stop = asyncio.create_task(stop_event.wait())
    pending = None
    try:
        while not stop_event.is_set():
            pending = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({pending, stop}, return_when=asyncio.FIRST_COMPLETED)
            if pending in done:
                success, item = pending.result()
                pending = None
                if not success:
                    if item is not None:
                        raise item
                    return
                yield item
            if stop in done:
                return
    finally:
        for task in (pending, stop, reader):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (pending, stop, reader) if task is not None), return_exceptions=True)


async def native_inputs(helper: Path, seconds: float, device_id: str | None, emit, *, stop_event=None):
    normalizer = None
    inflight = None

    def items(batch):
        events = sorted(batch.events, key=lambda event: event.before_frame_index)
        cursor = 0
        for index in range(len(batch.frames) + 1):
            while cursor < len(events) and events[cursor].before_frame_index == index:
                event = events[cursor]
                emit("normalization", asdict(event))
                if event.kind == "audio.gap":
                    next_start = batch.frames[index].start_time_ns if index < len(batch.frames) else None
                    yield InputGap(normalizer.track_id, event.reason, next_start)
                cursor += 1
            if index < len(batch.frames):
                frame = batch.frames[index]
                yield frame

    try:
        async with NativeCaptureProcess(helper, duration_s=seconds, device_id=device_id) as source, \
                aclosing(_capture_records(source, stop_event)) as records:
            async for record in records:
                if isinstance(record, CaptureFormat):
                    emit("capture.format", asdict(record))
                    normalizer = StreamingNormalizer(
                        record.pcm, record.track_id, record.capture_epoch, record.qpc_origin_100ns)
                elif isinstance(record, NativePcmChunk):
                    if normalizer is None:
                        raise ValueError("Missing native format")
                    inflight = asyncio.create_task(asyncio.to_thread(normalizer.push, record))
                    batch = await asyncio.shield(inflight)
                    inflight = None
                    for item in items(batch):
                        yield item
                elif isinstance(record, CaptureStopped):
                    emit("capture.stopped", asdict(record))
        if normalizer is not None:
            inflight = asyncio.create_task(asyncio.to_thread(normalizer.flush))
            batch = await asyncio.shield(inflight)
            inflight = None
            for item in items(batch):
                yield item
    finally:
        # An aborted live source may discard unsent tail, but must not run flush
        # concurrently with an already executing native resampling call.
        if inflight is not None:
            try:
                await asyncio.shield(inflight)
            except Exception as exc:
                emit("capture.pending_failed", {"error": type(exc).__name__})
        if normalizer is not None:
            await asyncio.to_thread(normalizer.flush)


async def run_live(args, *, input_factory=None, observer=None, stop_event=None, on_projection=None) -> dict:
    """Run the actual client; optional observers only enqueue work, never block.

    ``on_projection(projection)`` shares the exact reducer used for final saves.
    ``observer(kind, data, segment_id)`` receives accepted server events and local
    status events on this event loop. It must not perform blocking output/model
    work. Observer failures stop the session with its accepted captions saved.
    ``stop_event`` requests capture stop, queued-frame drain and RPC finish.
    """
    if not 1 <= args.queue_frames <= 128:
        raise ValueError("Client queue must hold 1-128 frames")
    if not math.isfinite(args.seconds) or not .1 <= args.seconds <= 3600:
        raise ValueError("Capture duration must be in [0.1, 3600]")
    if not math.isfinite(args.drain_timeout_s) or not 1 <= args.drain_timeout_s <= 120:
        raise ValueError("Drain timeout must be in [1, 120]")
    record_words = getattr(args, "record_word_timings", False)
    if type(record_words) is not bool:
        raise ValueError("record_word_timings must be a boolean")
    capabilities = (("caption_groups_v1", "caption_refinement_v1", "semantic_translation_v1", "caption_words_v1") if record_words
                    else ("caption_groups_v1", "caption_refinement_v1", "semantic_translation_v1"))
    if record_words and any((args.output_dir / name).exists() for name in
                            ("events.jsonl", "session.json", "summary.json", "captions.srt", "captions.vtt")):
        raise FileExistsError("Use a new output directory for word timing evidence")
    duration = args.seconds
    if args.input is not None and input_factory is None:
        with wave.open(str(args.input), "rb") as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (16000, 1, 2, "NONE"):
                raise ValueError("WAV source requires PCM16 mono 16 kHz")
            duration = source.getnframes() / 16000
        if not 0 < duration <= 3600:
            raise ValueError("Use a nonempty WAV of at most one hour")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_id = args.session_id or "session-" + uuid.uuid4().hex
    projection = RemoteCaptionProjection(session_id, target_language=args.target_language,
                                         capabilities=capabilities)
    source_mode = "injected_test_source" if input_factory else "paced_wav" if args.input else "native_wasapi_loopback"
    latency = ClientLatencyTracker(source_mode)
    if on_projection is not None:
        on_projection(projection)
    queue = asyncio.Queue(maxsize=args.queue_frames)
    end = object()
    tasks = set()
    failure = None
    source_failure = None
    cancelled = None
    counts = {"produced_frames": 0, "sent_frames": 0, "sent_gaps": 0, "server_events": 0,
              "sent_heartbeats": 0}
    started = time.monotonic()
    server_mode = "not_started"
    speaker_status = "not_configured"
    observer_failure = None

    def observe(kind, data, segment_id=None):
        nonlocal observer_failure
        if observer is not None and observer_failure is None:
            try:
                observer(kind, data, segment_id)
            except Exception as exc:
                observer_failure = exc
                raise

    with (args.output_dir / "events.jsonl").open("w", encoding="utf-8") as journal:
        def local(kind, data):
            if kind == "capture.format":
                latency.set_native_origin(data["qpc_origin_100ns"], track_id=data["track_id"])
            elif kind.endswith(".failed"):
                latency.note_error()
            journal.write(json.dumps({"origin": "client", "kind": kind, "data": data},
                                     ensure_ascii=False) + "\n")
            journal.flush()
            observe(kind, data)

        def result(message):
            nonlocal server_mode, speaker_status
            received_ns = latency.now_ns()
            accepted = projection.apply_message(message)
            data = json.loads(message.data_json)
            counts["server_events"] += 1
            if message.kind == "session.started":
                server_mode = data.get("engine_mode", "unspecified")
            if message.kind in ("session.started", "session.completed"):
                speaker_status = data.get("speaker_status", speaker_status)
            journal.write(json.dumps({"origin": "server", "session_id": message.session_id,
                "event_sequence": message.event_sequence, "kind": message.kind,
                "segment_id": message.segment_id, "server_elapsed_ms": message.server_elapsed_ms,
                "data": data}, ensure_ascii=False) + "\n")
            journal.flush()
            if accepted:
                if message.kind.endswith(".failed"):
                    latency.note_error()
                segment = projection.store.get_segment(message.segment_id) if message.segment_id else None
                metric_kinds = ()
                text = None
                if segment is not None and message.kind == "caption.source":
                    metric_kinds = ("source.first", "source.stable") if segment.source_state == "stable" else ("source.first",)
                    text = segment.source_text
                elif segment is not None and message.kind in ("translation.preview", "translation.completed"):
                    translated = segment.translation
                    if (translated is not None and translated.source_revision == segment.source_revision
                            and translated.target_language == segment.target_language):
                        metric_kinds = ("translation.first", "translation.completed") if message.kind == "translation.completed" else ("translation.first",)
                        text = translated.text
                for kind in metric_kinds:
                    sample = latency.observe_caption(segment.segment_id, segment.source_revision,
                        segment.start_ns, segment.end_ns, kind, text, track_id=segment.track_id,
                        capture_epoch=data.get("capture_epoch"), now_ns=received_ns)
                    if sample is not None:
                        journal.write(json.dumps({"origin": "client", "kind": "client.latency", "data": sample}) + "\n")
                        journal.flush()
                        observe("client.latency", sample, segment.segment_id)
                observe(message.kind, data, message.segment_id or None)
            if accepted and getattr(args, "show_text", False) and message.kind in ("caption.source", "translation.completed"):
                print(data.get("text", ""), flush=True)

        async def produce():
            nonlocal source_failure
            if input_factory:
                source = input_factory()
            elif args.input:
                origin = time.perf_counter_ns()
                latency.set_wav_origin(origin)
                source = wav_inputs(args.input, origin_ns=origin)
            else:
                source = native_inputs(args.helper, args.seconds, args.device_id, local, stop_event=stop_event)
            try:
                async with aclosing(source):
                    sequence = (_until_stopped(source, stop_event) if input_factory or args.input is not None else source)
                    async with aclosing(sequence):
                        async for item in sequence:
                            try:
                                queue.put_nowait(item)
                            except asyncio.QueueFull as exc:
                                raise RuntimeError("Client audio queue is full; live transmission stopped") from exc
                            if isinstance(item, AudioFrame):
                                counts["produced_frames"] += 1
            except Exception as exc:
                source_failure = {"type": type(exc).__name__, "message": str(exc)}
                local("source.failed", source_failure)
            # Bound waiting for the sender; no capture resources remain open here.
            async with asyncio.timeout(args.drain_timeout_s):
                await queue.put(end)

        try:
            async with GatewayClient(
                args.target, ca_cert=args.ca_cert, client_cert=args.client_cert,
                client_key=args.client_key, session_id=session_id,
                connect_timeout_s=args.connect_timeout_s,
                rpc_timeout_s=duration + args.drain_timeout_s + args.connect_timeout_s,
            ) as client:
                options = {"capabilities": capabilities}
                result(await client.open(source_language=args.source_language or "",
                                         target_language=args.target_language,
                                         translation_budget_ms=args.deadline_ms, **options))
                if record_words and "caption_words_v1" not in projection.capabilities:
                    local("word_trace.unavailable", {"reason": "server_capability_not_negotiated"})
                    raise GatewayProtocolError("Server does not support requested word timing records")

                async def send():
                    heartbeat_interval = getattr(client, "heartbeat_interval_s", None)
                    while True:
                        if heartbeat_interval is None:
                            item = await queue.get()
                        else:
                            # All writes remain in this sender; no timer task
                            # races audio, a gap, FinishSession, or half-close.
                            try:
                                item = await asyncio.wait_for(queue.get(), heartbeat_interval)
                            except TimeoutError:
                                await client.send_heartbeat()
                                counts["sent_heartbeats"] += 1
                                continue
                        if item is end:
                            await client.finish_sending()
                            return
                        if isinstance(item, InputGap):
                            latency.note_gap(item.track_id, item.reason)
                            await client.send_gap(item.track_id, item.reason, item.next_start_ns)
                            counts["sent_gaps"] += 1
                        else:
                            latency.observe_frame(item)
                            await client.send_frame(item)
                            counts["sent_frames"] += 1

                async def receive():
                    async for message in client.responses():
                        result(message)
                        if message.kind == "session.failed":
                            raise GatewayProtocolError("Server ended the session with failure")

                tasks = {asyncio.create_task(produce()), asyncio.create_task(send()),
                         asyncio.create_task(receive())}
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                for task in done:
                    if not task.cancelled() and task.exception() is not None:
                        raise task.exception()
                if pending:
                    await asyncio.gather(*pending)
        except asyncio.CancelledError as exc:
            cancelled = exc
            failure = {"type": "CancelledError", "message": "Live session cancelled"}
            local("session.cancelled", failure)
        except Exception as exc:
            # grpc debug diagnostics are not transcript content. Store a category,
            # leaving source/result details in the already accepted event journal.
            failure = {"type": type(exc).__name__, "message": "Live connection or protocol failed"}
            local("session.failed", failure)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    snapshot = projection.snapshot()
    (args.output_dir / "captions.srt").write_text(export_srt(snapshot), encoding="utf-8")
    (args.output_dir / "captions.vtt").write_text(export_webvtt(snapshot), encoding="utf-8")
    (args.output_dir / "session.json").write_text(export_json(snapshot), encoding="utf-8")
    terminal = projection.terminal
    terminal_counts = (terminal or {}).get("counts", {})
    unsuccessful = any(terminal_counts.get(key, 0) for key in
                       ("asr_errors", "asr_skipped", "translation_failed", "audio_gaps", "speaker_errors", "speaker_skipped"))
    summary = {
        "schema": "myvote.live_session", "schema_version": 1, "session_id": session_id,
        "status": "failed" if failure or source_failure or unsuccessful or projection.terminal_kind != "session.completed" else "completed",
        "source_mode": source_mode,
        "word_timings_requested": record_words,
        "negotiated_capabilities": sorted(projection.capabilities),
        "server_engine_mode": server_mode, "speaker_status": speaker_status,
        "counts": counts, "server_terminal": terminal, "failure": failure, "source_failure": source_failure,
        "wall_seconds": time.monotonic() - started,
        "client_latency": latency.summary(),
        "limitations": [
            "Server-local durations are not Windows audio-to-display latency",
            "This client measures receipt diagnostics; WPF rendering is logged separately",
            "No automatic reconnect or model-quality evaluation",
            "Aborted live source may leave unsent audio; accepted captions remain in the journal",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if cancelled is not None:
        raise cancelled
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--helper", type=Path)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--device-id")
    parser.add_argument("--target", required=True)
    parser.add_argument("--ca-cert", type=Path, required=True)
    parser.add_argument("--client-cert", type=Path, required=True)
    parser.add_argument("--client-key", type=Path, required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--source-language")
    parser.add_argument("--target-language", default="ko")
    parser.add_argument("--deadline-ms", type=float, default=2500)
    parser.add_argument("--queue-frames", type=int, default=32)
    parser.add_argument("--connect-timeout-s", type=float, default=10)
    parser.add_argument("--drain-timeout-s", type=float, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/live"))
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--record-word-timings", action="store_true",
                        help="Record exact ASR word timing metadata in events.jsonl; requires a supporting server")
    args = parser.parse_args()
    try:
        result = asyncio.run(run_live(args))
    except Exception as exc:
        parser.exit(2, f"Live setup failed ({type(exc).__name__})\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
