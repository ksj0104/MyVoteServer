"""One-session mTLS gRPC gateway for actual local engines.

Audio timestamps stay on the client capture timeline. Server elapsed values are
local durations, never client-to-caption latency. Native inference is fenced:
cancelling a Python await cannot make a still-running model safe to reuse.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import struct
import threading
import time
from typing import Any, Awaitable, Callable

import grpc

from .audio import AudioFrame, SAMPLE_NS
from .pipeline import PipelineConfig, PipelineEvent, StreamingSession
from .ingress_timing import INGRESS_SCOPE, IngressTimeline
from .rpc import session_pb2 as pb, session_pb2_grpc as rpc


@dataclass
class EngineBundle:
    transcriber: Any
    provider: Any
    vad: Any
    aclose: Callable[[], Awaitable[None]] | None = None
    engine_mode: str = "injected_test_engines"
    speaker_status: str = "not_configured"
    speaker_analyzer: Any = None


@dataclass(frozen=True)
class GatewayConfig:
    max_receive_queue: int = 64
    max_event_queue: int = 128
    max_event_bytes: int = 65536
    open_timeout_s: float = 10.
    idle_timeout_s: float = 30.
    finish_timeout_s: float = 30.
    result_write_timeout_s: float = 5.
    max_session_s: float = 3600.
    max_messages: int = 200000
    max_ingress_frames: int = 8192

    @property
    def heartbeat_interval_ms(self) -> int | None:
        # Leave at least two intervals of scheduling/network margin. Extremely
        # short test timeouts cannot advertise a safe integral millisecond value.
        if self.idle_timeout_s < .003:
            return None
        return max(1, math.floor(min(self.idle_timeout_s, 30.) * 1000 / 3))

    def __post_init__(self):
        for value in (self.max_receive_queue, self.max_event_queue, self.max_event_bytes, self.max_messages, self.max_ingress_frames):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Gateway capacities must be positive integers")
        if not 256 <= self.max_event_bytes <= 1_048_576:
            raise ValueError("Event payload limit must be between 256 bytes and 1 MiB")
        if self.max_ingress_frames > 200000:
            raise ValueError("Ingress history exceeds 200000 frame metadata records")
        for value in (self.open_timeout_s, self.idle_timeout_s, self.finish_timeout_s,
                      self.result_write_timeout_s, self.max_session_s):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Gateway timeouts must be finite and positive")


class GatewayFailure(Exception):
    def __init__(self, code: grpc.StatusCode, category: str):
        super().__init__(category)
        self.code, self.category = code, category


class NativeCallFence:
    """Close admission before retiring a session, then await its native calls.

    A default-executor wrapper that starts *after* close is rejected before it
    touches an engine. Already-running native calls hold the session lease.
    """

    def __init__(self):
        self._loop = asyncio.get_running_loop()
        self._lock = threading.Lock()
        self._open = True
        self._active = 0
        self._drained = self._loop.create_future()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def _resolve(self):
        if not self._drained.done():
            self._drained.set_result(None)

    def call(self, function, *args, **kwargs):
        with self._lock:
            if not self._open:
                raise RuntimeError("native session retired")
            self._active += 1
        try:
            return function(*args, **kwargs)
        finally:
            with self._lock:
                self._active -= 1
                drained = not self._open and self._active == 0
            if drained:
                try:
                    self._loop.call_soon_threadsafe(self._resolve)
                except RuntimeError:
                    # The process event loop has already shut down; no new RPC can reuse this lease.
                    pass

    async def run(self, function, *args, **kwargs):
        return await asyncio.to_thread(self.call, function, *args, **kwargs)

    def close(self):
        with self._lock:
            self._open = False
            drained = self._active == 0
        if drained:
            self._resolve()

    async def wait_idle(self):
        await asyncio.shield(self._drained)


class _FencedTranscriber:
    def __init__(self, target, fence: NativeCallFence):
        self.target, self.fence = target, fence

    def transcribe_pcm(self, *args, **kwargs):
        return self.fence.call(self.target.transcribe_pcm, *args, **kwargs)


class _Reblocker:
    """Combine arbitrary 1-512 sample wire frames into real 512-sample VAD input."""

    def __init__(self):
        self.samples: list[float] = []
        self.start_ns = 0
        self.track = self.epoch = ""
        self.sequence = 0

    def _take(self, size: int) -> AudioFrame:
        result = AudioFrame(self.track, self.epoch, self.sequence, self.start_ns, tuple(self.samples[:size]))
        del self.samples[:size]
        self.start_ns += size * SAMPLE_NS
        self.sequence += 1
        return result

    def push(self, frame: AudioFrame) -> list[AudioFrame]:
        if not self.samples:
            self.start_ns, self.track, self.epoch = frame.start_time_ns, frame.track_id, frame.capture_epoch
        self.samples.extend(frame.samples)
        return [self._take(512)] if len(self.samples) >= 512 else []

    def flush(self) -> list[AudioFrame]:
        return [self._take(len(self.samples))] if self.samples else []


def _identifier(value: str, field: str) -> str:
    if not value or len(value) > 128 or any(ord(char) < 32 for char in value):
        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, f"invalid_{field}")
    return value


def validate_open(message: pb.ClientMessage) -> pb.OpenSession:
    if message.WhichOneof("body") != "open":
        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "open_required_first")
    opening = message.open
    if opening.protocol_version != 1:
        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "unsupported_protocol")
    _identifier(opening.session_id, "session_id")
    for field, value, optional in (("source_language", opening.source_language, True),
                                   ("target_language", opening.target_language, False)):
        if (value or not optional) and not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,19}", value):
            raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, f"invalid_{field}")
    if not math.isfinite(opening.translation_budget_ms) or not 0 < opening.translation_budget_ms <= 120000:
        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "invalid_translation_budget")
    return opening


def decode_audio(message: pb.PcmFrame) -> AudioFrame:
    _identifier(message.track_id, "track_id")
    _identifier(message.capture_epoch, "capture_epoch")
    raw = message.pcm_f32le
    if not raw or len(raw) % 4 or len(raw) > 2048:
        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "invalid_pcm_length")
    samples = tuple(value[0] for value in struct.iter_unpack("<f", raw))
    if message.start_time_ns + len(samples) * SAMPLE_NS > (1 << 64) - 1:
        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "audio_time_overflow")
    try:
        return AudioFrame(message.track_id, message.capture_epoch, message.sequence,
                          message.start_time_ns, samples)
    except ValueError as exc:
        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "invalid_pcm_samples") from exc


_FINISH = object()


@dataclass(frozen=True)
class _ReceivedAudio:
    frame: AudioFrame
    received_at: float


class GatewayService(rpc.SpeechGatewayServicer):
    def __init__(self, engine_factory: Callable[[pb.OpenSession], EngineBundle], *,
                 config: GatewayConfig | None = None):
        self.engine_factory = engine_factory
        self.config = config or GatewayConfig()
        self._busy = False
        self._blocked_reason: str | None = None
        self._retirement: asyncio.Task | None = None
        self._fence: NativeCallFence | None = None

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def active_native_calls(self) -> int:
        return self._fence.active if self._fence else 0

    async def aclose(self, *, timeout_s: float = 5.) -> bool:
        """Wait for an already-retiring lease; stop the gRPC server first."""
        if self._retirement:
            try:
                await asyncio.wait_for(asyncio.shield(self._retirement), timeout_s)
            except TimeoutError:
                return False
        return not self._busy

    async def _retire(self, fence: NativeCallFence, bundle: EngineBundle | None):
        try:
            await fence.wait_idle()
            if bundle is not None:
                # All old native work ended, and admission remains closed.
                await asyncio.to_thread(bundle.vad.reset)
                if bundle.aclose:
                    await bundle.aclose()
        except Exception:
            self._blocked_reason = "engine_cleanup_failed"
        finally:
            self._busy = False
            self._fence = None

    async def StreamSession(self, request_iterator, context):
        auth = context.auth_context()
        if (b"ssl" not in auth.get("transport_security_type", ())
                or not (auth.get("x509_pem_cert") or context.peer_identities())):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "client_certificate_required")
        try:
            first = await asyncio.wait_for(anext(request_iterator), self.config.open_timeout_s)
            opening = validate_open(first)
        except StopAsyncIteration:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "open_required_first")
        except TimeoutError:
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "open_timeout")
        except GatewayFailure as exc:
            await context.abort(exc.code, exc.category)
        if self._blocked_reason:
            await context.abort(grpc.StatusCode.UNAVAILABLE, self._blocked_reason)
        if self._busy:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "engine_busy_or_native_call_draining")
        self._busy = True
        fence = self._fence = NativeCallFence()
        bundle: EngineBundle | None = None
        session: StreamingSession | None = None
        reader_task = worker_task = get_event = None
        started = time.monotonic()
        event_sequence = 0
        input_frames = 0
        heartbeats_received = 0
        events: asyncio.Queue[pb.ServerMessage] = asyncio.Queue(self.config.max_event_queue)
        received: asyncio.Queue[Any] = asyncio.Queue(self.config.max_receive_queue)
        ingress = IngressTimeline(self.config.max_ingress_frames)
        fatal = asyncio.get_running_loop().create_future()

        def fail(error: GatewayFailure):
            if not fatal.done():
                fatal.set_result(error)

        def message(kind, data, segment_id=""):
            try:
                raw = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            except (ValueError, TypeError) as exc:
                raise GatewayFailure(grpc.StatusCode.INTERNAL, "invalid_engine_event_data") from exc
            if len(raw.encode("utf-8")) > self.config.max_event_bytes:
                raise GatewayFailure(grpc.StatusCode.RESOURCE_EXHAUSTED, "event_payload_too_large")
            return pb.ServerMessage(session_id=opening.session_id,
                                    kind=kind, segment_id=segment_id or "", data_json=raw,
                                    server_elapsed_ms=(time.monotonic() - started) * 1000)

        async def send(result):
            nonlocal event_sequence
            # Number actual sends, including a fatal event that discards queued results.
            event_sequence += 1
            result.event_sequence = event_sequence
            try:
                await asyncio.wait_for(context.write(result), self.config.result_write_timeout_s)
            except TimeoutError:
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "slow_result_consumer")

        async def emit(kind, data, segment_id=""):
            if fatal.done():
                return
            try:
                events.put_nowait(message(kind, data, segment_id))
            except asyncio.QueueFull:
                fail(GatewayFailure(grpc.StatusCode.RESOURCE_EXHAUSTED, "slow_result_consumer"))
            except GatewayFailure as exc:
                fail(exc)

        async def sink(event: PipelineEvent):
            await emit(event.kind, event.data, event.segment_id)

        def terminal_counts():
            return {**(session.counts if session is not None else {}),
                    "heartbeats_received": heartbeats_received}

        async def read_input():
            nonlocal heartbeats_received
            finish_seen = False
            count = 0
            try:
                while True:
                    remaining = self.config.max_session_s - (time.monotonic() - started)
                    if remaining <= 0:
                        raise GatewayFailure(grpc.StatusCode.DEADLINE_EXCEEDED, "session_duration_limit")
                    try:
                        item = await asyncio.wait_for(anext(request_iterator), min(remaining, self.config.idle_timeout_s))
                        received_at = time.monotonic()
                    except TimeoutError:
                        # Classify by the deadline we waited for: event-loop
                        # timers can fire within their clock resolution before
                        # another monotonic read crosses the exact boundary.
                        category = ("session_duration_limit" if remaining <= self.config.idle_timeout_s
                                    else "input_idle_timeout")
                        raise GatewayFailure(grpc.StatusCode.DEADLINE_EXCEEDED, category)
                    except StopAsyncIteration:
                        if not finish_seen:
                            raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "finish_required_before_half_close")
                        received.put_nowait(_FINISH)
                        return
                    count += 1
                    if count > self.config.max_messages:
                        raise GatewayFailure(grpc.StatusCode.RESOURCE_EXHAUSTED, "input_message_limit")
                    body = item.WhichOneof("body")
                    if finish_seen:
                        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "message_after_finish")
                    if body == "finish":
                        finish_seen = True
                    elif body == "heartbeat":
                        # Activity is independent of audio, VAD, ingress anchors,
                        # and the worker queue. The message/session limits above
                        # still apply, including ordering after finish.
                        heartbeats_received += 1
                    elif body == "audio":
                        received.put_nowait(_ReceivedAudio(decode_audio(item.audio), received_at))
                    elif body == "gap":
                        _identifier(item.gap.track_id, "track_id")
                        if not item.gap.reason or len(item.gap.reason) > 256:
                            raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "invalid_gap_reason")
                        received.put_nowait(item.gap)
                    else:
                        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "unexpected_client_message")
            except asyncio.QueueFull:
                fail(GatewayFailure(grpc.StatusCode.RESOURCE_EXHAUSTED, "receive_queue_full"))
            except GatewayFailure as exc:
                fail(exc)
            except asyncio.CancelledError:
                raise
            except Exception:
                fail(GatewayFailure(grpc.StatusCode.INTERNAL, "request_stream_failed"))

        async def process_input():
            nonlocal session, input_frames
            assert bundle is not None
            try:
                if hasattr(bundle.vad, "preload"):
                    await fence.run(bundle.vad.preload)
                await fence.run(bundle.vad.reset)
                session = StreamingSession(opening.session_id, _FencedTranscriber(bundle.transcriber, fence),
                    bundle.provider, sink=sink, source_origin_monotonic=None,
                    ingress_timeline=ingress,
                    speaker_analyzer=bundle.speaker_analyzer, speaker_run_native=fence.run,
                    config=PipelineConfig(source_language=opening.source_language or None,
                        target_language=opening.target_language, translation_budget_ms=opening.translation_budget_ms))
                reblock = _Reblocker()
                last_wire: AudioFrame | None = None
                explicit_gap_pending = False
                pending_gap_start: int | None = None
                active_track: str | None = None

                async def feed_frames(frames):
                    for frame in frames:
                        probability = await fence.run(bundle.vad.probability, frame)
                        await session.feed(frame, probability)

                async def gap(track, reason, next_start):
                    nonlocal reblock
                    await feed_frames(reblock.flush())
                    ingress.gap()
                    await session.audio_gap(track, reason, next_start)
                    await fence.run(bundle.vad.reset)
                    reblock = _Reblocker()

                started_data = {"protocol_version": 1, "engine_mode": bundle.engine_mode,
                    "speaker_status": session.speaker_status, "timing_scope": "server_local_processing_only",
                    "translation_budget_scope": INGRESS_SCOPE,
                    "audio_format": "float32le_mono_16000", "vad_ready": True,
                    "asr_warmup": "not_performed", "translation_warmup": "not_performed"}
                if self.config.heartbeat_interval_ms is not None:
                    started_data["heartbeat_interval_ms"] = self.config.heartbeat_interval_ms
                await emit("session.started", started_data)
                while True:
                    item = await received.get()
                    if item is _FINISH:
                        await feed_frames(reblock.flush())
                        await session.finish(timeout_s=self.config.finish_timeout_s)
                        await emit("session.completed", {"counts": terminal_counts(),
                            "input_frames": input_frames, "speaker_status": session.speaker_status,
                            "ingress_timing": ingress.summary(),
                            "timing_scope": "server_local_processing_only"})
                        return
                    if isinstance(item, pb.AudioGap):
                        if active_track is not None and item.track_id != active_track:
                            raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "track_changed_within_session")
                        active_track = item.track_id
                        next_start = item.next_start_time_ns if item.HasField("next_start_time_ns") else None
                        if last_wire and next_start is not None and next_start < last_wire.end_time_ns:
                            raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "gap_time_rewinds_audio")
                        if pending_gap_start is not None and next_start is not None and next_start < pending_gap_start:
                            raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "gap_time_rewinds_barrier")
                        await gap(item.track_id, item.reason, next_start)
                        # Keep the time barrier; the next frame starts a fresh sequence/epoch expectation.
                        explicit_gap_pending = True
                        # Unknown next start adds no invented timestamp. last_wire still
                        # protects the end of all audio accepted before the gap.
                        pending_gap_start = next_start
                        continue
                    frame = item.frame
                    input_frames += 1
                    if active_track is not None and frame.track_id != active_track:
                        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "track_changed_within_session")
                    active_track = frame.track_id
                    if pending_gap_start is not None and frame.start_time_ns < pending_gap_start:
                        raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "audio_before_gap_barrier")
                    if last_wire is not None:
                        if frame.start_time_ns < last_wire.end_time_ns:
                            raise GatewayFailure(grpc.StatusCode.INVALID_ARGUMENT, "audio_time_rewinds_or_overlaps")
                        if not explicit_gap_pending and (frame.track_id != last_wire.track_id or frame.capture_epoch != last_wire.capture_epoch
                                or frame.sequence != last_wire.sequence + 1 or frame.start_time_ns != last_wire.end_time_ns):
                            await gap(frame.track_id, "wire_audio_discontinuity", frame.start_time_ns)
                    ingress.observe(frame, item.received_at)
                    await feed_frames(reblock.push(frame))
                    last_wire = frame
                    explicit_gap_pending = False
                    pending_gap_start = None
            except GatewayFailure as exc:
                fail(exc)
            except TimeoutError:
                fail(GatewayFailure(grpc.StatusCode.DEADLINE_EXCEEDED, "engine_drain_timeout"))
            except asyncio.CancelledError:
                raise
            except Exception:
                fail(GatewayFailure(grpc.StatusCode.INTERNAL, "engine_processing_failed"))

        try:
            # Fast synchronous construction only; model I/O belongs in fenced calls.
            try:
                bundle = self.engine_factory(opening)
                if not isinstance(bundle, EngineBundle):
                    bundle = None
                    raise TypeError("engine_factory must return EngineBundle")
            except Exception:
                await send(message("session.failed", {"error": "engine_factory_failed", "counts": terminal_counts()}))
                await context.abort(grpc.StatusCode.INTERNAL, "engine_factory_failed")
            reader_task = asyncio.create_task(read_input())
            worker_task = asyncio.create_task(process_input())

            while True:
                if fatal.done():
                    error = fatal.result()
                    await send(message("session.failed", {"error": error.category, "counts": terminal_counts()}))
                    await context.abort(error.code, error.category)
                if worker_task.done():
                    if events.empty():
                        return
                    await send(events.get_nowait())
                    continue
                get_event = asyncio.create_task(events.get())
                done, _ = await asyncio.wait({get_event, fatal, worker_task}, return_when=asyncio.FIRST_COMPLETED)
                if fatal in done:
                    get_event.cancel()
                    await asyncio.gather(get_event, return_exceptions=True)
                    get_event = None
                    continue
                # It may have completed after asyncio.wait formed its done set.
                if get_event.done():
                    result = get_event.result()
                    get_event = None
                    await send(result)
                else:
                    get_event.cancel()
                    await asyncio.gather(get_event, return_exceptions=True)
                    get_event = None
        finally:
            fence.close()
            for task in (reader_task, worker_task, get_event):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (reader_task, worker_task, get_event) if task), return_exceptions=True)
            try:
                if session is not None:
                    await session.close()
                    session.builder.reset()
            except Exception:
                self._blocked_reason = "pipeline_cleanup_failed"
            finally:
                self._retirement = asyncio.create_task(self._retire(fence, bundle))


def create_secure_server(service: GatewayService, *, certificate_chain: bytes, private_key: bytes,
                         client_ca: bytes, host: str = "127.0.0.1", port: int = 50051):
    """Create, but do not start, an mTLS-only aio server. Port 0 is for tests."""
    if not all(isinstance(value, bytes) and value.strip() for value in (certificate_chain, private_key, client_ca)):
        raise ValueError("Explicit server certificate, private key, and trusted client CA are required")
    if not host or isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("Invalid bind host or port")
    credentials = grpc.ssl_server_credentials([(private_key, certificate_chain)],
                                               root_certificates=client_ca, require_client_auth=True)
    server = grpc.aio.server(maximum_concurrent_rpcs=8, options=[
        ("grpc.max_receive_message_length", 16384),
        ("grpc.max_send_message_length", service.config.max_event_bytes + 4096),
    ])
    rpc.add_SpeechGatewayServicer_to_server(service, server)
    address = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"
    bound_port = server.add_secure_port(address, credentials)
    if not bound_port:
        raise RuntimeError("mTLS bind failed; insecure fallback is disabled")
    return server, bound_port


async def warmup_bundle(bundle: EngineBundle, *, translation_budget_s: float = 30.) -> dict[str, Any]:
    """Consume/close a dedicated startup bundle without publishing any captions.

    The caller constructs an English ASR adapter and a translation provider with
    at most 32 output tokens. MLX's process-local model cache can survive this
    disposable bundle. Per-session ONNX objects are still loaded separately.
    Native work has no hard timeout: cancellation waits for it before cleanup.
    """
    from .translation import LatestTranslationRunner, ProviderError, TranslationRequest

    fence = NativeCallFence()
    runner = None
    report: dict[str, Any] = {
        "scope": "synthetic_startup_diagnostic", "quality_verified": False,
        "asr_input": "one_second_zero_pcm_16000_mono", "asr_language": "en",
        "translation_input": "fixed_readiness_diagnostic_en_to_ko",
        "native_hard_timeout": False,
        "vad_scope": "dedicated_startup_instance; session_instances_load_separately",
        "speaker_warmup": "not_configured" if bundle.speaker_analyzer is None else "not_performed_no_preload_api",
    }
    started = time.monotonic()
    try:
        if (isinstance(translation_budget_s, bool) or not math.isfinite(translation_budget_s)
                or not 0 < translation_budget_s <= 30):
            raise ValueError("Startup translation budget must be in (0, 30] seconds")
        report["translation_budget_ms"] = translation_budget_s * 1000
        if getattr(bundle.provider, "max_tokens", 32) > 32:
            raise ValueError("Startup translation provider must use at most 32 output tokens")
        if getattr(bundle.transcriber, "language", "en") != "en":
            raise ValueError("Startup ASR must use explicit English, not language detection")
        if hasattr(bundle.vad, "preload"):
            await fence.run(bundle.vad.preload)
        await fence.run(bundle.vad.reset)
        if bundle.speaker_analyzer is not None and hasattr(bundle.speaker_analyzer, "preload"):
            await fence.run(bundle.speaker_analyzer.preload)
            report["speaker_warmup"] = "dedicated_startup_instance_preloaded"
        stage = time.monotonic()
        await fence.run(bundle.transcriber.transcribe_pcm, (0.,) * 16000,
                        sample_rate=16000, window_start_ns=0)
        report["asr_ms"] = (time.monotonic() - stage) * 1000
        terminal_kind = terminal_error = None

        async def sink(event):
            nonlocal terminal_kind, terminal_error
            # Deliberately retain no diagnostic response text and emit no
            # PipelineEvent/CaptionStore update during startup.
            if event.kind != "preview":
                terminal_kind, terminal_error = event.kind, event.error

        runner = LatestTranslationRunner(bundle.provider, sink, max_pending=1, concurrency=1, max_segments=1)
        stage = time.monotonic()
        await runner.submit(TranslationRequest("startup-readiness", 1, "Ready.", "en", "ko",
            budget_ms=translation_budget_s * 1000, deadline_monotonic=stage + translation_budget_s))
        if terminal_kind != "completed":
            raise ProviderError(terminal_error or "incomplete", "Startup diagnostic translation did not complete")
        report["translation_ms"] = (time.monotonic() - stage) * 1000
    finally:
        fence.close()

        async def cleanup():
            await fence.wait_idle()
            try:
                if runner is not None:
                    await runner.close()
            finally:
                try:
                    await asyncio.to_thread(bundle.vad.reset)
                finally:
                    if bundle.aclose:
                        await bundle.aclose()

        cleanup_task = asyncio.create_task(cleanup())
        cancelled = False
        while True:
            try:
                await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError:
                if cleanup_task.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
    report["elapsed_ms"] = (time.monotonic() - started) * 1000
    return report


async def _serve(args):
    import httpx
    from .asr import MlxWhisperWindowTranscriber
    from .audio import SileroOnnxVad
    from .translation import LlamaCppProvider, OllamaProvider, LMStudioProvider
    from .tls import load_tls_material
    from .speaker_stream import LocalSpeakerAnalyzer

    if bool(args.speaker_segmentation_model) != bool(args.speaker_embedding_model):
        raise ValueError("Provide both speaker segmentation and embedding model paths")
    if args.speaker_profile and not args.speaker_segmentation_model:
        raise ValueError("A speaker profile requires both speaker model paths")

    def factory(opening, *, max_tokens=None):
        client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=3), follow_redirects=False, trust_env=False)
        provider_type = {"llamacpp": LlamaCppProvider, "ollama": OllamaProvider, "lmstudio": LMStudioProvider}[args.backend]
        return EngineBundle(MlxWhisperWindowTranscriber(args.asr_model, language=opening.source_language or None),
            provider_type(client, args.endpoint, args.model,
                          max_tokens=args.max_tokens if max_tokens is None else max_tokens),
            SileroOnnxVad(args.vad_model), aclose=client.aclose, engine_mode="configured_local_engines",
            speaker_analyzer=LocalSpeakerAnalyzer.from_local_models(
                args.speaker_segmentation_model, args.speaker_embedding_model, profile_path=args.speaker_profile
            ) if args.speaker_segmentation_model else None)

    material = load_tls_material(args.cert, args.key, args.client_ca, role="server")
    if getattr(args, "warmup", False):
        print(json.dumps({"kind": "gateway.warmup.started", "scope": "synthetic_startup_diagnostic",
                          "native_hard_timeout": False, "quality_verified": False}), flush=True)
        try:
            report = await warmup_bundle(factory(pb.OpenSession(protocol_version=1, session_id="startup-readiness",
                source_language="en", target_language="ko", translation_budget_ms=30000), max_tokens=32))
        except Exception as exc:
            print(json.dumps({"kind": "gateway.warmup.failed", "error": getattr(exc, "category", type(exc).__name__),
                              "quality_verified": False}), flush=True)
            raise
        print(json.dumps({"kind": "gateway.warmup.completed", **report}), flush=True)
    service = GatewayService(factory)
    server, port = create_secure_server(service, certificate_chain=material.certificate_chain,
        private_key=material.private_key, client_ca=material.root_certificates, host=args.host, port=args.port)
    await server.start()
    print(json.dumps({"kind": "gateway.listening", "host": args.host, "port": port,
                      "transport": "mtls", "engines": "initialized_after_authenticated_open",
                      "startup_warmup": "synthetic_completed" if getattr(args, "warmup", False) else "not_performed",
                      "quality_verified": False}), flush=True)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(3)
        await service.aclose(timeout_s=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cert", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--client-ca", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--vad-model", type=Path, required=True)
    parser.add_argument("--backend", choices=("llamacpp", "ollama", "lmstudio"), default="llamacpp")
    parser.add_argument("--endpoint", help="Server origin; default port 1234 for LM Studio, otherwise 8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", action="store_true",
                        help="Before listening, run synthetic ASR/translation diagnostics; native work has no hard timeout")
    parser.add_argument("--speaker-segmentation-model", type=Path)
    parser.add_argument("--speaker-embedding-model", type=Path)
    parser.add_argument("--speaker-profile", type=Path)
    args = parser.parse_args()
    if args.endpoint is None:
        args.endpoint = "http://127.0.0.1:1234" if args.backend == "lmstudio" else "http://127.0.0.1:8080"
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
