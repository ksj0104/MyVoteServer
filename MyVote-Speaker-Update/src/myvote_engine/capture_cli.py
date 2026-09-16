"""Capture Windows system output to timestamped, normalized WAV parts.

Parts remain separate across capture discontinuities. Their manifest preserves
the original session times; playing the WAV files back to back is not the original
timeline. This CLI records audio only and does not send it to a model or server.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import struct
import time
import wave

from .audio import AudioFrame
from .capture import CaptureFormat, CaptureStopped, NativeCaptureProcess, WindowsQpcClock
from .benchmark import percentile
from .dsp import NativePcmChunk, NormalizedBatch, StreamingNormalizer


class CaptureArchive:
    """One normalized WAV per uninterrupted audio epoch, with original times."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.parts: list[dict] = []
        self._writer = None
        self._previous: AudioFrame | None = None
        self.frame_count = 0
        self.sample_count = 0

    def write(self, frame: AudioFrame) -> None:
        previous = self._previous
        if self._writer and (frame.track_id, frame.capture_epoch) == (previous.track_id, previous.capture_epoch):
            if frame.sequence != previous.sequence + 1 or frame.start_time_ns != previous.end_time_ns:
                raise ValueError("Discontinuous normalized frames require a new archive part")
        else:
            self.close_part()
            path = self.output_dir / f"audio-part-{len(self.parts) + 1:04d}.wav"
            self._writer = wave.open(str(path), "wb")
            self._writer.setnchannels(1)
            self._writer.setsampwidth(2)
            self._writer.setframerate(16000)
            self.parts.append({
                "file": path.name, "track_id": frame.track_id,
                "capture_epoch": frame.capture_epoch, "start_ns": frame.start_time_ns,
                "end_ns": frame.end_time_ns, "samples": 0, "frames": 0,
                "encoding": "pcm16le", "sample_rate": 16000, "channels": 1,
            })
        # Clip before asymmetric PCM16 quantization (+1 maps to 32767).
        samples = [max(-32768, min(32767, round(value * 32768))) for value in frame.samples]
        self._writer.writeframesraw(struct.pack("<" + "h" * len(samples), *samples))
        self.parts[-1]["end_ns"] = frame.end_time_ns
        self.parts[-1]["samples"] += len(frame.samples)
        self.parts[-1]["frames"] += 1
        self._previous = frame
        self.frame_count += 1
        self.sample_count += len(frame.samples)

    def close_part(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None
        self._previous = None


async def record_capture(args, *, source_factory=None) -> dict:
    """Run the real helper by default; injected sources are explicitly labeled."""
    if not math.isfinite(args.seconds) or not .1 <= args.seconds <= 3600:
        raise ValueError("Capture seconds must be finite and in [0.1, 3600]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = CaptureArchive(output_dir)
    source = (source_factory() if source_factory else NativeCaptureProcess(
        args.helper, duration_s=args.seconds, device_id=args.device_id,
    ))
    clock = WindowsQpcClock() if source_factory is None else None
    packet_latencies = []
    normalized_latencies = []
    copy_to_python = []
    normalization_calls = []
    header = None
    terminal = None
    normalizer = None
    counts = {"native_packets": 0, "native_frames": 0, "normalization_events": 0,
              "audio_gaps": 0, "clock_drift_events": 0}
    failure = None
    cancelled = None
    inflight = None
    started = time.monotonic()
    with (output_dir / "timeline.jsonl").open("w", encoding="utf-8") as timeline:
        def event(item):
            timeline.write(json.dumps(item, ensure_ascii=False) + "\n")
            timeline.flush()

        def consume(batch: NormalizedBatch):
            ordered = sorted(batch.events, key=lambda item: item.before_frame_index)
            if any(not 0 <= item.before_frame_index <= len(batch.frames) for item in ordered):
                raise ValueError("Invalid normalized event position")
            index = 0
            for offset in range(len(batch.frames) + 1):
                while index < len(ordered) and ordered[index].before_frame_index == offset:
                    item = ordered[index]
                    counts["normalization_events"] += 1
                    if item.kind == "audio.gap":
                        archive.close_part()
                        counts["audio_gaps"] += 1
                    if item.kind == "clock.drift":
                        counts["clock_drift_events"] += 1
                    event({"type": "normalization", **asdict(item)})
                    index += 1
                if offset < len(batch.frames):
                    frame = batch.frames[offset]
                    latency_ms = None
                    if clock is not None and header is not None:
                        latency_ms = ((clock.now_100ns() - header.qpc_origin_100ns) * 100
                                      - frame.end_time_ns) / 1_000_000
                        normalized_latencies.append(latency_ms)
                    archive.write(frame)
                    event({"type": "normalized_frame", "track_id": frame.track_id,
                           "capture_epoch": frame.capture_epoch, "sequence": frame.sequence,
                           "start_ns": frame.start_time_ns, "end_ns": frame.end_time_ns,
                           "samples": len(frame.samples), "part": archive.parts[-1]["file"],
                           "source_end_to_normalized_ready_signed_ms": latency_ms})

        try:
            async with source:
                async for record in source.records():
                    if isinstance(record, CaptureFormat):
                        header = record
                        normalizer = StreamingNormalizer(
                            record.pcm, track_id=record.track_id,
                            capture_epoch=record.capture_epoch,
                            qpc_origin_100ns=record.qpc_origin_100ns,
                        )
                        event({"type": "format", **asdict(record)})
                    elif isinstance(record, NativePcmChunk):
                        if normalizer is None:
                            raise ValueError("PCM arrived before format")
                        counts["native_packets"] += 1
                        counts["native_frames"] += record.frames
                        packet_ms = None
                        copied_ms = None
                        received_qpc = clock.now_100ns() if clock is not None else None
                        if clock is not None and not record.flags & 4:
                            packet_ms = ((received_qpc - record.qpc_position_100ns) / 10_000
                                         - record.frames / header.pcm.sample_rate * 1000)
                            packet_latencies.append(packet_ms)
                        if received_qpc is not None and record.copied_qpc_100ns is not None:
                            copied_ms = (received_qpc - record.copied_qpc_100ns) / 10_000
                            copy_to_python.append(copied_ms)
                        event({"type": "native_packet", "sequence": record.sequence,
                               "device_position": record.device_position,
                               "qpc_position_100ns": record.qpc_position_100ns,
                               "frames": record.frames, "flags": record.flags,
                               "source_end_to_python_signed_ms": packet_ms,
                               "copied_qpc_100ns": record.copied_qpc_100ns,
                               "copy_to_python_ms": copied_ms})
                        normalize_started = time.perf_counter_ns()
                        inflight = asyncio.create_task(asyncio.to_thread(normalizer.push, record))
                        batch = await asyncio.shield(inflight)
                        inflight = None
                        normalization_calls.append((time.perf_counter_ns() - normalize_started) / 1_000_000)
                        consume(batch)
                    elif isinstance(record, CaptureStopped):
                        terminal = record
                        event({"type": "stopped", **asdict(record)})
                    else:
                        raise ValueError("Unknown capture record")
        except asyncio.CancelledError as exc:
            cancelled = exc
            failure = {"type": "CancelledError", "message": "Capture was cancelled"}
            event({"type": "capture.cancelled", "error": failure})
        except Exception as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            event({"type": "capture.failed", "error": failure})
        finally:
            try:
                if inflight is not None:
                    # Cancelling a to_thread await does not stop native SoXR.
                    # Preserve its result, then flush on this same serial path.
                    try:
                        consume(await asyncio.shield(inflight))
                    except Exception as exc:
                        event({"type": "capture.pending_failed", "error": type(exc).__name__})
                        if failure is None:
                            failure = {"type": type(exc).__name__, "message": str(exc)}
                    inflight = None
                if normalizer is not None:
                    consume(await asyncio.to_thread(normalizer.flush))
            except Exception as exc:
                event({"type": "capture.flush_failed", "error": type(exc).__name__})
                if failure is None:
                    failure = {"type": type(exc).__name__, "message": str(exc)}
            finally:
                archive.close_part()
    if terminal is None and failure is None:
        failure = {"type": "MissingTerminal", "message": "Capture source did not finish explicitly"}
    status = ("failed" if failure or (terminal and terminal.error) else
              "partial" if counts["audio_gaps"] or counts["clock_drift_events"]
              or (terminal and terminal.dropped_packets) else
              "no_audio_packets" if not counts["native_packets"] else "captured")
    summary = {
        "schema": "myvote.capture_archive", "schema_version": 1, "status": status,
        "engine_mode": "injected_capture_source" if source_factory else "native_wasapi_loopback",
        "platform": platform.platform(), "python": platform.python_version(),
        "requested_seconds": args.seconds, "wall_seconds": time.monotonic() - started,
        "source_format": asdict(header) if header else None,
        "terminal": asdict(terminal) if terminal else None, "failure": failure,
        "counts": {**counts, "normalized_frames": archive.frame_count,
                   "normalized_samples": archive.sample_count},
        "normalization": dict(normalizer.stats) if normalizer is not None else None,
        "timing": {
            "scope": "same_windows_qpc_capture_to_normalization" if clock else "unmeasured_fixture",
            "source_clock_status": ("unmeasured_fixture" if clock is None else
                                    "negative_offsets_observed_unvalidated_endpoint_timing"
                                    if any(value < 0 for value in packet_latencies + normalized_latencies)
                                    else "not_acoustically_validated"),
            "packet_end_to_python_signed_p95_ms": percentile(packet_latencies, .95),
            "normalized_end_to_ready_signed_p50_ms": percentile(normalized_latencies, .50),
            "normalized_end_to_ready_signed_p95_ms": percentile(normalized_latencies, .95),
            "normalized_end_to_ready_signed_max_ms": max(normalized_latencies, default=None),
            "negative_packet_offsets": sum(value < 0 for value in packet_latencies),
            "negative_normalized_offsets": sum(value < 0 for value in normalized_latencies),
            "copy_to_python_p95_ms": percentile(copy_to_python, .95),
            "copy_to_python_sample_count": len(copy_to_python),
            "normalization_call_p95_ms": percentile(normalization_calls, .95),
            "normalization_call_max_ms": max(normalization_calls, default=None),
            "normalization_duration_clock": time.get_clock_info("perf_counter").implementation,
            "packet_samples": len(packet_latencies), "normalized_samples": len(normalized_latencies),
        },
        "parts": archive.parts,
        "limitations": [
            "Capture and normalization only; no ASR, translation, speaker recognition, network, or UI",
            "WAV parts have separate original start times; concatenating them removes real gaps",
            "No device packets is not proof of digital silence or a working transcription engine",
            "QPC drift is reported; adaptive device clock resampling is not implemented",
            "Signed source-clock offsets preserve negative values; they are not validated acoustic latency",
            "Copy-to-Python measures helper queue/pipe/decode only; normalization call time excludes SoXR lookahead",
        ],
    }
    (output_dir / "capture.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    if cancelled is not None:
        raise cancelled
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--device-id")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/windows-capture"))
    args = parser.parse_args()
    try:
        result = asyncio.run(record_capture(args))
    except Exception as exc:
        parser.exit(2, f"Capture setup failed ({type(exc).__name__}): {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["status"] in ("failed", "partial") else 0)


if __name__ == "__main__":
    main()
