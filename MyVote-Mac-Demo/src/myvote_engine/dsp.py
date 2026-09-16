"""Native Windows PCM to sample-timed mono 16 kHz frames using streaming SoXR.

Optional imports are lazy. Gap events have a frame insertion index so callers can
flush/reset their VAD after the old short tail and before the new capture epoch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import importlib
from typing import Literal

from .audio import AudioFrame, NS, SAMPLE_NS, SAMPLE_RATE


DATA_DISCONTINUITY = 1
SILENT = 2
TIMESTAMP_ERROR = 4
Encoding = Literal["float32le", "pcm16le", "pcm24le", "pcm32le"]
_WIDTHS = {"float32le": 4, "pcm16le": 2, "pcm24le": 3, "pcm32le": 4}


def _int(value, name, low=0, high=2**64-1):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


@dataclass(frozen=True)
class NativePcmFormat:
    sample_rate: int
    channels: int
    encoding: Encoding
    valid_bits: int
    channel_mask: int
    block_align: int

    def __post_init__(self):
        _int(self.sample_rate, "sample_rate", 8000, 192000)
        _int(self.channels, "channels", 1, 2)
        if self.encoding not in _WIDTHS:
            raise ValueError("unsupported native PCM encoding")
        bits = _WIDTHS[self.encoding] * 8
        _int(self.valid_bits, "valid_bits", 1, bits)
        if self.encoding == "float32le" and self.valid_bits != 32:
            raise ValueError("float32 PCM requires 32 valid bits")
        _int(self.block_align, "block_align", 1, 8)
        if self.block_align != self.channels * _WIDTHS[self.encoding]:
            raise ValueError("block_align must equal channels times container bytes")
        _int(self.channel_mask, "channel_mask", 0, 0xFFFFFFFF)
        allowed = (0, 1, 2, 4) if self.channels == 1 else (0, 3)
        if self.channel_mask not in allowed:
            raise ValueError("only unspecified/front mono or front-left/right stereo layouts are supported")


@dataclass(frozen=True)
class NativePcmChunk:
    sequence: int
    device_position: int
    qpc_position_100ns: int
    frames: int
    flags: int
    data: bytes
    copied_qpc_100ns: int | None = None

    def __post_init__(self):
        for name in ("sequence", "device_position", "qpc_position_100ns"):
            _int(getattr(self, name), name)
        if self.copied_qpc_100ns is not None:
            _int(self.copied_qpc_100ns, "copied_qpc_100ns")
        _int(self.frames, "frames", 1, 192000)
        _int(self.flags, "flags", 0, 7)
        if not isinstance(self.data, bytes) or len(self.data) > 192000 * 8:
            raise ValueError("native payload must be bounded immutable bytes")


@dataclass(frozen=True)
class NormalizationEvent:
    kind: str
    reason: str
    before_frame_index: int
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedBatch:
    frames: tuple[AudioFrame, ...] = ()
    events: tuple[NormalizationEvent, ...] = ()


class StreamingNormalizer:
    """One native stream, fixed format, bounded chunks, one serial calling thread.

    Output timestamps use the first valid QPC anchor plus output sample position,
    never processing/arrival time. SoXR delay is separately observable in stats.
    Downmix is mono passthrough or .5*left + .5*right. Surround is rejected.
    """

    def __init__(self, format: NativePcmFormat, track_id: str, capture_epoch: str,
                 qpc_origin_100ns: int, *, max_qpc_drift_ns: int = 20_000_000):
        if not isinstance(format, NativePcmFormat):
            raise ValueError("format must be NativePcmFormat")
        for name, value in (("track_id", track_id), ("capture_epoch", capture_epoch)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        _int(qpc_origin_100ns, "qpc_origin_100ns")
        _int(max_qpc_drift_ns, "max_qpc_drift_ns", 1)
        self.format = format
        self.track_id = track_id
        self.qpc_origin_100ns = qpc_origin_100ns
        self.max_qpc_drift_ns = max_qpc_drift_ns
        self._base_epoch = capture_epoch
        self.capture_epoch = capture_epoch
        self._gap_index = 0
        self._numpy = self._soxr = None
        self._closed = False
        self._last_emitted_end_ns = 0
        self._stats = dict(accepted_input_frames=0, output_samples=0, clipped_input_samples=0,
                           clipped_output_samples=0, silent_frames=0, dropped_frames=0,
                           dropped_chunks=0,
                           final_rounding_samples_trimmed=0,
                           gaps=0, qpc_drift_exceeded=0, last_qpc_drift_ns=0,
                           max_abs_qpc_drift_ns=0, qpc_steps=0, last_qpc_step_ns=0,
                           max_abs_qpc_step_ns=0)
        self._clear_run()

    def _clear_run(self):
        self._resampler = None
        self._pending = deque()
        self._last_chunk: NativePcmChunk | None = None
        self._anchor_qpc = self._anchor_device = None
        self._anchor_ns = None
        self._run_input_frames = self._output_position = self._frame_sequence = 0
        self._drift_alarm = False

    @property
    def stats(self) -> dict:
        result = dict(self._stats)
        result["pending_output_samples"] = len(self._pending)
        result["algorithmic_delay_samples"] = (
            float(self._resampler.delay()) if self._resampler is not None else 0.0)
        result["algorithmic_delay_ms"] = result["algorithmic_delay_samples"] / 16
        return result

    def _load(self):
        if self._numpy is not None:
            return
        try:
            numpy = importlib.import_module("numpy")
            soxr = importlib.import_module("soxr")
        except ImportError as exc:
            raise RuntimeError("install optional numpy and soxr packages for native PCM normalization") from exc
        self._numpy, self._soxr = numpy, soxr

    def _decode(self, chunk: NativePcmChunk):
        if chunk.frames > self.format.sample_rate:
            raise ValueError("one native PCM chunk must not exceed one second")
        expected = chunk.frames * self.format.block_align
        if chunk.flags & SILENT:
            if len(chunk.data) not in (0, expected):
                raise ValueError("silent PCM payload must be empty or contain exactly frames*block_align bytes")
        elif len(chunk.data) != expected:
            raise ValueError("native PCM payload length differs from frames*block_align")
        self._load()
        np = self._numpy
        if chunk.flags & SILENT:
            # WASAPI explicitly says to ignore the actual bytes for SILENT.
            return np.zeros(chunk.frames, dtype=np.float32), 0
        encoding = self.format.encoding
        if encoding == "float32le":
            values = np.frombuffer(chunk.data, dtype="<f4").reshape(chunk.frames, self.format.channels)
            if not np.isfinite(values).all():
                raise ValueError("non-finite native float PCM")
            clipped = int(np.count_nonzero((values < -1) | (values > 1)))
            values = np.clip(values, -1, 1)
        else:
            if encoding == "pcm24le":
                raw = np.frombuffer(chunk.data, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
                integers = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
                integers = (integers ^ 0x800000) - 0x800000
            else:
                integers = np.frombuffer(chunk.data, dtype="<i2" if encoding == "pcm16le" else "<i4")
                integers = integers.astype(np.int64)
            shift = _WIDTHS[encoding] * 8 - self.format.valid_bits
            if shift and np.any(integers & ((1 << shift) - 1)):
                raise ValueError("PCM valid bits must be left aligned with zero unused low bits")
            integers = integers >> shift
            values = (integers.astype(np.float64) / (1 << (self.format.valid_bits - 1)))
            values = values.astype(np.float32).reshape(chunk.frames, self.format.channels)
            clipped = 0
        mono = values[:, 0] if self.format.channels == 1 else .5 * values[:, 0] + .5 * values[:, 1]
        return np.ascontiguousarray(mono, dtype=np.float32), clipped

    def _output(self, values, *, final=False) -> list[AudioFrame]:
        np = self._numpy
        if values.ndim != 1 or len(values) > SAMPLE_RATE * 2 or not np.isfinite(values).all():
            raise ValueError("resampler produced invalid or unexpectedly unbounded output")
        if final:
            # Do not extend a completed source interval by the resampler's final
            # rounding. Trim at most a fractional output sample; never pad input.
            limit = self._run_input_frames * SAMPLE_RATE // self.format.sample_rate
            remaining = limit - self._output_position - len(self._pending)
            if remaining < 0 or len(values) > remaining + 1:
                raise ValueError("resampler output exceeds the completed source duration")
            self._stats["final_rounding_samples_trimmed"] += max(0, len(values) - remaining)
            values = values[:remaining]
        self._stats["clipped_output_samples"] += int(np.count_nonzero((values < -1) | (values > 1)))
        self._pending.extend(float(value) for value in np.clip(values, -1, 1))
        frames = []
        while len(self._pending) >= 512 or (final and self._pending):
            count = min(512, len(self._pending))
            samples = tuple(self._pending.popleft() for _ in range(count))
            frame = AudioFrame(self.track_id, self.capture_epoch, self._frame_sequence,
                               self._anchor_ns + self._output_position * SAMPLE_NS, samples)
            frames.append(frame)
            self._frame_sequence += 1
            self._output_position += count
            self._stats["output_samples"] += count
            self._last_emitted_end_ns = frame.end_time_ns
        return frames

    def _finish_run(self) -> list[AudioFrame]:
        if self._anchor_ns is None:
            return []
        tail = self._numpy.empty(0, dtype=self._numpy.float32)
        if self._resampler is not None:
            tail = self._resampler.resample_chunk(tail, last=True)
        frames = self._output(tail, final=True)
        expected = self._run_input_frames * SAMPLE_RATE // self.format.sample_rate
        if self._output_position != expected:
            raise ValueError("resampler did not produce the expected completed source duration")
        return frames

    def _gap(self, reason, chunk, frames, events, *, extra=None):
        previous = self._last_chunk
        frames.extend(self._finish_run())
        self._gap_index += 1
        self._stats["gaps"] += 1
        old_epoch = self.capture_epoch
        self.capture_epoch = f"{self._base_epoch}:gap-{self._gap_index}"
        data = {"previous_epoch": old_epoch, "capture_epoch": self.capture_epoch,
                "sequence": chunk.sequence, "device_position": chunk.device_position,
                "qpc_position_100ns": chunk.qpc_position_100ns, "flags": chunk.flags}
        if previous is not None:
            data["device_frame_delta"] = chunk.device_position - previous.device_position - previous.frames
        if extra:
            data.update(extra)
        events.append(NormalizationEvent("audio.gap", reason, len(frames), data))
        self._clear_run()

    def push(self, chunk: NativePcmChunk) -> NormalizedBatch:
        if self._closed:
            raise RuntimeError("normalizer is flushed; reset before further native chunks")
        if not isinstance(chunk, NativePcmChunk):
            raise ValueError("chunk must be NativePcmChunk")
        # Decode/validate before mutating a valid preceding run.
        mono, clipped = self._decode(chunk)
        frames, events = [], []
        if chunk.flags & TIMESTAMP_ERROR:
            self._gap("timestamp_error", chunk, frames, events)
            self._stats["dropped_frames"] += chunk.frames
            self._stats["dropped_chunks"] += 1
            return NormalizedBatch(tuple(frames), tuple(events))
        previous = self._last_chunk
        if previous is not None:
            if chunk.flags & DATA_DISCONTINUITY:
                self._gap("data_discontinuity", chunk, frames, events)
            elif chunk.sequence != previous.sequence + 1:
                self._gap("sequence_gap", chunk, frames, events)
            elif chunk.device_position != previous.device_position + previous.frames:
                self._gap("device_position_gap", chunk, frames, events)
            else:
                # An abrupt QPC jump can conceal a real wall-time hole even if
                # a device continues its sample counter. Do not compress that
                # jump into one continuous source timeline. Gradual cumulative
                # drift remains a separate observation below.
                step_ns = ((chunk.qpc_position_100ns - previous.qpc_position_100ns) * 100
                           - (chunk.device_position - previous.device_position) * NS
                           // self.format.sample_rate)
                self._stats["last_qpc_step_ns"] = step_ns
                self._stats["max_abs_qpc_step_ns"] = max(self._stats["max_abs_qpc_step_ns"], abs(step_ns))
                if abs(step_ns) > self.max_qpc_drift_ns:
                    self._stats["qpc_steps"] += 1
                    self._gap("qpc_step", chunk, frames, events,
                              extra={"qpc_step_ns": step_ns,
                                     "threshold_ns": self.max_qpc_drift_ns,
                                     "timestamp_uncertain": True})
        if self._anchor_ns is None:
            anchor_ns = (chunk.qpc_position_100ns - self.qpc_origin_100ns) * 100
            if anchor_ns < self._last_emitted_end_ns:
                self._gap("backward_qpc_anchor", chunk, frames, events)
                self._stats["dropped_frames"] += chunk.frames
                self._stats["dropped_chunks"] += 1
                return NormalizedBatch(tuple(frames), tuple(events))
            self._anchor_ns = anchor_ns
            self._anchor_qpc = chunk.qpc_position_100ns
            self._anchor_device = chunk.device_position
            if self.format.sample_rate != SAMPLE_RATE:
                self._resampler = self._soxr.ResampleStream(
                    self.format.sample_rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ")
        drift_ns = ((chunk.qpc_position_100ns - self._anchor_qpc) * 100
                    - (chunk.device_position - self._anchor_device) * NS // self.format.sample_rate)
        self._stats["last_qpc_drift_ns"] = drift_ns
        self._stats["max_abs_qpc_drift_ns"] = max(self._stats["max_abs_qpc_drift_ns"], abs(drift_ns))
        exceeded = abs(drift_ns) > self.max_qpc_drift_ns
        if exceeded:
            self._stats["qpc_drift_exceeded"] += 1
        if exceeded != self._drift_alarm:
            events.append(NormalizationEvent("clock.drift" if exceeded else "clock.recovered",
                "qpc_vs_device_position", len(frames),
                {"drift_ns": drift_ns, "threshold_ns": self.max_qpc_drift_ns,
                 "sequence": chunk.sequence, "capture_epoch": self.capture_epoch}))
            self._drift_alarm = exceeded
        self._stats["accepted_input_frames"] += chunk.frames
        self._stats["clipped_input_samples"] += clipped
        if chunk.flags & SILENT:
            self._stats["silent_frames"] += chunk.frames
        self._run_input_frames += chunk.frames
        result = mono if self._resampler is None else self._resampler.resample_chunk(mono)
        frames.extend(self._output(result))
        self._last_chunk = chunk
        return NormalizedBatch(tuple(frames), tuple(events))

    def flush(self) -> NormalizedBatch:
        """Drain SoXR once and emit the unpadded short tail, then close the stream."""
        if self._closed:
            return NormalizedBatch()
        frames = self._finish_run()
        self._closed = True
        self._clear_run()
        return NormalizedBatch(tuple(frames))

    def reset(self, capture_epoch: str) -> None:
        """Start an explicit new epoch after flush; accumulated stats are retained."""
        if not self._closed:
            raise RuntimeError("flush the previous stream before reset")
        if not isinstance(capture_epoch, str) or not capture_epoch.strip() or capture_epoch == self.capture_epoch:
            raise ValueError("reset requires a different nonempty capture epoch")
        self._base_epoch = self.capture_epoch = capture_epoch
        self._gap_index = 0
        self._closed = False
        self._clear_run()
