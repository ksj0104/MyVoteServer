"""Streaming PCM, optional local Silero VAD, and bounded speech windows.

The window builder knows no speech model. Its probabilities must come from a
real VAD at runtime; deterministic probabilities belong only in contract tests.
All timestamps are on the capture clock, with exact 16 kHz sample arithmetic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import struct
from typing import Iterator, Protocol, runtime_checkable
import wave


SAMPLE_RATE = 16_000
NS = 1_000_000_000
SAMPLE_NS = NS // SAMPLE_RATE


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class AudioFrame:
    track_id: str
    capture_epoch: str
    sequence: int
    start_time_ns: int
    samples: tuple[float, ...]
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("track_id must be a nonempty string")
        if not isinstance(self.capture_epoch, str) or not self.capture_epoch:
            raise ValueError("capture_epoch must be a nonempty string")
        _integer(self.sequence, "sequence")
        _integer(self.start_time_ns, "start_time_ns")
        if self.sample_rate != SAMPLE_RATE or isinstance(self.sample_rate, bool):
            raise ValueError("AudioFrame requires mono 16 kHz PCM; resample upstream")
        if not isinstance(self.samples, tuple) or not 1 <= len(self.samples) <= SAMPLE_RATE:
            raise ValueError("samples must be a nonempty tuple of at most one second")
        if any(isinstance(x, bool) or not isinstance(x, (float, int))
               or not math.isfinite(x) or not -1 <= x <= 1 for x in self.samples):
            raise ValueError("PCM samples must be finite numbers in [-1, 1]")

    @property
    def end_time_ns(self) -> int:
        return self.start_time_ns + len(self.samples) * SAMPLE_NS


@dataclass(frozen=True)
class SpeechWindow:
    segment_id: str
    track_id: str
    capture_epoch: str
    window_start_ns: int
    window_end_ns: int
    samples: tuple[float, ...]
    final: bool
    reason: str
    emit_start_ns: int
    last_speech_end_ns: int | None = None


class AudioDiscontinuity(ValueError):
    """A track, epoch, sequence, or capture timestamp changed unexpectedly.

    Flush the last good speech window, report the gap, reset both VAD and window
    builder, and then retry the new frame. No samples are fabricated for a gap.
    """


class _Continuity:
    def __init__(self) -> None:
        self.last: AudioFrame | None = None
        self.broken = False

    def check(self, frame: AudioFrame) -> None:
        if self.broken:
            raise AudioDiscontinuity("audio stream requires an explicit reset")
        previous = self.last
        if previous is not None and (
            frame.track_id != previous.track_id
            or frame.capture_epoch != previous.capture_epoch
            or frame.sequence != previous.sequence + 1
            or frame.start_time_ns != previous.end_time_ns
        ):
            self.broken = True
            raise AudioDiscontinuity("track, epoch, sequence, or sample timestamp is discontinuous")

    def accept(self, frame: AudioFrame) -> None:
        self.last = frame


def iter_wav_frames(path: str | Path, *, frame_samples: int = 512,
                    track_id: str = "system", capture_epoch: str = "replay-1",
                    start_time_ns: int = 0) -> Iterator[AudioFrame]:
    """Read PCM16 mono 16 kHz WAV incrementally, preserving an unpadded tail.

    The caller controls replay pacing. Reading a frame does not sleep or pretend
    that its capture timestamp is a server wall-clock measurement.
    """
    _integer(frame_samples, "frame_samples", 1)
    _integer(start_time_ns, "start_time_ns")
    if frame_samples > SAMPLE_RATE:
        raise ValueError("frame_samples must be at most one second")
    if not isinstance(track_id, str) or not track_id:
        raise ValueError("track_id must be a nonempty string")
    if not isinstance(capture_epoch, str) or not capture_epoch:
        raise ValueError("capture_epoch must be a nonempty string")
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels() != 1 or stream.getframerate() != SAMPLE_RATE
                or stream.getsampwidth() != 2 or stream.getcomptype() != "NONE"):
            raise ValueError("WAV requires uncompressed PCM16 mono 16 kHz")
        total_samples = stream.getnframes()
        offset = 0
        sequence = 0
        while offset < total_samples:
            count = min(frame_samples, total_samples - offset)
            raw = stream.readframes(count)
            if len(raw) != count * 2:
                raise ValueError("truncated PCM16 WAV")
            samples = tuple(item[0] / 32768.0 for item in struct.iter_unpack("<h", raw))
            yield AudioFrame(track_id, capture_epoch, sequence,
                             start_time_ns + offset * SAMPLE_NS, samples)
            offset += count
            sequence += 1


@runtime_checkable
class VoiceActivityDetector(Protocol):
    def probability(self, frame: AudioFrame) -> float: ...

    def reset(self) -> None: ...


class SileroOnnxVad:
    """CPU-only adapter for the official Silero ONNX recurrent I/O contract.

    A caller-supplied LOCAL model is mandatory; no weights or Python packages are
    downloaded. Imports/session initialization are lazy. One instance belongs to
    one serial stream. Use 512-sample frames, except a final short WAV tail; tail
    padding is model input only and never changes the capture timestamp.

    Contract source (checked 2026-09-15):
    https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self._numpy = None
        self._session = None
        self._state = None
        self._context = None
        self._continuity = _Continuity()
        self._tail_seen = False

    def preload(self) -> None:
        """Prepare the local model before starting capture-time measurements."""
        if self._session is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError("prepare an explicit local Silero ONNX model before inference")
        try:
            numpy = importlib.import_module("numpy")
            runtime = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise RuntimeError("install numpy and onnxruntime for the optional Silero VAD") from exc
        options = runtime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        session = runtime.InferenceSession(str(self.model_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        if {item.name for item in session.get_inputs()} != {"input", "state", "sr"}:
            raise ValueError("unsupported ONNX model: expected Silero input/state/sr inputs")
        if {item.name for item in session.get_outputs()} != {"output", "stateN"}:
            raise ValueError("unsupported ONNX model: expected Silero output/stateN outputs")
        self._numpy = numpy
        self._session = session
        self.reset()

    def reset(self) -> None:
        self._continuity = _Continuity()
        self._tail_seen = False
        if self._numpy is not None:
            self._state = self._numpy.zeros((2, 1, 128), dtype=self._numpy.float32)
            self._context = self._numpy.zeros((1, 64), dtype=self._numpy.float32)

    def probability(self, frame: AudioFrame) -> float:
        if len(frame.samples) > 512:
            raise ValueError("Silero VAD requires 512 samples or a final short tail")
        self.preload()
        self._continuity.check(frame)
        if self._tail_seen:
            raise AudioDiscontinuity("short VAD frame ended the stream; reset before further audio")
        numpy = self._numpy
        waveform = numpy.zeros((1, 512), dtype=numpy.float32)
        waveform[0, :len(frame.samples)] = frame.samples
        inputs = numpy.concatenate((self._context, waveform), axis=1)
        output, state = self._session.run(["output", "stateN"], {
            "input": inputs, "state": self._state,
            "sr": numpy.asarray(SAMPLE_RATE, dtype=numpy.int64),
        })
        if (output.shape != (1, 1) or state.shape != (2, 1, 128)
                or not numpy.isfinite(output).all() or not numpy.isfinite(state).all()):
            raise ValueError("Silero returned an invalid probability or recurrent state")
        probability = float(output[0, 0])
        if not 0 <= probability <= 1:
            raise ValueError("Silero probability must lie in [0, 1]")
        self._state = state
        self._context = inputs[:, -64:].copy()
        self._tail_seen = len(frame.samples) < 512
        self._continuity.accept(frame)
        return probability


class SpeechWindowBuilder:
    """One track's endpointed, incremental windows with bounded context.

    A final window may repeat the last incremental end timestamp: this signals
    finalization, not a second LocalAgreement vote. Forced continuations retain
    context before emit_start_ns, which consumers must not emit/vote on again.
    Durations are sample-exact. Endpoint decisions occur at frame boundaries.
    """

    def __init__(self, *, step_s: float = .5, min_window_s: float = .5,
                 endpoint_s: float = .32, pre_roll_s: float = .16,
                 pad_s: float = .096, max_window_s: float = 15.,
                 overlap_s: float = .5, threshold: float = .5,
                 negative_threshold: float = .35) -> None:
        durations = {"step_s": step_s, "min_window_s": min_window_s,
                     "endpoint_s": endpoint_s, "pre_roll_s": pre_roll_s,
                     "pad_s": pad_s, "max_window_s": max_window_s,
                     "overlap_s": overlap_s}
        for name, value in durations.items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative duration")
        if not (0 < step_s <= max_window_s <= 30 and 0 < min_window_s <= max_window_s
                and endpoint_s > 0 and pad_s <= endpoint_s
                and pre_roll_s < max_window_s and overlap_s < max_window_s):
            raise ValueError("invalid step, endpoint, padding, or maximum window duration")
        if (isinstance(threshold, bool) or isinstance(negative_threshold, bool)
                or not 0 <= negative_threshold <= threshold <= 1):
            raise ValueError("VAD thresholds must satisfy 0 <= negative <= positive <= 1")
        for name, value in durations.items():
            setattr(self, "_" + name.removesuffix("_s"), round(value * SAMPLE_RATE))
        if min(self._step, self._min_window, self._endpoint) < 1:
            raise ValueError("step, minimum window, and endpoint must be at least one sample")
        if not (self._overlap < self._max_window and self._pre_roll < self._max_window):
            raise ValueError("rounded overlap and pre-roll must be shorter than the maximum window")
        self.threshold = threshold
        self.negative_threshold = negative_threshold
        self._counter = 0
        self.reset()

    def reset(self) -> None:
        """Discard buffered audio and accept a new epoch; flush first if needed."""
        self._continuity = _Continuity()
        self._origin_ns = 0
        self._position = 0
        self._recent: deque[float] = deque(maxlen=self._pre_roll)
        self._samples: list[float] = []
        self._start = 0
        self._emit_start = 0
        self._segment_id: str | None = None
        self._track_id = ""
        self._epoch = ""
        self._last_speech_end = 0
        self._last_output_end = 0
        self._last_final_end = 0
        self._closed = False

    @property
    def buffered_samples(self) -> int:
        """Audio held for the current ASR window plus bounded onset pre-roll."""
        return len(self._samples) + len(self._recent)

    def _new_segment(self, start: int, emit_start: int, samples: list[float]) -> None:
        self._counter += 1
        self._segment_id = f"{self._track_id}:{self._epoch}:{self._counter}"
        self._start = start
        self._emit_start = emit_start
        self._samples = samples
        self._last_output_end = emit_start

    def _window(self, end: int, *, final: bool, reason: str) -> SpeechWindow:
        self._last_output_end = max(self._last_output_end, end)
        if final:
            self._last_final_end = max(self._last_final_end, end)
        return SpeechWindow(
            self._segment_id, self._track_id, self._epoch,
            self._origin_ns + self._start * SAMPLE_NS,
            self._origin_ns + end * SAMPLE_NS,
            tuple(self._samples[:end - self._start]), final, reason,
            self._origin_ns + self._emit_start * SAMPLE_NS,
            self._origin_ns + min(end, self._last_speech_end) * SAMPLE_NS,
        )

    def push(self, frame: AudioFrame, speech_probability: float) -> list[SpeechWindow]:
        if self._closed:
            raise RuntimeError("speech stream is flushed; reset before more audio")
        if (isinstance(speech_probability, bool) or not math.isfinite(speech_probability)
                or not 0 <= speech_probability <= 1):
            raise ValueError("speech_probability must be finite and lie in [0, 1]")
        self._continuity.check(frame)
        if self._continuity.last is None:
            self._origin_ns = frame.start_time_ns
            self._track_id, self._epoch = frame.track_id, frame.capture_epoch
        start = self._position
        end = start + len(frame.samples)
        active = self._segment_id is not None
        speech = speech_probability >= (self.negative_threshold if active else self.threshold)
        if speech and not active:
            window_start = start - len(self._recent)
            self._new_segment(window_start, max(window_start, self._last_final_end), list(self._recent))
        if self._segment_id is not None:
            self._samples.extend(frame.samples)
            if speech:
                self._last_speech_end = end
        self._recent.extend(frame.samples)
        self._position = end
        self._continuity.accept(frame)
        if self._segment_id is None:
            return []

        windows = []
        # Silence after the endpoint never consumes a maximum-window split.
        endpoint = not speech and end - self._last_speech_end >= self._endpoint
        visible_end = end if speech else min(end, self._last_speech_end + self._pad)
        while visible_end - self._start >= self._max_window:
            boundary = self._start + self._max_window
            windows.append(self._window(boundary, final=True, reason="max_window"))
            continuation_start = boundary - self._overlap
            remaining = self._samples[continuation_start - self._start:]
            self._new_segment(continuation_start, boundary, remaining)

        if endpoint:
            if visible_end > self._emit_start:
                windows.append(self._window(visible_end, final=True, reason="endpoint"))
            self._samples = []
            self._segment_id = None
        elif (visible_end - self._start >= self._min_window
              and visible_end - self._last_output_end >= self._step
              and visible_end > self._emit_start):
            windows.append(self._window(visible_end, final=False, reason="incremental"))
        return windows

    def flush(self, *, reason: str = "stop") -> list[SpeechWindow]:
        """Finalize the last real samples once, including a short utterance."""
        if self._closed:
            return []
        self._closed = True
        windows = []
        if self._segment_id is not None:
            end = min(self._position, self._last_speech_end + self._pad)
            if end > self._emit_start:
                windows.append(self._window(end, final=True, reason=reason))
        self._samples = []
        self._recent.clear()
        self._segment_id = None
        return windows
