"""Bounded asynchronous overlap separation and independent lane transcription.

This module does not assign speaker identities or replace captions. The owning
session validates parent revisions and atomically applies successful corrections.
Waveform correlation aligns anonymous output lanes; it is not a purity test.
"""

from __future__ import annotations

import asyncio
from array import array
from collections import deque
from dataclasses import dataclass, field
import math
import json
import time
import threading

from .asr import ASRHypothesis, TimedWord
from .audio import AudioFrame, SAMPLE_NS
from .inference_scheduler import AsrInferenceScheduler


class OverlapDeadlineExpired(RuntimeError):
    """Correction budget expired before a new native call was admitted."""


class OverlapRunRetired(RuntimeError):
    """The source run ended while work was waiting for native admission."""


@dataclass(frozen=True)
class OverlapStreamConfig:
    buffer_s: float = 20.0
    window_s: float = 8.0
    min_shared_s: float = .2
    min_correlation: float = .35
    permutation_margin: float = .10
    min_lane_rms: float = 1e-5

    def __post_init__(self):
        values = (self.buffer_s, self.window_s, self.min_shared_s,
                  self.min_correlation, self.permutation_margin, self.min_lane_rms)
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("overlap settings must be finite numbers")
        if not 1 <= self.window_s <= 10 or not self.window_s <= self.buffer_s <= 20:
            raise ValueError("overlap window must be 1..10 seconds within a <=20 second buffer")
        if not 0 < self.min_shared_s <= self.window_s:
            raise ValueError("shared context must fit the separation window")
        if not 0 < self.min_correlation <= 1 or not 0 < self.permutation_margin <= 1:
            raise ValueError("correlation gates must lie in (0,1]")
        if not 0 < self.min_lane_rms < 1:
            raise ValueError("lane RMS floor must lie in (0,1)")


@dataclass(frozen=True)
class OverlapRequest:
    request_id: str
    source_track_id: str
    capture_epoch: str
    run_id: int
    window_start_ns: int
    window_end_ns: int
    owned_start_ns: int
    owned_end_ns: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SeparatedLane:
    lane_id: str
    track_id: str
    hypothesis: ASRHypothesis
    # Whole words intersecting the ownership edge. Never clipped or fabricated.
    # The parent transaction owner can preserve the original when these matter.
    boundary_words: tuple[TimedWord, ...] = ()


@dataclass(frozen=True)
class OverlapAnalysis:
    request: OverlapRequest
    lanes: tuple[SeparatedLane, SeparatedLane]
    separation_identity: str
    separation_runtime_ms: float
    processing_ms: float
    shared_gain: float
    continuity_reason: str
    continuity_scores: tuple[float, float]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class _Work:
    request: OverlapRequest
    samples: tuple[float, ...]
    retired: threading.Event


class SeparatedLaneContinuity:
    """Match two permutations on shared samples, resetting ambiguous lane state."""

    def __init__(self, config=None):
        self.config = config or OverlapStreamConfig()
        self._previous = None
        self._generation = 0

    @staticmethod
    def _correlation(first, second, rms_floor):
        count = len(first)
        mean_a, mean_b = math.fsum(first) / count, math.fsum(second) / count
        energy_a = math.fsum((value - mean_a) ** 2 for value in first)
        energy_b = math.fsum((value - mean_b) ** 2 for value in second)
        if min(energy_a, energy_b) < count * rms_floor ** 2:
            return None
        dot = math.fsum((left - mean_a) * (right - mean_b) for left, right in zip(first, second))
        return min(1.0, abs(dot / math.sqrt(energy_a * energy_b)))

    def align(self, sources, *, start_ns, valid_end_ns, scope):
        """Called only on the ordered worker; scope is (track, epoch, run)."""
        previous = self._previous
        order, scores, reason = (0, 1), (0.0, 0.0), "initial"
        accepted = False
        if previous is not None:
            old_scope, old_start, old_end, old_sources = previous
            shared_start, shared_end = max(start_ns, old_start), min(valid_end_ns, old_end)
            if scope != old_scope:
                reason = "run_changed"
            elif start_ns < old_start or valid_end_ns <= old_end:
                reason = "nonadvancing_window"
            elif ((start_ns - old_start) % SAMPLE_NS
                  or shared_end - shared_start < round(self.config.min_shared_s * 1e9)):
                reason = "insufficient_shared_audio"
            else:
                old_first = (shared_start - old_start) // SAMPLE_NS
                new_first = (shared_start - start_ns) // SAMPLE_NS
                count = (shared_end - shared_start) // SAMPLE_NS
                correlations = [[self._correlation(old_sources[left][old_first:old_first + count],
                    sources[right][new_first:new_first + count], self.config.min_lane_rms)
                    for right in range(2)] for left in range(2)]
                if any(value is None for row in correlations for value in row):
                    reason = "insufficient_lane_energy"
                else:
                    scores = ((correlations[0][0] + correlations[1][1]) / 2,
                              (correlations[0][1] + correlations[1][0]) / 2)
                    order = (0, 1) if scores[0] >= scores[1] else (1, 0)
                    if min(correlations[index][order[index]] for index in range(2)) < self.config.min_correlation:
                        reason = "low_correlation"
                    elif abs(scores[0] - scores[1]) < self.config.permutation_margin:
                        reason = "ambiguous_permutation"
                    else:
                        accepted, reason = True, "matched" if order == (0, 1) else "permuted"
        if not accepted:
            self._generation += 1
            order = (0, 1)
        aligned = tuple(sources[index] for index in order)
        self._previous = (scope, start_ns, valid_end_ns, tuple(array("f", source) for source in aligned))
        lane_ids = tuple(f"run-{scope[2]}:generation-{self._generation}:lane-{index}" for index in range(2))
        return aligned, lane_ids, reason, scores


class OverlapStream:
    """20-second PCM history, one active work item and one newest pending item.

    ``request`` snapshots real samples immediately; unavailable samples are never
    padded. ``run_native`` should be the session's NativeCallFence.run. Closing
    stops admission/results; cancelling an await does not stop native inference.
    The shared ASR scheduler owns running calls until the synchronous call ends.
    """

    def __init__(self, separator, transcriber, sink, *, config=None,
                 run_native=None, asr_scheduler=None):
        self.separator, self.transcriber, self.sink = separator, transcriber, sink
        self.config = config or OverlapStreamConfig()
        self.run_native = run_native or asyncio.to_thread
        self.asr_scheduler = asr_scheduler or AsrInferenceScheduler(run_native=self.run_native)
        self._owns_scheduler = asr_scheduler is None
        self._frames = deque()
        self._sample_count = 0
        self._last = None
        self._run_id = 0
        self._run_start_ns = None
        self._run_retired = threading.Event()
        self._pending = None
        self._task = None
        self._available = asyncio.Event()
        self._closed = self._finishing = False
        self._continuity = SeparatedLaneContinuity(self.config)
        self._notices = deque(maxlen=64)
        self.counts = {"overlap_requested": 0, "overlap_completed": 0, "overlap_skipped": 0,
                       "overlap_errors": 0, "overlap_asr_calls": 0, "overlap_notices_dropped": 0}

    @property
    def buffer_bounds(self):
        if not self._frames:
            return None
        start, _ = self._frames[0]
        return (self._last.track_id, self._last.capture_epoch, self._run_id,
                start, self._last.end_time_ns)

    @property
    def run_start_ns(self):
        return self._run_start_ns

    def _ensure_worker(self):
        self._available.set()
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    def _notice(self, reason, request=None):
        self.counts["overlap_skipped"] += 1
        notice = {"kind": "overlap.skipped", "reason": reason}
        if request is not None:
            notice.update(request_id=request.request_id, source_track_id=request.source_track_id,
                          capture_epoch=request.capture_epoch, run_id=request.run_id,
                          start_ns=request.owned_start_ns, end_ns=request.owned_end_ns)
        if len(self._notices) == self._notices.maxlen:
            self.counts["overlap_notices_dropped"] += 1
        self._notices.append(notice)
        self._ensure_worker()

    def feed(self, frame: AudioFrame):
        if self._closed or self._finishing:
            raise RuntimeError("cannot feed a closing overlap stream")
        previous = self._last
        if previous and (frame.track_id == previous.track_id and frame.capture_epoch == previous.capture_epoch
                         and frame.start_time_ns < previous.end_time_ns):
            raise ValueError("overlap PCM cannot rewind within a capture epoch")
        if previous and (frame.track_id != previous.track_id or frame.capture_epoch != previous.capture_epoch
                         or frame.sequence != previous.sequence + 1 or frame.start_time_ns != previous.end_time_ns):
            self.gap()
        if self._run_start_ns is None:
            self._run_start_ns = frame.start_time_ns
        self._frames.append((frame.start_time_ns, array("f", frame.samples)))
        self._sample_count += len(frame.samples)
        maximum = round(self.config.buffer_s * 16000)
        while self._sample_count > maximum:
            start, oldest = self._frames.popleft()
            removed = min(len(oldest), self._sample_count - maximum)
            if removed < len(oldest):
                self._frames.appendleft((start + removed * SAMPLE_NS, oldest[removed:]))
            self._sample_count -= removed
        self._last = frame

    def gap(self):
        self._run_retired.set()
        self._run_retired = threading.Event()
        if self._pending is not None:
            self._notice("run_changed", self._pending.request)
        self._pending = None
        self._frames.clear()
        self._sample_count = 0
        self._last = None
        self._run_start_ns = None
        self._run_id += 1
        # An old native call retains its old object; it cannot mutate this run.
        self._continuity = SeparatedLaneContinuity(self.config)

    def suggest_window(self, owned_start_ns, owned_end_ns):
        bounds = self.buffer_bounds
        if bounds is None:
            return None
        _, _, _, available_start, available_end = bounds
        if not available_start <= owned_start_ns < owned_end_ns <= available_end:
            return None
        first = (owned_start_ns - available_start) // SAMPLE_NS
        last = (owned_end_ns - available_start + SAMPLE_NS - 1) // SAMPLE_NS
        maximum = round(self.config.window_s * 16000)
        if last - first > maximum:
            return None
        total = (available_end - available_start) // SAMPLE_NS
        start = max(0, first - (maximum - (last - first)) // 2)
        start = max(0, min(start, total - maximum))
        end = min(total, start + maximum)
        if not start <= first < last <= end or end - start < 16000:
            return None
        return available_start + start * SAMPLE_NS, available_start + end * SAMPLE_NS

    def request(self, *, request_id, source_track_id, capture_epoch, window_start_ns,
                window_end_ns, owned_start_ns, owned_end_ns, run_id=None, metadata=None):
        if self._closed or self._finishing:
            raise RuntimeError("cannot request work on a closing overlap stream")
        if any(not isinstance(value, str) or not value or len(value) > 512
               for value in (request_id, source_track_id, capture_epoch)):
            raise ValueError("overlap request identifiers must contain 1..512 characters")
        times = (window_start_ns, window_end_ns, owned_start_ns, owned_end_ns)
        if any(type(value) is not int or value < 0 for value in times):
            raise ValueError("overlap request times must be nonnegative integer nanoseconds")
        if not window_start_ns <= owned_start_ns < owned_end_ns <= window_end_ns:
            raise ValueError("owned range must lie inside the separation window")
        if run_id is not None and (type(run_id) is not int or run_id < 0):
            raise ValueError("overlap run_id must be a nonnegative integer")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("overlap metadata must be a bounded JSON object")
        try:
            serialized = json.dumps(metadata or {}, allow_nan=False, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("overlap metadata must be a bounded JSON object") from exc
        if len(serialized.encode("utf-8")) > 32768:
            raise ValueError("overlap metadata exceeds 32768 bytes")
        copied_metadata = json.loads(serialized)
        deadline = copied_metadata.get("deadline_monotonic")
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("overlap deadline must be a finite monotonic timestamp")
        request = OverlapRequest(request_id, source_track_id, capture_epoch,
            self._run_id if run_id is None else run_id, *times, copied_metadata)
        bounds = self.buffer_bounds
        if bounds is None or bounds[:3] != (source_track_id, capture_epoch, request.run_id):
            self._notice("run_unavailable", request)
            return False
        available_start, available_end = bounds[3:]
        if window_start_ns < available_start or window_end_ns > available_end:
            self._notice("audio_evicted" if window_start_ns < available_start else "audio_not_yet_received", request)
            return False
        if ((window_start_ns - available_start) % SAMPLE_NS or (window_end_ns - available_start) % SAMPLE_NS
                or not 16000 <= (window_end_ns - window_start_ns) // SAMPLE_NS
                <= round(self.config.window_s * 16000)):
            raise ValueError("separation window must be sample-aligned and between 1 second and its maximum")
        samples = []
        for start, block in self._frames:
            left = max(0, (window_start_ns - start) // SAMPLE_NS)
            right = min(len(block), (window_end_ns - start) // SAMPLE_NS)
            if right > left:
                samples.extend(block[left:right])
        if len(samples) * SAMPLE_NS != window_end_ns - window_start_ns:
            raise RuntimeError("overlap PCM buffer is discontinuous")
        if self._pending is not None:
            self._notice("latest_window_replaced", self._pending.request)
        self._pending = _Work(request, tuple(samples), self._run_retired)
        self.counts["overlap_requested"] += 1
        self._ensure_worker()
        return True

    def _current(self, request):
        bounds = self.buffer_bounds
        return (not self._closed and bounds is not None
                and bounds[:3] == (request.source_track_id, request.capture_epoch, request.run_id))

    def _separate(self, work, continuity):
        request = work.request
        if work.retired.is_set():
            raise OverlapRunRetired("source run retired before separation admission")
        self._check_deadline(request)
        result = self.separator.separate(work.samples, sample_rate=16000,
                                         window_start_ns=request.window_start_ns)
        if (result.window_start_ns != request.window_start_ns or result.window_end_ns != request.window_end_ns
                or len(result.sources) != 2 or any(len(source) != len(work.samples) for source in result.sources)
                or type(result.unmodeled_tail_samples) is not int
                or not 0 <= result.unmodeled_tail_samples < 16
                or any(not math.isfinite(value) for source in result.sources for value in source)):
            raise ValueError("separation changed the source timeline or returned invalid samples")
        valid_end = request.window_end_ns - result.unmodeled_tail_samples * SAMPLE_NS
        if request.owned_end_ns > valid_end:
            raise ValueError("owned range includes unmodeled separation tail")
        peak = max(abs(value) for source in result.sources for value in source)
        gain = min(1.0, .999 / peak) if peak else 1.0
        sources = tuple(tuple(value * gain for value in source) for source in result.sources)
        owned_first = (request.owned_start_ns - request.window_start_ns + SAMPLE_NS - 1) // SAMPLE_NS
        owned_last = (request.owned_end_ns - request.window_start_ns) // SAMPLE_NS
        if owned_last <= owned_first:
            raise ValueError("owned overlap range contains no complete source samples")
        # This is only a silence/energy floor, not a speech or separation-quality
        # classifier. Do not ask ASR to invent a second stream from a silent lane.
        for source in sources:
            energy = math.fsum(value * value for value in source[owned_first:owned_last])
            if energy < (owned_last - owned_first) * self.config.min_lane_rms ** 2:
                raise ValueError("separated owned lane has insufficient signal energy")
        aligned, lane_ids, reason, scores = continuity.align(sources,
            start_ns=request.window_start_ns, valid_end_ns=valid_end,
            scope=(request.source_track_id, request.capture_epoch, request.run_id))
        return result, aligned, lane_ids, reason, scores, gain

    @staticmethod
    def _check_deadline(request):
        deadline = request.metadata.get("deadline_monotonic")
        if deadline is not None and time.monotonic() >= deadline:
            raise OverlapDeadlineExpired("overlap correction admission deadline expired")

    def _transcribe(self, request, source, retired):
        # Recheck after waiting for a primary-priority resource slot.
        if retired.is_set():
            raise OverlapRunRetired("source run retired before secondary ASR admission")
        self._check_deadline(request)
        return self.transcriber.transcribe_pcm(source, sample_rate=16000,
                                               window_start_ns=request.window_start_ns)

    async def _process(self, work):
        request = work.request
        if not self._current(request):
            self._notice("run_changed", request)
            return
        started = time.perf_counter_ns()
        result, sources, lane_ids, reason, scores, gain = await self.run_native(
            self._separate, work, self._continuity)
        lanes = []
        for index, source in enumerate(sources):
            if not self._current(request):
                self._notice("run_changed", request)
                return
            self._check_deadline(request)
            hypothesis = await self.asr_scheduler.run_secondary(self._transcribe, request, source, work.retired)
            self.counts["overlap_asr_calls"] += 1
            if (not isinstance(hypothesis, ASRHypothesis)
                    or hypothesis.window_start_ns != request.window_start_ns
                    or hypothesis.window_end_ns != request.window_end_ns
                    or len(hypothesis.words) > 2048):
                raise ValueError("separated ASR changed the source timeline")
            owned = tuple(word for word in hypothesis.words
                if request.owned_start_ns <= word.start_time_ns
                and word.end_time_ns <= request.owned_end_ns
                and word.start_time_ns < request.owned_end_ns)
            boundary = tuple(word for word in hypothesis.words
                if word not in owned and word.start_time_ns < request.owned_end_ns
                and word.end_time_ns > request.owned_start_ns)
            lanes.append(SeparatedLane(lane_ids[index], f"{request.source_track_id}:sep:{lane_ids[index]}",
                ASRHypothesis(hypothesis.window_start_ns, hypothesis.window_end_ns, owned, hypothesis.language), boundary))
        if not self._current(request):
            self._notice("run_changed", request)
            return
        self.counts["overlap_completed"] += 1
        diagnostics = tuple(getattr(result, "diagnostics", getattr(result, "model_diagnostics", ())))
        algorithm = getattr(result, "algorithm_identity", None)
        if algorithm is not None:
            diagnostics += (f"frequency_algorithm:{algorithm}",)
        await self.sink(OverlapAnalysis(request, tuple(lanes), self.separator.identity,
            result.runtime_ms, (time.perf_counter_ns() - started) / 1e6, gain, reason, scores,
            diagnostics + ("lane_correlation_is_not_identity_or_purity",)))

    async def _loop(self):
        while True:
            await self._available.wait()
            self._available.clear()
            while self._notices:
                await self.sink(self._notices.popleft())
            while self._pending is not None:
                work, self._pending = self._pending, None
                try:
                    await self._process(work)
                except asyncio.CancelledError:
                    raise
                except OverlapRunRetired:
                    self._notice("run_changed", work.request)
                except Exception as exc:
                    self.counts["overlap_errors"] += 1
                    await self.sink({"kind": "overlap.failed", "request_id": work.request.request_id,
                        "source_track_id": work.request.source_track_id, "capture_epoch": work.request.capture_epoch,
                        "run_id": work.request.run_id, "start_ns": work.request.owned_start_ns,
                        "end_ns": work.request.owned_end_ns, "error": type(exc).__name__})
            while self._notices:
                await self.sink(self._notices.popleft())
            if self._finishing:
                return

    def finish_input(self):
        if not self._closed and not self._finishing:
            self._finishing = True
            self._available.set()

    async def drain(self, timeout_s):
        self.finish_input()
        if self._task is None:
            return True
        done, _ = await asyncio.wait({self._task}, timeout=timeout_s)
        if not done:
            await self.close()
            return False
        await self._task
        if self._owns_scheduler:
            await self.asr_scheduler.close(wait=True)
        return True

    async def close(self):
        self._closed = True
        self._run_retired.set()
        self._pending = None
        self._frames.clear()
        self._sample_count = 0
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self._owns_scheduler:
            await self.asr_scheduler.close()
