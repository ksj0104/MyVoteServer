"""Bounded, independent speaker work; translation never awaits model inference.

Segmentation and embedding execute sequentially on one worker. Source PCM stays
on its original track/epoch timeline. This module imports no optional runtime.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Callable

from .audio import AudioFrame, SAMPLE_NS
from .speakers import Assignment, OnlineSpeakerTracker, TrackerConfig


@dataclass(frozen=True)
class SpeakerStreamConfig:
    window_s: float = 10
    first_window_s: float = 3
    hop_s: float = 2
    observation_s: float = 1.5
    min_confidence: float = .80
    max_clipped_fraction: float = .01

    def __post_init__(self):
        for value in (self.window_s, self.first_window_s, self.hop_s, self.observation_s,
                      self.min_confidence, self.max_clipped_fraction):
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("Speaker settings must be finite numbers")
        if not 1 <= self.first_window_s <= self.window_s <= 10 or not .5 <= self.hop_s <= self.window_s:
            raise ValueError("Use 1-10 second windows and a 0.5-window second hop")
        if not 1 <= self.observation_s <= 3 or not .5 <= self.min_confidence <= 1:
            raise ValueError("Use 1-3 second observations and confidence in [0.5, 1]")
        if not 0 <= self.max_clipped_fraction <= .1:
            raise ValueError("Clipping gate must be in [0, 0.1]")


@dataclass(frozen=True)
class SpeakerWindow:
    track_id: str
    capture_epoch: str
    start_ns: int
    samples: tuple[float, ...]
    final: bool = False
    run_id: int = 0
    run_start_ns: int | None = None

    @property
    def end_ns(self):
        return self.start_ns + len(self.samples) * SAMPLE_NS


@dataclass(frozen=True)
class SpeakerAnalysis:
    track_id: str
    capture_epoch: str
    start_ns: int
    end_ns: int
    assignments: tuple[Assignment, ...]
    identities: tuple[tuple[str, str], ...]
    model_identity: str
    segmentation_identity: str
    processing_ms: float
    errors: tuple[tuple[int, int, str], ...] = ()
    cleared_candidates: int = 0


class LocalSpeakerAnalyzer:
    """Model-predicted clean spans -> independent evidence -> open-set IDs.

    Class confidence gates evidence; it is not a calibrated acoustic quality or
    identity probability. Defaults require evaluation on the selected languages.
    Both local adapters are created lazily inside the first worker call.
    """
    def __init__(self, segmentation_factory: Callable, embedding_factory: Callable, *,
                 tracker: OnlineSpeakerTracker | None = None,
                 config: SpeakerStreamConfig | None = None,
                 expected_segmentation_identity: str | None = None,
                 expected_embedding_identity: str | None = None):
        self.segmentation_factory, self.embedding_factory = segmentation_factory, embedding_factory
        self.config = config or SpeakerStreamConfig()
        self.tracker = tracker or OnlineSpeakerTracker(TrackerConfig(min_quality=self.config.min_confidence))
        self._segmenter = self._embedder = None
        self._committed_end = 0
        self._last_track = None
        self._last_run = None
        self._sample_sequence = 0
        self.expected_segmentation_identity = expected_segmentation_identity
        self.expected_embedding_identity = expected_embedding_identity

    @classmethod
    def from_local_models(cls, segmentation_model, embedding_model, *, profile_path=None, **kwargs):
        from .segmentation import LocalPyannoteSegmentation
        from .embedding import LocalWeSpeakerEmbedding
        if profile_path is not None:
            from .speaker_profile import load_speaker_profile
            if kwargs:
                raise ValueError("A profile cannot be combined with separate speaker settings")
            profile = load_speaker_profile(profile_path)
            kwargs = dict(config=profile.stream, tracker=OnlineSpeakerTracker(profile.tracker),
                          expected_segmentation_identity=profile.segmentation_identity,
                          expected_embedding_identity=profile.embedding_identity)
        config = kwargs.get("config") or SpeakerStreamConfig()
        return cls(lambda: LocalPyannoteSegmentation(segmentation_model, min_confidence=config.min_confidence),
                   lambda: LocalWeSpeakerEmbedding(embedding_model), **kwargs)

    def analyze(self, window: SpeakerWindow) -> SpeakerAnalysis:
        started = time.perf_counter_ns()
        if self._last_track is not None and self._last_track != window.track_id:
            raise ValueError("One speaker analyzer supports one audio track")
        self._last_track = window.track_id
        run = (window.capture_epoch, window.run_id)
        cleared = self.tracker.clear_candidates() if self._last_run is not None and self._last_run != run else 0
        self._last_run = run
        if self._segmenter is None:
            self._segmenter = self.segmentation_factory()
            if (self.expected_segmentation_identity is not None
                    and self._segmenter.identity != self.expected_segmentation_identity):
                self._segmenter = None
                raise ValueError("Speaker profile segmentation model/frontend identity mismatch")
        result = self._segmenter.segment(window.samples, sample_rate=16000, window_start_ns=window.start_ns)
        if (result.window_start_ns != window.start_ns or result.window_end_ns != window.end_ns
                or not window.start_ns <= result.trusted_end_ns <= window.end_ns):
            raise ValueError("Speaker segmentation changed the source timeline")
        limit = window.end_ns if window.final else result.trusted_end_ns
        cursor = max(window.start_ns, self._committed_end)
        assignments = []
        errors = []
        embedding_failed = False

        def unknown(start, end, reason):
            if end <= start:
                return
            self._sample_sequence += 1
            assignments.append(Assignment(f"speaker-sample-{self._sample_sequence}", window.track_id,
                                           start, end, "unknown", reason=reason))

        # Only mark permanently uncovered audio within this actual capture run;
        # a replaced rolling window may still be covered by the next snapshot.
        run_start = window.run_start_ns if window.run_start_ns is not None else window.start_ns
        gap_start = max(run_start, self._committed_end)
        if gap_start < window.start_ns:
            unknown(gap_start, window.start_ns, "speaker_unanalyzed")

        for span in result.spans:
            if (span.start_ns < window.start_ns or span.end_ns > window.end_ns or span.end_ns <= span.start_ns):
                raise ValueError("Invalid speaker segmentation interval")
            begin, end = max(cursor, span.start_ns), min(limit, span.end_ns)
            if end <= begin:
                continue
            if begin > cursor:
                unknown(cursor, begin, "segmentation_uncovered")
            start_index = (begin - window.start_ns + SAMPLE_NS - 1) // SAMPLE_NS
            stop_index = (end - window.start_ns) // SAMPLE_NS
            first = window.start_ns + start_index * SAMPLE_NS
            last = window.start_ns + stop_index * SAMPLE_NS
            if first > begin:
                unknown(begin, first, "sample_boundary")
            clean = (span.trusted and len(span.active_speakers) == 1
                     and math.isfinite(span.confidence) and span.confidence >= self.config.min_confidence)
            if not clean or stop_index - start_index < 16000:
                unknown(first, last, span.reason if not clean else "insufficient_clean_speech")
            else:
                # Merge a short remainder into the preceding observation. Never
                # credit the same source sample again in overlapping model windows.
                desired = round(self.config.observation_s * 16000)
                full, remainder = divmod(stop_index - start_index, desired)
                count = max(1, full + (remainder >= 16000))
                for index in range(count):
                    left = start_index + (stop_index - start_index) * index // count
                    right = start_index + (stop_index - start_index) * (index + 1) // count
                    pcm = window.samples[left:right]
                    obs_start = window.start_ns + left * SAMPLE_NS
                    obs_end = window.start_ns + right * SAMPLE_NS
                    clipped = sum(abs(value) >= .999 for value in pcm) / len(pcm)
                    if clipped > self.config.max_clipped_fraction or not any(pcm):
                        unknown(obs_start, obs_end, "clipped_or_zero_pcm")
                        continue
                    if embedding_failed:
                        unknown(obs_start, obs_end, "embedding_unavailable")
                        continue
                    try:
                        if self._embedder is None:
                            self._embedder = self.embedding_factory()
                            if (self.expected_embedding_identity is not None
                                    and self._embedder.identity != self.expected_embedding_identity):
                                self._embedder = None
                                raise ValueError("Speaker profile embedding model/frontend identity mismatch")
                        self._sample_sequence += 1
                        observation = self._embedder.embed_observation(
                            pcm, sample_id=f"speaker-sample-{self._sample_sequence}", track_id=window.track_id,
                            start_time_ns=obs_start, speech_duration_ns=obs_end - obs_start,
                            overlap=False, quality=span.confidence, min_quality=self.config.min_confidence)
                        assignments.append(self.tracker.observe(observation))
                    except Exception as exc:
                        # A later observation must not erase an earlier new-ID
                        # confirmation already applied to the tracker in this batch.
                        embedding_failed = True
                        errors.append((obs_start, obs_end, type(exc).__name__))
                        unknown(obs_start, obs_end, "embedding_failed")
            if last < end:
                unknown(max(first, last), end, "sample_boundary")
            cursor = end
        if cursor < limit:
            unknown(cursor, limit, "segmentation_uncovered")
        self._committed_end = max(self._committed_end, limit)
        self.tracker.expire(self._committed_end)
        return SpeakerAnalysis(window.track_id, window.capture_epoch, window.start_ns, window.end_ns,
            tuple(assignments), tuple((item.speaker_id, item.name) for item in self.tracker.speakers),
            self._embedder.identity if self._embedder is not None else "not_loaded",
            result.model_identity, (time.perf_counter_ns() - started) / 1e6, tuple(errors), cleared)


class SpeakerStream:
    """One running and one newest pending snapshot, with explicit overload events.

    ``run_native`` can be a gateway NativeCallFence.run to keep cancelled model
    calls from overlapping later sessions. Default to_thread cannot stop native
    inference, so callers must not reuse that analyzer after cancellation.
    """
    def __init__(self, analyzer, sink, *, config=None, run_native=None):
        self.analyzer, self.sink = analyzer, sink
        self.config = config or SpeakerStreamConfig()
        self.run_native = run_native or asyncio.to_thread
        self._frames = deque()
        self._sample_count = 0
        self._last = None
        self._run_id = 0
        self._run_start_ns = None
        self._scheduled_end = None
        self._pending = None
        self._available = asyncio.Event()
        self._task = None
        self._finishing = self._closed = False
        self._notices = deque(maxlen=16)
        self.counts = {"speaker_windows": 0, "speaker_errors": 0, "speaker_skipped": 0,
                       "speaker_observations": 0, "speaker_new": 0, "speaker_unknown": 0}

    def _offer(self, *, final=False):
        if not self._frames:
            return
        first, last = self._frames[0], self._frames[-1]
        if not final and self._scheduled_end is not None and last.end_time_ns <= self._scheduled_end:
            return
        samples = tuple(value for frame in self._frames for value in frame.samples)
        window = SpeakerWindow(first.track_id, first.capture_epoch, first.start_time_ns, samples,
                               final, self._run_id, self._run_start_ns)
        if self._pending is not None:
            previous = self._pending
            if (previous.track_id, previous.capture_epoch, previous.start_ns, previous.end_ns) != (
                    window.track_id, window.capture_epoch, window.start_ns, window.end_ns):
                self.counts["speaker_skipped"] += 1
                self._notices.append({"kind": "speaker.skipped", "start_ns": previous.start_ns,
                                      "end_ns": previous.end_ns, "reason": "latest_window_replaced"})
        self._pending = window
        self._scheduled_end = window.end_ns
        self._available.set()
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    def feed(self, frame: AudioFrame):
        if self._finishing or self._closed:
            raise RuntimeError("Cannot feed a closing speaker stream")
        previous = self._last
        if previous and frame.start_time_ns < previous.end_time_ns:
            raise ValueError("Speaker input cannot rewind or overlap the previous frame")
        if previous and (frame.track_id != previous.track_id or frame.capture_epoch != previous.capture_epoch
                         or frame.start_time_ns != previous.end_time_ns or frame.sequence != previous.sequence + 1):
            self.gap()
        if self._run_start_ns is None:
            self._run_start_ns = frame.start_time_ns
        self._frames.append(frame)
        self._sample_count += len(frame.samples)
        maximum = round(self.config.window_s * 16000)
        while self._sample_count > maximum:
            oldest = self._frames.popleft()
            excess = self._sample_count - maximum
            if len(oldest.samples) > excess:
                kept = AudioFrame(oldest.track_id, oldest.capture_epoch, oldest.sequence,
                                  oldest.start_time_ns + excess * SAMPLE_NS, oldest.samples[excess:])
                self._frames.appendleft(kept)
                self._sample_count -= excess
                break
            self._sample_count -= len(oldest.samples)
        self._last = frame
        if ((self._scheduled_end is None and self._sample_count >= round(self.config.first_window_s * 16000))
                or (self._scheduled_end is not None and frame.end_time_ns - self._scheduled_end >= round(self.config.hop_s * 1e9))):
            self._offer()

    def gap(self):
        self._offer(final=True)
        self._frames.clear()
        self._sample_count = 0
        self._last = self._scheduled_end = None
        self._run_id += 1
        self._run_start_ns = None

    def finish_input(self):
        if not self._finishing and not self._closed:
            self._offer(final=True)
            self._finishing = True
            self._available.set()

    async def _loop(self):
        while True:
            await self._available.wait()
            self._available.clear()
            while self._notices:
                await self.sink(self._notices.popleft())
            while self._pending is not None:
                window, self._pending = self._pending, None
                try:
                    result = await self.run_native(self.analyzer.analyze, window)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.counts["speaker_errors"] += 1
                    await self.sink({"kind": "speaker.failed", "start_ns": window.start_ns,
                                     "end_ns": window.end_ns, "error": type(exc).__name__})
                    continue
                self.counts["speaker_windows"] += 1
                self.counts["speaker_errors"] += len(result.errors)
                self.counts["speaker_observations"] += sum(a.status != "unknown" for a in result.assignments)
                self.counts["speaker_new"] += sum(a.status == "new" for a in result.assignments)
                self.counts["speaker_unknown"] += sum(a.status == "unknown" for a in result.assignments)
                await self.sink(result)
            while self._notices:
                await self.sink(self._notices.popleft())
            if self._finishing:
                return

    async def drain(self, timeout_s):
        self.finish_input()
        if self._task is not None:
            done, _ = await asyncio.wait({self._task}, timeout=timeout_s)
            if not done:
                self.counts["speaker_errors"] += 1
                await self.sink({"kind": "speaker.failed", "error": "speaker_drain_timeout"})
                await self.close()
                return False
            await self._task
        return True

    async def close(self):
        self._closed = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._pending = None
        self._frames.clear()
        self._sample_count = 0
