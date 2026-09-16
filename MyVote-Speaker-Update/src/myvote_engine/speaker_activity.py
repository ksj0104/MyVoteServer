"""Experimental separation of identity evidence from current-window activity.

A local segmentation head gets at most one fresh, contiguous embedding in a
window. Its tracker result may describe other clean activity of that head in
that same window. Activity is presentation data, never additional tracker
evidence. This assumes a local head is internally consistent; model errors can
therefore propagate to more activity, and this policy is deliberately opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

from .audio import SAMPLE_NS
from .segmentation import (FRAME_SHIFT_SAMPLES, OUTPUT_FRAMES, POWERSET,
                           RECEPTIVE_FIELD_SAMPLES)
from .speaker_stream import LocalSpeakerAnalyzer, SpeakerAnalysis, SpeakerWindow
from .speakers import Assignment, Observation


ACTIVITY_PARAMETERS = {
    "method": "window_head_activity_v1",
    "min_support_ns": 1_000_000_000,
    "max_support_ns": 3_000_000_000,
    "max_overlap_probability": .1,
    "max_heads": 3,
    "support_selection": "longest_contiguous_then_earliest",
    "pending_tail": False,
}


@dataclass(frozen=True)
class HeadResolution:
    """Exact learning evidence/result, scoped to one inference window only."""
    resolution_id: str
    track_id: str
    capture_epoch: str
    run_id: int
    window_start_ns: int
    window_end_ns: int
    local_head: int
    observation: Observation
    tracker_assignment: Assignment


@dataclass(frozen=True)
class DiarizationActivity:
    """A real clean activity interval; no duration or vote is credited here."""
    start_ns: int
    end_ns: int
    local_head: int
    resolution_id: str | None


@dataclass(frozen=True)
class _Cell:
    start_ns: int
    end_ns: int
    local_head: int | None
    confidence: float
    reason: str


class LocalSpeakerActivityAnalyzer(LocalSpeakerAnalyzer):
    def __init__(self, *args, activity_policy="trusted", **kwargs):
        if activity_policy not in ("trusted", "singleton"):
            raise ValueError("activity_policy must be trusted or singleton")
        super().__init__(*args, **kwargs)
        self.activity_policy = activity_policy
        self.last_resolutions: tuple[HeadResolution, ...] = ()
        self.last_activities: tuple[DiarizationActivity, ...] = ()
        self._activity_window_end = -1

    @classmethod
    def from_local_models(cls, segmentation_model, embedding_model, *,
                          profile_path=None, activity_policy="trusted", **kwargs):
        # Keep the inherited profile validation and identities. Retained raw
        # frames change only diagnostics, not the segmentation frontend identity.
        analyzer = super().from_local_models(segmentation_model, embedding_model,
                                              profile_path=profile_path, **kwargs)
        if activity_policy not in ("trusted", "singleton"):
            raise ValueError("activity_policy must be trusted or singleton")
        analyzer.activity_policy = activity_policy
        if activity_policy == "singleton":
            from .segmentation import LocalPyannoteSegmentation
            analyzer.segmentation_factory = lambda: LocalPyannoteSegmentation(
                segmentation_model, min_confidence=analyzer.config.min_confidence,
                retain_frames=True)
        return analyzer

    def runtime_configuration(self):
        result = super().runtime_configuration()
        result["speaker_activity"] = {"enabled": True,
            "activity_policy": self.activity_policy, "parameters": dict(ACTIVITY_PARAMETERS)}
        return result

    @staticmethod
    def _valid_heads(heads):
        return (isinstance(heads, tuple) and all(type(head) is int for head in heads)
                and heads in POWERSET)

    def _trusted_cells(self, result):
        cells = []
        for span in result.spans:
            trusted = span.trusted and span.confidence >= self.config.min_confidence
            reason = span.reason
            head = None
            if trusted and len(span.active_speakers) >= 2:
                reason = "predicted_overlap"
            elif trusted and not span.active_speakers:
                reason = "predicted_nonspeech"
            elif trusted and len(span.active_speakers) == 1:
                if (span.overlap_probability is not None
                        and span.overlap_probability <= ACTIVITY_PARAMETERS["max_overlap_probability"]):
                    head = span.active_speakers[0]
                    reason = "prediction"
                else:
                    reason = "overlap_uncertain"
            elif reason in ("predicted_nonspeech", "predicted_overlap"):
                reason = "untrusted_prediction"
            cells.append(_Cell(span.start_ns, span.end_ns, head, span.confidence, reason))
        return cells

    def _singleton_cells(self, result):
        frames = getattr(result, "frames", ())
        if not isinstance(frames, tuple) or len(frames) > OUTPUT_FRAMES:
            raise ValueError("Expected bounded immutable raw segmentation frames")
        if not frames and result.trusted_start_ns < result.trusted_end_ns:
            raise ValueError("singleton activity requires retained original powerset frames")
        cells = [_Cell(result.window_start_ns, result.trusted_start_ns,
                       None, 0., "edge_context")]
        previous_end = result.trusted_start_ns
        previous_index = None
        cell_offset = (RECEPTIVE_FIELD_SAMPLES - FRAME_SHIFT_SAMPLES + 1) // 2
        span_index = 0
        for frame in frames:
            probs = frame.probabilities
            if (type(frame.index) is not int or not 0 <= frame.index < OUTPUT_FRAMES
                    or previous_index is not None and frame.index != previous_index + 1
                    or type(frame.start_ns) is not int or type(frame.end_ns) is not int
                    or frame.start_ns != previous_end
                    or frame.start_ns != result.window_start_ns + (cell_offset + frame.index * FRAME_SHIFT_SAMPLES) * SAMPLE_NS
                    or frame.end_ns != frame.start_ns + FRAME_SHIFT_SAMPLES * SAMPLE_NS
                    or not frame.start_ns < frame.end_ns <= result.trusted_end_ns
                    or not isinstance(probs, tuple) or len(probs) != 7
                    or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                           for p in probs)
                    or not math.isclose(math.fsum(probs), 1., abs_tol=1e-6)
                    or not self._valid_heads(frame.active_speakers)):
                raise ValueError("Invalid retained powerset frame geometry/probabilities")
            winner = max(range(7), key=probs.__getitem__)
            if (frame.active_speakers != POWERSET[winner]
                    or not math.isclose(frame.confidence, probs[winner], abs_tol=1e-7)
                    or not math.isclose(frame.overlap_probability, math.fsum(probs[4:]), abs_tol=1e-7)):
                raise ValueError("Retained frame differs from original powerset probabilities")
            while span_index < len(result.spans) and result.spans[span_index].end_ns <= frame.start_ns:
                span_index += 1
            if span_index < len(result.spans):
                span = result.spans[span_index]
                if (span.trusted and span.start_ns < frame.end_ns
                        and (span.active_speakers != frame.active_speakers
                             or span.start_ns > frame.start_ns or span.end_ns < frame.end_ns
                             or span.confidence > frame.confidence + 1e-7)):
                    raise ValueError("Trusted support differs from original powerset head")
            head = None
            if len(frame.active_speakers) >= 2:
                # A pair remains blocking even when its class score is low.
                reason = "predicted_overlap"
            elif frame.overlap_probability > ACTIVITY_PARAMETERS["max_overlap_probability"]:
                reason = "overlap_uncertain"
            elif len(frame.active_speakers) == 1:
                head = frame.active_speakers[0]
                # Keep trusted support separate from broader lower-confidence
                # activity so extra zero/clipped PCM cannot erase its evidence
                # presentation or turn the whole extended span into learning.
                reason = "prediction" if frame.confidence >= self.config.min_confidence else "singleton_activity"
            elif frame.confidence >= self.config.min_confidence:
                reason = "predicted_nonspeech"
            else:
                reason = "low_confidence"
            cell = _Cell(frame.start_ns, frame.end_ns, head, frame.confidence, reason)
            if (cells and cells[-1].end_ns == cell.start_ns
                    and (cells[-1].local_head, cells[-1].reason) == (cell.local_head, cell.reason)):
                previous = cells[-1]
                cells[-1] = replace(previous, end_ns=cell.end_ns,
                                    confidence=min(previous.confidence, cell.confidence))
            else:
                cells.append(cell)
            previous_end, previous_index = frame.end_ns, frame.index
        if previous_end != result.trusted_end_ns:
            raise ValueError("Retained raw frames must cover the guarded timeline exactly")
        cells.append(_Cell(result.trusted_end_ns, result.window_end_ns, None, 0.,
                           "padding_edge" if result.padded_samples else "edge_context"))
        return cells

    def _partition(self, cells, window, cursor, limit):
        """One source disposition per nanosecond; PCM claims are sample-aligned."""
        result = []
        for cell in cells:
            begin, end = max(cursor, cell.start_ns), min(limit, cell.end_ns)
            if end <= begin:
                continue
            if begin > cursor:
                result.append(_Cell(cursor, begin, None, 0., "segmentation_uncovered"))
            if cell.local_head is None:
                result.append(replace(cell, start_ns=begin, end_ns=end))
            else:
                left = (begin - window.start_ns + SAMPLE_NS - 1) // SAMPLE_NS
                right = (end - window.start_ns) // SAMPLE_NS
                first, last = window.start_ns + left * SAMPLE_NS, window.start_ns + right * SAMPLE_NS
                if first >= last:
                    result.append(_Cell(begin, end, None, 0., "sample_boundary"))
                else:
                    if begin < first:
                        result.append(_Cell(begin, first, None, 0., "sample_boundary"))
                    pcm = window.samples[left:right]
                    clipped = sum(abs(value) >= .999 for value in pcm) / len(pcm)
                    if clipped > self.config.max_clipped_fraction or not any(pcm):
                        result.append(_Cell(first, last, None, 0., "clipped_or_zero_pcm"))
                    else:
                        result.append(replace(cell, start_ns=first, end_ns=last))
                    if last < end:
                        result.append(_Cell(last, end, None, 0., "sample_boundary"))
            cursor = end
        if cursor < limit:
            result.append(_Cell(cursor, limit, None, 0., "segmentation_uncovered"))
        return result

    def analyze(self, window: SpeakerWindow) -> SpeakerAnalysis:
        started = time.perf_counter_ns()
        if (not isinstance(window, SpeakerWindow) or type(window.start_ns) is not int
                or window.start_ns < 0 or not 1 <= len(window.samples) <= 160000
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or not -1 <= value <= 1 for value in window.samples)):
            raise ValueError("Speaker activity requires a bounded normalized mono16k window")
        if self._last_track is not None and self._last_track != window.track_id:
            raise ValueError("One speaker analyzer supports one audio track")
        if window.end_ns < self._activity_window_end:
            raise ValueError("Cannot rewind a speaker activity window")
        if self._segmenter is None:
            self._segmenter = self.segmentation_factory()
            if (self.expected_segmentation_identity is not None
                    and self._segmenter.identity != self.expected_segmentation_identity):
                self._segmenter = None
                raise ValueError("Speaker profile segmentation model/frontend identity mismatch")
        result = self._segmenter.segment(window.samples, sample_rate=16000,
                                          window_start_ns=window.start_ns)
        if (result.window_start_ns != window.start_ns or result.window_end_ns != window.end_ns
                or not window.start_ns <= result.trusted_start_ns <= result.trusted_end_ns <= window.end_ns
                or not isinstance(result.spans, tuple) or len(result.spans) > OUTPUT_FRAMES + 2):
            raise ValueError("Speaker segmentation changed the source timeline")
        previous_end = window.start_ns
        for span in result.spans:
            if (type(span.start_ns) is not int or type(span.end_ns) is not int
                    or not previous_end <= span.start_ns < span.end_ns <= window.end_ns
                    or not self._valid_heads(span.active_speakers)
                    or type(span.trusted) is not bool or not math.isfinite(span.confidence)
                    or not 0 <= span.confidence <= 1
                    or span.overlap_probability is not None and
                    (not math.isfinite(span.overlap_probability) or not 0 <= span.overlap_probability <= 1)):
                raise ValueError("Invalid speaker segmentation interval")
            if span.trusted and not result.trusted_start_ns <= span.start_ns < span.end_ns <= result.trusted_end_ns:
                raise ValueError("Trusted segmentation activity exceeds guarded source")
            previous_end = span.end_ns
        limit = window.end_ns if window.final else result.trusted_end_ns
        cursor = max(window.start_ns, self._committed_end)
        strict = self._trusted_cells(result)
        learning = self._partition(strict, window, cursor, limit)
        display = learning if self.activity_policy == "trusted" else self._partition(
            self._singleton_cells(result), window, cursor, limit)

        # Validate models and all partitions before changing run/candidate state.
        run = (window.capture_epoch, window.run_id)
        cleared = self.tracker.clear_candidates() if self._last_run is not None and self._last_run != run else 0
        self._last_track, self._last_run = window.track_id, run
        resolutions, failures, errors = {}, {}, []
        chosen = {}
        for cell in learning:
            if cell.local_head is None or cell.end_ns - cell.start_ns < ACTIVITY_PARAMETERS["min_support_ns"]:
                continue
            old = chosen.get(cell.local_head)
            if old is None or cell.end_ns - cell.start_ns > old.end_ns - old.start_ns:
                chosen[cell.local_head] = cell
        embedding_failed = False
        accepted_count = 0
        for head, cell in sorted(chosen.items(), key=lambda item: item[1].start_ns):
            first = cell.start_ns
            last = min(cell.end_ns, first + ACTIVITY_PARAMETERS["max_support_ns"])
            pcm = window.samples[(first - window.start_ns) // SAMPLE_NS:(last - window.start_ns) // SAMPLE_NS]
            if sum(abs(value) >= .999 for value in pcm) / len(pcm) > self.config.max_clipped_fraction or not any(pcm):
                failures[head] = "clipped_or_zero_pcm"
                continue
            if embedding_failed:
                failures[head] = "embedding_unavailable"
                continue
            try:
                if self._embedder is None:
                    self._embedder = self.embedding_factory()
                    if (self.expected_embedding_identity is not None
                            and self._embedder.identity != self.expected_embedding_identity):
                        self._embedder = None
                        raise ValueError("Speaker profile embedding model/frontend identity mismatch")
                self._sample_sequence += 1
                sample_id = f"speaker-evidence-{self._sample_sequence}"
                observation = self._embedder.embed_observation(pcm, sample_id=sample_id,
                    track_id=window.track_id, start_time_ns=first,
                    speech_duration_ns=last - first, overlap=False,
                    quality=cell.confidence, min_quality=self.config.min_confidence)
                if (not isinstance(observation, Observation) or observation.sample_id != sample_id
                        or observation.track_id != window.track_id
                        or observation.start_time_ns != first or observation.end_time_ns != last
                        or observation.speech_duration_ns != last - first or observation.overlap):
                    raise ValueError("Embedding changed the source evidence interval")
                observation = replace(observation, support_intervals=((first, last),),
                                      evidence_ready_ns=window.end_ns)
                assignment = self.tracker.observe(observation)
                resolution = HeadResolution(sample_id, window.track_id,
                    window.capture_epoch, window.run_id, window.start_ns, window.end_ns, head,
                    observation, assignment)
                resolutions[head] = resolution
                accepted_count += int(assignment.status != "unknown")
            except Exception as exc:
                # Previous successful tracker updates and their activity survive.
                embedding_failed = True
                failures[head] = "embedding_failed"
                errors.append((first, last, type(exc).__name__))

        assignments, activities, published, links = [], [], set(), []
        run_start = window.run_start_ns if window.run_start_ns is not None else window.start_ns
        gap_start = max(run_start, self._committed_end)
        if gap_start < window.start_ns:
            display.insert(0, _Cell(gap_start, window.start_ns, None, 0., "speaker_unanalyzed"))
        for cell in display:
            self._sample_sequence += 1
            sample_id = f"speaker-activity-{self._sample_sequence}"
            resolution = resolutions.get(cell.local_head)
            if cell.local_head is not None:
                activities.append(DiarizationActivity(cell.start_ns, cell.end_ns, cell.local_head,
                                                       resolution.resolution_id if resolution else None))
            if resolution is None:
                reason = cell.reason if cell.local_head is None else failures.get(cell.local_head, "insufficient_clean_speech")
                assignment = Assignment(sample_id, window.track_id, cell.start_ns, cell.end_ns,
                                        "unknown", reason=reason)
            else:
                original = resolution.tracker_assignment
                status = "existing" if original.status == "new" and resolution.resolution_id in published else original.status
                assignment = Assignment(sample_id, window.track_id, cell.start_ns, cell.end_ns,
                    status, original.speaker_id, original.candidate_id, original.similarity, original.reason)
                published.add(resolution.resolution_id)
                links.append((sample_id, resolution.resolution_id))
            assignments.append(assignment)
        overlaps = []
        for cell in strict:
            if cell.reason != "predicted_overlap":
                continue
            first = window.start_ns + ((max(window.start_ns, cell.start_ns) - window.start_ns + SAMPLE_NS - 1) // SAMPLE_NS) * SAMPLE_NS
            last = window.start_ns + ((min(limit, cell.end_ns) - window.start_ns) // SAMPLE_NS) * SAMPLE_NS
            if last > first:
                if overlaps and overlaps[-1][1] == first:
                    overlaps[-1] = (overlaps[-1][0], last)
                else:
                    overlaps.append((first, last))
        self._committed_end = max(self._committed_end, limit)
        self._activity_window_end = window.end_ns
        self.tracker.expire(window.end_ns)
        self.last_resolutions, self.last_activities = tuple(resolutions.values()), tuple(activities)
        return SpeakerAnalysis(window.track_id, window.capture_epoch, window.start_ns, window.end_ns,
            tuple(assignments), tuple((item.speaker_id, item.name) for item in self.tracker.speakers),
            self._embedder.identity if self._embedder is not None else "not_loaded", result.model_identity,
            (time.perf_counter_ns() - started) / 1e6, tuple(errors), cleared, (), tuple(overlaps),
            window.run_id, window.run_start_ns, window.final, accepted_count,
            assignment_origin="window_activity", activity_links=tuple(links),
            identity_supports=tuple(item.tracker_assignment for item in resolutions.values()))
