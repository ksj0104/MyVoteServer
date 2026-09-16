"""Experimental recovery of short clean speech using bounded raw PCM splices.

The baseline analyzer still performs segmentation exactly once, handles its
existing long observations/tail recovery, and classifies all nonpool audio.
Only otherwise-discarded 200 ms..<1 s clean fragments may be held here. Every
held interval must be entirely clean under one CURRENT window-local head before
pooling; a head number is never carried across windows as a person identity.

Concatenating PCM creates artificial feature frames at splice boundaries and
changes CMN statistics. This is not equivalent to pyannote's feature-mask pooling
and does not establish speaker purity. Thresholds below are a fixed experiment,
not calibrated guarantees. No additional model or optional dependency is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

from .audio import SAMPLE_NS
from .speaker_stream import LocalSpeakerAnalyzer, SpeakerAnalysis
from .speakers import Assignment


POOLING_PARAMETERS = {
    "method": "raw_pcm_splice_v1",
    "min_fragment_ns": 200_000_000,
    "max_fragment_exclusive_ns": 1_000_000_000,
    "min_pool_ns": 1_000_000_000,
    "max_pool_ns": 3_000_000_000,
    "max_supports": 32,
    "max_window_ns": 10_000_000_000,
    "head_scope": "current_segmentation_window_only",
    "evidence_clock": "current_analysis_window_end",
}


@dataclass(frozen=True)
class _Held:
    capture_epoch: str
    run_id: int
    start_ns: int
    end_ns: int


class _RecordingSegmenter:
    def __init__(self, target, owner):
        self.target, self.owner = target, owner

    @property
    def identity(self):
        return self.target.identity

    def segment(self, *args, **kwargs):
        result = self.target.segment(*args, **kwargs)
        self.owner._latest_segmentation = result
        return result


class LocalSpeakerPoolingAnalyzer(LocalSpeakerAnalyzer):
    """Opt-in subclass; ordinary LocalSpeakerAnalyzer behavior is unchanged.

    The inherited from_local_models supports the existing profile and model
    identity checks. Work runs on the same ordered SpeakerStream native worker.
    Held state is at most 32 intervals, without PCM or persistent head labels.
    """

    def __init__(self, segmentation_factory, embedding_factory, **kwargs):
        self._latest_segmentation = None
        self._pending_fragments = []
        self._readiness_watermark = -1
        self._pool_window_end = -1
        self._pool_run = None
        self._pool_counts = {"held_fragments": 0, "pooled_observations": 0,
            "pooled_supports": 0, "pooled_speech_ns": 0, "retired_fragments": 0,
            "embedding_errors": 0}
        super().__init__(lambda: _RecordingSegmenter(segmentation_factory(), self),
                         embedding_factory, **kwargs)

    def pooling_diagnostics(self):
        return {**self._pool_counts, "pending_supports": len(self._pending_fragments),
            "pending_speech_ns": sum(item.end_ns - item.start_ns for item in self._pending_fragments),
            "parameters": dict(POOLING_PARAMETERS)}

    def runtime_configuration(self):
        configuration = super().runtime_configuration()
        configuration["short_speech_pooling"] = {"enabled": True, **self.pooling_diagnostics()}
        return configuration

    def _prepare_observation(self, observation, window):
        if observation.speech_duration_ns != observation.end_time_ns - observation.start_time_ns:
            raise ValueError("pooling mode requires explicit clean support for ordinary observations")
        self._readiness_watermark = max(self._readiness_watermark, window.end_ns)
        return replace(observation,
            support_intervals=((observation.start_time_ns, observation.end_time_ns),),
            evidence_ready_ns=window.end_ns)

    def _expire_tracker(self, source_ns):
        self.tracker.expire(max(source_ns, self._readiness_watermark))

    def _unknown(self, item, reason):
        self._sample_sequence += 1
        return Assignment(f"speaker-pool-retired-{self._sample_sequence}", self._last_track,
                          item.start_ns, item.end_ns, "unknown", reason=reason)

    def _clean_head(self, item, window, spans):
        """Require complete coverage, same CURRENT head, and valid original PCM."""
        limit = window.end_ns if window.final else self._latest_segmentation.trusted_end_ns
        if item.start_ns < window.start_ns or item.end_ns > limit:
            return None
        cursor, head, confidence = item.start_ns, None, 1.0
        for span in spans:
            first, last = max(item.start_ns, span.start_ns), min(item.end_ns, span.end_ns)
            if last <= first:
                continue
            if (first > cursor or not span.trusted or not math.isfinite(span.confidence)
                    or span.confidence < self.config.min_confidence or len(span.active_speakers) != 1):
                return None
            current_head = span.active_speakers[0]
            if head is not None and current_head != head:
                return None
            head, confidence, cursor = current_head, min(confidence, span.confidence), max(cursor, last)
        if cursor < item.end_ns or head is None:
            return None
        left = (item.start_ns - window.start_ns) // SAMPLE_NS
        right = (item.end_ns - window.start_ns) // SAMPLE_NS
        pcm = window.samples[left:right]
        if (len(pcm) * SAMPLE_NS != item.end_ns - item.start_ns or not pcm or not any(pcm)
                or sum(abs(value) >= .999 for value in pcm) / len(pcm) > self.config.max_clipped_fraction):
            return None
        return head, confidence

    def _fanout(self, assignment, supports):
        # One tracker observation is one vote, even if several caption spans
        # receive its answer. Emit the new-ID semantic event only once.
        return tuple(replace(assignment, sample_id=f"{assignment.sample_id}:part-{index}",
            start_time_ns=start, end_time_ns=end, support_intervals=(),
            status="existing" if index and assignment.status == "new" else assignment.status)
            for index, (start, end) in enumerate(supports))

    def _revalidation_failure_reason(self, item, spans):
        intersections = [span for span in spans
                         if span.start_ns < item.end_ns and span.end_ns > item.start_ns]
        if any(span.trusted and math.isfinite(span.confidence)
               and span.confidence >= self.config.min_confidence and len(span.active_speakers) >= 2
               for span in intersections):
            # Conservatively block the whole retained support; the exact current
            # detector intervals remain available separately in overlap_regions.
            return "predicted_overlap"
        if any(span.reason == "overlap_uncertain" for span in intersections):
            return "overlap_uncertain"
        return "speaker_pool_revalidation_failed"

    def analyze(self, window):
        started = time.perf_counter_ns()
        run = (window.capture_epoch, window.run_id)
        if window.end_ns - window.start_ns > POOLING_PARAMETERS["max_window_ns"]:
            raise ValueError("short speech pooling is bounded to a ten second source window")
        if run == self._pool_run and window.end_ns < self._pool_window_end:
            raise ValueError("short speech pooling cannot rewind its evidence clock")
        self._latest_segmentation = None
        result = super().analyze(window)
        segmentation = self._latest_segmentation
        if segmentation is None:
            raise RuntimeError("pooling requires the actual current segmentation result")
        self._pool_run, self._pool_window_end = run, window.end_ns
        spans = tuple(sorted(segmentation.spans, key=lambda span: (span.start_ns, span.end_ns)))
        discarded = list(result.discarded_tails)
        assignments = list(result.assignments)
        accepted_count = sum(item.status != "unknown" for item in result.assignments)
        errors = list(result.errors)
        eligible = []

        def retire(item, reason):
            self._pool_counts["retired_fragments"] += 1
            assignment = self._unknown(item, reason)
            if item.capture_epoch != window.capture_epoch or item.run_id != window.run_id:
                discarded.append((item.capture_epoch, assignment))
            else:
                assignments.append(assignment)

        # Heads from the previous window are deliberately absent from _Held.
        for item in self._pending_fragments:
            if (item.capture_epoch, item.run_id) != run:
                retire(item, "speaker_pool_run_changed")
                continue
            if item.start_ns < window.start_ns:
                retire(item, "speaker_pool_evicted")
                continue
            clean = self._clean_head(item, window, spans)
            if clean is None:
                retire(item, self._revalidation_failure_reason(item, spans))
                continue
            eligible.append((item, *clean))
        removed = set()
        for assignment in result.assignments:
            duration = assignment.end_time_ns - assignment.start_time_ns
            if (assignment.status != "unknown" or assignment.reason != "insufficient_clean_speech"
                    or not POOLING_PARAMETERS["min_fragment_ns"] <= duration
                    < POOLING_PARAMETERS["max_fragment_exclusive_ns"]):
                continue
            item = _Held(window.capture_epoch, window.run_id, assignment.start_time_ns, assignment.end_time_ns)
            if any(item.start_ns < held.end_ns and item.end_ns > held.start_ns for held, _, _ in eligible):
                raise RuntimeError("short pooling attempted to hold overlapping source support")
            clean = self._clean_head(item, window, spans)
            if clean is not None:
                eligible.append((item, *clean))
                removed.add(assignment.sample_id)
                self._pool_counts["held_fragments"] += 1
        assignments = [assignment for assignment in assignments if assignment.sample_id not in removed]
        self._pending_fragments = []
        by_head = {}
        for item, head, confidence in sorted(eligible, key=lambda value: value[0].start_ns):
            by_head.setdefault(head, []).append((item, confidence))
        pools = []
        for fragments in by_head.values():
            group, duration = [], 0
            for item, confidence in fragments:
                item_duration = item.end_ns - item.start_ns
                if group and (duration + item_duration > POOLING_PARAMETERS["max_pool_ns"]
                              or len(group) == POOLING_PARAMETERS["max_supports"]):
                    pools.append(group)
                    group, duration = [], 0
                group.append((item, confidence))
                duration += item_duration
            if group:
                pools.append(group)
        # Ready time is identical, but source order makes allocation reproducible.
        pools.sort(key=lambda group: (group[-1][0].end_ns, group[0][0].start_ns))
        for group in pools:
            supports = tuple((item.start_ns, item.end_ns) for item, _ in group)
            duration = sum(end - start for start, end in supports)
            if duration < POOLING_PARAMETERS["min_pool_ns"]:
                for item, _ in group:
                    if window.final:
                        retire(item, "speaker_pool_final_insufficient")
                    elif len(self._pending_fragments) >= POOLING_PARAMETERS["max_supports"]:
                        retire(item, "speaker_pool_capacity")
                    else:
                        self._pending_fragments.append(item)
                continue
            if errors:
                for item, _ in group:
                    retire(item, "embedding_unavailable")
                continue
            try:
                if self._embedder is None:
                    self._embedder = self.embedding_factory()
                    if (self.expected_embedding_identity is not None
                            and self._embedder.identity != self.expected_embedding_identity):
                        self._embedder = None
                        raise ValueError("Speaker profile embedding model/frontend identity mismatch")
                pcm = tuple(value for start, end in supports for value in window.samples[
                    (start - window.start_ns) // SAMPLE_NS:(end - window.start_ns) // SAMPLE_NS])
                self._sample_sequence += 1
                observation = self._embedder.embed_observation(pcm,
                    sample_id=f"speaker-pool-{self._sample_sequence}", track_id=window.track_id,
                    start_time_ns=supports[0][0], speech_duration_ns=duration,
                    overlap=False, quality=min(confidence for _, confidence in group),
                    min_quality=self.config.min_confidence)
                observation = replace(observation, start_time_ns=supports[0][0], end_time_ns=supports[-1][1],
                                      support_intervals=supports, evidence_ready_ns=window.end_ns)
                self._readiness_watermark = max(self._readiness_watermark, window.end_ns)
                assignment = self.tracker.observe(observation)
                accepted_count += int(assignment.status != "unknown")
                assignments.extend(self._fanout(assignment, supports))
                self._pool_counts["pooled_observations"] += 1
                self._pool_counts["pooled_supports"] += len(supports)
                self._pool_counts["pooled_speech_ns"] += duration
            except Exception as exc:
                errors.append((supports[0][0], supports[-1][1], type(exc).__name__))
                self._pool_counts["embedding_errors"] += 1
                for item, _ in group:
                    retire(item, "speaker_pool_embedding_failed")
        return replace(result, assignments=tuple(sorted(assignments,
            key=lambda assignment: (assignment.start_time_ns, assignment.end_time_ns, assignment.sample_id))),
            identities=tuple((item.speaker_id, item.name) for item in self.tracker.speakers),
            model_identity=self._embedder.identity if self._embedder is not None else "not_loaded",
            processing_ms=(time.perf_counter_ns() - started) / 1e6, errors=tuple(errors),
            discarded_tails=tuple(discarded), accepted_observation_count=accepted_count)

    def discard_pending_tail(self, reason):
        baseline = super().discard_pending_tail(reason)
        pending, self._pending_fragments = self._pending_fragments, []
        if not pending:
            return baseline
        self._pool_counts["retired_fragments"] += len(pending)
        assignments = tuple(self._unknown(item, reason) for item in pending)
        if baseline is not None:
            if any(item.capture_epoch != baseline.capture_epoch for item in pending):
                return replace(baseline, discarded_tails=baseline.discarded_tails +
                               tuple((item.capture_epoch, assignment) for item, assignment in zip(pending, assignments)))
            return replace(baseline, assignments=baseline.assignments + assignments,
                           start_ns=min(baseline.start_ns, min(item.start_ns for item in pending)),
                           end_ns=max(baseline.end_ns, max(item.end_ns for item in pending)))
        return SpeakerAnalysis(self._last_track, pending[0].capture_epoch,
            min(item.start_ns for item in pending), max(item.end_ns for item in pending), assignments, (),
            self._embedder.identity if self._embedder is not None else "not_loaded",
            self._segmenter.identity if self._segmenter is not None else "not_loaded", 0.0,
            run_id=pending[0].run_id, final=True)
