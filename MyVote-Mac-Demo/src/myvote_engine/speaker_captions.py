"""Map supplied speaker assignments onto caption intervals without inference.

This reducer runs on the host's ordered event path. It never extracts a voice,
votes on a new identity, delays translation, or calls a model. All decisions use
the source audio timeline. Coverage is a duration fraction, not a probability.
Synthetic Assignment tests demonstrate mapping contracts, not voice accuracy.
"""

from __future__ import annotations

from collections import OrderedDict
from bisect import bisect_left
from dataclasses import dataclass
import math

from .captions import CaptionStore
from .speakers import Assignment


NS = 1_000_000_000


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"Invalid {name}")
    value.encode("utf-8")


@dataclass(frozen=True)
class SpeakerMappingConfig:
    min_coverage: float = .8
    max_unknown_fraction: float = 0.0
    history_ns: int = 120 * NS
    max_history: int = 4096
    max_captions: int = 1024

    def __post_init__(self):
        for name in ("min_coverage", "max_unknown_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite fraction")
        if self.min_coverage <= .5:
            raise ValueError("min_coverage must exceed one half of the caption interval")
        for name in ("history_ns", "max_history", "max_captions"):
            _integer(getattr(self, name), name, 1)


@dataclass(frozen=True)
class SpeakerPatch:
    segment_id: str
    speaker_id: str | None
    speaker_revision: int
    based_on_source_revision: int
    track_id: str
    capture_epoch: str
    start_ns: int
    end_ns: int
    reason: str
    coverage: float
    unknown_fraction: float

    def data(self) -> dict:
        """Payload for ``speaker.updated``; segment_id is the event envelope ID."""
        return {"speaker_id": self.speaker_id, "speaker_revision": self.speaker_revision,
                "based_on_source_revision": self.based_on_source_revision,
                "track_id": self.track_id, "capture_epoch": self.capture_epoch,
                "start_ns": self.start_ns, "end_ns": self.end_ns,
                "reason": self.reason, "coverage": self.coverage,
                "unknown_fraction": self.unknown_fraction}


@dataclass(frozen=True)
class _CaptionRef:
    segment_id: str
    track_id: str
    capture_epoch: str
    start_ns: int
    end_ns: int
    source_revision: int


@dataclass(frozen=True)
class _Evidence:
    assignment: Assignment
    capture_epoch: str


@dataclass(frozen=True)
class _Decision:
    speaker_id: str | None
    reason: str
    coverage: float
    unknown_fraction: float


class SpeakerCaptionReducer:
    """Bounded recent speaker evidence and caption references for one session.

    Register a caption after its source update is accepted by ``store``. Calls
    return only patches already accepted by ``store.update_speaker``; emit those
    on the independent speaker event path. The source and translation revisions
    never change. Local manual assignments and aliases remain authoritative.

    Explicit unknown/overlap intervals block automatic naming by default. Any
    second confirmed speaker overlapping a caption blocks naming, even when one
    speaker occupies most of the time. Unobserved edges instead reduce coverage.
    Repeated/overlapping windows contribute their time union, never extra votes.
    New evidence rechecks only intersecting captions, including earlier candidate
    intervals on confirmation. It never copies the session export or journal.

    Retention/capacity eviction is counted in ``stats``. Evicted evidence raises
    a timeline floor; captions crossing that floor are no longer reclassified
    from incomplete history. Historical stored names remain as last accepted.
    This is not a persistent diarization database or a session-wide reanalysis.
    """

    def __init__(self, store: CaptionStore, config: SpeakerMappingConfig | None = None):
        if not isinstance(store, CaptionStore):
            raise ValueError("A CaptionStore for this session is required")
        self.store = store
        self.config = config or SpeakerMappingConfig()
        self._history: OrderedDict[tuple[str, str, str], _Evidence] = OrderedDict()
        self._captions: OrderedDict[str, _CaptionRef] = OrderedDict()
        self._confirmed_candidates: dict[tuple[str, str, str], str] = {}
        self._watermark_ns = 0
        self._retention_floor_ns = 0
        self.stats = dict(assignments=0, duplicate_assignments=0, ignored_assignments=0,
                          expired_assignments=0, history_evictions=0, caption_evictions=0,
                          expired_captions=0, stale_captions=0, manual_preserved=0,
                          candidate_confirmations=0, speaker_patches=0, caption_rechecks=0)

    @property
    def retained_assignments(self) -> int:
        return len(self._history)

    @property
    def pending_captions(self) -> int:
        return len(self._captions)

    @property
    def retained_candidate_resolutions(self) -> int:
        return len(self._confirmed_candidates)

    @property
    def retention_floor_ns(self) -> int:
        return self._retention_floor_ns

    def _prune(self) -> None:
        floor = max(self._retention_floor_ns, self._watermark_ns - self.config.history_ns, 0)
        expired = [key for key, entry in self._history.items()
                   if entry.assignment.end_time_ns <= floor]
        for key in expired:
            del self._history[key]
            self.stats["expired_assignments"] += 1
        while len(self._history) > self.config.max_history:
            oldest = min(self._history, key=lambda key: self._history[key].assignment.end_time_ns)
            floor = max(floor, self._history.pop(oldest).assignment.end_time_ns)
            self.stats["history_evictions"] += 1
        self._retention_floor_ns = floor
        for key in tuple(self._captions):
            if self._captions[key].start_ns < floor:
                del self._captions[key]
                self.stats["expired_captions"] += 1
        candidates = {(item.assignment.track_id, item.capture_epoch, item.assignment.candidate_id)
                      for item in self._history.values() if item.assignment.candidate_id is not None}
        self._confirmed_candidates = {key: value for key, value in self._confirmed_candidates.items()
                                      if key in candidates}

    def expire(self, now_ns: int) -> None:
        _integer(now_ns, "now_ns")
        if now_ns < self._watermark_ns:
            raise ValueError("Expiry clock cannot move backwards")
        self._watermark_ns = now_ns
        self._prune()

    def register_caption(self, segment_id: str, *, track_id: str, capture_epoch: str,
                         start_ns: int, end_ns: int, source_revision: int) -> tuple[SpeakerPatch, ...]:
        for name, value in (("segment_id", segment_id), ("track_id", track_id),
                            ("capture_epoch", capture_epoch)):
            _identifier(value, name)
        _integer(start_ns, "start_ns")
        _integer(end_ns, "end_ns", start_ns + 1)
        _integer(source_revision, "source_revision", 1)
        current = self.store.get_segment(segment_id)
        if current is None or current.superseded_by:
            raise ValueError("Register an existing active source caption")
        if (current.track_id, current.start_ns, current.end_ns, current.source_revision) != (
                track_id, start_ns, end_ns, source_revision):
            raise ValueError("Caption registration differs from the current source revision/timing")
        previous = self._captions.get(segment_id)
        if previous is not None and previous.capture_epoch != capture_epoch:
            raise ValueError("A registered caption cannot move to another capture epoch")
        self._watermark_ns = max(self._watermark_ns, end_ns)
        self._prune()
        if start_ns < self._retention_floor_ns:
            self.stats["expired_captions"] += 1
            return ()
        if segment_id not in self._captions and len(self._captions) >= self.config.max_captions:
            oldest = min(self._captions, key=lambda key: self._captions[key].end_ns)
            del self._captions[oldest]
            self.stats["caption_evictions"] += 1
        reference = _CaptionRef(segment_id, track_id, capture_epoch, start_ns, end_ns, source_revision)
        self._captions[segment_id] = reference
        return self._patch((reference,))

    @staticmethod
    def _validate_assignment(item: Assignment, capture_epoch: str) -> None:
        if not isinstance(item, Assignment):
            raise ValueError("Expected a tracker Assignment")
        _identifier(capture_epoch, "capture_epoch")
        for name, value in (("sample_id", item.sample_id), ("track_id", item.track_id)):
            _identifier(value, name)
        _integer(item.start_time_ns, "assignment start_time_ns")
        _integer(item.end_time_ns, "assignment end_time_ns", item.start_time_ns + 1)
        if item.status not in ("unknown", "provisional", "existing", "new"):
            raise ValueError("Invalid assignment status")
        if item.speaker_id is not None:
            _identifier(item.speaker_id, "speaker_id")
        if item.candidate_id is not None:
            _identifier(item.candidate_id, "candidate_id")
        if item.status in ("existing", "new") and item.speaker_id is None:
            raise ValueError("Confirmed assignment requires a speaker ID")
        if item.status in ("unknown", "provisional") and item.speaker_id is not None:
            raise ValueError("Unconfirmed assignment cannot name a speaker")
        if item.status in ("provisional", "new") and item.candidate_id is None:
            raise ValueError("Candidate assignment requires a candidate ID")
        if item.status == "unknown" and item.candidate_id is not None:
            raise ValueError("Unknown assignment cannot identify a candidate")
        if item.similarity is not None and (isinstance(item.similarity, bool)
                or not math.isfinite(item.similarity) or not -1 <= item.similarity <= 1):
            raise ValueError("Invalid assignment similarity")
        if not isinstance(item.reason, str) or len(item.reason) > 1024:
            raise ValueError("Invalid assignment reason")

    def observe(self, assignment: Assignment, *, capture_epoch: str) -> tuple[SpeakerPatch, ...]:
        self._validate_assignment(assignment, capture_epoch)
        if assignment.reason in ("duplicate_sample", "out_of_order"):
            self.stats["ignored_assignments"] += 1
            return ()
        key = (assignment.track_id, capture_epoch, assignment.sample_id)
        previous = self._history.get(key)
        if previous is not None:
            if previous.assignment != assignment:
                raise ValueError("A speaker sample ID was reused with different evidence")
            self.stats["duplicate_assignments"] += 1
            return ()
        if assignment.end_time_ns <= self._retention_floor_ns:
            self.stats["expired_assignments"] += 1
            return ()
        candidate_key = (assignment.track_id, capture_epoch, assignment.candidate_id)
        if assignment.status == "new":
            resolved = self._confirmed_candidates.get(candidate_key)
            if resolved is not None and resolved != assignment.speaker_id:
                raise ValueError("A candidate cannot resolve to conflicting speaker identities")
        self._watermark_ns = max(self._watermark_ns, assignment.end_time_ns)
        self._history[key] = _Evidence(assignment, capture_epoch)
        self.stats["assignments"] += 1
        if assignment.status == "new":
            if candidate_key not in self._confirmed_candidates:
                self.stats["candidate_confirmations"] += 1
            self._confirmed_candidates[candidate_key] = assignment.speaker_id
        self._prune()
        affected = [(assignment.start_time_ns, assignment.end_time_ns)]
        if assignment.status == "new":
            # Resolving a candidate can change earlier provisional intervals as
            # well as this observation. The union avoids quadratic repeated
            # intersections when many candidate windows overlap each other.
            affected.extend((item.assignment.start_time_ns, item.assignment.end_time_ns)
                for item in self._history.values()
                if item.assignment.track_id == assignment.track_id
                and item.capture_epoch == capture_epoch
                and item.assignment.status == "provisional"
                and item.assignment.candidate_id == assignment.candidate_id)
        merged = []
        for start, end in sorted(affected):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        starts = [start for start, _ in merged]
        references = []
        for item in self._captions.values():
            if item.track_id != assignment.track_id or item.capture_epoch != capture_epoch:
                continue
            index = bisect_left(starts, item.end_ns) - 1
            if index >= 0 and merged[index][1] > item.start_ns:
                references.append(item)
        return self._patch(references)

    def _label(self, entry: _Evidence) -> str | None:
        assignment = entry.assignment
        if assignment.status in ("new", "existing"):
            return assignment.speaker_id
        if assignment.status == "provisional":
            return self._confirmed_candidates.get((assignment.track_id, entry.capture_epoch,
                                                   assignment.candidate_id))
        return None

    def _decision(self, caption: _CaptionRef) -> _Decision:
        boundaries: dict[int, list[tuple[str | None, int]]] = {}
        known_ids = set()
        for entry in self._history.values():
            item = entry.assignment
            if item.track_id != caption.track_id or entry.capture_epoch != caption.capture_epoch:
                continue
            start, end = max(item.start_time_ns, caption.start_ns), min(item.end_time_ns, caption.end_ns)
            if end <= start:
                continue
            label = self._label(entry)
            if label is not None:
                known_ids.add(label)
            boundaries.setdefault(start, []).append((label, 1))
            boundaries.setdefault(end, []).append((label, -1))
        active: dict[str | None, int] = {}
        support: dict[str, int] = {}
        unknown_ns = 0
        previous_ns = caption.start_ns
        for boundary, changes in sorted(boundaries.items()):
            elapsed = boundary - previous_ns
            if active.get(None, 0) > 0:
                unknown_ns += elapsed
            else:
                speakers = [key for key, value in active.items() if key is not None and value > 0]
                if len(speakers) == 1:
                    support[speakers[0]] = support.get(speakers[0], 0) + elapsed
                elif len(speakers) > 1:
                    unknown_ns += elapsed
            for label, delta in changes:
                active[label] = active.get(label, 0) + delta
                if active[label] == 0:
                    del active[label]
            previous_ns = boundary
        duration = caption.end_ns - caption.start_ns
        coverage = max(support.values(), default=0) / duration
        unknown_fraction = unknown_ns / duration
        if len(known_ids) > 1:
            return _Decision(None, "multiple_speakers", coverage, unknown_fraction)
        if unknown_fraction > self.config.max_unknown_fraction:
            return _Decision(None, "unknown_or_overlap_evidence", coverage, unknown_fraction)
        if not support or coverage < self.config.min_coverage:
            return _Decision(None, "insufficient_coverage", coverage, unknown_fraction)
        return _Decision(next(iter(support)), "dominant_confirmed_speaker", coverage, unknown_fraction)

    def _patch(self, references) -> tuple[SpeakerPatch, ...]:
        patches = []
        for caption in references:
            self.stats["caption_rechecks"] += 1
            segment = self.store.get_segment(caption.segment_id)
            if (segment is None or segment.superseded_by
                    or (segment.source_revision, segment.track_id, segment.start_ns, segment.end_ns)
                    != (caption.source_revision, caption.track_id, caption.start_ns, caption.end_ns)):
                self._captions.pop(caption.segment_id, None)
                self.stats["stale_captions"] += 1
                continue
            if segment.speaker_manual:
                self.stats["manual_preserved"] += 1
                continue
            decision = self._decision(caption)
            if segment.speaker_id == decision.speaker_id:
                continue
            revision = segment.speaker_revision + 1
            if self.store.update_speaker(caption.segment_id, decision.speaker_id, revision,
                                         based_on_source_revision=caption.source_revision):
                patches.append(SpeakerPatch(caption.segment_id, decision.speaker_id, revision,
                    caption.source_revision, caption.track_id, caption.capture_epoch,
                    caption.start_ns, caption.end_ns, decision.reason,
                    decision.coverage, decision.unknown_fraction))
                self.stats["speaker_patches"] += 1
        return tuple(patches)
