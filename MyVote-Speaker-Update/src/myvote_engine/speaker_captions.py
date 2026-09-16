"""Map supplied speaker assignments onto caption intervals without inference.

This reducer runs on the host's ordered event path. It never extracts a voice,
votes on a new identity, delays translation, or calls a model. All decisions use
the source audio timeline. Coverage is a duration fraction, not a probability.
Synthetic Assignment tests demonstrate mapping contracts, not voice accuracy.
"""

from __future__ import annotations

from collections import OrderedDict
from bisect import bisect_left
from dataclasses import dataclass, replace
import math

from .captions import CaptionStore
from .speakers import Assignment, support_intervals_for


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
    duration_ns: int = 0
    known_ns: int = 0
    unknown_ns: int = 0
    nonspeech_ns: int = 0
    overlap_ns: int = 0
    ambiguous_ns: int = 0
    conflict_ns: int = 0
    unobserved_ns: int = 0

    @property
    def evaluated_duration_ns(self) -> int:
        return self.duration_ns - self.nonspeech_ns

    @property
    def evaluated_coverage(self) -> float:
        # coverage remains the best confirmed speaker / full caption duration.
        # The guarded speech-only view is diagnostic; it cannot bypass raw .8.
        return (self.coverage * self.duration_ns / self.evaluated_duration_ns
                if self.evaluated_duration_ns else 0.0)

    @property
    def evaluated_unknown_fraction(self) -> float:
        return self.unknown_ns / self.evaluated_duration_ns if self.evaluated_duration_ns else 0.0


_TIME_FIELDS = ("duration_ns", "known_ns", "unknown_ns", "nonspeech_ns", "overlap_ns",
                "ambiguous_ns", "conflict_ns", "unobserved_ns", "evaluated_duration_ns")
_OVERLAP_REASONS = frozenset(("predicted_overlap", "overlap", "overlap_uncertain"))
_AMBIGUOUS_REASONS = frozenset(("ambiguous_existing", "ambiguous_candidate"))


def _diagnostic_bucket() -> dict:
    return {"count": 0, **dict.fromkeys(_TIME_FIELDS, 0),
            "coverage_sum": 0.0, "unknown_fraction_sum": 0.0,
            "evaluated_coverage_sum": 0.0, "evaluated_unknown_fraction_sum": 0.0}


def _aggregate_decision(bucket: dict, decision: _Decision) -> None:
    bucket["count"] += 1
    for name in _TIME_FIELDS:
        bucket[name] += getattr(decision, name)
    bucket["coverage_sum"] += decision.coverage
    bucket["unknown_fraction_sum"] += decision.unknown_fraction
    bucket["evaluated_coverage_sum"] += decision.evaluated_coverage
    bucket["evaluated_unknown_fraction_sum"] += decision.evaluated_unknown_fraction


class SpeakerCaptionReducer:
    """Bounded recent speaker evidence and caption references for one session.

    Register a caption after its source update is accepted by ``store``. Calls
    return only patches already accepted by ``store.update_speaker``; emit those
    on the independent speaker event path. The source and translation revisions
    never change. Local manual assignments and aliases remain authoritative.

    Explicit overlap, ambiguous identities, and conflicting evidence always
    block automatic naming. Other unknown intervals block naming by default. Any
    second confirmed speaker overlapping a caption blocks naming, even when one
    speaker occupies most of the time. Unobserved edges instead reduce coverage.
    Trusted predicted nonspeech alone is neutral; the confirmed speaker still
    needs coverage of at least min_coverage of the FULL caption interval.
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
        self._latest_decisions: dict[str, _Decision] = {}
        self._retired_decisions: dict[str, dict] = {}
        self._retirement_causes: dict[str, int] = {}
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

    def confirmed_speaker_for(self, *, track_id: str, capture_epoch: str,
                              start_ns: int, end_ns: int) -> str | None:
        """Read-only routing from acoustic evidence; never infer identity from text.

        Uses exactly the caption mapping guards, without creating a caption or
        adding votes, aliases, retention state or diagnostic observations.
        """
        _identifier(track_id, "track_id")
        _identifier(capture_epoch, "capture_epoch")
        _integer(start_ns, "start_ns")
        _integer(end_ns, "end_ns", start_ns)
        if end_ns <= start_ns or start_ns < self._retention_floor_ns:
            return None
        return self._decision(_CaptionRef("routing", track_id, capture_epoch,
                                         start_ns, end_ns, 1)).speaker_id

    def diagnostics(self) -> dict:
        """Read-only bounded aggregates; never copy text, aliases, or sample IDs.

        Latest decisions cover retained caption records at their last recheck.
        Retired records retain that last decision, not a later UI/manual state.
        Revisions replaced or records re-registered after capacity eviction are
        distinct tracking records, so counts are not unique session caption IDs.
        Time totals are per-caption unions; overlapping captions can share time.
        overlap/ambiguous/conflict are subsets of unknown_ns, not extra duration.
        """
        retained: dict[str, dict] = {}
        for decision in self._latest_decisions.values():
            _aggregate_decision(retained.setdefault(decision.reason, _diagnostic_bucket()), decision)
        return {"schema_version": 1, "scope": "last_decision_per_caption_tracking_record",
                "retained_latest": {"count": len(self._latest_decisions), "by_reason": retained},
                "retired_last": {"count": sum(row["count"] for row in self._retired_decisions.values()),
                    "by_reason": {key: dict(value) for key, value in self._retired_decisions.items()},
                    "retirement_causes": dict(self._retirement_causes)},
                "decision_rechecks": self.stats["caption_rechecks"],
                "min_raw_coverage": self.config.min_coverage,
                "max_unknown_fraction": self.config.max_unknown_fraction,
                "unknown_fraction_policy_denominator": "caption_duration_minus_exclusive_predicted_nonspeech"}

    def _retire_caption(self, segment_id: str, cause: str) -> None:
        self._captions.pop(segment_id, None)
        decision = self._latest_decisions.pop(segment_id, None)
        if decision is not None:
            _aggregate_decision(self._retired_decisions.setdefault(decision.reason, _diagnostic_bucket()), decision)
            self._retirement_causes[cause] = self._retirement_causes.get(cause, 0) + 1

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
                self._retire_caption(key, "retention_floor")
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
            self._retire_caption(oldest, "capacity")
            self.stats["caption_evictions"] += 1
        reference = _CaptionRef(segment_id, track_id, capture_epoch, start_ns, end_ns, source_revision)
        if segment_id in self._captions and self._captions[segment_id] != reference:
            self._retire_caption(segment_id, "source_replaced")
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
        support_intervals_for(item)
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
        affected = list(support_intervals_for(assignment))
        if assignment.status == "new":
            # Resolving a candidate can change earlier provisional intervals as
            # well as this observation. The union avoids quadratic repeated
            # intersections when many candidate windows overlap each other.
            affected.extend(interval
                for item in self._history.values()
                if item.assignment.track_id == assignment.track_id
                and item.capture_epoch == capture_epoch
                and item.assignment.status == "provisional"
                and item.assignment.candidate_id == assignment.candidate_id
                for interval in support_intervals_for(item.assignment))
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
        # Keep evidence types separate from opaque speaker IDs. Only exclusive
        # nonspeech cells can be neutral; it never erases contradictory evidence.
        boundaries = {caption.start_ns: [], caption.end_ns: []}
        known_ids = set()
        for entry in self._history.values():
            item = entry.assignment
            if item.track_id != caption.track_id or entry.capture_epoch != caption.capture_epoch:
                continue
            intersections = tuple((max(start, caption.start_ns), min(end, caption.end_ns))
                                  for start, end in support_intervals_for(item)
                                  if start < caption.end_ns and end > caption.start_ns)
            if not intersections:
                continue
            label = self._label(entry)
            if label is not None:
                known_ids.add(label)
                category = ("known", label)
            elif item.status == "unknown" and item.reason == "predicted_nonspeech":
                category = ("nonspeech", None)
            elif item.status == "unknown" and item.reason in _OVERLAP_REASONS:
                category = ("overlap", None)
            elif item.status == "unknown" and item.reason in _AMBIGUOUS_REASONS:
                category = ("ambiguous", None)
            else:
                category = ("unknown", None)
            for start, end in intersections:
                boundaries.setdefault(start, []).append((category, 1))
                boundaries.setdefault(end, []).append((category, -1))
        active: dict[tuple[str, str | None], int] = {}
        support: dict[str, int] = {}
        unknown_ns = nonspeech_ns = overlap_ns = ambiguous_ns = conflict_ns = unobserved_ns = 0
        previous_ns = caption.start_ns
        for boundary, changes in sorted(boundaries.items()):
            elapsed = boundary - previous_ns
            speakers = [label for (kind, label), value in active.items() if kind == "known" and value > 0]
            nonspeech = active.get(("nonspeech", None), 0) > 0
            overlap = active.get(("overlap", None), 0) > 0
            ambiguous = active.get(("ambiguous", None), 0) > 0
            conflict = len(speakers) > 1 or bool(speakers and nonspeech)
            if overlap:
                overlap_ns += elapsed
            if ambiguous:
                ambiguous_ns += elapsed
            if conflict:
                conflict_ns += elapsed
            if overlap or ambiguous or conflict or active.get(("unknown", None), 0) > 0:
                unknown_ns += elapsed
            elif len(speakers) == 1:
                support[speakers[0]] = support.get(speakers[0], 0) + elapsed
            elif nonspeech:
                nonspeech_ns += elapsed
            else:
                unobserved_ns += elapsed
            for label, delta in changes:
                active[label] = active.get(label, 0) + delta
                if active[label] == 0:
                    del active[label]
            previous_ns = boundary
        duration = caption.end_ns - caption.start_ns
        coverage = max(support.values(), default=0) / duration
        unknown_fraction = unknown_ns / duration
        decision = _Decision(None, "insufficient_coverage", coverage, unknown_fraction,
                             duration, sum(support.values()), unknown_ns, nonspeech_ns,
                             overlap_ns, ambiguous_ns, conflict_ns, unobserved_ns)
        if len(known_ids) > 1:
            return replace(decision, reason="multiple_speakers")
        if overlap_ns:
            return replace(decision, reason="explicit_overlap_evidence")
        if ambiguous_ns:
            return replace(decision, reason="ambiguous_identity_evidence")
        if conflict_ns:
            return replace(decision, reason="conflicting_evidence")
        if nonspeech_ns == duration:
            return replace(decision, reason="predicted_nonspeech_only")
        if decision.evaluated_unknown_fraction > self.config.max_unknown_fraction:
            return replace(decision, reason="unknown_or_overlap_evidence")
        if not support or coverage < self.config.min_coverage:
            return decision
        return replace(decision, speaker_id=next(iter(support)), reason="dominant_confirmed_speaker")

    def _patch(self, references) -> tuple[SpeakerPatch, ...]:
        patches = []
        for caption in references:
            self.stats["caption_rechecks"] += 1
            segment = self.store.get_segment(caption.segment_id)
            if (segment is None or segment.superseded_by
                    or (segment.source_revision, segment.track_id, segment.start_ns, segment.end_ns)
                    != (caption.source_revision, caption.track_id, caption.start_ns, caption.end_ns)):
                self._retire_caption(caption.segment_id, "stale")
                self.stats["stale_captions"] += 1
                continue
            decision = self._decision(caption)
            if segment.speaker_manual:
                self._latest_decisions[caption.segment_id] = replace(decision, reason="manual_preserved")
                self.stats["manual_preserved"] += 1
                continue
            self._latest_decisions[caption.segment_id] = decision
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
