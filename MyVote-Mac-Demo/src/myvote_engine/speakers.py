"""Session-local open-set tracking of *already extracted* speaker embeddings.

No audio model, voice activity detector, or overlap detector is implemented here.
Default thresholds are uncalibrated engineering examples, not accuracy claims.
Call from one ordered reducer: observations must have nondecreasing end times.
Time values use the capture host's session nanoseconds, never server wall time.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Iterable, Literal


Status = Literal["unknown", "provisional", "existing", "new"]


@dataclass(frozen=True)
class TrackerConfig:
    """Tune against held-out audio for the exact embedding model and frontend."""

    match_similarity: float = 0.82
    new_similarity: float = 0.60
    candidate_similarity: float = 0.82
    update_similarity: float = 0.90
    min_margin: float = 0.08
    min_quality: float = 0.70
    min_speech_ns: int = 1_000_000_000
    confirmation_speech_ns: int = 3_000_000_000
    min_confirmations: int = 3
    candidate_idle_ns: int = 20_000_000_000
    max_candidates: int = 16
    max_speakers: int = 128
    max_recent_samples: int = 4096
    prototype_update_rate: float = 0.10

    def __post_init__(self) -> None:
        thresholds = (self.new_similarity, self.match_similarity,
                      self.candidate_similarity, self.update_similarity)
        if not all(math.isfinite(x) and -1 <= x <= 1 for x in thresholds):
            raise ValueError("cosine thresholds must be finite and in [-1, 1]")
        if not self.new_similarity < self.match_similarity <= self.update_similarity:
            raise ValueError("require new_similarity < match_similarity <= update_similarity")
        if not self.new_similarity < self.candidate_similarity:
            raise ValueError("candidate_similarity must exceed the novelty threshold")
        if not math.isfinite(self.min_margin) or not 0 <= self.min_margin <= 2:
            raise ValueError("min_margin must be finite and in [0, 2]")
        if not math.isfinite(self.min_quality) or not 0 <= self.min_quality <= 1:
            raise ValueError("min_quality must be finite and in [0, 1]")
        if not math.isfinite(self.prototype_update_rate) or not 0 < self.prototype_update_rate <= 1:
            raise ValueError("prototype_update_rate must be in (0, 1]")
        for field in ("min_speech_ns", "confirmation_speech_ns", "min_confirmations",
                      "candidate_idle_ns", "max_candidates", "max_speakers", "max_recent_samples"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.min_confirmations < 2:
            raise ValueError("new identities require at least two independent observations")


@dataclass(frozen=True)
class Observation:
    sample_id: str
    track_id: str
    start_time_ns: int
    end_time_ns: int
    embedding: tuple[float, ...]
    speech_duration_ns: int
    overlap: bool = False
    quality: float = 1.0


@dataclass(frozen=True)
class Assignment:
    sample_id: str
    track_id: str
    start_time_ns: int
    end_time_ns: int
    status: Status
    speaker_id: str | None = None
    candidate_id: str | None = None
    similarity: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class SpeakerProfile:
    """Persist with the session and embedding model revision; never across models.

    Restoring profiles resumes confirmed identities. Pending candidates and replay
    deduplication are deliberately discarded; new evidence must reconfirm them.
    """

    speaker_id: str
    name: str
    prototype: tuple[float, ...]
    support_count: int
    last_seen_ns: int
    last_support_end_ns: int


@dataclass
class _Candidate:
    candidate_id: str
    prototype: tuple[float, ...]
    support_count: int
    speech_ns: int
    last_support_end_ns: int


def _normalize(values: Iterable[float]) -> tuple[float, ...]:
    vector = tuple(float(x) for x in values)
    if not vector or not all(math.isfinite(x) for x in vector):
        raise ValueError("embedding must be nonempty and finite")
    # Scaling first also handles very large/small finite inputs without overflow.
    scale = max(abs(x) for x in vector)
    if scale == 0:
        raise ValueError("zero embedding is not a speaker observation")
    scaled = tuple(x / scale for x in vector)
    norm = math.sqrt(math.fsum(x * x for x in scaled))
    return tuple(x / norm for x in scaled)


def _similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right))))


def _blend(left: tuple[float, ...], right: tuple[float, ...], weight: float) -> tuple[float, ...]:
    return _normalize((1 - weight) * a + weight * b for a, b in zip(left, right))


class OnlineSpeakerTracker:
    """One session, one embedding model, one ordered calling thread.

    Gray-zone matches remain unknown. Clean observations whose support windows
    overlap previously credited audio never increase confirmations or prototypes.
    Confirmed identities survive idle periods; only tentative candidates expire.
    Profiles can be persisted by the host to resume a session after a restart.
    """

    def __init__(self, config: TrackerConfig | None = None, *,
                 initial_speakers: Iterable[SpeakerProfile] = ()) -> None:
        self.config = config or TrackerConfig()
        self._speakers: dict[str, SpeakerProfile] = {}
        self._candidates: dict[str, _Candidate] = {}
        self._recent: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._dimension: int | None = None
        self._watermark_ns = -1
        self._speaker_sequence = 0
        self._candidate_sequence = 0
        for profile in initial_speakers:
            if profile.speaker_id in self._speakers:
                raise ValueError("duplicate restored speaker_id")
            prefix, separator, suffix = profile.speaker_id.rpartition("-")
            if prefix != "speaker" or not separator or not suffix.isdigit() or int(suffix) <= 0:
                raise ValueError("restored speaker_id must have form speaker-<positive integer>")
            if not profile.name.strip():
                raise ValueError("restored name cannot be empty")
            if (not isinstance(profile.support_count, int) or profile.support_count < 1
                    or not isinstance(profile.last_seen_ns, int)
                    or not isinstance(profile.last_support_end_ns, int)
                    or not 0 <= profile.last_support_end_ns <= profile.last_seen_ns):
                raise ValueError("invalid restored profile counters/times")
            vector = _normalize(profile.prototype)
            self._check_dimension(vector)
            self._speakers[profile.speaker_id] = SpeakerProfile(
                profile.speaker_id, profile.name, vector, profile.support_count,
                profile.last_seen_ns, profile.last_support_end_ns)
            self._speaker_sequence = max(self._speaker_sequence, int(suffix))
            self._watermark_ns = max(self._watermark_ns, profile.last_seen_ns)
        if len(self._speakers) > self.config.max_speakers:
            raise ValueError("restored profiles exceed max_speakers")

    @property
    def speakers(self) -> tuple[SpeakerProfile, ...]:
        return tuple(self._speakers.values())

    @property
    def pending_candidates(self) -> int:
        return len(self._candidates)

    def rename(self, speaker_id: str, name: str) -> SpeakerProfile:
        name = name.strip()
        if not name or len(name) > 256:
            raise ValueError("name must contain 1 to 256 characters")
        old = self._speakers[speaker_id]
        updated = SpeakerProfile(old.speaker_id, name, old.prototype, old.support_count,
                                 old.last_seen_ns, old.last_support_end_ns)
        self._speakers[speaker_id] = updated
        return updated

    def expire(self, now_ns: int) -> None:
        """Advance using the same capture timeline, including during silence."""
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative integer")
        if now_ns < self._watermark_ns:
            raise ValueError("expiry clock cannot go backwards")
        self._watermark_ns = now_ns
        expired = [key for key, item in self._candidates.items()
                   if now_ns - item.last_support_end_ns > self.config.candidate_idle_ns]
        for key in expired:
            del self._candidates[key]

    def clear_candidates(self) -> int:
        """Discard tentative evidence at an input gap, retaining confirmed IDs.

        Candidate IDs are never recycled; late results cannot confirm a new
        candidate using the identifier of evidence from before the gap.
        """
        count = len(self._candidates)
        self._candidates.clear()
        return count

    def _check_dimension(self, vector: tuple[float, ...]) -> None:
        if self._dimension is not None and self._dimension != len(vector):
            raise ValueError("embedding dimension changed; use a new tracker for a new model")
        self._dimension = len(vector)

    @staticmethod
    def _result(obs: Observation, status: Status, *, speaker_id: str | None = None,
                candidate_id: str | None = None, similarity: float | None = None,
                reason: str) -> Assignment:
        return Assignment(obs.sample_id, obs.track_id, obs.start_time_ns, obs.end_time_ns,
                          status, speaker_id, candidate_id, similarity, reason)

    def observe(self, obs: Observation) -> Assignment:
        if not obs.sample_id or not obs.track_id:
            raise ValueError("sample_id and track_id are required")
        times = (obs.start_time_ns, obs.end_time_ns, obs.speech_duration_ns)
        if any(isinstance(x, bool) or not isinstance(x, int) for x in times):
            raise ValueError("times/durations must be integer nanoseconds")
        if (obs.start_time_ns < 0 or obs.end_time_ns <= obs.start_time_ns
                or not 0 <= obs.speech_duration_ns <= obs.end_time_ns - obs.start_time_ns):
            raise ValueError("invalid observation range or speech duration")
        if not math.isfinite(obs.quality) or not 0 <= obs.quality <= 1:
            raise ValueError("quality must be finite and in [0, 1]")
        vector = _normalize(obs.embedding)
        if self._dimension is not None and len(vector) != self._dimension:
            raise ValueError("embedding dimension changed; use a new tracker for a new model")
        key = (obs.track_id, obs.sample_id)
        if key in self._recent:
            return self._result(obs, "unknown", reason="duplicate_sample")
        if obs.end_time_ns < self._watermark_ns:
            return self._result(obs, "unknown", reason="out_of_order")
        self.expire(obs.end_time_ns)
        self._recent[key] = None
        if len(self._recent) > self.config.max_recent_samples:
            self._recent.popitem(last=False)
        if obs.overlap:
            return self._result(obs, "unknown", reason="overlap")
        if obs.quality < self.config.min_quality:
            return self._result(obs, "unknown", reason="low_quality")
        if obs.speech_duration_ns < self.config.min_speech_ns:
            return self._result(obs, "unknown", reason="insufficient_speech")
        self._check_dimension(vector)
        ranked = sorted(((_similarity(vector, item.prototype), key)
                         for key, item in self._speakers.items()), reverse=True)
        best = ranked[0][0] if ranked else None
        runner_up = ranked[1][0] if len(ranked) > 1 else -1.0
        if ranked and best >= self.config.match_similarity:
            if best - runner_up < self.config.min_margin:
                return self._result(obs, "unknown", similarity=best, reason="ambiguous_existing")
            speaker = self._speakers[ranked[0][1]]
            prototype = speaker.prototype
            support_count = speaker.support_count
            last_support = speaker.last_support_end_ns
            if best >= self.config.update_similarity and obs.start_time_ns >= last_support:
                prototype = _blend(prototype, vector, self.config.prototype_update_rate)
                support_count += 1
                last_support = obs.end_time_ns
            self._speakers[speaker.speaker_id] = SpeakerProfile(
                speaker.speaker_id, speaker.name, prototype, support_count,
                obs.end_time_ns, last_support)
            return self._result(obs, "existing", speaker_id=speaker.speaker_id,
                                similarity=best, reason="matched_existing")
        if best is not None and best > self.config.new_similarity:
            return self._result(obs, "unknown", similarity=best, reason="ambiguous_existing")

        # Only observations unlike ALL confirmed speakers enter the novelty pool.
        pending = sorted(((_similarity(vector, item.prototype), key)
                          for key, item in self._candidates.items()), reverse=True)
        if pending and pending[0][0] >= self.config.candidate_similarity:
            candidate_score, candidate_key = pending[0]
            if len(pending) > 1 and candidate_score - pending[1][0] < self.config.min_margin:
                return self._result(obs, "unknown", similarity=candidate_score,
                                    reason="ambiguous_candidate")
            candidate = self._candidates[candidate_key]
            if obs.start_time_ns < candidate.last_support_end_ns:
                return self._result(obs, "provisional", candidate_id=candidate.candidate_id,
                                    similarity=candidate_score, reason="overlapping_evidence")
            weight = obs.speech_duration_ns / (candidate.speech_ns + obs.speech_duration_ns)
            candidate.prototype = _blend(candidate.prototype, vector, weight)
            candidate.support_count += 1
            candidate.speech_ns += obs.speech_duration_ns
            candidate.last_support_end_ns = obs.end_time_ns
        else:
            if pending and pending[0][0] > self.config.new_similarity:
                return self._result(obs, "unknown", similarity=pending[0][0],
                                    reason="ambiguous_candidate")
            if len(self._candidates) >= self.config.max_candidates:
                # Bound memory while leaving confirmed identity history untouched.
                oldest = min(self._candidates,
                             key=lambda key: self._candidates[key].last_support_end_ns)
                del self._candidates[oldest]
            self._candidate_sequence += 1
            candidate = _Candidate(f"candidate-{self._candidate_sequence:04d}", vector,
                                   1, obs.speech_duration_ns, obs.end_time_ns)
            self._candidates[candidate.candidate_id] = candidate
        if (candidate.support_count < self.config.min_confirmations
                or candidate.speech_ns < self.config.confirmation_speech_ns):
            return self._result(obs, "provisional", candidate_id=candidate.candidate_id,
                                similarity=best, reason="collecting_evidence")
        if len(self._speakers) >= self.config.max_speakers:
            del self._candidates[candidate.candidate_id]
            return self._result(obs, "unknown", reason="speaker_capacity")
        self._speaker_sequence += 1
        speaker_id = f"speaker-{self._speaker_sequence:04d}"
        self._speakers[speaker_id] = SpeakerProfile(
            speaker_id, f"Speaker {self._speaker_sequence}", candidate.prototype,
            candidate.support_count, obs.end_time_ns, candidate.last_support_end_ns)
        del self._candidates[candidate.candidate_id]
        return self._result(obs, "new", speaker_id=speaker_id,
                            candidate_id=candidate.candidate_id, similarity=best,
                            reason="confirmed_new_speaker")
