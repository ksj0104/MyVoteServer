"""Model-free caption projection and exports on the source audio timeline.

This is an in-memory research store, not the product's persistent event database.
One host owns revision allocation. Translation text passed here is cumulative,
never a token delta. Speaker fields have independent revisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from html import escape
import json
from threading import RLock
from typing import Literal


SourceState = Literal["provisional", "stable"]
ExportMode = Literal["source", "translation", "bilingual"]
MissingTranslation = Literal["source", "omit", "error"]
OverlapPolicy = Literal["preserve", "merge"]
NS_PER_MS = 1_000_000


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _text(value: str, name: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be non-empty text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be UTF-8 encodable") from exc


@dataclass(frozen=True)
class CaptionTranslation:
    source_revision: int
    provider_generation: int
    target_language: str
    text: str
    completed: bool = True

    def __post_init__(self) -> None:
        _integer(self.source_revision, "source_revision", 1)
        _integer(self.provider_generation, "provider_generation")
        _text(self.target_language, "target_language")
        _text(self.text, "translation text")
        if type(self.completed) is not bool:
            raise ValueError("completed must be a bool")


@dataclass(frozen=True)
class CaptionSegment:
    segment_id: str
    track_id: str
    start_ns: int
    end_ns: int
    source_text: str
    source_revision: int
    source_state: SourceState = "stable"
    target_language: str = "ko"
    speaker_id: str | None = None
    speaker_revision: int = 0
    speaker_manual: bool = False
    translation: CaptionTranslation | None = None
    superseded_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("segment_id", "track_id", "source_text", "target_language"):
            _text(getattr(self, name), name)
        _integer(self.start_ns, "start_ns")
        _integer(self.end_ns, "end_ns", self.start_ns + 1)
        _integer(self.source_revision, "source_revision", 1)
        _integer(self.speaker_revision, "speaker_revision")
        if self.source_state not in ("provisional", "stable"):
            raise ValueError("source_state must be provisional or stable")
        if self.speaker_id is not None:
            _text(self.speaker_id, "speaker_id")
        if type(self.speaker_manual) is not bool:
            raise ValueError("speaker_manual must be a bool")
        if self.translation is not None and not isinstance(self.translation, CaptionTranslation):
            raise ValueError("translation must be CaptionTranslation or None")
        if not isinstance(self.superseded_by, tuple):
            raise ValueError("superseded_by must be an immutable tuple")
        for segment_id in self.superseded_by:
            _text(segment_id, "superseding segment ID")
            if segment_id == self.segment_id:
                raise ValueError("A segment cannot supersede itself")

    @property
    def current_translation(self) -> CaptionTranslation | None:
        """Preview and stale translations remain inspectable but are not final cues.

        A completed historical translation retains its provider provenance after
        provider replacement. Replacement only rejects incoming old results.
        """
        item = self.translation
        if (item is not None and item.completed
                and item.source_revision == self.source_revision
                and item.target_language == self.target_language):
            return item
        return None


@dataclass(frozen=True)
class CaptionChange:
    event_seq: int
    event_type: str
    payload_json: str


@dataclass(frozen=True)
class CaptionSnapshot:
    session_id: str
    revision: int
    provider_generation: int
    segments: tuple[CaptionSegment, ...]
    speaker_aliases: tuple[tuple[str, str], ...] = ()
    changes: tuple[CaptionChange, ...] = ()


class CaptionStore:
    """Atomic projection/snapshot; immutable entries can safely outlive the store.

    Session-unique segment IDs are required across tracks. Accepted source,
    completed translation, speaker and generation changes have an in-memory
    journal. Streaming previews affect the snapshot revision but are not retained
    in that journal, avoiding quadratic storage of growing translation strings.
    """

    def __init__(self, session_id: str, provider_generation: int = 0):
        _text(session_id, "session_id")
        _integer(provider_generation, "provider_generation")
        self.session_id = session_id
        self._generation = provider_generation
        self._revision = 0
        self._segments: dict[str, CaptionSegment] = {}
        self._aliases: dict[str, str] = {}
        self._changes: list[CaptionChange] = []
        self._lock = RLock()

    def _record(self, event_type: str, payload: dict, *, retain: bool = True) -> None:
        self._revision += 1
        if retain:
            self._changes.append(CaptionChange(
                self._revision, event_type,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ))

    def upsert_source(self, segment: CaptionSegment) -> bool:
        """Apply a newer source revision without overwriting speaker/translation.

        New source revisions retain the previous translation as stale evidence.
        Source payloads cannot inject translation results or speaker overrides.
        """
        if segment.translation is not None or segment.superseded_by:
            raise ValueError("Source updates cannot supply translations or tombstones")
        if segment.speaker_revision or segment.speaker_manual:
            raise ValueError("Apply independent speaker revisions with update_speaker")
        with self._lock:
            current = self._segments.get(segment.segment_id)
            if current is not None:
                if current.track_id != segment.track_id:
                    raise ValueError("Segment IDs must remain on their original track")
                if current.superseded_by or segment.source_revision <= current.source_revision:
                    return False
                segment = replace(segment, speaker_id=current.speaker_id,
                                  speaker_revision=current.speaker_revision,
                                  speaker_manual=current.speaker_manual,
                                  translation=current.translation)
            self._segments[segment.segment_id] = segment
            self._record("transcript.updated", asdict(segment))
            return True

    def apply_translation(self, segment_id: str, source_revision: int,
                          provider_generation: int, text: str, *,
                          target_language: str = "ko", completed: bool = True) -> bool:
        incoming = CaptionTranslation(source_revision, provider_generation,
                                      target_language, text, completed)
        with self._lock:
            current = self._segments.get(segment_id)
            if (current is None or current.superseded_by
                    or source_revision != current.source_revision
                    or target_language != current.target_language
                    or provider_generation != self._generation):
                return False
            previous = current.translation
            same_request = (previous is not None
                            and previous.source_revision == source_revision
                            and previous.provider_generation == provider_generation
                            and previous.target_language == target_language)
            if same_request:
                if previous.completed or incoming == previous:
                    return False
                # Cumulative previews must grow; an older buffered chunk must not
                # erase a newer preview. Completion may normalize whitespace.
                if not completed and not text.startswith(previous.text):
                    return False
            self._segments[segment_id] = replace(current, translation=incoming)
            self._record("translation.completed" if completed else "translation.preview",
                         {"segment_id": segment_id, **asdict(incoming)}, retain=completed)
            return True

    def set_provider_generation(self, generation: int) -> bool:
        _integer(generation, "provider_generation")
        with self._lock:
            if generation < self._generation:
                raise ValueError("Provider generation cannot move backwards")
            if generation == self._generation:
                return False
            self._generation = generation
            self._record("provider.updated", {"provider_generation": generation})
            return True

    def update_speaker(self, segment_id: str, speaker_id: str | None,
                       speaker_revision: int, *, based_on_source_revision: int | None = None,
                       manual: bool = False) -> bool:
        _integer(speaker_revision, "speaker_revision", 1)
        if speaker_id is not None:
            _text(speaker_id, "speaker_id")
        if type(manual) is not bool:
            raise ValueError("manual must be a bool")
        if based_on_source_revision is not None:
            _integer(based_on_source_revision, "based_on_source_revision", 1)
        with self._lock:
            current = self._segments.get(segment_id)
            if (current is None or current.superseded_by
                    or speaker_revision <= current.speaker_revision
                    or (current.speaker_manual and not manual)
                    or (based_on_source_revision is not None
                        and based_on_source_revision != current.source_revision)):
                return False
            self._segments[segment_id] = replace(
                current, speaker_id=speaker_id, speaker_revision=speaker_revision,
                speaker_manual=manual,
            )
            self._record("speaker.override" if manual else "speaker.updated", {
                "segment_id": segment_id, "speaker_id": speaker_id,
                "speaker_revision": speaker_revision,
                "based_on_source_revision": based_on_source_revision, "manual": manual,
            })
            return True

    def get_segment(self, segment_id: str) -> CaptionSegment | None:
        """Read one immutable current caption without copying the session journal."""
        with self._lock:
            return self._segments.get(segment_id)

    def speaker_aliases(self) -> tuple[tuple[str, str], ...]:
        with self._lock:
            return tuple(self._aliases.items())

    def set_speaker_alias(self, speaker_id: str, name: str) -> bool:
        _text(speaker_id, "speaker_id")
        _text(name, "speaker alias")
        with self._lock:
            if self._aliases.get(speaker_id) == name:
                return False
            self._aliases[speaker_id] = name
            self._record("speaker.alias_updated", {"speaker_id": speaker_id, "name": name})
            return True

    def supersede_segment(self, segment_id: str, superseded_by: tuple[str, ...]) -> bool:
        """Tombstone an old ID; its delayed results cannot attach to replacements.

        Caller owns replacement source revisions and any time-span override
        migration. This store does not infer a speaker for a split/merged span.
        """
        if not superseded_by or not isinstance(superseded_by, tuple):
            raise ValueError("Supply one or more replacement IDs as a tuple")
        with self._lock:
            current = self._segments.get(segment_id)
            if current is None or current.superseded_by:
                return False
            updated = replace(current, superseded_by=superseded_by)
            self._segments[segment_id] = updated
            self._record("segment.superseded", {
                "segment_id": segment_id, "superseded_by": superseded_by,
            })
            return True

    def snapshot(self) -> CaptionSnapshot:
        with self._lock:
            return CaptionSnapshot(
                self.session_id, self._revision, self._generation,
                tuple(sorted(self._segments.values(), key=_segment_order)),
                tuple(sorted(self._aliases.items())), tuple(self._changes),
            )


def _segment_order(segment: CaptionSegment) -> tuple[int, int, str, str]:
    return segment.start_ns, segment.end_ns, segment.track_id, segment.segment_id


def _subtitle_text(value: str) -> str:
    # A blank line terminates a cue. Collapse empty lines and replace forbidden
    # control characters without letting transcript text create new cue blocks.
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(char if char in "\n\t" or ord(char) >= 32 else "\ufffd" for char in value)
    lines = [line.strip() for line in value.split("\n") if line.strip()]
    return escape("\n".join(lines), quote=False)


def _cue_text(segment: CaptionSegment, aliases: dict[str, str], mode: ExportMode,
              missing: MissingTranslation, webvtt: bool, speaker_labels: bool) -> str:
    translation = segment.current_translation
    if mode != "source" and translation is None:
        if missing == "error":
            raise ValueError(f"No completed current translation for {segment.segment_id}")
        if missing == "omit":
            return ""
    lines = []
    if mode in ("source", "bilingual") or translation is None:
        lines.append(_subtitle_text(segment.source_text))
    if mode != "source" and translation is not None:
        lines.append(_subtitle_text(translation.text))
    body = "\n".join(lines)
    if speaker_labels and segment.speaker_id is not None:
        # Alias whitespace cannot terminate a voice annotation or inject a cue.
        label = " ".join(aliases.get(segment.speaker_id, segment.speaker_id).split())
        label = _subtitle_text(label)
        if webvtt:
            # Voice metadata alone is not visibly rendered by every player.
            return f"<v {label}>{label}: {body}</v>"
        return f"{label}: {body}"
    return body


def _cues(snapshot: CaptionSnapshot, *, mode: ExportMode,
          missing_translation: MissingTranslation, include_provisional: bool,
          offset_ns: int, speaker_labels: bool, webvtt: bool,
          overlap: OverlapPolicy) -> list[tuple[int, int, str]]:
    if mode not in ("source", "translation", "bilingual"):
        raise ValueError("mode must be source, translation or bilingual")
    if missing_translation not in ("source", "omit", "error"):
        raise ValueError("missing_translation must be source, omit or error")
    if overlap not in ("preserve", "merge"):
        raise ValueError("overlap must be preserve or merge")
    if type(offset_ns) is not int:
        raise ValueError("offset_ns must be an integer")
    aliases = dict(snapshot.speaker_aliases)
    cues = []
    for segment in sorted(snapshot.segments, key=_segment_order):
        if segment.superseded_by or (segment.source_state == "provisional" and not include_provisional):
            continue
        body = _cue_text(segment, aliases, mode, missing_translation, webvtt, speaker_labels)
        if not body:
            continue
        start_ns, end_ns = segment.start_ns + offset_ns, segment.end_ns + offset_ns
        if start_ns < 0 or end_ns <= start_ns:
            raise ValueError("Export offset produces invalid source timing")
        # Expand by less than 1 ms on either side; never collapse a short cue or
        # round a valid positive source interval to zero duration.
        start_ms = start_ns // NS_PER_MS
        end_ms = (end_ns + NS_PER_MS - 1) // NS_PER_MS
        cues.append((start_ms, end_ms, body))
    if overlap == "preserve":
        return cues
    # Slice at every active-set boundary. Simultaneous text shares a cue; no
    # speaker is delayed to make an overlapping recording look sequential.
    boundaries: dict[int, list[tuple[bool, int]]] = {}
    for index, (start, end, _) in enumerate(cues):
        boundaries.setdefault(start, []).append((True, index))
        boundaries.setdefault(end, []).append((False, index))
    active: set[int] = set()
    merged = []
    previous_time = None
    for time_ms, changes in sorted(boundaries.items()):
        if previous_time is not None and time_ms > previous_time and active:
            body = "\n".join(cues[index][2] for index in sorted(active))
            merged.append((previous_time, time_ms, body))
        for entering, index in changes:
            if entering:
                active.add(index)
            else:
                active.discard(index)
        previous_time = time_ms
    return merged


def _timestamp(milliseconds: int, decimal: str) -> str:
    seconds, millis = divmod(milliseconds, 1000)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}{decimal}{millis:03d}"


def export_srt(snapshot: CaptionSnapshot, *, mode: ExportMode = "bilingual",
               missing_translation: MissingTranslation = "source",
               include_provisional: bool = False, offset_ns: int = 0,
               speaker_labels: bool = True, overlap: OverlapPolicy = "preserve") -> str:
    """Return Unicode SRT; caller writes UTF-8 (no BOM required).

    Default preserves overlaps. overlap='merge' creates non-overlapping cues
    containing all active text for players unable to display simultaneous cues.
    """
    cues = _cues(snapshot, mode=mode, missing_translation=missing_translation,
                 include_provisional=include_provisional, offset_ns=offset_ns,
                 speaker_labels=speaker_labels, webvtt=False, overlap=overlap)
    return "".join(f"{index}\n{_timestamp(start, ',')} --> {_timestamp(end, ',')}\n{body}\n\n"
                   for index, (start, end, body) in enumerate(cues, 1))


def export_webvtt(snapshot: CaptionSnapshot, *, mode: ExportMode = "bilingual",
                  missing_translation: MissingTranslation = "source",
                  include_provisional: bool = False, offset_ns: int = 0,
                  speaker_labels: bool = True, overlap: OverlapPolicy = "preserve") -> str:
    cues = _cues(snapshot, mode=mode, missing_translation=missing_translation,
                 include_provisional=include_provisional, offset_ns=offset_ns,
                 speaker_labels=speaker_labels, webvtt=True, overlap=overlap)
    return "WEBVTT\n\n" + "".join(
        f"{index}\n{_timestamp(start, '.')} --> {_timestamp(end, '.')}\n{body}\n\n"
        for index, (start, end, body) in enumerate(cues, 1)
    )


def export_json(snapshot: CaptionSnapshot) -> str:
    """Lossless projection metadata and retained accepted changes; no raw audio.

    Exact integer nanoseconds, stale/preview translations and tombstones are
    preserved. This format is not the complete product session database: word
    timings, embedding vectors, model manifests and raw audio are not inputs.
    """
    data = {
        "schema": "myvote.caption_snapshot", "schema_version": 1,
        "session_id": snapshot.session_id, "revision": snapshot.revision,
        "provider_generation": snapshot.provider_generation,
        "time_unit": "ns", "time_basis": "source_audio_session",
        "speaker_aliases": dict(snapshot.speaker_aliases),
        "segments": [{**asdict(segment),
                      "translation_current": segment.current_translation is not None}
                     for segment in sorted(snapshot.segments, key=_segment_order)],
        "changes": [{"event_seq": item.event_seq, "event_type": item.event_type,
                     "payload": json.loads(item.payload_json)} for item in snapshot.changes],
        "journal_scope": "accepted_source_completed_translation_speaker_provider_and_supersession",
    }
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
