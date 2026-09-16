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
    result_revision: int = 1

    def __post_init__(self) -> None:
        _integer(self.source_revision, "source_revision", 1)
        _integer(self.provider_generation, "provider_generation")
        _integer(self.result_revision, "result_revision", 1)
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
    source_track_id: str | None = None
    capture_epoch: str | None = None
    separation_group_id: str | None = None
    lane_id: str | None = None
    parent_segment_ids: tuple[str, ...] = ()
    delivery_class: str = "live"
    text_operation: str = "translation"
    source_language: str | None = None

    def __post_init__(self) -> None:
        for name in ("segment_id", "track_id", "source_text", "target_language"):
            _text(getattr(self, name), name)
        _integer(self.start_ns, "start_ns")
        _integer(self.end_ns, "end_ns", self.start_ns + 1)
        _integer(self.source_revision, "source_revision", 1)
        _integer(self.speaker_revision, "speaker_revision")
        if self.source_state not in ("provisional", "stable"):
            raise ValueError("source_state must be provisional or stable")
        if self.text_operation not in ("translation", "transcript_correction"):
            raise ValueError("Invalid caption text operation")
        if self.source_language is not None:
            _text(self.source_language, "source_language")
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
        for name in ("source_track_id", "capture_epoch", "separation_group_id", "lane_id"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        if not isinstance(self.parent_segment_ids, tuple):
            raise ValueError("parent_segment_ids must be an immutable tuple")
        for parent in self.parent_segment_ids:
            _text(parent, "parent segment ID")
            if parent == self.segment_id:
                raise ValueError("A segment cannot be its own parent")
        if len(set(self.parent_segment_ids)) != len(self.parent_segment_ids):
            raise ValueError("Duplicate parent segment ID")
        if self.delivery_class not in ("live", "overlap_correction"):
            raise ValueError("Invalid caption delivery class")
        if self.delivery_class == "overlap_correction" and not all((
                self.source_track_id, self.capture_epoch, self.separation_group_id,
                self.lane_id, self.parent_segment_ids)):
            raise ValueError("Overlap captions require complete capture and parent lineage")

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
class CaptionResultCorrection:
    """A compare-and-replace of a completed result; source stays immutable."""
    segment_id: str
    source_revision: int
    based_on_result_revision: int
    text: str

    def __post_init__(self):
        _text(self.segment_id, "segment_id")
        _integer(self.source_revision, "source_revision", 1)
        _integer(self.based_on_result_revision, "based_on_result_revision", 1)
        _text(self.text, "corrected result")
        if len(self.text) > 24000:
            raise ValueError("Corrected result exceeds limit")


@dataclass(frozen=True)
class CaptionParentRef:
    segment_id: str
    source_revision: int

    def __post_init__(self) -> None:
        _text(self.segment_id, "parent segment ID")
        _integer(self.source_revision, "parent source_revision", 1)


@dataclass(frozen=True)
class CaptionGroupReplacement:
    """One complete replacement, validated before any parent is hidden.

    The group bounds cover whole parent captions; callers must include all
    non-overlap words from those parents in their independently timed children.
    Lane IDs describe separation streams, never inferred person identities.
    """
    group_id: str
    group_revision: int
    source_track_id: str
    capture_epoch: str
    start_ns: int
    end_ns: int
    parents: tuple[CaptionParentRef, ...]
    children: tuple[CaptionSegment, ...]

    def __post_init__(self) -> None:
        for name in ("group_id", "source_track_id", "capture_epoch"):
            _text(getattr(self, name), name)
        _integer(self.group_revision, "group_revision", 1)
        _integer(self.start_ns, "start_ns")
        _integer(self.end_ns, "end_ns", self.start_ns + 1)
        if (not isinstance(self.parents, tuple) or not 1 <= len(self.parents) <= 16
                or not all(isinstance(item, CaptionParentRef) for item in self.parents)):
            raise ValueError("Supply 1..16 immutable parent references")
        if (not isinstance(self.children, tuple) or not 2 <= len(self.children) <= 32
                or not all(isinstance(item, CaptionSegment) for item in self.children)):
            raise ValueError("Supply 2..32 immutable child captions")
        parents = {item.segment_id for item in self.parents}
        children = {item.segment_id for item in self.children}
        if len(parents) != len(self.parents) or len(children) != len(self.children) or parents & children:
            raise ValueError("Group IDs must be unique and cannot form self references")
        lanes: dict[str, str] = {}
        referenced = set()
        for child in self.children:
            if (child.delivery_class != "overlap_correction"
                    or child.source_track_id != self.source_track_id
                    or child.capture_epoch != self.capture_epoch
                    or child.separation_group_id != self.group_id
                    or child.source_state != "stable" or child.superseded_by
                    or child.speaker_manual or not set(child.parent_segment_ids) <= parents
                    or child.start_ns < self.start_ns or child.end_ns > self.end_ns
                    or child.track_id == self.source_track_id):
                raise ValueError("Invalid child capture, timing or parent lineage")
            if child.lane_id in lanes and lanes[child.lane_id] != child.track_id:
                raise ValueError("A lane cannot change tracks within a group")
            lanes[child.lane_id] = child.track_id
            referenced.update(child.parent_segment_ids)
            if child.translation is not None and child.current_translation is None:
                raise ValueError("Group translations must be completed and current")
        separated_lanes = {lane for lane in lanes if not lane.startswith("context:")}
        context_lanes = set(lanes) - separated_lanes
        if (len(separated_lanes) != 2 or len(context_lanes) > 1
                or len(set(lanes.values())) != len(lanes) or referenced != parents):
            raise ValueError("A group needs two distinct lanes and every parent must be represented")
        for context in (item for item in self.children if item.lane_id in context_lanes):
            if any(context.start_ns < item.end_ns and context.end_ns > item.start_ns
                   for item in self.children if item.lane_id in separated_lanes):
                raise ValueError("Unseparated context cannot overlap separated child captions")
        for lane in lanes:
            ordered = sorted((item for item in self.children if item.lane_id == lane), key=lambda item: item.start_ns)
            if any(left.end_ns > right.start_ns for left, right in zip(ordered, ordered[1:])):
                raise ValueError("Captions on one separated lane cannot overlap")
        # The protocol budget is smaller than the desktop pipe envelope budget.
        encoded = json.dumps(asdict(self), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 65536:
            raise ValueError("Caption group exceeds 65536 bytes")


@dataclass(frozen=True)
class CaptionSnapshot:
    session_id: str
    revision: int
    provider_generation: int
    segments: tuple[CaptionSegment, ...]
    speaker_aliases: tuple[tuple[str, str], ...] = ()
    changes: tuple[CaptionChange, ...] = ()
    held_groups: tuple[CaptionGroupReplacement, ...] = ()


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
        self._groups: dict[str, CaptionGroupReplacement] = {}
        self._held_groups: set[str] = set()
        self._held_children: dict[str, str] = {}
        self._lock = RLock()

    @classmethod
    def from_snapshot(cls, snapshot: CaptionSnapshot) -> CaptionStore:
        """Restore an already parsed local snapshot without replaying inference.

        Archived stale translations and held alternatives stay inspectable.
        Applying a held group still revalidates its source/provider revisions.
        Applied group transactions are not in v1 snapshots; their final child
        captions and parent tombstones are, and remain the authoritative view.
        """
        if not isinstance(snapshot, CaptionSnapshot):
            raise ValueError("Expected a caption snapshot")
        _integer(snapshot.revision, "revision")
        store = cls(snapshot.session_id, snapshot.provider_generation)
        segments = {item.segment_id: item for item in snapshot.segments}
        aliases = dict(snapshot.speaker_aliases)
        if len(segments) != len(snapshot.segments) or len(aliases) != len(snapshot.speaker_aliases):
            raise ValueError("Duplicate snapshot identity")
        for child in segments.values():
            if child.delivery_class != "overlap_correction":
                continue
            for parent_id in child.parent_segment_ids:
                parent = segments.get(parent_id)
                if (parent is None or child.segment_id not in parent.superseded_by
                        or child.source_track_id != parent.source_track_id
                        or child.capture_epoch != parent.capture_epoch
                        or child.end_ns <= parent.start_ns or child.start_ns >= parent.end_ns):
                    raise ValueError("Archived overlap child has inconsistent parent lineage")
        groups, held_children = {}, {}
        for group in snapshot.held_groups:
            if group.group_id in groups or any(parent.segment_id not in segments for parent in group.parents):
                raise ValueError("Invalid archived group identity or missing parent")
            for child in group.children:
                if child.segment_id in segments or child.segment_id in held_children:
                    raise ValueError("Archived held child identity is already in use")
                held_children[child.segment_id] = group.group_id
            groups[group.group_id] = group
        store._revision = snapshot.revision
        store._segments, store._aliases = segments, aliases
        store._changes = list(snapshot.changes)
        store._groups = groups
        store._held_groups = set(groups)
        store._held_children = held_children
        return store

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
                lineage = ("source_track_id", "capture_epoch", "separation_group_id", "lane_id",
                           "parent_segment_ids", "delivery_class")
                if any(getattr(current, name) != getattr(segment, name) for name in lineage):
                    raise ValueError("Source updates cannot change capture or separation lineage")
                if current.superseded_by or segment.source_revision <= current.source_revision:
                    return False
                segment = replace(segment, speaker_id=current.speaker_id,
                                  speaker_revision=current.speaker_revision,
                                  speaker_manual=current.speaker_manual,
                                  translation=current.translation)
            if segment.segment_id in self._held_children:
                return False  # A held alternative is revised only as a complete group.
            self._segments[segment.segment_id] = segment
            self._record("transcript.updated", asdict(segment))
            return True

    def apply_translation(self, segment_id: str, source_revision: int,
                          provider_generation: int, text: str, *,
                          target_language: str = "ko", completed: bool = True) -> bool:
        incoming = CaptionTranslation(source_revision, provider_generation,
                                      target_language, text, completed)
        with self._lock:
            current = self._lookup_segment(segment_id)
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
            if segment_id in self._held_children and not completed:
                return False  # Alternatives are presented as complete groups.
            self._put_segment(replace(current, translation=incoming))
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

    def revise_translations(self, corrections: tuple[CaptionResultCorrection, ...], *,
                            provider_generation: int) -> bool:
        """Atomically revise up to two completed results after new context.

        Validation happens before any mutation. Replayed, stale, manual, hidden,
        or pending alternative captions reject the complete batch. A revised
        result has its own revision, never a fabricated new source revision.
        """
        _integer(provider_generation, "provider_generation")
        if (not isinstance(corrections, tuple) or not 1 <= len(corrections) <= 2
                or any(not isinstance(item, CaptionResultCorrection) for item in corrections)
                or len({item.segment_id for item in corrections}) != len(corrections)):
            raise ValueError("Supply one or two distinct immutable result corrections")
        with self._lock:
            if provider_generation != self._generation:
                return False
            updates = []
            for correction in corrections:
                current = self._lookup_segment(correction.segment_id)
                if (current is None or current.superseded_by or current.speaker_manual
                        or correction.segment_id in self._held_children
                        or current.source_revision != correction.source_revision):
                    return False
                previous = current.current_translation
                if (previous is None or previous.provider_generation != provider_generation
                        or previous.result_revision != correction.based_on_result_revision
                        or correction.text == previous.text):
                    return False
                result = replace(previous, text=correction.text,
                                 result_revision=previous.result_revision + 1)
                updates.append(replace(current, translation=result))
            for segment in updates:
                self._put_segment(segment)
            self._record("caption.results_revised", {
                "provider_generation": provider_generation,
                "corrections": [asdict(item) for item in corrections]})
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
            current = self._lookup_segment(segment_id)
            if (current is None or current.superseded_by
                    or (manual and segment_id in self._held_children)
                    or speaker_revision <= current.speaker_revision
                    or (current.speaker_manual and not manual)
                    or (based_on_source_revision is not None
                        and based_on_source_revision != current.source_revision)):
                return False
            self._put_segment(replace(
                current, speaker_id=speaker_id, speaker_revision=speaker_revision,
                speaker_manual=manual,
            ))
            self._record("speaker.override" if manual else "speaker.updated", {
                "segment_id": segment_id, "speaker_id": speaker_id,
                "speaker_revision": speaker_revision,
                "based_on_source_revision": based_on_source_revision, "manual": manual,
            })
            return True

    def get_segment(self, segment_id: str) -> CaptionSegment | None:
        """Read an active/tombstoned caption or a held alternative by exact ID.

        Held alternatives are deliberately excluded from snapshot.segments and
        SRT/VTT until explicitly applied, but can receive independent translations.
        """
        with self._lock:
            return self._lookup_segment(segment_id)

    def _lookup_segment(self, segment_id: str) -> CaptionSegment | None:
        # Internal mutation checks must not invoke an overridden public read
        # hook, or count as a second consumer read of a speaker caption patch.
        group_id = self._held_children.get(segment_id)
        if group_id is not None:
            return next(item for item in self._groups[group_id].children if item.segment_id == segment_id)
        return self._segments.get(segment_id)

    def _put_segment(self, segment: CaptionSegment) -> None:
        group_id = self._held_children.get(segment.segment_id)
        if group_id is None:
            self._segments[segment.segment_id] = segment
        else:
            group = self._groups[group_id]
            self._groups[group_id] = replace(group, children=tuple(
                segment if item.segment_id == segment.segment_id else item for item in group.children))

    def held_group_for_segment(self, segment_id: str) -> str | None:
        with self._lock:
            return self._held_children.get(segment_id)

    def get_group(self, group_id: str) -> CaptionGroupReplacement | None:
        with self._lock:
            return self._groups.get(group_id)

    def group_status(self, group_id: str) -> str | None:
        with self._lock:
            return ("held" if group_id in self._held_groups else "applied") if group_id in self._groups else None

    def _validate_group(self, group: CaptionGroupReplacement, max_segments: int,
                        *, applying_held: bool = False) -> bool:
        """Under the store lock, validate the whole transaction without mutation."""
        if not isinstance(group, CaptionGroupReplacement):
            raise ValueError("Expected CaptionGroupReplacement")
        _integer(max_segments, "max_segments", 1)
        previous = self._groups.get(group.group_id)
        if previous is not None and not applying_held and group.group_revision <= previous.group_revision:
            return False
        old_held = {item.segment_id for item in previous.children} if group.group_id in self._held_groups else set()
        current_ids = set(self._segments) | set(self._held_children)
        reusable = old_held if applying_held else set()
        if any(item.segment_id in current_ids - reusable for item in group.children):
            raise ValueError("Replacement children must have fresh session IDs")
        if len(current_ids - old_held) + len(group.children) > max_segments:
            raise ValueError("Caption segment limit reached; group was not stored")
        for parent in group.parents:
            current = self._segments.get(parent.segment_id)
            if (current is None or current.superseded_by
                    or current.source_revision != parent.source_revision):
                return False
            if (current.source_track_id != group.source_track_id
                    or current.capture_epoch != group.capture_epoch
                    or current.start_ns < group.start_ns or current.end_ns > group.end_ns):
                raise ValueError("Group must preserve parent capture epoch and complete time span")
            related = [child for child in group.children if parent.segment_id in child.parent_segment_ids]
            if any(child.end_ns <= current.start_ns or child.start_ns >= current.end_ns for child in related):
                raise ValueError("Child parent references must intersect in source time")
        for child in group.children:
            if child.translation is not None and child.translation.provider_generation != self._generation:
                raise ValueError("Group translation provider generation must be current")
        return True

    def replace_group(self, group: CaptionGroupReplacement, *, max_segments: int = 10000,
                      allow_manual: bool = False) -> Literal["applied", "held", "stale"]:
        """Atomically insert two lanes and tombstone their mixed parents.

        A manual parent assignment retains the original caption and stores the
        complete alternative group. Explicit local acceptance may replace it;
        parent speaker IDs are never copied onto either separated child.
        """
        if type(allow_manual) is not bool:
            raise ValueError("allow_manual must be a bool")
        with self._lock:
            if not self._validate_group(group, max_segments):
                return "stale"
            manual = any(self._segments[item.segment_id].speaker_manual for item in group.parents)
            status = "held" if manual and not allow_manual else "applied"
            if status == "held" and group.group_id not in self._held_groups and len(self._held_groups) >= 64:
                raise ValueError("Held caption group limit reached")
            self._commit_group(group, status)
            return status

    def _commit_group(self, group: CaptionGroupReplacement, status: str) -> None:
        # Build every immutable replacement before changing any group/index map.
        updates = {}
        if status == "applied":
            updates = {item.segment_id: item for item in group.children}
            for parent in group.parents:
                ids = tuple(item.segment_id for item in group.children if parent.segment_id in item.parent_segment_ids)
                updates[parent.segment_id] = replace(self._segments[parent.segment_id], superseded_by=ids)
        previous = self._groups.get(group.group_id)
        if group.group_id in self._held_groups:
            for item in previous.children:
                self._held_children.pop(item.segment_id)
            self._held_groups.remove(group.group_id)
        self._groups[group.group_id] = group
        if status == "held":
            self._held_groups.add(group.group_id)
            self._held_children.update({item.segment_id: group.group_id for item in group.children})
        else:
            self._segments.update(updates)
        self._record("caption.group_held" if status == "held" else "caption.group_replaced", asdict(group))

    def apply_held_group(self, group_id: str, group_revision: int, *,
                         max_segments: int = 10000) -> bool:
        """Explicit local action; stale parents/capacity are checked again."""
        _text(group_id, "group_id")
        _integer(group_revision, "group_revision", 1)
        with self._lock:
            group = self._groups.get(group_id)
            if (group_id not in self._held_groups or group.group_revision != group_revision
                    or not self._validate_group(group, max_segments, applying_held=True)):
                return False
            self._commit_group(group, "applied")
            return True

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
                tuple(self._groups[key] for key in sorted(self._held_groups)),
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
        "held_groups": [asdict(item) for item in snapshot.held_groups],
        "schema_extensions": ["caption_groups_v1"],
        "journal_scope": "accepted_source_completed_translation_speaker_provider_and_supersession",
    }
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
