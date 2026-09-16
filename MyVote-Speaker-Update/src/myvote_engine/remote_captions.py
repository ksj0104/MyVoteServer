"""Typed projection of gateway events into client-side source-timed captions.

No protobuf, speech model, or RPC dependency is imported. ``events.jsonl`` owned
by the gateway client is the event journal; this reducer keeps current state only.
The gateway can optionally run experimental local speaker models. This reducer
applies their patches and preserves local manual assignments and names.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import Any

from .captions import (CaptionGroupReplacement, CaptionParentRef, CaptionSegment, CaptionResultCorrection,
                       CaptionSnapshot, CaptionStore, CaptionTranslation)


MAX_DATA_BYTES = 262144
UINT64_MAX = 2**64 - 1


class RemoteCaptionError(ValueError):
    """An event cannot safely describe the active client's caption session."""


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= UINT64_MAX:
        raise RemoteCaptionError(f"{name} must be an integer in [{minimum}, {UINT64_MAX}]")
    return value


def _text(value: Any, name: str, maximum: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise RemoteCaptionError(f"Invalid {name}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RemoteCaptionError(f"Invalid Unicode in {name}") from exc
    return value


def _optional_text(value: Any, name: str, maximum: int) -> str | None:
    return None if value is None else _text(value, name, maximum)


def _json_object(data: Any) -> str:
    if not isinstance(data, dict):
        raise RemoteCaptionError("Event data must be an object")
    try:
        serialized = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > MAX_DATA_BYTES:
            raise RemoteCaptionError("Event data exceeds the projection limit")
    except (TypeError, ValueError, RecursionError, UnicodeEncodeError) as exc:
        raise RemoteCaptionError("Event data must be bounded finite JSON") from exc
    return serialized


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RemoteCaptionError("Duplicate event JSON key")
        result[key] = value
    return result


class _CurrentCaptionStore(CaptionStore):
    """The owning client records wire events; do not duplicate its journal in RAM."""

    def _record(self, event_type: str, payload: dict, *, retain: bool = True) -> None:
        super()._record(event_type, payload, retain=False)


@dataclass(frozen=True)
class TranslationStatus:
    kind: str
    source_revision: int
    provider_generation: int
    error: str | None = None


@dataclass(frozen=True)
class _SourceIdentity:
    revision: int
    track_id: str
    target_language: str


@dataclass(frozen=True)
class _PreviewIdentity:
    revision: int
    source_track_id: str
    capture_epoch: str
    target_language: str
    frontier_ns: int
    cleared: bool


def parse_caption_group(data: dict) -> CaptionGroupReplacement:
    """Parse bounded wire values before the store performs contextual validation."""
    if set(data) != {"group_id", "group_revision", "source_track_id", "capture_epoch",
                     "start_ns", "end_ns", "parents", "children"}:
        raise RemoteCaptionError("Invalid caption group fields")
    if not isinstance(data["parents"], list) or not 1 <= len(data["parents"]) <= 16:
        raise RemoteCaptionError("Invalid caption group parents")
    if not isinstance(data["children"], list) or not 2 <= len(data["children"]) <= 32:
        raise RemoteCaptionError("Invalid caption group children")
    parents = []
    for item in data["parents"]:
        if not isinstance(item, dict) or set(item) != {"segment_id", "source_revision"}:
            raise RemoteCaptionError("Invalid caption parent reference")
        parents.append(CaptionParentRef(_text(item["segment_id"], "segment_id", 512),
                                       _integer(item["source_revision"], "source_revision", 1)))
    children = []
    allowed = set(CaptionSegment.__dataclass_fields__)
    for item in data["children"]:
        if not isinstance(item, dict) or set(item) - allowed:
            raise RemoteCaptionError("Invalid caption child fields")
        values = dict(item)
        for field, maximum in (("segment_id", 512), ("track_id", 256), ("source_text", 12000),
                               ("target_language", 64), ("source_track_id", 256),
                               ("capture_epoch", 256), ("separation_group_id", 512), ("lane_id", 256)):
            values[field] = _text(values[field], field, maximum)
        for field, minimum in (("start_ns", 0), ("end_ns", 1), ("source_revision", 1)):
            values[field] = _integer(values[field], field, minimum)
        values["speaker_revision"] = _integer(values.get("speaker_revision", 0), "speaker_revision")
        values["speaker_id"] = _optional_text(values.get("speaker_id"), "speaker_id", 256)
        for field in ("parent_segment_ids", "superseded_by"):
            entries = values.get(field, [])
            if not isinstance(entries, list) or len(entries) > 16:
                raise RemoteCaptionError(f"Invalid {field}")
            values[field] = tuple(_text(value, field, 512) for value in entries)
        translation = values.get("translation")
        if translation is not None:
            if (not isinstance(translation, dict)
                    or set(translation) - set(CaptionTranslation.__dataclass_fields__)
                    or not {"source_revision", "provider_generation", "target_language", "text", "completed"} <= set(translation)):
                raise RemoteCaptionError("Invalid child translation")
            values["translation"] = CaptionTranslation(
                _integer(translation["source_revision"], "source_revision", 1),
                _integer(translation["provider_generation"], "provider_generation"),
                _text(translation["target_language"], "target_language", 64),
                _text(translation["text"], "translation text", 24000), translation["completed"],
                _integer(translation.get("result_revision", 1), "result_revision", 1))
        children.append(CaptionSegment(**values))
    return CaptionGroupReplacement(
        _text(data["group_id"], "group_id", 512), _integer(data["group_revision"], "group_revision", 1),
        _text(data["source_track_id"], "source_track_id", 256),
        _text(data["capture_epoch"], "capture_epoch", 256),
        _integer(data["start_ns"], "start_ns"), _integer(data["end_ns"], "end_ns", 1),
        tuple(parents), tuple(children))


class RemoteCaptionProjection:
    """Reduce one ordered session stream, retaining at most ``max_segments`` IDs.

    ``apply`` returns False for stale sequence/revision/results or a patch blocked
    by a local manual speaker override. Unknown events and provisional transcript
    events consume their sequence and return True without changing the store.
    Sequence gaps are allowed here; the RPC client owns complete-stream checks.
    Bad session identity or malformed typed payload raises RemoteCaptionError.
    Validation failure does not advance the event sequence or mutate the store.

    Local speaker names and overrides use ``store.set_speaker_alias`` and
    ``store.update_speaker(..., manual=True)``. No model is called by these APIs.
    """

    def __init__(self, session_id: str, *, target_language: str = "ko",
                 track_id: str = "system", max_segments: int = 10000,
                 capabilities: tuple[str, ...] = ("caption_groups_v1", "caption_refinement_v1", "semantic_translation_v1")):
        self.session_id = _text(session_id, "session_id", 256)
        self.target_language = _text(target_language, "target_language", 64)
        self.track_id = _text(track_id, "track_id", 256)
        _integer(max_segments, "max_segments", 1)
        if max_segments > 10000:
            raise RemoteCaptionError("Remote projection supports at most 10000 segment IDs")
        self.max_segments = max_segments
        if (not isinstance(capabilities, tuple) or len(capabilities) > 4
                or any(item not in ("caption_groups_v1", "caption_words_v1", "caption_refinement_v1", "semantic_translation_v1") for item in capabilities)
                or len(set(capabilities)) != len(capabilities)):
            raise RemoteCaptionError("Unsupported client caption capabilities")
        self.supported_capabilities = frozenset(capabilities)
        self.capabilities: frozenset[str] = frozenset()
        self._started = False
        self.store = _CurrentCaptionStore(self.session_id)
        self.last_event_sequence = 0
        self.provider_generation = 0
        self.terminal_kind: str | None = None
        self._terminal_json: str | None = None
        self._sources: dict[str, _SourceIdentity] = {}
        self._translation_status: dict[str, TranslationStatus] = {}
        # Ephemeral display state is deliberately outside CaptionStore/snapshot.
        # Retain a bounded revision/retirement record, never transcript history.
        self._preview_id: str | None = None
        self._preview_json: str | None = None
        self._preview_frontier_ns = -1
        self._preview_identities: dict[str, _PreviewIdentity] = {}
        self.last_event_kind: str | None = None
        self.stats = dict(accepted_events=0, source_updates=0, translation_previews=0,
                          translation_completed=0, translation_failed=0,
                          speaker_updates=0, ignored_events=0, stale_events=0)
        self.stats.update(group_updates=0, groups_held=0)
        self.stats.update(transcript_previews=0)

    @property
    def segment_count(self) -> int:
        return len(self._sources)

    @property
    def terminal(self) -> dict | None:
        return json.loads(self._terminal_json) if self._terminal_json is not None else None

    @property
    def terminal_counts(self) -> dict[str, int] | None:
        terminal = self.terminal
        return terminal.get("counts") if terminal is not None else None

    @property
    def translation_status(self) -> dict[str, TranslationStatus]:
        return dict(self._translation_status)

    def snapshot(self) -> CaptionSnapshot:
        return self.store.snapshot()

    @property
    def transcript_preview(self) -> dict | None:
        """Last accepted display payload, including clear, detached from state."""
        return json.loads(self._preview_json) if self._preview_json is not None else None

    def _segment_id(self, segment_id: str | None) -> str:
        return _text(segment_id, "segment_id", 512)

    def _set_generation(self, generation: int) -> None:
        if generation > self.provider_generation:
            self.store.set_provider_generation(generation)
            self.provider_generation = generation

    def apply(self, session_id: str, event_sequence: int, kind: str,
              segment_id: str | None, data: dict) -> bool:
        _text(session_id, "session_id", 256)
        if session_id != self.session_id:
            raise RemoteCaptionError("Event belongs to a different caption session")
        _integer(event_sequence, "event_sequence", 1)
        _text(kind, "kind", 128)
        if segment_id is not None:
            _text(segment_id, "segment_id", 512, empty=True)
        if event_sequence <= self.last_event_sequence:
            self.stats["stale_events"] += 1
            return False
        if self.terminal_kind is not None:
            raise RemoteCaptionError("Event arrived after the session terminal")
        serialized = _json_object(data)
        # Use a detached, standard JSON object throughout validation and mutation.
        payload = json.loads(serialized)
        try:
            accepted = self._apply_typed(kind, segment_id or None, payload, serialized)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RemoteCaptionError):
                raise
            raise RemoteCaptionError(f"Invalid {kind} payload: {exc}") from exc
        self.last_event_sequence = event_sequence
        self.last_event_kind = kind
        self.stats["accepted_events"] += 1
        if not accepted:
            self.stats["stale_events"] += 1
        return accepted

    def _apply_typed(self, kind: str, segment_id: str | None, data: dict,
                     serialized: str) -> bool:
        if kind == "transcript.preview":
            return self._apply_transcript_preview(data, serialized)
        if kind == "caption.results_revised":
            if "caption_refinement_v1" not in self.capabilities or segment_id is not None:
                raise RemoteCaptionError("Context refinement was not negotiated or has an invalid envelope")
            if set(data) != {"provider_generation", "corrections"}:
                raise RemoteCaptionError("Invalid context correction fields")
            generation = _integer(data["provider_generation"], "provider_generation")
            rows = data["corrections"]
            if not isinstance(rows, list) or not 1 <= len(rows) <= 2:
                raise RemoteCaptionError("Expected one or two context corrections")
            corrections = []
            for row in rows:
                if not isinstance(row, dict) or set(row) != set(CaptionResultCorrection.__dataclass_fields__):
                    raise RemoteCaptionError("Invalid context correction")
                corrections.append(CaptionResultCorrection(
                    _text(row["segment_id"], "segment_id", 512),
                    _integer(row["source_revision"], "source_revision", 1),
                    _integer(row["based_on_result_revision"], "based_on_result_revision", 1),
                    _text(row["text"], "corrected text", 24000)))
            # A revision event never changes the active provider generation.
            accepted = self.store.revise_translations(tuple(corrections), provider_generation=generation)
            return accepted
        if kind == "caption.source":
            segment_id = self._segment_id(segment_id)
            revision = _integer(data["source_revision"], "source_revision", 1)
            text = _text(data["text"], "source text", 12000)
            start = _integer(data["start_ns"], "start_ns")
            end = _integer(data["end_ns"], "end_ns")
            track = _text(data.get("track_id", self.track_id), "track_id", 256)
            target = _text(data.get("target_language", self.target_language), "target_language", 64)
            if data.get("source_state", "stable") != "stable":
                raise RemoteCaptionError("caption.source must describe stable source text")
            segment = CaptionSegment(segment_id, track, start, end, text, revision,
                                     target_language=target,
                                     text_operation=data.get("text_operation", "translation"),
                                     source_language=_optional_text(data.get("source_language"), "source_language", 64),
                                     source_track_id=_optional_text(data.get("source_track_id"), "source_track_id", 256),
                                     capture_epoch=_optional_text(data.get("capture_epoch"), "capture_epoch", 256))
            if any(data.get(name) for name in ("separation_group_id", "lane_id", "parent_segment_ids")) or data.get("delivery_class", "live") != "live":
                raise RemoteCaptionError("Separated source children require an atomic caption group")
            current = self._sources.get(segment_id)
            if current is not None and revision <= current.revision:
                return False
            if current is None and len(self._sources) >= self.max_segments:
                raise RemoteCaptionError("Caption segment limit reached; source was not stored")
            if current is not None and current.track_id != track:
                raise RemoteCaptionError("Source segment cannot move to another track")
            accepted = self.store.upsert_source(segment)
            if accepted:
                self._sources[segment_id] = _SourceIdentity(revision, track, target)
                self._translation_status.pop(segment_id, None)
                self.stats["source_updates"] += 1
            return accepted

        if kind == "caption.group_replaced":
            if "caption_groups_v1" not in self.capabilities:
                raise RemoteCaptionError("Caption groups were not negotiated at session start")
            if segment_id is not None:
                raise RemoteCaptionError("A caption group cannot name one envelope segment")
            group = parse_caption_group(data)
            previous = self.store.get_group(group.group_id)
            if previous is not None and group.group_revision <= previous.group_revision:
                return False
            new_ids = {item.segment_id for item in group.children}
            if new_ids & set(self._sources):
                raise RemoteCaptionError("Group children must use fresh session IDs")
            if len(self._sources) + len(new_ids) > self.max_segments:
                raise RemoteCaptionError("Caption segment limit reached; group was not stored")
            status = self.store.replace_group(group, max_segments=self.max_segments)
            if status == "stale":
                return False
            for child in group.children:
                self._sources[child.segment_id] = _SourceIdentity(child.source_revision, child.track_id, child.target_language)
                if child.translation is not None:
                    item = child.translation
                    self._translation_status[child.segment_id] = TranslationStatus(
                        "translation.completed", item.source_revision, item.provider_generation)
            self.stats["group_updates"] += 1
            self.stats["groups_held"] += status == "held"
            return True

        if kind in ("translation.preview", "translation.completed", "translation.failed", "translation.cancelled"):
            segment_id = self._segment_id(segment_id)
            current = self._sources.get(segment_id)
            stored = self.store.get_segment(segment_id)
            completed_or_preview = kind in ("translation.preview", "translation.completed")
            # Pipeline failures before a TranslationRequest exists contain only
            # the clause ID and error. Bind those to the current clause identity.
            revision_value = data.get("source_revision", current.revision if current else None)
            generation_value = data.get("provider_generation", self.provider_generation)
            if completed_or_preview and ("source_revision" not in data or "provider_generation" not in data):
                raise RemoteCaptionError("Translation text requires source revision and provider generation")
            revision = _integer(revision_value, "source_revision", 1)
            generation = _integer(generation_value, "provider_generation")
            target = _text(data.get("target_language", current.target_language if current else self.target_language),
                           "target_language", 64)
            error = _optional_text(data.get("error"), "translation error", 2048)
            text = (_text(data.get("text", ""), "translation text", 24000,
                          empty=kind != "translation.completed"))
            if (current is None or stored is None or stored.superseded_by
                    or current.revision != revision or current.target_language != target
                    or generation < self.provider_generation):
                return False
            previous = self._translation_status.get(segment_id)
            if (previous is not None and previous.source_revision == revision
                    and previous.provider_generation == generation
                    and previous.kind in ("translation.completed", "translation.failed", "translation.cancelled")):
                return False
            if kind == "translation.preview" and not text.strip():
                self.stats["ignored_events"] += 1
                return True
            if kind == "translation.preview" and self.store.held_group_for_segment(segment_id) is not None:
                return False
            held_id = self.store.held_group_for_segment(segment_id)
            if held_id is not None and kind == "translation.completed":
                # A growing held group can exceed the atomic event budget.
                # Construct the complete prospective group before advancing a
                # provider generation or touching any caption state.
                group = self.store.get_group(held_id)
                incoming = CaptionTranslation(revision, generation, target, text)
                replace(group, children=tuple(replace(item, translation=incoming)
                    if item.segment_id == segment_id else item for item in group.children))
            # Stale/unknown source events above cannot spuriously advance the
            # generation and invalidate otherwise valid translation jobs.
            self._set_generation(generation)
            if completed_or_preview:
                accepted = self.store.apply_translation(
                    segment_id, revision, generation, text, target_language=target,
                    completed=kind == "translation.completed")
                if not accepted:
                    return False
                counter = "translation_completed" if kind == "translation.completed" else "translation_previews"
            else:
                # The source and any non-final preview remain preserved. Export
                # ignores the preview; this status explains why it was unfinished.
                counter = "translation_failed"
            self._translation_status[segment_id] = TranslationStatus(kind, revision, generation, error)
            self.stats[counter] += 1
            return True

        if kind == "provider.updated":
            generation = _integer(data["provider_generation"], "provider_generation")
            if generation <= self.provider_generation:
                return False
            self._set_generation(generation)
            return True

        if kind == "speaker.alias_updated":
            speaker_id = _text(data["speaker_id"], "speaker_id", 256)
            name = _text(data["name"], "speaker name", 256)
            aliases = dict(self.store.speaker_aliases())
            if speaker_id in aliases:
                return False  # Preserve a name assigned locally by the user.
            if len(aliases) >= 128:
                raise RemoteCaptionError("Speaker name capacity exceeded")
            return self.store.set_speaker_alias(speaker_id, name)

        if kind == "speaker.updated":
            segment_id = self._segment_id(segment_id)
            speaker_id = _optional_text(data["speaker_id"], "speaker_id", 256)
            revision = _integer(data["speaker_revision"], "speaker_revision", 1)
            basis = data.get("based_on_source_revision")
            if basis is not None:
                _integer(basis, "based_on_source_revision", 1)
            if "manual" in data and data["manual"] is not False:
                raise RemoteCaptionError("Automatic speaker updates cannot request a manual override")
            accepted = self.store.update_speaker(segment_id, speaker_id, revision,
                                                 based_on_source_revision=basis)
            if accepted:
                self.stats["speaker_updates"] += 1
            return accepted

        if kind in ("session.completed", "session.failed"):
            if kind == "session.completed" and "counts" not in data:
                raise RemoteCaptionError("Completed session must contain terminal counts")
            if "counts" in data:
                counts = data["counts"]
                if not isinstance(counts, dict) or len(counts) > 128:
                    raise RemoteCaptionError("Invalid terminal counts")
                for name, value in counts.items():
                    _text(name, "count name", 128)
                    _integer(value, f"counts.{name}")
            if kind == "session.failed":
                _text(data["error"], "session error", 2048)
            self.terminal_kind = kind
            self._terminal_json = serialized
            return True

        if kind == "session.started":
            if self._started:
                raise RemoteCaptionError("Session start was already accepted")
            if _integer(data["protocol_version"], "protocol_version", 1) != 1:
                raise RemoteCaptionError("Unsupported gateway protocol")
            if data.get("engine_mode") not in ("configured_local_engines", "injected_test_engines"):
                raise RemoteCaptionError("Invalid gateway engine mode")
            _text(data["speaker_status"], "speaker_status", 128)
            capabilities = data.get("capabilities", [])
            if (not isinstance(capabilities, list) or len(capabilities) > 32
                    or any(not isinstance(item, str) for item in capabilities)
                    or len(set(capabilities)) != len(capabilities)
                    or any(item not in self.supported_capabilities for item in capabilities)):
                raise RemoteCaptionError("Invalid or unsupported negotiated capabilities")
            self.capabilities = frozenset(capabilities)
            self._started = True

        if kind == "transcript.updated":
            # These are window-level hypotheses. Even a stable window is not a
            # caption.source clause and must not create duplicate exported text.
            self._segment_id(segment_id)
            if "revision" in data:
                _integer(data["revision"], "transcript revision", 1)
            if "text" in data:
                _text(data["text"], "transcript text", 24000, empty=True)
            if "source_state" in data and data["source_state"] not in ("provisional", "stable"):
                raise RemoteCaptionError("Invalid transcript source state")
        self.stats["ignored_events"] += 1
        return True

    def _apply_transcript_preview(self, data: dict, serialized: str) -> bool:
        if "semantic_translation_v1" not in self.capabilities:
            raise RemoteCaptionError("Semantic transcript previews were not negotiated")
        if set(data) != {"preview_id", "revision", "source_text", "stable_text", "source_language",
                         "target_language", "source_track_id", "capture_epoch", "start_ns", "end_ns", "state"}:
            raise RemoteCaptionError("Invalid transcript preview fields")
        identity = _text(data["preview_id"], "preview_id", 512)
        revision = _integer(data["revision"], "preview revision", 1)
        source = _text(data["source_text"], "preview source text", 24000, empty=True)
        stable = _text(data["stable_text"], "preview stable text", 24000, empty=True)
        _text(data["source_language"], "source_language", 64)
        target = _text(data["target_language"], "target_language", 64)
        track = _text(data["source_track_id"], "source_track_id", 256)
        epoch = _text(data["capture_epoch"], "capture_epoch", 256)
        start = _integer(data["start_ns"], "preview start_ns")
        end = _integer(data["end_ns"], "preview end_ns")
        state = data["state"]
        if state not in ("listening", "buffering", "cleared") or end < start:
            raise RemoteCaptionError("Invalid transcript preview state or interval")
        if state == "cleared" and (source != "" or stable != "" or start != end):
            raise RemoteCaptionError("Cleared preview must contain empty text at one audio boundary")
        if target != self.target_language:
            raise RemoteCaptionError("Transcript preview target differs from the session")
        previous = self._preview_identities.get(identity)
        if previous is not None:
            if (track, epoch, target) != (previous.source_track_id, previous.capture_epoch, previous.target_language):
                raise RemoteCaptionError("Transcript preview lineage cannot change")
            if identity != self._preview_id or revision <= previous.revision:
                return False
            if previous.cleared and state != "cleared" and end <= previous.frontier_ns:
                return False
        else:
            if end <= self._preview_frontier_ns:
                return False
            current = self._preview_identities.get(self._preview_id)
            if state == "cleared" and current is not None and not current.cleared:
                return False  # A different/unknown ID cannot clear a live preview.
        # Same-ID provisional timestamps and auto-detected language may change.
        # Only new IDs are held behind the monotonic audio frontier. Retired IDs
        # in the bounded history cannot take over the live display again.
        self._preview_identities[identity] = _PreviewIdentity(revision, track, epoch, target,
            max(previous.frontier_ns if previous else -1, end), state == "cleared")
        self._preview_id = identity
        self._preview_json = serialized
        self._preview_frontier_ns = max(self._preview_frontier_ns, end)
        while len(self._preview_identities) > 256:
            retired = next(key for key in self._preview_identities if key != identity)
            del self._preview_identities[retired]
        self.stats["transcript_previews"] += 1
        return True

    def apply_message(self, message: Any) -> bool:
        """Accept a ServerMessage-like object without importing generated protobuf."""
        try:
            raw = _text(message.data_json, "data_json", MAX_DATA_BYTES, empty=False)
            if len(raw.encode("utf-8")) > MAX_DATA_BYTES:
                raise RemoteCaptionError("Event JSON exceeds the projection limit")
            elapsed = message.server_elapsed_ms
            if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed) or elapsed < 0):
                raise RemoteCaptionError("Invalid server processing duration")
            data = json.loads(raw, object_pairs_hook=_unique_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  RemoteCaptionError("Non-finite event JSON")))
            return self.apply(message.session_id, message.event_sequence, message.kind,
                              message.segment_id or None, data)
        except (AttributeError, TypeError, ValueError, RecursionError) as exc:
            if isinstance(exc, RemoteCaptionError):
                raise
            raise RemoteCaptionError("Invalid gateway event message") from exc
