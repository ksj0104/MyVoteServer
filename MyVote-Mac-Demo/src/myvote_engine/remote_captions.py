"""Typed projection of gateway events into client-side source-timed captions.

No protobuf, speech model, or RPC dependency is imported. ``events.jsonl`` owned
by the gateway client is the event journal; this reducer keeps current state only.
The gateway can optionally run experimental local speaker models. This reducer
applies their patches and preserves local manual assignments and names.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

from .captions import CaptionSegment, CaptionSnapshot, CaptionStore


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
                 track_id: str = "system", max_segments: int = 10000):
        self.session_id = _text(session_id, "session_id", 256)
        self.target_language = _text(target_language, "target_language", 64)
        self.track_id = _text(track_id, "track_id", 256)
        _integer(max_segments, "max_segments", 1)
        if max_segments > 10000:
            raise RemoteCaptionError("Remote projection supports at most 10000 segment IDs")
        self.max_segments = max_segments
        self.store = _CurrentCaptionStore(self.session_id)
        self.last_event_sequence = 0
        self.provider_generation = 0
        self.terminal_kind: str | None = None
        self._terminal_json: str | None = None
        self._sources: dict[str, _SourceIdentity] = {}
        self._translation_status: dict[str, TranslationStatus] = {}
        self.last_event_kind: str | None = None
        self.stats = dict(accepted_events=0, source_updates=0, translation_previews=0,
                          translation_completed=0, translation_failed=0,
                          speaker_updates=0, ignored_events=0, stale_events=0)

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
                                     target_language=target)
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

        if kind in ("translation.preview", "translation.completed", "translation.failed", "translation.cancelled"):
            segment_id = self._segment_id(segment_id)
            current = self._sources.get(segment_id)
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
            if (current is None or current.revision != revision or current.target_language != target
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
            if _integer(data["protocol_version"], "protocol_version", 1) != 1:
                raise RemoteCaptionError("Unsupported gateway protocol")
            if data.get("engine_mode") not in ("configured_local_engines", "injected_test_engines"):
                raise RemoteCaptionError("Invalid gateway engine mode")
            _text(data["speaker_status"], "speaker_status", 128)

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
