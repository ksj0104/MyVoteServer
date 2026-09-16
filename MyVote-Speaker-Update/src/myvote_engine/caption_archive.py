"""Bounded local caption archive loading; no models, capture, RPC or credentials."""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re

from .captions import (CaptionChange, CaptionGroupReplacement, CaptionParentRef, CaptionSegment,
                       CaptionSnapshot, CaptionStore, CaptionTranslation)


MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_SEGMENTS = 10000
MAX_ARCHIVE_CHANGES = 100000
MAX_HISTORY_ENTRIES = 100
MAX_SCAN_ENTRIES = 10000
MAX_LIST_CANDIDATES = 300
MAX_INTEGER = 2**63 - 1
_SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")


class ArchiveError(ValueError):
    def __init__(self, category="history_invalid"):
        super().__init__(category)
        self.category = category


def _integer(value, minimum=0, maximum=MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ArchiveError()
    return value


def _text(value, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ArchiveError()
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ArchiveError() from exc
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError()
        result[key] = value
    return result


def _finite(token):
    value = float(token)
    if not math.isfinite(value):
        raise ArchiveError()
    return value


def _depth(value, depth=0):
    if depth > 30:
        raise ArchiveError()
    if isinstance(value, dict):
        for item in value.values():
            _depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _depth(item, depth + 1)


def _object(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ArchiveError()
    return value


def _array(value, maximum):
    if not isinstance(value, list) or len(value) > maximum:
        raise ArchiveError("history_limit")
    return value


def _segment(value):
    required = {"segment_id", "track_id", "start_ns", "end_ns", "source_text", "source_revision"}
    allowed = {field.name for field in fields(CaptionSegment)} | {"translation_current"}
    values = dict(_object(value, required, allowed - required))
    marked_current = values.pop("translation_current", None)
    if "translation_current" in value and type(marked_current) is not bool:
        raise ArchiveError()
    for name, maximum in (("segment_id", 512), ("track_id", 256), ("source_text", 12000),
                          ("target_language", 64), ("speaker_id", 256), ("source_track_id", 256),
                          ("capture_epoch", 256), ("separation_group_id", 512), ("lane_id", 256),
                          ("source_language", 64)):
        if name in values and values[name] is not None:
            _text(values[name], maximum)
    for name in ("start_ns", "end_ns", "source_revision", "speaker_revision"):
        if name in values:
            _integer(values[name])
    for name in ("parent_segment_ids", "superseded_by"):
        values[name] = tuple(_text(item, 512) for item in _array(values.get(name, []), 32))
        if len(set(values[name])) != len(values[name]):
            raise ArchiveError()
    translation = values.get("translation")
    if translation is not None:
        translation = dict(_object(translation, {"source_revision", "provider_generation", "target_language", "text"},
                                   {"completed", "result_revision"}))
        for name in ("source_revision", "provider_generation", "result_revision"):
            if name in translation:
                _integer(translation[name])
        _text(translation["target_language"], 64)
        _text(translation["text"], 24000)
        values["translation"] = CaptionTranslation(**translation)
    segment = CaptionSegment(**values)
    if marked_current is not None and marked_current != (segment.current_translation is not None):
        raise ArchiveError()
    return segment


def _group(value):
    values = dict(_object(value, {field.name for field in fields(CaptionGroupReplacement)}))
    for name in ("group_id", "source_track_id", "capture_epoch"):
        _text(values[name], 512 if name == "group_id" else 256)
    for name in ("group_revision", "start_ns", "end_ns"):
        _integer(values[name])
    parents = []
    for parent in _array(values["parents"], 16):
        _object(parent, {"segment_id", "source_revision"})
        parents.append(CaptionParentRef(_text(parent["segment_id"], 512), _integer(parent["source_revision"], 1)))
    values["parents"] = tuple(parents)
    values["children"] = tuple(_segment(item) for item in _array(values["children"], 32))
    return CaptionGroupReplacement(**values)


def decode_snapshot(raw: bytes, *, expected_session_id: str | None = None) -> CaptionSnapshot:
    """Validate a v1 export completely before changing the selected desktop view."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_ARCHIVE_BYTES:
        raise ArchiveError("history_limit")
    try:
        data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique, parse_float=_finite,
                          parse_constant=lambda _: (_ for _ in ()).throw(ArchiveError()))
        _depth(data)
        _object(data, {"schema", "schema_version", "session_id", "revision", "provider_generation",
                       "time_unit", "time_basis", "segments", "speaker_aliases"},
                      {"changes", "held_groups", "schema_extensions", "journal_scope"})
        if (data["schema"] != "myvote.caption_snapshot" or type(data["schema_version"]) is not int
                or data["schema_version"] != 1 or data["time_unit"] != "ns"
                or data["time_basis"] != "source_audio_session"):
            raise ArchiveError()
        session_id = _text(data["session_id"], 256)
        if expected_session_id is not None and session_id != expected_session_id:
            raise ArchiveError()
        aliases = data["speaker_aliases"]
        if not isinstance(aliases, dict) or len(aliases) > 128:
            raise ArchiveError("history_limit")
        aliases = tuple((_text(key, 256), _text(name, 256)) for key, name in aliases.items())
        segments = tuple(_segment(item) for item in _array(data["segments"], MAX_ARCHIVE_SEGMENTS))
        held = tuple(_group(item) for item in _array(data.get("held_groups", []), 64))
        if len(segments) + sum(len(group.children) for group in held) > MAX_ARCHIVE_SEGMENTS:
            raise ArchiveError("history_limit")
        known_speakers = {key for key, _ in aliases} | {item.speaker_id for item in segments if item.speaker_id}
        known_speakers.update(child.speaker_id for group in held for child in group.children if child.speaker_id)
        if len(known_speakers) > 128:
            raise ArchiveError("history_limit")
        revision = _integer(data["revision"])
        changes, previous = [], 0
        for item in _array(data.get("changes", []), MAX_ARCHIVE_CHANGES):
            _object(item, {"event_seq", "event_type", "payload"})
            sequence = _integer(item["event_seq"], previous + 1, revision)
            changes.append(CaptionChange(sequence, _text(item["event_type"], 256),
                json.dumps(item["payload"], ensure_ascii=False, allow_nan=False, separators=(",", ":"))))
            previous = sequence
        extensions = data.get("schema_extensions", [])
        if not isinstance(extensions, list) or any(item != "caption_groups_v1" for item in extensions) or len(extensions) > 1:
            raise ArchiveError()
        snapshot = CaptionSnapshot(session_id, revision, _integer(data["provider_generation"]),
                                   segments, aliases, tuple(changes), held)
        CaptionStore.from_snapshot(snapshot)  # Validate cross-record identities before committing.
        return snapshot
    except ArchiveError:
        raise
    except (ValueError, TypeError, KeyError, UnicodeDecodeError, RecursionError, OverflowError) as exc:
        raise ArchiveError() from exc


def archive_folder(root: Path, session_id: str) -> Path:
    if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
        raise ArchiveError("history_not_found")
    root = root.resolve(strict=True)
    folder = root / session_id
    if folder.is_symlink() or getattr(folder, "is_junction", lambda: False)():
        raise ArchiveError("history_invalid")
    if not folder.is_dir():
        raise ArchiveError("history_not_found")
    if folder.resolve(strict=True).parent != root:
        raise ArchiveError("history_invalid")
    return folder


def load_archive(root: Path, session_id: str) -> tuple[CaptionSnapshot, Path]:
    folder = archive_folder(root, session_id)
    path = folder / "session.json"
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True).parent != folder:
        raise ArchiveError("history_not_found")
    with path.open("rb") as stream:
        raw = stream.read(MAX_ARCHIVE_BYTES + 1)
    return decode_snapshot(raw, expected_session_id=session_id), folder


def visible_count(snapshot):
    return sum(not item.superseded_by for item in snapshot.segments)


def target_language(snapshot):
    languages = {item.target_language for item in snapshot.segments}
    return next(iter(languages)) if len(languages) == 1 else "mixed" if languages else "und"


def list_archives(root: Path) -> dict:
    candidates, skipped, scanned, truncated = [], 0, 0, False
    root = root.resolve(strict=True)
    with os.scandir(root) as entries:
        for entry in entries:
            scanned += 1
            if scanned > MAX_SCAN_ENTRIES:
                truncated = True
                break
            if not _SESSION_ID.fullmatch(entry.name):
                continue
            try:
                folder = archive_folder(root, entry.name)
                path = folder / "session.json"
                if path.is_symlink() or not path.is_file():
                    continue
                candidates.append((path.stat().st_mtime_ns, entry.name))
            except (ArchiveError, OSError):
                skipped += 1
    candidates.sort(reverse=True)
    if len(candidates) > MAX_LIST_CANDIDATES:
        truncated = True
    sessions = []
    for modified_ns, session_id in candidates[:MAX_LIST_CANDIDATES]:
        if len(sessions) == MAX_HISTORY_ENTRIES:
            truncated = True
            break
        try:
            snapshot, folder = load_archive(root, session_id)
            state = "saved"
            summary = folder / "desktop-summary.json"
            if summary.is_file() and not summary.is_symlink():
                with summary.open("rb") as stream:
                    summary_raw = stream.read(128 * 1024 + 1)
                if len(summary_raw) <= 128 * 1024:
                    try:
                        parsed = json.loads(summary_raw)
                        if isinstance(parsed, dict) and parsed.get("status") in ("completed", "failed"):
                            state = parsed["status"]
                    except (ValueError, UnicodeDecodeError, RecursionError):
                        pass
            sessions.append({"session_id": session_id,
                "updated_at_utc": datetime.fromtimestamp(modified_ns / 1e9, timezone.utc).isoformat().replace("+00:00", "Z"),
                "caption_count": visible_count(snapshot), "target_language": target_language(snapshot), "state": state})
        except (ArchiveError, OSError, ValueError, OverflowError):
            skipped += 1
    return {"sessions": sessions, "truncated": truncated, "skipped_invalid": skipped}


class ArchivedProjection:
    """Editable saved captions, deliberately lacking live/RPC event admission."""
    def __init__(self, snapshot):
        self.session_id = snapshot.session_id
        self.provider_generation = snapshot.provider_generation
        self.max_segments = MAX_ARCHIVE_SEGMENTS
        self.store = CaptionStore.from_snapshot(snapshot)

    def snapshot(self):
        return self.store.snapshot()
