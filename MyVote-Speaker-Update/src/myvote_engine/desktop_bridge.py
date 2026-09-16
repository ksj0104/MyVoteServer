"""Version 1 JSONL desktop control, reusing the actual live client/reducer.

stdin/stdout are owned pipes. No model, capture or RPC is reimplemented here.
Native Windows pipe polling needs no indefinitely blocked stdin thread. stdout
uses one bounded daemon writer; a slow/disconnected UI is an explicit failure,
never permission to discard final captions or manual edits silently.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import ctypes
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

from .captions import export_json, export_srt, export_webvtt
from .caption_archive import (ArchiveError, ArchivedProjection, list_archives, load_archive,
                              target_language, visible_count)
from .remote_captions import RemoteCaptionProjection


MAX_LINE_BYTES = 1024 * 1024
MAX_COMMANDS = 10000


class DesktopError(ValueError):
    def __init__(self, category, command_id=""):
        super().__init__(category)
        self.category = category
        self.command_id = command_id


def _text(value, maximum, *, empty=False):
    if (not isinstance(value, str) or len(value) > maximum
            or (not empty and not value.strip()) or any(ord(c) < 32 for c in value)):
        raise DesktopError("invalid_data")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DesktopError("invalid_data") from exc
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DesktopError("invalid_command")
        result[key] = value
    return result


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise DesktopError("invalid_command")
    return result


def decode_command(line):
    if not isinstance(line, bytes) or not line.endswith(b"\n") or len(line) > MAX_LINE_BYTES:
        raise DesktopError("invalid_command")
    try:
        item = json.loads(line.decode("utf-8"), object_pairs_hook=_unique, parse_float=_float,
                          parse_constant=lambda _: (_ for _ in ()).throw(DesktopError("invalid_command")))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise DesktopError("invalid_command") from exc
    if not isinstance(item, dict):
        raise DesktopError("invalid_command")
    try:
        command_id = _text(item.get("id"), 128)
    except DesktopError:
        command_id = ""
    if not command_id or set(item) != {"version", "id", "kind", "data"}:
        raise DesktopError("invalid_command", command_id)
    if type(item["version"]) is not int or item["version"] != 1:
        raise DesktopError("invalid_version", command_id)
    if item["kind"] not in ("start", "stop", "create_speaker", "rename_speaker", "assign_speaker",
                            "apply_caption_group", "export", "history_list", "history_open", "history_page"):
        raise DesktopError("invalid_command", command_id)
    if not isinstance(item["data"], dict):
        raise DesktopError("invalid_data", command_id)
    return item


class PipeLines:
    """Bounded polling of one redirected stdin pipe, including Windows EOF."""
    def __init__(self, fd):
        self.fd = fd
        self._peek = None
        if sys.platform == "win32":
            import msvcrt
            self._handle = msvcrt.get_osfhandle(fd)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self._peek = kernel.PeekNamedPipe
            self._peek.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                   ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
            self._peek.restype = ctypes.c_int
        else:
            os.set_blocking(fd, False)

    def _read(self):
        if self._peek is not None:
            available = ctypes.c_uint32()
            if not self._peek(self._handle, None, 0, None, ctypes.byref(available), None):
                if ctypes.get_last_error() in (109, 232, 233):
                    return b""
                raise DesktopError("stdin_pipe_failed")
            if not available.value:
                return None
            return os.read(self.fd, min(65536, available.value))
        try:
            return os.read(self.fd, 65536)
        except BlockingIOError:
            return None

    async def __aiter__(self):
        buffer = bytearray()
        while True:
            block = self._read()
            if block is None:
                await asyncio.sleep(.01)
                continue
            if not block:
                if buffer:
                    raise DesktopError("truncated_input")
                return
            buffer.extend(block)
            while (end := buffer.find(b"\n")) >= 0:
                if end + 1 > MAX_LINE_BYTES:
                    raise DesktopError("input_line_limit")
                line = bytes(buffer[:end + 1])
                del buffer[:end + 1]
                yield line
            if len(buffer) > MAX_LINE_BYTES:
                raise DesktopError("input_line_limit")


class PipeWriter:
    """One raw os.write worker; only pending previews may be superseded.

    Sequence/QPC are assigned at dequeue, so coalescing creates no wire sequence
    gaps. A blocked native write lives in a daemon thread, outside asyncio's
    default executor and interpreter stdout locks. All buffers have fixed caps.
    """
    def __init__(self, fd, *, max_lines=256, max_bytes=4 * MAX_LINE_BYTES, clock=None):
        self.fd, self.max_lines, self.max_bytes = fd, max_lines, max_bytes
        self.clock = clock
        self._queue = deque()
        self._bytes = 0
        self._condition = threading.Condition()
        self._closing = self._writing = False
        self._error = None
        self._sequence = 0
        self._worker = threading.Thread(target=self._run, name="desktop-stdout", daemon=True)
        self._worker.start()

    @property
    def queued_bytes(self):
        with self._condition:
            return self._bytes

    def emit(self, kind, session_id, data, *, preview=False, key=None):
        message = {"schema": "myvote.desktop", "version": 1, "seq": 2**64 - 1,
                   "kind": kind, "session_id": session_id, "data": data, "emitted_qpc_100ns": None}
        try:
            raw = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
            raise DesktopError("invalid_output") from exc
        size = len(raw) + 32  # newline and a worst-case QPC replacing null
        if size > MAX_LINE_BYTES:
            raise DesktopError("output_line_limit")
        # JSON round-trip detaches mutable callback data from the writer thread.
        message = json.loads(raw)
        with self._condition:
            if self._error or self._closing:
                raise DesktopError(self._error or "output_closed")
            if key is not None:
                for item in tuple(self._queue):
                    if item[1] == key and item[2]:
                        self._queue.remove(item)
                        self._bytes -= item[3]
            if len(self._queue) >= self.max_lines or self._bytes + size > self.max_bytes:
                raise DesktopError("output_backpressure")
            self._queue.append((message, key, preview, size))
            self._bytes += size
            self._condition.notify()

    async def send(self, kind, session_id, data, *, timeout_s=5, **kwargs):
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.emit(kind, session_id, data, **kwargs)
                return
            except DesktopError as exc:
                if exc.category != "output_backpressure" or time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(.01)

    def _run(self):
        while True:
            with self._condition:
                while not self._queue and not self._closing:
                    self._condition.wait()
                if not self._queue:
                    return
                message, _, _, size = self._queue.popleft()
                self._bytes -= size
                self._writing = True
            try:
                self._sequence += 1
                message["seq"] = self._sequence
                message["emitted_qpc_100ns"] = self.clock() if self.clock is not None else None
                raw = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
                if len(raw) > MAX_LINE_BYTES:
                    raise DesktopError("output_line_limit")
                view = memoryview(raw)
                while view:
                    count = os.write(self.fd, view)
                    if count <= 0:
                        raise OSError("closed output")
                    view = view[count:]
            except Exception:
                with self._condition:
                    self._error = "output_pipe_failed"
                    self._writing = False
                    self._condition.notify_all()
                return
            with self._condition:
                self._writing = False
                self._condition.notify_all()

    async def aclose(self, timeout_s=5):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        deadline = time.monotonic() + timeout_s
        while self._worker.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(.01)
        if self._worker.is_alive() or self._error:
            raise DesktopError(self._error or "output_backpressure")


async def _actual_live(args, **kwargs):
    from .live import run_live
    return await run_live(args, **kwargs)


def _save_snapshot(snapshot, folder, mode):
    folder.mkdir(parents=True, exist_ok=True)
    for name, contents in (("captions.srt", export_srt(snapshot, mode=mode)),
                           ("captions.vtt", export_webvtt(snapshot, mode=mode)),
                           ("session.json", export_json(snapshot))):
        pending = folder / (name + ".tmp")
        pending.write_text(contents, encoding="utf-8")
        pending.replace(folder / name)


class DesktopBridge:
    """One current session; finished projection remains editable/exportable.

    A supplied live_runner is test injection and is labeled in finished output;
    the production CLI exposes no fixture source or fake engine switch.
    """
    def __init__(self, helper, output_root, writer, *, live_runner=None):
        self.helper, self.output_root = Path(helper), Path(output_root)
        if not self.helper.is_absolute() or not self.helper.is_file() or self.helper.suffix.lower() not in (".exe", ".dll"):
            raise DesktopError("invalid_helper")
        if not self.output_root.is_absolute():
            raise DesktopError("invalid_output_root")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.runner = live_runner or _actual_live
        self.injected = live_runner is not None
        self.session_id = ""
        self.output_dir = None
        self.projection = None
        self._task = None
        self._stop = None
        self._seen = set()
        self._known_speakers = set()
        self._segment_ids = set()
        self._statuses = {}
        self._state = "idle"
        self._pending_timing = None
        self._history_token = None
        self._history_items = ()

    @property
    def active(self):
        return self._task is not None and not self._task.done()

    def _emit(self, kind, data, *, preview=False, key=None):
        self.writer.emit(kind, self.session_id, data, preview=preview, key=key)

    def _caption(self, segment_id):
        segment = self.projection.store.get_segment(segment_id)
        if segment is None:
            return None
        result = asdict(segment)
        aliases = dict(self.projection.store.speaker_aliases())
        result["speaker_name"] = aliases.get(segment.speaker_id, segment.speaker_id)
        status = self._statuses.get(segment_id)
        if status is not None and status[1] == segment.source_revision and status[2] == self.projection.provider_generation:
            state = status[0]
        else:
            state = None
        translation = segment.translation
        if translation is not None:
            valid = (translation.source_revision == segment.source_revision
                     and translation.target_language == segment.target_language
                     and (translation.completed or (translation.provider_generation == self.projection.provider_generation
                                                    and state not in ("failed", "cancelled"))))
            if not valid:
                result["translation"] = None
            elif translation.completed:
                state = "completed"
        result["translation_status"] = state
        return result

    def _speakers(self):
        aliases = dict(self.projection.store.speaker_aliases())
        return {"items": [{"speaker_id": key, "name": aliases.get(key, key)}
                          for key in sorted(self._known_speakers)]}

    def _caption_group(self, group_id):
        group = self.projection.store.get_group(group_id)
        if group is None:
            raise DesktopError("unknown_caption_group")
        result = asdict(group)
        result["status"] = self.projection.store.group_status(group_id)
        result["children"] = [self._caption(child.segment_id) for child in group.children]
        speakers = {item["speaker_id"] for item in result["children"] if item["speaker_id"] is not None}
        if len(self._known_speakers | speakers) > 128:
            raise DesktopError("speaker_limit")
        self._known_speakers.update(speakers)
        self._segment_ids.update(child.segment_id for child in group.children)
        return result

    def _observe(self, kind, data, segment_id=None):
        if kind == "transcript.preview":
            self._pending_timing = None
            # run_live invokes observers only after the projection accepted the
            # event. Forward its detached validated state, never the callback's
            # mutable data and never a stored caption/export entry.
            if not isinstance(self.projection, RemoteCaptionProjection):
                return
            preview = self.projection.transcript_preview
            if preview is not None and preview == data:
                self._emit("transcript_preview", preview, preview=preview["state"] != "cleared",
                           key=("transcript_preview", self.session_id))
            return
        if kind == "caption.results_revised":
            self._pending_timing = None
            for correction in data["corrections"]:
                caption_id = correction["segment_id"]
                caption = self._caption(caption_id)
                if caption is not None:
                    self._emit("caption", caption, key=caption_id)
            return
        if kind == "client.latency":
            # This callback and the next caption callback run without an await.
            # Retain one timing only; later manual/speaker changes must not be
            # counted as a new source/translation display latency.
            if data.get("clock_kind") == "windows_qpc_ns":
                self._pending_timing = (segment_id, data["source_revision"], {
                    "client_received_qpc_100ns": data["client_received_clock_ns"] // 100,
                    "source_end_qpc_100ns": (data["timeline_origin_clock_ns"] + data["end_ns"]) // 100,
                    "metric_kind": data["kind"], "diagnostic_only": True})
            return
        timing, self._pending_timing = self._pending_timing, None
        if kind == "session.started":
            if self._state != "stopping":
                self._state = "running"
            self._emit("session", {"state": self._state, "server_engine_mode": data.get("engine_mode"),
                                   "speaker_status": data.get("speaker_status")})
        if kind in ("translation.preview", "translation.completed", "translation.failed", "translation.cancelled") and segment_id is not None:
            self._statuses[segment_id] = (kind.split(".", 1)[1], data.get("source_revision"),
                                           data.get("provider_generation"))
        if kind == "caption.group_replaced":
            self._emit("caption_group", self._caption_group(data["group_id"]))
            return
        if segment_id and kind in ("translation.preview", "translation.completed", "translation.failed",
                                   "translation.cancelled", "speaker.updated"):
            held_group = self.projection.store.held_group_for_segment(segment_id)
            if held_group is not None:
                self._emit("caption_group", self._caption_group(held_group))
                return
        if kind in ("caption.source", "translation.preview", "translation.completed", "translation.failed",
                    "translation.cancelled", "speaker.updated", "segment.superseded") and segment_id:
            caption = self._caption(segment_id)
            if caption is not None:
                if timing is not None and timing[:2] == (segment_id, caption["source_revision"]):
                    caption["timing"] = timing[2]
                self._segment_ids.add(segment_id)
                if caption["speaker_id"] is not None:
                    if caption["speaker_id"] not in self._known_speakers and len(self._known_speakers) >= 128:
                        raise DesktopError("speaker_limit")
                    self._known_speakers.add(caption["speaker_id"])
                self._emit("caption", caption, preview=kind == "translation.preview", key=segment_id)
        if kind == "speaker.alias_updated":
            if data["speaker_id"] not in self._known_speakers and len(self._known_speakers) >= 128:
                raise DesktopError("speaker_limit")
            self._known_speakers.add(data["speaker_id"])
            self._emit("speakers", self._speakers())
        if kind == "provider.updated":
            for group in self.projection.snapshot().held_groups:
                self._emit("caption_group", self._caption_group(group.group_id))
            for key in self._segment_ids:
                item = self.projection.store.get_segment(key)
                if (item is not None and self.projection.store.held_group_for_segment(key) is None
                        and item.translation is not None and not item.translation.completed):
                    self._emit("caption", self._caption(key), key=key)
        notices = {"source.failed": ("error", "source_failed"),
                   "session.failed": ("error", "connection_failed"),
                   "speaker.failed": ("warning", "speaker_degraded"),
                   "audio.gap": ("warning", "audio_gap")}
        if kind in notices:
            level, code = notices[kind]
            self._emit("notice", {"level": level, "message_code": code})

    def _on_projection(self, projection):
        if not isinstance(projection, RemoteCaptionProjection) or projection.session_id != self.session_id:
            raise DesktopError("projection_mismatch")
        self.projection = projection
        self._known_speakers.update(key for key, _ in projection.store.speaker_aliases())

    async def _run(self, args):
        try:
            summary = await self.runner(args, observer=self._observe, stop_event=self._stop,
                                        on_projection=self._on_projection)
        except BaseException as exc:
            archive_error = None
            if self.projection is not None:
                try:
                    await asyncio.to_thread(_save_snapshot, self.projection.snapshot(), self.output_dir, "bilingual")
                except Exception as archive_exc:
                    archive_error = type(archive_exc).__name__
            summary = {"session_id": self.session_id, "status": "failed",
                       "source_mode": "injected_test_source" if self.injected else "native_wasapi_loopback",
                       "failure": {"type": type(exc).__name__, "message": "session_setup_failed"}}
            if archive_error:
                summary["archive_error"] = archive_error
            await self.writer.send("notice", self.session_id, {"level": "error", "message_code": "session_setup_failed"})
        self._state = "finished"
        summary = {**summary, "output_dir": str(self.output_dir),
                   "bridge_mode": "injected_test_runner" if self.injected else "actual_live_client"}
        try:
            (self.output_dir / "desktop-summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            summary["status"] = "failed"
            summary["archive_error"] = type(exc).__name__
            await self.writer.send("notice", self.session_id, {"level": "error", "message_code": "archive_failed"})
        await self.writer.send("finished", self.session_id, summary)

    @staticmethod
    def _fields(data, required, optional=()):
        if not set(required) <= set(data) or set(data) - set(required) - set(optional):
            raise DesktopError("invalid_data")

    def _start(self, data):
        if self.active:
            raise DesktopError("session_busy")
        self._fields(data, ("target", "ca_cert", "client_cert", "client_key", "target_language"), ("source_language",))
        from .gateway_client import validate_target
        try:
            target = validate_target(_text(data["target"], 512))
        except ValueError as exc:
            raise DesktopError("invalid_data") from exc
        paths = {}
        for name in ("ca_cert", "client_cert", "client_key"):
            path = Path(_text(data[name], 4096))
            if not path.is_absolute() or not path.is_file():
                raise DesktopError("invalid_path")
            paths[name] = path
        source_language = data.get("source_language")
        if source_language is None:
            source_language = ""
        _text(source_language, 64, empty=True)
        target_language = _text(data["target_language"], 64)
        next_session_id = uuid.uuid4().hex
        next_output_dir = self.output_root / next_session_id
        next_output_dir.mkdir(exist_ok=False)
        args = argparse.Namespace(input=None, helper=self.helper, seconds=3600, device_id=None,
            target=target, **paths, session_id=next_session_id, source_language=source_language,
            target_language=target_language, deadline_ms=2500, queue_frames=32,
            connect_timeout_s=10, drain_timeout_s=60, output_dir=next_output_dir, show_text=False)
        # A failed output reservation must not destroy the editable prior session.
        self.writer.emit("session", next_session_id, {"state": "connecting"})
        self.session_id = next_session_id
        self.output_dir = next_output_dir
        self.projection = None
        self._known_speakers.clear()
        self._segment_ids.clear()
        self._statuses.clear()
        self._pending_timing = None
        self._history_token, self._history_items = None, ()
        self._stop = asyncio.Event()
        self._state = "connecting"
        self._task = asyncio.create_task(self._run(args))

    def _require_projection(self):
        if self.projection is None:
            raise DesktopError("no_session")

    async def _edited(self, kind, data):
        self._history_token, self._history_items = None, ()
        # Keep a separate local journal; the wire journal remains server events.
        with (self.output_dir / "desktop-actions.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps({"kind": kind, "data": data}, ensure_ascii=False, allow_nan=False) + "\n")
        if not self.active:
            await asyncio.to_thread(_save_snapshot, self.projection.snapshot(), self.output_dir, "bilingual")

    async def _open_history(self, data):
        if self.active:
            raise DesktopError("session_busy")
        self._fields(data, ("session_id",))
        snapshot, folder = await asyncio.to_thread(load_archive, self.output_root, data["session_id"])
        projection = ArchivedProjection(snapshot)
        # The archive keeps its original session ID on disk. Every desktop view
        # gets a fresh wire identity so retired live messages cannot affect it.
        view_id, token = uuid.uuid4().hex, uuid.uuid4().hex
        items = tuple(("caption", item.segment_id) for item in snapshot.segments)
        items += tuple(("caption_group", group.group_id) for group in snapshot.held_groups)
        speakers = {key for key, _ in snapshot.speaker_aliases}
        speakers.update(item.speaker_id for item in snapshot.segments if item.speaker_id is not None)
        speakers.update(child.speaker_id for group in snapshot.held_groups for child in group.children
                        if child.speaker_id is not None)
        self.projection, self.output_dir, self.session_id = projection, folder, view_id
        self._known_speakers = speakers
        self._segment_ids = {item.segment_id for item in snapshot.segments}
        self._statuses.clear()
        self._pending_timing = None
        self._state = "history"
        self._history_token, self._history_items = token, items
        return {"session_id": view_id, "archive_session_id": snapshot.session_id, "output_dir": str(folder),
                "snapshot_token": token, "caption_count": visible_count(snapshot), "item_count": len(items),
                "target_language": target_language(snapshot)}

    def _history_page(self, data):
        self._fields(data, ("snapshot_token", "offset", "limit"))
        if (self.active or self._history_token is None or data["snapshot_token"] != self._history_token):
            raise DesktopError("history_stale")
        offset, limit = data["offset"], data["limit"]
        if (type(offset) is not int or not 0 <= offset <= len(self._history_items)
                or type(limit) is not int or not 1 <= limit <= 100):
            raise DesktopError("invalid_data")
        payload = {"session_id": self.session_id, "snapshot_token": self._history_token, "offset": offset,
                   "items": [], "speakers": self._speakers(), "next_offset": None}
        # A bounded pull response cannot overflow the desktop writer or the UI's
        # bounded message channel. Large captions/groups reduce page row count.
        size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")) + 4096
        for kind, identifier in self._history_items[offset:offset + limit]:
            item = {"kind": kind, "data": self._caption(identifier) if kind == "caption" else self._caption_group(identifier)}
            item_size = len(json.dumps(item, ensure_ascii=False, allow_nan=False).encode("utf-8")) + 2
            if size + item_size > 512 * 1024:
                if not payload["items"]:
                    raise DesktopError("history_limit")
                break
            payload["items"].append(item)
            size += item_size
        following = offset + len(payload["items"])
        payload["next_offset"] = following if following < len(self._history_items) else None
        return payload

    async def command(self, item):
        command_id, kind, data = item["id"], item["kind"], item["data"]
        result = {"id": command_id, "ok": False}
        try:
            if command_id in self._seen:
                raise DesktopError("duplicate_command")
            if len(self._seen) >= MAX_COMMANDS:
                raise DesktopError("command_limit")
            self._seen.add(command_id)
            if kind == "start":
                self._start(data)
            elif kind == "history_list":
                self._fields(data, ())
                result.update(await asyncio.to_thread(list_archives, self.output_root))
            elif kind == "history_open":
                result.update(await self._open_history(data))
            elif kind == "history_page":
                result.update(self._history_page(data))
            elif kind == "stop":
                self._fields(data, ())
                if not self.active:
                    raise DesktopError("no_session")
                self._state = "stopping"
                self._stop.set()
                self._emit("session", {"state": "stopping"})
            elif kind == "create_speaker":
                self._require_projection()
                self._fields(data, ("name",))
                name = _text(data["name"], 256)
                known = self._known_speakers | {key for key, _ in self.projection.store.speaker_aliases()}
                if len(known) >= 128:
                    raise DesktopError("speaker_limit")
                speaker_id = "manual-" + uuid.uuid4().hex
                self.projection.store.set_speaker_alias(speaker_id, name)
                self._known_speakers.add(speaker_id)
                await self.writer.send("speakers", self.session_id, self._speakers())
                await self._edited(kind, {"speaker_id": speaker_id, "name": name})
                result["speaker_id"] = speaker_id
            elif kind == "rename_speaker":
                self._require_projection()
                self._fields(data, ("speaker_id", "name"))
                speaker_id, name = _text(data["speaker_id"], 256), _text(data["name"], 256)
                if speaker_id not in self._known_speakers:
                    raise DesktopError("unknown_speaker")
                self.projection.store.set_speaker_alias(speaker_id, name)
                await self.writer.send("speakers", self.session_id, self._speakers())
                await self._edited(kind, data)
            elif kind == "assign_speaker":
                self._require_projection()
                self._fields(data, ("segment_id", "speaker_id"))
                segment_id = _text(data["segment_id"], 512)
                speaker_id = data["speaker_id"]
                if speaker_id is not None and _text(speaker_id, 256) not in self._known_speakers:
                    raise DesktopError("unknown_speaker")
                segment = self.projection.store.get_segment(segment_id)
                if (segment is None or segment.superseded_by
                        or self.projection.store.held_group_for_segment(segment_id) is not None):
                    raise DesktopError("unknown_segment")
                self.projection.store.update_speaker(segment_id, speaker_id, segment.speaker_revision + 1,
                    based_on_source_revision=segment.source_revision, manual=True)
                await self.writer.send("caption", self.session_id, self._caption(segment_id), key=segment_id)
                await self._edited(kind, data)
            elif kind == "apply_caption_group":
                self._require_projection()
                self._fields(data, ("group_id", "group_revision"))
                group_id = _text(data["group_id"], 512)
                revision = data["group_revision"]
                if type(revision) is not int or not 1 <= revision <= 2**64 - 1:
                    raise DesktopError("invalid_data")
                if not self.projection.store.apply_held_group(
                        group_id, revision, max_segments=self.projection.max_segments):
                    raise DesktopError("stale_caption_group")
                await self.writer.send("caption_group", self.session_id, self._caption_group(group_id))
                await self._edited(kind, data)
            elif kind == "export":
                self._require_projection()
                self._fields(data, ("mode",))
                mode = data["mode"]
                if mode not in ("source", "translation", "bilingual"):
                    raise DesktopError("invalid_data")
                folder = self.output_dir / "exports" / uuid.uuid4().hex
                await asyncio.to_thread(_save_snapshot, self.projection.snapshot(), folder, mode)
                result["output_dir"] = str(folder)
            result["ok"] = True
        except (DesktopError, ArchiveError) as exc:
            result["error"] = exc.category
        except OSError:
            result["error"] = "export_failed" if kind == "export" else "archive_failed"
        except Exception:
            result["error"] = "export_failed" if kind == "export" else "command_failed"
        await self.writer.send("command_result", self.session_id, result)

    async def shutdown(self):
        if self.active:
            self._stop.set()
            self._state = "stopping"
            try:
                self._emit("session", {"state": "stopping"})
            except DesktopError:
                pass  # A lost UI pipe cannot prevent owned capture/RPC cleanup.
            try:
                await asyncio.wait_for(asyncio.shield(self._task), 75)
            except TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
        elif self._task is not None:
            await self._task

    async def serve(self, lines):
        await self.writer.send("ready", "", {})
        try:
            async for line in lines:
                try:
                    item = decode_command(line)
                except DesktopError as exc:
                    await self.writer.send("command_result", self.session_id,
                                           {"id": exc.command_id, "ok": False, "error": exc.category})
                    continue
                await self.command(item)
        finally:
            await self.shutdown()


async def _main(args):
    clock = None
    if sys.platform == "win32":
        from .capture import WindowsQpcClock
        clock = WindowsQpcClock().now_100ns
    writer = PipeWriter(sys.stdout.fileno(), clock=clock)
    try:
        bridge = DesktopBridge(args.helper, args.output_root, writer)
        await bridge.serve(PipeLines(sys.stdin.fileno()))
    finally:
        await writer.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(_main(args))
    except Exception as exc:
        # No exception text, endpoint, paths, audio or transcript on stderr.
        code = exc.category if isinstance(exc, DesktopError) else type(exc).__name__
        print(f"desktop_bridge_failed:{code}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
