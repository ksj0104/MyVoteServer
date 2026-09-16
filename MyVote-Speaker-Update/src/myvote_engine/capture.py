"""Bounded JSONL bridge to the owned Windows WASAPI capture helper.

The helper exposes native device positions and QPC timestamps. Python does not
replace them with pipe-arrival time. This module neither records to disk nor
opens an inference server; callers decide where the captured audio goes.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import ctypes
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import AsyncIterator

from .dsp import NativePcmChunk, NativePcmFormat


MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_PCM_BYTES = 1024 * 1024


class CaptureProtocolError(ValueError):
    pass


class WindowsQpcClock:
    """The same Windows QPC domain as the native helper, expressed in 100 ns.

    This is usable across processes on this computer, never across Windows/Mac.
    Use QueryPerformanceCounter directly so no Python clock-origin assumption is
    required. Frequency is stable for the current boot and cached once.
    """

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("WASAPI timing requires the Windows QPC clock")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._counter = kernel.QueryPerformanceCounter
        frequency = kernel.QueryPerformanceFrequency
        for function in (self._counter, frequency):
            function.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
            function.restype = ctypes.c_int
        value = ctypes.c_longlong()
        if not frequency(ctypes.byref(value)) or value.value <= 0:
            raise OSError("QueryPerformanceFrequency failed")
        self.frequency = value.value

    def now_100ns(self) -> int:
        value = ctypes.c_longlong()
        if not self._counter(ctypes.byref(value)):
            raise OSError("QueryPerformanceCounter failed")
        return value.value * 10_000_000 // self.frequency


def _integer(value, name: str, *, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise CaptureProtocolError(f"Invalid {name}")
    return value


def _string(value, name: str, *, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise CaptureProtocolError(f"Invalid {name}")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CaptureProtocolError("Duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class CaptureFormat:
    pcm: NativePcmFormat
    track_id: str
    capture_epoch: str
    qpc_origin_100ns: int
    endpoint_id: str


@dataclass(frozen=True)
class CaptureStopped:
    reason: str
    packets: int
    frames: int
    dropped_packets: int
    error: dict | None


CaptureRecord = CaptureFormat | NativePcmChunk | CaptureStopped


class CaptureDecoder:
    """Validate one header, bounded native packets, and exactly one terminal.

    Sequence gaps are retained for the normalizer to report. A sequence rewind,
    malformed packet, missing terminal, or data after the terminal is an error.
    """

    def __init__(self):
        self.header: CaptureFormat | None = None
        self.stopped: CaptureStopped | None = None
        self._last_sequence = -1
        self.received_packets = 0
        self.received_frames = 0

    def decode(self, line: bytes) -> CaptureRecord:
        if self.stopped is not None:
            raise CaptureProtocolError("Data after capture terminal")
        if not isinstance(line, bytes) or not line.endswith(b"\n") or len(line) > MAX_LINE_BYTES:
            raise CaptureProtocolError("Truncated or oversized capture line")
        try:
            item = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object,
                              parse_constant=lambda _: (_ for _ in ()).throw(CaptureProtocolError("Non-finite JSON")))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise CaptureProtocolError("Invalid capture JSON") from exc
        if not isinstance(item, dict):
            raise CaptureProtocolError("Capture record must be an object")
        try:
            kind = item["type"]
            if kind == "format":
                if self.header is not None or item["protocol"] != 1 or isinstance(item["protocol"], bool):
                    raise CaptureProtocolError("Duplicate header or unsupported protocol")
                pcm = NativePcmFormat(
                    sample_rate=item["sample_rate"], channels=item["channels"],
                    encoding=item["encoding"], valid_bits=item["valid_bits"],
                    channel_mask=item["channel_mask"], block_align=item["block_align"],
                )
                self.header = CaptureFormat(
                    pcm, _string(item["track_id"], "track_id", maximum=128),
                    _string(item["capture_epoch"], "capture_epoch", maximum=128),
                    _integer(item["qpc_origin_100ns"], "QPC origin"),
                    _string(item["endpoint_id"], "endpoint_id"),
                )
                return self.header
            if kind == "stopped":
                # A device-open failure may produce only a terminal diagnostic.
                error = item.get("error")
                if error is not None and not isinstance(error, dict):
                    raise CaptureProtocolError("Invalid capture error")
                stopped = CaptureStopped(
                    _string(item["reason"], "stop reason", maximum=128),
                    _integer(item["packets"], "packet count"),
                    _integer(item["frames"], "frame count"),
                    _integer(item["dropped_packets"], "drop count"), error,
                )
                if (stopped.packets < self.received_packets or stopped.frames < self.received_frames
                        or stopped.dropped_packets > stopped.packets
                        or stopped.packets - stopped.dropped_packets != self.received_packets
                        or self._last_sequence >= stopped.packets):
                    raise CaptureProtocolError("Capture terminal counts do not match received packets")
                lost_frames = stopped.frames - self.received_frames
                max_packet_frames = self.header.pcm.sample_rate if self.header else 192000
                if not stopped.dropped_packets <= lost_frames <= stopped.dropped_packets * max_packet_frames:
                    raise CaptureProtocolError("Capture terminal frame count does not match packet loss")
                if self.header is None and error is None:
                    raise CaptureProtocolError("Successful capture is missing its format")
                self.stopped = stopped
                return stopped
            if kind != "packet" or self.header is None:
                raise CaptureProtocolError("Unknown record or packet before format")
            sequence = _integer(item["sequence"], "sequence")
            if sequence <= self._last_sequence:
                raise CaptureProtocolError("Capture sequence rewound")
            frames = _integer(item["frames"], "packet frames", maximum=self.header.pcm.sample_rate)
            if frames == 0:
                raise CaptureProtocolError("Empty capture packet")
            expected = frames * self.header.pcm.block_align
            encoded = item["data_b64"]
            if (expected > MAX_PCM_BYTES or not isinstance(encoded, str)
                    or len(encoded) != 4 * ((expected + 2) // 3)):
                raise CaptureProtocolError("PCM payload size mismatch")
            try:
                payload = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise CaptureProtocolError("Invalid base64 PCM") from exc
            if len(payload) != expected:
                raise CaptureProtocolError("PCM payload size mismatch")
            chunk = NativePcmChunk(
                sequence=sequence,
                device_position=_integer(item["device_position"], "device position"),
                qpc_position_100ns=_integer(item["qpc_position_100ns"], "QPC position"),
                frames=frames, flags=_integer(item["flags"], "capture flags", maximum=7),
                data=payload,
                copied_qpc_100ns=(_integer(item["copied_qpc_100ns"], "copied QPC position")
                                  if item.get("copied_qpc_100ns") is not None else None),
            )
            if chunk.flags & 2 and any(payload):
                raise CaptureProtocolError("Silent packet contains nonzero PCM")
            self._last_sequence = sequence
            self.received_packets += 1
            self.received_frames += frames
            return chunk
        except KeyError as exc:
            raise CaptureProtocolError(f"Capture record missing field {exc.args[0]}") from exc

    def finish(self) -> None:
        if self.stopped is None:
            raise CaptureProtocolError("Capture helper ended without a terminal record")


class NativeCaptureProcess:
    """Launch and clean up only this call's owned helper process.

    Use as an async context manager and consume ``records()`` once. Cancellation
    closes the owned helper; it never kills shared audio or inference services.
    stderr is drained with bounded retention, and is not echoed into a transcript.
    """

    def __init__(self, helper_path: str | Path, *, duration_s: float = 10,
                 device_id: str | None = None):
        if not math.isfinite(duration_s) or not .1 <= duration_s <= 3600:
            raise ValueError("Capture duration must be between 0.1 and 3600 seconds")
        path = Path(helper_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in (".exe", ".dll"):
            raise FileNotFoundError("Provide the built MyVote.Capture .exe or .dll")
        if device_id is not None:
            _string(device_id, "device_id")
        self.command = (["dotnet", str(path)] if path.suffix.lower() == ".dll" else [str(path)])
        self.command.extend(["--seconds", str(duration_s), "--control-stdin"])
        if device_id:
            self.command.extend(["--device-id", device_id])
        self.duration_s = duration_s
        self.process: asyncio.subprocess.Process | None = None
        self.decoder = CaptureDecoder()
        self.stderr_tail = bytearray()
        self._stderr_task: asyncio.Task | None = None
        self._consumed = False
        self._stop_requested = False

    async def __aenter__(self):
        if sys.platform != "win32":
            raise RuntimeError("Native WASAPI capture requires Windows")
        self.process = await asyncio.create_subprocess_exec(
            *self.command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=MAX_LINE_BYTES, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        return self

    async def request_stop(self) -> None:
        """Ask the owned helper to finish; keep reading through its terminal.

        Sending this command does not discard already copied packets or replace
        normal terminal validation. The consumer bounds its stop/drain wait.
        Repeated requests and an already-exited helper are harmless.
        """
        if self.process is None:
            raise RuntimeError("Stop requires an active capture context")
        if self._stop_requested or self.process.returncode is not None:
            return
        self._stop_requested = True
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(b"stop\n")
            async with asyncio.timeout(1):
                await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # A concurrent natural completion may close stdin first. records()
            # still requires a valid terminal and exit status before success.
            pass

    async def _drain_stderr(self):
        assert self.process and self.process.stderr
        while block := await self.process.stderr.read(4096):
            self.stderr_tail.extend(block)
            if len(self.stderr_tail) > 8192:
                del self.stderr_tail[:-8192]

    async def records(self) -> AsyncIterator[CaptureRecord]:
        if self.process is None or self._consumed:
            raise RuntimeError("Use one capture record consumer inside the async context")
        self._consumed = True
        assert self.process.stdout is not None
        # An endpoint may emit no packets during silence, so the timeout covers
        # the requested capture duration plus bounded startup/drain allowance.
        async with asyncio.timeout(self.duration_s + 15):
            while True:
                try:
                    line = await self.process.stdout.readline()
                except (ValueError, asyncio.LimitOverrunError) as exc:
                    raise CaptureProtocolError("Capture line exceeds transport limit") from exc
                if not line:
                    break
                yield self.decoder.decode(line)
            self.decoder.finish()
            code = await self.process.wait()
            if code != 0:
                raise RuntimeError(f"Capture helper failed with exit code {code}")

    async def __aexit__(self, exc_type, exc, traceback):
        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.close()
        if self.process is not None and self.process.returncode is None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
        # A paused StreamReader can prevent process.wait() from completing even
        # after kill has set returncode. Discard unread stdout before waiting.
        if self.process is not None and self.process.stdout is not None:
            while await self.process.stdout.read(65536):
                pass
        if self._stderr_task is not None:
            await self._stderr_task
        if self.process is not None:
            await self.process.wait()
