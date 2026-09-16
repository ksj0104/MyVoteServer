"""Async mTLS gRPC client for normalized audio and independent result events."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
from pathlib import Path
import re
import struct

import grpc

from .audio import AudioFrame
from .rpc import session_pb2 as pb, session_pb2_grpc as rpc
from .tls import load_tls_material


MAX_MESSAGE_BYTES = 256 * 1024


class GatewayProtocolError(ValueError):
    pass


def validate_target(target: str) -> str:
    if not isinstance(target, str) or len(target) > 512:
        raise ValueError("Expected TLS hostname:port or [IPv6]:port")
    host, separator, port = target.rpartition(":")
    if not separator or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError("TLS target requires an explicit valid port")
    if host.startswith("[") and host.endswith("]"):
        ipaddress.IPv6Address(host[1:-1])
    elif not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", host):
        raise ValueError("TLS target requires a hostname or IP address")
    return target


def frame_message(frame: AudioFrame) -> pb.ClientMessage:
    if not 1 <= len(frame.samples) <= 512:
        raise ValueError("Gateway frames require 1-512 mono samples at 16 kHz")
    return pb.ClientMessage(audio=pb.PcmFrame(
        track_id=frame.track_id, capture_epoch=frame.capture_epoch,
        sequence=frame.sequence, start_time_ns=frame.start_time_ns,
        pcm_f32le=struct.pack("<" + "f" * len(frame.samples), *frame.samples),
    ))


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GatewayProtocolError("Duplicate event data key")
        result[key] = value
    return result


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise GatewayProtocolError("Nonfinite event data")
    return number


class GatewayClient:
    """One authenticated stream; one sender and one receiver at a time.

    No insecure channel, hostname override, automatic reconnect, or cloud fallback.
    A new connection is a new session; the UI retains previous session records.
    """

    def __init__(self, target: str, *, ca_cert: Path, client_cert: Path,
                 client_key: Path, session_id: str, connect_timeout_s: float = 10,
                 rpc_timeout_s: float = 120):
        self.target = validate_target(target)
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("session_id must be 1-128 ASCII letters, digits, '_' or '-'")
        if any(not math.isfinite(value) or value <= 0 for value in (connect_timeout_s, rpc_timeout_s)):
            raise ValueError("Connection and RPC timeouts must be finite and positive")
        self.session_id = session_id
        self.material = load_tls_material(client_cert, client_key, ca_cert, role="client")
        self.connect_timeout_s = connect_timeout_s
        self.rpc_timeout_s = rpc_timeout_s
        self.channel = None
        self.call = None
        self.last_sequence = 0
        self._terminal = False
        self._finished_sending = False
        self._receiving = False
        self.heartbeat_interval_s: float | None = None

    async def __aenter__(self):
        credentials = grpc.ssl_channel_credentials(
            root_certificates=self.material.root_certificates,
            private_key=self.material.private_key,
            certificate_chain=self.material.certificate_chain,
        )
        self.channel = grpc.aio.secure_channel(self.target, credentials, options=(
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ("grpc.enable_retries", 0),
        ))
        try:
            async with asyncio.timeout(self.connect_timeout_s):
                await self.channel.channel_ready()
        except BaseException:
            await self.channel.close()
            self.channel = None
            raise
        return self

    def _validate_event(self, event: pb.ServerMessage) -> dict:
        if (event.session_id != self.session_id or event.event_sequence <= self.last_sequence
                or not event.kind or len(event.kind) > 128 or len(event.segment_id) > 512
                or not math.isfinite(event.server_elapsed_ms) or event.server_elapsed_ms < 0):
            raise GatewayProtocolError("Invalid server session/event ordering or metadata")
        if event.event_sequence != self.last_sequence + 1:
            raise GatewayProtocolError("Server event sequence contains a gap")
        if len(event.data_json.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise GatewayProtocolError("Server event data exceeds limit")
        try:
            data = json.loads(event.data_json, object_pairs_hook=_json_object, parse_float=_finite_json_float,
                              parse_constant=lambda _: (_ for _ in ()).throw(GatewayProtocolError("Nonfinite event data")))
        except (json.JSONDecodeError, RecursionError) as exc:
            raise GatewayProtocolError("Invalid server event JSON") from exc
        if not isinstance(data, dict) or self._terminal:
            raise GatewayProtocolError("Event after terminal or non-object data")
        self.last_sequence = event.event_sequence
        if event.kind in ("session.completed", "session.failed"):
            self._terminal = True
        return data

    async def open(self, *, source_language: str = "", target_language: str = "ko",
                   translation_budget_ms: float = 2500) -> pb.ServerMessage:
        if self.channel is None or self.call is not None:
            raise RuntimeError("Open one stream inside the client context")
        self.call = rpc.SpeechGatewayStub(self.channel).StreamSession(timeout=self.rpc_timeout_s)
        await self.call.write(pb.ClientMessage(open=pb.OpenSession(
            protocol_version=1, session_id=self.session_id, source_language=source_language,
            target_language=target_language, translation_budget_ms=translation_budget_ms,
        )))
        async with asyncio.timeout(self.connect_timeout_s):
            first = await self.call.read()
        if first is grpc.aio.EOF:
            raise GatewayProtocolError("Server ended before session start")
        data = self._validate_event(first)
        if first.kind != "session.started":
            raise GatewayProtocolError("Server did not acknowledge session start")
        interval_ms = data.get("heartbeat_interval_ms")
        if interval_ms is not None:
            if type(interval_ms) is not int or not 1 <= interval_ms <= 60000:
                raise GatewayProtocolError("Invalid advertised heartbeat interval")
            self.heartbeat_interval_s = interval_ms / 1000
        return first

    async def send_heartbeat(self) -> None:
        """Keep an idle application stream alive; never represent this as audio.

        Only the existing single sender may call this. Older servers do not
        advertise the capability and never receive an unknown oneof field.
        """
        if self.call is None or self._finished_sending or self.heartbeat_interval_s is None:
            raise RuntimeError("Heartbeat requires an open stream and server capability")
        await self.call.write(pb.ClientMessage(heartbeat=pb.Heartbeat()))

    async def send_frame(self, frame: AudioFrame) -> None:
        if self.call is None or self._finished_sending:
            raise RuntimeError("Audio requires an open sending stream")
        await self.call.write(frame_message(frame))

    async def send_gap(self, track_id: str, reason: str, next_start_ns: int | None) -> None:
        if self.call is None or self._finished_sending:
            raise RuntimeError("Gap requires an open sending stream")
        await self.call.write(pb.ClientMessage(gap=pb.AudioGap(
            track_id=track_id, reason=reason, next_start_time_ns=next_start_ns,
        )))

    async def finish_sending(self) -> None:
        if self.call is None or self._finished_sending:
            raise RuntimeError("Finish requires an open sending stream")
        self._finished_sending = True
        await self.call.write(pb.ClientMessage(finish=pb.FinishSession()))
        await self.call.done_writing()

    async def responses(self):
        if self.call is None or self._receiving:
            raise RuntimeError("Use one receiver after session open")
        self._receiving = True
        while True:
            event = await self.call.read()
            if event is grpc.aio.EOF:
                if not self._terminal:
                    raise GatewayProtocolError("Server ended without a session terminal")
                return
            self._validate_event(event)
            yield event

    async def __aexit__(self, exc_type, exc, traceback):
        if self.call is not None:
            self.call.cancel()
        if self.channel is not None:
            await self.channel.close(grace=.5)
