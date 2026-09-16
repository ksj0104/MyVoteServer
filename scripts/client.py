"""Authenticated ASR-text example client; never logs credentials.

The old Windows/gRPC audio client does not use this protocol. Reconnects obtain
a fresh snapshot only: already sent ASR events aren't automatically replayed.
Text display is disabled unless --show-text is explicitly selected.
"""

import argparse
import asyncio
import contextlib
import inspect
import json
import math
import os
import sys
import uuid
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from dotenv import load_dotenv
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake


SAFE_ERROR_CODES = frozenset({
    "STALE_SEQUENCE", "SOURCE_LIMIT", "SESSION_LIMIT", "LANGUAGE_MISMATCH",
    "SPEAKER_MISMATCH", "SESSION_MISMATCH", "CONFIG_REQUIRES_RESET", "MODEL_NOT_ALLOWED",
    "GLOSSARY_LIMIT", "INVALID_MESSAGE", "FINAL_REQUIRES_FINAL", "RATE_LIMIT",
    "SOURCE_RECONCILIATION_PENDING", "PIPELINE_ERROR", "INCOMPLETE_SOURCE",
    "QUEUE_OVERFLOW", "TRANSLATION_ERROR", "SNAPSHOT_TOO_LARGE",
    "SNAPSHOT_LIMIT", "UNSUPPORTED_SOURCE_LANGUAGE", "TRANSLATION_TIMEOUT",
    "BACKEND_UNAVAILABLE", "BACKEND_RATE_LIMITED", "BACKEND_INVALID_RESPONSE",
    "BACKEND_INCOMPLETE_RESPONSE", "BACKEND_RESPONSE_LIMIT",
    "invalid_model", "invalid_temperature", "backend_closed", "input_too_large",
    "rate_limited", "upstream_http_error", "response_too_large", "timeout",
    "transport_error", "incomplete_response", "invalid_response", "output_too_large",
    "invalid_backend_result", "queue_full", "stale_revision", "scheduler_closed",
})


class ClientError(Exception):
    """A deliberately safe error message, without tokens or server text."""


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def base_url(value):
    url = urlsplit(value)
    if (url.scheme not in ("http", "https") or not url.hostname
            or url.username or url.password or url.query or url.fragment):
        raise argparse.ArgumentTypeError("use an HTTP(S) API URL without credentials or query")
    return value.rstrip("/")


def add_connection_options(parser):
    parser.add_argument("--url", type=base_url,
                        default=os.getenv("STREAMING_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("STREAMING_API_KEY"),
                        help="admin key; prefer STREAMING_API_KEY in the environment or .env")
    parser.add_argument("--session-id", default=os.getenv("STREAMING_SESSION_ID"))
    parser.add_argument("--session-token", default=os.getenv("STREAMING_SESSION_TOKEN"),
                        help="session capability; prefer STREAMING_SESSION_TOKEN")
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-language", default="ko")
    parser.add_argument("--timeout", type=positive_number, default=90,
                        help="maximum wait for each matching final snapshot, seconds")
    parser.add_argument("--reconnects", type=int, choices=range(0, 6), default=2,
                        help="maximum connection recovery attempts; never replays ASR")
    parser.add_argument("--keep-session", action="store_true",
                        help="do not delete a session created by this process")
    parser.add_argument("--show-text", action="store_true",
                        help="explicitly display translated text on this terminal")


def safe_event(event, *, show_text=False):
    """Print only known fields; text display requires an explicit client option."""
    kind = event.get("type")
    if kind in ("session_state", "translation_update", "translation_final"):
        revision = event.get("revision")
        sequence = event.get("last_sequence")
        final = event.get("final")
        if type(revision) is int and type(sequence) is int and type(final) is bool:
            print(f"{kind}: revision={revision} last_sequence={sequence} final={final}", flush=True)
        else:
            print(f"{kind}: snapshot received", flush=True)
        if show_text and isinstance(event.get("full_text"), str):
            # json encoding prevents terminal escape/control sequences from running.
            print(json.dumps({"full_text": event["full_text"]}, ensure_ascii=False), flush=True)


class StreamingClient:
    def __init__(self, args):
        self.args = args
        self.url = base_url(args.url)
        self.session_id = args.session_id
        self.token = args.session_token
        self.created = False
        self.http = None
        self.websocket = None
        self.reader = None
        self.snapshot = {}
        self.next_sequence = 0
        self.error = None
        self.closing = False
        self.ready = asyncio.Event()
        self.changed = asyncio.Condition()
        self.send_lock = asyncio.Lock()

    @staticmethod
    def _headers(token):
        if not isinstance(token, str) or not token or any(c in token for c in "\r\n"):
            raise ClientError("Missing or invalid authentication credential")
        return {"Authorization": "Bearer " + token}

    async def _request(self, method, path, *, token, payload=None):
        try:
            response = await self.http.request(
                method, self.url + path, headers=self._headers(token), json=payload)
        except httpx.HTTPError as exc:
            raise ClientError("REST connection failed (" + type(exc).__name__ + ")") from None
        if response.status_code >= 400:
            raise ClientError(f"REST request rejected: HTTP {response.status_code}")
        if response.status_code == 204 or not response.content:
            return {}
        try:
            data = response.json()
        except ValueError:
            raise ClientError("REST returned invalid JSON") from None
        if not isinstance(data, dict):
            raise ClientError("REST returned an unexpected response shape")
        return data

    async def __aenter__(self):
        self.http = httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False)
        try:
            if bool(self.session_id) != bool(self.token):
                raise ClientError("Resuming requires both --session-id and a session token")
            if not self.session_id:
                if not self.args.api_key or len(self.args.api_key) < 24:
                    raise ClientError("Set STREAMING_API_KEY to the server's admin key (at least 24 characters)")
                data = await self._request("POST", "/sessions", token=self.args.api_key, payload={
                    "source_language": self.args.source_language,
                    "target_language": self.args.target_language,
                })
                self.session_id, self.token = data.get("session_id"), data.get("session_token")
                if not isinstance(self.session_id, str) or not isinstance(self.token, str):
                    raise ClientError("Session creation response is missing capability fields")
                self.created = True
            await self._open()
            self.reader = asyncio.create_task(self._receive(), name="streaming-client-receiver")
            return self
        except BaseException:
            await self.close()
            raise

    async def __aexit__(self, *unused):
        await self.close()

    async def _update(self, event):
        if not isinstance(event, dict):
            raise ClientError("WebSocket returned an unexpected message shape")
        kind = event.get("type")
        if kind in ("session_state", "translation_update", "translation_final"):
            if event.get("session_id") != self.session_id:
                raise ClientError("Snapshot belongs to a different session")
            sequence = event.get("last_sequence")
            if type(sequence) is not int:
                raise ClientError("Snapshot has no valid last_sequence")
            self.next_sequence = max(self.next_sequence, sequence + 1)
            self.snapshot = event
            safe_event(event, show_text=self.args.show_text)
        elif kind == "error":
            sequence = event.get("last_sequence")
            if type(sequence) is int:
                self.next_sequence = max(self.next_sequence, sequence + 1)
            code = event.get("code")
            if not isinstance(code, str) or code not in SAFE_ERROR_CODES:
                code = "SERVER_ERROR"
            if code == "SOURCE_RECONCILIATION_PENDING" and event.get("recoverable") is True:
                print("source correction pending: send its explicit final with the same utterance_id", flush=True)
            else:
                raise ClientError("Server error: " + code)
        elif kind != "pong":
            raise ClientError("Unknown WebSocket event type")
        async with self.changed:
            self.changed.notify_all()

    async def _open(self):
        parsed = urlsplit(self.url)
        path = parsed.path.rstrip("/") + "/ws/translate/" + quote(self.session_id, safe="")
        ws_url = urlunsplit(("wss" if parsed.scheme == "https" else "ws", parsed.netloc, path, "", ""))
        options = dict(additional_headers=self._headers(self.token), open_timeout=10,
                       ping_interval=20, ping_timeout=20, close_timeout=3,
                       max_size=524288, max_queue=16)
        # websockets 15 adds proxy support; 14 connects directly already.
        if "proxy" in inspect.signature(connect).parameters:
            options["proxy"] = None
        websocket = await connect(ws_url, **options)
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=10)
            event = json.loads(raw)
            await self._update(event)
            if event.get("type") != "session_state":
                raise ClientError("Connection did not begin with session_state")
        except BaseException:
            await websocket.close()
            raise
        self.websocket = websocket
        self.ready.set()

    async def _receive(self):
        attempts = 0
        try:
            while not self.closing:
                try:
                    async for raw in self.websocket:
                        await self._update(json.loads(raw))
                    if self.closing:
                        return
                    raise ClientError("WebSocket connection closed")
                except (ConnectionClosed, OSError, TimeoutError, ClientError) as exc:
                    self.ready.clear()
                    # Application errors are not transient connection failures.
                    if isinstance(exc, ClientError) and str(exc) != "WebSocket connection closed":
                        raise
                    while not self.closing:
                        if attempts >= self.args.reconnects:
                            raise ClientError("Connection recovery exhausted; no ASR was replayed") from None
                        attempts += 1
                        await asyncio.sleep(min(0.5 * 2 ** (attempts - 1), 4))
                        try:
                            await self._open()
                            print("connection recovered from current snapshot; no ASR replay", flush=True)
                            break
                        except (ConnectionClosed, InvalidHandshake, OSError, TimeoutError):
                            continue
        except asyncio.CancelledError:
            raise
        except (ClientError, ValueError, InvalidHandshake, OSError, TimeoutError) as exc:
            self.error = exc if isinstance(exc, ClientError) else ClientError("Connection failed (" + type(exc).__name__ + ")")
            self.ready.set()  # Wake a blocked sender so it can observe the error.
            async with self.changed:
                self.changed.notify_all()

    async def send(self, event):
        await asyncio.wait_for(self.ready.wait(), timeout=self.args.timeout)
        async with self.send_lock:
            if self.error:
                raise self.error
            try:
                await self.websocket.send(json.dumps(event, ensure_ascii=False, allow_nan=False))
            except (ConnectionClosed, OSError):
                # Acceptance is ambiguous. Never guess by resending the frame.
                raise ClientError("Send interrupted; inspect the next session snapshot before continuing") from None

    async def send_asr(self, text, utterance_id, *, final=False):
        await asyncio.wait_for(self.ready.wait(), timeout=self.args.timeout)
        sequence = self.next_sequence
        self.next_sequence += 1
        await self.send({"type": "asr_final" if final else "asr_partial",
                         "sequence": sequence, "text": text,
                         "utterance_id": utterance_id, "is_final": final})
        return sequence

    async def wait_final(self, sequence, utterance_id):
        def completed():
            return (self.snapshot.get("final") is True
                    and self.snapshot.get("last_sequence", -1) >= sequence
                    and any(item.get("utterance_id") == utterance_id
                            for item in self.snapshot.get("segments", []) if isinstance(item, dict)))

        async with asyncio.timeout(self.args.timeout):
            async with self.changed:
                await self.changed.wait_for(lambda: self.error is not None or completed())
        if self.error:
            raise self.error
        return self.snapshot

    async def disconnect_and_recover(self):
        """An explicit simulator fault injection; doesn't replay submitted source."""
        self.ready.clear()
        await self.websocket.close()
        await asyncio.wait_for(self.ready.wait(), timeout=self.args.timeout)
        if self.error:
            raise self.error

    async def reset(self):
        previous_generation = self.snapshot.get("generation_id")
        await self.send({"type": "reset_context"})
        async with asyncio.timeout(self.args.timeout):
            async with self.changed:
                await self.changed.wait_for(lambda: self.error is not None or
                    self.snapshot.get("generation_id") != previous_generation)
        if self.error:
            raise self.error

    async def close(self):
        self.closing = True
        if self.reader:
            self.reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reader
        if self.websocket:
            with contextlib.suppress(Exception):
                await self.websocket.close()
        if self.http:
            if self.created and not self.args.keep_session:
                try:
                    await self._request("DELETE", "/sessions/" + quote(self.session_id, safe=""), token=self.token)
                except ClientError:
                    print("Session cleanup failed; server inactivity expiry still applies", file=sys.stderr)
            await self.http.aclose()


async def interactive(args):
    async with StreamingClient(args) as client:
        if args.text is not None:
            utterance = "client-" + uuid.uuid4().hex
            sequence = await client.send_asr(args.text, utterance, final=True)
            await client.wait_final(sequence, utterance)
            return
        print("Each line is a new final ASR utterance. :reset clears context; :quit exits.")
        while True:
            try:
                text = await asyncio.to_thread(input, "ASR> ")
            except EOFError:
                return
            if text == ":quit":
                return
            if text == ":reset":
                await client.reset()
                continue
            if not text.strip():
                continue
            utterance = "client-" + uuid.uuid4().hex
            sequence = await client.send_asr(text, utterance, final=True)
            await client.wait_final(sequence, utterance)


def run_cli(operation, args):
    try:
        asyncio.run(operation(args))
    except KeyboardInterrupt:
        return 130
    except (ClientError, TimeoutError, httpx.HTTPError, InvalidHandshake, ConnectionClosed, OSError, ValueError) as exc:
        message = str(exc) if isinstance(exc, ClientError) else type(exc).__name__
        print("Client stopped: " + message, file=sys.stderr)
        return 1
    return 0


def main():
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    add_connection_options(parser)
    parser.add_argument("--text", help="send one complete final utterance and wait for its final snapshot")
    return run_cli(interactive, parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
