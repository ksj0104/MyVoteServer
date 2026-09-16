"""One bounded writer per session. Model I/O never runs inside the receive loop."""
import asyncio
import json
import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from app.api.auth import bearer
from app.core.models import ASREvent, GlossaryUpdate, SessionConfig

router = APIRouter()


@router.websocket("/ws/translate/{session_id}")
async def translate_socket(websocket: WebSocket, session_id: str):
    manager = websocket.app.state.manager
    settings = websocket.app.state.settings
    token = bearer(websocket.headers.get("authorization"))
    await websocket.accept()
    writer = None
    receive = None
    session = None
    attached = False
    try:
        if not token:
            # Browser clients cannot set Authorization headers. Do not put
            # capability tokens into URL/query logs; authenticate in frame one.
            first_packet = await asyncio.wait_for(websocket.receive(), 5)
            raw = first_packet.get("text")
            if first_packet["type"] != "websocket.receive" or not isinstance(raw, str):
                await websocket.close(code=1008)
                return
            if len(raw.encode("utf-8")) > 1024:
                await websocket.close(code=1008)
                return
            first = json.loads(raw)
            if not isinstance(first, dict) or set(first) != {"type", "token"} or first["type"] != "authenticate" or not isinstance(first["token"], str):
                await websocket.close(code=1008)
                return
            token = first["token"]
        session = manager.authorized(session_id, token)
        if session is None:
            await websocket.close(code=1008)
            return
        queue = await session.attach()
        attached = True

        async def send_messages():
            while True:
                event = await queue.get()
                if event is None:
                    await websocket.close(code=1013)
                    return
                enqueued_at = event.pop("_enqueued_at", None)
                if enqueued_at is not None:
                    session.metrics.observe("client_queue_wait_ms", (asyncio.get_running_loop().time() - enqueued_at) * 1000)
                encoded = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                if len(encoded.encode("utf-8")) > settings.max_snapshot_bytes:
                    await websocket.send_json({"type": "error", "code": "SNAPSHOT_LIMIT",
                                               "message": "Snapshot limit reached", "recoverable": False})
                    await websocket.close(code=1009)
                    return
                started = asyncio.get_running_loop().time()
                await asyncio.wait_for(websocket.send_text(encoded), 10)
                session.metrics.observe("client_write_ms", (asyncio.get_running_loop().time() - started) * 1000)
                if event.get("type") == "translation_update" and enqueued_at is not None:
                    session.metrics.observe("model_result_to_client_ms", (asyncio.get_running_loop().time() - enqueued_at) * 1000)

        writer = asyncio.create_task(send_messages(), name="translation-ws-writer")
        while True:
            receive = asyncio.create_task(websocket.receive())
            done, _ = await asyncio.wait((receive, writer), return_when=asyncio.FIRST_COMPLETED)
            if writer in done:
                receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)
                await writer
                break
            packet = receive.result()
            if packet["type"] == "websocket.disconnect":
                break
            raw = packet.get("text")
            if raw is None:
                session.error("INVALID_MESSAGE", "Only UTF-8 JSON text frames are supported")
                continue
            if len(raw.encode("utf-8")) > settings.max_event_bytes:
                await websocket.close(code=1009)
                break
            try:
                if not session.admit_event():
                    session.error("RATE_LIMIT", "Too many events; slow down input")
                    continue
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("INVALID_MESSAGE")
                kind = data.get("type")
                if kind in ("asr_partial", "asr_final"):
                    await session.ingest(ASREvent.model_validate(data))
                elif kind == "glossary_update":
                    await session.set_glossary(GlossaryUpdate.model_validate({k: v for k, v in data.items() if k != "type"}).entries)
                    session.emit(session.snapshot())
                elif kind == "session_config":
                    await session.configure(SessionConfig.model_validate({k: v for k, v in data.items() if k != "type"}))
                elif kind == "reset_context" and set(data) == {"type"}:
                    await session.reset()
                elif kind == "ping" and set(data) <= {"type"}:
                    session.touch()
                    session.emit({"type": "pong", "last_sequence": session.last_sequence})
                else:
                    raise ValueError("INVALID_MESSAGE")
            except (ValidationError, json.JSONDecodeError, RecursionError):
                session.error("INVALID_MESSAGE", "Message does not match the bounded protocol schema")
            except ValueError as error:
                code = str(error)
                known = {"STALE_SEQUENCE", "SOURCE_LIMIT", "SESSION_LIMIT", "LANGUAGE_MISMATCH", "SPEAKER_MISMATCH", "SESSION_MISMATCH", "CONFIG_REQUIRES_RESET", "MODEL_NOT_ALLOWED", "UNSUPPORTED_SOURCE_LANGUAGE", "GLOSSARY_LIMIT", "INVALID_MESSAGE", "FINAL_REQUIRES_FINAL"}
                session.error(code if code in known else "INVALID_MESSAGE", "Input rejected; state is preserved")
    except (WebSocketDisconnect, TimeoutError, json.JSONDecodeError, ValueError, RecursionError):
        if not attached:
            try:
                await websocket.close(code=1008)
            except (RuntimeError, WebSocketDisconnect):
                pass
    finally:
        # ASGI servers may cancel the scope during disconnect. Finish bounded
        # cleanup without orphaning either receive tasks or session ownership.
        with anyio.CancelScope(shield=True):
            tasks = [task for task in (writer, receive) if task is not None]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if attached:
                await session.detach()
