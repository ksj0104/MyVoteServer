from fastapi import APIRouter, Depends, HTTPException, Request, Response
import json
from app.api.auth import require_admin, require_session
from app.core.models import GlossaryUpdate, SessionConfig

router = APIRouter()


@router.post("/sessions", status_code=201, dependencies=[Depends(require_admin)])
async def create_session(request: Request, config: SessionConfig):
    try:
        session, token = await request.app.state.manager.create(config)
    except ValueError as error:
        code = str(error)
        raise HTTPException(429 if code == "SESSION_CAPACITY" else 422, code) from None
    return {"session_id": session.session_id, "session_token": token,
            "websocket_path": f"/ws/translate/{session.session_id}", "protocol_version": "1"}


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str):
    session = require_session(request, session_id)
    async with session.lock:
        session.touch()
        snapshot = session.snapshot()
        if len(json.dumps(snapshot, ensure_ascii=False).encode("utf-8")) > session.settings.max_snapshot_bytes:
            raise HTTPException(413, "Snapshot limit reached; reset or start a new session")
        return snapshot


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(request: Request, session_id: str):
    require_session(request, session_id)
    await request.app.state.manager.delete(session_id)
    return Response(status_code=204)


@router.post("/sessions/{session_id}/glossary")
async def update_glossary(request: Request, session_id: str, body: GlossaryUpdate):
    session = require_session(request, session_id)
    try:
        await session.set_glossary(body.entries)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    return {"entries": len(session.glossary), "applies_to": "future_translation_jobs"}
