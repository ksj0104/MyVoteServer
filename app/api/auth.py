import secrets
from fastapi import HTTPException, Request


def bearer(value: str | None) -> str:
    if not value or len(value) > 512:
        return ""
    scheme, _, token = value.partition(" ")
    return token if scheme.lower() == "bearer" else ""


def require_admin(request: Request) -> None:
    supplied = bearer(request.headers.get("authorization"))
    expected = request.app.state.settings.streaming_api_key.get_secret_value()
    if not supplied or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})


def require_session(request: Request, session_id: str):
    session = request.app.state.manager.authorized(session_id, bearer(request.headers.get("authorization")))
    if session is None:
        # Do not reveal whether a session belongs to another user.
        raise HTTPException(404, "Session not found")
    return session
