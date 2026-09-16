"""Bounded in-process sessions and per-session capability authentication."""
import asyncio
import hashlib
import secrets
import time
from app.core.config import Settings
from app.core.models import SessionConfig
from app.metrics.collector import Metrics
from app.session.state import TranslationSession


class SessionManager:
    def __init__(self, settings: Settings, scheduler, metrics: Metrics) -> None:
        self.settings, self.scheduler, self.metrics = settings, scheduler, metrics
        self.sessions: dict[str, TranslationSession] = {}
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="session-cleanup")

    async def create(self, config: SessionConfig) -> tuple[TranslationSession, str]:
        if len(self.sessions) >= self.settings.max_sessions:
            raise ValueError("SESSION_CAPACITY")
        token = secrets.token_urlsafe(32)
        session = TranslationSession(secrets.token_urlsafe(18), config, self.settings, self.scheduler, self.metrics)
        session.token_hash = hashlib.sha256(token.encode()).digest()
        self.sessions[session.session_id] = session
        self.metrics.gauge("active_sessions", len(self.sessions))
        return session, token

    def authorized(self, session_id: str, token: str) -> TranslationSession | None:
        session = self.sessions.get(session_id)
        if session is None or session.closed or not token or len(token) > 256:
            return None
        if not secrets.compare_digest(session.token_hash, hashlib.sha256(token.encode()).digest()):
            return None
        return session

    async def delete(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is not None:
            await session.close()
        self.metrics.gauge("active_sessions", len(self.sessions))

    async def cleanup(self) -> None:
        cutoff = time.monotonic() - self.settings.session_inactivity_seconds
        for session_id, session in tuple(self.sessions.items()):
            if session.last_activity < cutoff:
                await self.delete(session_id)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(min(60, self.settings.session_inactivity_seconds))
            await self.cleanup()

    async def close(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
        for session_id in tuple(self.sessions):
            await self.delete(session_id)
