"""ASGI entry point. Use one process; model workers are asyncio tasks, not uvicorn replicas."""
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.api import health, sessions, websocket
from app.core.config import Settings
from app.metrics.collector import Metrics
from app.session.manager import SessionManager
from app.translation.openai_backend import OpenAITranslationBackend
from app.translation.scheduler import TranslationScheduler


def create_app(settings: Settings | None = None, *, backend=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = settings or Settings()
        if len(config.streaming_api_key.get_secret_value()) < 24:
            raise RuntimeError("Set STREAMING_API_KEY to a random secret of at least 24 characters before starting")
        metrics = Metrics()
        provider = backend or OpenAITranslationBackend(
            config.translation_base_url, default_model=config.translation_model,
            api_key=config.translation_api_key.get_secret_value(), timeout_s=config.translation_timeout)
        scheduler = TranslationScheduler(provider, workers=config.translation_workers,
                                         max_pending=config.max_pending_jobs,
                                         max_pending_per_session=config.max_pending_per_session, metrics=metrics)
        manager = SessionManager(config, scheduler, metrics)
        application.state.settings, application.state.metrics = config, metrics
        application.state.scheduler, application.state.manager = scheduler, manager
        await scheduler.start()
        await manager.start()
        try:
            yield
        finally:
            try:
                await manager.close()
                await scheduler.close(timeout_s=config.translation_timeout + 1)
            finally:
                await provider.close()

    application = FastAPI(title="MyVote streaming translation", version="1.0.0", lifespan=lifespan)

    @application.middleware("http")
    async def bounded_body(request: Request, call_next):
        limit = request.app.state.settings.max_event_bytes
        # Cap chunked bodies too; Content-Length alone is not an input bound.
        if request.method in ("POST", "PUT", "PATCH"):
            size = 0
            chunks = []
            try:
                async with asyncio.timeout(10):
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > limit:
                            return JSONResponse({"detail": "Request body too large"}, status_code=413)
                        chunks.append(chunk)
            except TimeoutError:
                return JSONResponse({"detail": "Request body timed out"}, status_code=408)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    application.include_router(sessions.router)
    application.include_router(health.router)
    application.include_router(websocket.router)
    return application


app = create_app()
