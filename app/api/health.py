from fastapi import APIRouter, Depends, Request, Response
from app.api.auth import require_admin

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    # Liveness is intentionally not an expensive model inference/readiness test.
    return {"status": "ok", "protocol_version": "1", "backend_quality_verified": False,
            "storage": "in_memory", "gateway": "asr_text"}


@router.get("/metrics", dependencies=[Depends(require_admin)])
async def metrics(request: Request):
    scheduler = request.app.state.scheduler
    request.app.state.metrics.gauge("active_workers", scheduler.active_count)
    request.app.state.metrics.gauge("queued_jobs", scheduler.pending_count)
    request.app.state.metrics.gauge("worker_utilization", scheduler.active_count / scheduler.worker_count)
    return Response(request.app.state.metrics.render(), media_type="text/plain; version=0.0.4")
