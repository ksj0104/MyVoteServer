"""Bounded round-robin admission with one physical backend job per session.

Accepted finals are never evicted for queue pressure. Explicit session/segment
cancellation may retire them. A cancellation-resistant backend retains its
worker and session slot until it actually returns; its result is discarded.
Callbacks must also recheck application generation/revision before publishing.
"""

import asyncio
from collections import deque
from dataclasses import dataclass
import math
import time

from .backend import TranslationError
from .worker import TranslationJob, record_metric, translate_job


class QueueCapacityError(TranslationError):
    def __init__(self):
        super().__init__("queue_full", retryable=True)


class StaleJobError(TranslationError):
    def __init__(self):
        super().__init__("stale_revision")


class SchedulerClosedError(TranslationError):
    def __init__(self):
        super().__init__("scheduler_closed")


@dataclass
class _Entry:
    job: TranslationJob
    callback: object


@dataclass
class _Active:
    entry: _Entry
    task: asyncio.Task | None = None
    stale: bool = False


def _same_segment(left, right):
    return (left.session_id, left.generation_id, left.segment_id) == (
        right.session_id, right.generation_id, right.segment_id)


class TranslationScheduler:
    def __init__(self, backend, *, workers=4, max_pending=128, max_pending_per_session=32, metrics=None):
        for value in (workers, max_pending, max_pending_per_session):
            if type(value) is not int or value < 1:
                raise ValueError("Scheduler capacities must be positive integers")
        if workers > 64:
            raise ValueError("At most 64 workers are supported")
        self.backend, self.metrics = backend, metrics
        self.worker_count = workers
        self.max_pending, self.max_pending_per_session = max_pending, max_pending_per_session
        self._condition = asyncio.Condition()
        self._pending = {}
        self._ready = deque()
        self._ready_set = set()
        self._active = {}
        self._workers = []
        self._started = False
        self._closing = False

    @property
    def pending_count(self):
        return sum(len(queue) for queue in self._pending.values())

    @property
    def active_count(self):
        return len(self._active)

    async def start(self):
        async with self._condition:
            if self._closing:
                raise SchedulerClosedError()
            if not self._started:
                self._started = True
                self._workers = [asyncio.create_task(self._worker(), name=f"text-translation-worker-{index}")
                                 for index in range(self.worker_count)]

    def _mark_ready(self, session_id):
        if (self._pending.get(session_id) and session_id not in self._active
                and session_id not in self._ready_set and not self._closing):
            self._ready.append(session_id)
            self._ready_set.add(session_id)

    def _forget_ready(self, session_id):
        self._ready_set.discard(session_id)
        self._ready = deque(item for item in self._ready if item != session_id)

    def _cancel_active(self, active):
        if not active.stale:
            active.stale = True
            if active.task is not None and not active.task.done():
                active.task.cancel()
            record_metric(self.metrics, "translation_cancelled")

    async def submit(self, job, callback):
        if not isinstance(job, TranslationJob) or not callable(callback):
            raise TypeError("submit requires TranslationJob and an async callback")
        await self.start()
        async with self._condition:
            if self._closing:
                raise SchedulerClosedError()
            queue = list(self._pending.get(job.session_id, ()))
            active = self._active.get(job.session_id)
            peers = queue + ([active.entry] if active is not None and not active.stale else [])
            if any(_same_segment(entry.job, job) and entry.job.source_revision > job.source_revision
                   for entry in peers):
                raise StaleJobError()
            replacements = [index for index, entry in enumerate(queue)
                            if not entry.job.final and _same_segment(entry.job, job)
                            and entry.job.source_revision <= job.source_revision]
            if (self.pending_count - len(replacements) + 1 > self.max_pending
                    or len(queue) - len(replacements) + 1 > self.max_pending_per_session):
                record_metric(self.metrics, "translation_queue_rejected")
                raise QueueCapacityError()
            entry = _Entry(job, callback)
            if replacements:
                replacement_set = set(replacements)
                queue = [entry if index == replacements[0] else old
                         for index, old in enumerate(queue)
                         if index not in replacement_set or index == replacements[0]]
                record_metric(self.metrics, "translation_partial_coalesced", len(replacements))
            else:
                queue.append(entry)
            self._pending[job.session_id] = deque(queue)
            # A newly accepted revision can cancel a speculative active partial,
            # but never an accepted final without explicit invalidate/cancel.
            if (active is not None and not active.entry.job.final and _same_segment(active.entry.job, job)
                    and active.entry.job.source_revision <= job.source_revision):
                self._cancel_active(active)
            self._mark_ready(job.session_id)
            record_metric(self.metrics, "translation_submitted")
            self._condition.notify_all()

    async def cancel_session(self, session_id):
        async with self._condition:
            self._pending.pop(session_id, None)
            self._forget_ready(session_id)
            if session_id in self._active:
                self._cancel_active(self._active[session_id])
            self._condition.notify_all()

    async def invalidate(self, session_id, segment_ids):
        if isinstance(segment_ids, (str, bytes)):
            raise TypeError("segment_ids must be a collection of identifiers")
        segment_ids = set(segment_ids)
        if any(not isinstance(value, str) or not value for value in segment_ids):
            raise ValueError("Invalid segment identifier")
        async with self._condition:
            queue = self._pending.get(session_id)
            if queue is not None:
                remaining = deque(entry for entry in queue if entry.job.segment_id not in segment_ids)
                if remaining:
                    self._pending[session_id] = remaining
                else:
                    self._pending.pop(session_id, None)
                    self._forget_ready(session_id)
            active = self._active.get(session_id)
            if active is not None and active.entry.job.segment_id in segment_ids:
                self._cancel_active(active)
            self._condition.notify_all()

    async def _take(self):
        async with self._condition:
            while not self._closing:
                while self._ready:
                    session_id = self._ready.popleft()
                    self._ready_set.discard(session_id)
                    queue = self._pending.get(session_id)
                    if not queue or session_id in self._active:
                        continue
                    active = _Active(queue.popleft())
                    if not queue:
                        self._pending.pop(session_id, None)
                    self._active[session_id] = active
                    return active
                await self._condition.wait()
            return None

    async def _execute(self, active):
        if active.stale:
            return
        job = active.entry.job
        record_metric(self.metrics, "translation_queue_wait_ms",
                      max(0, time.monotonic() - job.created_at) * 1000, observation=True)
        active.task = asyncio.create_task(translate_job(self.backend, job), name="text-translation-request")
        result, error = None, None
        try:
            result = await asyncio.shield(active.task)
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                self._cancel_active(active)
                raise
            error = TranslationError("backend_cancelled", retryable=True)
        except TranslationError as exc:
            error = exc
        except Exception:
            error = TranslationError("backend_error", retryable=True)
        if active.stale or self._closing:
            record_metric(self.metrics, "translation_stale_discarded")
            return
        if result is not None:
            record_metric(self.metrics, "translation_latency_ms", result.latency_ms, observation=True)
        record_metric(self.metrics, "translation_failed" if error is not None else "translation_completed")
        try:
            await active.entry.callback(job, result, error)
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
            record_metric(self.metrics, "translation_callback_failed")
        except Exception:
            record_metric(self.metrics, "translation_callback_failed")

    async def _worker(self):
        while (active := await self._take()) is not None:
            try:
                await self._execute(active)
            finally:
                async with self._condition:
                    session_id = active.entry.job.session_id
                    if self._active.get(session_id) is active:
                        self._active.pop(session_id)
                    self._mark_ready(session_id)
                    self._condition.notify_all()

    async def close(self, *, timeout_s=5):
        """Cancel admitted work; timeout leaves resistant jobs tracked and closed.

        This scheduler does not own/close the backend. The application closes it
        separately. Calling close again can await a previously resistant job.
        """
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Shutdown timeout must be finite and positive")
        async with self._condition:
            self._closing = True
            self._pending.clear()
            self._ready.clear()
            self._ready_set.clear()
            for active in self._active.values():
                self._cancel_active(active)
            self._condition.notify_all()
        if self._workers:
            done, pending = await asyncio.wait(self._workers, timeout=timeout_s)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    record_metric(self.metrics, "translation_worker_failed")
            if pending:
                raise TimeoutError("Translation workers are still draining cancelled requests")
