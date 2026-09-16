"""Serialize native ASR calls, giving queued primary captions priority.

An active native call cannot be preempted. Cancelling its awaiting client does
not release the execution slot: the owned worker waits for the actual call.
The gateway must still fence all native calls against cross-session reuse.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import math
import time


class SecondaryAdmissionTimeout(RuntimeError):
    """A correction could not enter the ASR resource within its waiting budget."""


@dataclass
class _Call:
    function: object
    args: tuple
    kwargs: dict
    future: asyncio.Future
    queued_at: float


class AsrInferenceScheduler:
    """Event-loop-owned primary/secondary queues and one synchronous call slot.

    ``secondary_admission`` can inspect primary window deadlines or backlog.
    It is checked again before each individual secondary call, including lane B.
    It is not a latency guarantee: an already started call holds the resource.
    ``run_native`` defaults to to_thread; a gateway can inject NativeCallFence.run.
    """

    def __init__(self, *, run_native=None, secondary_admission=None,
                 secondary_max_wait_s=2.0, max_primary_pending=8, max_secondary_pending=2):
        if (isinstance(secondary_max_wait_s, bool) or not math.isfinite(secondary_max_wait_s)
                or not 0 < secondary_max_wait_s <= 120):
            raise ValueError("secondary wait must be in (0,120] seconds")
        for limit in (max_primary_pending, max_secondary_pending):
            if type(limit) is not int or not 1 <= limit <= 64:
                raise ValueError("scheduler queue capacities must be integers in [1,64]")
        self.run_native = run_native or asyncio.to_thread
        self.secondary_admission = secondary_admission or (lambda: True)
        self.secondary_max_wait_s = secondary_max_wait_s
        self._limits = (max_primary_pending, max_secondary_pending)
        self._queues = (deque(), deque())
        self._changed = asyncio.Event()
        self._worker = None
        self._active = None
        self._closed = False
        self.counts = {"primary_calls": 0, "secondary_calls": 0,
                       "secondary_admission_timeouts": 0, "cancelled_queued": 0}

    @property
    def primary_pending(self):
        return sum(not item.future.done() for item in self._queues[0])

    @property
    def active(self):
        return self._active is not None

    def notify_admission_changed(self):
        self._changed.set()

    async def run_primary(self, function, *args, **kwargs):
        return await self._submit(0, function, args, kwargs)

    async def run_secondary(self, function, *args, **kwargs):
        return await self._submit(1, function, args, kwargs)

    async def _submit(self, priority, function, args, kwargs):
        if self._closed:
            raise RuntimeError("ASR scheduler is closed")
        queue = self._queues[priority]
        queue = deque(item for item in queue if not item.future.done())
        self._queues = tuple(queue if index == priority else existing
                             for index, existing in enumerate(self._queues))
        if len(queue) >= self._limits[priority]:
            raise RuntimeError("ASR scheduler queue is full")
        future = asyncio.get_running_loop().create_future()
        call = _Call(function, args, kwargs, future, time.monotonic())
        queue.append(call)
        self._changed.set()
        if self._worker is None:
            self._worker = asyncio.create_task(self._loop())
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Future cancellation never cancels the scheduler/native owner.
            future.cancel()
            self._changed.set()
            raise

    async def _next(self):
        while True:
            for queue in self._queues:
                while queue and queue[0].future.done():
                    queue.popleft()
                    self.counts["cancelled_queued"] += 1
            if self._queues[0]:
                return 0, self._queues[0].popleft()
            if not self._queues[1]:
                return None
            call = self._queues[1][0]
            if time.monotonic() - call.queued_at >= self.secondary_max_wait_s:
                self._queues[1].popleft()
                self.counts["secondary_admission_timeouts"] += 1
                call.future.set_exception(SecondaryAdmissionTimeout("secondary ASR admission expired"))
                continue
            try:
                admitted = bool(self.secondary_admission())
            except Exception as exc:
                self._queues[1].popleft()
                call.future.set_exception(exc)
                continue
            if admitted:
                return 1, self._queues[1].popleft()
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), .025)
            except TimeoutError:
                pass

    async def _loop(self):
        try:
            while (next_call := await self._next()) is not None:
                priority, call = next_call
                self._active = call
                self.counts["primary_calls" if priority == 0 else "secondary_calls"] += 1
                try:
                    result = await self.run_native(call.function, *call.args, **call.kwargs)
                except BaseException as exc:
                    if not call.future.done():
                        call.future.set_exception(exc)
                else:
                    if not call.future.done():
                        call.future.set_result(result)
                finally:
                    self._active = None
        finally:
            self._worker = None

    async def close(self, *, wait=False):
        """Stop new/queued calls. Optionally wait for the active native owner."""
        self._closed = True
        for queue in self._queues:
            while queue:
                future = queue.popleft().future
                if not future.done():
                    future.cancel()
        self._changed.set()
        worker = self._worker
        if wait and worker is not None:
            await asyncio.shield(worker)
