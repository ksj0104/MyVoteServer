"""Complete-prefix selection with optional per-lane transcription-inactivity drain.

Install after semantic_boundary_routing and before importing the semantic
pipeline. Legacy mode preserves source-only incomplete retirement. Inactivity
mode translates the exact residual snapshot as spoken after no new transcription
or an explicit drain; it never asserts that silence implies a complete sentence.
The verified coordinator still owns snapshots, cancellation and publication.
"""

from dataclasses import replace
import asyncio
import math
import time


QUALITY_MIN_REQUEST_INTERVAL_S = 1.5
QUALITY_MAX_HOLD_S = 12
QUALITY_MAX_TOTAL_AGE_S = 45
QUALITY_REQUEST_TIMEOUT_S = 20


def _metadata(max_hold_s, max_total_age_s, request_timeout_s, min_request_interval_s,
              inactivity_flush_s):
    result = {
        "semantic_quality_policy": "complete-thought-bounded-wait-v2",
        "semantic_min_request_interval_ms": int(min_request_interval_s * 1000),
        "semantic_max_hold_ms": int(max_hold_s * 1000),
        "semantic_total_budget_ms": int(max_total_age_s * 1000),
        "semantic_request_timeout_ms": int(request_timeout_s * 1000),
    }
    if inactivity_flush_s:
        result.update(semantic_quality_policy="complete-prefix-inactivity-v3",
                      semantic_inactivity_flush_ms=int(inactivity_flush_s * 1000),
                      semantic_dispatch_policy="complete-prefix-or-transcript-inactivity-v1")
    return result


def install_quality_policy(routing_module, *, max_hold_s=QUALITY_MAX_HOLD_S,
                           max_total_age_s=QUALITY_MAX_TOTAL_AGE_S,
                           request_timeout_s=QUALITY_REQUEST_TIMEOUT_S,
                           min_request_interval_s=QUALITY_MIN_REQUEST_INTERVAL_S,
                           inactivity_flush_s=0):
    """Install once in the supplied routing module, without changing parsers."""
    from myvote_engine.translation import ProviderError
    timings = (max_hold_s, max_total_age_s, request_timeout_s, min_request_interval_s)
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not .001 <= value <= 120 for value in timings):
        raise ValueError("Quality timings must be finite positive seconds, at most 120")
    if max_hold_s > max_total_age_s or request_timeout_s > max_total_age_s:
        raise ValueError("Hold and request budgets cannot exceed source lifetime")
    if min_request_interval_s >= max_total_age_s:
        raise ValueError("Request pacing must be shorter than source lifetime")
    if (type(inactivity_flush_s) not in (int, float) or not math.isfinite(inactivity_flush_s)
            or not 0 <= inactivity_flush_s < max_total_age_s):
        raise ValueError("Inactivity flush must be zero (disabled) or shorter than source lifetime")
    signature = (*timings, inactivity_flush_s)
    coordinator_type = routing_module.SemanticTranslationCoordinator
    router_type = routing_module.SemanticTranslationRouter
    marker = "_myvote_semantic_quality_policy_v1"
    installed = (getattr(coordinator_type, marker, False),
                 getattr(router_type, marker, False))
    if all(installed):
        if coordinator_type._myvote_quality_timings != signature:
            raise ValueError("Quality policy already installed with different timings")
        return _metadata(*signature)
    if any(installed):
        raise RuntimeError("Partially installed semantic quality policy")

    class CompleteThoughtCoordinator(coordinator_type):
        _myvote_semantic_quality_policy_v1 = True
        _myvote_quality_original = coordinator_type
        _myvote_quality_timings = signature

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_transcription_at = None

        def note_transcription(self):
            """New source, not a heartbeat: never touch another speaker's lane."""
            if self._closed or not self._pending or not inactivity_flush_s:
                return False
            self._last_transcription_at = time.monotonic()
            self._ensure_worker()
            return True

        async def append(self, unit, *, boundary=False):
            await super().append(unit, boundary=boundary)
            if any(item.unit_id == unit.unit_id for item in self._pending):
                self.note_transcription()

        def _inactivity_due(self, now):
            return (bool(inactivity_flush_s) and self._last_transcription_at is not None
                    and now >= self._last_transcription_at + inactivity_flush_s)

        async def reset(self):
            try:
                return await super().reset()
            finally:
                self._last_transcription_at = None

        async def _loop(self):
            # Adapted from the verified loop. An age-based recheck is allowed
            # once per unchanged head, not continuously at the new fast input
            # cadence. Hard drains and newly appended source remain immediate
            # candidates. This avoids burning the GPU on the same WAIT text.
            rechecked = None
            try:
                while self._pending and not self._closed:
                    self._wake.clear()
                    units, explicit_force = self._head()
                    now = time.monotonic()
                    oldest = units[0].created_at_monotonic
                    if now >= oldest + self.config.max_total_age_s:
                        consumed = self._remove_prefix(len(units))
                        self._active_units = consumed
                        try:
                            await self._failure(consumed, "source_deadline", force=True)
                        finally:
                            self._active_units = ()
                            self._changed.set()
                        continue
                    ids = tuple(unit.unit_id for unit in units)
                    key = (self._generation, ids)
                    hard = explicit_force or self._closing
                    idle_due = self._inactivity_due(now)
                    age_due = not inactivity_flush_s and now >= oldest + self.config.max_hold_s
                    recheck_due = age_due and rechecked != key
                    fresh = ids != self._last_attempt
                    physical_busy = self._request_task is not None and not self._request_task.done()
                    eligible = fresh or hard or idle_due or recheck_due
                    # Meaning-ready input is eligible immediately. Only repeated
                    # calls share a short pacing interval, never a sentence timer.
                    due = (now if inactivity_flush_s and (hard or idle_due) else
                           max(self._last_started + self.config.min_request_interval_s,
                               oldest + (0 if inactivity_flush_s or hard or recheck_due
                                         else self.config.min_request_interval_s)))
                    if not physical_busy and eligible and now >= due:
                        if age_due:
                            rechecked = key
                        await self._query(units, hard or idle_due or recheck_due)
                        continue
                    deadlines = [oldest + self.config.max_total_age_s]
                    if inactivity_flush_s and not idle_due and self._last_transcription_at is not None:
                        deadlines.append(self._last_transcription_at + inactivity_flush_s)
                    elif not inactivity_flush_s and not age_due:
                        deadlines.append(oldest + self.config.max_hold_s)
                    if not physical_busy and eligible and due > now:
                        deadlines.append(due)
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=max(.001, min(deadlines) - now))
                    except TimeoutError:
                        pass
            finally:
                self._changed.set()

        async def _collect(self, request):
            generation = self._generation
            try:
                return await super()._collect(request)
            except ProviderError as exc:
                # A single transient stage timeout need not throw away the
                # immutable source snapshot. Retry once inside the SAME caller
                # deadline; never extend source age, overlap physical requests,
                # retry malformed output, or publish after a reset.
                if (exc.category != "timeout" or request.deadline_monotonic is None
                        or self._closed or generation != self._generation):
                    raise
                remaining = request.deadline_monotonic - time.monotonic()
                delay = self.config.min_request_interval_s
                if remaining <= delay:
                    raise
                await asyncio.sleep(delay)
                if self._closed or generation != self._generation:
                    raise asyncio.CancelledError
                self.counts["quality_retries"] = self.counts.get("quality_retries", 0) + 1
                return await super()._collect(replace(
                    request, budget_ms=max(.001, (request.deadline_monotonic - time.monotonic()) * 1000)))

        async def _query_active(self, units, forced):
            generation = self._generation
            waits = self.counts["waits"]
            started = time.monotonic()
            # max_hold is a recheck threshold, not a speech endpoint. Retiring
            # a partial here loses the prefix before its ending can arrive.
            # Only explicit stop/gap/flush boundaries may retire a WAIT; the
            # inherited loop still enforces the absolute source lifetime.
            terminal = self._closing or any(unit.unit_id in self._boundaries for unit in units)
            if inactivity_flush_s:
                idle_due = self._inactivity_due(started)
                residual = terminal or idle_due
                if idle_due and not terminal:
                    self.counts["inactivity_flushes"] = self.counts.get("inactivity_flushes", 0) + 1
                # Explicit residual translation is not a claim of grammatical
                # completeness. Provider translates this immutable snapshot as
                # spoken and strict forced parsing requires its final source ID.
                await super()._query_active(units, residual)
                return
            # Preserve the model/parser contract. Internal hard drain and age
            # limits no longer instruct the model to swallow unfinished source.
            await super()._query_active(units, False)
            if (terminal and self.counts["waits"] > waits
                    and generation == self._generation and not self._closed
                    and tuple(self._pending[:len(units)]) == units):
                # Retire only this unchanged snapshot, never a later append.
                # The outer _query keeps _active_units until publication settles.
                self._remove_prefix(len(units))
                await self._failure(units, "incomplete_source", force=True, started=started)

    class CompleteThoughtRouter(router_type):
        _myvote_semantic_quality_policy_v1 = True
        _myvote_quality_original = router_type

        def note_transcription(self, scope):
            if self._closed or self._closing or self._resetting:
                return False
            coordinator = self._lanes.get(scope)
            return coordinator.note_transcription() if coordinator is not None else False

        def __init__(self, *args, **kwargs):
            config = kwargs.get("config")
            if config is None:
                config = routing_module.SemanticTranslationConfig()
            kwargs["config"] = replace(
                config,
                min_request_interval_s=min_request_interval_s,
                max_hold_s=max_hold_s,
                max_total_age_s=max_total_age_s,
                request_timeout_s=request_timeout_s,
            )
            super().__init__(*args, **kwargs)

    routing_module.SemanticTranslationCoordinator = CompleteThoughtCoordinator
    routing_module.SemanticTranslationRouter = CompleteThoughtRouter
    return _metadata(*signature)
