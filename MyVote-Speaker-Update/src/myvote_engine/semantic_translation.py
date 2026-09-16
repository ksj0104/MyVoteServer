"""Bounded, asynchronous semantic-prefix decisions over immutable ASR units.

The provider chooses a contiguous source prefix and returns its finished text in
one request. This module never guesses a translation when that request fails.
Original payloads are returned to the caller on both commit and retirement.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import json
import math
import time
import uuid

from .translation import ProviderError, SemanticTranslationRequest, SemanticUnit


@dataclass(frozen=True)
class SemanticTranslationConfig:
    min_request_interval_s: float = .8
    max_hold_s: float = 4
    request_timeout_s: float = 2.5
    max_total_age_s: float = 8
    max_units: int = 64
    max_source_chars: int = 4000
    max_response_chars: int = 32768
    max_result_chars: int = 24000

    def __post_init__(self):
        for name in ("min_request_interval_s", "max_hold_s", "request_timeout_s", "max_total_age_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 120:
                raise ValueError(f"Invalid {name}")
        if self.max_hold_s > self.max_total_age_s:
            raise ValueError("Semantic hold must not exceed total age")
        for name, limit in (("max_units", 64), ("max_source_chars", 12000),
                            ("max_response_chars", 65536), ("max_result_chars", 24000)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"Invalid {name}")


@dataclass(frozen=True)
class SemanticPendingUnit:
    unit_id: str
    text: str
    source_language: str
    source_track_id: str
    capture_epoch: str
    created_at_monotonic: float
    payload: object = field(repr=False, compare=False, default=None)
    lane_id: str = "mixed"

    def __post_init__(self):
        for name, limit in (("unit_id", 512), ("text", 12000), ("source_language", 63),
                            ("source_track_id", 512), ("capture_epoch", 512), ("lane_id", 512)):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError(f"Invalid semantic unit {name}")
            value.encode("utf-8")
        if (type(self.created_at_monotonic) not in (int, float)
                or not math.isfinite(self.created_at_monotonic)):
            raise ValueError("Semantic unit timestamp must be a finite local monotonic time")

    @property
    def scope(self):
        return self.source_track_id, self.capture_epoch, self.source_language, self.lane_id


@dataclass(frozen=True)
class SemanticDecision:
    action: str
    through_id: str | None = None
    text: str | None = None
    stage_metrics: dict[str, float] = field(default_factory=dict)


class InvalidSemanticResponse(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidSemanticResponse("Duplicate semantic JSON field")
        result[key] = value
    return result


def _reject_constant(value):
    raise InvalidSemanticResponse("Nonfinite semantic JSON constant")


def parse_semantic_response(text: str, units, *, force_flush: bool = False,
                            max_response_chars: int = 32768,
                            max_result_chars: int = 24000) -> SemanticDecision:
    """Validate the whole packet before authorizing one immutable source prefix."""
    if not isinstance(text, str) or len(text) > max_response_chars:
        raise InvalidSemanticResponse("Semantic response is too large or not text")
    ids = tuple(unit.unit_id for unit in units)
    if not ids or len(set(ids)) != len(ids):
        raise InvalidSemanticResponse("Semantic request units must be nonempty and unique")
    try:
        text.encode("utf-8")
        packet = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise InvalidSemanticResponse("Invalid semantic JSON") from exc
    if type(packet) is not dict:
        raise InvalidSemanticResponse("Semantic response must be an object")
    action = packet.get("action")
    if action == "wait":
        if set(packet) != {"action"} or force_flush:
            raise InvalidSemanticResponse("A forced semantic request must commit all units")
        return SemanticDecision("wait")
    if action != "commit" or set(packet) != {"action", "through_id", "text"}:
        raise InvalidSemanticResponse("Unexpected semantic response fields")
    through_id, result = packet["through_id"], packet["text"]
    if not isinstance(through_id, str) or through_id not in ids:
        raise InvalidSemanticResponse("Semantic prefix ends at an unknown unit")
    if force_flush and through_id != ids[-1]:
        raise InvalidSemanticResponse("A forced semantic request must commit all units")
    if not isinstance(result, str) or not result.strip() or len(result) > max_result_chars:
        raise InvalidSemanticResponse("Invalid semantic result text")
    try:
        result.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidSemanticResponse("Invalid semantic Unicode text") from exc
    return SemanticDecision("commit", through_id, result)


@dataclass(frozen=True)
class SemanticCommit:
    units: tuple[SemanticPendingUnit, ...]
    text: str
    request_id: str
    force_flush: bool
    buffer_wait_ms: float
    model_ms: float
    total_ms: float
    stage_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticFailure:
    units: tuple[SemanticPendingUnit, ...]
    reason: str
    request_id: str = ""
    force_flush: bool = False
    buffer_wait_ms: float = 0
    model_ms: float = 0
    total_ms: float = 0
    stage_metrics: dict[str, float] = field(default_factory=dict)


class SemanticTranslationCoordinator:
    """One logical/physical provider job, bounded source, and no ASR blocking.

    append() only adds immutable source and may report capacity retirement. flush()
    waits for the source present at its call; subsequent appends are independent.
    Failed source is retired through on_failure so the caller can retain its raw
    display. Provider cancellation that ignores cancellation never authorizes a
    late commit and prevents another physical request until cleanup actually ends.
    """
    def __init__(self, provider, on_commit, on_failure, *, target_language="ko",
                 config: SemanticTranslationConfig | None = None, context_provider=None):
        self.provider, self.on_commit, self.on_failure = provider, on_commit, on_failure
        self.target_language = target_language
        self.config = config or SemanticTranslationConfig()
        self.context_provider = context_provider
        self._pending: list[SemanticPendingUnit] = []
        self._boundaries: set[str] = set()
        self._context: list[str] = []
        self._context_scope = None
        self._last_started = -math.inf
        self._last_attempt: tuple[str, ...] = ()
        self._generation = 0
        self._worker: asyncio.Task | None = None
        self._request_task: asyncio.Task | None = None
        self._active_units: tuple[SemanticPendingUnit, ...] = ()
        self._wake = asyncio.Event()
        self._changed = asyncio.Event()
        self._closed = False
        self._closing = False
        self._callback_lock = asyncio.Lock()
        self.counts = {key: 0 for key in ("requests", "waits", "commits", "committed_units",
                       "failures", "retired_units", "timeouts", "invalid", "stale")}

    @property
    def pending_units(self):
        return tuple(self._pending)

    @property
    def unsettled_units(self):
        """Source whose model result or publication callback has not settled."""
        items = {unit.unit_id: unit for unit in (*self._active_units, *self._pending)}
        return tuple(items.values())

    @property
    def is_idle(self):
        return (not self.unsettled_units
            and (self._worker is None or self._worker.done())
            and (self._request_task is None or self._request_task.done()))

    def _ensure_worker(self):
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._loop(), name="myvote-semantic-translation")
        self._wake.set()

    def _remove_prefix(self, count):
        removed = tuple(self._pending[:count])
        del self._pending[:count]
        self._boundaries.difference_update(item.unit_id for item in removed)
        self._changed.set()
        return removed

    async def _failure(self, units, reason, *, request_id="", force=False, started=None):
        if not units:
            return
        now = time.monotonic()
        started = now if started is None else started
        failure = SemanticFailure(tuple(units), reason, request_id, force,
            max(0, started - units[0].created_at_monotonic) * 1000,
            max(0, now - started) * 1000,
            max(0, now - units[0].created_at_monotonic) * 1000)
        self.counts["failures"] += 1
        self.counts["retired_units"] += len(units)
        async with self._callback_lock:
            await self.on_failure(failure)

    async def append(self, unit: SemanticPendingUnit, *, boundary=False):
        if self._closed or self._closing:
            raise RuntimeError("Semantic coordinator is closed")
        if not isinstance(unit, SemanticPendingUnit):
            raise TypeError("Expected an immutable semantic unit")
        if unit.created_at_monotonic > time.monotonic() + .1:
            raise ValueError("Semantic source must use this process's current monotonic clock")
        if any(item.unit_id == unit.unit_id for item in self._pending):
            raise ValueError("Duplicate pending semantic unit ID")
        if len(unit.text) > self.config.max_source_chars:
            await self._failure((unit,), "source_too_large")
            return
        # Retiring the newest unit leaves an existing in-flight prefix immutable.
        # Audio/source display remains the caller's responsibility on_failure.
        if (len(self._pending) >= self.config.max_units
                or sum(len(item.text) for item in self._pending) + len(unit.text) > self.config.max_source_chars):
            if self._pending:
                self._boundaries.add(self._pending[-1].unit_id)
                self._ensure_worker()
            await self._failure((unit,), "buffer_capacity")
            return
        self._pending.append(unit)
        if boundary:
            self._boundaries.add(unit.unit_id)
        self._ensure_worker()

    def _head(self):
        if not self._pending:
            return (), False
        scope = self._pending[0].scope
        units = []
        forced = False
        for unit in self._pending:
            if unit.scope != scope:
                forced = True
                break
            units.append(unit)
            if unit.unit_id in self._boundaries:
                forced = True
                break
        return tuple(units), forced

    def request_flush(self):
        """Seal currently pending source without awaiting inference or its sink.

        Useful for a VAD/final endpoint with no new stable words. A later append
        remains after this boundary and is never consumed by its forced request.
        """
        if self._closed or not self._pending:
            return False
        self._boundaries.add(self._pending[-1].unit_id)
        self._ensure_worker()
        return True

    def _request_done(self, task):
        if self._request_task is task:
            self._request_task = None
        if not task.cancelled():
            task.exception()  # Retrieve late failures from cancellation-resistant providers.
        self._wake.set()

    async def _collect(self, request):
        stream = self.provider.semantic_stream(request)
        parts, size = [], 0
        try:
            async for chunk in stream:
                if not isinstance(chunk.text, str):
                    raise InvalidSemanticResponse("Invalid provider chunk text")
                size += len(chunk.text)
                if size > self.config.max_response_chars:
                    raise InvalidSemanticResponse("Semantic response is too large")
                parts.append(chunk.text)
                if chunk.completed:
                    if chunk.finish_reason != "stop":
                        raise InvalidSemanticResponse("Semantic stream did not finish normally")
                    decision = parse_semantic_response("".join(parts), request.units,
                        force_flush=request.force_flush,
                        max_response_chars=self.config.max_response_chars,
                        max_result_chars=self.config.max_result_chars)
                    metrics = {}
                    usage = getattr(chunk, "usage", None)
                    stages = usage.get("stage_ms") if type(usage) is dict else None
                    if type(stages) is dict:
                        for key in ("selection", "translation"):
                            value = stages.get(key)
                            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                                metrics[key] = float(value)
                    return replace(decision, stage_metrics=metrics)
            raise InvalidSemanticResponse("Semantic stream ended without completion")
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()

    async def _query(self, units, forced):
        self._active_units = units
        try:
            await self._query_active(units, forced)
        finally:
            self._active_units = ()
            self._changed.set()

    async def _query_active(self, units, forced):
        started, generation = time.monotonic(), self._generation
        deadline = min(started + self.config.request_timeout_s,
                       units[0].created_at_monotonic + self.config.max_total_age_s)
        context = list(self._context[-2:]) if self._context_scope == units[0].scope else []
        if self.context_provider is not None:
            external = self.context_provider()
            if (not isinstance(external, tuple) or len(external) > 2
                    or any(not isinstance(text, str) or not text.strip() for text in external)):
                raise ValueError("Semantic external context requires up to two immutable source spans")
            context.extend(external)
        while context and sum(map(len, context)) > 12000:
            context.pop(0)
        request = SemanticTranslationRequest(uuid.uuid4().hex,
            tuple(SemanticUnit(unit.unit_id, unit.text) for unit in units),
            units[0].source_language, target_language=self.target_language,
            context=tuple(context),
            force_flush=forced, budget_ms=max(.001, (deadline - started) * 1000),
            deadline_monotonic=deadline)
        self._last_started = started
        self._last_attempt = tuple(unit.unit_id for unit in units)
        self.counts["requests"] += 1
        task = self._request_task = asyncio.create_task(self._collect(request))
        task.add_done_callback(self._request_done)
        reason = None
        try:
            done, _ = await asyncio.wait((task,), timeout=max(0, deadline - time.monotonic()))
            if not done or time.monotonic() >= deadline:
                task.cancel()
                self.counts["timeouts"] += 1
                reason = "model_timeout"
            else:
                decision = task.result()
        except asyncio.CancelledError:
            task.cancel()
            raise
        except InvalidSemanticResponse:
            self.counts["invalid"] += 1
            reason = "invalid_model_response"
        except ProviderError as exc:
            # Categories are a closed diagnostic vocabulary. Never forward an
            # arbitrary provider message, response body or unrecognized category.
            category = exc.category if isinstance(exc.category, str) else None
            if category == "timeout":
                self.counts["timeouts"] += 1
                reason = "model_timeout"
            elif category in ("invalid_output", "incomplete"):
                self.counts["invalid"] += 1
                reason = "invalid_model_response"
            elif category in ("unsupported_language", "source_limit", "invalid_source", "unsupported_profile"):
                reason = category
            else:
                reason = "provider_error"
        except Exception:
            reason = "provider_error"
        # All authorization is rechecked after awaits. Later appended source is
        # excluded from this snapshot and can never be consumed by its result.
        if (generation != self._generation or self._closed
                or tuple(self._pending[:len(units)]) != units):
            self.counts["stale"] += 1
            return
        if reason is not None:
            self._remove_prefix(len(units))
            await self._failure(units, reason, request_id=request.request_id, force=forced, started=started)
            return
        if decision.action == "wait":
            self.counts["waits"] += 1
            return
        count = next(index + 1 for index, unit in enumerate(units) if unit.unit_id == decision.through_id)
        consumed = self._remove_prefix(count)
        now = time.monotonic()
        commit = SemanticCommit(consumed, decision.text, request.request_id, forced,
            max(0, started - consumed[0].created_at_monotonic) * 1000,
            max(0, now - started) * 1000,
            max(0, now - consumed[0].created_at_monotonic) * 1000,
            stage_metrics=decision.stage_metrics)
        async with self._callback_lock:
            if generation != self._generation or self._closed:
                self.counts["stale"] += 1
                return
            await self.on_commit(commit)
        self.counts["commits"] += 1
        self.counts["committed_units"] += len(consumed)
        if self._context_scope != consumed[0].scope:
            self._context = []
        self._context_scope = consumed[0].scope
        # ASR word pieces already carry their original spacing. Inserting spaces
        # corrupts Korean/Japanese tokens and changes the source supplied as context.
        self._context.append("".join(unit.text for unit in consumed).strip())
        self._context = self._context[-2:]
        while len(self._context) > 1 and sum(map(len, self._context)) > 12000:
            self._context.pop(0)

    async def _loop(self):
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
                forced = explicit_force or now >= oldest + self.config.max_hold_s or self._closing
                fresh = tuple(unit.unit_id for unit in units) != self._last_attempt
                physical_busy = self._request_task is not None and not self._request_task.done()
                due = max(self._last_started + self.config.min_request_interval_s,
                          oldest + (0 if forced else self.config.min_request_interval_s))
                if not physical_busy and (fresh or forced) and now >= due:
                    await self._query(units, forced)
                    continue
                deadlines = [oldest + self.config.max_total_age_s]
                if not forced:
                    deadlines.append(oldest + self.config.max_hold_s)
                if not physical_busy and (fresh or forced) and due > now:
                    deadlines.append(due)
                delay = max(.001, min(deadlines) - now)
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=delay)
                except TimeoutError:
                    pass
        finally:
            self._changed.set()

    async def flush(self):
        if self._closed or not self.unsettled_units:
            return
        ids = {unit.unit_id for unit in self.unsettled_units}
        self.request_flush()
        await self.wait_for_units(ids)

    async def wait_for_units(self, unit_ids):
        """Wait for an already sealed source checkpoint, without sealing new input."""
        ids = frozenset(unit_ids)
        if self._closed or not ids:
            return
        self._ensure_worker()
        while any(unit.unit_id in ids for unit in self.unsettled_units) and not self._closed:
            self._changed.clear()
            # Worker callback failures are programming/sink errors, not a reason
            # to silently claim that a final flush completed successfully.
            changed = asyncio.create_task(self._changed.wait())
            try:
                done, _ = await asyncio.wait((changed, self._worker), return_when=asyncio.FIRST_COMPLETED)
                if self._worker in done:
                    self._worker.result()
                    break
            finally:
                changed.cancel()
                await asyncio.gather(changed, return_exceptions=True)

    async def reset(self):
        self._generation += 1
        worker, self._worker = self._worker, None
        if self._request_task is not None:
            self._request_task.cancel()
        if worker is not None and not worker.done():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        pending = self._remove_prefix(len(self._pending))
        self._context.clear()
        self._context_scope = None
        self._last_attempt = ()
        if pending:
            await self._failure(pending, "reset")

    async def close(self, *, flush=True):
        if self._closed:
            return
        self._closing = True
        try:
            if flush:
                await self.flush()
        finally:
            self._closed = True
            await self.reset()
