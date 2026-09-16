"""Bounded speaker-lane buffers sharing an externally admitted semantic provider.

The caller supplies lane identity from available speaker evidence. Unknown and
mixed lanes are opaque keys here; this router never guesses speaker identities.
Per-lane requests may finish out of order. Each result still owns exactly its
original source IDs, word timings and payloads.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import time

from .semantic_translation import (
    SemanticFailure, SemanticPendingUnit, SemanticTranslationConfig,
    SemanticTranslationCoordinator,
)


class SemanticTranslationRouter:
    def __init__(self, provider, on_commit, on_failure, *, target_language="ko",
                 config: SemanticTranslationConfig | None = None,
                 max_lanes=8, max_total_units=64):
        if type(max_lanes) is not int or not 1 <= max_lanes <= 32:
            raise ValueError("Semantic router max_lanes must be in [1, 32]")
        if type(max_total_units) is not int or not 1 <= max_total_units <= 64:
            raise ValueError("Semantic router max_total_units must be in [1, 64]")
        self.provider, self.on_commit, self.on_failure = provider, on_commit, on_failure
        self.target_language = target_language
        self.max_lanes, self.max_total_units = max_lanes, max_total_units
        self._config = config or SemanticTranslationConfig()
        self._lanes: OrderedDict[tuple, SemanticTranslationCoordinator] = OrderedDict()
        self._retired_counts: dict[str, int] = {}
        self._own_counts = {"failures": 0, "retired_units": 0}
        self._last_source_scope = None
        self._dialogue: list[tuple[tuple, str]] = []
        self._generation = 0
        self._closed = False
        self._closing = False
        self._resetting = False

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        if not isinstance(value, SemanticTranslationConfig):
            raise TypeError("Expected SemanticTranslationConfig")
        self._config = value
        for coordinator in self._lanes.values():
            coordinator.config = value

    @property
    def pending_units(self):
        return tuple(unit for coordinator in self._lanes.values() for unit in coordinator.pending_units)

    @property
    def counts(self):
        # Retiring an idle LRU lane never makes session diagnostics decrease.
        result = {key: 0 for key in ("requests", "waits", "commits", "committed_units",
                  "failures", "retired_units", "timeouts", "invalid", "stale")}
        for counters in (self._retired_counts, self._own_counts,
                         *(coordinator.counts for coordinator in self._lanes.values())):
            for key, value in counters.items():
                result[key] = result.get(key, 0) + value
        return result

    def _retire(self, scope):
        coordinator = self._lanes.pop(scope)
        for key, value in coordinator.counts.items():
            self._retired_counts[key] = self._retired_counts.get(key, 0) + value

    async def _failure(self, unit, reason):
        self._own_counts["failures"] += 1
        self._own_counts["retired_units"] += 1
        age_ms = max(0, time.monotonic() - unit.created_at_monotonic) * 1000
        await self.on_failure(SemanticFailure((unit,), reason,
            buffer_wait_ms=age_ms, total_ms=age_ms))

    def _context_for(self, scope):
        # Dialogue belongs to this session/router and capture run, never a shared
        # provider object. A different capture epoch cannot inherit old context.
        return tuple(text for previous, text in self._dialogue if previous[:2] == scope[:2])

    async def _committed(self, result):
        generation = self._generation
        await self.on_commit(result)
        if generation != self._generation or self._closed:
            return
        label = f"[{result.units[0].lane_id}] "
        source = "".join(unit.text for unit in result.units).strip()
        # Bounded context excerpts preserve source characters and their spacing;
        # generated translations/corrections never become dialogue context.
        excerpt = label + source[-max(1, 4000 - len(label)):]
        self._dialogue.append((result.units[0].scope, excerpt))
        self._dialogue = self._dialogue[-2:]

    async def append(self, unit: SemanticPendingUnit, *, boundary=False):
        if self._closed or self._closing or self._resetting:
            raise RuntimeError("Semantic router is closed")
        if not isinstance(unit, SemanticPendingUnit):
            raise TypeError("Expected an immutable semantic unit")
        if unit.created_at_monotonic > time.monotonic() + .1:
            raise ValueError("Semantic source must use this process's current monotonic clock")
        pending = self.pending_units
        if any(item.unit_id == unit.unit_id for item in pending):
            raise ValueError("Duplicate pending semantic unit ID across lanes")
        max_chars = min(4000, self.config.max_source_chars)
        if len(unit.text) > max_chars:
            self.request_flush()
            await self._failure(unit, "source_too_large")
            return
        if (len(pending) >= self.max_total_units
                or sum(len(item.text) for item in pending) + len(unit.text) > max_chars):
            self.request_flush()
            await self._failure(unit, "buffer_capacity")
            return
        scope = unit.scope
        coordinator = self._lanes.get(scope)
        if coordinator is None:
            if len(self._lanes) >= self.max_lanes:
                idle = next((key for key, lane in self._lanes.items() if lane.is_idle), None)
                if idle is None:
                    self.request_flush()
                    await self._failure(unit, "buffer_capacity")
                    return
                self._retire(idle)
            coordinator = SemanticTranslationCoordinator(self.provider, self._committed, self.on_failure,
                target_language=self.target_language, config=self.config,
                context_provider=lambda scope=scope: self._context_for(scope))
            self._lanes[scope] = coordinator
        self._lanes.move_to_end(scope)
        source_scope = scope[:3]
        if self._last_source_scope is not None and source_scope != self._last_source_scope:
            self.request_flush()
        self._last_source_scope = source_scope
        await coordinator.append(unit, boundary=boundary)
        if boundary:
            self.request_flush()

    def request_flush(self):
        if self._closed:
            return False
        changed = False
        for coordinator in tuple(self._lanes.values()):
            changed = coordinator.request_flush() or changed
        return changed

    async def flush(self):
        if self._closed:
            return
        snapshots = tuple((coordinator, tuple(unit.unit_id for unit in coordinator.unsettled_units))
                          for coordinator in self._lanes.values())
        self.request_flush()
        # Capture all checkpoints before the first await. A later append is not
        # accidentally sealed by scheduling several child flush coroutines.
        await asyncio.gather(*(coordinator.wait_for_units(ids) for coordinator, ids in snapshots))

    async def reset(self):
        self._generation += 1
        self._resetting = True
        self._dialogue.clear()
        coordinators = tuple(self._lanes.items())
        try:
            await asyncio.gather(*(coordinator.reset() for _, coordinator in coordinators))
        finally:
            self._resetting = False
        # Keep a cancellation-resistant provider owner until it becomes idle;
        # its physical in-flight job must still consume a lane and its counters.
        for scope, coordinator in coordinators:
            if self._lanes.get(scope) is coordinator and coordinator.is_idle:
                self._retire(scope)
        self._last_source_scope = None

    async def close(self, *, flush=True):
        if self._closed:
            return
        self._closing = True
        try:
            if flush:
                await self.flush()
        finally:
            self._closed = True
            self._dialogue.clear()
            coordinators = tuple(self._lanes.items())
            await asyncio.gather(*(coordinator.close(flush=False) for _, coordinator in coordinators))
            for scope, coordinator in coordinators:
                if self._lanes.get(scope) is coordinator and coordinator.is_idle:
                    self._retire(scope)
