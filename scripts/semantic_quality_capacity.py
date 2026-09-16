"""Quality-only original-word capacity with bounded request projections.

Install after semantic_boundary_routing, before semantic_quality_policy. The
verified engine files and its <=64-unit request/config validation remain intact.
Only model requests group consecutive words; pending units, payloads, source
clocks and publication ownership always remain the original immutable objects.

The append methods and >64-unit query lifecycle below are adapted from the
verified semantic-translation-2026-09-16 semantic_routing/semantic_translation
methods. Keep their admission, cancellation and post-await authorization checks
in step with that manifest-bound release. Small queries delegate unchanged.
"""

import asyncio
import time
import uuid


QUALITY_PENDING_UNITS = 256
REQUEST_UNIT_LIMIT = 64


def _capacity(value):
    if type(value) is not int or not 1 <= value <= QUALITY_PENDING_UNITS:
        raise ValueError("Quality pending unit capacity must be in [1, 256]")
    return value


def project_request_units(units, semantic_unit_type):
    """Preserve exact source and expose only original block-ending IDs."""
    if not 1 <= len(units) <= QUALITY_PENDING_UNITS:
        raise ValueError("Quality request projection requires 1..256 original units")
    width = (len(units) + REQUEST_UNIT_LIMIT - 1) // REQUEST_UNIT_LIMIT
    return tuple(semantic_unit_type(block[-1].unit_id,
                                   "".join(unit.text for unit in block))
                 for offset in range(0, len(units), width)
                 for block in (units[offset:offset + width],))


def install_quality_capacity(routing_module):
    """Install bounded subclasses in one gateway's routing module only."""
    from myvote_engine import semantic_translation as semantic

    metadata = {
        "semantic_quality_capacity": "original-words-request-blocks-v1",
        "semantic_pending_unit_capacity": QUALITY_PENDING_UNITS,
        "semantic_request_unit_limit": REQUEST_UNIT_LIMIT,
    }
    coordinator_type = routing_module.SemanticTranslationCoordinator
    router_type = routing_module.SemanticTranslationRouter
    marker = "_myvote_quality_capacity_v1"
    installed = (getattr(coordinator_type, marker, False),
                 getattr(router_type, marker, False))
    if all(installed):
        return metadata
    if any(installed):
        raise RuntimeError("Partially installed semantic quality capacity")
    if not all(getattr(cls, "_myvote_soft_lane_boundaries_v1", False)
               for cls in (coordinator_type, router_type)):
        raise ValueError("Install natural semantic routing before quality capacity")
    if any(getattr(cls, "_myvote_semantic_quality_policy_v1", False)
           for cls in (coordinator_type, router_type)):
        raise ValueError("Install quality capacity before the quality policy")

    class QualityCapacityCoordinator(coordinator_type):
        _myvote_quality_capacity_v1 = True
        _myvote_capacity_original = coordinator_type

        def __init__(self, *args, max_pending_units=QUALITY_PENDING_UNITS, **kwargs):
            self.max_pending_units = _capacity(max_pending_units)
            super().__init__(*args, **kwargs)

        async def append(self, unit, *, boundary=False):
            if self._closed or self._closing:
                raise RuntimeError("Semantic coordinator is closed")
            if not isinstance(unit, semantic.SemanticPendingUnit):
                raise TypeError("Expected an immutable semantic unit")
            if unit.created_at_monotonic > time.monotonic() + .1:
                raise ValueError("Semantic source must use this process's current monotonic clock")
            if any(item.unit_id == unit.unit_id for item in self._pending):
                raise ValueError("Duplicate pending semantic unit ID")
            max_chars = min(4000, self.config.max_source_chars)
            if len(unit.text) > max_chars:
                await self._failure((unit,), "source_too_large")
                return
            # Only this count differs from upstream. Never evict or rewrite an
            # existing in-flight original prefix to admit its newest word.
            if (len(self._pending) >= self.max_pending_units
                    or sum(len(item.text) for item in self._pending) + len(unit.text) > max_chars):
                if self._pending:
                    self._boundaries.add(self._pending[-1].unit_id)
                    self._ensure_worker()
                await self._failure((unit,), "buffer_capacity")
                return
            self._pending.append(unit)
            if boundary:
                self._boundaries.add(unit.unit_id)
            self._ensure_worker()

        async def _query_active(self, units, forced):
            if len(units) <= REQUEST_UNIT_LIMIT:
                return await super()._query_active(units, forced)
            # The lifecycle is the verified algorithm; only request.units is a
            # projection. In particular, all snapshot checks use original units.
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
            request = semantic.SemanticTranslationRequest(uuid.uuid4().hex,
                project_request_units(units, semantic.SemanticUnit),
                units[0].source_language, target_language=self.target_language,
                context=tuple(context), force_flush=forced,
                budget_ms=max(.001, (deadline - started) * 1000),
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
            except semantic.InvalidSemanticResponse:
                self.counts["invalid"] += 1
                reason = "invalid_model_response"
            except semantic.ProviderError as exc:
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
            if (generation != self._generation or self._closed
                    or tuple(self._pending[:len(units)]) != units):
                self.counts["stale"] += 1
                return
            if reason is not None:
                self._remove_prefix(len(units))
                await self._failure(units, reason, request_id=request.request_id,
                                    force=forced, started=started)
                return
            if decision.action == "wait":
                self.counts["waits"] += 1
                return
            # Every exposed block ID is its last original ID. Strict parsing in
            # inherited _collect rejects IDs inside a block or outside a request.
            count = next(index + 1 for index, unit in enumerate(units)
                         if unit.unit_id == decision.through_id)
            consumed = self._remove_prefix(count)
            now = time.monotonic()
            commit = semantic.SemanticCommit(consumed, decision.text, request.request_id, forced,
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
            self._context.append("".join(unit.text for unit in consumed).strip())
            self._context = self._context[-2:]
            while len(self._context) > 1 and sum(map(len, self._context)) > 12000:
                self._context.pop(0)

    class QualityCapacityRouter(router_type):
        _myvote_quality_capacity_v1 = True
        _myvote_capacity_original = router_type

        def __init__(self, *args, max_total_units=QUALITY_PENDING_UNITS, **kwargs):
            capacity = _capacity(max_total_units)
            # The verified constructor continues validating every other input.
            super().__init__(*args, max_total_units=min(capacity, REQUEST_UNIT_LIMIT), **kwargs)
            self.max_total_units = capacity

        async def append(self, unit, *, boundary=False):
            if self._closed or self._closing or self._resetting:
                raise RuntimeError("Semantic router is closed")
            if not isinstance(unit, semantic.SemanticPendingUnit):
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
                # Resolve dynamically so the subsequently installed quality
                # policy still wraps the child coordinator and force contract.
                coordinator = routing_module.SemanticTranslationCoordinator(
                    self.provider, self._committed, self.on_failure,
                    target_language=self.target_language, config=self.config,
                    context_provider=lambda scope=scope: self._context_for(scope),
                    max_pending_units=self.max_total_units)
                self._lanes[scope] = coordinator
            self._lanes.move_to_end(scope)
            source_scope = scope[:3]
            if self._last_source_scope is not None and source_scope != self._last_source_scope:
                self.request_flush()
            self._last_source_scope = source_scope
            await coordinator.append(unit, boundary=False)
            if boundary:
                self.request_boundary(scope)

    routing_module.SemanticTranslationCoordinator = QualityCapacityCoordinator
    routing_module.SemanticTranslationRouter = QualityCapacityRouter
    return metadata
