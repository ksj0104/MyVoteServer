"""Scoped natural boundaries for the checksum-verified semantic router.

A natural boundary stops a request from crossing a turn, but permits the model
to wait or select a shorter prefix. Explicit drain, capacity and source-scope
changes still use the upstream hard flush. Source ownership, deadlines, strict
response parsing and failure publication stay in the original coordinator.
Install before importing semantic_pipeline or constructing router instances.
"""

import time


def _lane_generation(scope):
    prefix, separator, _ = scope[3].partition(":")
    if separator and prefix.startswith("boundary-g") and prefix[len("boundary-g"):].isdigit():
        return prefix
    return None


def install_routing(routing_module):
    """Install once in the supplied verified routing module; return metadata."""
    coordinator_type = routing_module.SemanticTranslationCoordinator
    router_type = routing_module.SemanticTranslationRouter
    marker = "_myvote_soft_lane_boundaries_v1"
    installed = (getattr(coordinator_type, marker, False),
                 getattr(router_type, marker, False))
    metadata = {"semantic_boundary_routing": "soft-lane-v1"}
    if all(installed):
        return metadata
    if any(installed):
        raise RuntimeError("Partially installed semantic boundary routing")

    class NaturalBoundaryCoordinator(coordinator_type):
        _myvote_soft_lane_boundaries_v1 = True
        _myvote_boundary_original = coordinator_type

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._soft_boundaries = set()

        def request_boundary(self):
            """Seal only the current source prefix, without forcing all of it."""
            if self._closed or not self._pending:
                return False
            self._soft_boundaries.add(self._pending[-1].unit_id)
            self._ensure_worker()
            return True

        def resume_boundary(self, unit_id, *, allow_after_hold=False):
            """Reopen exactly one pending VAD tail after acoustic continuity checks.

            The caller must establish guarded source continuity. Hard drains,
            expired buffers and markers inside a later turn are never reopened.
            In-flight requests retain their original immutable snapshots.
            """
            limit = (self.config.max_total_age_s
                     if allow_after_hold and getattr(self, "_myvote_semantic_quality_policy_v1", False)
                     else self.config.max_hold_s)
            if (self._closed or self._closing or not self._pending
                    or self._pending[-1].unit_id != unit_id
                    or unit_id not in self._soft_boundaries
                    or unit_id in self._boundaries
                    or time.monotonic() >= self._pending[0].created_at_monotonic + limit):
                return False
            self._soft_boundaries.remove(unit_id)
            self._ensure_worker()
            return True

        def _head(self):
            units, forced = super()._head()
            for index, unit in enumerate(units):
                if unit.unit_id in self._soft_boundaries:
                    # A later hard boundary must not force this earlier natural
                    # prefix. The identical boundary can explicitly be hardened.
                    return units[:index + 1], unit.unit_id in self._boundaries
            return units, forced

        def _remove_prefix(self, count):
            removed = super()._remove_prefix(count)
            self._soft_boundaries.difference_update(unit.unit_id for unit in removed)
            return removed

        def request_flush(self):
            if self._closed:
                return False
            # Preserve turn separation during a hard session drain. Otherwise a
            # soft head could remain a wait despite a hard boundary further on.
            self._boundaries.update(self._soft_boundaries)
            return super().request_flush()

        async def reset(self):
            try:
                return await super().reset()
            finally:
                self._soft_boundaries.clear()

    class NaturalBoundaryRouter(router_type):
        _myvote_soft_lane_boundaries_v1 = True
        _myvote_boundary_original = router_type

        def _context_for(self, scope):
            # A delayed commit from before a gap must not become context for
            # the next generation. Generic legacy lanes remain mutually usable,
            # but never mix with an explicitly generation-tagged lane.
            generation = _lane_generation(scope)
            return tuple(text for previous, text in self._dialogue
                         if previous[:2] == scope[:2]
                         and _lane_generation(previous) == generation)[-2:]

        async def append(self, unit, *, boundary=False):
            # Upstream admission/capacity/source-scope safeguards still run.
            # Only its boundary=True global flush is deliberately bypassed.
            result = await super().append(unit, boundary=False)
            if boundary:
                self.request_boundary(unit.scope)
            return result

        def request_boundary(self, scope):
            if self._closed:
                return False
            coordinator = self._lanes.get(scope)
            return coordinator.request_boundary() if coordinator is not None else False

        def resume_boundary(self, scope, unit_id, *, allow_after_hold=False):
            if self._closed or self._closing:
                return False
            coordinator = self._lanes.get(scope)
            return coordinator.resume_boundary(unit_id, allow_after_hold=allow_after_hold) if coordinator is not None else False

        def request_flush(self, scope=None):
            if scope is None:
                return super().request_flush()
            if self._closed:
                return False
            coordinator = self._lanes.get(scope)
            return coordinator.request_flush() if coordinator is not None else False

    routing_module.SemanticTranslationCoordinator = NaturalBoundaryCoordinator
    routing_module.SemanticTranslationRouter = NaturalBoundaryRouter
    return metadata
