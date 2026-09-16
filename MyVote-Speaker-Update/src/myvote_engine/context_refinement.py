"""Bounded review of completed caption results using later, immutable source text.

This is a secondary provider job, not a transcript summary or a second source
revision. Admission prefers waiting live translations; an in-flight backend
request cannot be physically preempted by that preference.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
import time

from .priority_translation import AdmittedTranslationProvider
from .translation import ContextClause, ContextReviewRequest, ContextReviewTarget


@dataclass(frozen=True)
class ContextRefinementConfig:
    max_clauses: int = 64
    max_source_chars: int = 16000
    max_targets: int = 2
    max_revisions_per_target: int = 2
    max_source_age_s: float = 120
    first_chunk_timeout_s: float = 2
    request_timeout_s: float = 4
    min_request_interval_s: float = 2
    max_response_chars: int = 32768
    max_correction_chars: int = 24000
    close_timeout_s: float = .1

    def __post_init__(self):
        for name, limit in (("max_clauses", 64), ("max_source_chars", 16000),
                            ("max_targets", 2), ("max_revisions_per_target", 2),
                            ("max_response_chars", 65536), ("max_correction_chars", 24000)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{name} must be an integer in [1, {limit}]")
        for name in ("max_source_age_s", "first_chunk_timeout_s", "request_timeout_s",
                     "min_request_interval_s", "close_timeout_s"):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or (name not in ("min_request_interval_s", "close_timeout_s") and value == 0)):
                raise ValueError(f"Invalid {name}")
        if self.first_chunk_timeout_s > self.request_timeout_s or self.request_timeout_s > 120:
            raise ValueError("First chunk timeout must not exceed total timeout, at most 120 seconds")


@dataclass(frozen=True)
class ContextCorrection:
    segment_id: str
    source_revision: int
    result_revision: int
    text: str


@dataclass(frozen=True)
class _Source:
    clause: ContextClause
    start_ns: int
    end_ns: int
    # Source geometry and lineage must not change while a request is in flight.
    identity: tuple


@dataclass(frozen=True)
class ContextRefinementSnapshot:
    request_id: str
    run_token: int
    context_version: int
    provider_generation: int
    provider: object = field(repr=False, compare=False)
    targets: tuple[ContextReviewTarget, ...]
    sources: tuple[_Source, ...]


class _InvalidResponse(ValueError):
    pass


class _StaleRequest(Exception):
    pass


def _identity(caption):
    return (caption.track_id, caption.start_ns, caption.end_ns, caption.source_track_id,
            caption.capture_epoch, caption.separation_group_id, caption.lane_id,
            caption.parent_segment_ids, caption.delivery_class, caption.text_operation,
            caption.source_language)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidResponse("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise _InvalidResponse("Nonfinite JSON constant")


def parse_context_corrections(text: str, targets: tuple[ContextReviewTarget, ...], *,
                              max_text_chars: int = 24000) -> tuple[ContextCorrection, ...]:
    """Validate the entire packet before returning any authorized correction."""
    try:
        packet = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise _InvalidResponse("Invalid context review JSON") from exc
    if type(packet) is not dict or set(packet) != {"corrections"} or type(packet["corrections"]) is not list:
        raise _InvalidResponse("Expected only a corrections array")
    allowed = {target.segment_id: target for target in targets}
    rows = packet["corrections"]
    if len(rows) > len(allowed):
        raise _InvalidResponse("Too many corrections")
    corrections, seen = [], set()
    for row in rows:
        if type(row) is not dict or set(row) != {"segment_id", "source_revision", "result_revision", "text"}:
            raise _InvalidResponse("Unexpected correction fields")
        key = row["segment_id"]
        if not isinstance(key, str) or key not in allowed or key in seen:
            raise _InvalidResponse("Unknown or repeated correction target")
        seen.add(key)
        target = allowed[key]
        if (type(row["source_revision"]) is not int or type(row["result_revision"]) is not int
                or row["source_revision"] != target.source_revision
                or row["result_revision"] != target.result_revision):
            raise _InvalidResponse("Correction revision does not match request")
        value = row["text"]
        if not isinstance(value, str) or not value.strip() or len(value) > max_text_chars:
            raise _InvalidResponse("Invalid correction text")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise _InvalidResponse("Invalid Unicode correction") from exc
        if value != target.current_text:
            corrections.append(ContextCorrection(key, target.source_revision, target.result_revision, value))
    return tuple(corrections)


class _CurrentProvider:
    """Recheck after secondary admission, before sending anything to the backend."""
    def __init__(self, owner, snapshot):
        self.owner, self.snapshot = owner, snapshot

    async def stream(self, request):
        if not self.owner.is_current(self.snapshot) or time.monotonic() >= request.deadline_monotonic:
            raise _StaleRequest()
        stream = self.snapshot.provider.stream(request)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()


class ContextRefinementCoordinator:
    def __init__(self, session, admission, *, config: ContextRefinementConfig | None = None):
        self.session, self.admission = session, admission
        self.config = config or ContextRefinementConfig()
        self._history: OrderedDict[str, _Source] = OrderedDict()
        self._revisions: dict[str, int] = {}
        self._last_reviewed_context: dict[str, int] = {}
        self._scope = None
        self._run_token = 0
        self._context_version = 0
        self._attempted_version = -1
        self._pending: int | None = None
        self._task: asyncio.Task | None = None
        self._request_task: asyncio.Task | None = None
        self._closed = False
        self._last_started = -math.inf
        self._request_sequence = 0
        self.counts = {f"context_refinement_{name}": 0 for name in (
            "requests", "applied_batches", "applied_corrections", "unchanged", "invalid",
            "stale", "timeouts", "provider_errors", "cancelled", "pending_replaced", "evicted",
        )}

    @property
    def history(self) -> tuple[ContextClause, ...]:
        return tuple(item.clause for item in self._history.values())

    @property
    def pending_count(self) -> int:
        return int(self._pending is not None)

    @property
    def active(self) -> bool:
        return self._request_task is not None and not self._request_task.done()

    def _count(self, name, amount=1):
        self.counts[f"context_refinement_{name}"] += amount

    def _provider(self):
        runner = self.session._translator
        provider = runner.provider
        while isinstance(provider, AdmittedTranslationProvider):
            provider = provider.provider
        return provider, runner.generation

    def _visible(self, caption):
        held = getattr(self.session.store, "held_group_for_segment", None)
        return (caption is not None and caption.source_state == "stable" and not caption.superseded_by
                and not (held is not None and held(caption.segment_id) is not None))

    def _source_current(self, source):
        caption = self.session.store.get_segment(source.clause.segment_id)
        return (self._visible(caption) and caption.source_revision == source.clause.source_revision
                and caption.source_text == source.clause.source_text and _identity(caption) == source.identity)

    def register_caption(self, caption, source_language=None):
        if self._closed or not self._visible(caption):
            return
        scope = (caption.source_track_id or caption.track_id, caption.capture_epoch)
        if self._scope is not None and scope != self._scope:
            self.gap()
        self._scope = scope
        try:
            clause = ContextClause(caption.segment_id, caption.source_revision, caption.source_text,
                                   source_language or caption.source_language or "auto")
        except ValueError:
            return
        source = _Source(clause, caption.start_ns, caption.end_ns, _identity(caption))
        previous = self._history.get(caption.segment_id)
        if previous == source:
            return
        self._history[caption.segment_id] = source
        # Stable source ordering, including a late separated child or revision.
        self._history = OrderedDict(sorted(self._history.items(), key=lambda row: (row[1].end_ns, row[1].start_ns)))
        while (len(self._history) > self.config.max_clauses
               or sum(len(item.clause.source_text) for item in self._history.values()) > self.config.max_source_chars):
            key, _ = self._history.popitem(last=False)
            self._revisions.pop(key, None)
            self._last_reviewed_context.pop(key, None)
            self._count("evicted")
        self._context_version += 1
        self._schedule()

    def on_translation_completed(self, segment_id):
        if segment_id in self._history:
            self._schedule()

    def _schedule(self):
        if self._closed or self._context_version == self._attempted_version or len(self._history) < 2:
            return
        if self._pending is not None and self._pending != self._context_version:
            self._count("pending_replaced")
        self._pending = self._context_version
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="caption-context-review")

    def _snapshot(self):
        sources = tuple(source for source in self._history.values() if self._source_current(source))
        if len(sources) < 2:
            return None
        newest_end = max(source.end_ns for source in sources)
        provider, generation = self._provider()
        targets = []
        # Unchanged older captions must not monopolize every later context.
        # Review the least recently considered candidates first, with source
        # chronology as the stable tie breaker. This is separate from edit caps.
        review_order = sorted(sources, key=lambda source: (
            self._last_reviewed_context.get(source.clause.segment_id, -1), source.end_ns, source.start_ns))
        for source in review_order:
            key = source.clause.segment_id
            if (source.end_ns >= newest_end
                    or newest_end - source.end_ns > self.config.max_source_age_s * 1e9
                    or self._revisions.get(key, 0) >= self.config.max_revisions_per_target):
                continue
            caption = self.session.store.get_segment(key)
            result = caption.current_translation
            if caption.speaker_manual or result is None or result.provider_generation != generation:
                continue
            try:
                targets.append(ContextReviewTarget(key, caption.source_revision, result.result_revision,
                                                   caption.source_text, result.text, source.clause.source_language,
                                                   caption.target_language))
            except ValueError:
                continue
            if len(targets) == self.config.max_targets:
                break
        if not targets:
            return None
        self._request_sequence += 1
        return ContextRefinementSnapshot(f"context-review-{self._run_token}-{self._request_sequence}",
                                         self._run_token, self._context_version, generation, provider,
                                         tuple(targets), sources)

    def is_current(self, snapshot: ContextRefinementSnapshot) -> bool:
        if self._closed or getattr(self.session, "_closed", False) or snapshot.run_token != self._run_token:
            return False
        provider, generation = self._provider()
        if provider is not snapshot.provider or generation != snapshot.provider_generation:
            return False
        if any(not self._source_current(source) for source in snapshot.sources):
            return False
        for target in snapshot.targets:
            caption = self.session.store.get_segment(target.segment_id)
            if not self._visible(caption) or caption.speaker_manual:
                return False
            result = caption.current_translation
            if (caption.source_revision != target.source_revision or caption.source_text != target.source_text
                    or caption.target_language != target.target_language or result is None
                    or result.provider_generation != snapshot.provider_generation
                    or result.result_revision != target.result_revision or result.text != target.current_text):
                return False
        return True

    async def _collect(self, snapshot, request, first_chunk):
        provider = AdmittedTranslationProvider(_CurrentProvider(self, snapshot), self.admission, secondary=True)
        stream = provider.stream(request)
        parts, size = [], 0
        try:
            async for chunk in stream:
                if not isinstance(chunk.text, str):
                    raise _InvalidResponse("Invalid provider text")
                if chunk.text:
                    first_chunk.set()
                    size += len(chunk.text)
                    if size > self.config.max_response_chars:
                        raise _InvalidResponse("Context response is too large")
                    parts.append(chunk.text)
                if chunk.completed:
                    if chunk.finish_reason != "stop":
                        raise _InvalidResponse("Review did not finish normally")
                    return parse_context_corrections("".join(parts), snapshot.targets,
                                                     max_text_chars=self.config.max_correction_chars)
            raise _InvalidResponse("Review ended without completion")
        finally:
            await stream.aclose()

    async def _review(self, snapshot):
        started = time.monotonic()
        self._last_started = started
        deadline = started + self.config.request_timeout_s
        request = ContextReviewRequest(snapshot.request_id, snapshot.targets,
                                       tuple(source.clause for source in snapshot.sources),
                                       budget_ms=self.config.request_timeout_s * 1000,
                                       deadline_monotonic=deadline)
        first_chunk = asyncio.Event()
        task = self._request_task = asyncio.create_task(self._collect(snapshot, request, first_chunk))
        first = asyncio.create_task(first_chunk.wait())
        self._count("requests")
        try:
            done, _ = await asyncio.wait((task, first), timeout=max(0, started + self.config.first_chunk_timeout_s - time.monotonic()),
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise TimeoutError()
            if not task.done():
                done, _ = await asyncio.wait((task,), timeout=max(0, deadline - time.monotonic()))
                if not done:
                    raise TimeoutError()
            corrections = task.result()
            if time.monotonic() >= deadline:
                raise TimeoutError()
            if not self.is_current(snapshot):
                self._count("stale")
            elif not corrections:
                self._count("unchanged")
            elif await self.session.apply_context_corrections(corrections, snapshot) is True:
                self._count("applied_batches")
                self._count("applied_corrections", len(corrections))
                for item in corrections:
                    if item.segment_id in self._history:
                        self._revisions[item.segment_id] = self._revisions.get(item.segment_id, 0) + 1
            else:
                self._count("stale")
        except TimeoutError:
            self._count("timeouts")
        except _StaleRequest:
            self._count("stale")
        except _InvalidResponse:
            self._count("invalid")
        except asyncio.CancelledError:
            self._count("cancelled")
        except Exception:
            self._count("provider_errors")
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            if not task.done():
                task.cancel()
            # Keep this one active owner until generator cleanup actually exits.
            # drain/close use asyncio.wait, so a noncooperative backend cannot
            # make those public calls wait indefinitely or admit a second job.
            await asyncio.gather(task, return_exceptions=True)
            self._request_task = None

    async def _loop(self):
        try:
            while not self._closed and self._pending is not None:
                wait = self._last_started + self.config.min_request_interval_s - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                if self._closed or self._pending is None:
                    break
                self._pending = None
                snapshot = self._snapshot()
                if snapshot is None:
                    continue
                self._attempted_version = snapshot.context_version
                for target in snapshot.targets:
                    self._last_reviewed_context[target.segment_id] = snapshot.context_version
                await self._review(snapshot)
        except asyncio.CancelledError:
            pass

    def gap(self):
        self._run_token += 1
        self._context_version += 1
        self._pending = None
        self._history.clear()
        self._revisions.clear()
        self._last_reviewed_context.clear()
        self._scope = None
        if self._request_task is not None and not self._request_task.done():
            self._request_task.cancel()

    async def drain(self, timeout_s: float) -> bool:
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("Drain timeout must be finite and nonnegative")
        task = self._task
        if task is None or task.done():
            return True
        done, _ = await asyncio.wait((task,), timeout=timeout_s)
        return bool(done)

    async def close(self):
        self._closed = True
        self.gap()
        # A cadence sleep has no provider ownership and can end immediately.
        if self._task is not None and not self._task.done() and self._request_task is None:
            self._task.cancel()
        await self.drain(self.config.close_timeout_s)
