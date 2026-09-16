"""Per-session streaming state machine; only immutable, semantically ready source is scheduled.

Revision rollback uses source segment boundaries, never guessed source/target
character alignment. Independent sessions share workers; a session assembles its
next context only after the previous result, preserving causal context order.
"""
import asyncio
from collections import deque
from dataclasses import dataclass, field
import logging
import time
from app.context.glossary import matching_entries
from app.context.manager import ContextEntry, ContextManager
from app.core.config import Settings
from app.core.models import ASREvent, GlossaryUpdate, SessionConfig
from app.metrics.collector import Metrics
from app.streaming.commit import DraftCommitter
from app.streaming.policy import ReadWritePolicy
from app.streaming.revision import RevisionSegment, plan_revision
from app.streaming.segmentation import SemanticSegmenter
from app.streaming.stability import StabilityTracker
from app.translation.backend import TranslationResult
from app.translation.scheduler import QueueCapacityError, TranslationJob
from app.utils.text import tokenize

logger = logging.getLogger(__name__)
BACKEND_ERRORS = {
    "timeout": "TRANSLATION_TIMEOUT", "transport_error": "BACKEND_UNAVAILABLE",
    "upstream_http_error": "BACKEND_UNAVAILABLE", "rate_limited": "BACKEND_RATE_LIMITED",
    "invalid_response": "BACKEND_INVALID_RESPONSE", "incomplete_response": "BACKEND_INCOMPLETE_RESPONSE",
    "response_too_large": "BACKEND_RESPONSE_LIMIT", "output_too_large": "BACKEND_RESPONSE_LIMIT",
    "input_too_large": "SOURCE_LIMIT",
}


@dataclass
class Utterance:
    utterance_id: str
    tracker: StabilityTracker
    speaker_id: str | None
    text: str = ""
    stable_end: int = 0
    selected_end: int = 0
    revision: int = 0
    sequence: int = 0
    final: bool = False
    pause_ms: float | None = None
    created_at: float = field(default_factory=time.monotonic)
    changed_at: float = field(default_factory=time.monotonic)
    buffer_since: float = field(default_factory=time.monotonic)
    incomplete_reported: bool = False


@dataclass
class Segment:
    segment_id: str
    utterance_id: str
    start: int
    end: int
    source: str
    source_revision: int
    speaker_id: str | None
    sequence: int
    final_source: bool
    committer: DraftCommitter
    created_at: float = field(default_factory=time.monotonic)
    translation: str = ""
    committed: bool = False
    status: str = "ready"


class TranslationSession:
    def __init__(self, session_id: str, config: SessionConfig, settings: Settings, scheduler, metrics: Metrics):
        self.session_id, self.settings, self.scheduler, self.metrics = session_id, settings, scheduler, metrics
        self.lock = asyncio.Lock()
        self.token_hash = b""
        self.closed = False
        self.last_sequence = -1
        self.generation_id = 0
        self.revision = 0
        self.last_activity = time.monotonic()
        self.created_at = self.last_activity
        self.first_translation_at: float | None = None
        self.first_asr_at: float | None = None
        self.utterances: dict[str, Utterance] = {}
        self.segments: list[Segment] = []
        self.glossary: dict[str, str] = {}
        self._auto_id: str | None = None
        self._next_utterance = 0
        self._next_segment = 0
        self._active_segment: str | None = None
        self._timers: dict[str, asyncio.Task] = {}
        self._queue: asyncio.Queue | None = None
        self._events: deque[float] = deque()
        self._set_config(config)

    def _set_config(self, config: SessionConfig) -> None:
        self.segmenter = SemanticSegmenter()
        if not self.segmenter.supports(config.source_language):
            raise ValueError("UNSUPPORTED_SOURCE_LANGUAGE")
        if config.translation.model not in (None, self.settings.translation_model):
            raise ValueError("MODEL_NOT_ALLOWED")
        self.config = config
        self.threshold = (config.streaming.stability_threshold if config.streaming.stability_threshold is not None
                          else self.settings.stability_threshold)
        self.policy = ReadWritePolicy(
            stability_threshold=self.threshold, min_stable_tokens=self.settings.min_stable_tokens,
            min_segment_tokens=self.settings.min_segment_tokens, target_segment_tokens=self.settings.target_segment_tokens,
            max_latency_ms=config.streaming.max_latency_ms or self.settings.max_segment_latency_ms,
            pause_threshold_ms=self.settings.pause_threshold_ms)
        count = config.translation.max_context_segments
        self.context = ContextManager(self.settings.context_segments if count is None else min(count, self.settings.context_segments),
                                      self.settings.max_context_tokens)

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def state_bytes(self) -> int:
        return sum(len(u.text.encode("utf-8")) for u in self.utterances.values()) + sum(
            len(s.translation.encode("utf-8")) for s in self.segments)

    def admit_event(self) -> bool:
        now = time.monotonic()
        while self._events and self._events[0] < now - 1:
            self._events.popleft()
        if len(self._events) >= self.settings.max_events_per_second:
            self.metrics.increment("rate_limited_events")
            return False
        self._events.append(now)
        return True

    def emit(self, event: dict) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait({**event, "_enqueued_at": time.monotonic()})
        except asyncio.QueueFull:
            # Never silently drop a committed result. Disconnect a slow consumer;
            # retained state is recovered by authenticated GET/new connection.
            self.metrics.increment("slow_client_disconnects")
            while not self._queue.empty():
                self._queue.get_nowait()
            self._queue.put_nowait(None)

    def error(self, code: str, message: str, *, recoverable: bool = True) -> None:
        self.emit({"type": "error", "code": code, "message": message,
                   "recoverable": recoverable, "last_sequence": self.last_sequence})

    def snapshot(self, kind: str = "session_state") -> dict:
        visible = [segment for segment in self.segments if segment.translation]
        split = next((i for i, segment in enumerate(visible) if not segment.committed), len(visible))
        committed = " ".join(segment.translation for segment in visible[:split])
        draft = " ".join(segment.translation for segment in visible[split:])
        return {
            "type": kind, "session_id": self.session_id, "protocol_version": "1",
            "generation_id": self.generation_id, "revision": self.revision, "last_sequence": self.last_sequence,
            "source_language": self.config.source_language, "target_language": self.config.target_language,
            "committed_source": " ".join(u.text[:u.stable_end].strip() for u in self.utterances.values()).strip(),
            "unstable_source": " ".join(u.text[u.stable_end:].strip() for u in self.utterances.values()).strip(),
            "committed": committed, "draft": draft, "full_text": " ".join(s.translation for s in visible),
            "final": bool(self.utterances) and all(u.final and not u.text[u.selected_end:].strip() for u in self.utterances.values())
                     and bool(self.segments) and all(s.committed for s in self.segments),
            "segments": [{"segment_id": s.segment_id, "utterance_id": s.utterance_id, "source": s.source,
                          "translation": s.translation, "committed": s.committed, "source_revision": s.source_revision,
                          "speaker_id": s.speaker_id, "status": s.status} for s in self.segments],
        }

    async def attach(self) -> asyncio.Queue:
        async with self.lock:
            if self._queue is not None or self.closed:
                raise ValueError("SESSION_ALREADY_CONNECTED")
            self._queue = asyncio.Queue(self.settings.outgoing_queue_size)
            self.touch()
            self.emit(self.snapshot())
            for utterance in self.utterances.values():
                self._arm(utterance.utterance_id, 0)
            for segment in self.segments:
                if segment.translation and not segment.committed:
                    if segment.committer.ready():
                        self._commit(segment)
                    else:
                        self._arm_commit(segment)
            await self._dispatch_next()
            return self._queue

    async def detach(self) -> None:
        async with self.lock:
            self._queue = None
            # A callback may already be waiting on this lock when cancellation
            # is requested. Invalidate the attempt, not the retained subtitles.
            self.generation_id += 1
            self._cancel_timers()
            await self.scheduler.cancel_session(self.session_id)
            for segment in self.segments:
                if segment.status == "running":
                    segment.status = "ready"
            self._active_segment = None

    async def configure(self, config: SessionConfig) -> None:
        async with self.lock:
            if self.utterances:
                raise ValueError("CONFIG_REQUIRES_RESET")
            self._set_config(config)
            self.touch()
            self.emit(self.snapshot())

    async def set_glossary(self, entries: dict[str, str]) -> None:
        async with self.lock:
            merged = {**self.glossary, **entries}
            try:
                GlossaryUpdate(entries=merged)
            except ValueError:
                raise ValueError("GLOSSARY_LIMIT") from None
            self.glossary = merged
            self.touch()

    async def ingest(self, event: ASREvent) -> None:
        started = time.monotonic()
        async with self.lock:
            if event.session_id not in (None, self.session_id):
                raise ValueError("SESSION_MISMATCH")
            if event.sequence <= self.last_sequence:
                raise ValueError("STALE_SEQUENCE")
            if event.language not in (None, self.config.source_language) or event.target_language not in (None, self.config.target_language):
                raise ValueError("LANGUAGE_MISMATCH")
            if len(event.text) > self.settings.max_source_chars:
                raise ValueError("SOURCE_LIMIT")
            utterance_id = event.utterance_id or self._auto_id
            if event.utterance_id is None and (utterance_id is None or self.utterances[utterance_id].final):
                self._next_utterance += 1
                utterance_id = f"auto:{self._next_utterance}"
            utterance = self.utterances.get(utterance_id)
            if utterance and utterance.speaker_id != event.speaker_id:
                raise ValueError("SPEAKER_MISMATCH")
            if utterance and utterance.final and not event.is_final:
                raise ValueError("FINAL_REQUIRES_FINAL")
            retained = sum(len(u.text) for u in self.utterances.values()) - (len(utterance.text) if utterance else 0)
            retained_bytes = self.state_bytes() - (len(utterance.text.encode("utf-8")) if utterance else 0)
            if (retained + len(event.text) > self.settings.max_session_chars
                    or retained_bytes + len(event.text.encode("utf-8")) > self.settings.max_state_bytes
                    or (utterance is None and len(self.utterances) >= self.settings.max_utterances)):
                raise ValueError("SESSION_LIMIT")
            if utterance is None:
                utterance = Utterance(utterance_id, StabilityTracker(self.settings.stability_history, self.threshold,
                                      pause_threshold_ms=self.settings.pause_threshold_ms,
                                      max_source_chars=self.settings.max_source_chars,
                                      agreement_weight=self.settings.stability_agreement_weight,
                                      confidence_weight=self.settings.stability_confidence_weight,
                                      age_weight=self.settings.stability_age_weight,
                                      pause_weight=self.settings.stability_pause_weight,
                                      hold_last_token=self.settings.hold_partial_tail), event.speaker_id)
                self.utterances[utterance_id] = utterance
            if event.utterance_id is None:
                self._auto_id = utterance_id
            self.last_sequence = event.sequence
            self.touch()
            self.metrics.increment("asr_events")
            if self.first_asr_at is None:
                self.first_asr_at = time.monotonic()
            tokens = tokenize(event.text)
            confidences = None
            if event.tokens and len(event.tokens) == len(tokens) and all(a.text.strip() == b.text for a, b in zip(event.tokens, tokens)):
                confidences = [token.confidence for token in event.tokens]
            stability = utterance.tracker.observe(event.text, event.sequence, is_final=event.is_final,
                                                  confidences=confidences, pause_ms=event.pause_ms)
            if stability.prefix_conflict and not event.is_final:
                self.error("SOURCE_RECONCILIATION_PENDING", "A committed prefix changed; waiting for final ASR confirmation")
                return
            old_segments = [s for s in self.segments if s.utterance_id == utterance_id]
            plan = plan_revision(utterance.text, event.text, [RevisionSegment(s.start, s.end) for s in old_segments])
            if plan.rollback_segment_index is not None and plan.rollback_segment_index < len(old_segments):
                first = old_segments[plan.rollback_segment_index]
                await self._rollback(self.segments.index(first))
            if event.text != utterance.text:
                utterance.revision += 1
                utterance.changed_at = time.monotonic()
                utterance.incomplete_reported = False
            utterance.text, utterance.final = event.text, event.is_final
            utterance.sequence, utterance.pause_ms = event.sequence, event.pause_ms
            utterance.stable_end = stability.stable_char_end
            utterance.tracker.commit(utterance.stable_end, allow_final_reconcile=event.is_final)
            self.revision += 1
            self.emit(self.snapshot())
            self._arm(utterance_id, 0 if event.is_final else self.settings.translation_debounce_ms / 1000)
            # A final identical hypothesis commits existing complete draft segments
            # without forcing a second model request.
            for segment in old_segments:
                if segment in self.segments and event.is_final and segment.translation and not segment.committed and segment.committer.ready():
                    self._commit(segment)
                if segment in self.segments and event.is_final and segment.status == "failed":
                    segment.status = "ready"
                    self.metrics.increment("retried_jobs")
            self.metrics.observe("asr_processing_ms", (time.monotonic() - started) * 1000)

    async def _rollback(self, index: int) -> None:
        affected = self.segments[index:]
        identifiers = {s.segment_id for s in affected}
        # Invalidate state before awaiting cancellation: a backend may ignore it.
        self.segments[index:] = []
        for segment in affected:
            u = self.utterances[segment.utterance_id]
            u.selected_end = min(u.selected_end, segment.start)
            u.incomplete_reported = False
            self._arm(u.utterance_id, 0)
        if self._active_segment in identifiers:
            self._active_segment = None
        await self.scheduler.invalidate(self.session_id, identifiers)
        self.metrics.increment("source_revisions")
        self.revision += 1
        self.emit(self.snapshot("translation_update"))

    def _arm(self, utterance_id: str, delay: float) -> None:
        if self.closed or self._queue is None:
            return
        old = self._timers.pop(utterance_id, None)
        if old is not None and old is not asyncio.current_task():
            old.cancel()
        generation = self.generation_id

        async def run():
            try:
                await asyncio.sleep(delay)
                await self._evaluate(utterance_id, generation)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Unexpected policy bugs are visible, but do not leak raw input.
                self.metrics.increment("pipeline_errors")
                logger.error("Streaming policy task failed", exc_info=False)
                self.error("PIPELINE_ERROR", "Streaming policy failed; reset or retry the session")
            finally:
                if self._timers.get(utterance_id) is asyncio.current_task():
                    self._timers.pop(utterance_id, None)

        self._timers[utterance_id] = asyncio.create_task(run(), name="semantic-decision")

    async def _evaluate(self, utterance_id: str, generation: int) -> None:
        async with self.lock:
            if self.closed or generation != self.generation_id or utterance_id not in self.utterances:
                return
            started = time.monotonic()
            # A final from a later utterance must not race ahead of an earlier
            # debounce/reconciliation. Visit source order, not timer wake order.
            for u in self.utterances.values():
                if not u.text[u.selected_end:].strip() or u.incomplete_reported:
                    continue
                if not await self._select_utterance(u):
                    break
            self.metrics.observe("segment_decision_ms", (time.monotonic() - started) * 1000)
            await self._dispatch_next()

    async def _select_utterance(self, u: Utterance) -> bool:
        stability = u.tracker.refresh()
        if stability is None:
            return False
        if not stability.prefix_conflict:
            u.stable_end = stability.stable_char_end
            u.tracker.commit(u.stable_end, allow_final_reconcile=u.final)
        while u.text[u.selected_end:].strip():
            if len(self.segments) >= self.settings.max_segments:
                self.error("SESSION_LIMIT", "Segment limit reached; start a new session")
                return False
            waiting = sum(s.status in ("ready", "running") for s in self.segments)
            if waiting >= self.settings.max_pending_per_session:
                self.metrics.increment("backpressure_events")
                self._arm(u.utterance_id, 0.2)
                return False
            count = self.settings.context_segments
            context = "\n".join(s.source for s in self.segments[-count:]) if count else ""
            candidate = self.segmenter.select(
                u.text, u.stable_end, start=u.selected_end, context=context,
                language=self.config.source_language, max_tokens=self.settings.max_segment_tokens,
                allow_oversize_complete=self.settings.allow_oversize_complete)
            age_ms = (time.monotonic() - u.buffer_since) * 1000
            decision = await self.policy.decide(
                candidate, stability_score=stability.score, stable_token_count=stability.stable_token_count,
                age_ms=age_ms, pause_ms=u.pause_ms, is_final=u.final, prefix_conflict=stability.prefix_conflict)
            if decision.action != "WRITE" or decision.segment is None:
                if u.final or age_ms >= self.settings.max_buffer_age_ms:
                    if not u.incomplete_reported:
                        self.error("INCOMPLETE_SOURCE", "Unfinished source retained; no forced fragment translation")
                        self.metrics.increment("incomplete_source")
                        u.incomplete_reported = True
                else:
                    self._arm(u.utterance_id, min(0.5, self.settings.max_segment_latency_ms / 1000))
                return u.incomplete_reported
            chosen = decision.segment
            self._next_segment += 1
            segment = Segment(
                f"s{self._next_segment}", u.utterance_id, chosen.start, chosen.end,
                u.text[chosen.start:chosen.end], u.revision, u.speaker_id, u.sequence, u.final,
                DraftCommitter(self.settings.translation_stability_count, self.settings.translation_commit_delay_ms))
            self.segments.append(segment)
            u.selected_end = chosen.end
            u.buffer_since = time.monotonic()
            self.metrics.observe("arrival_to_segmentation_ms", (segment.created_at - u.changed_at) * 1000)
        return True

    async def _dispatch_next(self) -> None:
        if self.closed or self._queue is None or self._active_segment is not None:
            return
        segment = next((s for s in self.segments if s.status == "ready"), None)
        if segment is None:
            return
        before = self.segments[:self.segments.index(segment)]
        self.context.replace([ContextEntry(s.segment_id, s.source, s.translation, s.speaker_id)
                              for s in before if s.translation and s.status == "complete"])
        source_context, translation_context = self.context.snapshot()
        enqueued_at = time.monotonic()
        job = TranslationJob(session_id=self.session_id, generation_id=self.generation_id, sequence=segment.sequence,
                             source_revision=segment.source_revision, segment_id=segment.segment_id, source=segment.source,
                             source_language=self.config.source_language, target_language=self.config.target_language,
                             source_context=tuple(source_context), translation_context=tuple(translation_context),
                             glossary=matching_entries(segment.source, self.glossary), final=segment.final_source,
                             created_at=enqueued_at, model=self.settings.translation_model,
                             temperature=self.config.translation.temperature if self.config.translation.temperature is not None else self.settings.translation_temperature)
        try:
            await self.scheduler.submit(job, self._result)
        except QueueCapacityError:
            self.error("QUEUE_OVERFLOW", "Translation queue is full; source retained for retry")
            self._arm(segment.utterance_id, 0.2)
            return
        segment.status = "running"
        self._active_segment = segment.segment_id
        self.metrics.observe("segmentation_to_queue_ms", (enqueued_at - segment.created_at) * 1000)

    async def _result(self, job, result: TranslationResult | None, error: Exception | None) -> None:
        async with self.lock:
            segment = next((s for s in self.segments if s.segment_id == job.segment_id), None)
            if (self.closed or job.generation_id != self.generation_id or segment is None
                    or segment.source_revision != job.source_revision or segment.status != "running"
                    or self._active_segment != segment.segment_id):
                self.metrics.increment("discarded_stale_results")
                return
            if self._active_segment == segment.segment_id:
                self._active_segment = None
            if error is not None or result is None:
                segment.status = "failed"
                code = BACKEND_ERRORS.get(getattr(error, "code", None), "TRANSLATION_ERROR")
                self.error(code, "Translation failed; source is retained and the session remains usable")
            elif (sum(len(s.translation) for s in self.segments) + len(result.text) > self.settings.max_translation_chars
                  or self.state_bytes() + len(result.text.encode("utf-8")) > self.settings.max_state_bytes):
                segment.status = "failed"
                self.error("SESSION_LIMIT", "Translation history limit reached; start a new session")
            else:
                segment.translation, segment.status = result.text, "complete"
                segment.committer.propose(segment.start, segment.end, segment.source, result.text,
                                           source_revision=segment.source_revision, is_final=segment.final_source)
                self.revision += 1
                self.emit(self.snapshot("translation_update"))
                now = time.monotonic()
                if self.first_translation_at is None:
                    self.first_translation_at = now
                    self.metrics.observe("first_translation_ms", (now - (self.first_asr_at or self.created_at)) * 1000)
                self.metrics.observe("translation_end_to_end_ms", (now - segment.created_at) * 1000)
                if self.settings.raw_transcript_logging:
                    logger.info("translation_debug %s", {"source": segment.source, "translation": segment.translation,
                                                       "source_revision": segment.source_revision})
                if segment.committer.ready():
                    self._commit(segment)
                else:
                    self._arm_commit(segment)
            # Resume retained final source after bounded queue/backpressure clears.
            for u in self.utterances.values():
                if u.text[u.selected_end:].strip() and not u.incomplete_reported:
                    self._arm(u.utterance_id, 0)
            await self._dispatch_next()

    def _commit(self, segment: Segment) -> None:
        if segment.committed:
            return
        segment.committer.mark_committed()
        segment.committed = True
        self.revision += 1
        self.metrics.observe("finalization_ms", (time.monotonic() - segment.created_at) * 1000)
        self.emit(self.snapshot("translation_final"))

    def _arm_commit(self, segment: Segment) -> None:
        key = f"commit:{segment.segment_id}"
        generation = self.generation_id

        async def commit():
            try:
                await asyncio.sleep(self.settings.translation_commit_delay_ms / 1000)
                async with self.lock:
                    if not self.closed and generation == self.generation_id and segment in self.segments and segment.committer.ready():
                        self._commit(segment)
            finally:
                if self._timers.get(key) is asyncio.current_task():
                    self._timers.pop(key, None)

        self._timers[key] = asyncio.create_task(commit(), name="translation-commit")

    def _cancel_timers(self) -> None:
        for task in self._timers.values():
            task.cancel()
        self._timers.clear()

    async def reset(self) -> None:
        async with self.lock:
            self.generation_id += 1
            self.revision += 1
            self._cancel_timers()
            self.utterances.clear()
            self.segments.clear()
            self.context.replace([])
            self._auto_id = self._active_segment = None
            self.first_translation_at = self.first_asr_at = None
            await self.scheduler.cancel_session(self.session_id)
            self.touch()
            self.emit(self.snapshot())

    async def close(self) -> None:
        async with self.lock:
            self.closed = True
            self.generation_id += 1
            self._cancel_timers()
            await self.scheduler.cancel_session(self.session_id)
            if self._queue is not None:
                while not self._queue.empty():
                    self._queue.get_nowait()
                self._queue.put_nowait(None)
