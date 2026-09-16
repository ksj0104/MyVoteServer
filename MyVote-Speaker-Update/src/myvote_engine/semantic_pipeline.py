"""Connect immutable ASR words, ephemeral source previews and semantic captions."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import time

from .captions import CaptionSegment
from .semantic_translation import SemanticPendingUnit, SemanticTranslationConfig
from .semantic_routing import SemanticTranslationRouter
from .translation import TranslationEvent, text_operation_for


@dataclass
class _Preview:
    window: object
    update: object
    language: str
    submitted: int = 0
    retired: set[int] = field(default_factory=set)
    revision: int = 0
    last_payload: dict | None = None
    retired_end_ns: int = 0


@dataclass(frozen=True)
class _Source:
    preview: _Preview
    index: int
    word: object
    commit: object
    timing: object
    window: object
    update_revision: int
    update_reason: str


class SemanticCaptionPipeline:
    def __init__(self, session, provider):
        self.session = session
        self._unit_sequence = self._caption_sequence = 0
        self._last_route = None
        self._timings = {}
        base_provider = provider
        while hasattr(base_provider, "provider"):
            base_provider = base_provider.provider
        orchestrated = getattr(base_provider, "semantic_mode", None) == "orchestrated"
        self.coordinator = SemanticTranslationRouter(
            provider, self._commit, self._failure,
            target_language=session.config.target_language,
            config=SemanticTranslationConfig(
                request_timeout_s=(1.0 if orchestrated else 0.0)
                    + min(2.5, session.config.translation_budget_ms / 1000)))
        self.update_counts()

    def update_counts(self):
        self.session.counts.update({"semantic_" + key: value
                                    for key, value in self.coordinator.counts.items()})

    async def accept(self, window, state, update, language, *, final, timing):
        source_language = self.session.config.source_language
        if not source_language or source_language.lower() in ("auto", "und", "unknown"):
            source_language = language or "auto"
        # Retained word payloads do not retain a rolling window's PCM allocation.
        metadata = replace(window, samples=())
        if state.semantic_state is None:
            state.semantic_state = _Preview(metadata, update, source_language)
        preview = state.semantic_state
        preview.window, preview.update, preview.language = metadata, (
            replace(update, provisional_words=()) if final else update), source_language
        await self._preview(preview)
        fresh = update.stable_words[preview.submitted:]
        sources = tuple(_Source(preview, preview.submitted + index, word,
                        state.pending_word_timings.popleft(), timing, metadata,
                        update.revision, update.reason) for index, word in enumerate(fresh))
        preview.submitted = len(update.stable_words)
        # Reconcile the following ASR window immediately, independent of LLM speed.
        self.session._recent_words.extend(fresh)
        for index, source in enumerate(sources):
            self._unit_sequence += 1
            speaker_id = None
            if self.session._speaker_mapper is not None:
                speaker_id = self.session._speaker_mapper.confirmed_speaker_for(
                    track_id=window.track_id, capture_epoch=window.capture_epoch,
                    start_ns=source.word.start_time_ns, end_ns=source.word.end_time_ns)
            # Unconfirmed speech is local to this utterance, never one global
            # fictional speaker. Later acoustic evidence can still patch labels.
            lane = f"speaker:{speaker_id}" if speaker_id else f"unassigned:{window.segment_id}"
            route = (window.track_id, window.capture_epoch, source_language, lane)
            if self._last_route is not None and route != self._last_route:
                # A-B-A turns must not become one A caption spanning B's speech.
                self.coordinator.request_flush()
            self._last_route = route
            # Oversized ASR tokens are rejected by the 4000-character router
            # limit. Keep the full word in payload for the failure/source path.
            unit = SemanticPendingUnit(f"u{self._unit_sequence}", source.word.text[:12000],
                source_language, window.track_id, window.capture_epoch,
                source.commit.ready_at, source, lane_id=lane)
            await self.coordinator.append(unit, boundary=final and index == len(sources) - 1)
        if final and not sources:
            self.coordinator.request_flush()
        self.update_counts()

    async def _preview(self, preview):
        if self.session._closed:
            return
        update, window = preview.update, preview.window
        stable = tuple(word for index, word in enumerate(update.stable_words)
                       if index not in preview.retired)
        words = stable + update.provisional_words
        # Only the latest 4000 characters are a live preview; full source is
        # preserved in immutable pending units and finalized caption storage.
        source_text = "".join(word.text for word in words).strip()[-4000:]
        stable_text = "".join(word.text for word in stable).strip()[-4000:]
        start = max(window.emit_start_ns, words[0].start_time_ns) if words else preview.retired_end_ns
        end = max(start, max(word.end_time_ns for word in words)) if words else start
        payload = dict(preview_id=window.segment_id, source_text=source_text,
            stable_text=stable_text, source_language=preview.language,
            target_language=self.session.config.target_language,
            source_track_id=window.track_id, capture_epoch=window.capture_epoch,
            start_ns=start, end_ns=end,
            state="buffering" if stable else "listening" if words else "cleared")
        if payload == preview.last_payload:
            return
        preview.last_payload = payload
        preview.revision += 1
        await self.session._emit("transcript.preview", window.segment_id,
                                 revision=preview.revision, **payload)

    async def _retire_preview(self, units):
        previews = {}
        for unit in units:
            source = unit.payload
            preview = source.preview
            preview.retired.add(source.index)
            preview.retired_end_ns = max(preview.retired_end_ns, source.word.end_time_ns)
            previews[id(preview)] = preview
        for preview in previews.values():
            await self._preview(preview)

    def metrics(self, clause_id):
        timing = self._timings.get(clause_id)
        if timing is None:
            return None
        result = dict(timing)
        if self.session.ingress_timeline is not None:
            anchor = self.session._ingress_anchors.get(clause_id)
            result["server_ingress_age_ms"] = None if anchor is None else max(0, time.monotonic() - anchor) * 1000
        return result

    async def _source(self, result):
        session = self.session
        if session._closed:
            return None
        sources = tuple(unit.payload for unit in result.units)
        words = tuple(source.word for source in sources)
        first, last = sources[0], sources[-1]
        window = first.window
        text = "".join(word.text for word in words).strip()
        start_ns = max(window.emit_start_ns, words[0].start_time_ns)
        end_ns = max(word.end_time_ns for word in words)
        if end_ns <= start_ns:
            session.counts["asr_errors"] += 1
            await session._emit("asr.incomplete", window.segment_id,
                start_ns=start_ns, end_ns=end_ns, source_text=text, reason="invalid_word_timing")
            await self._retire_preview(result.units)
            return None
        self._caption_sequence += 1
        clause_id = f"{window.segment_id}:semantic-{self._caption_sequence}"
        language = result.units[0].source_language
        caption = CaptionSegment(segment_id=clause_id, track_id=window.track_id,
            start_ns=start_ns, end_ns=end_ns, source_text=text, source_revision=1,
            source_state="stable", target_language=session.config.target_language,
            source_track_id=window.track_id, capture_epoch=window.capture_epoch,
            source_language=language,
            text_operation=text_operation_for(language, session.config.target_language))
        session.store.upsert_source(caption)
        session._caption_times[clause_id] = (start_ns, end_ns)
        if session.ingress_timeline is not None:
            session._ingress_anchors[clause_id] = session.ingress_timeline.anchor(
                window.track_id, window.capture_epoch, start_ns, end_ns)
        self._timings[clause_id] = dict(
            translation_budget_scope="semantic_stable_source_clock",
            translation_budget_ms=self.coordinator.config.max_total_age_s * 1000,
            semantic_buffer_wait_ms=result.buffer_wait_ms, semantic_model_ms=result.model_ms,
            semantic_total_ms=result.total_ms)
        self._timings[clause_id].update({"semantic_" + key + "_ms": value
                                        for key, value in result.stage_metrics.items()})
        data = dict(text=text, source_revision=1, start_ns=start_ns, end_ns=end_ns,
            track_id=window.track_id, source_track_id=window.track_id,
            capture_epoch=window.capture_epoch, source_language=language,
            text_operation=caption.text_operation, semantic_request_id=result.request_id,
            semantic_force_flush=result.force_flush, **self.metrics(clause_id))
        data["source_timing"] = last.commit.caption_metrics(first_word=first.commit,
            emitting_window=last.timing, caption_end_ns=end_ns,
            source_created_at=time.monotonic(), ingress_anchor=session._ingress_anchors.get(clause_id))
        if session._word_trace is not None:
            trace = session._word_trace.build(data, words,
                stabilizer_revision=last.update_revision, commit_reason=last.update_reason,
                window_start_ns=min(source.window.window_start_ns for source in sources),
                window_end_ns=max(source.window.window_end_ns for source in sources),
                emit_start_ns=window.emit_start_ns)
            if trace is not None:
                data["word_trace"] = trace
            session.counts.update(session._word_trace.counts)
        if session._context_refinement is not None:
            session._context_refinement.register_caption(caption, language)
        session.counts["clauses"] += 1
        await session._emit("caption.source", clause_id, **data)
        if session._speaker_mapper is not None:
            await session._speaker_patches(session._speaker_mapper.register_caption(
                clause_id, track_id=window.track_id, capture_epoch=window.capture_epoch,
                start_ns=start_ns, end_ns=end_ns, source_revision=1))
        if session._overlap is not None:
            await session._overlap.register_caption(caption, words,
                session._correction_anchor(clause_id, end_ns), language)
        session._context.append(text)
        session._correction_context.append(text)
        return clause_id

    async def _commit(self, result):
        clause_id = await self._source(result)
        if clause_id is not None:
            await self.session._on_translation(TranslationEvent(clause_id, 1,
                self.session._translator.generation, "completed", result.text,
                elapsed_ms=result.model_ms, queue_ms=result.buffer_wait_ms))
            await self._retire_preview(result.units)
        self.update_counts()

    async def _failure(self, result):
        clause_id = await self._source(result)
        if clause_id is not None:
            await self.session._on_translation(TranslationEvent(clause_id, 1,
                self.session._translator.generation, "failed", error="semantic_" + result.reason,
                elapsed_ms=result.model_ms, queue_ms=result.buffer_wait_ms))
            await self._retire_preview(result.units)
        self.update_counts()
