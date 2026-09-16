"""Continuous single-track research pipeline, with independent input and inference.

The caller supplies actual PCM and VAD probabilities. No fake ASR, speaker model,
or translations are substituted when an engine fails. The synchronous ASR adapter
runs off the event loop; cancelling its thread does NOT stop native inference.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass, field
import math
import time
import unicodedata
from typing import Awaitable, Callable, Protocol

from .asr import ASRHypothesis, LocalAgreementStabilizer, TimedWord, TranscriptUpdate
from .audio import AudioFrame, AudioDiscontinuity, SpeechWindow, SpeechWindowBuilder
from .captions import CaptionSegment, CaptionStore
from .ingress_timing import INGRESS_SCOPE, IngressTimeline
from .translation import (
    LatestTranslationRunner, ProviderError, TranslationEvent, TranslationProvider,
    TranslationRequest,
)


class WindowTranscriber(Protocol):
    def transcribe_pcm(self, samples, *, sample_rate: int,
                       window_start_ns: int) -> ASRHypothesis: ...


@dataclass(frozen=True)
class PipelineEvent:
    kind: str
    sequence: int
    segment_id: str | None
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    source_language: str | None = None
    target_language: str = "ko"
    clause_target_s: float = 2.0
    clause_max_chars: int = 100
    translation_budget_ms: float = 2500
    max_asr_pending: int = 4
    max_translation_pending: int = 8
    context_clauses: int = 2

    def __post_init__(self):
        for value in (self.clause_target_s, self.translation_budget_ms):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Timing budgets must be finite and positive")
        if self.translation_budget_ms > 120000:
            raise ValueError("Translation budget exceeds the supported limit")
        for value in (self.clause_max_chars, self.max_asr_pending, self.max_translation_pending):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Capacity limits must be positive integers")
        if not 0 <= self.context_clauses <= 4:
            raise ValueError("Use zero to four context clauses")


class ClauseAssembler:
    """Emit each stable word once, without waiting for a long sentence to end.

    Punctuation/time/size boundaries are heuristics to benchmark, not a semantic
    clause detector. Context is supplied to the translator separately.
    """

    def __init__(self, target_s: float = 2, max_chars: int = 100):
        self.target_ns = round(target_s * 1e9)
        self.max_chars = max_chars
        self.consumed = 0
        self.pending: list[TimedWord] = []

    def accept(self, update: TranscriptUpdate, *, final: bool = False,
               playhead_ns: int | None = None) -> list[tuple[TimedWord, ...]]:
        if len(update.stable_words) < self.consumed:
            raise ValueError("Stable prefix rewound; start a new clause assembler")
        fresh = update.stable_words[self.consumed:]
        self.consumed = len(update.stable_words)
        result = []
        for word in fresh:
            self.pending.append(word)
            chars = sum(len(item.text) for item in self.pending)
            if (word.text.rstrip().endswith((".", "?", "!", "。", "？", "！"))
                    or word.end_time_ns - self.pending[0].start_time_ns >= self.target_ns
                    or chars >= self.max_chars):
                result.append(tuple(self.pending))
                self.pending.clear()
        if self.pending and (final or (playhead_ns is not None
                                      and playhead_ns - self.pending[0].start_time_ns >= self.target_ns)):
            result.append(tuple(self.pending))
            self.pending.clear()
        return result


@dataclass
class _SegmentState:
    stabilizer: LocalAgreementStabilizer
    assembler: ClauseAssembler
    clause_index: int = 0
    last_update: TranscriptUpdate | None = None
    last_window: SpeechWindow | None = None
    last_hypothesis: ASRHypothesis | None = None


class StreamingSession:
    """One audio track and one ordered ASR worker, translation in a separate queue.

    source_origin_monotonic is ONLY valid when input and measurement clocks are
    in this process (e.g. paced WAV replay). On a network server leave it None and
    let the capture host perform final audio-to-display deadline validation.
    """

    def __init__(self, session_id: str, transcriber: WindowTranscriber,
                 provider: TranslationProvider, *, builder: SpeechWindowBuilder | None = None,
                 config: PipelineConfig | None = None,
                 sink: Callable[[PipelineEvent], Awaitable[None]] | None = None,
                 source_origin_monotonic: float | None = None,
                 ingress_timeline: IngressTimeline | None = None,
                 speaker_analyzer=None, speaker_config=None, speaker_run_native=None):
        if source_origin_monotonic is not None and ingress_timeline is not None:
            raise ValueError("Choose one source clock scope for translation deadlines")
        self.config = config or PipelineConfig()
        self.builder = builder or SpeechWindowBuilder()
        self.transcriber = transcriber
        self.store = CaptionStore(session_id)
        self.sink = sink
        self.source_origin_monotonic = source_origin_monotonic
        self.ingress_timeline = ingress_timeline
        self._ingress_anchors: dict[str, float | None] = {}
        self._translation_deadlines: dict[str, float] = {}
        self._event_sequence = 0
        self._pending: OrderedDict[str, SpeechWindow] = OrderedDict()
        self._available = asyncio.Event()
        self._states: dict[str, _SegmentState] = {}
        self._context: deque[str] = deque(maxlen=self.config.context_clauses)
        self._translation_tasks: set[asyncio.Task] = set()
        self._translation_errors: list[tuple[str, Exception]] = []
        self._asr_task: asyncio.Task | None = None
        self._finishing = False
        self._closed = False
        self._last_frame: AudioFrame | None = None
        self._caption_times: dict[str, tuple[int, int]] = {}
        self._recent_words: deque[TimedWord] = deque(maxlen=128)
        self._recent_track: tuple[str, str] | None = None
        self._translator = LatestTranslationRunner(
            provider, self._on_translation,
            max_pending=self.config.max_translation_pending,
        )
        self.counts = {"frames": 0, "asr_windows": 0, "asr_errors": 0,
                       "asr_cached_finalizations": 0,
                       "boundary_duplicates": 0,
                       "asr_skipped": 0, "clauses": 0, "translation_completed": 0,
                       "translation_failed": 0, "audio_gaps": 0}
        self._speaker = self._speaker_mapper = None
        if speaker_analyzer is not None:
            from .speaker_stream import SpeakerStream
            from .speaker_captions import SpeakerCaptionReducer
            self._speaker_mapper = SpeakerCaptionReducer(self.store)
            self._speaker = SpeakerStream(speaker_analyzer, self._on_speaker,
                config=speaker_config or getattr(speaker_analyzer, "config", None), run_native=speaker_run_native)
            self.counts.update(self._speaker.counts)

    @property
    def speaker_status(self):
        if self._speaker is None:
            return "not_configured"
        return "degraded" if self._speaker.counts["speaker_errors"] else "configured_experimental"

    async def _speaker_patches(self, patches):
        for patch in patches:
            await self._emit("speaker.updated", patch.segment_id, **patch.data())

    async def _on_speaker(self, result):
        if isinstance(result, dict):
            kind = result["kind"]
            await self._emit(kind, **{key: value for key, value in result.items() if key != "kind"})
        else:
            if result.cleared_candidates:
                await self._emit("speaker.candidates_reset", track_id=result.track_id,
                                 capture_epoch=result.capture_epoch, count=result.cleared_candidates)
            for start_ns, end_ns, error in result.errors:
                await self._emit("speaker.failed", track_id=result.track_id, capture_epoch=result.capture_epoch,
                                 start_ns=start_ns, end_ns=end_ns, error=error)
            for speaker_id, name in result.identities:
                if self.store.set_speaker_alias(speaker_id, name):
                    await self._emit("speaker.alias_updated", speaker_id=speaker_id, name=name)
            unknown_reasons = {}
            for index, assignment in enumerate(result.assignments):
                if assignment.status == "unknown":
                    unknown_reasons[assignment.reason] = unknown_reasons.get(assignment.reason, 0) + 1
                else:
                    await self._emit("speaker.observed", track_id=result.track_id,
                        capture_epoch=result.capture_epoch, start_ns=assignment.start_time_ns,
                        end_ns=assignment.end_time_ns, sample_id=assignment.sample_id,
                        status=assignment.status, speaker_id=assignment.speaker_id,
                        candidate_id=assignment.candidate_id, similarity=assignment.similarity,
                        reason=assignment.reason)
                await self._speaker_patches(self._speaker_mapper.observe(assignment, capture_epoch=result.capture_epoch))
                if index % 16 == 15:
                    await asyncio.sleep(0)  # Yield long diagnostic batches to audio/results.
            await self._emit("speaker.analyzed", track_id=result.track_id, capture_epoch=result.capture_epoch,
                start_ns=result.start_ns, end_ns=result.end_ns, processing_ms=result.processing_ms,
                embedding_identity=result.model_identity, segmentation_identity=result.segmentation_identity,
                unknown_reasons=unknown_reasons, mapping_stats=dict(self._speaker_mapper.stats))
        if self._speaker is not None:
            self.counts.update(self._speaker.counts)

    async def _emit(self, kind: str, segment_id: str | None = None, **data) -> None:
        self._event_sequence += 1
        if self.sink:
            await self.sink(PipelineEvent(kind, self._event_sequence, segment_id, data))

    async def start(self) -> None:
        if self._closed or self._finishing:
            raise RuntimeError("Session is closed")
        if self._asr_task is None:
            self._asr_task = asyncio.create_task(self._asr_loop())

    async def feed(self, frame: AudioFrame, speech_probability: float) -> None:
        if self._closed or self._finishing:
            raise RuntimeError("Cannot feed a closing session")
        if self._last_frame and frame.start_time_ns < self._last_frame.end_time_ns:
            raise AudioDiscontinuity("Late or overlapping audio must not be replayed as a new segment")
        await self.start()
        self.counts["frames"] += 1
        try:
            windows = self.builder.push(frame, speech_probability)
        except AudioDiscontinuity as exc:
            # Finish audio already retained; never silently join across a gap.
            for window in self.builder.flush(reason="discontinuity"):
                await self._enqueue(window)
            self.builder.reset()
            self.counts["audio_gaps"] += 1
            await self._emit("audio.gap", track_id=frame.track_id,
                             next_start_ns=frame.start_time_ns, reason=str(exc))
            windows = self.builder.push(frame, speech_probability)
        self._last_frame = frame
        if self._speaker is not None:
            self._speaker.feed(frame)
        for window in windows:
            await self._enqueue(window)

    async def _enqueue(self, window: SpeechWindow) -> None:
        if window.segment_id in self._pending:
            # Same segment needs only its newest PCM snapshot, including endpoint.
            self._pending[window.segment_id] = window
        else:
            if len(self._pending) >= self.config.max_asr_pending:
                _, removed = self._pending.popitem(last=False)
                self.counts["asr_skipped"] += 1
                if removed.final:
                    self._states.pop(removed.segment_id, None)
                await self._emit("asr.skipped_overload", removed.segment_id,
                                 start_ns=removed.emit_start_ns, end_ns=removed.window_end_ns,
                                 final=removed.final)
            self._pending[window.segment_id] = window
        self._available.set()

    async def audio_gap(self, track_id: str, reason: str, next_start_ns: int | None) -> None:
        """Finish retained speech before an explicit capture/transport gap.

        The caller resets its VAD separately, after the preceding short tail.
        Keep the last source time so resetting an epoch cannot replay old audio.
        """
        if (not isinstance(track_id, str) or not track_id or not isinstance(reason, str)
                or not reason or (next_start_ns is not None and
                (isinstance(next_start_ns, bool) or not isinstance(next_start_ns, int) or next_start_ns < 0))):
            raise ValueError("Invalid audio gap metadata")
        if self._last_frame and next_start_ns is not None and next_start_ns < self._last_frame.end_time_ns:
            raise AudioDiscontinuity("Gap cannot rewind the source audio clock")
        await self.start()
        for window in self.builder.flush(reason="discontinuity"):
            await self._enqueue(window)
        self.builder.reset()
        if self._speaker is not None:
            self._speaker.gap()
        self._context.clear()
        self._recent_words.clear()
        self._recent_track = None
        self.counts["audio_gaps"] += 1
        await self._emit("audio.gap", track_id=track_id, next_start_ns=next_start_ns, reason=reason)

    async def _asr_loop(self) -> None:
        while True:
            await self._available.wait()
            while self._pending:
                _, window = self._pending.popitem(last=False)
                await self._process_window(window)
            self._available.clear()
            if self._finishing:
                return

    async def _process_window(self, window: SpeechWindow) -> None:
        started = time.monotonic()
        state = self._states.setdefault(window.segment_id, _SegmentState(
            LocalAgreementStabilizer(window.segment_id),
            ClauseAssembler(self.config.clause_target_s, self.config.clause_max_chars),
        ))
        try:
            cached = (window.final and state.last_window is not None
                      and window.window_start_ns == state.last_window.window_start_ns
                      and window.window_end_ns == state.last_window.window_end_ns
                      and window.samples == state.last_window.samples)
            if cached:
                hypothesis = state.last_hypothesis
                self.counts["asr_cached_finalizations"] += 1
            else:
                hypothesis = await asyncio.to_thread(
                    self.transcriber.transcribe_pcm, window.samples,
                    sample_rate=16000, window_start_ns=window.window_start_ns,
                )
                self.counts["asr_windows"] += 1
            assert hypothesis is not None
            if (hypothesis.window_start_ns != window.window_start_ns
                    or hypothesis.window_end_ns != window.window_end_ns):
                raise ValueError("ASR changed the source audio window timeline")
            # Reconcile context against already emitted boundary words. Checking
            # end time alone duplicates a word clipped at the preceding boundary.
            track = (window.track_id, window.capture_epoch)
            if track != self._recent_track:
                self._recent_words.clear()
                self._recent_track = track
            kept = []
            for word in hypothesis.words:
                if (word.end_time_ns < window.emit_start_ns
                        or (word.end_time_ns == window.emit_start_ns
                            and word.start_time_ns < window.emit_start_ns)):
                    continue
                if self._is_boundary_duplicate(word, window.emit_start_ns):
                    self.counts["boundary_duplicates"] += 1
                    await self._emit("asr.boundary_duplicate_suppressed", window.segment_id,
                                     text=word.text, start_ns=word.start_time_ns, end_ns=word.end_time_ns)
                else:
                    kept.append(word)
            hypothesis = ASRHypothesis(
                hypothesis.window_start_ns, hypothesis.window_end_ns,
                tuple(kept),
                hypothesis.language,
            )
            update = state.stabilizer.accept(hypothesis)
            state.last_window = window
            state.last_hypothesis = hypothesis
            if update:
                state.last_update = update
                await self._emit("transcript.updated", window.segment_id,
                                 revision=update.revision, text=update.text,
                                 stable_text=update.stable_text,
                                 source_state=update.source_state,
                                 source_start_ns=window.emit_start_ns,
                                 source_end_ns=window.window_end_ns,
                                 asr_ms=(time.monotonic() - started) * 1000)
                await self._emit_clauses(window, state, update, hypothesis.language)
            elif state.last_update:
                # Time can advance while the stable words are unchanged. A short
                # pending clause must still be released by the latency boundary.
                await self._emit_clauses(window, state, state.last_update, hypothesis.language)
            if window.final:
                flushed = state.stabilizer.flush()
                if flushed:
                    state.last_update = flushed
                    await self._emit("transcript.updated", window.segment_id,
                                     revision=flushed.revision, text=flushed.text,
                                     stable_text=flushed.stable_text, source_state="stable",
                                     source_start_ns=window.emit_start_ns,
                                     source_end_ns=window.window_end_ns, reason="flush")
                if state.last_update:
                    await self._emit_clauses(window, state, state.last_update,
                                            hypothesis.language, final=True)
                self._states.pop(window.segment_id, None)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.counts["asr_errors"] += 1
            await self._emit("asr.failed", window.segment_id,
                             error=type(exc).__name__, start_ns=window.emit_start_ns,
                             end_ns=window.window_end_ns)
            if window.final:
                if state.last_update:
                    # Preserve genuinely agreed words even when the final model
                    # invocation fails. Do not promote its unverified suffix.
                    await self._emit_clauses(window, state, state.last_update,
                                            self.config.source_language, final=True)
                    start_ns = (state.last_update.stable_words[-1].end_time_ns
                                if state.last_update.stable_words else window.emit_start_ns)
                    await self._emit("asr.incomplete", window.segment_id,
                                     start_ns=start_ns, end_ns=window.window_end_ns,
                                     provisional_text=state.last_update.provisional_text,
                                     reason="final_inference_failed")
                self._states.pop(window.segment_id, None)

    def _is_boundary_duplicate(self, word: TimedWord, boundary_ns: int) -> bool:
        if word.start_time_ns >= boundary_ns:
            return False  # Adjacent genuine repeated words must survive.
        def key(text: str) -> str:
            return unicodedata.normalize("NFKC", text).strip().casefold().strip(".,?!。？！")
        for previous in reversed(self._recent_words):
            if previous.end_time_ns < boundary_ns - 500_000_000:
                break
            overlap = min(word.end_time_ns, previous.end_time_ns) - max(word.start_time_ns, previous.start_time_ns)
            shorter = min(word.end_time_ns - word.start_time_ns,
                          previous.end_time_ns - previous.start_time_ns)
            if (key(word.text) == key(previous.text) and shorter > 0
                    and overlap >= shorter * .6
                    and abs(word.start_time_ns - previous.start_time_ns) <= 250_000_000):
                return True
        return False

    async def _emit_clauses(self, window: SpeechWindow, state: _SegmentState,
                            update: TranscriptUpdate, language: str | None,
                            *, final: bool = False) -> None:
        for words in state.assembler.accept(update, final=final,
                                             playhead_ns=window.window_end_ns):
            text = "".join(word.text for word in words).strip()
            start_ns = max(window.emit_start_ns, words[0].start_time_ns)
            end_ns = max(word.end_time_ns for word in words)
            if not text:
                continue
            if end_ns <= start_ns:
                # Whisper can return a nonempty word with equal start/end.
                # Keep an explicit loss record instead of inventing subtitle time.
                self.counts["asr_errors"] += 1
                await self._emit("asr.incomplete", window.segment_id,
                                 start_ns=start_ns, end_ns=end_ns,
                                 source_text=text, reason="invalid_word_timing")
                continue
            state.clause_index += 1
            clause_id = f"{window.segment_id}:clause-{state.clause_index}"
            self.store.upsert_source(CaptionSegment(
                segment_id=clause_id, track_id=window.track_id,
                start_ns=start_ns, end_ns=end_ns, source_text=text,
                source_revision=1, source_state="stable",
                target_language=self.config.target_language,
            ))
            self._caption_times[clause_id] = (start_ns, end_ns)
            if self.ingress_timeline is not None:
                self._ingress_anchors[clause_id] = self.ingress_timeline.anchor(
                    window.track_id, window.capture_epoch, start_ns, end_ns)
            self._recent_words.extend(words)
            self.counts["clauses"] += 1
            await self._emit("caption.source", clause_id, text=text,
                             source_revision=1, start_ns=start_ns, end_ns=end_ns,
                             track_id=window.track_id, capture_epoch=window.capture_epoch)
            if self._speaker_mapper is not None:
                await self._speaker_patches(self._speaker_mapper.register_caption(
                    clause_id, track_id=window.track_id, capture_epoch=window.capture_epoch,
                    start_ns=start_ns, end_ns=end_ns, source_revision=1))
            budget = self.config.translation_budget_ms
            now = time.monotonic()
            deadline = now + budget / 1000
            timing_error = None
            if self.source_origin_monotonic is not None:
                deadline = min(deadline, self.source_origin_monotonic + end_ns / 1e9 + budget / 1000)
            elif self.ingress_timeline is not None:
                anchor = self._ingress_anchors[clause_id]
                if anchor is None:
                    timing_error = "ingress_timing_unavailable"
                elif anchor > now:
                    timing_error = "ingress_clock_invalid"
                else:
                    deadline = anchor + budget / 1000
            budget = (deadline - now) * 1000
            self._translation_deadlines[clause_id] = deadline
            context = tuple(self._context)
            self._context.append(text)
            if timing_error or budget <= 0:
                self.counts["translation_failed"] += 1
                await self._emit("translation.failed", clause_id,
                                 error=timing_error or ("ingress_deadline" if self.ingress_timeline is not None else "source_deadline"),
                                 source_revision=1, provider_generation=self._translator.generation,
                                 audio_to_event_ms=self._audio_elapsed_ms(end_ns),
                                 **self._ingress_metrics(clause_id))
                continue
            try:
                task = self._translator.submit(TranslationRequest(
                    clause_id, 1, text, self.config.source_language or language or "auto",
                    self.config.target_language, context, min(budget, self.config.translation_budget_ms),
                    deadline_monotonic=deadline,
                ))
                self._translation_tasks.add(task)
                def completed(done: asyncio.Task, segment_id=clause_id):
                    self._translation_tasks.discard(done)
                    if not done.cancelled() and done.exception() is not None:
                        self._translation_errors.append((segment_id, done.exception()))
                task.add_done_callback(completed)
            except ProviderError as exc:
                self.counts["translation_failed"] += 1
                await self._emit("translation.failed", clause_id, error=exc.category,
                                 source_revision=1, provider_generation=self._translator.generation,
                                 audio_to_event_ms=self._audio_elapsed_ms(end_ns),
                                 **self._ingress_metrics(clause_id))

    def _ingress_metrics(self, clause_id):
        if self.ingress_timeline is None:
            return {}
        anchor = self._ingress_anchors.get(clause_id)
        return dict(translation_budget_scope=INGRESS_SCOPE,
                    server_ingress_age_ms=None if anchor is None else (time.monotonic() - anchor) * 1000,
                    translation_budget_ms=self.config.translation_budget_ms)

    def _audio_elapsed_ms(self, end_ns: int) -> float | None:
        if self.source_origin_monotonic is None:
            return None
        return (time.monotonic() - self.source_origin_monotonic - end_ns / 1e9) * 1000

    async def _on_translation(self, event: TranslationEvent) -> None:
        if event.segment_id not in self._caption_times:
            return
        if event.kind == "preview" and not event.text.strip():
            return
        if event.kind in ("preview", "completed"):
            deadline = self._translation_deadlines.get(event.segment_id)
            if deadline is not None and time.monotonic() >= deadline:
                raise ProviderError("timeout", "Translation expired before caption acceptance")
            accepted = self.store.apply_translation(
                event.segment_id, event.revision, event.generation, event.text,
                target_language=self.config.target_language,
                completed=event.kind == "completed",
            )
            if not accepted:
                return
        if event.kind == "completed":
            self.counts["translation_completed"] += 1
        elif event.kind in ("failed", "cancelled"):
            self.counts["translation_failed"] += 1
        _, end_ns = self._caption_times[event.segment_id]
        await self._emit("translation." + event.kind, event.segment_id,
                         text=event.text, source_revision=event.revision,
                         provider_generation=event.generation, error=event.error,
                         queue_ms=event.queue_ms, provider_elapsed_ms=event.elapsed_ms,
                         audio_to_event_ms=self._audio_elapsed_ms(end_ns), usage=event.usage,
                         **self._ingress_metrics(event.segment_id))

    async def finish(self, *, timeout_s: float = 60) -> None:
        if self._closed:
            return
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("finish timeout must be finite and positive")
        deadline = time.monotonic() + timeout_s
        await self.start()
        if self._speaker is not None:
            self._speaker.finish_input()
        for window in self.builder.flush():
            await self._enqueue(window)
        self._finishing = True
        self._available.set()
        assert self._asr_task is not None
        done, _ = await asyncio.wait({self._asr_task}, timeout=max(0, deadline - time.monotonic()))
        if not done:
            self._asr_task.cancel()
            await self._emit("pipeline.failed", error="asr_drain_timeout")
            await self.close()
            raise TimeoutError("ASR drain timed out; native inference may still be running")
        await self._asr_task
        tasks = set(self._translation_tasks)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=max(0, deadline - time.monotonic()))
            if pending:
                await self._emit("pipeline.failed", error="translation_drain_timeout")
                await self.close()
                raise TimeoutError("Translation drain timed out")
            for task in tasks:
                if not task.cancelled():
                    task.exception()  # Callback retains errors even after task removal.
        for segment_id, error in self._translation_errors:
            self.counts["translation_failed"] += 1
            await self._emit("translation.failed", segment_id,
                             error="unexpected_" + type(error).__name__)
        self._translation_errors.clear()
        if self._speaker is not None:
            await self._speaker.drain(max(0, deadline - time.monotonic()))
            self.counts.update(self._speaker.counts)
        await self.close()

    async def close(self) -> None:
        self._closed = True
        if self._asr_task and not self._asr_task.done():
            self._asr_task.cancel()
            await asyncio.gather(self._asr_task, return_exceptions=True)
        self._pending.clear()
        self._states.clear()
        await self._translator.close()
        if self._speaker is not None:
            await self._speaker.close()
            self.counts.update(self._speaker.counts)
