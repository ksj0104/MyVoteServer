"""Opt-in semantic boundaries over the verified engine's unchanged ASR.

Only semantic buffering/routing changes. Stable words, source clocks, acoustic
speaker decisions, preview rendering, caption publication and deadlines retain
the upstream implementations. Audio gaps are stamped onto queued windows so a
slow ASR completion cannot accidentally bridge a newer capture discontinuity.
"""

from dataclasses import dataclass, fields, replace
import hashlib
from types import SimpleNamespace


def install_pipeline(semantic_module, streaming_module, audio_module, *, quality_first=False):
    if type(quality_first) is not bool:
        raise ValueError("quality_first must be a bool")
    metadata = {"semantic_boundary_pipeline": "natural-prefix-v1"}
    if quality_first:
        metadata.update(semantic_endpoint_resume="evidence-aware-vad-continuation-v2",
                        semantic_endpoint_resume_max_gap_ms=1500,
                        semantic_continuity="anonymous-acoustic-turn-v2")
    if getattr(semantic_module.SemanticCaptionPipeline, "_myvote_natural_boundaries_v1", False):
        if semantic_module.SemanticCaptionPipeline._myvote_endpoint_resume != quality_first:
            raise ValueError("Semantic pipeline is already installed with a different endpoint policy")
        return metadata
    original_pipeline = semantic_module.SemanticCaptionPipeline
    original_session = streaming_module.StreamingSession
    from myvote_engine.speaker_captions import _CaptionRef

    # These are routing permissions, not speaker assignments. Explicit adverse
    # evidence is never erased by a larger amount of benign/confirmed evidence.
    continuity_reasons = frozenset(("dominant_confirmed_speaker", "insufficient_coverage",
                                    "unknown_or_overlap_evidence"))

    @dataclass(frozen=True)
    class BoundaryWindow(audio_module.SpeechWindow):
        semantic_generation: int = 0

    window_fields = fields(audio_module.SpeechWindow)

    def stamp(window, generation):
        return BoundaryWindow(**{field.name: getattr(window, field.name) for field in window_fields},
                              semantic_generation=generation)

    class NaturalBoundaryPipeline(original_pipeline):
        _myvote_natural_boundaries_v1 = True
        _myvote_endpoint_resume = quality_first

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._boundary_generation = None
            self._boundary_continuation = None
            self._boundary_speaker = None
            self._boundary_turn = None
            self._boundary_endpoint = None
            self._boundary_anchor = None
            self._boundary_adverse = False
            self._continuity_cache = {}
            self._quality_endpoint_resume = self._myvote_endpoint_resume

        def break_continuity(self, generation):
            self.coordinator.request_flush()
            self._last_route = None
            self._boundary_speaker = self._boundary_turn = None
            self._boundary_continuation = None
            self._boundary_endpoint = None
            self._boundary_anchor = None
            self._boundary_adverse = False
            self._boundary_generation = generation

        def _speaker_for(self, window, word):
            mapper = self.session._speaker_mapper
            if mapper is None:
                return None
            return mapper.confirmed_speaker_for(
                track_id=window.track_id, capture_epoch=window.capture_epoch,
                start_ns=word.start_time_ns, end_ns=word.end_time_ns)

        def _pending_confirmed_as(self, route, speaker):
            # None can also mean overlap/conflict. Never substitute a previous
            # speaker for it. Reuse an unknown lane only with acoustic evidence
            # for every still-pending source word; never move frozen units.
            units = [unit for unit in self.coordinator.pending_units if unit.scope == route]
            return bool(units) and all(
                self._speaker_for(unit.payload.window, unit.payload.word) == speaker for unit in units)

        def _continuity_evidence(self, window, word):
            mapper = self.session._speaker_mapper
            stats = getattr(mapper, "stats", {})
            key = (id(mapper), window.track_id, window.capture_epoch,
                   word.start_time_ns, word.end_time_ns,
                   getattr(mapper, "retention_floor_ns", 0),
                   stats.get("assignments", 0), stats.get("candidate_confirmations", 0))
            if key not in self._continuity_cache:
                self._continuity_cache[key] = self._read_continuity_evidence(window, word)
            return self._continuity_cache[key]

        def _read_continuity_evidence(self, window, word):
            """Inspect exact acoustic guards without assigning any caption name.

            Lack of a confirmed label is not positive evidence of a new turn.
            The original reducer remains solely responsible for caption names.
            Unknown identity can still conceal a real change of speaker; opaque
            continuity lanes deliberately make no identity claim about it.
            """
            mapper = self.session._speaker_mapper
            if mapper is None:
                return None, True
            if (word.end_time_ns <= word.start_time_ns
                    or word.start_time_ns < getattr(mapper, "retention_floor_ns", 0)):
                return None, False
            decide = getattr(mapper, "_decision", None)
            if decide is None:
                speaker = self._speaker_for(window, word)
                # An opaque adapter cannot distinguish overlap from missing
                # evidence. Keep its unknowns conservative, not silently safe.
                return speaker, speaker is not None
            decision = decide(_CaptionRef("semantic-continuity", window.track_id,
                              window.capture_epoch, word.start_time_ns, word.end_time_ns, 1))
            allowed = (decision.reason in continuity_reasons
                       and not any(getattr(decision, name, 0) for name in
                                   ("overlap_ns", "ambiguous_ns", "conflict_ns")))
            return decision.speaker_id, allowed

        def _compatible_continuity(self, route, sources, anchor=None, *, require_pending=False):
            pending = [unit for unit in self.coordinator.pending_units if unit.scope == route]
            if require_pending and not pending:
                return False, anchor
            confirmed = set() if anchor is None else {anchor}
            for source in (*[unit.payload for unit in pending], *sources):
                speaker, allowed = self._continuity_evidence(source.window, source.word)
                if not allowed:
                    return False, anchor
                if speaker is not None:
                    confirmed.add(speaker)
                    if len(confirmed) > 1:
                        return False, anchor
            return True, next(iter(confirmed), None)

        def _count_boundary(self, key):
            self.session.counts[key] = self.session.counts.get(key, 0) + 1

        def _note_provisional_activity(self, preview, window, source_scope, generation):
            """Extend only the compatible active lane's transcription clock.

            This is not a stable-word promotion or a boundary resume. Unknown,
            benign evidence follows the same anonymous continuity permission as
            stable appends; it never assigns a speaker. Explicit conflict,
            overlap or another confirmed speaker cannot keep this lane alive.
            """
            notify = getattr(self.coordinator, "note_transcription", None)
            words = preview.update.provisional_words
            if (not self._quality_endpoint_resume or not callable(notify)
                    or self.session._closed or not words):
                return
            route = preview._boundary_route
            if route is not None:
                if (route != self._last_route or route[:3] != source_scope
                        or preview._boundary_turn != self._boundary_turn):
                    return
            else:
                # A provisional-only next VAD window can keep its predecessor
                # alive, but cannot withdraw the marker before stable evidence.
                candidate = preview._boundary_endpoint
                if candidate is None:
                    return
                old_generation, old_scope, old_turn, route, _, _, end_ns = candidate
                if (old_generation != generation or old_scope != source_scope
                        or route != self._last_route or old_turn != self._boundary_turn
                        or not 0 <= words[0].start_time_ns - end_ns <= 1_500_000_000):
                    return
            pending = tuple(unit for unit in self.coordinator.pending_units if unit.scope == route)
            if (not pending or words[0].start_time_ns < pending[-1].payload.word.end_time_ns):
                return
            # Reuse the stable-word routing guard instead of inventing a
            # stricter timer-only speaker requirement. Otherwise anonymous
            # ongoing transcription would be forced while it is still changing.
            sources = tuple(SimpleNamespace(window=window, word=word) for word in words)
            compatible, _ = self._compatible_continuity(route, sources,
                                                        self._boundary_anchor, require_pending=True)
            if not compatible:
                return
            notify(route)
            self._count_boundary("semantic_transcription_activity_notes")

        async def accept(self, window, state, update, language, *, final, timing):
            # Reduce each exact source interval only once per acoustic evidence
            # revision in this batch. Never carry proof across ASR accepts;
            # assignments arriving during awaits change the cache key as well.
            self._continuity_cache.clear()
            source_language = self.session.config.source_language
            if not source_language or source_language.lower() in ("auto", "und", "unknown"):
                source_language = language or "auto"
            generation = getattr(window, "semantic_generation", 0)
            if self._boundary_generation != generation:
                self.break_continuity(generation)
            source_scope = (window.track_id, window.capture_epoch, source_language)
            metadata = replace(window, samples=())
            if state.semantic_state is None:
                state.semantic_state = semantic_module._Preview(metadata, update, source_language)
                turn = window.segment_id
                continuation = self._boundary_continuation
                if (continuation is not None and continuation[:2] == (generation, source_scope)
                        and window.emit_start_ns == continuation[2]
                        and window.window_start_ns <= continuation[2]):
                    turn = continuation[3]
                self._boundary_continuation = None
                state.semantic_state._boundary_turn = turn
                state.semantic_state._boundary_route = None
                # The first result in the next window may contain only a
                # provisional suffix. Defer identity/continuity checks until
                # that window supplies its first immutable stable word.
                state.semantic_state._boundary_endpoint = self._boundary_endpoint
                self._boundary_endpoint = None
            preview = state.semantic_state
            preview.window, preview.update, preview.language = metadata, (
                replace(update, provisional_words=()) if final else update), source_language
            await self._preview(preview)
            transcription = "".join(word.text for word in (
                *preview.update.stable_words, *preview.update.provisional_words)).strip()
            signature = hashlib.sha256(transcription.encode("utf-8")).digest() if transcription else None
            changed = signature != getattr(preview, "_boundary_transcription_signature", None)
            preview._boundary_transcription_signature = signature
            fresh = update.stable_words[preview.submitted:]
            # append() accounts for fresh immutable words itself. Only a real
            # text change without new stable words needs this optional hook;
            # repeated hypotheses, timestamp-only revisions and empty updates
            # cannot postpone another lane's inactivity flush.
            if changed and transcription and not fresh and not final:
                self._note_provisional_activity(preview, window, source_scope, generation)
            sources = tuple(semantic_module._Source(preview, preview.submitted + index, word,
                            state.pending_word_timings.popleft(), timing, metadata,
                            update.revision, update.reason) for index, word in enumerate(fresh))
            preview.submitted = len(update.stable_words)
            self.session._recent_words.extend(fresh)
            candidate = preview._boundary_endpoint
            if sources:
                preview._boundary_endpoint = None
                if self._quality_endpoint_resume and candidate is not None:
                    old_generation, old_scope, old_turn, old_route, old_speaker, unit_id, end_ns = candidate
                    first = sources[0].word
                    gap_ns = first.start_time_ns - end_ns
                    compatible, anchor = self._compatible_continuity(
                        old_route, sources, old_speaker, require_pending=True)
                    if (old_generation == generation and old_scope == source_scope
                            and old_route == self._last_route and old_turn == self._boundary_turn
                            and 0 <= gap_ns <= 1_500_000_000
                            and compatible
                            and self.coordinator.resume_boundary(old_route, unit_id, allow_after_hold=True)):
                        # Only the exact previous VAD endpoint is withdrawn.
                        # Earlier speaker-turn/hard boundaries and in-flight
                        # source snapshots retain their original ownership.
                        preview._boundary_turn = old_turn
                        self._boundary_anchor = anchor
                        self._count_boundary("semantic_endpoint_resumes")
                    elif not compatible:
                        self._count_boundary("semantic_endpoint_rejected_evidence")
            turn = preview._boundary_turn
            for source in sources:
                self._unit_sequence += 1
                speaker = None if self._quality_endpoint_resume else self._speaker_for(window, source.word)
                lane = f"speaker:{speaker}" if speaker else f"unassigned:{turn}"
                route = (*source_scope, f"boundary-g{generation}:{lane}")
                same_turn = (self._last_route is not None and self._last_route[:3] == source_scope
                             and self._boundary_turn == turn)
                if self._quality_endpoint_resume:
                    # Anonymous lane identity is about immutable source
                    # continuity, never a guessed automatic speaker label.
                    speaker, allowed = self._continuity_evidence(window, source.word)
                    route = (*source_scope, f"boundary-g{generation}:continuity:{turn}:r{self._unit_sequence}")
                    reuse, anchor = False, speaker
                    if same_turn and allowed and not self._boundary_adverse:
                        reuse, anchor = self._compatible_continuity(
                            self._last_route, (source,), self._boundary_anchor)
                        if not reuse:
                            anchor = speaker
                    elif same_turn and not allowed and self._boundary_adverse:
                        # Keep an adverse interval separate from its neighbors,
                        # without spending a new lane on every overlap word.
                        reuse, anchor = True, None
                    if reuse:
                        route = self._last_route
                        if allowed and (speaker is None or self._boundary_speaker is None):
                            self._count_boundary("semantic_continuity_uncertain_bridges")
                    elif same_turn:
                        if not allowed or self._boundary_adverse:
                            self._count_boundary("semantic_continuity_adverse_boundaries")
                        else:
                            self._count_boundary("semantic_continuity_speaker_boundaries")
                    self._boundary_anchor, self._boundary_adverse = anchor, not allowed
                elif same_turn and self._boundary_speaker is not None and speaker is None:
                    # A confirmed lane may itself have an old "unassigned"
                    # opaque key after evidence-based promotion. Losing speaker
                    # evidence must still start a distinct uncertainty run.
                    route = (*source_scope, f"boundary-g{generation}:unassigned:{turn}:r{self._unit_sequence}")
                if not self._quality_endpoint_resume and same_turn and (speaker == self._boundary_speaker or (
                        self._boundary_speaker is None and speaker is not None
                        and self._pending_confirmed_as(self._last_route, speaker))):
                    route = self._last_route
                if self._last_route is not None and (route != self._last_route or not same_turn):
                    self.coordinator.request_boundary(self._last_route)
                self._last_route = preview._boundary_route = route
                self._boundary_speaker, self._boundary_turn = speaker, turn
                unit = semantic_module.SemanticPendingUnit(f"u{self._unit_sequence}", source.word.text[:12000],
                    source_language, window.track_id, window.capture_epoch,
                    source.commit.ready_at, source, lane_id=route[-1])
                await self.coordinator.append(unit)
            if final:
                if window.reason == "max_window":
                    # An ASR allocation limit is not the end of a spoken thought.
                    self._boundary_continuation = (generation, source_scope, window.window_end_ns, turn)
                    self._boundary_endpoint = None
                elif window.reason == "endpoint":
                    # Let the model split a ready prefix first, even if the VAD
                    # reports an endpoint. The coordinator owns timing policy.
                    if preview._boundary_route is not None:
                        self.coordinator.request_boundary(preview._boundary_route)
                    self._boundary_continuation = None
                    self._boundary_endpoint = None
                    if (self._quality_endpoint_resume and preview._boundary_route is not None
                            and preview._boundary_route == self._last_route):
                        units = [unit for unit in self.coordinator.pending_units
                                 if unit.scope == preview._boundary_route]
                        if units and units[-1].payload.preview is preview:
                            self._boundary_endpoint = (
                                generation, source_scope, turn, preview._boundary_route,
                                self._boundary_anchor, units[-1].unit_id,
                                units[-1].payload.word.end_time_ns,
                            )
                elif window.reason == "transcript_inactivity":
                    # The normal ASR finalization owns any newly stable words.
                    # Only this preview's lane has earned an inactivity drain;
                    # another speaker's live lane must never be globally forced.
                    if preview._boundary_route is not None:
                        self.coordinator.request_flush(preview._boundary_route)
                    self._boundary_continuation = None
                    self._boundary_endpoint = None
                else:
                    # Stop, gap and unknown final reasons remain hard boundaries.
                    self.coordinator.request_flush()
                    self._boundary_continuation = None
                    self._boundary_endpoint = None
            self.update_counts()

    class BoundaryStreamingSession(original_session):
        def __init__(self, *args, **kwargs):
            self._boundary_capture_generation = 0
            self._boundary_processing = False
            super().__init__(*args, **kwargs)

        def _boundary_cut(self, *, retag_pending=False):
            previous_capture = self._boundary_capture_generation
            if retag_pending:
                # Called when the oldest queued ASR window was skipped. All
                # surviving pending windows follow the hole; the active one is
                # older and retains its immutable generation. Preserve earlier
                # gap partitions instead of merging every survivor into one run.
                remap = {}
                next_generation = previous_capture
                for key, window in tuple(self._pending.items()):
                    old = getattr(window, "semantic_generation", 0)
                    if old not in remap:
                        next_generation += 1
                        remap[old] = next_generation
                    self._pending[key] = stamp(window, remap[old])
                self._boundary_capture_generation = remap.get(previous_capture, next_generation + 1)
            else:
                self._boundary_capture_generation += 1
            self._boundary_cut_if_drained()

        def _boundary_cut_if_drained(self):
            if (self._semantic is not None and not self._boundary_processing and not self._pending
                    and self._semantic._boundary_generation != self._boundary_capture_generation):
                self._semantic.break_continuity(self._boundary_capture_generation)

        async def _emit(self, kind, segment_id=None, **data):
            if kind == "audio.gap":
                self._boundary_cut()
            elif kind == "asr.skipped_overload":
                self._boundary_cut(retag_pending=True)
            return await super()._emit(kind, segment_id, **data)

        async def _enqueue(self, window):
            generation = self._boundary_capture_generation
            wrapped = stamp(window, generation)
            await super()._enqueue(wrapped)
            # Overload notification occurs inside super()._enqueue before it
            # inserts the new window. Stamp that new survivor after the cut too.
            if (generation != self._boundary_capture_generation
                    and self._pending.get(window.segment_id) is wrapped):
                self._pending[window.segment_id] = stamp(wrapped, self._boundary_capture_generation)

        async def _process_window(self, window, *, queued_at=None):
            self._boundary_processing = True
            errors = self.counts.get("asr_errors", 0)
            try:
                return await super()._process_window(window, queued_at=queued_at)
            finally:
                self._boundary_processing = False
                if window.final and self.counts.get("asr_errors", 0) > errors:
                    self._boundary_cut(retag_pending=True)
                self._boundary_cut_if_drained()

    semantic_module.SemanticCaptionPipeline = NaturalBoundaryPipeline
    streaming_module.StreamingSession = BoundaryStreamingSession
    return metadata
