"""Turn bounded separation/ASR results into complete, atomic caption corrections.

The first mixed caption remains visible until two separated utterances and all
replacement translations are ready. Original words outside detected overlap are
retained verbatim. This experimental path does not establish speaker identity.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, replace
import json
import math
import time

from .audio import SAMPLE_NS
from .captions import CaptionGroupReplacement, CaptionParentRef, CaptionSegment, CaptionTranslation
from .priority_translation import AdmittedTranslationProvider
from .translation import ProviderError, TranslationRequest, _closing_stream, text_operation_for


@dataclass(frozen=True)
class OverlapCaptionConfig:
    correction_budget_ms: float = 15000
    context_s: float = 8
    max_event_bytes: int = 65536
    max_recent_parents: int = 256

    def __post_init__(self):
        for value in (self.correction_budget_ms, self.context_s):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Overlap times must be finite numbers")
        if not 1000 <= self.correction_budget_ms <= 60000 or not 1 <= self.context_s <= 10:
            raise ValueError("Use a 1-60s correction deadline and 1-10s separation context")
        if type(self.max_event_bytes) is not int or not 256 <= self.max_event_bytes <= 1048576:
            raise ValueError("Invalid correction event byte limit")
        if type(self.max_recent_parents) is not int or not 1 <= self.max_recent_parents <= 256:
            raise ValueError("Retain at most 256 recent parent clauses")


@dataclass(frozen=True)
class _Parent:
    caption: CaptionSegment
    words: tuple
    anchor: float | None
    language: str


def _inside_overlap(word, regions):
    midpoint = (word.start_time_ns + word.end_time_ns) // 2
    return any(start <= midpoint < end for start, end in regions)


def _merged_regions(regions):
    """One ordered union, so repeated segmentation windows do not duplicate text."""
    result = []
    for start, end in sorted(set(regions)):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return tuple(result)


class OverlapCaptionCoordinator:
    def __init__(self, session, separator, scheduler, admission, *, config=None, run_native=None):
        from .overlap_stream import OverlapStream, OverlapStreamConfig
        self.session = session
        self.config = config or OverlapCaptionConfig()
        self.admission = admission
        self.parents = OrderedDict()
        self.regions = deque(maxlen=64)
        self.attempted = set()
        self._sequence = 0
        self._closed = False
        self.counts = dict(overlap_groups_completed=0, overlap_groups_skipped=0,
                           overlap_translations_completed=0)
        self.stream = OverlapStream(separator, session.transcriber, self._on_result,
                                    config=OverlapStreamConfig(window_s=self.config.context_s),
                                    run_native=run_native, asr_scheduler=scheduler)

    def feed(self, frame):
        self.stream.feed(frame)
        cutoff = frame.end_time_ns - 20_000_000_000
        for key, value in tuple(self.parents.items()):
            if value.caption.end_ns < cutoff:
                self.parents.pop(key)
                self.attempted.discard(key)

    def gap(self):
        self.stream.gap()
        self.parents.clear()
        self.attempted.clear()
        self.regions.clear()

    async def register_caption(self, caption, words, anchor, language):
        if self._closed:
            return
        self.parents[caption.segment_id] = _Parent(caption, tuple(words), anchor, language or "auto")
        while len(self.parents) > self.config.max_recent_parents:
            key, _ = self.parents.popitem(last=False)
            self.attempted.discard(key)
        await self._schedule()

    async def observe_analysis(self, analysis):
        if self._closed or not getattr(analysis, "overlap_regions", ()):
            return
        self.regions.append((analysis.track_id, analysis.capture_epoch, analysis.run_id,
                             analysis.run_start_ns, tuple(analysis.overlap_regions)))
        await self._schedule()

    async def _skip(self, reason, *, parent_id=None, **data):
        self.counts["overlap_groups_skipped"] += 1
        await self.session._emit("overlap.skipped", parent_id, reason=reason,
                                 delivery_class="overlap_correction", **data)

    async def _schedule(self):
        for parent_id, parent in tuple(self.parents.items()):
            if parent_id in self.attempted:
                continue
            caption = parent.caption
            regions = []
            run_id = None
            for track, epoch, run, run_start, spans in self.regions:
                if (track != caption.track_id or epoch != caption.capture_epoch
                        or (run_start is not None and caption.start_ns < run_start)):
                    continue
                for start, end in spans:
                    start, end = max(start, caption.start_ns), min(end, caption.end_ns)
                    if end > start:
                        regions.append((start, end))
                        run_id = run
            regions = _merged_regions(regions)
            if not regions or not any(_inside_overlap(word, regions) for word in parent.words):
                continue
            # One attempt owns this parent. Repeated segmentation observations
            # cannot vote twice or emit duplicate separated clauses.
            self.attempted.add(parent_id)
            if parent.anchor is None:
                await self._skip("source_timing_unavailable", parent_id=parent_id)
                continue
            deadline = parent.anchor + self.config.correction_budget_ms / 1000
            if time.monotonic() >= deadline:
                await self._skip("correction_deadline", parent_id=parent_id)
                continue
            window = self.stream.suggest_window(caption.start_ns, caption.end_ns)
            if window is None:
                await self._skip("audio_context_unavailable", parent_id=parent_id)
                continue
            self._sequence += 1
            group_id = f"overlap-{self._sequence}"
            self.stream.request(
                request_id=group_id, source_track_id=caption.track_id,
                capture_epoch=caption.capture_epoch, run_id=run_id,
                window_start_ns=window[0], window_end_ns=window[1],
                owned_start_ns=caption.start_ns, owned_end_ns=caption.end_ns,
                metadata={"parent_id": parent_id, "source_revision": caption.source_revision,
                          "overlap_regions": tuple(regions), "deadline_monotonic": deadline},
            )

    def _children(self, result, parent):
        request, caption = result.request, parent.caption
        regions = request.metadata["overlap_regions"]
        children = []
        lineage = dict(source_track_id=caption.track_id, capture_epoch=caption.capture_epoch,
                       separation_group_id=request.request_id, parent_segment_ids=(caption.segment_id,),
                       delivery_class="overlap_correction", source_revision=1,
                       target_language=caption.target_language, source_language=caption.source_language,
                       text_operation=caption.text_operation)

        def add(words, track, lane, speaker=None):
            text = "".join(word.text for word in words).strip()
            if not text or len(text) > 12000:
                raise ValueError("Invalid separated caption text")
            start, end = min(word.start_time_ns for word in words), max(word.end_time_ns for word in words)
            if start < caption.start_ns or end > caption.end_ns or end <= start:
                raise ValueError("Replacement words exceed their parent")
            children.append(CaptionSegment(
                segment_id=f"{request.request_id}:caption-{len(children) + 1}",
                track_id=track, lane_id=lane, start_ns=start, end_ns=end,
                source_text=text, speaker_id=speaker, **lineage))

        # Crossing source words cannot be represented by inventing a clipped
        # transcript boundary. Keep the complete mixed caption for this case.
        for lane in result.lanes:
            if any(_inside_overlap(word, regions) for word in lane.boundary_words):
                raise ValueError("Separated word crosses parent boundary")
            # Separate disjoint overlap regions into distinct captions. Joining
            # their words would create a long cue across original context that
            # must remain visible between those regions.
            for region in _merged_regions(tuple(tuple(item) for item in regions)):
                words = tuple(word for word in lane.hypothesis.words
                              if word.end_time_ns > word.start_time_ns and _inside_overlap(word, (region,)))
                if not words:
                    raise ValueError("Both separated lanes need timed speech in every overlap region")
                add(words, lane.track_id, lane.lane_id)
        if len(result.lanes) != 2 or result.lanes[0].lane_id == result.lanes[1].lane_id:
            raise ValueError("Two distinct separation lanes required")

        # Preserve all original non-overlap words, including intervening speech.
        # Their translations are generated afresh; a whole parent's translation
        # must not be copied onto a fragment or onto both separated people.
        pending = []
        for word in parent.words:
            if _inside_overlap(word, regions):
                if pending:
                    add(pending, f"{caption.track_id}:sep:{request.request_id}:context",
                        f"context:{request.request_id}", caption.speaker_id)
                    pending = []
            else:
                pending.append(word)
        if pending:
            add(pending, f"{caption.track_id}:sep:{request.request_id}:context",
                f"context:{request.request_id}", caption.speaker_id)
        if len(children) > 12:
            raise ValueError("Too many replacement clauses")
        return tuple(children)

    async def _translate(self, child, language, provider, deadline, generation):
        remaining = (deadline - time.monotonic()) * 1000
        if remaining <= 0:
            raise TimeoutError
        source_language = self.session.config.source_language
        if not source_language or source_language.lower() in ("auto", "und", "unknown"):
            source_language = language or child.source_language or "auto"
        child = replace(child, source_language=source_language,
                        text_operation=text_operation_for(source_language, child.target_language))
        request = TranslationRequest(child.segment_id, 1, child.source_text,
                                     source_language,
                                     child.target_language, budget_ms=min(remaining, 60000),
                                     deadline_monotonic=deadline)
        text = ""
        async with _closing_stream(provider.stream(request)) as stream:
            async for chunk in stream:
                text += chunk.text
                if len(text) > 24000:
                    raise ProviderError("invalid_output", "Correction exceeds output limit")
                if chunk.completed:
                    if not text.strip():
                        raise ProviderError("invalid_output", "Empty correction translation")
                    self.counts["overlap_translations_completed"] += 1
                    return replace(child, translation=CaptionTranslation(1, generation, child.target_language, text))
        raise ProviderError("incomplete", "Correction translation did not complete")

    async def _on_result(self, result):
        if isinstance(result, dict):
            if result.get("kind", "").startswith("overlap."):
                await self.session._emit(result["kind"], **{key: value for key, value in result.items() if key != "kind"})
            return
        request = result.request
        parent_id = request.metadata["parent_id"]
        parent = self.parents.get(parent_id)
        if self._closed or parent is None:
            await self._skip("parent_expired", parent_id=parent_id)
            return
        current = self.session.store.get_segment(parent_id)
        if (current is None or current.superseded_by
                or current.source_revision != request.metadata["source_revision"]):
            await self._skip("parent_changed", parent_id=parent_id)
            return
        deadline = request.metadata["deadline_monotonic"]
        generation = self.session._translator.generation
        primary_provider = self.session._translator.provider
        provider = AdmittedTranslationProvider(
            getattr(primary_provider, "provider", primary_provider), self.admission, secondary=True)
        try:
            children = self._children(result, replace(parent, caption=current))
            async with asyncio.timeout_at(deadline):
                translated = []
                for child in children:
                    translated.append(await self._translate(child, parent.language, provider, deadline, generation))
            if (self.session._translator.generation != generation
                    or self.session._translator.provider is not primary_provider or self._closed):
                await self._skip("provider_changed", parent_id=parent_id)
                return
            bounds = self.stream.buffer_bounds
            if bounds is None or bounds[:3] != (request.source_track_id, request.capture_epoch, request.run_id):
                await self._skip("run_changed", parent_id=parent_id)
                return
            current = self.session.store.get_segment(parent_id)
            if (current is None or current.superseded_by
                    or current.source_revision != request.metadata["source_revision"]):
                await self._skip("parent_changed", parent_id=parent_id)
                return
            # A late confirmed identity belongs only to unchanged source-word
            # context, never automatically to either separated voice. Refresh
            # it after translation, when the latest parent patch is known.
            translated = [replace(child, speaker_id=current.speaker_id)
                          if child.lane_id.startswith("context:") else child for child in translated]
            group = CaptionGroupReplacement(
                request.request_id, 1, current.track_id, current.capture_epoch,
                current.start_ns, current.end_ns,
                (CaptionParentRef(parent_id, current.source_revision),), tuple(translated))
            payload = asdict(group)
            if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")) > self.config.max_event_bytes:
                await self._skip("group_event_too_large", parent_id=parent_id)
                return
            accepted = self.session.store.replace_group(group)
            if accepted != "applied":
                await self._skip("parent_changed", parent_id=parent_id)
                return
            await self.session._emit("caption.group_replaced", **payload)
            refinement = getattr(self.session, "_context_refinement", None)
            if refinement is not None:
                for child in sorted(translated, key=lambda item: (item.start_ns, item.end_ns, item.segment_id)):
                    refinement.register_caption(child, child.source_language or parent.language or "auto")
                for child in translated:
                    refinement.on_translation_completed(child.segment_id)
            self.counts["overlap_groups_completed"] += 1
            await self.session._emit("overlap.completed", parent_id,
                group_id=request.request_id, delivery_class="overlap_correction",
                correction_budget_ms=self.config.correction_budget_ms,
                source_age_ms=(time.monotonic() - parent.anchor) * 1000,
                timing_scope=self.session._correction_timing_scope(),
                separation_identity=result.separation_identity,
                speaker_identity_verified=False)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            await self._skip("correction_deadline", parent_id=parent_id)
        except Exception as exc:
            await self._skip("correction_" + getattr(exc, "category", type(exc).__name__), parent_id=parent_id)

    async def drain(self, timeout_s):
        if not await self.stream.drain(timeout_s):
            await self._skip("correction_drain_timeout")

    async def close(self):
        self._closed = True
        await self.stream.close()
        self.parents.clear()
        self.regions.clear()
        self.attempted.clear()
