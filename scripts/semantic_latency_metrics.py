"""Observe first-word preview latency without changing source ownership or budgets.

The clock starts just before a successful server ``transcript.preview`` emit,
not at audio capture, ASR stabilization, or the last word of a sentence. This
measures server events, not delivery acknowledgements or client rendering.

Word identity is canonical text plus EXACT source timestamps. A timestamp
revision is deliberately not guessed to be the same word: unmatched words use
their existing stable-ready clock, explicitly labelled as a fallback. Preview
history and caption anchors are independently bounded to 256 entries. Truncated
or evicted words can therefore also use that fallback. No raw text is retained
in the additional history, and no source deadline, unit, or PCM is modified.
"""

from collections import OrderedDict
import hashlib
import math
import time
import unicodedata


MAX_OBSERVED_WORDS = 256
MAX_CAPTION_ANCHORS = 256
PREVIEW_CHAR_LIMIT = 4000
LATENCY_SCOPE = "first_source_preview_to_server_translation_event"


def _word_key(word):
    canonical = " ".join(unicodedata.normalize("NFKC", word.text).split()).casefold()
    return (word.start_time_ns, word.end_time_ns,
            hashlib.sha256(canonical.encode("utf-8")).digest())


def _visible_word_keys(preview):
    """Mirror the upstream preview's suffix clipping, excluding partial words."""
    words = tuple(word for index, word in enumerate(preview.update.stable_words)
                  if index not in preview.retired) + tuple(preview.update.provisional_words)
    raw = "".join(word.text for word in words)
    right = len(raw.rstrip())
    left = max(len(raw) - len(raw.lstrip()), right - PREVIEW_CHAR_LIMIT)
    offset = 0
    keys = []
    for word in words:
        text = word.text
        start = offset + len(text) - len(text.lstrip())
        end = offset + len(text.rstrip())
        if left <= start < end <= right:
            keys.append(_word_key(word))
        offset += len(text)
    return tuple(keys)


def install_latency_metrics(semantic_module, *, target_latency_s=2):
    """Wrap the currently installed pipeline; preserve all lifecycle methods."""
    if (type(target_latency_s) not in (int, float) or not math.isfinite(target_latency_s)
            or not 0 < target_latency_s <= 120):
        raise ValueError("Target latency must be finite positive seconds, at most 120")
    target_ms = target_latency_s * 1000
    metadata = {
        "semantic_latency_metrics": "first-preview-word-v1",
        "semantic_latency_scope": LATENCY_SCOPE,
        "semantic_latency_target_ms": target_ms,
        "semantic_latency_word_match": "canonical-text-exact-timestamps",
    }
    original = semantic_module.SemanticCaptionPipeline
    marker = "_myvote_first_preview_latency_v1"
    if getattr(original, marker, False):
        if original._myvote_latency_target_ms != target_ms:
            raise ValueError("Latency metrics already installed with a different target")
        return metadata

    class FirstPreviewLatencyPipeline(original):
        _myvote_first_preview_latency_v1 = True
        _myvote_latency_target_ms = target_ms
        _myvote_latency_original = original
        _latency_clock = staticmethod(time.monotonic)

        def __init__(self, *args, **kwargs):
            self._latency_anchors = OrderedDict()
            super().__init__(*args, **kwargs)

        async def _preview(self, preview):
            before_revision = preview.revision
            first_seen_at = self._latency_clock()
            keys = _visible_word_keys(preview)
            retired = {_word_key(word) for index, word in enumerate(preview.update.stable_words)
                       if index in preview.retired}
            # Do not claim an observation on exceptions, cancellation, unchanged
            # payloads, or closed sessions. Upstream mutation semantics stay intact.
            result = await super()._preview(preview)
            if preview.revision == before_revision:
                return result
            previous = getattr(preview, "_latency_first_seen", ())
            observed = OrderedDict((key, value) for key, value in dict(previous).items()
                                   if key not in retired)
            current = set(keys)
            for key in keys:
                observed.setdefault(key, first_seen_at)
            # Prefer retaining the oldest still-visible word. Obsolete revision
            # history is evicted first; a crowded preview never resets its head.
            for key in tuple(observed):
                if len(observed) <= MAX_OBSERVED_WORDS:
                    break
                if key not in current:
                    del observed[key]
            while len(observed) > MAX_OBSERVED_WORDS:
                observed.popitem(last=True)
            preview._latency_first_seen = observed
            return result

        async def _source(self, result):
            first = result.units[0].payload
            observed = getattr(first.preview, "_latency_first_seen", {})
            anchor = observed.get(_word_key(first.word))
            fallback = anchor is None
            if fallback:
                anchor = first.commit.ready_at
            clause_id = await super()._source(result)
            if clause_id is not None:
                self._latency_anchors[clause_id] = (anchor, fallback)
                while len(self._latency_anchors) > MAX_CAPTION_ANCHORS:
                    self._latency_anchors.popitem(last=False)
            return clause_id

        def metrics(self, clause_id):
            result = super().metrics(clause_id)
            anchor = self._latency_anchors.get(clause_id)
            if result is None or anchor is None:
                return result
            first_seen_at, fallback = anchor
            age_ms = max(0, self._latency_clock() - first_seen_at) * 1000
            return dict(result,
                        semantic_first_word_age_ms=age_ms,
                        semantic_latency_target_ms=target_ms,
                        semantic_latency_exceeded=age_ms > target_ms,
                        semantic_latency_anchor_fallback=fallback)

    semantic_module.SemanticCaptionPipeline = FirstPreviewLatencyPipeline
    return metadata
