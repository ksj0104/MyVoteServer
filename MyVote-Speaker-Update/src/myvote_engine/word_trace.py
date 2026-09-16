"""Bounded optional evidence of the words that actually formed a source cue.

This records existing ASR output. It does not align, clip, split or retime words,
and a successful trace does not establish transcript or speaker accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .asr import TimedWord


CAPABILITY = "caption_words_v1"
MAX_WORDS = 128
MAX_EVENT_BYTES = 65536
MAX_SESSION_BYTES = 1048576


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8"))


def _integer(value, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError("Invalid word trace integer")
    return value


@dataclass(frozen=True)
class _Word:
    index: int
    text: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class _Trace:
    schema: str
    schema_version: int
    origin: str
    source_revision: int
    stabilizer_revision: int
    commit_reason: str
    window_start_ns: int
    window_end_ns: int
    emit_start_ns: int
    words: tuple[_Word, ...]
    boundary_eligible: bool
    boundary_ineligible_reasons: tuple[str, ...]


class WordTraceRecorder:
    """One session's bounded trace budget; omission never mutates a caption.

    The caller emits the returned payload inside the same caption.source event.
    Counters contain no text. There is no retained history or new model call.
    """

    def __init__(self, *, max_event_bytes=MAX_EVENT_BYTES,
                 max_session_bytes=MAX_SESSION_BYTES):
        if (type(max_event_bytes) is not int or not 256 <= max_event_bytes <= MAX_EVENT_BYTES
                or type(max_session_bytes) is not int or not 0 <= max_session_bytes <= MAX_SESSION_BYTES):
            raise ValueError("Word trace bounds exceed supported limits")
        self.max_event_bytes = max_event_bytes
        self.max_session_bytes = max_session_bytes
        self.counts = {"word_trace_emitted": 0, "word_trace_bytes": 0,
            "word_trace_omitted_invalid": 0, "word_trace_omitted_word_limit": 0,
            "word_trace_omitted_event_limit": 0, "word_trace_omitted_session_limit": 0}

    def _omit(self, reason):
        self.counts["word_trace_omitted_" + reason] += 1
        return None

    def build(self, caption, words, *, stabilizer_revision, window_start_ns,
              window_end_ns, emit_start_ns, commit_reason="unspecified"):
        try:
            if not isinstance(words, tuple) or not words:
                return self._omit("invalid")
            if len(words) > MAX_WORDS:
                return self._omit("word_limit")
            revision = _integer(caption["source_revision"], 1)
            _integer(stabilizer_revision, 1)
            if commit_reason not in ("agreement", "hypothesis", "flush", "unspecified"):
                return self._omit("invalid")
            _integer(window_start_ns)
            _integer(window_end_ns, window_start_ns + 1)
            _integer(emit_start_ns, window_start_ns)
            if emit_start_ns > window_end_ns:
                return self._omit("invalid")
            start, end = _integer(caption["start_ns"]), _integer(caption["end_ns"], 1)
            if end <= start or not isinstance(caption["text"], str):
                return self._omit("invalid")
            captured, reasons = [], set()
            previous_start = -1
            previous_end = -1
            for index, word in enumerate(words):
                if not isinstance(word, TimedWord) or not word.text.strip() or len(word.text) > 24000:
                    return self._omit("invalid")
                first, last = _integer(word.start_time_ns), _integer(word.end_time_ns)
                if first < previous_start or last < first:
                    return self._omit("invalid")
                if first == last:
                    reasons.add("zero_duration_word")
                if first < previous_end:
                    reasons.add("overlapping_word_times")
                if first < start or last > end:
                    reasons.add("word_outside_caption")
                if first < window_start_ns or last > window_end_ns:
                    reasons.add("word_outside_commit_window")
                previous_start, previous_end = first, max(previous_end, last)
                captured.append(_Word(index, word.text, first, last))
            if "".join(word.text for word in captured).strip() != caption["text"]:
                return self._omit("invalid")
            trace = asdict(_Trace("myvote.committed_words", 1, "stable_word_assembler",
                revision, stabilizer_revision, commit_reason, window_start_ns, window_end_ns,
                emit_start_ns, tuple(captured), not reasons, tuple(sorted(reasons))))
            size = _size({**caption, "word_trace": trace})
            if size > self.max_event_bytes:
                return self._omit("event_limit")
            extra = size - _size(caption)
            if self.counts["word_trace_bytes"] + extra > self.max_session_bytes:
                return self._omit("session_limit")
            self.counts["word_trace_emitted"] += 1
            self.counts["word_trace_bytes"] += extra
            return trace
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
            return self._omit("invalid")
