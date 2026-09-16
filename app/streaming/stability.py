"""Local, bounded ASR stability tracking; audio timestamps are not clocks.

N distinct hypotheses are needed for an interim stable prefix. Confidence and
pause are optional evidence, not reasons to force an unstable token stable.
ASR finality settles source recognition only; it says nothing about meaning.
"""
from collections import deque
from dataclasses import dataclass
import math
import time

from app.utils.text import common_prefix_length, tokenize, token_char_end


@dataclass(frozen=True)
class StabilityResult:
    text: str
    sequence: int
    stable_token_count: int
    stable_char_end: int
    score: float
    confidence: float | None
    age_ms: float
    prefix_conflict: bool
    reconcile_required: bool
    is_final: bool


class StabilityTracker:
    def __init__(self, window_size=4, threshold=0.75, *, clock=time.monotonic,
                 age_horizon_ms=1200, pause_threshold_ms=500,
                 max_source_chars=8000, agreement_weight=0.35,
                 confidence_weight=0.30, age_weight=0.20, pause_weight=0.15,
                 hold_last_token=False):
        if not 2 <= window_size <= 32 or not 0 <= threshold <= 1:
            raise ValueError("Invalid stability history or threshold")
        if age_horizon_ms <= 0 or pause_threshold_ms <= 0 or max_source_chars < 1:
            raise ValueError("Stability limits must be positive")
        self.window_size = window_size
        self.threshold = threshold
        self.clock = clock
        self.age_horizon_ms = age_horizon_ms
        self.pause_threshold_ms = pause_threshold_ms
        self.max_source_chars = max_source_chars
        self.hold_last_token = bool(hold_last_token)
        self.weights = (agreement_weight, confidence_weight, age_weight, pause_weight)
        if any(not math.isfinite(value) or value < 0 for value in self.weights) or not any(self.weights):
            raise ValueError("Stability weights must be nonnegative and have a positive sum")
        self.reset()

    def reset(self):
        self._history = deque(maxlen=self.window_size)
        self._tokens = ()
        self._seen = []
        self._text = ""
        self._sequence = -1
        self._final = False
        self._confidences = ()
        self._pause_ms = None
        self._committed_text = ""
        self._last_time = None
        self._result = None

    def _now(self):
        value = float(self.clock())
        if not math.isfinite(value):
            raise ValueError("Monotonic clock must be finite")
        # Also make injected test clocks conservative when moved backwards.
        self._last_time = value if self._last_time is None else max(value, self._last_time)
        return self._last_time

    @staticmethod
    def _probability(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) and 0 <= value <= 1 else None

    def observe(self, text, sequence, *, is_final=False, confidences=None, pause_ms=None):
        if not isinstance(text, str) or len(text) > self.max_source_chars:
            raise ValueError("Source exceeds tracker character limit")
        if type(sequence) is not int or sequence <= self._sequence:
            raise ValueError("ASR sequence must strictly increase")
        now = self._now()
        tokens = tokenize(text)
        values = tuple(token.text for token in tokens)
        retained = common_prefix_length((tuple(token.text for token in self._tokens), values))
        self._seen = self._seen[:retained] + [now] * (len(tokens) - retained)
        self._history.append(values)
        self._tokens, self._text, self._sequence = tokens, text, sequence
        self._final = bool(is_final)
        if confidences is None:
            self._confidences = (None,) * len(tokens)
        elif isinstance(confidences, (int, float)):
            self._confidences = (self._probability(confidences),) * len(tokens)
        else:
            supplied = tuple(confidences)
            self._confidences = (tuple(map(self._probability, supplied))
                                 if len(supplied) == len(tokens) else (None,) * len(tokens))
        self._pause_ms = (float(pause_ms) if isinstance(pause_ms, (int, float))
                          and not isinstance(pause_ms, bool) and math.isfinite(pause_ms)
                          and pause_ms >= 0 else None)
        return self._evaluate(now)

    def refresh(self):
        """Reevaluate age without counting a timer tick as another ASR result."""
        if self._sequence < 0:
            return None
        return self._evaluate(self._now())

    def _evaluate(self, now):
        protected_end = len(self._committed_text)
        conflict = (not self._text.startswith(self._committed_text)
                    or bool(protected_end and protected_end not in {token.end for token in self._tokens}))
        agreement = common_prefix_length(tuple(self._history)) if len(self._history) == self.window_size else 0
        available = len(self._tokens) if self._final else agreement
        if (self.hold_last_token and not self._final and self._tokens
                and available == len(self._tokens) and self._tokens[-1].end == len(self._text)
                and any(char.isalnum() for char in self._tokens[-1].text)):
            # A repeated "hosp" might still become "hospital". Wait for a
            # lexical delimiter or explicit final, not a dictionary guess.
            available -= 1
        scores = []
        stable = 0
        agreement_weight, confidence_weight, age_weight, pause_weight = self.weights
        for index in range(available):
            age = max(0.0, (now - self._seen[index]) * 1000)
            weighted = agreement_weight + age_weight * min(1.0, age / self.age_horizon_ms)
            weight = agreement_weight + age_weight
            confidence = self._confidences[index]
            if confidence is not None:
                weighted += confidence_weight * confidence
                weight += confidence_weight
            if self._pause_ms is not None:
                weighted += pause_weight * min(1.0, self._pause_ms / self.pause_threshold_ms)
                weight += pause_weight
            score = 1.0 if self._final else weighted / weight if weight else 0.0
            scores.append(score)
            if stable == index and score + 1e-12 >= self.threshold:
                stable += 1
        if conflict and not self._final:
            stable = 0
        if not conflict:
            committed_count = sum(token.end <= len(self._committed_text) for token in self._tokens)
            stable = max(stable, committed_count)
        known = [value for value in self._confidences[:available] if value is not None]
        self._result = StabilityResult(
            text=self._text, sequence=self._sequence, stable_token_count=stable,
            stable_char_end=token_char_end(self._tokens, stable),
            score=min(scores[:stable] or scores or [0.0]),
            confidence=sum(known) / len(known) if known else None,
            age_ms=max(0.0, (now - self._seen[0]) * 1000) if self._seen else 0.0,
            prefix_conflict=conflict, reconcile_required=conflict and self._final,
            is_final=self._final,
        )
        return self._result

    def commit(self, char_end, *, allow_final_reconcile=False):
        """Protect an exact source prefix; replacement requires an explicit final."""
        if self._result is None or type(char_end) is not int:
            raise ValueError("No observed source to commit")
        if char_end not in {0, *(token.end for token in self._tokens)}:
            raise ValueError("Commit must end on a source token boundary")
        if not 0 <= char_end <= self._result.stable_char_end:
            raise ValueError("Cannot commit beyond the stable source")
        if self._result.prefix_conflict or char_end < len(self._committed_text):
            if not (allow_final_reconcile and self._final):
                raise ValueError("Committed source replacement requires explicit final reconciliation")
        self._committed_text = self._text[:char_end]
        return self._evaluate(self._now())
