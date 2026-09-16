"""READ/WRITE preferences never overrule source completeness."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str
    segment: object | None = None


class TranslationPolicy(Protocol):
    async def decide(self, candidate, *, stability_score, stable_token_count,
                     age_ms=0, pause_ms=None, is_final=False, prefix_conflict=False) -> PolicyDecision:
        ...


class ReadWritePolicy:
    def __init__(self, stability_threshold=0.75, min_stable_tokens=3,
                 min_segment_tokens=5, target_segment_tokens=10,
                 max_latency_ms=1500, pause_threshold_ms=500):
        if not 0 <= stability_threshold <= 1:
            raise ValueError("Invalid stability threshold")
        if min(min_stable_tokens, min_segment_tokens, target_segment_tokens,
               max_latency_ms, pause_threshold_ms) < 1:
            raise ValueError("Policy limits must be positive")
        self.stability_threshold = stability_threshold
        self.min_stable_tokens = min_stable_tokens
        self.min_segment_tokens = min_segment_tokens
        self.target_segment_tokens = target_segment_tokens
        self.max_latency_ms = max_latency_ms
        self.pause_threshold_ms = pause_threshold_ms

    async def decide(self, candidate, *, stability_score, stable_token_count,
                     age_ms=0, pause_ms=None, is_final=False, prefix_conflict=False):
        if prefix_conflict:
            return PolicyDecision("READ", "committed_prefix_conflict")
        if candidate is None or not candidate.complete:
            return PolicyDecision("READ", "incomplete_source")
        if not self.stability_threshold <= stability_score <= 1 or stable_token_count < candidate.token_count:
            return PolicyDecision("READ", "unstable_source")
        if candidate.strong_boundary:
            return PolicyDecision("WRITE", "complete_boundary", candidate)
        if stable_token_count < self.min_stable_tokens:
            return PolicyDecision("READ", "short_stable_prefix")
        preferred = candidate.token_count >= self.target_segment_tokens
        due = (is_final or age_ms >= self.max_latency_ms
               or (pause_ms is not None and pause_ms >= self.pause_threshold_ms))
        if preferred or (candidate.token_count >= self.min_segment_tokens and due):
            return PolicyDecision("WRITE", "complete_source", candidate)
        return PolicyDecision("READ", "collecting_complete_context")
