"""Paired rolling context with a conservative UTF-8 byte token upper bound."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextEntry:
    segment_id: str
    source: str
    translation: str
    speaker_id: str | None = None


class ContextManager:
    def __init__(self, max_segments: int = 3, max_tokens: int = 1500) -> None:
        self.max_segments, self.max_tokens = max_segments, max_tokens
        self.entries: list[ContextEntry] = []

    def replace(self, entries: list[ContextEntry]) -> None:
        self.entries = []
        budget = self.max_tokens
        for entry in reversed(entries):
            # Generic backend tokenizers differ. Bytes are a conservative budget,
            # not a claim of exact token counting. Drop pairs together.
            cost = len((entry.source + entry.translation + (entry.speaker_id or "")).encode("utf-8")) + 32
            if len(self.entries) >= self.max_segments or cost > budget:
                break
            self.entries.insert(0, entry)
            budget -= cost

    def snapshot(self) -> tuple[list[str], list[str]]:
        return ([f"[{e.speaker_id}] {e.source}" if e.speaker_id else e.source for e in self.entries],
                [e.translation for e in self.entries])
