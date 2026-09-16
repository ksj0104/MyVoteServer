"""Small transport-independent contract for the text translation gateway."""

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class TranslationResult:
    text: str
    latency_ms: float


class TranslationError(Exception):
    """A closed diagnostic code, never an upstream response or source text."""

    def __init__(self, code: str, *, retryable: bool = False):
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class TranslationBackend(Protocol):
    async def translate(
        self, source: str, source_language: str, target_language: str,
        source_context: Sequence[str] = (), translation_context: Sequence[str] = (),
        glossary: Mapping[str, str] | None = None, *, model: str | None = None,
        temperature: float | None = None,
    ) -> TranslationResult: ...
