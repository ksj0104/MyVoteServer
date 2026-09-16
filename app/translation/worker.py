"""Immutable work descriptions and worker helpers; no session state ownership."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import logging
import time
from types import MappingProxyType

from .backend import TranslationError, TranslationResult


@dataclass(frozen=True)
class TranslationJob:
    session_id: str
    generation_id: int | str
    sequence: int
    source_revision: int
    segment_id: str
    source: str
    source_language: str
    target_language: str
    source_context: tuple[str, ...] = ()
    translation_context: tuple[str, ...] = ()
    glossary: Mapping[str, str] = field(default_factory=dict)
    final: bool = False
    created_at: float = field(default_factory=time.monotonic)
    model: str | None = None
    temperature: float | None = None

    def __post_init__(self):
        for value, limit in ((self.session_id, 128), (self.segment_id, 256), (self.source, 32000),
                             (self.source_language, 32), (self.target_language, 32)):
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError("Invalid translation job text or identifier")
        if (type(self.generation_id) not in (int, str) or isinstance(self.generation_id, int) and self.generation_id < 0
                or isinstance(self.generation_id, str) and (not self.generation_id.strip() or len(self.generation_id) > 256)):
            raise ValueError("Invalid generation identifier")
        if (type(self.sequence) is not int or self.sequence < 0
                or type(self.source_revision) is not int or self.source_revision < 1):
            raise ValueError("Sequence must be nonnegative and source revision positive")
        if type(self.final) is not bool:
            raise ValueError("final must be boolean")
        if type(self.created_at) not in (int, float) or not math.isfinite(self.created_at):
            raise ValueError("created_at must be a finite monotonic timestamp")
        for name in ("source_context", "translation_context"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) > 32 or any(
                    not isinstance(item, str) or not item.strip() or len(item) > 32000 for item in value):
                raise ValueError("Invalid translation context")
            object.__setattr__(self, name, tuple(value))
        if not isinstance(self.glossary, Mapping) or len(self.glossary) > 1000 or any(
                not isinstance(key, str) or not key.strip() or len(key) > 512
                or not isinstance(value, str) or not value.strip() or len(value) > 512
                for key, value in self.glossary.items()):
            raise ValueError("Invalid translation glossary")
        object.__setattr__(self, "glossary", MappingProxyType(dict(self.glossary)))
        total_chars = (len(self.source) + sum(map(len, self.source_context))
                       + sum(map(len, self.translation_context))
                       + sum(len(key) + len(value) for key, value in self.glossary.items()))
        if total_chars > 160000:
            raise ValueError("Translation job exceeds its bounded source/context budget")


def record_metric(metrics, name, value=1, *, observation=False):
    if metrics is None:
        return
    try:
        if callable(metrics):
            metrics(name, value)
        else:
            getattr(metrics, "observe" if observation else "increment")(name, value)
    except Exception:
        # Diagnostics must not disable translation; make the failure observable
        # without including exception strings that may contain private data.
        logging.getLogger(__name__).warning("translation_metrics_hook_failed")


async def translate_job(backend, job):
    result = await backend.translate(
        job.source, job.source_language, job.target_language, job.source_context,
        job.translation_context, job.glossary, model=job.model, temperature=job.temperature)
    if (not isinstance(result, TranslationResult) or not isinstance(result.text, str) or not result.text.strip()
            or type(result.latency_ms) not in (int, float) or not math.isfinite(result.latency_ms)
            or result.latency_ms < 0):
        raise TranslationError("invalid_backend_result")
    return result
