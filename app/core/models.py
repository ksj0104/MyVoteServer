"""Bounded wire schema. ASR text is cumulative within one utterance, not a delta."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

Identifier = Annotated[str, StringConstraints(min_length=1, max_length=128, pattern=r"^[\w.:-]+$")]
Language = Annotated[str, StringConstraints(min_length=2, max_length=32, pattern=r"^[A-Za-z][A-Za-z0-9_-]+$")]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ASRToken(WireModel):
    text: str = Field(min_length=1, max_length=1000)
    start: float | None = Field(None, ge=0)
    end: float | None = Field(None, ge=0)
    confidence: float | None = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def ordered_times(self):
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("Token end precedes start")
        return self


class ASREvent(WireModel):
    type: Literal["asr_partial", "asr_final"]
    session_id: Identifier | None = None
    sequence: int = Field(ge=0, strict=True)
    text: str = Field(max_length=32000)
    language: Language | None = None
    target_language: Language | None = None
    is_final: bool = False
    timestamp: float | None = Field(None, ge=0)
    tokens: list[ASRToken] = Field(default_factory=list, max_length=4096)
    pause_ms: float | None = Field(None, ge=0, le=60000)
    speaker_id: Identifier | None = None
    utterance_id: Identifier | None = None

    @model_validator(mode="after")
    def normalize_final(self):
        self.is_final = self.is_final or self.type == "asr_final"
        return self


class TranslationOptions(WireModel):
    model: str | None = Field(None, min_length=1, max_length=200)
    temperature: float | None = Field(None, ge=0, le=0.2)
    max_context_segments: int | None = Field(None, ge=0, le=10)


class StreamingOptions(WireModel):
    max_latency_ms: int | None = Field(None, ge=100, le=20000)
    stability_threshold: float | None = Field(None, ge=0, le=1)


class SessionConfig(WireModel):
    source_language: Language = "en"
    target_language: Language = "ko"
    translation: TranslationOptions = Field(default_factory=TranslationOptions)
    streaming: StreamingOptions = Field(default_factory=StreamingOptions)


class GlossaryUpdate(WireModel):
    entries: dict[str, str] = Field(max_length=1000)

    @field_validator("entries")
    @classmethod
    def bounded_entries(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not k.strip() or len(k) > 128 or not v.strip() or len(v) > 256 for k, v in value.items()):
            raise ValueError("Glossary entries must have nonempty keys <=128 and values <=256 characters")
        if sum(len(k) + len(v) for k, v in value.items()) > 64000:
            raise ValueError("Glossary too large")
        return value
