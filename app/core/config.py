"""Validated deployment limits. Client overrides may narrow, never bypass, these limits."""
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    streaming_api_key: SecretStr = SecretStr("")
    translation_base_url: str = "http://127.0.0.1:1234/v1"
    translation_api_key: SecretStr = SecretStr("lm-studio")
    translation_model: str = "google/gemma-4-26b-a4b"
    translation_timeout: float = Field(30, ge=0.05, le=120)
    translation_workers: int = Field(4, ge=1, le=32)
    translation_temperature: float = Field(0.1, ge=0, le=0.2)
    stability_history: int = Field(4, ge=2, le=8)
    stability_threshold: float = Field(0.75, ge=0, le=1)
    stability_agreement_weight: float = Field(0.35, ge=0, le=1)
    stability_confidence_weight: float = Field(0.30, ge=0, le=1)
    stability_age_weight: float = Field(0.20, ge=0, le=1)
    stability_pause_weight: float = Field(0.15, ge=0, le=1)
    hold_partial_tail: bool = True
    min_stable_tokens: int = Field(3, ge=1, le=32)
    min_segment_tokens: int = Field(5, ge=1, le=64)
    target_segment_tokens: int = Field(10, ge=1, le=128)
    max_segment_tokens: int = Field(24, ge=1, le=256)
    allow_oversize_complete: bool = True
    max_segment_latency_ms: int = Field(1500, ge=100, le=20000)
    max_buffer_age_ms: int = Field(20000, ge=1000, le=120000)
    pause_threshold_ms: int = Field(500, ge=100, le=5000)
    translation_debounce_ms: int = Field(150, ge=0, le=2000)
    translation_stability_count: int = Field(2, ge=1, le=8)
    translation_commit_delay_ms: int = Field(700, ge=0, le=10000)
    context_segments: int = Field(3, ge=0, le=10)
    max_context_tokens: int = Field(1500, ge=0, le=8000)
    max_sessions: int = Field(32, ge=1, le=512)
    session_inactivity_seconds: float = Field(1800, ge=1, le=86400)
    max_pending_jobs: int = Field(128, ge=1, le=4096)
    max_pending_per_session: int = Field(16, ge=1, le=256)
    max_utterances: int = Field(64, ge=2, le=1024)
    max_segments: int = Field(256, ge=4, le=2048)
    max_source_chars: int = Field(8000, ge=128, le=32000)
    max_session_chars: int = Field(64000, ge=1024, le=256000)
    max_translation_chars: int = Field(128000, ge=1024, le=512000)
    max_state_bytes: int = Field(65536, ge=1024, le=131072)
    max_snapshot_bytes: int = Field(524288, ge=524288, le=1048576)
    max_event_bytes: int = Field(65536, ge=1024, le=262144)
    max_events_per_second: int = Field(100, ge=1, le=1000)
    outgoing_queue_size: int = Field(64, ge=2, le=1024)
    raw_transcript_logging: bool = False

    @field_validator("translation_base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        from urllib.parse import urlsplit
        url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Use an HTTP(S) backend URL without credentials, query or fragment")
        return value.rstrip("/")

    @model_validator(mode="after")
    def ordered_limits(self):
        if not self.min_segment_tokens <= self.target_segment_tokens <= self.max_segment_tokens:
            raise ValueError("Segment token limits must be ordered")
        if self.max_segment_latency_ms > self.max_buffer_age_ms:
            raise ValueError("Decision latency must not exceed the buffer lifetime")
        if self.stability_agreement_weight + self.stability_age_weight <= 0:
            raise ValueError("Always-available stability evidence must have positive weight")
        return self
