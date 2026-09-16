"""Strict local runtime profiles for an explicitly calibrated speaker setup.

A profile supplies complete settings and exact model/frontend identities. It
does not certify accuracy, retrieve models, or apply settings by itself. The
runtime must compare both identities with the adapters it actually loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, get_type_hints

from .speaker_stream import SpeakerStreamConfig
from .speakers import TrackerConfig


PROFILE_SCHEMA = "myvote.speaker_profile"
SCHEMA_VERSION = 1
MAX_PROFILE_BYTES = 16 * 1024
MAX_IDENTITY_CHARACTERS = 4096
_REQUIRED_FIELDS = frozenset({"schema", "schema_version", "tracker", "stream",
                              "embedding_identity", "segmentation_identity"})


@dataclass(frozen=True)
class SpeakerRuntimeProfile:
    tracker: TrackerConfig
    stream: SpeakerStreamConfig
    embedding_identity: str
    segmentation_identity: str
    # Informational provenance only; this field does not affect runtime settings.
    metadata: Any = None
    # Digest of the exact bytes parsed by load_speaker_profile, never a reread.
    source_sha256: str | None = None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    return value


def _reject_constant(token):
    raise ValueError(f"nonstandard JSON number: {token}")


def _identity(value, name):
    if (not isinstance(value, str) or not 1 <= len(value) <= MAX_IDENTITY_CHARACTERS
            or not value.strip() or not value.isprintable()):
        raise ValueError(f"{name} must contain 1..4096 printable characters")
    # No trimming, prefix rewriting, path interpretation, or case normalization.
    return value


def _config(value, config_type, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a complete settings object")
    expected = {field.name for field in fields(config_type)}
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise ValueError(f"{name} fields must be complete: missing={missing}, unknown={unknown}")
    hints = get_type_hints(config_type)
    normalized = {}
    for key, item in value.items():
        kind = hints[key]
        if kind is int:
            if type(item) is not int:
                raise ValueError(f"{name}.{key} must be an integer, not a bool or float")
            normalized[key] = item
        elif kind is float:
            if type(item) not in (int, float):
                raise ValueError(f"{name}.{key} must be a number, not a bool")
            try:
                number = float(item)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{name}.{key} must be a finite number") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name}.{key} must be a finite number")
            normalized[key] = number
        else:
            # A future settings type needs an explicit schema decision instead
            # of silently broadening what a version-1 profile can configure.
            raise ValueError(f"unsupported version-1 settings type: {name}.{key}")
    # Keep the existing settings' range and relationship rules authoritative.
    return config_type(**normalized)


def load_speaker_profile(path: str | Path) -> SpeakerRuntimeProfile:
    """Read <=16 KiB of UTF-8 JSON and reject incomplete/ambiguous profiles.

    ``tracker`` and ``stream`` must include every current dataclass field; no
    implicit defaults are filled. Metadata can be any finite JSON value and
    stays informational. Duplicate keys are rejected at every nesting level,
    including metadata. Nonfinite exponent overflow is rejected during parsing.
    Filesystem errors propagate; all malformed profiles raise ValueError.
    """
    source = Path(path).expanduser()
    with source.open("rb") as stream:
        raw = stream.read(MAX_PROFILE_BYTES + 1)
    if not raw or len(raw) > MAX_PROFILE_BYTES:
        raise ValueError("speaker profile must be nonempty and at most 16 KiB")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_float=_finite_float, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"invalid speaker profile JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("speaker profile must be a JSON object")
    missing = _REQUIRED_FIELDS - set(value)
    unknown = set(value) - _REQUIRED_FIELDS - {"metadata"}
    if missing or unknown:
        raise ValueError(f"speaker profile fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if value["schema"] != PROFILE_SCHEMA:
        raise ValueError("unsupported speaker profile schema")
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("speaker profile schema_version must be integer 1")
    return SpeakerRuntimeProfile(
        tracker=_config(value["tracker"], TrackerConfig, "tracker"),
        stream=_config(value["stream"], SpeakerStreamConfig, "stream"),
        embedding_identity=_identity(value["embedding_identity"], "embedding_identity"),
        segmentation_identity=_identity(value["segmentation_identity"], "segmentation_identity"),
        metadata=value.get("metadata"),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
