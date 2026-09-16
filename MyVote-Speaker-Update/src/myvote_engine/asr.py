"""Timed-word LocalAgreement and an optional native Mac MLX Whisper adapter.

The stabilizer is model independent. The adapter runs one completed PCM window;
capture, rolling-window scheduling, VAD, and network serving belong upstream.
No model is downloaded automatically and no real ASR performance is asserted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib
from itertools import islice
import math
from pathlib import Path
import platform
import struct
from typing import Iterable, Literal
import unicodedata
import wave


NS = 1_000_000_000


@dataclass(frozen=True)
class TimedWord:
    text: str
    start_time_ns: int
    end_time_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("word text must be nonempty")
        times = (self.start_time_ns, self.end_time_ns)
        if any(isinstance(x, bool) or not isinstance(x, int) for x in times):
            raise ValueError("word times must be integer nanoseconds")
        if self.start_time_ns < 0 or self.end_time_ns < self.start_time_ns:
            raise ValueError("invalid word interval")


@dataclass(frozen=True)
class ASRHypothesis:
    window_start_ns: int
    window_end_ns: int
    words: tuple[TimedWord, ...]
    language: str | None = None

    def __post_init__(self) -> None:
        for value in (self.window_start_ns, self.window_end_ns):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("window times must be integer nanoseconds")
        if self.window_start_ns < 0 or self.window_end_ns <= self.window_start_ns:
            raise ValueError("invalid audio window")
        previous_start = self.window_start_ns
        for word in self.words:
            if (word.start_time_ns < previous_start or word.end_time_ns > self.window_end_ns):
                raise ValueError("words must be ordered and lie inside the audio window")
            previous_start = word.start_time_ns


@dataclass(frozen=True)
class TranscriptUpdate:
    segment_id: str
    revision: int
    source_state: Literal["provisional", "stable"]
    stable_words: tuple[TimedWord, ...]
    provisional_words: tuple[TimedWord, ...]
    reason: str

    @property
    def stable_text(self) -> str:
        return "".join(word.text for word in self.stable_words).strip()

    @property
    def provisional_text(self) -> str:
        return "".join(word.text for word in self.provisional_words).strip()

    @property
    def text(self) -> str:
        # Backends preserve leading spaces within words; do not add spaces to CJK.
        return "".join(word.text for word in self.stable_words + self.provisional_words).strip()


class UncommittedAudioLostError(ValueError):
    """The caller advanced its window beyond a word that is not stable yet."""


def _canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


class LocalAgreementStabilizer:
    """One live segment with an append-only stable prefix and revisable suffix.

    Stable here means agreement in this realtime pass, not an immutable transcript.
    Later manual/offline corrections belong in the host's revision reducer.
    ``revision`` is local to this instance; the host maps it to source_revision or
    seeds initial_revision and stable_words when resuming this same segment.
    End time must advance to count as new evidence. A repeated window cannot vote
    repeatedly. Call flush before discarding uncommitted audio at a forced boundary.
    """

    def __init__(self, segment_id: str, *, agreement_passes: int = 2,
                 timestamp_tolerance_ns: int = 250_000_000,
                 initial_revision: int = 0,
                 stable_words: Iterable[TimedWord] = ()) -> None:
        if not segment_id:
            raise ValueError("segment_id is required")
        if isinstance(agreement_passes, bool) or not isinstance(agreement_passes, int) or agreement_passes < 2:
            raise ValueError("agreement_passes must be an integer >= 2")
        if (isinstance(timestamp_tolerance_ns, bool) or not isinstance(timestamp_tolerance_ns, int)
                or timestamp_tolerance_ns < 0):
            raise ValueError("timestamp tolerance must be a nonnegative integer")
        if isinstance(initial_revision, bool) or not isinstance(initial_revision, int) or initial_revision < 0:
            raise ValueError("initial_revision must be a nonnegative integer")
        self.segment_id = segment_id
        self.agreement_passes = agreement_passes
        self.timestamp_tolerance_ns = timestamp_tolerance_ns
        self._revision = initial_revision
        self._stable = tuple(stable_words)
        if any(a.start_time_ns > b.start_time_ns for a, b in zip(self._stable, self._stable[1:])):
            raise ValueError("restored stable words must be ordered")
        self._pending: tuple[TimedWord, ...] = ()
        self._history: deque[tuple[TimedWord, ...]] = deque(maxlen=agreement_passes - 1)
        self._last_window_end = -1
        self._closed = False

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def closed(self) -> bool:
        return self._closed

    def _after_stable(self, words: tuple[TimedWord, ...]) -> tuple[TimedWord, ...]:
        if not self._stable:
            return words
        boundary = self._stable[-1].end_time_ns
        result = []
        for word in words:
            if word.end_time_ns <= boundary:
                continue
            # A redecoded boundary word may move a little later. Only discard it
            # when most of its duration is already owned by the stable prefix;
            # an adjacent repeated word ("go go") must survive.
            overlap_ns = max(0, boundary - word.start_time_ns)
            duration_ns = max(1, word.end_time_ns - word.start_time_ns)
            if (overlap_ns * 2 >= duration_ns
                    and word.end_time_ns <= boundary + self.timestamp_tolerance_ns):
                continue
            result.append(word)
        return tuple(result)

    def _agrees(self, left: TimedWord, right: TimedWord) -> bool:
        return (_canonical(left.text) == _canonical(right.text)
                and abs(left.start_time_ns - right.start_time_ns) <= self.timestamp_tolerance_ns
                and abs(left.end_time_ns - right.end_time_ns) <= self.timestamp_tolerance_ns)

    def _update(self, reason: str) -> TranscriptUpdate:
        self._revision += 1
        return TranscriptUpdate(self.segment_id, self._revision,
                                "provisional" if self._pending else "stable",
                                self._stable, self._pending, reason)

    def accept(self, hypothesis: ASRHypothesis) -> TranscriptUpdate | None:
        if self._closed:
            raise RuntimeError("segment is flushed; create a new stabilizer")
        if hypothesis.window_end_ns < self._last_window_end:
            raise ValueError("stale ASR hypothesis")
        if hypothesis.window_end_ns == self._last_window_end:
            return None
        if (self._pending and hypothesis.window_start_ns >
                self._pending[0].start_time_ns + self.timestamp_tolerance_ns):
            raise UncommittedAudioLostError("window dropped uncommitted audio; flush or retain the prefix")
        current = self._after_stable(hypothesis.words)
        previous = (self._stable, self._pending)
        histories = [self._after_stable(words) for words in self._history]
        common = 0
        if len(histories) == self.agreement_passes - 1:
            for index, word in enumerate(current):
                if not all(index < len(words) and self._agrees(word, words[index]) for words in histories):
                    break
                common += 1
        self._stable += current[:common]
        self._pending = current[common:]
        self._history.append(current)
        self._last_window_end = hypothesis.window_end_ns
        if previous == (self._stable, self._pending):
            return None
        return self._update("agreement" if common else "hypothesis")

    def flush(self) -> TranscriptUpdate | None:
        """Commit the latest suffix on stop/endpoint, without pretending agreement."""
        if self._closed:
            return None
        self._closed = True
        if not self._pending:
            return None
        self._stable += self._pending
        self._pending = ()
        self._history.clear()
        return self._update("flush")


class MlxWhisperWindowTranscriber:
    """Optional native Apple Silicon adapter for a prepared LOCAL model directory.

    Uses the official mlx-whisper waveform API with word_timestamps=True. Importing
    this module or constructing the adapter does not import MLX/numpy or download
    weights. The Mac runtime and actual model inference remain to be verified.
    Serialize calls for a model; mlx-whisper's model cache is process global.
    """

    def __init__(self, model_path: str | Path, *, language: str | None = None,
                 max_window_s: float = 30.0) -> None:
        if not math.isfinite(max_window_s) or not 1 / 16000 <= max_window_s <= 30:
            raise ValueError("max_window_s must be between one sample and 30 seconds")
        if language is not None:
            if not isinstance(language, str):
                raise ValueError("ASR language must be a string or None")
            language = language.strip().lower()
            if language in ("", "auto"):
                # Public CLI/wire callers may spell automatic detection as
                # 'auto'; mlx-whisper detects only when language is None.
                language = None
        self.model_path = Path(model_path).expanduser().resolve()
        self.language = language
        self.max_window_samples = math.floor(max_window_s * 16000)

    def transcribe_pcm(self, samples: Iterable[float], *, sample_rate: int = 16000,
                       window_start_ns: int = 0) -> ASRHypothesis:
        if sample_rate != 16000:
            raise ValueError("adapter requires mono 16 kHz PCM; resample upstream")
        if (isinstance(window_start_ns, bool) or not isinstance(window_start_ns, int)
                or window_start_ns < 0):
            raise ValueError("window_start_ns must be a nonnegative integer")
        bounded = tuple(islice(samples, self.max_window_samples + 1))
        if len(bounded) > self.max_window_samples:
            raise ValueError("PCM exceeds the configured maximum ASR window")
        waveform = tuple(float(x) for x in bounded)
        if not waveform or not all(math.isfinite(x) and -1 <= x <= 1 for x in waveform):
            raise ValueError("PCM must be a nonempty sequence of finite floats in [-1, 1]")
        if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
            raise RuntimeError("mlx-whisper adapter requires native Apple Silicon macOS")
        if not self.model_path.is_dir():
            raise FileNotFoundError("prepare a local MLX Whisper model directory before inference")
        try:
            numpy = importlib.import_module("numpy")
            mlx_whisper = importlib.import_module("mlx_whisper")
        except ImportError as exc:
            raise RuntimeError("install the optional Mac ASR dependencies in the Mac engine environment") from exc
        result = mlx_whisper.transcribe(
            numpy.asarray(waveform, dtype=numpy.float32),
            path_or_hf_repo=str(self.model_path), language=self.language,
            # None suppresses language announcements and per-window progress.
            # False still prints both in mlx-whisper's public API.
            task="transcribe", word_timestamps=True, verbose=None,
            temperature=0.0, condition_on_previous_text=False)
        duration_ns = len(waveform) * NS // sample_rate
        window_end_ns = window_start_ns + duration_ns
        words = []
        for segment in result.get("segments", []):
            if segment.get("text", "").strip() and not segment.get("words"):
                raise ValueError("ASR returned text without required word timestamps")
            for item in segment.get("words", []):
                text = item.get("word", "")
                if not text.strip():
                    continue
                start, end = float(item["start"]), float(item["end"])
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                    raise ValueError("ASR returned invalid word timestamps")
                # Timestamps near the padded window edge may overshoot slightly.
                # Never invent timings for text fully outside the supplied audio.
                if start * NS > duration_ns + 20_000_000 or end * NS > duration_ns + 20_000_000:
                    raise ValueError("ASR word falls outside the supplied audio window")
                start_ns = window_start_ns + min(duration_ns, round(start * NS))
                end_ns = window_start_ns + min(duration_ns, round(end * NS))
                words.append(TimedWord(text, start_ns, end_ns))
        if result.get("text", "").strip() and not words:
            raise ValueError("ASR returned text without required word timestamps")
        return ASRHypothesis(window_start_ns, window_end_ns, tuple(words), result.get("language"))

    def transcribe_wav(self, path: str | Path, *, window_start_ns: int = 0) -> ASRHypothesis:
        """Read PCM16 mono 16 kHz WAV; other formats need an upstream decoder."""
        with wave.open(str(path), "rb") as stream:
            if (stream.getnchannels() != 1 or stream.getframerate() != 16000
                    or stream.getsampwidth() != 2 or stream.getcomptype() != "NONE"):
                raise ValueError("WAV adapter requires uncompressed PCM16 mono 16 kHz")
            frame_count = stream.getnframes()
            if frame_count > self.max_window_samples:
                raise ValueError("WAV exceeds the configured maximum ASR window; slice it upstream")
            raw = stream.readframes(frame_count)
        if len(raw) != frame_count * 2:
            raise ValueError("truncated PCM16 WAV")
        samples = (value[0] / 32768.0 for value in struct.iter_unpack("<h", raw))
        return self.transcribe_pcm(samples, window_start_ns=window_start_ns)
