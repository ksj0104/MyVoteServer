"""Local WeSpeaker ONNX embeddings, separate from speech/overlap detection.

The frontend follows the upstream WeSpeaker ONNX inference recipe (see
docs/speaker-embedding.md). This is a synchronous worker API, never a callback
for the capture or translation event loop. No model downloads are performed.
"""

from __future__ import annotations

import hashlib
import importlib
from itertools import islice
import math
from pathlib import Path
from typing import Iterable, Protocol

from .speakers import Observation


SAMPLE_RATE = 16_000
NS = 1_000_000_000
FRONTEND_ID = "wespeaker-kaldi80-hamming25-shift10-cmn-v1"
MAX_MODEL_BYTES = 512 * 1024 * 1024


class SpeakerEmbedding(Protocol):
    """One model/frontend identity per tracker; vectors cannot cross identities."""

    @property
    def identity(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, samples: Iterable[float], sample_rate: int = SAMPLE_RATE
              ) -> tuple[float, ...]: ...


class LocalWeSpeakerEmbedding:
    """CPU-only, single-file WeSpeaker ONNX model with normalized mono PCM.

    Supported graph contract: float32 ``feats[1,T,80]`` -> ``embs[1,D]``.
    Default D=256 matches the official VoxCeleb ResNet34 export. A compatible
    custom export needs its explicit dimension and separately calibrated tracker.
    Models with external weight files are not supported: load from model bytes.

    One instance belongs to one worker. Loading and inference are synchronous.
    ``embed`` extracts a vector only; it says nothing about speech, quality,
    number of speakers, or overlap. Use independently measured evidence when
    creating an Observation, or use ``embed_observation`` to enforce that gate.
    """

    def __init__(self, model_path: str | Path, *, dimension: int = 256,
                 min_window_s: float = 1.0, max_window_s: float = 5.0,
                 intra_op_threads: int = 1) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 4096:
            raise ValueError("dimension must be an integer in [1, 4096]")
        if (isinstance(intra_op_threads, bool) or not isinstance(intra_op_threads, int)
                or not 1 <= intra_op_threads <= 4):
            raise ValueError("intra_op_threads must be an integer in [1, 4]")
        if (isinstance(min_window_s, bool) or isinstance(max_window_s, bool)
                or not math.isfinite(min_window_s) or not math.isfinite(max_window_s)
                or not 1.0 <= min_window_s <= max_window_s <= 5.0):
            raise ValueError("embedding windows must satisfy 1 <= min <= max <= 5 seconds")
        self.model_path = Path(model_path).expanduser().resolve()
        if self.model_path.suffix.lower() != ".onnx" or not self.model_path.is_file():
            raise FileNotFoundError("provide a predownloaded local single-file WeSpeaker .onnx model")
        # Keep exactly the bytes identified by the checksum: replacing the file
        # after construction cannot silently mix embeddings from different models.
        with self.model_path.open("rb") as source:
            self._model_bytes = source.read(MAX_MODEL_BYTES + 1)
        if not self._model_bytes or len(self._model_bytes) > MAX_MODEL_BYTES:
            raise ValueError("ONNX model must be nonempty and at most 512 MiB")
        self._dimension = dimension
        self._identity = (
            f"wespeaker-onnx:sha256:{hashlib.sha256(self._model_bytes).hexdigest()}:"
            f"{FRONTEND_ID}:d{dimension}")
        self.min_window_samples = math.ceil(min_window_s * SAMPLE_RATE)
        self.max_window_samples = math.floor(max_window_s * SAMPLE_RATE)
        self.intra_op_threads = intra_op_threads
        self._session = None
        self._torch = None
        self._kaldi = None
        self._input_frames: int | None = None

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def dimension(self) -> int:
        return self._dimension

    def _prepare_pcm(self, samples: Iterable[float], sample_rate: int) -> tuple[float, ...]:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate != SAMPLE_RATE:
            raise ValueError("embedding requires mono 16 kHz PCM; resample upstream")
        bounded = tuple(islice(samples, self.max_window_samples + 1))
        if not self.min_window_samples <= len(bounded) <= self.max_window_samples:
            raise ValueError("PCM length must fit the configured 1-5 second embedding window")
        if any(isinstance(value, (bool, str, bytes)) for value in bounded):
            raise ValueError("PCM must contain numeric normalized samples")
        try:
            waveform = tuple(float(value) for value in bounded)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PCM must be one-dimensional normalized mono samples") from exc
        if not all(math.isfinite(value) and -1 <= value <= 1 for value in waveform):
            raise ValueError("PCM samples must be finite and normalized to [-1, 1]")
        if not any(waveform):
            raise ValueError("zero PCM cannot supply speaker evidence")
        return waveform

    def _ensure_runtime(self) -> None:
        if self._session is not None:
            return
        try:
            torch = importlib.import_module("torch")
            kaldi = importlib.import_module("torchaudio.compliance.kaldi")
            ort = importlib.import_module("onnxruntime")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "install compatible optional numpy, torch, torchaudio, and onnxruntime "
                "packages in the speaker worker environment") from exc
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = self.intra_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = ort.InferenceSession(
            self._model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "feats" or inputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 3 or inputs[0].shape[2] != 80):
            raise ValueError("WeSpeaker model must accept one float32 feats[B,T,80] input")
        if (len(outputs) != 1 or outputs[0].name != "embs" or outputs[0].type != "tensor(float)"
                or len(outputs[0].shape) != 2 or outputs[0].shape[1] != self.dimension):
            raise ValueError("WeSpeaker model must output float32 embs[B,configured_dimension]")
        for batch in (inputs[0].shape[0], outputs[0].shape[0]):
            if isinstance(batch, int) and batch != 1:
                raise ValueError("WeSpeaker model must allow batch size 1")
        frames = inputs[0].shape[1]
        if isinstance(frames, int) and frames <= 0:
            raise ValueError("invalid fixed frame count in WeSpeaker graph")
        self._input_frames = frames if isinstance(frames, int) else None
        self._torch, self._kaldi, self._session = torch, kaldi, session
        # ONNX Runtime owns the initialized graph now. The checksum remains stable.
        self._model_bytes = b""

    def _compute_features(self, waveform: tuple[float, ...]):
        torch = self._torch
        # Match WeSpeaker infer_onnx.py: PCM16 amplitude, Kaldi defaults with
        # 80 filters / Hamming window, then per-utterance mean (no variance) norm.
        pcm16_scale = torch.tensor([waveform], dtype=torch.float32, device="cpu") * 32768.0
        with torch.inference_mode():
            features = self._kaldi.fbank(
                pcm16_scale, num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
                dither=0.0, sample_frequency=16000.0, window_type="hamming",
                use_energy=False, snip_edges=True, round_to_power_of_two=True,
                preemphasis_coefficient=0.97, remove_dc_offset=True,
                low_freq=20.0, high_freq=0.0, energy_floor=1.0,
                raw_energy=True, htk_compat=False, use_log_fbank=True,
                use_power=True, subtract_mean=False)
            centered = features - features.mean(dim=0, keepdim=True)
            return centered.unsqueeze(0).contiguous().numpy()

    def _embed_prepared(self, waveform: tuple[float, ...]) -> tuple[float, ...]:
        self._ensure_runtime()
        features = self._compute_features(waveform)
        expected_frames = 1 + (len(waveform) - 400) // 160
        if tuple(features.shape) != (1, expected_frames, 80) or str(features.dtype) != "float32":
            raise ValueError("frontend returned invalid float32 features[B,T,80]")
        if self._input_frames is not None and self._input_frames != expected_frames:
            raise ValueError("PCM window does not match the ONNX graph's fixed frame count")
        output = self._session.run(output_names=["embs"], input_feed={"feats": features})
        if len(output) != 1 or tuple(output[0].shape) != (1, self.dimension):
            raise ValueError("WeSpeaker returned the wrong embedding shape")
        if str(output[0].dtype) != "float32":
            raise ValueError("WeSpeaker returned a non-float32 embedding")
        vector = tuple(float(value) for value in output[0].tolist()[0])
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("WeSpeaker returned a non-finite embedding")
        scale = max(abs(value) for value in vector)
        if scale == 0:
            raise ValueError("WeSpeaker returned a zero embedding")
        scaled = tuple(value / scale for value in vector)
        norm = math.sqrt(math.fsum(value * value for value in scaled))
        return tuple(value / norm for value in scaled)

    def embed(self, samples: Iterable[float], sample_rate: int = SAMPLE_RATE
              ) -> tuple[float, ...]:
        """Return a finite unit vector. No speech/quality/overlap decision is made."""
        return self._embed_prepared(self._prepare_pcm(samples, sample_rate))

    def embed_observation(self, samples: Iterable[float], sample_rate: int = SAMPLE_RATE,
                          *, sample_id: str, track_id: str, start_time_ns: int,
                          speech_duration_ns: int, overlap: bool, quality: float,
                          min_quality: float = 0.70,
                          min_speech_ns: int = NS) -> Observation:
        """Require external evidence and reject ineligible audio before inference.

        No defaults for overlap, quality, or speech duration: a missing overlap
        detector must leave the assignment unknown, not assume clean speech.
        Thresholds are engineering defaults; align them with the chosen tracker.
        """
        if (not isinstance(sample_id, str) or not sample_id.strip()
                or not isinstance(track_id, str) or not track_id.strip()):
            raise ValueError("sample_id and track_id are required")
        for name, value in (("start_time_ns", start_time_ns),
                            ("speech_duration_ns", speech_duration_ns),
                            ("min_speech_ns", min_speech_ns)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be nonnegative integer nanoseconds")
        if min_speech_ns < NS:
            raise ValueError("speaker evidence requires at least one second of speech")
        if not isinstance(overlap, bool):
            raise ValueError("overlap must be an explicitly observed bool")
        for name, value in (("quality", quality), ("min_quality", min_quality)):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if overlap or quality < min_quality or speech_duration_ns < min_speech_ns:
            raise ValueError("speaker evidence is overlapping, low-quality, or too short")
        waveform = self._prepare_pcm(samples, sample_rate)
        duration_ns = len(waveform) * NS // SAMPLE_RATE
        if speech_duration_ns > duration_ns:
            raise ValueError("speech duration exceeds the supplied PCM interval")
        vector = self._embed_prepared(waveform)
        return Observation(sample_id, track_id, start_time_ns, start_time_ns + duration_ns,
                           vector, speech_duration_ns, overlap=overlap, quality=quality)
