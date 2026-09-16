"""Local, bounded pyannote powerset segmentation for a speaker worker.

This adapter estimates speech/overlap; it neither assigns persistent speaker IDs
nor measures acoustic quality. A trusted span means it passed this adapter's
engineering gates, not that the audio is proven to contain only one speaker.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from itertools import islice
import math
from pathlib import Path
from typing import Iterable, Protocol


SAMPLE_RATE = 16_000
NS = 1_000_000_000
WINDOW_SAMPLES = 160_000
MIN_SAMPLES = 16_000
RECEPTIVE_FIELD_SAMPLES = 991
FRAME_SHIFT_SAMPLES = 270
OUTPUT_FRAMES = 589
MAX_MODEL_BYTES = 64 * 1024 * 1024
# pyannote Powerset(3, 2), in combinations-by-cardinality order.
POWERSET = ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
FRONTEND_ID = "fixed10s-logpowerset7-center-cells-v1"
_METADATA = {
    "model_type": "pyannote-segmentation-3.0", "version": "1",
    "sample_rate": "16000", "window_size": "160000", "num_speakers": "3",
    "powerset_max_classes": "2", "num_classes": "7",
    "receptive_field_size": "991", "receptive_field_shift": "270",
}


@dataclass(frozen=True, slots=True)
class PowersetPrediction:
    active_speakers: tuple[int, ...]
    confidence: float
    overlap_probability: float
    probabilities: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class PowersetFrame:
    """Original powerset prediction for one complete, guarded center cell.

    Unlike a gated span, this preserves the model's winning local speaker slots
    even when the score is too low to use as clean embedding evidence. The seven
    probabilities follow POWERSET order. Local slots only identify heads within
    this inference window; they are not persistent speaker identities.
    """

    index: int
    start_ns: int
    end_ns: int
    active_speakers: tuple[int, ...]
    probabilities: tuple[float, ...]
    confidence: float
    overlap_probability: float


@dataclass(frozen=True, slots=True)
class SegmentationSpan:
    """Half-open source time interval; local slots are meaningful in one result.

    Empty slots with trusted=True mean predicted nonspeech. Empty slots with
    trusted=False mean unknown. Missing overlap probability means no estimate,
    as on unprocessed/edge audio. Merged confidence is the minimum frame score;
    merged overlap probability is the maximum, never an optimistic average.
    """

    start_ns: int
    end_ns: int
    active_speakers: tuple[int, ...]
    confidence: float
    trusted: bool
    reason: str
    overlap_probability: float | None


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    spans: tuple[SegmentationSpan, ...]
    model_identity: str
    window_start_ns: int
    window_end_ns: int
    padded_samples: int
    trusted_start_ns: int
    trusted_end_ns: int
    frames: tuple[PowersetFrame, ...] = ()


class SpeakerSegmentation(Protocol):
    @property
    def identity(self) -> str: ...

    def segment(self, samples: Iterable[float], *, sample_rate: int = SAMPLE_RATE,
                window_start_ns: int = 0) -> SegmentationResult: ...


def decode_log_probabilities(row: Iterable[float]) -> PowersetPrediction:
    """Decode exactly seven normalized log probabilities, not arbitrary logits.

    Confidence is the winning powerset class probability. Overlap probability
    is the sum of the three two-speaker class probabilities. These model scores
    are not calibrated probabilities of an error-free acoustic observation.
    """
    bounded = tuple(islice(row, 8))
    if len(bounded) != 7 or any(isinstance(x, (bool, str, bytes)) for x in bounded):
        raise ValueError("expected seven numeric powerset log probabilities")
    try:
        values = tuple(float(x) for x in bounded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid powerset log probabilities") from exc
    if not all(math.isfinite(x) and x <= 1e-5 for x in values):
        raise ValueError("powerset log probabilities must be finite and nonpositive")
    probabilities = tuple(math.exp(x) for x in values)
    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError("powerset output must be normalized log probabilities")
    # Correct only float32 rounding after verifying the log-probability contract.
    probabilities = tuple(value / total for value in probabilities)
    winner = max(range(7), key=probabilities.__getitem__)
    return PowersetPrediction(POWERSET[winner], probabilities[winner],
                             math.fsum(probabilities[4:]), probabilities)


class LocalPyannoteSegmentation:
    """Single-file CPU ONNX segmentation; no network or implicit downloads.

    A call accepts 1..160000 real, normalized mono16k samples. At least one
    second is required for inference; shorter input returns unknown. Inference
    always uses the trained 10-second shape, right-padding shorter input with
    zeros. Padding is never returned as source audio or speaker evidence.

    Use one instance in one bounded speaker worker, away from capture/translation.
    The bidirectional LSTM and temporal instance normalization use full-window
    context. The 991-sample convolution footprint and edge guard do not make
    inference causal or prove padding harmless.
    """

    def __init__(self, model_path: str | Path, *, edge_guard_s: float = 0.25,
                 min_confidence: float = 0.80, max_overlap_probability: float = 0.10,
                 intra_op_threads: int = 1, retain_frames: bool = False) -> None:
        for name, value, low, high in (
            ("edge_guard_s", edge_guard_s, 0.0, 2.0),
            ("min_confidence", min_confidence, 0.5, 1.0),
            ("max_overlap_probability", max_overlap_probability, 0.0, 0.5),
        ):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        if (isinstance(intra_op_threads, bool) or not isinstance(intra_op_threads, int)
                or not 1 <= intra_op_threads <= 4):
            raise ValueError("intra_op_threads must be an integer in [1, 4]")
        if not isinstance(retain_frames, bool):
            raise ValueError("retain_frames must be a boolean")
        self.model_path = Path(model_path).expanduser().resolve()
        if self.model_path.suffix.lower() != ".onnx" or not self.model_path.is_file():
            raise FileNotFoundError("provide a predownloaded local pyannote segmentation .onnx")
        with self.model_path.open("rb") as source:
            self._model_bytes = source.read(MAX_MODEL_BYTES + 1)
        if not self._model_bytes or len(self._model_bytes) > MAX_MODEL_BYTES:
            raise ValueError("segmentation ONNX must be nonempty and at most 64 MiB")
        self._identity = (f"pyannote-segmentation-onnx:sha256:"
                          f"{hashlib.sha256(self._model_bytes).hexdigest()}:{FRONTEND_ID}")
        self.edge_guard_samples = math.ceil(edge_guard_s * SAMPLE_RATE)
        self.min_confidence = float(min_confidence)
        self.max_overlap_probability = float(max_overlap_probability)
        self.intra_op_threads = intra_op_threads
        self.retain_frames = retain_frames
        self._session = None
        self._numpy = None

    @property
    def identity(self) -> str:
        return self._identity

    def preload(self) -> None:
        """Load/validate the local graph without claiming a warmed-up inference."""
        if self._session is not None:
            return
        try:
            np = importlib.import_module("numpy")
            ort = importlib.import_module("onnxruntime")
        except (ImportError, OSError) as exc:
            raise RuntimeError("install optional numpy and onnxruntime in the speaker worker") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = ort.InferenceSession(self._model_bytes, sess_options=options,
                                       providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "x" or inputs[0].type != "tensor(float)"
                or len(inputs[0].shape) != 3 or inputs[0].shape[1] != 1):
            raise ValueError("segmentation must accept one float32 x[B,1,T] input")
        if (len(outputs) != 1 or outputs[0].name != "y" or outputs[0].type != "tensor(float)"
                or len(outputs[0].shape) != 3 or outputs[0].shape[2] != 7):
            raise ValueError("segmentation must output one float32 y[B,F,7]")
        for actual, expected in ((inputs[0].shape[0], 1), (outputs[0].shape[0], 1),
                                 (inputs[0].shape[2], WINDOW_SAMPLES),
                                 (outputs[0].shape[1], OUTPUT_FRAMES)):
            if isinstance(actual, int) and actual != expected:
                raise ValueError("segmentation graph does not support the fixed 10-second shape")
        metadata = session.get_modelmeta().custom_metadata_map
        for name, expected in _METADATA.items():
            if metadata.get(name) != expected:
                raise ValueError(f"unsupported segmentation metadata: {name}")
        self._numpy, self._session = np, session
        self._model_bytes = b""

    def _prepare_pcm(self, samples: Iterable[float], sample_rate: int,
                     window_start_ns: int) -> tuple[float, ...]:
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int)
                or sample_rate != SAMPLE_RATE):
            raise ValueError("segmentation requires normalized mono 16 kHz PCM")
        if (isinstance(window_start_ns, bool) or not isinstance(window_start_ns, int)
                or window_start_ns < 0):
            raise ValueError("window_start_ns must be nonnegative integer nanoseconds")
        bounded = tuple(islice(samples, WINDOW_SAMPLES + 1))
        if not 1 <= len(bounded) <= WINDOW_SAMPLES:
            raise ValueError("segmentation accepts 1..160000 real samples")
        if any(isinstance(x, (bool, str, bytes)) for x in bounded):
            raise ValueError("PCM must contain normalized numeric samples")
        try:
            waveform = tuple(float(x) for x in bounded)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PCM must be one-dimensional normalized mono samples") from exc
        if not all(math.isfinite(x) and -1 <= x <= 1 for x in waveform):
            raise ValueError("PCM samples must be finite and normalized to [-1, 1]")
        return waveform

    def segment(self, samples: Iterable[float], *, sample_rate: int = SAMPLE_RATE,
                window_start_ns: int = 0) -> SegmentationResult:
        waveform = self._prepare_pcm(samples, sample_rate, window_start_ns)
        end_ns = window_start_ns + len(waveform) * NS // SAMPLE_RATE
        if len(waveform) < MIN_SAMPLES:
            return SegmentationResult(
                (SegmentationSpan(window_start_ns, end_ns, (), 0.0, False,
                                  "too_short", None),),
                self.identity, window_start_ns, end_ns, WINDOW_SAMPLES - len(waveform),
                window_start_ns, window_start_ns)
        self.preload()
        pcm = self._numpy.zeros((1, 1, WINDOW_SAMPLES), dtype=self._numpy.float32)
        pcm[0, 0, :len(waveform)] = waveform
        output = self._session.run(output_names=["y"], input_feed={"x": pcm})
        if (len(output) != 1 or tuple(output[0].shape) != (1, OUTPUT_FRAMES, 7)
                or str(output[0].dtype) != "float32"):
            raise ValueError("segmentation returned the wrong float32 [1,589,7] output")
        # Validate every row, even the padding region; a corrupt graph is an error.
        predictions = tuple(decode_log_probabilities(row) for row in output[0].tolist()[0])
        if len(predictions) != OUTPUT_FRAMES:
            raise ValueError("segmentation returned the wrong frame count")
        return self._spans(predictions, len(waveform), window_start_ns)

    def _spans(self, predictions: tuple[PowersetPrediction, ...], real_samples: int,
               start_ns: int) -> SegmentationResult:
        # Frame i convolution footprint: [270*i, 270*i+991). Quantize the midpoint
        # cell between adjacent support centers to sample boundaries, rounding
        # both shared boundaries up; the cell is [361+270*i, 631+270*i).
        cell_offset = (RECEPTIVE_FIELD_SAMPLES - FRAME_SHIFT_SAMPLES + 1) // 2
        first = (self.edge_guard_samples + FRAME_SHIFT_SAMPLES - 1) // FRAME_SHIFT_SAMPLES
        last = min(OUTPUT_FRAMES - 1,
                   (real_samples - self.edge_guard_samples - RECEPTIVE_FIELD_SAMPLES)
                   // FRAME_SHIFT_SAMPLES)
        end_ns = start_ns + real_samples * NS // SAMPLE_RATE
        if first > last:
            return SegmentationResult(
                (SegmentationSpan(start_ns, end_ns, (), 0.0, False, "edge_context", None),),
                self.identity, start_ns, end_ns, WINDOW_SAMPLES - real_samples,
                start_ns, start_ns)
        trusted_start = start_ns + (cell_offset + first * FRAME_SHIFT_SAMPLES) * NS // SAMPLE_RATE
        trusted_end = start_ns + (cell_offset + (last + 1) * FRAME_SHIFT_SAMPLES) * NS // SAMPLE_RATE
        spans: list[SegmentationSpan] = []
        frames: list[PowersetFrame] = []

        def append(span: SegmentationSpan) -> None:
            if span.start_ns >= span.end_ns:
                return
            if (spans and spans[-1].end_ns == span.start_ns
                    and (spans[-1].active_speakers, spans[-1].trusted, spans[-1].reason)
                    == (span.active_speakers, span.trusted, span.reason)):
                old = spans[-1]
                overlap = (None if old.overlap_probability is None or span.overlap_probability is None
                           else max(old.overlap_probability, span.overlap_probability))
                spans[-1] = SegmentationSpan(old.start_ns, span.end_ns, old.active_speakers,
                                              min(old.confidence, span.confidence), old.trusted,
                                              old.reason, overlap)
            else:
                spans.append(span)

        append(SegmentationSpan(start_ns, trusted_start, (), 0.0, False, "edge_context", None))
        for index in range(first, last + 1):
            prediction = predictions[index]
            active = prediction.active_speakers
            trusted = prediction.confidence >= self.min_confidence
            reason = "prediction" if trusted else "low_confidence"
            if (trusted and len(active) < 2
                    and prediction.overlap_probability > self.max_overlap_probability):
                trusted, reason = False, "overlap_uncertain"
            left = start_ns + (cell_offset + index * FRAME_SHIFT_SAMPLES) * NS // SAMPLE_RATE
            right = start_ns + (cell_offset + (index + 1) * FRAME_SHIFT_SAMPLES) * NS // SAMPLE_RATE
            # A real segment() call decodes all seven values for every row. Old
            # scripted _spans callers remain compatible with the default mode;
            # opt-in callers must supply the actual posterior for every cell.
            if self.retain_frames:
                probabilities = prediction.probabilities
                if (not isinstance(probabilities, tuple) or len(probabilities) != 7
                        or any(isinstance(value, bool) or not isinstance(value, (int, float))
                               or not math.isfinite(value) or not 0 <= value <= 1
                               for value in probabilities)
                        or not math.isclose(math.fsum(probabilities), 1.0,
                                            rel_tol=0.0, abs_tol=1e-12)):
                    raise ValueError("retained frames require seven normalized probabilities")
                winner = max(range(7), key=probabilities.__getitem__)
                if (prediction.active_speakers != POWERSET[winner]
                        or prediction.confidence != probabilities[winner]
                        or prediction.overlap_probability != math.fsum(probabilities[4:])):
                    raise ValueError("retained frame summary must match its probabilities")
                frames.append(PowersetFrame(
                    index, left, right, active, probabilities,
                    prediction.confidence, prediction.overlap_probability))
            append(SegmentationSpan(left, right, active if trusted else (),
                                    prediction.confidence, trusted, reason,
                                    prediction.overlap_probability))
        append(SegmentationSpan(trusted_end, end_ns, (), 0.0, False,
                                "padding_edge" if real_samples < WINDOW_SAMPLES else "edge_context",
                                None))
        return SegmentationResult(tuple(spans), self.identity, start_ns, end_ns,
                                  WINDOW_SAMPLES - real_samples, trusted_start, trusted_end,
                                  tuple(frames))
