"""Bounded encoded audio inputs, canonicalized to request-owned mono 16 kHz PCM."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import hashlib
import io
import math
import struct
import time
from typing import Any, Callable, Sequence

import numpy as np

from .images import DEFAULT_LIMITS
from .images_http import ImageInputError


@dataclass(frozen=True, slots=True)
class AudioLimits:
    """The native Gemma audio region, within the existing encoded-media envelope.

    Gemma E2B accepts at most 30 seconds per audio segment. Cumulative decoded
    retention is supplied by the caller from its admitted prompt-token budget.
    """

    max_encoded_bytes: int = DEFAULT_LIMITS.max_encoded_bytes
    max_total_encoded_bytes: int = DEFAULT_LIMITS.max_total_encoded_bytes
    sample_rate: int = 16_000
    max_seconds: float = 30.0
    timeout_seconds: float = DEFAULT_LIMITS.timeout_seconds
    total_timeout_seconds: float = DEFAULT_LIMITS.total_timeout_seconds

    def __post_init__(self) -> None:
        for name in ("max_encoded_bytes", "max_total_encoded_bytes", "sample_rate"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_rate >= 2**31:
            raise ValueError("audio sample rate must fit the decoder's signed integer range")
        for name in ("max_seconds", "timeout_seconds", "total_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        samples = self.sample_rate * self.max_seconds
        if not math.isfinite(samples) or samples < 1 or samples > np.iinfo(np.intp).max // 4:
            raise ValueError("audio sample limit exceeds the supported array range")

    @property
    def max_samples(self) -> int:
        return int(self.sample_rate * self.max_seconds)


DEFAULT_AUDIO_LIMITS = AudioLimits()
_FORMATS = {"wav": "wav", "mp3": "mp3"}


@dataclass(frozen=True, slots=True)
class AudioSource:
    data: str
    format: str


@dataclass(frozen=True, slots=True)
class AudioInput:
    samples: np.ndarray
    sample_rate: int = 16_000
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.samples, np.ndarray) or self.samples.ndim != 1
                or self.samples.dtype != np.float32 or self.samples.size == 0
                or type(self.sample_rate) is not int or not 0 < self.sample_rate < 2**32
                or not np.isfinite(self.samples).all()):
            raise ImageInputError("audio input requires nonempty finite mono float32 samples and a positive rate")
        # Decoder/processor callers cannot mutate a previously prepared request.
        samples = np.array(self.samples, dtype="<f4", order="C", copy=True)
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)
        h = hashlib.sha256(b"tensorfold-audio-v1\0" + struct.pack(">IQ", self.sample_rate, len(samples)))
        h.update(memoryview(samples).cast("B"))
        object.__setattr__(self, "content_hash", h.hexdigest())


def audio_source(value: Any, limits: AudioLimits = DEFAULT_AUDIO_LIMITS) -> AudioSource:
    """Validate the Chat content-part payload before any base64 allocation."""
    if (not isinstance(value, dict) or not isinstance(value.get("data"), str) or not value["data"]
            or not isinstance(value.get("format"), str) or value["format"] not in _FORMATS):
        raise ImageInputError("input_audio requires nonempty base64 data and format wav or mp3")
    data = value["data"]
    if len(data) > 4 * ((limits.max_encoded_bytes + 2) // 3):
        raise ImageInputError("audio exceeds the encoded byte limit")
    if not data.isascii():
        raise ImageInputError("audio data must use ASCII base64")
    return AudioSource(data, value["format"])


def _bytes(source: AudioSource, limit: int) -> bytes:
    if (not isinstance(source, AudioSource) or not isinstance(source.format, str) or source.format not in _FORMATS
            or not isinstance(source.data, str) or not source.data
            or len(source.data) > 4 * ((limit + 2) // 3) or not source.data.isascii()):
        raise ImageInputError("audio exceeds the encoded byte limit or has invalid source metadata")
    try:
        data = base64.b64decode(source.data, validate=True)
    except (ValueError, binascii.Error):
        raise ImageInputError("invalid audio base64 encoding") from None
    if not data or len(data) > limit:
        raise ImageInputError("audio is empty or exceeds the encoded byte limit")
    return data


def _decode(data: bytes, format: str, limits: AudioLimits, remaining_samples: int,
            deadline: float, cancelled: Callable[[], bool] | None = None) -> AudioInput:
    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            from .replay import MediaCancelled
            raise MediaCancelled("audio preparation cancelled")

    check_cancelled()
    try:
        import av
    except ImportError:
        raise ImageInputError("audio inputs require PyAV") from None
    sample_limit = min(limits.max_samples, remaining_samples)
    if sample_limit <= 0:
        raise ImageInputError("audio exceeds the decoded sample budget")
    chunks, count = [], 0
    try:
        with av.open(io.BytesIO(data), mode="r", format=_FORMATS[format]) as container:
            if len(container.streams.audio) != 1:
                raise ImageInputError("audio must contain exactly one audio stream")
            stream = container.streams.audio[0]
            stream.codec_context.thread_count = 1
            channels = int(stream.codec_context.channels)
            rate = int(stream.codec_context.sample_rate)
            if channels not in (1, 2) or rate <= 0:
                raise ImageInputError("audio must have a valid sample rate and mono or stereo channels")
            if stream.duration is not None and stream.time_base is not None:
                seconds = float(stream.duration * stream.time_base)
                if not math.isfinite(seconds) or seconds < 0:
                    raise ImageInputError("audio duration is invalid")
                # Container duration may include codec priming. Actual canonical
                # decoded samples below own the native duration limit.
            converter = av.AudioResampler(format="fltp", layout=stream.codec_context.layout, rate=rate)
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=limits.sample_rate)

            def retain(frame) -> None:
                nonlocal count
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise ImageInputError("audio decoding exceeds the time limit")
                samples = frame.to_ndarray()
                if samples.ndim != 2 or samples.shape[0] != 1 or samples.dtype != np.float32:
                    raise ImageInputError("audio decoder returned an unsupported sample layout")
                n = samples.shape[1]
                if count + n > sample_limit:
                    if count + n > limits.max_samples:
                        raise ImageInputError("Gemma audio segments are limited to 30 seconds")
                    raise ImageInputError("audio request exceeds the decoded sample budget")
                mono = samples[0]
                if not np.isfinite(mono).all():
                    raise ImageInputError("audio samples must be finite")
                chunks.append(np.array(mono, dtype=np.float32, copy=True))
                count += n

            def convert(frame) -> None:
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise ImageInputError("audio decoding exceeds the time limit")
                samples = frame.to_ndarray()
                if (samples.ndim != 2 or samples.shape[0] != channels or samples.dtype != np.float32
                        or not np.isfinite(samples).all()):
                    raise ImageInputError("audio decoder returned invalid sample data")
                # Mean channels before resampling, matching canonical mono
                # conversion rather than FFmpeg's energy-preserving mix gain.
                mono = samples[0] if channels == 1 else samples.mean(axis=0, dtype=np.float32)
                converted = av.AudioFrame.from_ndarray(mono[None, :], format="fltp", layout="mono")
                converted.sample_rate = rate
                for output in resampler.resample(converted):
                    retain(output)

            for frame in container.decode(stream):
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise ImageInputError("audio decoding exceeds the time limit")
                if frame.samples * limits.sample_rate > limits.max_samples * rate:
                    raise ImageInputError("Gemma audio segments are limited to 30 seconds")
                for floating in converter.resample(frame):
                    convert(floating)
            for floating in converter.resample(None):
                convert(floating)
            for output in resampler.resample(None):
                retain(output)
    except ImageInputError:
        raise
    except (OSError, ValueError, EOFError, av.error.FFmpegError) as exc:
        raise ImageInputError(f"audio bytes are invalid or unsupported ({type(exc).__name__}); use WAV or MP3") from None
    if not chunks or count == 0:
        raise ImageInputError("audio contains no decoded samples")
    return AudioInput(np.concatenate(chunks), limits.sample_rate)


def load_audios(sources: Sequence[AudioSource], *, max_total_samples: int,
                limits: AudioLimits = DEFAULT_AUDIO_LIMITS,
                cancelled: Callable[[], bool] | None = None) -> list[AudioInput]:
    """Bound encoded bytes, canonical decoded retention and work time cumulatively."""
    if type(max_total_samples) is not int or max_total_samples <= 0:
        raise ValueError("audio preparation requires a positive cumulative sample budget")
    if not isinstance(sources, (tuple, list)):
        raise ImageInputError("audio sources must be a sequence")
    if cancelled is not None and not callable(cancelled):
        raise ValueError("audio cancellation must be a callback or None")
    if len(sources) > max_total_samples:
        raise ImageInputError("audio source count exceeds the decoded sample budget")
    used_bytes, used_samples = 0, 0
    deadline = time.monotonic() + limits.total_timeout_seconds
    output = []
    for source in sources:
        remaining = min(limits.max_encoded_bytes, limits.max_total_encoded_bytes - used_bytes)
        if remaining <= 0 or time.monotonic() >= deadline:
            raise ImageInputError("audio request exceeds the total byte or time limit")
        data = _bytes(source, remaining)
        used_bytes += len(data)
        audio = _decode(data, source.format, limits, max_total_samples - used_samples,
                        min(deadline, time.monotonic() + limits.timeout_seconds), cancelled)
        used_samples += len(audio.samples)
        output.append(audio)
    return output
