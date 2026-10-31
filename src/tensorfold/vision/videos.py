"""CPU video inputs: bounded bytes, frames sampled at a fixed rate, decoded straight to the tower's resolution."""

from __future__ import annotations

import hashlib
import io
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .images import _data_bytes
from .images_http import ImageInputError, fetch_image

VIDEO_MEDIA_TYPES = {"video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov", "video/x-matroska": "mkv"}


@dataclass(frozen=True, slots=True)
class VideoLimits:
    max_videos: int = 2
    max_encoded_bytes: int = 64 * 1024 * 1024
    max_total_encoded_bytes: int = 96 * 1024 * 1024
    max_dimension: int = 8192
    max_seconds: float = 3600.0          # of footage; frames past the sampled ones are decoded and dropped
    fps: float = 2.0                      # frames sampled a second (Qwen3-VL's rate)
    min_frames: int = 4
    max_frames: int = 256                 # a longer video is sampled more sparsely, over its whole length
    timeout_seconds: float = 20.0
    total_timeout_seconds: float = 120.0
    max_redirects: int = 3
    max_url_chars: int = 4096


DEFAULT_VIDEO_LIMITS = VideoLimits()


@dataclass(frozen=True, slots=True)
class VideoSource:
    url: str


@dataclass(frozen=True, slots=True)
class VideoInput:
    frames: np.ndarray            # [frames, height, width, 3] uint8, already at the tower's resolution
    indices: tuple[int, ...]      # each frame's index in the source
    fps: float                    # the source's frame rate
    timestamp_seconds: tuple[float, ...] | None = None  # opt-in actual presentation times, relative to first frame
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.frames, np.ndarray) or self.frames.dtype != np.uint8 or self.frames.ndim != 4
                or self.frames.shape[-1] != 3 or len(self.indices) != self.frames.shape[0] or not self.indices
                or not math.isfinite(self.fps) or self.fps <= 0):
            raise ImageInputError("video input requires RGB frames, one index each and a positive frame rate")
        if self.timestamp_seconds is not None and (
                len(self.timestamp_seconds) != len(self.indices)
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
                       for x in self.timestamp_seconds)
                or any(b < a for a, b in zip(self.timestamp_seconds, self.timestamp_seconds[1:]))):
            raise ImageInputError("video presentation times must be finite, ordered and match the selected frames")
        self.frames.setflags(write=False)
        digest = hashlib.sha256(b"tensorfold-video-v1\0" + np.asarray(self.frames.shape, dtype=">u4").tobytes())
        digest.update(np.asarray(self.indices, dtype=">u8").tobytes())
        digest.update(self.frames.tobytes())
        object.__setattr__(self, "content_hash", digest.hexdigest())

    def timestamps(self, temporal: int) -> list[float]:
        """Each group of ``temporal`` frames' time: the mean of its first and last frame (Qwen3-VL's processor)."""
        indices = list(self.indices)
        if len(indices) % temporal:
            indices.extend(indices[-1] for _ in range(temporal - len(indices) % temporal))
        seconds = [i / self.fps for i in indices]
        return [(seconds[i] + seconds[i + temporal - 1]) / 2 for i in range(0, len(seconds), temporal)]


def video_source(value: Any, limits: VideoLimits, allow_urls: bool) -> VideoSource:
    if not isinstance(value, dict) or not isinstance(value.get("url"), str) or not value["url"]:
        raise ImageInputError("video_url must contain a non-empty url string")
    url = value["url"]
    if url.startswith("data:"):
        if len(url) > limits.max_encoded_bytes * 3 // 2 + 256:
            raise ImageInputError("video data URL exceeds the encoded byte limit")
    elif not url.startswith("https://"):
        raise ImageInputError("videos require data URLs or public HTTPS URLs")
    elif not allow_urls:
        raise ImageInputError("video URLs are off on this server; send the video as a data URL, or start the server "
                              "with --vision-urls")
    elif len(url) > limits.max_url_chars:
        raise ImageInputError("video URL is too long")
    return VideoSource(url)


def _av():
    try:
        import av
    except ImportError:
        raise ImageInputError("video inputs require PyAV (pip install av)") from None
    return av


def sample_indices(total: int, fps: float, limits: VideoLimits) -> np.ndarray:
    """Qwen3-VL's frame choice: ``limits.fps`` frames a second (at least min_frames, at most max_frames) spread
    evenly over the whole video."""
    count = int(total / fps * limits.fps)
    count = min(max(count, limits.min_frames), limits.max_frames, total)
    return np.linspace(0, total - 1, count).round().astype(int)


def _presentation_geometry(data: bytes, limits: VideoLimits, deadline: float) -> tuple[int, float | None]:
    """Probe an unindexed clip without retaining frames or inventing its rate."""
    av = _av()
    count, origin, last = 0, None, None
    with av.open(io.BytesIO(data), mode="r") as probe:
        stream = probe.streams.video[0]
        for frame in probe.decode(stream):
            if time.monotonic() >= deadline:
                raise ImageInputError("video timeline scan exceeds the time limit")
            if frame.pts is None or frame.time_base is None:
                raise ImageInputError("video presentation timestamps are unavailable")
            seconds = float(frame.pts * frame.time_base)
            if not math.isfinite(seconds):
                raise ImageInputError("video presentation timestamps are invalid")
            if origin is None:
                origin = seconds
            if (seconds < origin or (last is not None and seconds < last)
                    or seconds - origin > limits.max_seconds):
                raise ImageInputError("video presentation timeline is invalid or exceeds the time limit")
            last = seconds
            count += 1
    return count, ((count - 1) / (last - origin)
                   if count >= 2 and last is not None and origin is not None and last > origin else None)


def decode_video(data: bytes, size: Callable[[int, int, int], tuple[int, int]], limits: VideoLimits,
                 deadline: float, *, preserve_timestamps: bool = False, minimum_frames: int = 2) -> VideoInput:
    """Decode the sampled frames of one video, each scaled to ``size(frames, height, width)`` (bicubic)."""
    if type(minimum_frames) is not int or minimum_frames <= 0:
        raise ValueError("minimum decoded video frames must be positive")
    av = _av()
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.video:
                raise ImageInputError("the video has no video stream")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            width, height = int(stream.codec_context.width or 0), int(stream.codec_context.height or 0)
            if min(width, height) <= 0 or max(width, height) > limits.max_dimension:
                raise ImageInputError("video dimensions are missing or exceed the pixel limit")
            rate = stream.average_rate or stream.guessed_rate
            measured = None
            if preserve_timestamps and (not stream.frames or rate is None):
                measured = _presentation_geometry(data, limits, deadline)
            fps = float(rate) if rate is not None else (measured[1] if measured is not None else 24.0)
            seconds = float(stream.duration * stream.time_base) if stream.duration else (
                container.duration / 1e6 if container.duration else 0.0)
            total = int(stream.frames or (measured[0] if measured is not None else round(seconds * fps)))
            if fps is None or not 0 < fps <= 1000 or total <= 0:
                raise ImageInputError("the video's frame count or rate is missing")
            if total / fps > limits.max_seconds:
                raise ImageInputError(f"videos are limited to {limits.max_seconds / 60:.0f} minutes")
            wanted = sample_indices(total, fps, limits)
            out_h, out_w = size(len(wanted), height, width)
            keep = set(int(i) for i in wanted)
            frames, indices, timestamps = [], [], []
            origin, previous_seconds = None, None
            decoded = 0
            for n, frame in enumerate(container.decode(stream)):
                if (n > wanted[-1] and not preserve_timestamps) or time.monotonic() >= deadline:
                    break
                decoded += 1
                if preserve_timestamps and decoded > total:
                    raise ImageInputError("decoded video frame count differs from its metadata")
                if min(frame.width, frame.height) <= 0 or max(frame.width, frame.height) > limits.max_dimension:
                    raise ImageInputError("decoded video dimensions exceed the pixel limit")
                if preserve_timestamps:
                    if frame.pts is None or frame.time_base is None:
                        raise ImageInputError("video presentation timestamps are unavailable")
                    seconds = float(frame.pts * frame.time_base)
                    if not math.isfinite(seconds):
                        raise ImageInputError("video presentation timestamps are invalid")
                    if origin is None:
                        origin = seconds
                    seconds -= origin
                    if (seconds < 0 or seconds > limits.max_seconds
                            or (previous_seconds is not None and seconds < previous_seconds)):
                        raise ImageInputError("video presentation timeline is invalid or exceeds the time limit")
                    previous_seconds = seconds
                if n in keep:
                    frames.append(frame.to_ndarray(format="rgb24", width=out_w, height=out_h,
                                                   interpolation="BICUBIC"))
                    indices.append(n)
                    if preserve_timestamps:
                        timestamps.append(seconds)
            if time.monotonic() >= deadline:
                raise ImageInputError("video decoding exceeds the time limit")
            if preserve_timestamps and decoded != total:
                raise ImageInputError("decoded video frame count differs from its metadata")
    except ImageInputError:
        raise
    except (OSError, ValueError, MemoryError) as exc:
        raise ImageInputError(f"video bytes are invalid or unsupported ({type(exc).__name__}); use MP4 or "
                              "WebM") from None
    if len(frames) < minimum_frames:
        if minimum_frames == 2:
            raise ImageInputError("the video has fewer than two decodable frames")
        raise ImageInputError(f"the video has fewer than {minimum_frames} decodable frames")
    return VideoInput(np.stack(frames), tuple(indices), fps,
                      tuple(timestamps) if preserve_timestamps else None)


def load_videos(sources: list[VideoSource], size: Callable[[int, int, int], tuple[int, int]], *,
                limits: VideoLimits = DEFAULT_VIDEO_LIMITS, allow_urls: bool = False,
                preserve_timestamps: bool = False, minimum_frames: int = 2) -> list[VideoInput]:
    """Bound encoded bytes and decoding time across the videos of one request."""
    if len(sources) > limits.max_videos:
        raise ImageInputError(f"a request supports at most {limits.max_videos} videos")
    total = 0
    deadline = time.monotonic() + limits.total_timeout_seconds
    output = []
    for source in sources:
        remaining = min(limits.max_encoded_bytes, limits.max_total_encoded_bytes - total)
        if remaining <= 0 or time.monotonic() >= deadline:
            raise ImageInputError("video request exceeds the total byte or time limit")
        if source.url.startswith("data:"):
            data = _data_bytes(source.url, remaining)
        elif allow_urls:
            data, _ = fetch_image(source.url, max_bytes=remaining,
                                  deadline=min(deadline, time.monotonic() + limits.timeout_seconds),
                                  max_redirects=limits.max_redirects, max_url_chars=limits.max_url_chars,
                                  media_types=VIDEO_MEDIA_TYPES)
        else:
            raise ImageInputError("video URLs are off on this server")
        total += len(data)
        if preserve_timestamps:
            output.append(decode_video(data, size, limits, deadline, preserve_timestamps=True,
                                       minimum_frames=minimum_frames))
        elif minimum_frames != 2:
            output.append(decode_video(data, size, limits, deadline, minimum_frames=minimum_frames))
        else:
            output.append(decode_video(data, size, limits, deadline))
    return output
