"""CPU-owned Gemma E2B media prepared through the pinned native HF components.

The encoders and PLE remain model-owned. Only transport, admission, ordering and
request lifetime are TensorFold responsibilities here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .audio import AudioInput, AudioLimits, AudioSource, DEFAULT_AUDIO_LIMITS, load_audios
from .images import DEFAULT_LIMITS, ImageInput, ImageLimits, ImageSource, load_images, split_images
from .images_http import ImageInputError
from .videos import DEFAULT_VIDEO_LIMITS, VideoInput, VideoLimits, VideoSource, load_videos


_ENCODER_KEYS = frozenset({
    "pixel_values", "image_position_ids", "pixel_values_videos", "video_position_ids",
    "input_features", "input_features_mask", "audio_feature_lengths",
})
_TEXT_KEYS = frozenset({"attention_mask", "mm_token_type_ids"})


def _feature_hash(inputs: Mapping[str, Any]) -> str:
    """Hash the actual native encoder inputs, not only their placeholder IDs."""
    h = hashlib.sha256(b"tensorfold-gemma-e2b-features-v1\0")
    for name in sorted(_ENCODER_KEYS & inputs.keys()):
        tensor = inputs[name]
        array = tensor.detach().contiguous().numpy()
        h.update(json.dumps((name, str(array.dtype), list(array.shape)), separators=(",", ":")).encode())
        h.update(memoryview(array).cast("B"))
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedE2BMedia:
    token_ids: tuple[int, ...]
    model_inputs: Mapping[str, Any]       # request-owned CPU tensors; never mutated by the GPU adapter
    image_hashes: tuple[str, ...]
    video_hashes: tuple[str, ...]
    audio_hashes: tuple[str, ...]
    processor_fingerprint: str
    feature_hash: str
    media_token_ids: tuple[int, ...]
    vocab_size: int = 262_144
    blocks: tuple = ()                   # E2B uses causal text attention, including media positions

    @property
    def media_key(self) -> tuple:
        return (self.processor_fingerprint, self.feature_hash, self.image_hashes,
                self.video_hashes, self.audio_hashes)

    @property
    def host_bytes(self) -> int:
        """Retained feature buffers plus a conservative prompt-ID object bound."""
        return (sum(value.numel() * value.element_size() for value in self.model_inputs.values())
                + sys.getsizeof(self.token_ids) + len(self.token_ids) * sys.getsizeof(self.vocab_size))

    @property
    def visual_tokens(self) -> int:
        image, video, _ = self.media_token_ids
        return sum(token in (image, video) for token in self.token_ids)

    @property
    def audio_tokens(self) -> int:
        return self.token_ids.count(self.media_token_ids[2])

    def continued(self, tokens: Sequence[int]) -> "PreparedE2BMedia":
        import torch

        tokens = tuple(tokens)
        if (tokens[:len(self.token_ids)] != self.token_ids
                or any(type(token) is not int or not 0 <= token < self.vocab_size for token in tokens)
                or any(token in self.media_token_ids for token in tokens[len(self.token_ids):])):
            raise ValueError("a continued E2B media prompt may append plain text tokens only")
        extra = len(tokens) - len(self.token_ids)
        inputs = dict(self.model_inputs)
        if extra:
            for name, fill in (("attention_mask", 1), ("mm_token_type_ids", 0)):
                if name in inputs:
                    prior = inputs[name]
                    tail = torch.full((1, extra), fill, dtype=prior.dtype, device="cpu")
                    inputs[name] = torch.cat((prior, tail), dim=1)
        return replace(self, token_ids=tokens, model_inputs=MappingProxyType(inputs))


def _substitute(text: str, marker: str, replacements: Sequence[str]) -> str:
    if text.count(marker) != len(replacements):
        raise ImageInputError("rendered Gemma prompt must contain exactly one marker per media input")
    remaining = iter(replacements)
    return re.sub(re.escape(marker), lambda _: next(remaining), text)


def _video_identity(video: VideoInput) -> str:
    if video.timestamp_seconds is None:
        raise ImageInputError("Gemma E2B video requires actual presentation timestamps")
    h = hashlib.sha256(b"tensorfold-e2b-video-v1\0" + video.content_hash.encode())
    h.update(np.asarray(video.timestamp_seconds, dtype=">f8").tobytes())
    return h.hexdigest()


class GemmaE2BProcessor:
    """Native image/video/audio processors, with bounded CPU request staging."""

    audio = True
    videos = True
    responses_video = True

    def __init__(self, raw: dict, processor: Any, size_function: Callable, *,
                 fingerprint: str, allow_urls: bool = False,
                 image_limits: ImageLimits = DEFAULT_LIMITS,
                 video_limits: VideoLimits = DEFAULT_VIDEO_LIMITS,
                 audio_limits: AudioLimits = DEFAULT_AUDIO_LIMITS):
        self.raw, self.processor, self.size_function = raw, processor, size_function
        self.processor_fingerprint = fingerprint
        self.allow_urls = allow_urls
        self.image_limits, self.video_limits, self.audio_limits = image_limits, video_limits, audio_limits
        text = raw.get("text_config", raw)
        if raw.get("model_type") != "gemma4" or not raw.get("audio_config") or not raw.get("vision_config"):
            raise ValueError("E2B media requires a complete Gemma4 audio/vision checkpoint")
        self.vocab, self.context_limit = text["vocab_size"], text["max_position_embeddings"]
        self.media_token_ids = tuple(raw[name] for name in ("image_token_id", "video_token_id", "audio_token_id"))
        self._processor_lock = threading.Lock()
        for name, expected in zip(("image_token", "video_token", "audio_token"), self.media_token_ids):
            marker = getattr(processor, name, None)
            if (not isinstance(marker, str) or not marker
                    or processor.tokenizer.convert_tokens_to_ids(marker) != expected):
                raise ValueError("native Gemma processor marker differs from checkpoint: " + name)
        if processor.feature_extractor.sampling_rate != audio_limits.sample_rate:
            raise ValueError("Gemma audio feature sample rate differs from canonical decoder")
        if audio_limits.max_seconds > 30:
            raise ValueError("native Gemma audio supports at most 30 seconds per segment")
        if processor.audio_seq_length != 750 or processor.audio_ms_per_token != 40:
            raise ValueError("unsupported Gemma audio segment/token geometry")
        for component in (processor.image_processor, processor.video_processor):
            if (type(component.patch_size) is not int or component.patch_size <= 0
                    or type(component.pooling_kernel_size) is not int or component.pooling_kernel_size <= 0
                    or type(component.max_soft_tokens) is not int or component.max_soft_tokens <= 0):
                raise ValueError("invalid native Gemma processor geometry")

    @classmethod
    def from_directory(cls, model_dir: str | Path, *, allow_urls: bool = False,
                       image_limits: ImageLimits = DEFAULT_LIMITS,
                       video_limits: VideoLimits = DEFAULT_VIDEO_LIMITS,
                       audio_limits: AudioLimits = DEFAULT_AUDIO_LIMITS) -> "GemmaE2BProcessor":
        import transformers
        from transformers import AutoProcessor
        from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
        from tensorfold.checkpoint_metadata import read_config

        root = Path(model_dir)
        raw = read_config(root)
        processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        h = hashlib.sha256(b"tensorfold-gemma-e2b-processor-v1\0" + transformers.__version__.encode())
        for name in ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja"):
            path = root / name
            if path.exists():
                h.update(name.encode())
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
        return cls(raw, processor, get_aspect_ratio_preserving_size, fingerprint=h.hexdigest(),
                   allow_urls=allow_urls, image_limits=image_limits, video_limits=video_limits,
                   audio_limits=audio_limits)

    def _size(self, height: int, width: int, component) -> tuple[int, int]:
        return self.size_function(height=height, width=width, patch_size=component.patch_size,
                                  max_patches=component.max_soft_tokens * component.pooling_kernel_size**2,
                                  pooling_kernel_size=component.pooling_kernel_size)

    def video_size(self, frames: int, height: int, width: int) -> tuple[int, int]:
        if type(frames) is not int or frames <= 0:
            raise ImageInputError("Gemma video must contain frames")
        return self._size(height, width, self.processor.video_processor)

    def resource_geometry(self, context_limit: int | None = None,
                          max_visual_tokens: int | None = None) -> dict[str, int]:
        """Input geometry for admission; actual encoder workspace is model-owned."""
        context = self.context_limit if context_limit is None else context_limit
        image, video = self.processor.image_processor, self.processor.video_processor
        return {
            "max_prompt_tokens": context,
            "max_visual_tokens": self.image_limits.max_visual_tokens if max_visual_tokens is None else max_visual_tokens,
            "max_images": self.image_limits.max_images,
            "max_video_frames": self.video_limits.max_videos * self.video_limits.max_frames,
            "image_patches_per_input": image.max_soft_tokens * image.pooling_kernel_size**2,
            "video_patches_per_frame": video.max_soft_tokens * video.pooling_kernel_size**2,
            "patch_pixels": 3 * image.patch_size**2,
            "max_audio_samples_per_clip": self.audio_limits.max_samples,
            "max_padded_audio_samples": context * self.audio_limits.sample_rate * self.processor.audio_ms_per_token // 1000,
        }

    def prepare_messages(self, messages: list[dict], render: Callable[[list[dict]], str], *,
                         max_prompt_tokens: int | None, max_visual_tokens: int | None = None,
                         image_limits: ImageLimits | None = None) -> PreparedE2BMedia:
        context = self.context_limit if max_prompt_tokens is None else max_prompt_tokens
        if type(context) is not int or not 0 < context <= self.context_limit:
            raise ValueError("Gemma media requires a positive admitted context within the checkpoint limit")
        limits = self._request_image_limits(image_limits)
        template, sources = split_images(
            messages, limits=limits, allow_urls=self.allow_urls, allow_videos=True,
            video_limits=self.video_limits, allow_audio=True, audio_limits=self.audio_limits)
        audio_sources = [s for s in sources if isinstance(s, AudioSource)]
        sample_budget = self.resource_geometry(context)["max_padded_audio_samples"]
        # Each segment needs its BOA/EOA markers even before soft tokens. The
        # actual feature extraction below is per clip, never B*longest padding.
        if len(audio_sources) * 2 > context:
            raise ImageInputError("audio segment markers exceed the admitted context")
        images = load_images([s for s in sources if isinstance(s, ImageSource)],
                             limits=limits, allow_urls=self.allow_urls)
        videos = load_videos([s for s in sources if isinstance(s, VideoSource)], self.video_size,
                             limits=self.video_limits, allow_urls=self.allow_urls, preserve_timestamps=True,
                             minimum_frames=1)
        audios = load_audios(audio_sources, max_total_samples=sample_budget, limits=self.audio_limits)
        return self.prepare(render(template), images, videos=videos, audios=audios,
                            max_prompt_tokens=context, max_visual_tokens=max_visual_tokens, image_limits=limits)

    def _request_image_limits(self, limits: ImageLimits | None) -> ImageLimits:
        if limits is None:
            return self.image_limits
        # App count/token overrides cannot relax the frontend's encoded/decoded
        # byte, pixel, URL or time budgets, and never mutate shared state.
        return replace(self.image_limits, max_images=limits.max_images,
                       max_visual_tokens=limits.max_visual_tokens)

    def prepare(self, rendered_prompt: str, images: Sequence[ImageInput], *,
                videos: Sequence[VideoInput] = (), audios: Sequence[AudioInput] = (),
                max_prompt_tokens: int | None, max_visual_tokens: int | None = None,
                image_limits: ImageLimits | None = None) -> PreparedE2BMedia:
        import torch
        from transformers.video_utils import VideoMetadata

        context = self.context_limit if max_prompt_tokens is None else max_prompt_tokens
        limits = self._request_image_limits(image_limits)
        visual_limit = limits.max_visual_tokens if max_visual_tokens is None else max_visual_tokens
        if (not isinstance(rendered_prompt, str) or not (images or videos or audios)
                or type(context) is not int or not 0 < context <= self.context_limit
                or type(visual_limit) is not int or visual_limit <= 0):
            raise ValueError("Gemma media requires inputs and positive admitted prompt/media budgets")
        if len(images) > limits.max_images or len(videos) > self.video_limits.max_videos:
            raise ImageInputError("Gemma media exceeds the admitted image/video count")
        # Compute native spatial output counts before allocating padded patches.
        image_count = 0
        for item in images:
            h, w = self._size(item.height, item.width, self.processor.image_processor)
            image_count += h * w // (self.processor.image_processor.patch_size *
                                     self.processor.image_processor.pooling_kernel_size)**2
        video_count = 0
        for item in videos:
            if len(item.indices) > self.video_limits.max_frames:
                raise ImageInputError("Gemma video exceeds the sampled frame limit")
            h, w = self.video_size(len(item.indices), int(item.frames.shape[1]), int(item.frames.shape[2]))
            video_count += len(item.indices) * h * w // (
                self.processor.video_processor.patch_size * self.processor.video_processor.pooling_kernel_size)**2
            _video_identity(item)
        if image_count + video_count > visual_limit:
            raise ImageInputError("Gemma processed images/videos exceed the visual-token budget")
        sample_budget = self.resource_geometry(context)["max_padded_audio_samples"]
        if audios:
            if (len(audios) * 2 > context
                    or sum(len(a.samples) for a in audios) > sample_budget
                    or any(a.sample_rate != self.audio_limits.sample_rate
                           or len(a.samples) > self.audio_limits.max_samples for a in audios)):
                raise ImageInputError("audio segments exceed the admitted context/sample budget")
        kwargs: dict[str, Any] = {}
        with self._processor_lock:
            text = rendered_prompt
            if images:
                values = self.processor.image_processor(
                    images=[x.to_pil() for x in images], do_convert_rgb=True, return_tensors="pt")
                text = _substitute(text, self.processor.image_token,
                                   [self.processor.replace_image_token(values, i) for i in range(len(images))])
                kwargs.update({k: values[k] for k in ("pixel_values", "image_position_ids")})
            elif self.processor.image_token in text:
                raise ImageInputError("image placeholders have no matching image input")
            if videos:
                pixels, positions, replacements = [], [], []
                for item in videos:
                    class PresentationMetadata(VideoMetadata):
                        @property
                        def timestamps(self):
                            return list(self.presentation_times)
                    metadata = PresentationMetadata(total_num_frames=max(item.indices) + 1, fps=item.fps,
                                                    frames_indices=list(item.indices))
                    metadata.presentation_times = item.timestamp_seconds
                    values = self.processor.video_processor(
                        videos=[np.array(item.frames, copy=True)], do_sample_frames=False,
                        video_metadata=[metadata], return_metadata=True, return_tensors="pt")
                    pixels.append(values["pixel_values_videos"])
                    positions.append(values["video_position_ids"])
                    replacements.append(self.processor.replace_video_token(values, 0))
                text = _substitute(text, self.processor.video_token, replacements)
                # Native encoder flattens (video,frame) and encodes frames
                # independently. Coalescing the frame axis avoids padding or
                # dropping clips with unequal frame counts.
                kwargs["pixel_values_videos"] = torch.cat(pixels, dim=1)
                kwargs["video_position_ids"] = torch.cat(positions, dim=1)
            elif self.processor.video_token in text:
                raise ImageInputError("video placeholders have no matching video input")
            if audios:
                features, masks, lengths, replacements = [], [], [], []
                batch_longest = max(len(audio.samples) for audio in audios)
                extraction_lengths = [min(batch_longest, max(
                    len(audio.samples), self.processor.feature_extractor.frame_length + 1)) for audio in audios]
                # Semicausal left padding is shorter than the analysis window,
                # so mel rows cannot exceed padded_samples // hop + 1. Reserve
                # retained feature rows before any per-clip extraction or cat.
                hop = self.processor.feature_extractor.hop_length
                row_budget = sum(((length + 127) // 128 * 128) // hop + 1
                                 for length in extraction_lengths)
                retained_rows = 0
                for audio, length in zip(audios, extraction_lengths):
                    # A public per-clip extractor bounds native waveform/FFT
                    # padding independently of the number of short neighbors.
                    # Reserve one analysis window for sub-frame clips; all
                    # extra positions remain invalid in the native mask.
                    values = self.processor.feature_extractor(
                        [np.asarray(audio.samples)], sampling_rate=self.audio_limits.sample_rate,
                        padding="max_length", truncation=False, max_length=length,
                        pad_to_multiple_of=128, return_tensors="pt")
                    if values["input_features"].shape[1] == 0:
                        raise ImageInputError("audio batch produces no native feature frames")
                    retained_rows += values["input_features"].shape[1]
                    if retained_rows > row_budget:
                        raise ImageInputError("native audio feature geometry exceeds the reserved row budget")
                    features.append(values["input_features"])
                    masks.append(values["input_features_mask"])
                    lengths.append(values["input_features"].shape[1])
                    replacements.append(self.processor.replace_audio_token(values, 0))
                text = _substitute(text, self.processor.audio_token, replacements)
                kwargs["input_features"] = torch.cat(features, dim=1)
                kwargs["input_features_mask"] = torch.cat(masks, dim=1)
                kwargs["audio_feature_lengths"] = torch.tensor(lengths, dtype=torch.int64)
            elif self.processor.audio_token in text:
                raise ImageInputError("audio placeholders have no matching audio input")
            tokenized = self.processor.tokenizer(
                text, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)
            tokenized["mm_token_type_ids"] = torch.tensor(
                self.processor.create_mm_token_type_ids(tokenized["input_ids"]), dtype=torch.int64)
        ids = tokenized["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ImageInputError("native Gemma processor returned invalid input IDs")
        tokens = tuple(int(i) for i in ids[0].tolist())
        if not tokens or len(tokens) > context:
            raise ImageInputError(f"context length exceeds {context} tokens after native Gemma media expansion")
        if any(not 0 <= token < self.vocab for token in tokens):
            raise ImageInputError("native Gemma processor returned an out-of-vocabulary token")
        kwargs.update({k: tokenized[k] for k in _TEXT_KEYS if k in tokenized})
        for name, value in kwargs.items():
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise ImageInputError("native Gemma processor must return request-owned CPU tensors")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ImageInputError("native Gemma processor returned nonfinite feature data")
            kwargs[name] = value.detach().contiguous()
        return PreparedE2BMedia(tokens, MappingProxyType(kwargs), tuple(i.content_hash for i in images),
                                tuple(_video_identity(v) for v in videos), tuple(a.content_hash for a in audios),
                                self.processor_fingerprint, _feature_hash(kwargs), self.media_token_ids,
                                vocab_size=self.vocab)

    @staticmethod
    def continued(prepared: PreparedE2BMedia, tokens: Sequence[int]) -> PreparedE2BMedia:
        return prepared.continued(tokens)
