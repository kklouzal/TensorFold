"""Native Gemma 4 vision, with CPU processor data and request-owned features.

The checkpoint owns resizing, patch order and placeholder expansion. The native
tower consumes those patches; it never loads checkpoint Python or converts the
floating-point vision tower to a different weight format. Language positions
remain ordinary sequential positions. ``blocks`` must remain whole during
prefill because sliding attention is bidirectional inside each visual block.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ProcessorGeometry:
    image_soft_tokens: int
    video_soft_tokens: int
    patch_size: int
    pooling_kernel_size: int

    @property
    def image_patches(self) -> int:
        return self.image_soft_tokens * self.pooling_kernel_size**2

    @property
    def video_patches(self) -> int:
        return self.video_soft_tokens * self.pooling_kernel_size**2


def processor_geometry(model_dir: str | Path) -> ProcessorGeometry:
    """Resolve processor geometry without importing Torch or reading model weights.

    Missing component fields use the installed Transformers 5.17 Gemma4 class
    defaults. Explicit checkpoint fields always take precedence. The instantiated
    processor is checked against this preparation result by ``from_directory``.
    """
    from tensorfold.checkpoint_metadata import read_config, read_json

    c = validate_vision_config(read_config(model_dir))
    metadata = read_json(model_dir, "processor_config.json") or {}
    components = []
    for name, default in (("image_processor", 280), ("video_processor", 70)):
        options = metadata.get(name, {})
        if not isinstance(options, dict):
            raise ValueError(f"Gemma processor {name} metadata must be an object")
        patch, pool, count = (options.get("patch_size", 16), options.get("pooling_kernel_size", 3),
                              options.get("max_soft_tokens", default))
        if (type(patch) is not int or type(pool) is not int or type(count) is not int
                or patch != c["patch_size"] or pool != c["pooling_kernel_size"]
                or count not in (70, 140, 280, 560, 1120)):
            raise ValueError(f"Gemma processor {name} geometry is unsupported or differs from the tower")
        components.append(count)
    return ProcessorGeometry(*components, c["patch_size"], c["pooling_kernel_size"])


def validate_vision_config(raw: dict) -> dict:
    """Validate the native vision contract before loading any tensor payload."""
    config = raw.get("vision_config")
    if raw.get("model_type") != "gemma4" or not isinstance(config, dict):
        raise ValueError("Gemma CUDA vision requires a complete Gemma 4 vision checkpoint")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "patch_size", "pooling_kernel_size",
                "position_embedding_size"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"invalid Gemma vision configuration: {key}")
    h, heads, dim = (config[k] for k in ("hidden_size", "num_attention_heads", "head_dim"))
    if h != heads * dim or config["num_key_value_heads"] != heads or dim % 4:
        raise ValueError("Gemma vision requires equal query/key/value heads and axial head width divisible by four")
    rope = config.get("rope_parameters") or {}
    if (rope.get("rope_type", "default") not in ("default", "axial")
            or not isinstance(rope.get("rope_theta"), (int, float))
            or not math.isfinite(rope["rope_theta"]) or rope["rope_theta"] <= 0):
        raise ValueError("Gemma vision requires the checkpoint's static axial rotary frequencies")
    eps = config.get("rms_norm_eps")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Gemma vision requires a finite positive RMS epsilon")
    if (config.get("hidden_activation") != "gelu_pytorch_tanh" or config.get("attention_bias", False)
            or config.get("use_clipped_linears", False) or config.get("attention_dropout", 0.0) != 0.0):
        raise ValueError("unsupported Gemma vision activation, biased/clipped linears or attention dropout")
    if type(raw.get("text_config", {}).get("hidden_size")) is not int or raw["text_config"]["hidden_size"] <= 0:
        raise ValueError("Gemma vision requires the language embedding width")
    for key in ("image_token_id", "video_token_id"):
        if type(raw.get(key)) is not int or raw[key] < 0:
            raise ValueError(f"Gemma vision requires a valid {key}")
    if raw["image_token_id"] == raw["video_token_id"]:
        raise ValueError("Gemma image and video token IDs must differ")
    return config


def vision_shapes(raw: dict) -> dict[str, tuple[int, ...]]:
    """The floating-point tower's exact tensor schema, separate from the projector."""
    c = validate_vision_config(raw)
    h, m, d = c["hidden_size"], c["intermediate_size"], c["head_dim"]
    shapes = {"patch_embedder.input_proj.weight": (h, 3 * c["patch_size"]**2),
              "patch_embedder.position_embedding_table": (2, c["position_embedding_size"], h)}
    if c.get("standardize", False):
        shapes.update(std_bias=(h,), std_scale=(h,))
    for i in range(c["num_hidden_layers"]):
        base = f"encoder.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm"):
            shapes[base + name + ".weight"] = (h,)
        for name in ("q", "k", "v", "o"):
            shapes[base + f"self_attn.{name}_proj.linear.weight"] = (h, h)
        for name in ("q", "k"):
            shapes[base + f"self_attn.{name}_norm.weight"] = (d,)
        for name in ("gate", "up"):
            shapes[base + f"mlp.{name}_proj.linear.weight"] = (m, h)
        shapes[base + "mlp.down_proj.linear.weight"] = (h, m)
    return {"vision_tower." + key: value for key, value in shapes.items()}


def vision_weight_bytes(raw: dict, headers: Mapping[str, dict]) -> int:
    """Validate tower headers and return resident bytes without reading payloads.

    The shared checkpoint reader validates ranges, overlap, file size and index
    identity. This function validates the family schema. The caller accounts for
    the native affine projector separately, using its own per-module metadata.
    """
    sizes = {"BF16": 2}
    shapes = vision_shapes(raw)
    actual = {key: value for key, value in headers.items() if key.startswith("vision_tower.")}
    if set(actual) != set(shapes):
        raise ValueError("Gemma vision checkpoint must contain the complete floating-point tower")
    for key, shape in shapes.items():
        if tuple(actual[key].get("shape", ())) != shape or actual[key].get("dtype") not in sizes:
            raise ValueError(f"invalid or unsupported Gemma vision tensor: {key}")
    return sum(math.prod(shapes[key]) * sizes[info["dtype"]] for key, info in actual.items())


def projector_spec(raw: dict, headers: Mapping[str, dict]) -> tuple[int, int]:
    """Validate original packed projector tensors against their own quantization."""
    from .weights import quant_spec

    c = validate_vision_config(raw)
    name = "embed_vision.embedding_projection"
    bits, group = quant_spec(raw, name)
    h, width = raw["text_config"]["hidden_size"], c["hidden_size"]
    if width % group or width * bits % 32:
        raise ValueError("Gemma vision projector width does not fit its quantization layout")
    expected = {name + ".weight": ((h, width * bits // 32), "U32"),
                name + ".scales": ((h, width // group), "BF16"),
                name + ".biases": ((h, width // group), "BF16")}
    found = {key for key in headers if key.startswith("embed_vision.")}
    if found != set(expected):
        raise ValueError("Gemma vision requires its complete native affine projector")
    for key, (shape, dtype) in expected.items():
        if tuple(headers[key].get("shape", ())) != shape or headers[key].get("dtype") != dtype:
            raise ValueError(f"invalid Gemma vision projector tensor: {key}")
    return bits, group


@dataclass(frozen=True, slots=True)
class VisionFrame:
    pixels: np.ndarray                         # [padded patches, 3 * patch_size**2], CPU float32
    positions: np.ndarray                      # [padded patches, 2], CPU int64; (-1,-1) is padding
    soft_tokens: int
    kind: str                                  # image or video


@dataclass(frozen=True, slots=True)
class PreparedGemmaVision:
    token_ids: tuple[int, ...]
    frames: tuple[VisionFrame, ...]              # prompt order, including one entry per video frame
    blocks: tuple[tuple[int, int], ...]
    image_hashes: tuple[str, ...]
    video_hashes: tuple[str, ...]

    @property
    def visual_tokens(self) -> int:
        return sum(end - start for start, end in self.blocks)


@dataclass(frozen=True, slots=True)
class EncodedGemmaVision:
    rows: tuple[int, ...]
    features: Any                              # [visual rows, text hidden], owned by this request
    blocks: tuple[tuple[int, int], ...]


def continued(prepared: PreparedGemmaVision, tokens: Sequence[int], config: dict) -> PreparedGemmaVision:
    """Reuse immutable prepared media when a continuation adds text tokens only."""
    tokens = tuple(int(token) for token in tokens)
    if tokens[:len(prepared.token_ids)] != prepared.token_ids:
        raise ValueError("a continued Gemma media prompt must start with the prepared prompt's tokens")
    if any(token in (config["image_token_id"], config["video_token_id"])
           for token in tokens[len(prepared.token_ids):]):
        raise ValueError("a continued Gemma media prompt may add text only")
    return replace(prepared, token_ids=tokens)


def media_blocks(tokens: Sequence[int], image_token: int, video_token: int,
                 kinds_and_counts: Sequence[tuple[str, int]]) -> tuple[tuple[int, int], ...]:
    """Match every expanded image/frame run to exactly one processor result."""
    runs, cursor = [], 0
    for kind, count in kinds_and_counts:
        if kind not in ("image", "video") or type(count) is not int or count <= 0:
            raise ValueError("invalid Gemma visual frame kind or soft-token count")
        while cursor < len(tokens) and tokens[cursor] not in (image_token, video_token):
            cursor += 1
        expected = image_token if kind == "image" else video_token
        end = cursor + count
        if end > len(tokens) or any(token != expected for token in tokens[cursor:end]):
            raise ValueError("Gemma image/frame placeholders do not match processed features")
        if end < len(tokens) and tokens[end] == expected:
            raise ValueError("Gemma image/frame placeholder run is longer than its features")
        runs.append((cursor, end))
        cursor = end
    if any(token in (image_token, video_token) for token in tokens[cursor:]):
        raise ValueError("Gemma prompt contains visual placeholders without features")
    return tuple(runs)


def _frame(pixels, positions, config: dict, kind: str) -> VisionFrame:
    """Own and validate standard-processor output before accelerator allocation."""
    p, xy = np.array(pixels, copy=True), np.array(positions, copy=True)
    k, patch = config["pooling_kernel_size"], config["patch_size"]
    if (p.ndim != 2 or p.shape[1] != 3 * patch**2 or p.shape[0] == 0 or p.shape[0] % k**2
            or p.dtype != np.float32 or xy.shape != (p.shape[0], 2)
            or not np.issubdtype(xy.dtype, np.integer) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)):
        raise ValueError("Gemma processor returned invalid patch pixels or positions")
    padding = np.all(xy == -1, axis=-1)
    n = int((~padding).sum())
    if n == 0 or n % k**2 or np.any(padding[:n]) or np.any(~padding[n:]):
        raise ValueError("Gemma patch padding must follow a nonempty pool-aligned spatial grid")
    valid = xy[:n]
    width, height = int(valid[:, 0].max()) + 1, int(valid[:, 1].max()) + 1
    if (width <= 0 or height <= 0 or width % k or height % k or width * height != n
            or max(width, height) > config["position_embedding_size"]
            or not np.array_equal(valid[:, 0], np.arange(n) % width)
            or not np.array_equal(valid[:, 1], np.arange(n) // width)):
        raise ValueError("Gemma patch positions must describe the processor's row-major spatial grid")
    if np.any(xy[n:] != -1):
        raise ValueError("Gemma patch padding positions must be (-1, -1)")
    xy = xy.astype(np.int64, copy=False)
    p.setflags(write=False)
    xy.setflags(write=False)
    return VisionFrame(p, xy, n // k**2, kind)


def _video_batch(pixels: np.ndarray, positions: np.ndarray, counts: Sequence[int]):
    """Flatten pinned5.17 [video, frame, patch, channel] output after shape proof."""
    if (pixels.ndim != 4 or positions.ndim != 4 or pixels.shape[:2] != positions.shape[:2]
            or pixels.shape[0] != len(counts) or any(count != pixels.shape[1] for count in counts)
            or pixels.shape[2] != positions.shape[2] or positions.shape[3] != 2):
        raise ValueError("Gemma processor video batch does not match the declared video/frame counts")
    frames = sum(counts)
    return pixels.reshape(frames, *pixels.shape[2:]), positions.reshape(frames, *positions.shape[2:])


class GemmaVisionProcessor:
    """Local standard Transformers processor; decoding and resource limits stay external."""

    def __init__(self, raw: dict, processor: Any, size_function: Any, geometry: ProcessorGeometry | None = None):
        self.config = validate_vision_config(raw)
        self.processor, self.size_function = processor, size_function
        # Fast tokenizers configure padding/truncation state during __call__.
        # Keep that mutable CPU component private to one call; GPU encoding has
        # no shared request scratch and uses the caller's execution stream.
        self._processor_lock = threading.Lock()
        self.image_token, self.video_token = raw["image_token_id"], raw["video_token_id"]
        tokenizer = processor.tokenizer
        for name, token_id in (("image_token", self.image_token), ("video_token", self.video_token)):
            token = getattr(processor, name, None)
            if not isinstance(token, str) or not token or tokenizer.convert_tokens_to_ids(token) != token_id:
                raise ValueError(f"Gemma processor {name} differs from the checkpoint")
        for name in ("image_processor", "video_processor"):
            component = getattr(processor, name)
            # The pinned video processor defaults to normalization with mean0
            # and std1. Those exact finite operations preserve rescaled pixels;
            # a nonidentity normalization would change the tower's input.
            mean = np.asarray(getattr(component, "image_mean", None))
            std = np.asarray(getattr(component, "image_std", None))
            identity = (mean.shape in ((), (1,), (3,)) and std.shape in ((), (1,), (3,))
                        and np.all(mean == 0) and np.all(std == 1))
            if (component.patch_size != self.config["patch_size"]
                    or component.pooling_kernel_size != self.config["pooling_kernel_size"]
                    or not component.do_resize or not component.do_rescale or (component.do_normalize and not identity)
                    or component.rescale_factor != 1 / 255):
                raise ValueError(f"Gemma {name} is incompatible with the native vision tower")
        actual = ProcessorGeometry(processor.image_processor.max_soft_tokens, processor.video_processor.max_soft_tokens,
                                   self.config["patch_size"], self.config["pooling_kernel_size"])
        if geometry is not None and actual != geometry:
            raise ValueError("Gemma instantiated processor geometry differs from the admitted checkpoint metadata")
        self.geometry = actual

    @classmethod
    def from_directory(cls, model_dir: str | Path):
        from tensorfold.checkpoint_metadata import read_config, read_json, read_text
        from transformers import GemmaTokenizer
        from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
        from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
        from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
        from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
        from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor

        path = Path(model_dir)
        raw = read_config(path)
        validate_vision_config(raw)
        geometry = processor_geometry(path)
        metadata = read_json(path, "processor_config.json") or {}
        if metadata.get("processor_class", "Gemma4Processor") != "Gemma4Processor":
            raise ValueError("native Gemma vision requires standard Gemma4Processor metadata")
        components = {}
        for name, component_class, type_key in (
                ("image_processor", Gemma4ImageProcessor, "image_processor_type"),
                ("video_processor", Gemma4VideoProcessor, "video_processor_type"),
                ("feature_extractor", Gemma4AudioFeatureExtractor, "feature_extractor_type")):
            options = metadata.get(name, {})
            if not isinstance(options, dict) or options.get(type_key, component_class.__name__) != component_class.__name__:
                raise ValueError(f"native Gemma vision requires standard {name} metadata")
            components[name] = component_class.from_dict(dict(options))
        tokenizer = GemmaTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        template = read_text(path, "chat_template.jinja")
        if template is None:
            template = metadata.get("chat_template")
        # The MLX checkpoint stores all component settings in this combined
        # file. Transformers 5.17's video from_pretrained path instead requires
        # video_preprocessor_config.json even when a combined file is present.
        # Construct its standard classes from the authoritative nested data;
        # an absent component uses that pinned class's documented defaults.
        options = {key: value for key, value in metadata.items()
                   if key not in {*components, "processor_class", "chat_template"}}
        processor = Gemma4Processor(tokenizer=tokenizer, chat_template=template, **components, **options)
        return cls(raw, processor, get_aspect_ratio_preserving_size, geometry=geometry)

    def _size(self, height: int, width: int, component) -> tuple[int, int]:
        if any(type(value) is not int or value <= 0 for value in (height, width)):
            raise ValueError("Gemma media dimensions must be positive integers")
        return self.size_function(height=height, width=width, patch_size=component.patch_size,
                                  max_patches=component.max_soft_tokens * component.pooling_kernel_size**2,
                                  pooling_kernel_size=component.pooling_kernel_size)

    def video_size(self, frames: int, height: int, width: int) -> tuple[int, int]:
        if type(frames) is not int or frames <= 0:
            raise ValueError("Gemma video must contain frames")
        return self._size(height, width, self.processor.video_processor)

    def prepare(self, rendered_prompt: str, images: Sequence[Any], *, videos: Sequence[Any] = (),
                max_prompt_tokens: int | None, max_visual_tokens: int = 4096,
                max_image_tokens: int | None = None) -> PreparedGemmaVision:
        if (not images and not videos) or type(max_visual_tokens) is not int or max_visual_tokens <= 0:
            raise ValueError("Gemma preprocessing needs media and a positive visual-token budget")
        if max_prompt_tokens is not None and (type(max_prompt_tokens) is not int or max_prompt_tokens <= 0):
            raise ValueError("Gemma preprocessing needs a positive prompt-token budget")
        if max_image_tokens is not None and (type(max_image_tokens) is not int or max_image_tokens <= 0):
            raise ValueError("Gemma image-token budget must be positive")
        if (rendered_prompt.count(self.processor.image_token) != len(images)
                or rendered_prompt.count(self.processor.video_token) != len(videos)):
            raise ValueError("Gemma rendered prompt must contain exactly one marker per image or video")
        # Determine exact output counts before the processor allocates patch arrays.
        image_counts, video_counts = [], []
        for image in images:
            h, w = self._size(image.height, image.width, self.processor.image_processor)
            count = h * w // (self.config["patch_size"] * self.config["pooling_kernel_size"])**2
            if max_image_tokens is not None and count > max_image_tokens:
                raise ValueError("Gemma image exceeds the image-token budget")
            image_counts.append(count)
        for video in videos:
            h, w = self.video_size(int(video.frames.shape[0]), int(video.frames.shape[1]), int(video.frames.shape[2]))
            count = h * w // (self.config["patch_size"] * self.config["pooling_kernel_size"])**2
            video_counts.append((len(video.indices), count))
        if len({frames for frames, _ in video_counts}) > 1:
            # Pinned5.17 stacks clips along a video axis. The initial native
            # trial admits that declared region and never pads or drops frames.
            raise ValueError("Gemma video requests require clips with the same sampled frame count")
        if sum(image_counts) + sum(n * count for n, count in video_counts) > max_visual_tokens:
            raise ValueError("Gemma processed media exceeds the visual-token budget")
        kwargs = dict(text=rendered_prompt, return_tensors="np", add_special_tokens=False,
                      padding=False, truncation=False)
        if images:
            kwargs["images"] = [[image.to_pil() for image in images]]
        if videos:
            from transformers.video_utils import VideoMetadata

            kwargs.update(videos=[np.array(video.frames, copy=True) for video in videos], do_sample_frames=False,
                          video_metadata=[VideoMetadata(total_num_frames=max(video.indices) + 1, fps=video.fps,
                                                        frames_indices=list(video.indices)) for video in videos])
        with self._processor_lock:
            result = self.processor(**kwargs)
        ids = np.asarray(result["input_ids"])
        if ids.ndim != 2 or ids.shape[0] != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Gemma processor returned invalid prompt token IDs")
        tokens = tuple(int(i) for i in ids[0])
        if max_prompt_tokens is not None and len(tokens) > max_prompt_tokens:
            raise ValueError(f"context length exceeds {max_prompt_tokens} tokens after Gemma media expansion")
        by_kind = {"image": [], "video": []}
        for kind, pixels_key, positions_key, expected in (
                ("image", "pixel_values", "image_position_ids", image_counts),
                ("video", "pixel_values_videos", "video_position_ids", [c for n, c in video_counts for _ in range(n)])):
            if not expected:
                continue
            pixels, positions = np.asarray(result[pixels_key]), np.asarray(result[positions_key])
            if kind == "video":
                pixels, positions = _video_batch(pixels, positions, [n for n, _ in video_counts])
            if pixels.ndim != 3 or positions.ndim != 3 or len(pixels) != len(expected) or len(positions) != len(expected):
                raise ValueError("Gemma processor returned an incorrect image/frame count")
            for p, xy, count in zip(pixels, positions, expected):
                frame = _frame(p, xy, self.config, kind)
                if frame.soft_tokens != count:
                    raise ValueError("Gemma processor geometry differs from its declared resize geometry")
                by_kind[kind].append(frame)
        # Reorder processor image/video batches to the actual mixed prompt order.
        queues = {kind: iter(frames) for kind, frames in by_kind.items()}
        frames, cursor = [], 0
        while cursor < len(tokens):
            token = tokens[cursor]
            if token not in (self.image_token, self.video_token):
                cursor += 1
                continue
            kind = "image" if token == self.image_token else "video"
            frame = next(queues[kind], None)
            if frame is None:
                raise ValueError("Gemma visual placeholders have no matching processed image/frame")
            frames.append(frame)
            cursor += frame.soft_tokens
        if any(next(queue, None) is not None for queue in queues.values()):
            raise ValueError("Gemma processed images/frames have no matching prompt placeholders")
        blocks = media_blocks(tokens, self.image_token, self.video_token, [(f.kind, f.soft_tokens) for f in frames])
        return PreparedGemmaVision(tokens, tuple(frames), blocks, tuple(i.content_hash for i in images),
                                   tuple(v.content_hash for v in videos))


def _norm(x, weight, eps: float):
    import torch

    values = x.float()
    values = values * torch.pow(values.square().mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        values = values * weight.float()
    return values.to(x.dtype)


def _rope(x, positions, inverse):
    import torch

    axes = x.chunk(2, dim=-1)
    rotated = []
    for i, values in enumerate(axes):
        frequency = positions[..., i, None].float() * inverse
        angle = torch.cat((frequency, frequency), dim=-1)
        cos, sin = angle.cos().to(x.dtype).unsqueeze(2), angle.sin().to(x.dtype).unsqueeze(2)
        first, second = values.chunk(2, dim=-1)
        rotated.append(values * cos + torch.cat((-second, first), dim=-1) * sin)
    return torch.cat(rotated, dim=-1)


class GemmaCudaVision:
    """Read-only resident tower; encode scratch and returned features are request owned.

    ``weights`` and ``projector`` are loaded once by the native model owner.
    All tower parameters must retain their checkpoint floating-point dtype.
    The projector is the native affine Linear with its own module quantization.
    Calls run on the engine's execution stream; model readiness must be published
    before the engine starts concurrent request streams.
    No global mutable scratch, input mutation, cross-request feature cache or
    hidden CPU model fallback exists here.
    """

    def __init__(self, raw: dict, weights: Mapping[str, Any], projector: Any,
                 frontend: GemmaVisionProcessor | None = None):
        import torch

        self.config = validate_vision_config(raw)
        schema = vision_shapes(raw)
        selected = {key: value for key, value in weights.items() if key.startswith("vision_tower.")}
        if set(selected) != set(schema):
            raise ValueError("Gemma vision requires its complete floating-point tensor schema")
        first = selected["vision_tower.patch_embedder.input_proj.weight"]
        if first.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Gemma vision requires native floating-point tower weights")
        for key, value in selected.items():
            if tuple(value.shape) != schema[key] or value.device != first.device or value.dtype != first.dtype:
                raise ValueError(f"Gemma vision tensor shape/device/dtype differs: {key}")
        self.weights = {key.removeprefix("vision_tower."): value for key, value in selected.items()}
        self.projector, self.frontend = projector, frontend
        self.raw_config = raw
        self.device, self.dtype = first.device, first.dtype
        self.hidden_size = raw["text_config"]["hidden_size"]
        self.image_token, self.video_token = raw["image_token_id"], raw["video_token_id"]
        self.media_tokens = frozenset((self.image_token, self.video_token))
        self.videos = True
        self.image_limits = None
        self.image_tokens = 4096
        c = self.config
        self.inverse = 1 / (c["rope_parameters"]["rope_theta"] ** (
            torch.arange(0, c["head_dim"] // 2, 2, dtype=torch.float32, device=self.device) / (c["head_dim"] // 2)))

    def prepare(self, *args, **kwargs):
        if self.frontend is None:
            raise ValueError("Gemma vision CPU processor is not configured")
        return self.frontend.prepare(*args, **kwargs)

    def continued(self, prepared: PreparedGemmaVision, tokens: Sequence[int]) -> PreparedGemmaVision:
        return continued(prepared, tokens, self.raw_config)

    def video_size(self, *args):
        if self.frontend is None:
            raise ValueError("Gemma vision CPU processor is not configured")
        return self.frontend.video_size(*args)

    def warm(self) -> None:
        """Prepare the native operators using one minimal spatial pooling block."""
        import torch

        c = self.config
        k = c["pooling_kernel_size"]
        positions = np.stack(np.meshgrid(np.arange(k), np.arange(k), indexing="xy"), axis=-1).reshape(-1, 2)
        frame = _frame(np.zeros((k**2, 3 * c["patch_size"]**2), dtype=np.float32), positions, c, "image")
        with torch.inference_mode():
            self._features(frame)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def _features(self, frame: VisionFrame):
        import torch
        from torch.nn import functional as F

        w, c = self.weights, self.config
        # Copy request arrays: Torch must never borrow immutable NumPy storage.
        pixels = torch.tensor(frame.pixels, dtype=torch.float32, device=self.device)[None]
        positions = torch.tensor(frame.positions, dtype=torch.int64, device=self.device)[None]
        padding = (positions == -1).all(-1)
        xy = positions.clamp_min(0)
        hidden = F.linear((2 * (pixels - 0.5)).to(self.dtype), w["patch_embedder.input_proj.weight"])
        table = w["patch_embedder.position_embedding_table"]
        positional = F.embedding(xy[..., 0], table[0]) + F.embedding(xy[..., 1], table[1])
        hidden = hidden + positional.masked_fill(padding[..., None], 0)
        mask = (~padding)[:, None, None, :]
        eps, heads, dim = c["rms_norm_eps"], c["num_attention_heads"], c["head_dim"]
        for layer in range(c["num_hidden_layers"]):
            b = f"encoder.layers.{layer}."
            normalized = _norm(hidden, w[b + "input_layernorm.weight"], eps)
            projected = []
            for name in ("q", "k", "v"):
                values = F.linear(normalized, w[b + f"self_attn.{name}_proj.linear.weight"])
                values = values.reshape(1, -1, heads, dim)
                values = _norm(values, w.get(b + f"self_attn.{name}_norm.weight"), eps)
                if name != "v":
                    values = _rope(values, positions, self.inverse)
                projected.append(values.transpose(1, 2))
            attention = F.scaled_dot_product_attention(*projected, attn_mask=mask, dropout_p=0.0, scale=1.0)
            attention = attention.transpose(1, 2).reshape_as(hidden)
            attention = F.linear(attention, w[b + "self_attn.o_proj.linear.weight"])
            hidden = hidden + _norm(attention, w[b + "post_attention_layernorm.weight"], eps)
            normalized = _norm(hidden, w[b + "pre_feedforward_layernorm.weight"], eps)
            gate = F.gelu(F.linear(normalized, w[b + "mlp.gate_proj.linear.weight"]), approximate="tanh")
            up = F.linear(normalized, w[b + "mlp.up_proj.linear.weight"])
            mlp = F.linear(gate * up, w[b + "mlp.down_proj.linear.weight"])
            hidden = hidden + _norm(mlp, w[b + "post_feedforward_layernorm.weight"], eps)
        return self.projector(_norm(self._pool(hidden, positions, padding), None, eps))

    def _pool(self, hidden, positions, padding):
        """Spatial pooling and FP32 cancellation before restoring working precision."""
        from torch.nn import functional as F

        c, w = self.config, self.weights
        hidden = hidden.masked_fill(padding[..., None], 0)
        xy = positions.clamp_min(0)
        k = c["pooling_kernel_size"]
        output_length = hidden.shape[1] // k**2
        width = xy[..., 0].max(-1, keepdim=True).values + 1
        pool_ids = xy[..., 0] // k + (width // k) * (xy[..., 1] // k)
        pool_weights = F.one_hot(pool_ids, output_length).float() / k**2
        pooled = (pool_weights.transpose(1, 2) @ hidden.float()).to(hidden.dtype)
        valid = pool_weights.ne(0).any(1)
        # Installed Transformers 5.17 keeps expansion and standardization in
        # FP32; rounding either operation to BF16 before cancellation changes
        # image features. Restore the working dtype only after standardization.
        pooled = (pooled.float() * math.sqrt(c["hidden_size"]))[valid]
        if c.get("standardize", False):
            pooled = (pooled - w["std_bias"].float()) * w["std_scale"].float()
        return pooled.to(self.dtype)

    def encode(self, prepared: PreparedGemmaVision, prompt: Sequence[int]) -> EncodedGemmaVision:
        import torch

        if tuple(prompt) != prepared.token_ids or len(prepared.frames) != len(prepared.blocks):
            raise ValueError("Gemma prepared vision belongs to different prompt tokens")
        blocks = media_blocks(prompt, self.image_token, self.video_token,
                              [(f.kind, f.soft_tokens) for f in prepared.frames])
        if blocks != prepared.blocks:
            raise ValueError("Gemma prepared vision has inconsistent visual blocks")
        with torch.inference_mode():
            # One image/frame a call bounds attention scratch independently of
            # image/video count. Features from separate requests never alias.
            parts = [self._features(frame) for frame in prepared.frames]
            if any(tuple(part.shape) != (frame.soft_tokens, self.hidden_size)
                   for part, frame in zip(parts, prepared.frames)):
                raise ValueError("Gemma tower output does not match its image/frame placeholders")
            features = torch.cat(parts, dim=0).contiguous()
        rows = tuple(i for start, end in blocks for i in range(start, end))
        return EncodedGemmaVision(rows, features, blocks)
