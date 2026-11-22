"""DeepSeek-V4.1-Flash image input on SM70: the server frontend, preprocessing, placeholder expansion and prompt-cache
ids, the mmproj ViT + aligner (tensor parallel, resumable by blocks), and the vision router bias.

Provenance (arithmetic and plan ported, recorded): the friend's overlay patch 150
(``patches/dsvis_files/local_ds41_vision_tower.py`` / ``local_ds41_vision.py``, snapshot lp-snap-c1es8-...), which
reimplements deepseek-ai/DeepSeek-V4.1-Flash-1017 ``inference/vision.py`` and ``inference/image_processor.py``
(rev 1bae0e10, MIT) on the llama.cpp mmproj (projector ``deepseek41v``, BF16).

Graph (upstream):
  patches (3 x 14 x 14 RGB, (x/255 - 0.5)/0.5) -> Linear 588 -> 1024 (bias)
  32 x [x + attn(rms(x)); x + mlp(rms(x))]: RMSNorm eps 1e-6, fused qkv (bias), 2-D rotate-half RoPE (first 16
        frequencies from the patch row, last 16 from the column, theta 1e4), full bidirectional attention over the
        image, wo (bias); SwiGLU 1024 -> 2 x 2816 -> 1024 (no bias)
  rms(x) -> 3 x 3 unfold (zero pad right/bottom, channel-major c*9 + ki*3 + kj) -> Linear 9216 -> 5120 (bias)
        -> GELU(erf) -> Linear 5120 -> 5120 (bias)
  span: [START] ([IMAGE] * n_llm_w + [NEWLINE]) * n_llm_h + [END]; every position carries id 129264
        (<｜deepseek_image｜>); START/NEWLINE/END are learned 5120-vectors of the mmproj.

Placement: TP8 over the lane's eight ranks (``VisionTower``: Megatron heads / MLP columns, FP32 all-reduce of the
row-parallel partials), so each GPU holds 1/8 of the 0.91 GiB tower and every rank ends with the same span (no
feature broadcast to the backbone).

Precision (the friend's production numerics, kept except at the row-parallel outputs; V100 has no BF16): BF16
weights widened to FP16 (checked at load: finite, exact in FP16's normal range, |error| <= 2**-25 on its subnormal
grid), FP16 GEMM operands with FP32 accumulation (cuBLAS reduced-precision reduction off while the tower runs),
FP16 column-parallel outputs (qkv, gate/up, mm.1); attn_out, ffn_down and mm.2 keep FP32 through the cross-rank sum
and the bias (his round them to FP16); the residual stream, RMSNorm, RoPE, GELU input and the patch embedding are
FP32 (``compute=float32`` runs everything in FP32: the GPU oracle). Residual magnitudes reach ~1.7e3 after layer 12, so an FP16 residual
loses ~3 bits there (the friend's ``fp16res`` variant, min token cosine 0.938 on carrots.jpeg).

Routing and Engram contract (upstream model.py, patch 150): image-span tokens route the MoE with the vision
bias ``bias_vl`` (``exp_probs_b_vl``, 40 x 384 FP32) instead of the text bias, routing weights still from the
unbiased scores; image-span tokens are dead Engram n-gram members (``valid & ~image`` into the hash state, its
commit and the Engram gate). Attention, compressor, indexer and positions have no image special case.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

IMAGE_TOKEN_ID = 129264
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
PATCH = 14
MERGE = 3
MAX_TOKENS = 1024
MIN_PIXELS = 544 * 544
DIM = 1024
HEADS = 16
INTER = 2816
LAYERS = 32
OUT_DIM = 5120
ROPE_THETA = 10000.0
EPS = 1e-6
PAD_COLOR = (127, 127, 127)
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)

BIAS_LAYERS, BIAS_EXPERTS = 40, 384
# sha256 of layers.0..39.ffn.gate.bias_vl (40 x 384 little-endian F32) of deepseek-ai/DeepSeek-V4.1-Flash-1017
# rev 1bae0e10 (ISTA convert_ds41_upstream.py output exp_probs_b_vl.f32; patch 150's BIAS_VL_SHA256)
BIAS_VL_SHA256 = "710b11267b4db392d1fdb558f36ba8e0e91a91f4e8180c953c912e18eacf843d"

_VISION_CONFIG = dict(model_type="deepseek_v41_vision", num_hidden_layers=LAYERS, hidden_size=DIM,
                      num_attention_heads=HEADS, intermediate_size=INTER, patch_size=PATCH,
                      downsample_ratio=MERGE, max_image_tokens=MAX_TOKENS, min_pixels=MIN_PIXELS, max_wh_ratio=None)
_MMPROJ_FIELDS = {"clip.projector_type": "deepseek41v", "clip.vision.patch_size": PATCH,
                  "clip.vision.embedding_length": DIM, "clip.vision.feed_forward_length": INTER,
                  "clip.vision.block_count": LAYERS, "clip.vision.attention.head_count": HEADS,
                  "clip.vision.projection_dim": OUT_DIM, "clip.vision.projector.scale_factor": MERGE,
                  "clip.vision.image_min_pixels": MIN_PIXELS}


def check_config(raw: dict) -> None:
    """Refuse a checkpoint whose vision contract differs from the one ported here (config.json)."""
    vision = raw.get("vision_config")
    if not isinstance(vision, dict):
        raise ValueError("DeepSeek-V4.1 vision requires config.json vision_config")
    wrong = {k: vision.get(k) for k, v in _VISION_CONFIG.items() if vision.get(k, None) != v}
    if vision.get("rope_theta", ROPE_THETA) != ROPE_THETA:
        wrong["rope_theta"] = vision.get("rope_theta")
    if wrong or raw.get("image_token_id") != IMAGE_TOKEN_ID:
        raise ValueError(f"unsupported DeepSeek-V4.1 vision_config {wrong} / image_token_id {raw.get('image_token_id')}")


# ------------------------------------------------------------------------------------- resize plan (upstream)
def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int) -> tuple[int, int]:
    return math.ceil((best_height // PATCH) / MERGE), math.ceil((best_width // PATCH) / MERGE)


def _solve_resize_ratio(height, width):
    r = height / width
    max_w_float = math.sqrt((MAX_TOKENS - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = PATCH * MERGE
    if max_w_float < 1.0:
        return (MAX_TOKENS - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (MAX_TOKENS - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / PATCH) * PATCH, math.floor(width * beta / PATCH) * PATCH


def plan_image_grid(width: int, height: int) -> tuple[int, int, int, int]:
    """(n_llm_h, n_llm_w, best_height, best_width): image_processor.plan_image_grid (max_wh_ratio None)."""
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("image size must be positive integers")
    if width * height < MIN_PIXELS:
        scale = (MIN_PIXELS / (width * height)) ** 0.5
        width, height = int(width * scale), int(height * scale)
    best_width = math.ceil(width / PATCH) * PATCH
    best_height = math.ceil(height / PATCH) * PATCH
    n_llm_h, n_llm_w = llm_grid(best_height, best_width)
    if num_image_tokens(n_llm_h, n_llm_w) > MAX_TOKENS:
        best_height, best_width = _solve_resize_ratio(height, width)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width)
        if num_image_tokens(n_llm_h, n_llm_w) > MAX_TOKENS:
            raise AssertionError("DeepSeek-V4.1 resize plan exceeds max_image_tokens")
    return n_llm_h, n_llm_w, best_height, best_width


def image_token_types(n_llm_h: int, n_llm_w: int) -> np.ndarray:
    types = [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
    return np.asarray(types, dtype=np.int64)


# ------------------------------------------------------------------------------------------- preprocessing
@dataclass(frozen=True, slots=True)
class ImageFrame:
    """One image's ViT input: uint8 patches [n_vit_h * n_vit_w, 3, 14, 14] in reading order (CPU, owned)."""

    patches: np.ndarray
    n_vit_h: int
    n_vit_w: int
    content_hash: str            # tensorfold.vision.images.ImageInput.content_hash of the source pixels

    @property
    def n_llm_h(self) -> int:
        return -(-self.n_vit_h // MERGE)

    @property
    def n_llm_w(self) -> int:
        return -(-self.n_vit_w // MERGE)

    @property
    def tokens(self) -> int:
        return num_image_tokens(self.n_llm_h, self.n_llm_w)


def preprocess(image) -> ImageFrame:
    """``tensorfold.vision.images.ImageInput`` (decoded, oriented RGB) -> ImageFrame.

    Pixels exactly as image_processor.load_image: resize plan, ``ImageOps.pad`` (aspect-preserving resize, then
    centered grey-127 padding), patchify. The (x/255 - 0.5)/0.5 normalization runs on the device in FP32."""
    from PIL import ImageOps

    pil = image.to_pil()
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(pil.width, pil.height)
    padded = ImageOps.pad(pil, (best_width, best_height), color=PAD_COLOR)
    n_vit_h, n_vit_w = best_height // PATCH, best_width // PATCH
    x = np.asarray(padded, dtype=np.uint8).transpose(2, 0, 1)
    patches = np.ascontiguousarray(
        x.reshape(3, n_vit_h, PATCH, n_vit_w, PATCH).transpose(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, PATCH, PATCH))
    patches.flags.writeable = False
    frame = ImageFrame(patches, n_vit_h, n_vit_w, image.content_hash)
    if (frame.n_llm_h, frame.n_llm_w) != (n_llm_h, n_llm_w):
        raise AssertionError("DeepSeek-V4.1 token grid differs from its patch grid")
    return frame


# ------------------------------------------------------------------------------------ prompt placeholders
@dataclass(frozen=True, slots=True)
class PreparedVision:
    """A prompt whose image placeholders are expanded: ``token_ids`` with one span per frame at ``blocks``
    ((start, end) prompt positions, prompt order); every span position carries IMAGE_TOKEN_ID."""

    token_ids: tuple[int, ...]
    frames: tuple[ImageFrame, ...]
    blocks: tuple[tuple[int, int], ...]
    image_hashes: tuple[str, ...] = field(default=())

    def continued(self, tokens: Sequence[int]) -> "PreparedVision":
        """The same media for a prompt that only appends text tokens (a continuation turn)."""
        tokens = tuple(int(t) for t in tokens)
        if tokens[:len(self.token_ids)] != self.token_ids or IMAGE_TOKEN_ID in tokens[len(self.token_ids):]:
            raise ValueError("a continued DeepSeek-V4.1 image prompt must extend the prepared prompt with text only")
        return PreparedVision(tokens, self.frames, self.blocks, self.image_hashes)


def expand_placeholders(prompt_ids: Sequence[int], frames: Sequence[ImageFrame]) -> PreparedVision:
    """image_processor.prepare_vl_inputs: each single IMAGE_TOKEN_ID of the rendered prompt (one per image block,
    prompt order) becomes that frame's span. The rendered prompt must hold exactly one placeholder per frame."""
    prompt = [int(t) for t in prompt_ids]
    count = sum(t == IMAGE_TOKEN_ID for t in prompt)
    if count != len(frames):
        raise ValueError(f"the prompt has {count} image placeholders for {len(frames)} images")
    tokens, blocks, it = [], [], iter(frames)
    for t in prompt:
        if t != IMAGE_TOKEN_ID:
            tokens.append(t)
            continue
        frame = next(it)
        blocks.append((len(tokens), len(tokens) + frame.tokens))
        tokens.extend([IMAGE_TOKEN_ID] * frame.tokens)
    return PreparedVision(tuple(tokens), tuple(frames), tuple(blocks), tuple(f.content_hash for f in frames))


class Frontend:
    """The CUDA server's image frontend of this lane (``engine.vision`` on rank 0; ``server.prompts.prepare_images``
    calls ``prepare``). The server's V4.1 renderer writes one trusted ``<｜deepseek_image｜>`` per image part
    (``prompts_v41``, images on); ``prepare`` encodes that prompt with the renderer's literal encoder (request text
    never becomes the image token), preprocesses each image (``preprocess``) and expands the placeholders."""

    videos = False
    audio = False

    def __init__(self, model_dir, *, allow_urls: bool = False) -> None:
        from tensorfold.token_metadata import load_tokenizer

        from ..serving_v41 import V41ChatTemplate

        self.allow_urls = bool(allow_urls)
        self.template = V41ChatTemplate(model_dir)
        self.tokenizer = load_tokenizer(model_dir)
        if self.tokenizer.token_to_id(IMAGE_PLACEHOLDER) != IMAGE_TOKEN_ID:
            raise ValueError(f"the tokenizer's {IMAGE_PLACEHOLDER} is not id {IMAGE_TOKEN_ID}")

    def prepare(self, rendered_prompt, images: Sequence[Any], *, max_prompt_tokens: int | None = None,
                max_visual_tokens: int | None = None, **_unused) -> PreparedVision:
        tokens = self.template.encode(self.tokenizer, rendered_prompt)
        if not images or sum(t == IMAGE_TOKEN_ID for t in tokens) != len(images):
            raise ValueError("the DeepSeek-V4.1 prompt must hold exactly one image placeholder per image")
        frames = [preprocess(image) for image in images]
        visual = sum(f.tokens for f in frames)
        if max_visual_tokens is not None and visual > max_visual_tokens:
            raise ValueError(f"the request's images take {visual} visual tokens; at most {max_visual_tokens} are "
                             "admitted (--vision-image-tokens)")
        prepared = expand_placeholders(tokens, frames)
        if max_prompt_tokens is not None and len(prepared.token_ids) > max_prompt_tokens:
            raise ValueError(f"context length exceeds {max_prompt_tokens} tokens after image expansion")
        return prepared

    @staticmethod
    def continued(prepared: PreparedVision, ids) -> PreparedVision:
        return prepared.continued(ids)


def prefix_ids(prepared: PreparedVision, vocab: int) -> list[int]:
    """The prompt ids the prompt-prefix cache keys on (and the engine's forwards carry): every image-span position
    holds an id >= ``vocab`` derived from its frame's content (sha256 of the source pixels and the patch grid), word
    ``j % 8`` of the digest at span offset ``j``, so two prompts share a prefix only where their images are the same
    pixels. ``model_ids`` maps them back to IMAGE_TOKEN_ID; ids < ``vocab`` are the prompt's own."""
    room = (1 << 31) - 1 - vocab
    ids = list(prepared.token_ids)
    for frame, (s, e) in zip(prepared.frames, prepared.blocks):
        digest = hashlib.sha256(f"ds41-image-v1\0{frame.content_hash}\0{frame.n_vit_h}x{frame.n_vit_w}".encode()).digest()
        words = [vocab + int.from_bytes(digest[4 * k:4 * k + 4], "little") % room for k in range(8)]
        for j in range(e - s):
            ids[s + j] = words[j % 8]
    return ids


# ------------------------------------------------------------------------------------------- mmproj reading
def shard_axes(name: str) -> tuple[int, int] | None:
    """The TP split of mmproj tensor ``name``: (axis, pieces), every one of the ``pieces`` equal blocks along
    ``axis`` cut into ``world`` contiguous parts (rank r takes part r of each), or None (replicated). Projections
    split over output features. Fused qkv splits q | k | v separately to keep whole heads together."""
    if name.endswith(("attn_qkv.weight", "attn_qkv.bias")):
        return 0, 3
    if name.endswith(("attn_out.weight", "attn_out.bias", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")) or \
            name in ("mm.1.weight", "mm.1.bias", "mm.2.weight", "mm.2.bias"):
        return 0, 1
    return None


def take_shard(value, name: str, rank: int, world: int):
    """Rank ``rank``'s part of ``value`` (numpy array or torch tensor, full shape) under ``shard_axes(name)``."""
    spec = shard_axes(name) if world > 1 else None
    if spec is None:
        return value
    axis, pieces = spec
    size = value.shape[axis]
    if size % (pieces * world):
        raise ValueError(f"{name}: dimension {size} does not split into {pieces} x {world} parts")
    part = size // (pieces * world)
    shape = tuple(value.shape)
    blocks = value.reshape(shape[:axis] + (pieces, size // pieces) + shape[axis + 1:])
    index = (slice(None),) * (axis + 1) + (slice(rank * part, (rank + 1) * part),)
    return blocks[index].reshape(shape[:axis] + (pieces * part,) + shape[axis + 1:])


def read_mmproj(path: str | Path, rank: int = 0, world: int = 1) -> dict[str, Any]:
    """The deepseek41v mmproj as {name: CPU torch tensor}: F32 tensors FP32, BF16 matrices FP16 (exactness
    checked), shapes in torch order; with ``world`` > 1 only rank ``rank``'s TP part of each split tensor
    (``shard_axes``; cut before widening, so a rank reads and converts its part only). Reads through TensorFold's
    GGUF inventory (``gguf_file._shard``); the full shapes are checked before any cut."""
    import torch

    from tensorfold.gguf_file import _shard

    path = Path(path).resolve(strict=True)
    metadata, entries = _shard(path)
    wrong = {k: metadata.get(k) for k, v in _MMPROJ_FIELDS.items() if metadata.get(k) != v}
    eps = metadata.get("clip.vision.attention.layer_norm_epsilon")
    if wrong or not isinstance(eps, float) or abs(eps - EPS) > 1e-12:
        raise ValueError(f"{path}: not the DeepSeek-V4.1 mmproj ({wrong}, eps {eps})")
    need = ["v.patch_embd.weight", "v.patch_embd.bias", "v.post_ln.weight", "mm.1.weight", "mm.1.bias", "mm.2.weight",
            "mm.2.bias", "v.token_embd.img_start", "v.token_embd.img_end", "v.image_newline"]
    for i in range(LAYERS):
        need += [f"v.blk.{i}.{n}" for n in ("attn_qkv.weight", "attn_qkv.bias", "attn_out.weight", "attn_out.bias",
                                           "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ln1.weight",
                                           "ln2.weight")]
    names = {t.name for t in entries}
    missing = [n for n in need if n not in names]
    if missing or len(entries) != len(need):
        raise ValueError(f"{path}: tensor set differs from deepseek41v (missing {missing[:4]})")
    expected = {"v.patch_embd.weight": (DIM, 3, PATCH, PATCH), "mm.1.weight": (OUT_DIM, DIM * MERGE * MERGE),
                "mm.2.weight": (OUT_DIM, OUT_DIM), "v.image_newline": (OUT_DIM,)}
    for i in range(LAYERS):
        expected.update({f"v.blk.{i}.attn_qkv.weight": (3 * DIM, DIM), f"v.blk.{i}.attn_qkv.bias": (3 * DIM,),
                         f"v.blk.{i}.attn_out.weight": (DIM, DIM), f"v.blk.{i}.attn_out.bias": (DIM,),
                         f"v.blk.{i}.ffn_gate.weight": (INTER, DIM),
                         f"v.blk.{i}.ffn_up.weight": (INTER, DIM), f"v.blk.{i}.ffn_down.weight": (DIM, INTER)})
    expected["mm.1.bias"] = expected["mm.2.bias"] = (OUT_DIM,)
    for t in entries:
        if t.name in expected and tuple(t.shape) != expected[t.name]:
            raise ValueError(f"{path}: {t.name} has shape {tuple(t.shape)}, expected {expected[t.name]}")
        if shard_axes(t.name) is not None and t.name not in expected:
            raise AssertionError(f"{t.name}: a split tensor without a checked shape")
    data = np.memmap(path, dtype=np.uint8, mode="r")
    tensors = {}
    try:
        for t in entries:
            raw = data[t.offset:t.offset + t.nbytes]
            if t.ggml_type == 0:
                part = take_shard(np.frombuffer(raw, dtype="<f4").reshape(t.shape), t.name, rank, world)
                value = torch.from_numpy(np.ascontiguousarray(part, dtype=np.float32).copy())
            elif t.ggml_type == 30:
                part = take_shard(np.frombuffer(raw, dtype="<u2").reshape(t.shape), t.name, rank, world)
                wide = torch.from_numpy(np.ascontiguousarray(part).astype(np.int32) << 16).view(torch.float32)
                half = wide.half()
                # FP16 holds every BF16 weight in its normal range exactly; values below 2**-14 round to the
                # subnormal grid (|error| <= 2**-25). This file: 88,173 of 484,442,112 values inexact, max error
                # 2.98e-8, max |w| 2.33 (measured 2026-11-21); anything else is refused.
                if not bool(torch.isfinite(half).all()) or float((half.float() - wide).abs().max()) > 2.0**-25:
                    raise ValueError(f"{path}: {t.name} does not fit FP16")
                value = half
            else:
                raise ValueError(f"{path}: {t.name} has ggml type {t.ggml_type}; the deepseek41v mmproj is F32/BF16")
            tensors[t.name] = value
    finally:
        del data
    return tensors


def read_bias_vl(path: str | Path) -> tuple[np.ndarray, str]:
    """exp_probs_b_vl.f32 (40 x 384 little-endian F32) -> (table, sha256). The backbone GGUF of this lane carries
    no ``blk.N.exp_probs_b_vl.bias`` (checked 2026-11-21 on ds41-uncensored-gsq-rco), so the file is required.
    The caller logs whether the digest equals BIAS_VL_SHA256 (the upstream checkpoint's)."""
    raw = Path(path).read_bytes()
    if len(raw) != BIAS_LAYERS * BIAS_EXPERTS * 4:
        raise ValueError(f"{path}: {len(raw)} bytes, expected 40 x 384 F32")
    table = np.frombuffer(raw, dtype="<f4").reshape(BIAS_LAYERS, BIAS_EXPERTS).astype(np.float32)
    if not np.isfinite(table).all():
        raise ValueError(f"{path}: non-finite vision router bias")
    return table, hashlib.sha256(raw).hexdigest()


def route_with_vision(logits, bias, bias_vl, image, top_k: int, scale: float):
    """Per-row routing: rows where ``image`` is True use ``bias_vl``, others ``bias`` (patch 150 combine_routes:
    the vendored fused router run once per bias, rows selected). ``image`` None: the text router alone."""
    import torch

    from .kernels.router import route_topk

    text_p, text_i = route_topk(logits, bias, top_k, scale)
    if image is None:
        return text_p, text_i
    vl_p, vl_i = route_topk(logits, bias_vl, top_k, scale)
    rows = image.reshape(-1, 1)
    return torch.where(rows, vl_p, text_p).contiguous(), torch.where(rows, vl_i, text_i).contiguous()


# ----------------------------------------------------------------------------------------------- the tower
def rope_cos_sin(n_h: int, n_w: int, device):
    """vision.get_vision_cos_sin: [n, 1, 32] FP32."""
    import torch

    dim = DIM // HEADS // 2
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = (torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq).flatten(1).to(device)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


class VisionTower:
    """Device-resident ViT + aligner of TP rank ``rank`` of ``world`` (``world`` 1: the whole tower on one device);
    ``encode(frame)`` -> the frame's whole span [frame.tokens, 5120] (delimiters included) in ``out_dtype``.

    Each TP rank owns whole attention heads and its output rows of every projection. Before attn_out, ffn_down
    and mm.2, lossless gathers assemble the input features in rank order. Each local GEMM reduces over the full
    input dimension before the original GEMM and bias roundings. Another gather reconstructs the full outputs.
    The residual stream, norms, patch embedding, RoPE and delimiters stay replicated. There is no floating-point
    reduction between ranks. GEMM tiling may still differ with output width.

    ``compute`` torch.float16 (default) uses FP16 GEMM operands and FP32 accumulation. The individual GEMM and
    bias operations round to FP16 and the residual adds in FP32. ``torch.float32`` is the GPU oracle."""

    def __init__(self, tensors: dict, device, *, compute=None, rank: int = 0, world: int = 1, comm=None):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.compute = torch.float16 if compute is None else compute
        if self.compute not in (torch.float16, torch.float32):
            raise ValueError("the DeepSeek-V4.1 tower computes in FP16 (production) or FP32 (oracle)")
        if type(world) is not int or world < 1 or not 0 <= rank < world or HEADS % world or INTER % world \
                or OUT_DIM % world:
            raise ValueError(f"the DeepSeek-V4.1 tower does not split over rank {rank} of {world}")
        if world > 1 and comm is None:
            raise ValueError("a tensor-parallel DeepSeek-V4.1 tower needs the ranks' collectives")
        self.rank, self.world, self.comm = rank, world, comm
        self.heads = HEADS // world
        width = self.heads * (DIM // HEADS)
        shapes = {"attn_qkv.weight": (3 * width, DIM), "attn_qkv.bias": (3 * width,),
                  "attn_out.weight": (DIM // world, DIM), "attn_out.bias": (DIM // world,),
                  "ffn_gate.weight": (INTER // world, DIM), "ffn_up.weight": (INTER // world, DIM),
                  "ffn_down.weight": (DIM // world, INTER)}
        aligner = {"mm.1.weight": (OUT_DIM // world, DIM * MERGE * MERGE), "mm.1.bias": (OUT_DIM // world,),
                   "mm.2.weight": (OUT_DIM // world, OUT_DIM), "mm.2.bias": (OUT_DIM // world,)}
        wrong = [n for n, shape in aligner.items() if tuple(tensors[n].shape) != shape]
        wrong += [f"v.blk.{i}.{n}" for i in range(LAYERS) for n, shape in shapes.items()
                  if tuple(tensors[f"v.blk.{i}.{n}"].shape) != shape]
        if wrong:
            raise ValueError(f"mmproj parts {wrong[:3]} are not rank {rank} of {world}'s (read_mmproj)")

        def mat(name):
            return tensors[name].to(self.compute).to(self.device).contiguous()

        def vec(name):
            return tensors[name].float().to(self.device).contiguous()

        self.patch_w = tensors["v.patch_embd.weight"].float().reshape(DIM, -1).to(self.device).contiguous()
        self.patch_b = vec("v.patch_embd.bias")
        self.blocks = []
        for i in range(LAYERS):
            p = f"v.blk.{i}."
            self.blocks.append(dict(
                ln1=vec(p + "ln1.weight"), ln2=vec(p + "ln2.weight"),
                qkv=mat(p + "attn_qkv.weight"), qkv_b=vec(p + "attn_qkv.bias"),
                wo=mat(p + "attn_out.weight"), wo_b=vec(p + "attn_out.bias"),
                gate_up=torch.cat([mat(p + "ffn_gate.weight"), mat(p + "ffn_up.weight")], 0).contiguous(),
                down=mat(p + "ffn_down.weight")))
        self.post_ln = vec("v.post_ln.weight")
        self.mm1, self.mm1_b = mat("mm.1.weight"), vec("mm.1.bias")
        self.mm2, self.mm2_b = mat("mm.2.weight"), vec("mm.2.bias")
        self.start, self.end = vec("v.token_embd.img_start"), vec("v.token_embd.img_end")
        self.newline = vec("v.image_newline")

    def nbytes(self) -> int:
        own = [self.patch_w, self.patch_b, self.post_ln, self.mm1, self.mm1_b, self.mm2, self.mm2_b, self.start,
               self.end, self.newline]
        own += [t for b in self.blocks for t in b.values()]
        return sum(t.numel() * t.element_size() for t in own)

    # FP16: operand and output rounding points of cuBLAS FP16 GEMMs with FP32 accumulation
    def _linear(self, x, weight, bias=None):
        F = self.torch.nn.functional
        y = F.linear(x.to(self.compute), weight)
        if bias is not None:
            y = (y.float() + bias).to(self.compute)
        return y

    def _gather_linear(self, x, weight, bias=None):
        """Compute the rank's output rows with the full rank-ordered input, then gather the result."""
        if self.world == 1:
            return self._linear(x, weight, bias)
        return self.comm.gather(self._linear(self.comm.gather(x.to(self.compute)), weight, bias))

    def _rms(self, x, weight):
        x = x.float()
        return weight * (x * self.torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS))

    def _rotary(self, x, cos, sin):
        x1, x2 = x.float().chunk(2, dim=-1)
        return self.torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def _attention(self, q, k, v):
        """[heads, n, 64] each; memory-efficient SDPA only (FP32 softmax/accumulation; no n x n materialization)."""
        torch = self.torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q, k, v = (t.to(self.compute).unsqueeze(0) for t in (q, k, v))
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)[0]

    def begin(self, patches, n_h: int, n_w: int) -> dict:
        """A resumable encode (``run`` advances it block by block, ``span`` finishes it): the patch embedding of
        ``patches`` [n_h * n_w, 3, 14, 14] (uint8 pixels, or float already normalized). Every step queues GPU work
        only (no host wait), so the caller may run other forwards between blocks; with ``world`` > 1 every rank
        runs the same steps in the same order (their collectives pair up)."""
        torch = self.torch
        F = torch.nn.functional
        if patches.shape != (n_h * n_w, 3, PATCH, PATCH):
            raise ValueError(f"{tuple(patches.shape)} patches for a {n_h} x {n_w} grid")
        x = patches.to(self.device, non_blocking=True)
        x = ((x.float() / 255 - 0.5) / 0.5) if x.dtype == torch.uint8 else x.float()
        x = F.linear(x.flatten(1), self.patch_w, self.patch_b)             # FP32 residual stream
        cos, sin = rope_cos_sin(n_h, n_w, self.device)
        return {"x": x, "cos": cos, "sin": sin, "n_h": n_h, "n_w": n_w, "next": 0}

    def run(self, st: dict, stop: int) -> None:
        """Blocks [st["next"], stop) of the ViT."""
        F = self.torch.nn.functional
        x, cos, sin = st["x"], st["cos"], st["sin"]
        n, hd = x.shape[0], DIM // HEADS
        for b in self.blocks[st["next"]:stop]:
            h = self._linear(self._rms(x, b["ln1"]), b["qkv"], b["qkv_b"])
            q, k, v = (t.reshape(n, self.heads, hd) for t in h.chunk(3, dim=-1))
            q, k = self._rotary(q, cos, sin), self._rotary(k, cos, sin)
            o = self._attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
            x = x + self._gather_linear(o.transpose(0, 1).reshape(n, self.heads * hd), b["wo"], b["wo_b"])
            gate, up = self._linear(self._rms(x, b["ln2"]), b["gate_up"]).float().chunk(2, dim=-1)
            x = x + self._gather_linear(F.silu(gate) * up, b["down"])
        st["x"], st["next"] = x, max(st["next"], stop)

    def aligner(self, st: dict):
        """Post-norm, 3 x 3 merge and aligner: [ceil(n_h/3) * ceil(n_w/3), 5120] in reading order."""
        F = self.torch.nn.functional
        if st["next"] != LAYERS:
            raise RuntimeError("the DeepSeek-V4.1 aligner runs after the last ViT block")
        n_h, n_w = st["n_h"], st["n_w"]
        x = self._rms(st["x"], self.post_ln).view(n_h, n_w, DIM).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % MERGE, 0, -n_h % MERGE))
        x = F.unfold(x.unsqueeze(0), MERGE, stride=MERGE).squeeze(0).transpose(0, 1)
        y = self._linear(x, self.mm1, self.mm1_b)
        y = F.gelu(y.float()).to(self.compute)
        return self._gather_linear(y, self.mm2, self.mm2_b)

    def grid(self, patches, n_h: int, n_w: int):
        """ViT + aligner: [ceil(n_h/3) * ceil(n_w/3), 5120] FP32 in reading order."""
        st = self.begin(patches, n_h, n_w)
        self.run(st, LAYERS)
        return self.aligner(st)

    def span(self, st: dict, out_dtype=None):
        """The span [tokens, 5120] of the finished encode ``st``: START, rows of (grid row, NEWLINE), END."""
        torch = self.torch
        out_dtype = torch.float16 if out_dtype is None else out_dtype
        n_llm_h, n_llm_w = -(-st["n_h"] // MERGE), -(-st["n_w"] // MERGE)
        rows = self.aligner(st).view(n_llm_h, n_llm_w, OUT_DIM)
        newline = self.newline.view(1, 1, OUT_DIM).expand(n_llm_h, 1, OUT_DIM)
        body = torch.cat([rows, newline], dim=1).reshape(-1, OUT_DIM)
        span = torch.cat([self.start.view(1, -1), body, self.end.view(1, -1)], dim=0)
        if span.shape[0] != num_image_tokens(n_llm_h, n_llm_w):
            raise AssertionError("DeepSeek-V4.1 span length differs from its plan")
        return span.to(out_dtype)

    def start_frame(self, frame: ImageFrame) -> dict:
        """``begin`` for a frame's patches (pinned host copy, so the upload does not wait on the host)."""
        return self.begin(pinned_patches(frame, self.device.type == "cuda"), frame.n_vit_h, frame.n_vit_w)

    def encode(self, frame: ImageFrame, out_dtype=None):
        """The frame's whole span [frame.tokens, 5120] (delimiters included). ``out_dtype`` default FP16 (the
        friend's merge into his FP16 inputs_embeds; the backbone lifts the stream to FP32 after). With ``world``
        > 1 every rank encodes the same frame at once."""
        torch = self.torch
        reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        try:
            with torch.inference_mode():
                st = self.start_frame(frame)
                self.run(st, LAYERS)
                span = self.span(st, out_dtype)
        finally:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
        if span.shape[0] != frame.tokens:
            raise AssertionError("DeepSeek-V4.1 span length differs from its frame")
        return span


def pinned_patches(frame: ImageFrame, pin: bool = True):
    """The frame's uint8 patches as an owned (pinned with ``pin``) CPU tensor [n, 3, 14, 14]."""
    import torch

    patches = torch.from_numpy(np.array(frame.patches))
    return patches.pin_memory() if pin else patches
