"""DeepSeek-V4.1-Flash image input on SM70: preprocessing, placeholder expansion, the mmproj ViT + aligner, and the
vision router bias.

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

Precision (the friend's production numerics, kept for parity; V100 has no BF16): BF16 weights widened to FP16
(checked at load: finite, exact in FP16's normal range, |error| <= 2**-25 on its subnormal grid), FP16 GEMM
operands with FP32 accumulation (cuBLAS reduced-precision reduction off while the tower runs), FP16 linear
outputs; the residual stream, RMSNorm, RoPE, GELU input and the patch embedding are FP32 (``compute=float32``
runs everything in FP32: the GPU oracle). Residual magnitudes reach ~1.7e3 after layer 12, so an FP16 residual
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


def image_rows(ids):
    """Device bool mask of image-span rows of a step (upstream rule: input id == IMAGE_TOKEN_ID). Text can never
    carry the id: the V4.1 renderer refuses the literal placeholder in text."""
    return ids == IMAGE_TOKEN_ID


# ------------------------------------------------------------------------------------------- mmproj reading
def read_mmproj(path: str | Path) -> dict[str, Any]:
    """The deepseek41v mmproj as {name: CPU torch tensor}: F32 tensors FP32, BF16 matrices FP16 (exactness
    checked), shapes in torch order. Reads through TensorFold's GGUF inventory (``gguf_file._shard``)."""
    import torch

    from tensorfold.gguf_file import _shard

    path = Path(path).resolve(strict=True)
    metadata, entries = _shard(path)
    wrong = {k: metadata.get(k) for k, v in _MMPROJ_FIELDS.items() if metadata.get(k) != v}
    eps = metadata.get("clip.vision.attention.layer_norm_epsilon")
    if wrong or not isinstance(eps, float) or abs(eps - EPS) > 1e-12:
        raise ValueError(f"{path}: not the DeepSeek-V4.1 mmproj ({wrong}, eps {eps})")
    data = np.memmap(path, dtype=np.uint8, mode="r")
    tensors = {}
    try:
        for t in entries:
            raw = data[t.offset:t.offset + t.nbytes]
            if t.ggml_type == 0:
                value = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy()).reshape(t.shape)
            elif t.ggml_type == 30:
                wide = torch.from_numpy((np.frombuffer(raw, dtype="<u2").astype(np.int32) << 16)).view(torch.float32)
                half = wide.half()
                # FP16 holds every BF16 weight in its normal range exactly; values below 2**-14 round to the
                # subnormal grid (|error| <= 2**-25). This file: 88,173 of 484,442,112 values inexact, max error
                # 2.98e-8, max |w| 2.33 (measured 2026-11-21); anything else is refused.
                if not bool(torch.isfinite(half).all()) or float((half.float() - wide).abs().max()) > 2.0**-25:
                    raise ValueError(f"{path}: {t.name} does not fit FP16")
                value = half.reshape(t.shape)
            else:
                raise ValueError(f"{path}: {t.name} has ggml type {t.ggml_type}; the deepseek41v mmproj is F32/BF16")
            tensors[t.name] = value
    finally:
        del data
    need = ["v.patch_embd.weight", "v.patch_embd.bias", "v.post_ln.weight", "mm.1.weight", "mm.1.bias", "mm.2.weight",
            "mm.2.bias", "v.token_embd.img_start", "v.token_embd.img_end", "v.image_newline"]
    for i in range(LAYERS):
        need += [f"v.blk.{i}.{n}" for n in ("attn_qkv.weight", "attn_qkv.bias", "attn_out.weight", "attn_out.bias",
                                           "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ln1.weight",
                                           "ln2.weight")]
    missing = [n for n in need if n not in tensors]
    if missing or len(tensors) != len(need):
        raise ValueError(f"{path}: tensor set differs from deepseek41v (missing {missing[:4]})")
    expected = {"v.patch_embd.weight": (DIM, 3, PATCH, PATCH), "mm.1.weight": (OUT_DIM, DIM * MERGE * MERGE),
                "mm.2.weight": (OUT_DIM, OUT_DIM), "v.blk.0.attn_qkv.weight": (3 * DIM, DIM),
                "v.blk.0.ffn_down.weight": (DIM, INTER), "v.image_newline": (OUT_DIM,)}
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{path}: {name} has shape {tuple(tensors[name].shape)}, expected {shape}")
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
    """Device-resident ViT + aligner of one rank; ``encode(frame)`` -> the frame's whole span
    [frame.tokens, 5120] (delimiters included) in ``out_dtype``.

    ``compute`` torch.float16 (default): the friend's production numerics (module docstring). torch.float32: every
    operand FP32 (the GPU oracle; ~8x slower GEMMs on V100)."""

    def __init__(self, tensors: dict, device, *, compute=None):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.compute = torch.float16 if compute is None else compute
        if self.compute not in (torch.float16, torch.float32):
            raise ValueError("the DeepSeek-V4.1 tower computes in FP16 (production) or FP32 (oracle)")

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

    def grid(self, patches, n_h: int, n_w: int):
        """ViT + aligner: [ceil(n_h/3) * ceil(n_w/3), 5120] in reading order (compute dtype)."""
        torch = self.torch
        F = torch.nn.functional
        if patches.shape != (n_h * n_w, 3, PATCH, PATCH):
            raise ValueError(f"{tuple(patches.shape)} patches for a {n_h} x {n_w} grid")
        x = patches.to(self.device)
        x = ((x.float() / 255 - 0.5) / 0.5) if x.dtype == torch.uint8 else x.float()
        x = F.linear(x.flatten(1), self.patch_w, self.patch_b)             # FP32 residual stream
        cos, sin = rope_cos_sin(n_h, n_w, self.device)
        n, hd = x.shape[0], DIM // HEADS
        for b in self.blocks:
            h = self._linear(self._rms(x, b["ln1"]), b["qkv"], b["qkv_b"])
            q, k, v = (t.reshape(n, HEADS, hd) for t in h.chunk(3, dim=-1))
            q, k = self._rotary(q, cos, sin), self._rotary(k, cos, sin)
            o = self._attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
            o = self._linear(o.transpose(0, 1).reshape(n, DIM), b["wo"], b["wo_b"])
            x = x + o.float()
            gate, up = self._linear(self._rms(x, b["ln2"]), b["gate_up"]).float().chunk(2, dim=-1)
            x = x + self._linear((F.silu(gate) * up).to(self.compute), b["down"]).float()
        x = self._rms(x, self.post_ln).view(n_h, n_w, DIM).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % MERGE, 0, -n_h % MERGE))
        x = F.unfold(x.unsqueeze(0), MERGE, stride=MERGE).squeeze(0).transpose(0, 1)
        y = self._linear(x, self.mm1, self.mm1_b)
        y = F.gelu(y.float()).to(self.compute)
        return self._linear(y, self.mm2, self.mm2_b)

    def encode(self, frame: ImageFrame, out_dtype=None):
        """The frame's span [frame.tokens, 5120]: START, rows of (grid row, NEWLINE), END. ``out_dtype`` default
        FP16 (the friend's merge into his FP16 inputs_embeds; the backbone lifts the stream to FP32 after)."""
        torch = self.torch
        out_dtype = torch.float16 if out_dtype is None else out_dtype
        patches = torch.from_numpy(np.array(frame.patches))
        reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        try:
            with torch.inference_mode():
                grid = self.grid(patches, frame.n_vit_h, frame.n_vit_w)
                rows = grid.float().view(frame.n_llm_h, frame.n_llm_w, OUT_DIM)
                newline = self.newline.view(1, 1, OUT_DIM).expand(frame.n_llm_h, 1, OUT_DIM)
                body = torch.cat([rows, newline], dim=1).reshape(-1, OUT_DIM)
                span = torch.cat([self.start.view(1, -1), body, self.end.view(1, -1)], dim=0)
        finally:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
        if span.shape[0] != frame.tokens:
            raise AssertionError("DeepSeek-V4.1 span length differs from its plan")
        return span.to(out_dtype)


@dataclass(frozen=True, slots=True)
class EncodedVision:
    rows: Any        # device int64 [V]: prompt positions of every image-span token, ascending
    features: Any    # device [V, 5120]: the spans concatenated in prompt order


def encode(tower: VisionTower, prepared: PreparedVision, start: int = 0, out_dtype=None) -> EncodedVision:
    """Every span of ``prepared`` that ends after prompt position ``start`` (a prefix hit before ``start`` keeps
    earlier spans out); spans straddling ``start`` are encoded whole and their rows before ``start`` dropped."""
    torch = tower.torch
    rows, features = [], []
    for frame, (s, e) in zip(prepared.frames, prepared.blocks):
        if e <= start:
            continue
        span = tower.encode(frame, out_dtype)
        lo = max(s, start)
        rows.append(torch.arange(lo, e, device=tower.device))
        features.append(span[lo - s:])
    if not rows:
        empty = torch.empty(0, OUT_DIM, dtype=out_dtype or torch.float16, device=tower.device)
        return EncodedVision(torch.empty(0, dtype=torch.int64, device=tower.device), empty)
    return EncodedVision(torch.cat(rows), torch.cat(features))
