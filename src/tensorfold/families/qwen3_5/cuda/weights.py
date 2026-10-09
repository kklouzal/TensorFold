"""Load each projection's declared affine format without changing its packed checkpoint words."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from tensorfold.cuda.tensor_file import checkpoint_path, read_metadata_json
from typing import Any

import torch


@dataclass
class QLinear:
    weight: torch.Tensor      # original MLX words, or the optimized four-bit tile layout
    scales: torch.Tensor | None
    biases: torch.Tensor | None
    layout: str = "mlx"       # "mlx" as stored, or "tiled" (``qmm_fast.tile``)
    rows: int = 0             # N when tiled (the tiled words are padded to 64 columns)
    gs: int = 64              # inputs per quantization group
    bits: int = 4

    @property
    def n(self) -> int:
        return self.rows if self.layout == "tiled" else int(self.weight.shape[0])

    @property
    def k(self) -> int:
        if self.layout == "dense":
            return int(self.weight.shape[1])
        return int(self.weight.shape[1]) * 64 if self.layout == "tiled" else int(self.scales.shape[1]) * self.gs

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases) if t is not None)

    @cached_property
    def fast(self) -> bool:
        return (self.bits, self.gs) == (4, 64) and self.layout != "dense" and all(
            t.dtype == torch.bfloat16 for t in (self.scales, self.biases))


@dataclass
class Plain:
    """A weight an EXL3 pack stores unquantized (embedding, GDN in_proj_a/b), read by ``b16.matmul`` one warp an output."""

    weight: torch.Tensor      # (N, K) fp16 or bf16
    layout: str = "b16"

    @property
    def n(self) -> int:
        return int(self.weight.shape[0])

    @property
    def k(self) -> int:
        return int(self.weight.shape[1])

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from .b16 import matmul

        return matmul(x, self.weight)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Prompt rows on the bf16 mma (chunk-invariant bits, not decode's)."""

        from .b16 import prompt

        return prompt(x, self.weight)


@dataclass
class Exl3:
    """An EXL3 trellis projection on ``tensorfold.cuda.exl3``'s row-invariant linear, any row count in 128-row calls."""

    layer: object             # tensorfold.cuda.exl3.linear.Exl3Linear
    layout: str = "exl3"
    workspace: object = None  # the pack's shared ``tensorfold.cuda.exl3.prefill.Workspace`` (prompts only)

    @property
    def n(self) -> int:
        return int(self.layer.n)

    @property
    def k(self) -> int:
        return int(self.layer.k)

    def nbytes(self) -> int:
        lin = self.layer
        parts = [lin.words, lin.suh, lin.svh, lin.counters] + ([lin.bias] if lin.bias is not None else [])
        return sum(t.numel() * t.element_size() for t in parts)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 128:
            return self.layer(x)
        out = torch.empty((x.shape[0], self.n), dtype=x.dtype, device=x.device)
        for r0 in range(0, x.shape[0], 128):                  # rows are independent, so slices keep their bits
            self.layer(x[r0:r0 + 128], out=out[r0:r0 + 128])
        return out

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Prompt rows through ``cuda/exl3/prefill.py``: bits never depend on the row count."""

        from tensorfold.cuda.exl3.prefill import matmul

        return matmul(self.layer, x, torch.empty((x.shape[0], self.n), dtype=torch.bfloat16, device=x.device),
                      self.workspace)


@dataclass
class Config:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    k_heads: int              # GDN key heads
    v_heads: int              # GDN value heads
    dk: int
    dv: int
    conv_kernel: int
    interval: int             # every interval-th layer is full attention
    eps: float
    rope_dims: int
    rope_theta: float
    eos: tuple[int, ...]
    experts: int = 0          # routed experts a MoE layer picks from (0: dense MLPs)
    top_k: int = 0
    moe_width: int = 0
    mrope_section: tuple[int, int, int] = (11, 11, 10)

    @classmethod
    def read(cls, model_dir: str | Path) -> "Config":
        raw = read_metadata_json(checkpoint_path(model_dir, "config.json"))
        t = raw.get("text_config", raw)
        if not isinstance(t, dict):
            raise ValueError("text_config must be a JSON object")
        rope = t.get("rope_parameters")
        rope = {} if rope is None else rope
        if not isinstance(rope, dict):
            raise ValueError("rope_parameters must be a JSON object")
        head_dim = int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"])
        gen = Path(model_dir) / "generation_config.json"
        found: list[int] = []
        # EXL3 packs name the chat turn's end, <|im_end|>, in generation_config.json only
        for value in (raw.get("eos_token_id", t.get("eos_token_id")),
                      read_metadata_json(checkpoint_path(model_dir, "generation_config.json")).get("eos_token_id") if gen.exists() else None):
            found += [] if value is None else [int(e) for e in value] if isinstance(value, list) else [int(value)]
        if not found:
            raise ValueError("no eos_token_id in config.json or generation_config.json")
        eos = tuple(dict.fromkeys(found))
        config = cls(
            hidden=int(t["hidden_size"]), intermediate=int(t.get("intermediate_size", 0)),
            layers=int(t["num_hidden_layers"]), heads=int(t["num_attention_heads"]),
            kv_heads=int(t["num_key_value_heads"]), head_dim=head_dim, vocab=int(t["vocab_size"]),
            k_heads=int(t["linear_num_key_heads"]), v_heads=int(t["linear_num_value_heads"]),
            dk=int(t["linear_key_head_dim"]), dv=int(t["linear_value_head_dim"]),
            conv_kernel=int(t["linear_conv_kernel_dim"]), interval=int(t.get("full_attention_interval", 4)),
            eps=float(t.get("rms_norm_eps", 1e-6)),
            rope_dims=int(head_dim * float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))),
            rope_theta=float(rope.get("rope_theta", t.get("rope_theta") or 10000000.0)),
            eos=eos, experts=int(t.get("num_experts", 0)), top_k=int(t.get("num_experts_per_tok", 0)),
            moe_width=int(t.get("moe_intermediate_size", 0)),
            mrope_section=tuple(rope.get("mrope_section", (11, 11, 10))),
        )
        from triton.language.core import TRITON_MAX_TENSOR_NUMEL

        config.validate_geometry(max_tensor_numel=TRITON_MAX_TENSOR_NUMEL)
        return config

    def validate_geometry(self, *, max_tensor_numel: int) -> None:
        """Admit normalized fields for the current CUDA paths before reading weights.

        Use the installed Triton tensor capacity. Attention/GDN constraints apply
        only to layers selected by this configuration; unused metadata is retained.
        Optional precision-specific plans validate their additional spans separately.
        """
        if type(self.hidden) is not int or self.hidden <= 0:
            raise ValueError("positive hidden width required for embedding and normalization")
        block = 1 << (self.hidden - 1).bit_length()
        if type(max_tensor_numel) is not int or max_tensor_numel < 64:
            raise ValueError("installed Triton tensor capacity must admit64-cell normalization")
        if block < 64 or block > max_tensor_numel:
            raise ValueError("normalization block must admit complete64-cell groups within installed Triton capacity")
        if type(self.vocab) is not int or self.vocab <= 0:
            raise ValueError("positive vocabulary required for model token emission")
        if type(self.layers) is not int or self.layers < 0:
            raise ValueError("layer count must be nonnegative")
        if type(self.interval) is not int or self.interval <= 0:
            raise ValueError("full-attention interval must be positive")
        if type(self.experts) is not int or self.experts < 0:
            raise ValueError("expert count must be nonnegative")
        if type(self.head_dim) is not int or self.head_dim < 0:
            raise ValueError("head width must be nonnegative")
        if type(self.rope_dims) is not int or not 0 <= self.rope_dims <= self.head_dim:
            raise ValueError("frequency allocation/rotation must lie within the head width")
        attention = self.layers >= self.interval
        linear = self.layers > 0 and self.interval > 1
        if attention:
            if (type(self.heads) is not int or self.heads <= 0 or type(self.kv_heads) is not int
                    or self.kv_heads <= 0 or self.heads % self.kv_heads or self.heads // self.kv_heads > 16):
                raise ValueError("attention requires positive whole query/KV head groups of at most sixteen queries")
            # The required shared tree attention wrapper admits D128/256 and at
            # most sixteen query heads per KV head. Its largest K/V block has64D
            # elements, so the installed provider capacity applies to that product.
            if self.head_dim not in (128, 256) or 64 * self.head_dim > max_tensor_numel or self.rope_dims < 2:
                raise ValueError("attention requires128/256-cell heads within installed64D tensor capacity and at least one stored frequency")
        if linear:
            if (type(self.k_heads) is not int or self.k_heads <= 0 or type(self.v_heads) is not int
                    or self.v_heads <= 0 or self.v_heads % self.k_heads):
                raise ValueError("GDN requires positive whole value/key head groups")
            # Required by shared GDN tree and prefill C++ entry contracts, not hardware inference.
            if type(self.dk) is not int or self.dk != 128 or type(self.dv) is not int or self.dv != 128 or type(self.conv_kernel) is not int or self.conv_kernel < 1:
                raise ValueError("current GDN prompt/decode paths require128-key/value dimensions and positive convolution width")
        if self.layers:
            if self.experts:
                if (type(self.top_k) is not int or not 1 <= self.top_k <= self.experts
                        or type(self.moe_width) is not int or self.moe_width <= 0):
                    raise ValueError("routed layers require bounded expert picks and positive expert width")
            elif type(self.intermediate) is not int or self.intermediate <= 0:
                raise ValueError("dense layers require positive intermediate width")

    def is_linear(self, layer: int) -> bool:
        return (layer + 1) % self.interval != 0

    def validate_fp8_prefill(self, *, max_tensor_numel: int, world: int = 1) -> None:
        """Admit the selected FP8 glue's rounded local-head blocks at startup.

        ``world`` is the rank count of the weights' eventual prefill layout,
        before payload loading/sharding. BF16 and checkpoint-specific prompt
        math retain their own geometry contract.
        """
        if type(world) is not int or world not in (1, 2):
            raise ValueError("FP8 prefill layout requires one or two declared ranks")
        if type(max_tensor_numel) is not int or max_tensor_numel < 64:
            raise ValueError("installed Triton tensor capacity must admit64-cell FP8 glue")
        active = []
        if self.layers >= self.interval:
            active.append((self.heads, self.head_dim, "attention"))
        if self.layers > 0 and self.interval > 1:
            active.append((self.v_heads, self.dv, "GDN"))
        for heads, width, role in active:
            if heads % world or heads // world <= 0:
                raise ValueError(role + " head count must split over the prefill ranks")
            local = heads // world
            rounded = 1 << (local - 1).bit_length()
            if rounded * width > max_tensor_numel:
                raise ValueError(role + " FP8 prefill rounded-head block exceeds installed Triton tensor capacity")


@dataclass
class GDN:
    qkv: QLinear
    z: QLinear
    b: QLinear
    a: QLinear
    out: QLinear
    conv: torch.Tensor        # (conv_dim, kernel) bf16
    A_log: torch.Tensor       # (Hv,) fp32
    dt_bias: torch.Tensor     # (Hv,) fp32
    norm: torch.Tensor        # (Dv,) bf16
    zba: QLinear | None = None  # [z | b | a] stacked into one projection (``qmm_fast.stack``)


@dataclass
class Attention:
    q: QLinear                # (heads * head_dim * 2, hidden): [q_h | gate_h] per head
    k: QLinear
    v: QLinear
    o: QLinear
    q_norm: torch.Tensor      # (head_dim,) bf16
    k_norm: torch.Tensor
    kv: QLinear | None = None   # [k | v] stacked


@dataclass
class Layer:
    linear: bool
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    gdn: GDN | None
    attn: Attention | None
    gate: QLinear | None
    up: QLinear | None
    down: QLinear | None
    moe: Any = None           # routed experts and a shared expert (``tensorfold.cuda.moe``) instead of gate/up/down


@dataclass
class Weights:
    config: Config
    embed: QLinear | Plain
    layers: list[Layer]
    norm: torch.Tensor
    head: Any                                        # QLinear, Exl3, or an NVFP4 checkpoint's linear
    inv_freq: torch.Tensor | None = None             # (rope_dims/2,) fp32
    quant: str = "mlx"                               # "exl3": an EXL3 pack (prompt glue then stays in bf16); "nvfp4"
    prompt_rows: int = 4096                          # a prompt chunk's rows, sized to the GPU: any count, the same bits

    @cached_property
    def attention_origin(self) -> torch.Tensor:
        """The model's address-only view; frequency storage is fixed for its lifetime."""
        from tensorfold.cuda.kernels.attention import address_origin

        return address_origin(self.inv_freq)

    @cached_property
    def fast_prefill(self) -> bool:
        """Whether every projection has an FP8 prompt kernel (run when prompts take FP8)."""

        if self.quant == "exl3" or getattr(self, "precision", "full") == "checkpoint":
            return False                             # EXL3 prompt glue stays bf16; checkpoint math has its own
        for layer in self.layers:
            modules = [m for m in (layer.gate, layer.up, layer.down) if m is not None]    # a MoE layer's are None
            modules += [layer.gdn.qkv, layer.gdn.z, layer.gdn.b, layer.gdn.a, layer.gdn.out] if layer.gdn else []
            modules += [layer.attn.q, layer.attn.k, layer.attn.v, layer.attn.o] if layer.attn else []
            if self.quant == "nvfp4":                # NVFP4 and FP8 have one; bf16 gates only with their e4m3 copies
                if any(not hasattr(q, "prefill8") for q in modules):
                    return False
            elif any(not q.fast for q in modules):
                return False
        return True

    def nbytes(self) -> int:
        total = self.embed.nbytes() + self.head.nbytes()
        for layer in self.layers:
            mods = [m for m in (layer.gate, layer.up, layer.down) if m is not None]
            mods += [layer.gdn.qkv, layer.gdn.z, layer.gdn.b, layer.gdn.a, layer.gdn.out] if layer.gdn else []
            mods += [layer.attn.q, layer.attn.k, layer.attn.v, layer.attn.o] if layer.attn else []
            total += sum(m.nbytes() for m in mods)
        return total


class _Tensors:
    """Checkpoint tensors read one at a time, so the weights never sit in device memory twice while they pack."""

    def __init__(self, model_dir: Path, device: str, skip=None) -> None:
        from tensorfold.cuda.direct_read import SafeTensors

        skip = skip or (lambda name: name.startswith("vision_tower") or ".mtp." in name or name.startswith("mtp."))
        self.device, self.files = device, SafeTensors(sorted(model_dir.glob("*.safetensors")))
        self.where = {name: None for name in self.files.keys() if not skip(name)}

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def __iter__(self):
        return iter(list(self.where))

    def pop(self, name: str) -> torch.Tensor:
        del self.where[name]
        return self.files.get(name, self.device)

    def close(self) -> None:
        self.files.close()                    # the reader's pinned staging goes back to the system


def _fp8_prefill_metadata(cfg: Config, raw: dict, reader: _Tensors, prefix: str,
                        mlp_names: tuple[str, ...]) -> bool:
    """Prove the existing fast_prefill predicate from already-read headers."""
    from tensorfold.quantization import resolve_affine

    for i in range(cfg.layers):
        base = prefix + f"model.layers.{i}."
        names = [base + "mlp." + name for name in mlp_names]
        names += ([base + "linear_attn." + name for name in
                   ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")]
                  if cfg.is_linear(i) else
                  [base + "self_attn." + name for name in ("q_proj", "k_proj", "v_proj", "o_proj")])
        for name in names:
            spec = resolve_affine(raw, name)
            if spec is None or (spec.bits, spec.group_size) != (4, 64):
                return False
            for suffix in (".scales", ".biases"):
                info = reader.files.where.get(name + suffix)
                if info is None or info[3] != "BF16":
                    return False
    return True


def _validate_fp8_prefill(cfg: Config, world: int) -> None:
    from triton.language.core import TRITON_MAX_TENSOR_NUMEL

    cfg.validate_fp8_prefill(max_tensor_numel=TRITON_MAX_TENSOR_NUMEL, world=world)


def _close_failed_checkpoint_impl(reader, primary: BaseException) -> None:
    """Drain one failed reader exactly once and retain its failure context.

    Native cause/context is captured before foreign cleanup. Every distinct
    cleanup/annotation exception survives without formatting the exception or
    letting malformed notes replace the original primary object.
    """
    previous = (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary))
    try:
        reader.close()
    except BaseException as cleanup:
        others = []
        for error in (*previous, cleanup):
            if error is not None and error is not primary and all(error is not item for item in others):
                others.append(error)
        if cleanup is primary and not others:
            raise primary
        annotation_failed = False
        try:
            BaseException.add_note(primary, "Qwen checkpoint reader cleanup also failed; its owned staging remains retained")
        except BaseException as annotation:
            annotation_failed = True
            if annotation is not primary and all(annotation is not item for item in others):
                others.append(annotation)
        if len(others) == 1 and not annotation_failed:
            raise primary from others[0]
        if others:
            raise primary from BaseExceptionGroup("Qwen checkpoint retained native and cleanup failures", others)
        raise primary


def _close_failed_checkpoint(reader, primary: BaseException) -> None:
    """Retain reader failure context if cold cleanup transport exhausts.

    A failed drain is attempted once. Captured native fields and the reader
    remain frame-owned if diagnostics/grouping cannot allocate; that transport
    failure becomes the explicit cause without replacing the original primary.
    The caller may retire these references by clearing the failure traceback.
    """
    native_cause = BaseException.__cause__.__get__(primary)
    native_context = BaseException.__context__.__get__(primary)
    if native_cause is primary:
        native_cause = None
    if native_context is primary:
        native_context = None
    try:
        _close_failed_checkpoint_impl(reader, primary)
    except BaseException as failure:
        if failure is primary:
            raise
        # Keep captured native fields in this traceback through foreign close.
        raise primary from failure


def load(model_dir: str | Path, device: str = "cuda", *, tiled: bool = False, mlp=None,
         prefill_world: int = 1, prefill_mlp: tuple[str, ...] | None = None) -> Weights:
    """MLX affine 4-bit, EXL3 or NVFP4 weights; ``tiled`` packs projections as read.

    ``prefill_world`` describes the eventual rank layout before loading its full
    weights. A custom ``mlp`` may declare its consumed prompt projection names
    through ``prefill_mlp`` (empty for routed experts); otherwise its selected
    FP8 geometry is checked after construction, before any prompt execution.
    """

    from .exl3_load import load_exl3, quant_config
    from .nvfp4_load import load_nvfp4, quantized

    model_dir = Path(model_dir)
    if (prefill_mlp is not None and (type(prefill_mlp) is not tuple
            or any(type(name) is not str or name not in ("gate_proj", "up_proj", "down_proj") for name in prefill_mlp)
            or len(set(prefill_mlp)) != len(prefill_mlp))):
        raise ValueError("custom prefill MLP metadata must name its unique gate/up/down projections")
    if quant_config(model_dir) is not None:
        return load_exl3(model_dir, device)
    if quantized(model_dir):
        return load_nvfp4(model_dir, device)
    cfg = Config.read(model_dir)
    raw = read_metadata_json(checkpoint_path(model_dir, "config.json"))
    t = _Tensors(model_dir, device)
    try:
        prefix = "language_model." if any(k.startswith("language_model.") for k in t) else ""
        from tensorfold.cuda import prompt_precision

        preflight = mlp is None or prefill_mlp is not None
        mlp_names = ("gate_proj", "up_proj", "down_proj") if mlp is None else prefill_mlp
        if (prompt_precision.fp8() and preflight
                and _fp8_prefill_metadata(cfg, raw, t, prefix, mlp_names)):
            _validate_fp8_prefill(cfg, prefill_world)

        def get(name: str) -> torch.Tensor:
            return t.pop(prefix + name)

        def qlinear(name: str, pack: bool = True) -> QLinear:
            from tensorfold.quantization import resolve_affine, validate_shapes

            w = get(name + ".weight")
            spec = resolve_affine(raw, prefix + name)
            if spec is None:
                if (prefix + name + ".scales") in t or (prefix + name + ".biases") in t:
                    raise ValueError(f"{name} has packed weights but no enabled affine metadata")
                if w.ndim != 2 or w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(f"{name} needs floating weights or declared affine metadata")
                return QLinear(w.contiguous(), None, None, layout="dense", bits=0, gs=0)
            if w.dtype not in (torch.int32, torch.uint32):
                raise ValueError(f"{name} declares affine quantization but its words are not 32-bit integers")
            w = w.view(torch.int32) if w.dtype != torch.int32 else w
            scales, biases = get(name + ".scales"), get(name + ".biases")
            if any(value.dtype not in (torch.bfloat16, torch.float16, torch.float32) for value in (scales, biases)):
                raise ValueError(f"{name} needs floating-point affine scales and biases")
            validate_shapes(w.shape, scales.shape, biases.shape, spec)
            q = QLinear(w.contiguous(), scales.contiguous(), biases.contiguous(), gs=spec.group_size, bits=spec.bits)
            if tiled and pack:
                from .qmm_fast import tile

                return tile(q)                                 # tile() leaves any format but 4-bit g64 as stored
            return q

        layers = []
        for i in range(cfg.layers):
            p = f"model.layers.{i}."
            gdn = attn = None
            if cfg.is_linear(i):
                gdn = GDN(qkv=qlinear(p + "linear_attn.in_proj_qkv"), z=qlinear(p + "linear_attn.in_proj_z"),
                          b=qlinear(p + "linear_attn.in_proj_b"), a=qlinear(p + "linear_attn.in_proj_a"),
                          out=qlinear(p + "linear_attn.out_proj"),
                          conv=get(p + "linear_attn.conv1d.weight").reshape(-1, cfg.conv_kernel).contiguous(),
                          A_log=get(p + "linear_attn.A_log").float().contiguous(),
                          dt_bias=get(p + "linear_attn.dt_bias").float().contiguous(),
                          norm=get(p + "linear_attn.norm.weight").contiguous())
            else:
                attn = Attention(q=qlinear(p + "self_attn.q_proj"), k=qlinear(p + "self_attn.k_proj"),
                                 v=qlinear(p + "self_attn.v_proj"), o=qlinear(p + "self_attn.o_proj"),
                                 q_norm=get(p + "self_attn.q_norm.weight").contiguous(),
                                 k_norm=get(p + "self_attn.k_norm.weight").contiguous())
            fields = mlp(p + "mlp.", get, qlinear, cfg) if mlp is not None else \
                {"gate": qlinear(p + "mlp.gate_proj"), "up": qlinear(p + "mlp.up_proj"), "down": qlinear(p + "mlp.down_proj")}
            layers.append(Layer(linear=cfg.is_linear(i), input_norm=get(p + "input_layernorm.weight").contiguous(),
                                post_norm=get(p + "post_attention_layernorm.weight").contiguous(), gdn=gdn, attn=attn,
                                **{"gate": None, "up": None, "down": None, **fields}))
        w = Weights(config=cfg, embed=qlinear("model.embed_tokens", pack=False), layers=layers,
                    norm=get("model.norm.weight"), head=qlinear("lm_head"))
        half = cfg.rope_dims // 2
        inv = cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
        w.inv_freq = inv.to(torch.float32).to(device)
        if prompt_precision.fp8() and not preflight and w.fast_prefill:
            _validate_fp8_prefill(cfg, prefill_world)
        if any(not layer.linear for layer in w.layers):
            _ = w.attention_origin                    # resolve before an attention runtime/capture
        left = list(t)
    except BaseException as error:
        _close_failed_checkpoint(t, error)
        raise
    t.close()
    if left:
        raise ValueError(f"unused checkpoint tensors: {left[:5]} ...")
    if tiled:
        torch.cuda.empty_cache()
    return w
