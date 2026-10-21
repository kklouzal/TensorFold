"""ModelOpt and compressed-tensors checkpoints of Qwen3.8-27B into ``Weights``: each projection by its tensors' scheme."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tensorfold.checkpoint_metadata import read_config
from tensorfold.cuda import prompt_precision

from .weights import Plain

SUFFIXES = ("weight", "weight_packed", "weight_scale", "weight_scale_2", "weight_global_scale", "input_scale",
            "input_global_scale")


def quantized(model_dir: Path) -> bool:
    """Whether config.json names ModelOpt or compressed-tensors weights (NVFP4, FP8 or bf16 per projection)."""

    from tensorfold.cuda.nvfp4.format import is_quantized

    return is_quantized(model_dir)


def skipped(name: str) -> bool:
    """Tensors the text model never reads: the vision tower and the MTP head."""

    return (name.startswith(("model.visual.", "visual.", "vision_tower", "mtp.")) or ".visual." in name
            or ".mtp." in name)


@dataclass
class Plain8(Plain):
    """A bf16 projection (the GDN gates) with an e4m3 copy made at load for --prefill-fp8's prompts."""

    rows8: object = None      # tensorfold.cuda.nvfp4.linear.Fp8Linear

    def prefill8(self, xq):
        return self.rows8.prefill8(xq)

    def nbytes(self) -> int:
        return super().nbytes() + self.rows8.nbytes()


@dataclass(frozen=True)
class MLPContext:
    """Borrowed loading context for a family-specific MLP constructor.

    Names are checkpoint-absolute. ``read`` consumes each tensor exactly once,
    on the requested device; an expert factory can therefore build its final
    CPU authority without first placing expert payload on the GPU. The callback
    must finish before returning and retain tensors, not this reader context.
    ``read_f32_scalars`` consumes a bounded CPU F32[1] batch once after its complete read;
    names already consumed by either callback fail before any batch I/O.
    ``prepare_cpu_run`` checks a complete equal-byte physical run without I/O.
    Its optional loading-only callable owns fresh CPU backing per bounded field
    interval and consumes that interval only after a successful synchronous read.
    Unsupported reader/topology returns None; metadata/resource errors propagate.
    ``used`` records actual native activation math for the startup declaration.
    """

    info: dict[str, tuple[str, list[int]]]
    metadata: Any
    read: Callable
    read_f32_scalars: Callable
    prepare_cpu_run: Callable
    linear: Callable
    checkpoint_math: bool
    own: dict[str, bool]
    used: dict[str, bool]


def weight_bytes(name: str, info: dict) -> tuple[int, int]:
    """A tensor as loaded: stored bytes with outputs padded to 128 (FP8 channel scales as fp32), gates' e4m3 copies,
    A_log and dt_bias in fp32."""

    if skipped(name):
        return 0, 0
    shape, dtype = list(info["shape"]), info["dtype"]
    if name.endswith((".A_log", ".dt_bias")):
        return math.prod(shape) * 4, 0
    from tensorfold.cuda.capacity import SIZES

    if name.endswith(".weight_scale") and dtype in ("F32", "F16", "BF16") and math.prod(shape) > 1:
        return -(-math.prod(shape) // 128) * 128 * 4, 0       # an output channel's scale each (``Fp8Linear.cols``)
    if len(shape) == 2 and dtype in ("U8", "F8_E4M3") and not name.endswith("_scale"):
        shape[0] = -(-shape[0] // 128) * 128
    amount = math.prod(shape) * SIZES[dtype]
    if name.endswith(("in_proj_a.weight", "in_proj_b.weight")) and len(shape) == 2 and prompt_precision.fp8():
        npad = -(-shape[0] // 128) * 128
        amount += npad * shape[1] + shape[1] // 64 * npad * 2
    return amount, 0


def admission(geometry):
    """The MLX path's geometry (FP8 prompts: plus the widest NVFP4 projection's e4m3 staging) and tensor bytes."""

    from tensorfold.cuda.geometry import with_fixed

    def with_staging(text):
        d, i = int(text["hidden_size"]), int(text["intermediate_size"])
        return with_fixed(geometry(text), d * i + d * i // 32 + (4 << 20)) if prompt_precision.fp8() else geometry(text)

    return with_staging, weight_bytes


FULL_LINE = "full (bf16 activations, the stored weights exactly)"


def maths() -> tuple[dict[str, bool], str]:
    """Each format's math on this GPU under --precision, and the startup line naming it; no supported GPU is refused."""

    from tensorfold.cuda import precision

    if precision.mode() != precision.CHECKPOINT or not torch.cuda.is_available():
        return {"nvfp4": False, "fp8": False}, FULL_LINE
    major, minor = torch.cuda.get_device_capability()
    own = precision.own_math((major, minor))
    if all(own.values()):
        return own, ("checkpoint (the checkpoint's own math: its NVFP4 layers FP4 x FP4, per-16 scales under its "
                     "static input scales; its FP8 layers FP8 x FP8)")
    gpu = f"{torch.cuda.get_device_name()}, SM {major}.{minor}"
    if own["fp8"]:
        return own, (f"checkpoint where this GPU has it ({gpu}): its FP8 layers FP8 x FP8 under their static input "
                     "scales; its NVFP4 layers W4A16 (bf16 activations: the block-scaled FP4 mma is SM 12.x's)")
    return own, f"full ({gpu} has neither the block-scaled FP4 nor the FP8 mma): bf16 activations, the stored weights"


def load_nvfp4(model_dir: str | Path, device: str = "cuda", *, eos_token_ids=None, mlp=None, rope=None):
    """NVFP4 and FP8 projections at full precision or in the checkpoint's own math, bf16 ones as stored (``rope``:
    a static YaRN policy for a default-RoPE checkpoint)."""

    from tensorfold.cuda.capacity import headers
    from tensorfold.cuda.nvfp4 import format as fmt
    from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear, Staging

    from .weights import (
        GDN,
        Attention,
        Config,
        Layer,
        Weights,
        _close_failed_checkpoint,
        _rotary_frequencies,
        _Tensors,
        _validate_fp8_prefill,
    )

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir, **{name: value for name, value in (("eos_token_ids", eos_token_ids), ("rope", rope))
                                    if value is not None})
    block = read_config(model_dir).get("quantization_config") or {}
    metadata = fmt.LinearMetadata(block)
    reciprocal = str(block.get("quant_method", "")).lower() == "compressed-tensors"
    info = {n: (i["dtype"], i["shape"]) for n, i in headers(model_dir).items() if not skipped(n)}
    root = "model.language_model." if any(n.startswith("model.language_model.") for n in info) else "model."
    t = _Tensors(model_dir, device, skip=skipped)
    try:
        staging = Staging()                                   # one e4m3 copy at a time, shared by every NVFP4 projection
        own, _ = maths()
        used = {"nvfp4": False, "fp8": False}
        per_token = {"fp8": False}                           # FP8 layers whose rows take their own scales
        from tensorfold.cuda import precision

        checkpoint_math = precision.mode() == precision.CHECKPOINT
        if prompt_precision.fp8() and not any(own.values()):
            _validate_fp8_prefill(cfg, 1)

        def act_scale(name: str, got: dict, kind: str, activation: int | None,
                      weight_scale: float) -> tuple[float | None, float | None]:
            """Only declared, supported activation math consumes a static calibration."""
            if activation is None or not own[kind]:
                return None, None
            field = "input_scale" if "input_scale" in got else "input_global_scale"
            original = float(got[field].float().reshape(-1)[0])
            scale = fmt.calibration(original, name=name + "." + field,
                                    reciprocal=field == "input_global_scale")
            # CT's quantizer consumes this stored multiplier directly. Taking
            # two FP32 reciprocals can change its bits; alpha still uses scale.
            inverse = original if reciprocal and kind == "nvfp4" and field == "input_global_scale" else None
            fmt.native_calibration(scale, weight_scale, name=name,
                                   **({"act_inverse": inverse} if inverse is not None else {}))
            used[kind] = True
            return scale, inverse

        def linear(name: str, prompt: bool = True):
            parts = {s: info[f"{name}.{s}"] for s in SUFFIXES if f"{name}.{s}" in info}
            if not parts:
                raise ValueError(f"the checkpoint has no projection {name}")
            kind, activation = metadata.validate(name, parts, checkpoint=checkpoint_math)
            if kind not in ("nvfp4", "fp8", "fp8channel", "bf16"):
                raise ValueError(f"{name}: {kind} projections are not read on Qwen3.8-27B yet")
            if kind == "fp8channel" and prompt_precision.fp8():
                raise ValueError(f"{name}: --prefill-fp8 reads per-tensor FP8 weights; this checkpoint's "
                                 "per-channel FP8 projections run FP8 prompts under --precision checkpoint")
            got = {s: t.pop(f"{name}.{s}") for s in parts}
            weight = got["weight"] if "weight" in got else got.get("weight_packed")
            if kind == "nvfp4":
                g = got.get("weight_global_scale" if reciprocal else "weight_scale_2")
                if g is None:
                    raise ValueError(f"{name}: an NVFP4 weight without its global scale")
                g = fmt.calibration(float(g.float().reshape(-1)[0]), name=name + ".weight_global_scale",
                                    reciprocal=reciprocal)
                act, inverse = act_scale(name, got, "nvfp4", activation, g)
                lin = Fp4Linear.from_checkpoint(weight, got["weight_scale"], g, act=act, act_inverse=inverse)
                lin.staging = staging
                return lin
            if kind == "fp8channel":
                # Per-token rows under checkpoint math on an FP8-mma GPU (``activation`` is only returned when the
                # checkpoint declares them); otherwise W8A16 with each channel's scale in the epilogue.
                token = activation is not None and own["fp8"]
                if token:
                    used["fp8"] = per_token["fp8"] = True
                return Fp8Linear.from_checkpoint(weight, got["weight_scale"], token=token)
            if kind == "fp8":
                s = got["weight_scale"].float().reshape(-1)
                if s.numel() != 1:
                    raise ValueError(f"{name}: FP8 with {s.numel()} scales; the CUDA engine reads one scale a tensor")
                scale = fmt.calibration(float(s[0]), name=name + ".weight_scale")
                return Fp8Linear.from_checkpoint(weight, scale,
                                                 act=act_scale(name, got, "fp8", activation, scale)[0])
            w = weight.to(torch.bfloat16).contiguous()
            return Plain8(w, rows8=Fp8Linear.from_bf16(w)) if prompt and prompt_precision.fp8() else Plain(w)

        def get(name: str) -> torch.Tensor:
            return t.pop(root + name).contiguous()

        def read(name: str, *, device=None) -> torch.Tensor:
            if device is None:
                return t.pop(name)
            del t.where[name]
            return t.files.get(name, device)

        def read_f32_scalars(names, *, max_bytes: int) -> torch.Tensor:
            names = tuple(names)
            for name in names:
                if name not in t.where:
                    raise KeyError(name)
            values = t.files.get_f32_scalars(names, max_bytes=max_bytes)
            for name in names:
                del t.where[name]
            return values

        def prepare_cpu_run(names, *, dtype: str, field_bytes: int, max_read_bytes: int):
            names = tuple(names)
            for name in names:
                if name not in t.where:
                    raise KeyError(name)
            run = t.files.prepare_cpu_run(names, dtype=dtype, field_bytes=field_bytes,
                                          max_read_bytes=max_read_bytes)
            if run is None:
                return None

            def read_run(first: int, stop: int):
                if (type(first) is not int or type(stop) is not int
                        or not 0 <= first < stop <= len(names)):
                    raise ValueError("owned CPU run requires a nonempty in-range field interval")
                selected = names[first:stop]
                for name in selected:
                    if name not in t.where:
                        raise KeyError(name)
                value = run.read(first, stop)
                for name in selected:
                    del t.where[name]
                return value

            return read_run

        context = (MLPContext(info, metadata, read, read_f32_scalars, prepare_cpu_run, linear, checkpoint_math, own, used)
                   if mlp is not None else None)

        def norm(name: str) -> torch.Tensor:
            """A centred RMSNorm weight (stored as gamma - 1) with its 1 back, as the MLX converter wrote it."""

            w = get(name)
            return (w.float() + 1.0).to(w.dtype)

        layers = []
        for i in range(cfg.layers):
            p = f"layers.{i}."
            q = root + p
            gdn = attn = None
            if cfg.is_linear(i):
                gdn = GDN(qkv=linear(q + "linear_attn.in_proj_qkv"), z=linear(q + "linear_attn.in_proj_z"),
                          b=linear(q + "linear_attn.in_proj_b"), a=linear(q + "linear_attn.in_proj_a"),
                          out=linear(q + "linear_attn.out_proj"),
                          conv=get(p + "linear_attn.conv1d.weight").reshape(-1, cfg.conv_kernel).contiguous(),
                          A_log=get(p + "linear_attn.A_log").float().contiguous(),
                          dt_bias=get(p + "linear_attn.dt_bias").float().contiguous(),
                          norm=get(p + "linear_attn.norm.weight"))
            else:
                attn = Attention(q=linear(q + "self_attn.q_proj"), k=linear(q + "self_attn.k_proj"),
                                 v=linear(q + "self_attn.v_proj"), o=linear(q + "self_attn.o_proj"),
                                 q_norm=norm(p + "self_attn.q_norm.weight"), k_norm=norm(p + "self_attn.k_norm.weight"))
            input_norm = norm(p + "input_layernorm.weight")
            post_norm = norm(p + "post_attention_layernorm.weight")
            fields = (mlp(q + "mlp.", context, cfg) if mlp is not None else
                      {"gate": linear(q + "mlp.gate_proj"), "up": linear(q + "mlp.up_proj"),
                       "down": linear(q + "mlp.down_proj")})
            layers.append(Layer(linear=cfg.is_linear(i), input_norm=input_norm,
                                post_norm=post_norm, gdn=gdn, attn=attn,
                                **{"gate": None, "up": None, "down": None, **fields}))
        # Layer packing has finished; unused staging bins need not coexist
        # with the late embedding/head allocations in the reserved footprint.
        torch.cuda.empty_cache()
        if not any(n.startswith("lm_head.") for n in info):
            raise ValueError("this checkpoint ties its head to the embedding; the CUDA engine reads a separate lm_head")
        w = Weights(config=cfg, embed=Plain(get("embed_tokens.weight").to(torch.bfloat16)), layers=layers,
                    norm=norm("norm.weight"), head=linear("lm_head", prompt=False), quant="nvfp4")
        w.precision = "checkpoint" if any(used.values()) else "full"
        w.own = used
        if any(used.values()):
            fp8 = "FP8 layers FP8 x FP8" + (" (rows quantized a token at a time)" if per_token["fp8"] else "")
            native = [label for kind, label in (("nvfp4", "NVFP4 layers FP4 x FP4"), ("fp8", fp8)) if used[kind]]
            line = "checkpoint (declared " + "; ".join(native) + "; other layers use bf16 activations)"
        else:
            line = FULL_LINE
        print(f"[tensorfold] precision: {line}", flush=True)
        w.inv_freq = _rotary_frequencies(cfg, device)
        if any(not layer.linear for layer in w.layers):
            _ = w.attention_origin                    # resolve before an attention runtime/capture
        left = list(t)
    except BaseException as error:
        _close_failed_checkpoint(t, error)
        raise
    t.close()
    if left:
        raise ValueError(f"unused checkpoint tensors: {left[:5]} ...")
    torch.cuda.empty_cache()
    return w
