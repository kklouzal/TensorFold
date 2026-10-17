"""NVFP4 / FP8 / MXFP8 checkpoints (ModelOpt, compressed-tensors): each linear's scheme, and a numpy dequantizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tensorfold.checkpoint_metadata import read_config
from tensorfold.checkpoint_quantization import LinearQuantization, effective_config

METHODS = ("modelopt", "compressed-tensors")
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=np.float32)


def config_block(config: dict) -> dict | None:
    """The quantization block of config.json (top level or text config) when it names ModelOpt or compressed-tensors."""

    block = effective_config(config, None, where="checkpoint").get("quantization_config")
    return block if block is not None and block.get("quant_method") in METHODS else None


def require_config(config: dict, *, where: str, tested: str, help: str) -> None:
    """Refuse a ModelOpt / compressed-tensors checkpoint whose weights are not NVFP4, FP8 or MXFP8."""

    block = config_block(config) or {}
    method = str(block.get("quant_method", "")).lower()
    formats = {str(block.get("format", "") or "")}
    groups = block.get("config_groups") or {}
    for group in groups.values():
        weights = group.get("weights") or {}
        formats.add(str(group.get("format", "") or ""))
        bits, kind = weights.get("num_bits"), str(weights.get("type", "float"))
        if kind != "float" or bits not in (4, 8):
            raise ValueError(f"this {method} checkpoint stores {bits}-bit {kind} weights; {where} reads NVFP4 and FP8 "
                             f"weights of this format. Tested checkpoints: {tested}. {help}")
    bad = {f for f in formats if f and f not in ("nvfp4-pack-quantized", "float-quantized", "naive-quantized",
                                                 "mixed-precision")}
    if bad:
        raise ValueError(f"this {method} checkpoint's weights are stored as {', '.join(sorted(bad))}; {where} reads "
                         f"NVFP4 and FP8 weights. Tested checkpoints: {tested}. {help}")


def is_quantized(model_dir: str | Path) -> bool:
    """Whether config.json names ModelOpt or compressed-tensors quantization."""

    try:
        (Path(model_dir) / "config.json").lstat()
    except FileNotFoundError:
        return False
    return config_block(read_config(model_dir)) is not None


def scheme(tensors: dict[str, tuple[str, list[int]]]) -> str:
    """One linear's scheme from its tensors' (dtype, shape) by suffix: nvfp4, fp8, mxfp8 or bf16."""

    w = tensors.get("weight") or tensors.get("weight_packed")
    s = tensors.get("weight_scale")
    if w is None:
        raise ValueError("a linear without weight or weight_packed")
    if w[0] == "U8" and s is not None and s[0] == "F8_E4M3":
        return "nvfp4"
    if w[0] == "F8_E4M3" and s is not None and s[0] == "U8":
        return "mxfp8"
    si = tensors.get("weight_scale_inv")
    if w[0] == "F8_E4M3" and si is not None and si[0] == "F32" and len(si[1]) == 2:
        return "fp8block"                      # ModelOpt FP8_PB_WO / DeepSeek: an fp32 scale per 128x128 block
    if w[0] == "F8_E4M3":
        return "fp8"
    if w[0] in ("BF16", "F16", "F32"):
        return "bf16"
    raise ValueError(f"unknown weight storage {w[0]} with scale {s[0] if s else None}")


class LinearMetadata:
    """A prepared declaration/storage cross-check for one model's linear reads."""

    def __init__(self, block):
        self.policy = LinearQuantization(block)
        self.reciprocal = block.get("quant_method") == "compressed-tensors"

    def validate(self, name, tensors, *, checkpoint: bool):
        """Return (stored kind, declared activation bits or None), before payload reads.

        ``checkpoint=False`` is an explicit/stored-weight math policy; it may
        disable declared activation quantization but never reinterpret weights.
        """
        if "weight" in tensors and "weight_packed" in tensors:
            raise ValueError(f"{name}: both weight and weight_packed are present")
        kind = scheme(tensors)
        weight = tensors.get("weight", tensors.get("weight_packed"))
        _, shape = weight
        if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f"{name}: linear weight must have a positive two-dimensional shape")
        n, k = shape[0], shape[1] * (2 if kind == "nvfp4" else 1)
        spec = self.policy.resolve(name)
        if spec is None:
            if kind != "bf16" and (checkpoint or self.policy.excluded(name)):
                raise ValueError(f"{name}: stored {kind} weights have no matching quantization declaration")
        else:
            declared = spec.get("weights")
            if declared is None:
                if kind != "bf16":
                    raise ValueError(f"{name}: metadata declares unquantized weights but storage is {kind}")
            elif declared.get("type", "float") != "float" or declared.get("num_bits") != (
                    4 if kind == "nvfp4" else 8 if kind in ("fp8", "mxfp8", "fp8block") else 16):
                raise ValueError(f"{name}: stored {kind} weights disagree with quantization metadata")
            elif kind == "nvfp4" and declared.get("group_size", 16) != 16:
                raise ValueError(f"{name}: stored NVFP4 blocks require group_size 16")
            if declared is not None:
                strategy = declared.get("strategy")
                allowed = (None, "group", "tensor_group") if kind == "nvfp4" else (None, "tensor")
                if kind in ("nvfp4", "fp8") and strategy not in allowed:
                    raise ValueError(f"{name}: unsupported weight scale strategy {strategy!r}")
                scale_dtype = declared.get("scale_dtype")
                expected_dtype = "torch.float8_e4m3fn" if kind == "nvfp4" else "torch.float32"
                if scale_dtype is not None and scale_dtype != expected_dtype:
                    raise ValueError(f"{name}: scale_dtype metadata disagrees with {kind} scale storage")
        if spec is not None:
            declared = spec.get("weights") or {}
            if (declared.get("dynamic", False) is not False or declared.get("symmetric", True) is not True
                    or declared.get("actorder") is not None or declared.get("zp_dtype") is not None
                    or spec.get("pre_quant_scale") or spec.get("has_zero_point") or spec.get("lora_rank")):
                raise ValueError(f"{name}: unsupported transformed/asymmetric weight quantization metadata")
        if kind == "nvfp4":
            if k % 16 or tensors.get("weight_scale") != ("F8_E4M3", [n, k // 16]):
                raise ValueError(f"{name}: NVFP4 block-scale shape/dtype disagrees with weight storage")
            if "weight_scale_2" in tensors and "weight_global_scale" in tensors:
                raise ValueError(f"{name}: conflicting weight global-scale conventions")
            self._scalar(name, tensors, "weight_global_scale" if self.reciprocal else "weight_scale_2")
        elif kind == "fp8":
            self._scalar(name, tensors, "weight_scale")
        activation = None if spec is None else spec.get("input_activations")
        if checkpoint and spec is not None and spec.get("output_activations") is not None:
            raise ValueError(f"{name}: checkpoint output quantization is unsupported; use --precision full")
        if not checkpoint or activation is None:
            return kind, None
        bits = activation.get("num_bits")
        dynamic = activation.get("dynamic", False)
        valid = (activation.get("type", "float") == "float" and
                 (kind == "nvfp4" and bits == 4 and activation.get("group_size", 16) == 16
                  and dynamic in (False, "local")
                  and activation.get("strategy") in (None, "group", "tensor_group")
                  and activation.get("scale_dtype") in (None, "torch.float8_e4m3fn")
                  or kind == "fp8" and bits == 8 and dynamic is False
                  and activation.get("strategy", "tensor") in ("tensor", None)))
        if not valid or activation.get("symmetric", True) is not True or activation.get("zp_dtype") is not None:
            raise ValueError(f"{name}: unsupported checkpoint activation quantization; use --precision full")
        scales = [key for key in ("input_scale", "input_global_scale") if key in tensors]
        if len(scales) != 1:
            raise ValueError(f"{name}: checkpoint activation math requires one static input calibration")
        self._scalar(name, tensors, scales[0])
        return kind, bits

    @staticmethod
    def _scalar(name, tensors, field):
        import math

        value = tensors.get(field)
        if (value is None or value[0] not in ("F32", "F16", "BF16")
                or not isinstance(value[1], (list, tuple))
                or any(type(n) is not int or n <= 0 for n in value[1]) or math.prod(value[1]) != 1):
            raise ValueError(f"{name}: {field} must be one floating-point calibration value")


def calibration(value, *, name, reciprocal=False):
    """A finite positive FP32 scalar, taking a reciprocal only when requested."""
    import math

    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name}: calibration must be finite and positive")
    scale = 1.0 / value if reciprocal else value
    import struct

    try:
        narrowed = struct.unpack("f", struct.pack("f", scale))[0]
    except OverflowError:
        narrowed = float("inf")
    if not math.isfinite(narrowed) or narrowed <= 0:
        raise ValueError(f"{name}: calibration must remain finite and positive in FP32")
    return scale


def e4m3(bits: np.ndarray) -> np.ndarray:
    """e4m3fn bytes as float32 (no infinities; 0x7f and 0xff are NaN)."""

    b = bits.astype(np.int32)
    sign = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
    exp, man = (b >> 3) & 0xF, b & 0x7
    value = np.where(exp == 0, man / 8.0 * 2.0 ** -6, (1 + man / 8.0) * 2.0 ** (exp - 7.0)).astype(np.float32)
    return np.where((b & 0x7F) == 0x7F, np.float32(np.nan), sign * value)


def e8m0(bits: np.ndarray) -> np.ndarray:
    """e8m0 exponent bytes as float32 powers of two (255 is NaN)."""

    b = bits.astype(np.int32)
    return np.where(b == 255, np.float32(np.nan), np.ldexp(np.float32(1.0), b - 127).astype(np.float32))


def dequant(scheme_name: str, weight: np.ndarray, scale: np.ndarray | None = None, global_scale: float = 1.0,
            *, reciprocal_global: bool = False) -> np.ndarray:
    """[N, K] float32 of one stored weight: e2m1 pairs with e4m3 per 16 x global, e4m3 x scale, or e4m3 x e8m0 per 32."""

    if scheme_name == "nvfp4":
        n, half = weight.shape
        codes = np.empty((n, 2 * half), dtype=np.uint8)
        codes[:, 0::2], codes[:, 1::2] = weight & 0xF, weight >> 4
        g = 1.0 / global_scale if reciprocal_global else global_scale
        return E2M1[codes] * np.repeat(e4m3(scale), 16, axis=1) * np.float32(g)
    if scheme_name == "fp8":
        return e4m3(weight) * np.float32(np.asarray(scale, dtype=np.float32).reshape(-1)[0])
    if scheme_name == "mxfp8":
        return e4m3(weight) * np.repeat(e8m0(scale), 32, axis=1)
    if scheme_name == "fp8block":              # ``scale`` = weight_scale_inv [ceil(N/128), K/128]
        n, k = weight.shape
        s = np.asarray(scale, dtype=np.float32)
        return e4m3(weight) * np.repeat(np.repeat(s, 128, axis=0)[:n], k // s.shape[1], axis=1)
    return weight.astype(np.float32)


def native_calibration(act: float, weight_scale: float, *, name: str) -> float:
    """Admit the exact FP32 inverse/product used by checkpoint native math.

    Called only for an actually selected native projection. Reuse the owned
    arithmetic helpers so validation and subsequent native preparation share
    their rounding and subnormal contract; return the original scale unchanged.
    """
    import math

    from .checkpoint import _inv, alpha

    inverse, product = _inv(act), alpha(act, weight_scale)
    if not (math.isfinite(inverse) and inverse > 0 and math.isfinite(product) and product > 0):
        raise ValueError(f"{name}: native FP32 inverse and joint calibration must remain finite and positive")
    return act
