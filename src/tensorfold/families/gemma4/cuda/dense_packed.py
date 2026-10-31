"""Packed-only projection boundary for a future dense Gemma loader.

Native HF text attention/MLP callers invoke projection modules; they do not read
projection `.weight`. Embeddings, tied heads, PLE and media remain native BF16.
Install only after meta-model plain tensors have been placed. Packed tensors
and their prepared operator are immutable, borrowed by one exclusive inference
lane and retired by its model owner AFTER all consumers drain. There is no BF16
weight facade, whole-matrix dequantization, calibration or target integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from tensorfold.cuda.nvfp4.format import LinearMetadata, calibration, native_calibration
from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear, Fp8BlockLinear

DTYPES = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32,
          "F16": torch.float16, "BF16": torch.bfloat16}


@dataclass(frozen=True)
class PackedProjectionSpec:
    """CPU storage/compute declaration; owns no payload or producer state."""
    name: str
    shape: tuple[int, int]
    kind: str
    activation_bits: int | None
    reciprocal_globals: bool
    parts: tuple[tuple[str, str, tuple[int, ...]], ...]

    @classmethod
    def from_header(cls, name: str, parts: Mapping, metadata: LinearMetadata, *, expected_shape):
        if (not isinstance(name, str) or not name or not isinstance(metadata, LinearMetadata)
                or not isinstance(expected_shape, (tuple, list)) or len(expected_shape) != 2
                or any(type(value) is not int or value < 1 for value in expected_shape)):
            raise ValueError("packed projection requires its module name, metadata and positive expected[N,K]")
        normalized = {}
        for suffix, value in parts.items():
            if (not isinstance(suffix, str) or not isinstance(value, (tuple, list)) or len(value) != 2
                    or value[0] not in DTYPES or not isinstance(value[1], (tuple, list))
                    or any(type(size) is not int or size < 1 for size in value[1])):
                raise ValueError(f"{name}: malformed packed tensor header")
            normalized[suffix] = value[0], list(value[1])
        kind, bits = metadata.validate(name, normalized, checkpoint=True)
        if kind not in ("nvfp4", "fp8", "fp8channel", "fp8block"):
            raise ValueError(f"{name}: unsupported dense packed representation {kind}")
        if metadata.reciprocal and kind != "nvfp4":
            raise ValueError(f"{name}: compressed-tensors FP8 conventions are not admitted by this adapter")
        weight = "weight_packed" if "weight_packed" in normalized else "weight"
        n, k = normalized[weight][1]
        if kind == "nvfp4":
            k *= 2
        if (n, k) != tuple(expected_shape) or k % 64:
            raise ValueError(f"{name}: packed shape differs from native[N,K] or K/64 contract")
        allowed = {weight}
        if kind == "nvfp4":
            allowed |= {"weight_scale", "weight_global_scale" if metadata.reciprocal else "weight_scale_2"}
        elif kind in ("fp8", "fp8channel"):
            allowed.add("weight_scale")
        else:
            if bits is not None:
                raise ValueError(f"{name}: FP8 block projection requires BF16 activation math")
            allowed.add("weight_scale_inv")
            if k % 128 or normalized["weight_scale_inv"] != ("F32", [(n+127)//128, k//128]):
                raise ValueError(f"{name}: FP8 block scales require original128x128 geometry")
            declared = metadata.policy.resolve(name) or {}
            if (declared.get("weights") or {}).get("block_structure") != [128, 128]:
                raise ValueError(f"{name}: FP8 block metadata must explicitly declare128x128")
        if bits is not None and kind != "fp8channel":
            allowed.add("input_global_scale" if metadata.reciprocal else "input_scale")
        if set(normalized) != allowed:
            raise ValueError(f"{name}: packed tensor inventory differs from declared compute/storage")
        return cls(name, (n, k), kind, bits, metadata.reciprocal,
                   tuple((key, dtype, tuple(shape)) for key, (dtype, shape) in sorted(normalized.items())))


class PackedProjection(nn.Module):
    """Actual packed buffers plus existing native operator, inference-only.

    `.device`, `.compute_dtype`, dimensions and `.bias=None` are real metadata;
    `.weight` intentionally does not exist. No original BF16 Parameter is owned.
    Moving/loading a prepared module would desynchronize operator aliases, so
    placement and storage replacement require construction of a fresh owner.
    """
    def __init__(self, spec: PackedProjectionSpec, operation, prompt_policy):
        storage = tensor = None
        try:
            super().__init__()
            if not isinstance(spec, PackedProjectionSpec):
                raise ValueError("prepared projection requires validated storage metadata")
            expected = {"nvfp4": Fp4Linear, "fp8": Fp8Linear,
                        "fp8channel": Fp8Linear, "fp8block": Fp8BlockLinear}[spec.kind]
            if type(operation) is not expected or (operation.n, operation.k) != spec.shape:
                raise ValueError("prepared projection operator differs from declared storage/shape")
            if spec.kind == "nvfp4":
                if (operation.act is None) != (spec.activation_bits is None):
                    raise ValueError("NVFP4 operator activation math differs from metadata")
                calibration(operation.scale, name=spec.name+".prepared_weight_global")
                if (operation.act_inverse is not None) != (spec.reciprocal_globals and spec.activation_bits is not None):
                    raise ValueError("NVFP4 prepared inverse convention differs from metadata")
                if operation.act is not None:
                    native_calibration(operation.act, operation.scale, name=spec.name,
                                       act_inverse=operation.act_inverse)
            elif spec.kind in ("fp8", "fp8channel"):
                calibration(operation.scale, name=spec.name+".prepared_weight_global")
                if operation.act is not None:
                    calibration(operation.act, name=spec.name+".prepared_input_global")
                if ((operation.cols is not None) != (spec.kind == "fp8channel")
                        or (operation.act is not None or operation.token) != (spec.activation_bits is not None)
                        or operation.token != (spec.kind == "fp8channel" and spec.activation_bits is not None)):
                    raise ValueError("FP8 operator activation/channel math differs from metadata")
            if type(prompt_policy.prompt) is not bool:
                raise ValueError("packed projection needs an explicit boolean prompt policy")
            self.spec, self._prompt_policy = spec, prompt_policy
            self.in_features, self.out_features = spec.shape[1], spec.shape[0]
            self.bias = None
            fields = ("words", "bs") if spec.kind == "nvfp4" else ("w8", "cols", "groups") if spec.kind != "fp8block" else ("w8", "bs", "groups")
            storage = [(name, getattr(operation, name)) for name in fields if getattr(operation, name) is not None]
            devices = {tensor.device for _, tensor in storage}
            if len(devices) != 1 or next(iter(devices)).type != "cuda":
                raise ValueError("packed projection storage must own one CUDA device")
            for name, tensor in storage:
                if not tensor.is_contiguous():
                    raise ValueError("prepared projection storage must be contiguous")
                self.register_buffer(name, tensor)
            self._operation = operation
            self.train(False)
        except BaseException:
            if hasattr(self, "_buffers"):
                self._buffers.clear()
            self._operation = None
            raise
        finally:
            operation = storage = tensor = None

    @property
    def device(self):
        return next(iter(self._buffers.values())).device

    @property
    def compute_dtype(self):
        return torch.bfloat16

    def _apply(self, function, recurse=True):
        raise RuntimeError("packed projection placement is immutable; build a fresh prepared owner")

    def _load_from_state_dict(self, *args, **kwargs):
        raise RuntimeError("packed projection storage is immutable; reload its validated checkpoint")

    @classmethod
    def from_checkpoint(cls, spec: PackedProjectionSpec, parts: Mapping[str, torch.Tensor], prompt_policy):
        """Prepare existing kernels from admitted tensors, without requantizing.

        Packed matrix/block scales are on one CUDA device. Global calibrations
        are CPU scalars, validated once while loading; no per-forward readback.
        Caller owns parts and loading staging; this module retains only prepared
        operator storage, not the original row-major source or payload mapping.
        """
        operation = weight = scales = value = None
        try:
            if not isinstance(spec, PackedProjectionSpec) or set(parts) != {part[0] for part in spec.parts}:
                raise ValueError("packed projection payload inventory differs")
            for suffix, dtype, shape in spec.parts:
                value = parts[suffix]
                if not isinstance(value, torch.Tensor) or value.dtype != DTYPES[dtype] or tuple(value.shape) != shape:
                    raise ValueError(f"{spec.name}.{suffix}: payload dtype/shape differs")
            weight = parts.get("weight_packed", parts.get("weight"))
            if weight.device.type != "cuda":
                raise ValueError("packed projection weight must be on its selected CUDA device")
            scales = parts.get("weight_scale_inv", parts.get("weight_scale"))
            if scales.numel() > 1 and scales.device != weight.device:
                raise ValueError("packed projection block/channel scales must share its CUDA device")

            def scalar(key, reciprocal=False):
                value = parts[key]
                if value.device.type != "cpu" or value.numel() != 1:
                    raise ValueError(f"{spec.name}.{key}: calibration must be a CPU scalar")
                return calibration(value.item(), name=f"{spec.name}.{key}", reciprocal=reciprocal)

            if spec.kind == "nvfp4":
                if not bool((torch.isfinite(scales.float()) & (scales.float() >= 0)).all()):
                    raise ValueError(f"{spec.name}: NVFP4 scales must be finite nonnegative")
                global_name = "weight_global_scale" if spec.reciprocal_globals else "weight_scale_2"
                scale = scalar(global_name, spec.reciprocal_globals)
                act = inverse = None
                if spec.activation_bits is not None:
                    input_name = "input_global_scale" if spec.reciprocal_globals else "input_scale"
                    act = scalar(input_name, spec.reciprocal_globals)
                    inverse = scalar(input_name) if spec.reciprocal_globals else None
                    native_calibration(act, scale, name=spec.name, act_inverse=inverse)
                operation = Fp4Linear.from_checkpoint(weight, scales, scale, act=act, act_inverse=inverse)
            else:
                if not bool(torch.isfinite(weight).all()):
                    raise ValueError(f"{spec.name}: FP8 codes must be finite")
                if not bool((torch.isfinite(scales) & (scales > 0)).all()):
                    raise ValueError(f"{spec.name}: FP8 scales must be finite positive")
                if spec.kind == "fp8block":
                    operation = Fp8BlockLinear.from_checkpoint(weight, scales)
                elif spec.kind == "fp8channel":
                    operation = Fp8Linear.from_checkpoint(weight, scales, token=spec.activation_bits is not None)
                else:
                    act = scalar("input_global_scale" if spec.reciprocal_globals else "input_scale",
                                 spec.reciprocal_globals) if spec.activation_bits is not None else None
                    operation = Fp8Linear.from_checkpoint(weight, scalar("weight_scale"), act=act)
            return cls(spec, operation, prompt_policy)
        finally:
            operation = parts = weight = scales = value = None

    def forward(self, value):
        if torch.is_grad_enabled():
            raise RuntimeError("packed projections are inference-only")
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.in_features
                or value.dtype != self.compute_dtype or value.device != self.device):
            raise ValueError("packed projection requires BF16 activations with its input width/device")
        shape = (*value.shape[:-1], self.out_features)
        if value.numel() == 0:
            return value.new_empty(shape)
        rows = value.reshape(-1, self.in_features).contiguous()
        # FP8BlockLinear.prefill follows a separate process precision flag.
        # Preserve this descriptor's W8A16 contract by using its exact lane op.
        output = (self._operation.prefill(rows) if self._prompt_policy.prompt and self.spec.kind != "fp8block"
                  else self._operation(rows))
        return output.reshape(shape)
