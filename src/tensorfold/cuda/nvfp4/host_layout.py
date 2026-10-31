"""Header-only NVFP4 RAM-expert validation; no CUDA or torch imports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import prod


def routed_tensor(name):
    return ".mlp.experts." in name


def bank_bases(infos, text, mtp=True):
    """Validate every routed projection before allocating authority or a pool."""
    bases = {n.split(".experts.", 1)[0] for n in infos if routed_tensor(n)
             and (mtp or not n.startswith("mtp."))}
    main = {int(m.group(1)) for b in bases if (m := re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", b))
            and not b.startswith("mtp.")}
    layers = text["num_hidden_layers"]
    if len(main) != layers or min(main, default=-1) != 0 or max(main, default=-1) != layers - 1:
        raise ValueError("--vram-experts: NVFP4 checkpoint must contain every configured main expert layer")
    e, width, dims = text["num_experts"], text["moe_intermediate_size"], text["hidden_size"]

    def expect(name, shape, dtypes):
        info = infos.get(name)
        if info is None or list(info["shape"]) != list(shape) or info["dtype"] not in dtypes:
            raise ValueError(f"--vram-experts: invalid NVFP4 expert field {name}; expected {shape} {dtypes}")
        return info

    for base in bases:
        draft = base.startswith("mtp.")
        if not (re.fullmatch(r"(?:model(?:\.language_model)?|language_model\.model)\.layers\.\d+\.mlp", base)
                or base == "mtp.layers.0.mlp"):
            raise ValueError(f"--vram-experts: unrecognized NVFP4 expert layer {base}")
        if base + ".experts.gate_up_proj" in infos:
            if not draft:
                raise ValueError("--vram-experts: only MTP experts may be stacked BF16")
            expect(base + ".experts.gate_up_proj", (e, 2 * width, dims), ("BF16", "F16", "F32"))
            expect(base + ".experts.down_proj", (e, dims, width), ("BF16", "F16", "F32"))
            continue
        for expert in range(e):
            for p, n, k in (("gate_proj", width, dims), ("up_proj", width, dims), ("down_proj", dims, width)):
                b = f"{base}.experts.{expert}.{p}"
                info = infos.get(b + ".weight", {})
                if info.get("dtype") == "F8_E4M3" and draft:
                    expect(b + ".weight", (n, k), ("F8_E4M3",))
                    if b + ".weight_scale_inv" in infos:
                        expect(b + ".weight_scale_inv", (-(-n // 128), -(-k // 128)), ("BF16", "F16", "F32"))
                    else:
                        scale = infos.get(b + ".weight_scale", {})
                        if scale.get("dtype") not in ("BF16", "F16", "F32") or list(scale.get("shape", ())) not in (
                                [], [1], [n], [n, 1]):
                            raise ValueError(f"--vram-experts: invalid FP8 MTP expert scale {b}")
                else:
                    expect(b + ".weight", (n, k // 2), ("U8",))
                    expect(b + ".weight_scale", (n, k // 16), ("F8_E4M3",))
                    scale = infos.get(b + ".weight_scale_2", {})
                    if scale.get("dtype") not in ("BF16", "F16", "F32") or list(scale.get("shape", ())) not in ([], [1]):
                        raise ValueError(f"--vram-experts: invalid NVFP4 expert global scale {b}")
    return tuple(sorted(bases))


_A4_EXPERT = re.compile(
    r"^(?P<bank>(?:model(?:\.language_model)?|language_model\.model)\.layers\.(?P<layer>\d+)\.mlp)"
    r"\.experts\.(?P<expert>\d+)\.(?P<projection>[^.]+)\.(?P<field>[^.]+)$")
_A4_FIELDS = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
_A4_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def authority_fields_a4(count, width, dims):
    """Six logical native-A4 fields, separate from Flash Next's W4A16 blocks.

    A projection blob holds checkpoint FP4 code words, then its tiled E4M3
    block scales. Weight global fields hold prepared FP32 reciprocals; input
    fields retain original CT FP32 multipliers exactly. Native alpha separately
    uses FP32(1/input_multiplier) times the prepared weight reciprocal.
    """
    if type(count) is not int or not 1 <= count <= 1023:
        raise ValueError("native A4 expert count must lie in [1, 1023]")
    if any(type(n) is not int or not 0 < n < 2**31 or n % 64 for n in (width, dims)):
        raise ValueError("native A4 expert dimensions must be positive int32 multiples of 64")
    blob = 9 * width * dims // 16
    fields = (((count, 2, blob), "U8"), ((count, blob), "U8"),
              ((count, 2), "F32"), ((count, 1), "F32"), ((count, 2), "F32"), ((count, 1), "F32"))
    if count * 2 * blob > 2**63 - 1:
        raise ValueError("native A4 authority exceeds tensor storage size limits")
    return fields


def plan_scratch_bytes_a4(pairs, width, dims):
    """Retained bytes of ExpertsA4.run_plan's per-buffer scratch for ``pairs`` planned pairs: the gate/up activation
    FP4 codes and E4M3 block scales of two projections, then the down projection's. The fused gate/up kernel keeps
    its BF16 gate/up values and activation in registers; a shared-input row plan is the caller's to add."""
    return pairs * (dims + dims // 8 + width // 2 + width // 16)


def routed_tensor_a4(name):
    """Main routed CT projection fields only; shared and secondary drafts stay separate."""
    match = _A4_EXPERT.fullmatch(name)
    return bool(match and match["projection"] in _A4_PROJECTIONS and match["field"] in _A4_FIELDS)


def bank_bases_a4(infos, text):
    """Validate complete main-bank CT NVFP4 headers before any authority allocation."""
    layers, count = text["num_hidden_layers"], text["num_experts"]
    width, dims = text["moe_intermediate_size"], text["hidden_size"]
    authority_fields_a4(count, width, dims)
    if type(layers) is not int or layers <= 0:
        raise ValueError("native A4 layer count must be a positive integer")
    banks = {}
    layer_ids = set()
    for name, info in infos.items():
        match = _A4_EXPERT.fullmatch(name)
        if match is None:
            continue
        if match["projection"] not in _A4_PROJECTIONS or match["field"] not in _A4_FIELDS:
            raise ValueError(f"native A4 expert field is unsupported: {name}")
        expert = int(match["expert"])
        if not 0 <= expert < count:
            raise ValueError(f"native A4 expert index is outside configured bank: {name}")
        layer_ids.add(int(match["layer"]))
        bank = banks.setdefault(match["bank"], {})
        key = (expert, match["projection"], match["field"])
        if key in bank:
            raise ValueError(f"duplicate native A4 expert field: {name}")
        bank[key] = info
    if len(layer_ids) != layers or min(layer_ids, default=-1) != 0 or max(layer_ids, default=-1) != layers - 1:
        raise ValueError("native A4 checkpoint must contain every configured main expert layer")
    if len(banks) != layers:
        raise ValueError("native A4 checkpoint has conflicting main expert bank prefixes")
    for name, bank in banks.items():
        if len(bank) != count * 12:
            raise ValueError(f"{name}: native A4 expert bank is incomplete")
        for (expert, projection, field), info in bank.items():
            n, k = (dims, width) if projection == "down_proj" else (width, dims)
            dtype, shape = (("U8", [n, k // 2]) if field == "weight_packed" else
                            ("F8_E4M3", [n, k // 16]) if field == "weight_scale" else ("F32", [1]))
            if info.get("dtype") != dtype or list(info.get("shape", ())) != shape:
                raise ValueError(f"{name}.experts.{expert}.{projection}.{field}: invalid native A4 storage")
    return tuple(sorted(banks))


@dataclass(frozen=True)
class A4Field:
    shape: tuple[int, ...]
    dtype: str
    offset: int
    size: int


@dataclass(frozen=True)
class A4EntryLayout:
    entry_bytes: int
    fields: tuple[A4Field, ...]
    padding: tuple[tuple[int, int], ...]


def entry_layout_a4(width, dims, *, count=1):
    """Canonical 16-byte entry layout for native A4 host authority and hot cells."""
    fields, padding, cursor = [], [], 0
    for shape, dtype in authority_fields_a4(count, width, dims):
        offset = -(-cursor // 16) * 16
        if offset != cursor:
            padding.append((cursor, offset))
        size = prod(shape[1:]) * (1 if dtype == "U8" else 4)
        fields.append(A4Field(shape[1:], dtype, offset, size))
        cursor = offset + size
    entry = -(-cursor // 16) * 16
    if entry != cursor:
        padding.append((cursor, entry))
    if count * entry > 2**63 - 1:
        raise ValueError("native A4 padded authority exceeds tensor storage size limits")
    return A4EntryLayout(entry, tuple(fields), tuple(padding))


def authority_fields_modelopt(count, width, dims, *, native_a4):
    """Original Flash W4A16 fields or native A4's six direct-calibration fields."""
    if native_a4:
        return authority_fields_a4(count, width, dims)
    if type(count) is not int or not 1 <= count <= 1023:
        raise ValueError("ModelOpt expert count must lie in [1, 1023]")
    if any(type(n) is not int or not 0 < n < 2**31 or n % 32 for n in (width, dims)):
        raise ValueError("ModelOpt expert dimensions must be positive int32 multiples of32")
    if count * width * dims * 9 // 8 > 2**63 - 1:
        raise ValueError("ModelOpt authority exceeds tensor storage size limits")
    return (((count, width // 32, dims // 32, 2, 144), "I32"),
            ((count, dims // 32, width // 32, 1, 144), "I32"),
            ((count, 2), "F32"), ((count, 1), "F32"))
