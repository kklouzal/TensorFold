"""Bounded native SM70 unique-page attention, selected only by a qualified profile.

All four phases borrow caller-owned contiguous buffers. The owner constructs
the private descriptor table and retains every pointed-to tensor through stream
completion and graph retirement. ``layer`` is a borrowed CUDA int32[1] control,
updated on the producing stream; graph nodes never capture its Python value.
Compilation is not numerical qualification;
the Torch owner implementation remains the independent execution reference.
"""

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import struct

from .paged_attention_descriptor import ABI_VERSION, BYTES, SCHEMA_ID, WORDS


KERNEL_FAMILY = "sm70-paged-owner-q8-compact-v1"
RN_KERNEL_FAMILY = "sm70-paged-owner-rn-ex2-p2051-v1"
RN_GRAPH_FAMILY = "owner-attention-rn-five-phase-v1"
RN_MAX_SELECTED = 2051
RN_LOG2E_BITS = 0x3fb8aa3b
RN_ALGORITHM = "qk-fma8-shuffle5/ordered32-ml/ex2-nonftz/half-global-p/pv-stride32-shuffle5"
RN_CUDA_FLAGS = ("-O3", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
DESCRIPTOR_WORDS = WORDS
DESCRIPTOR_BYTES = BYTES
LAYER_BYTES = 4
HEADS = 12
DIM = 256


def workspace_bytes(rows):
    """Requested table/global-normalization bytes; communication slabs are reused."""

    if type(rows) is not int or rows < 0:
        raise ValueError("paged attention rows must be a nonnegative integer")
    return rows * (DESCRIPTOR_BYTES + HEADS * 2 * 4)


def generation_bytes(slots):
    if type(slots) is not int or slots < 0:
        raise ValueError("history generation slots must be a nonnegative integer")
    return slots * 2 * 8


def _format(bits, codec):
    if type(bits) is not int or type(codec) is not int or (bits, codec) not in ((8, 0), (4, 9), (4, 10)):
        raise ValueError("paged attention requires ordinary Q8, pinned Iso4 codec10 or Planar4 codec9")


def _controls(group_start, page_tokens):
    if type(group_start) is not int or group_start not in (0, 4):
        raise ValueError("a KV-head group starts at rank zero or four")
    if type(page_tokens) is not int or not 0 < page_tokens <= 2**31 - 1 or page_tokens % 4:
        raise ValueError("history pages require positive int32 complete pooling groups")


def _scale(scale):
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("attention scale must be positive and finite")
    try:
        value = struct.unpack("f", struct.pack("f", float(scale)))[0]
    except (OverflowError, ValueError):
        raise ValueError("attention scale must be positive and finite FP32") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("attention scale must be positive and finite FP32")
    return value


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .rotorquant_q4_tables import verify
    from .source_cache import stage_sources

    verify()
    prepared = _native()
    root = Path(__file__).parent
    sources = [root / "paged_attention_bindings.cpp", root / "paged_attention.cu"]
    headers = [root / name for name in ("paged_attention_descriptor.cuh", "packed_kv_decode.cuh",
                                       "rotorquant_q4.cuh", "rotorquant_q4_tables.cuh")]
    digest = hashlib.sha256(prepared.__name__.encode())
    for source in [*sources, *headers]:
        digest.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=headers, source_root=root.parent)
    module = build.load("tf_sm70_paged_attention_" + digest.hexdigest()[:20], staged, need=(7, 0),
                        extra_cuda_cflags=list(RN_CUDA_FLAGS),
                        extra_cflags=["-O3"])
    if (getattr(module, "descriptor_abi", None), getattr(module, "descriptor_words", None),
            getattr(module, "descriptor_schema_id", None)) != (ABI_VERSION, WORDS, SCHEMA_ID):
        raise RuntimeError("native owner attention descriptor ABI differs from its caller schema")
    from .attention_backends import OWNER_FAMILY, OWNER_MAX_KEYS
    if (getattr(module, 'aten_owner_family', None), getattr(module, 'aten_owner_max_keys', None)) != (
            OWNER_FAMILY, OWNER_MAX_KEYS):
        raise RuntimeError('ATen owner tile native boundary differs from its declared family')
    if (getattr(module, "rn_family", None), getattr(module, "rn_graph_family", None),
            getattr(module, "rn_max_selected", None), getattr(module, "rn_log2e_bits", None),
            getattr(module, "rn_algorithm", None)) != (
                RN_KERNEL_FAMILY, RN_GRAPH_FAMILY, RN_MAX_SELECTED, RN_LOG2E_BITS, RN_ALGORITHM):
        raise RuntimeError("native RN numerical-family/constant ABI differs from its caller")
    return module


def local_statistics(queries, selections, row_table, out, *, group_start, page_tokens, layer,
                     key_bits, key_codec, scale):
    _controls(group_start, page_tokens)
    _format(key_bits, key_codec)
    scale = _scale(scale)
    extension().local_statistics(queries, selections, row_table, out, group_start, page_tokens,
                                 layer, key_bits, key_codec, scale)
    return out


def merge_statistics(gathered, row_table, out, *, group_start, page_tokens, layer):
    _controls(group_start, page_tokens)
    extension().merge_statistics(gathered, row_table, out, group_start, page_tokens, layer)
    return out


def normalized_partial(queries, selections, row_table, global_statistics, out, *, group_start,
                       page_tokens, layer, key_bits, key_codec, value_bits, value_codec, scale):
    _controls(group_start, page_tokens)
    _format(key_bits, key_codec)
    _format(value_bits, value_codec)
    if (key_bits, key_codec) != (value_bits, value_codec):
        raise ValueError("paged attention requires one homogeneous Q8 or compact Q4 key/value profile")
    scale = _scale(scale)
    extension().normalized_partial(queries, selections, row_table, global_statistics, out,
                                   group_start, page_tokens, layer, key_bits, key_codec,
                                   value_bits, value_codec, scale)
    return out


def merge_values(gathered, row_table, out, *, rank, page_tokens, layer, value_bits, value_codec):
    if type(rank) is not int or rank not in range(8):
        raise ValueError("a local query rank must lie in0..7")
    _controls(rank // 4 * 4, page_tokens)
    _format(value_bits, value_codec)
    extension().merge_values(gathered, row_table, out, rank, page_tokens, layer, value_bits, value_codec)
    return out


def rn_probability_bytes(rows):
    if type(rows) is not int or not 0 <= rows <= 2**31 - 1:
        raise ValueError("RN Half-P rows must be a bounded nonnegative integer")
    return rows * HEADS * RN_MAX_SELECTED * 2


def _rn_heads(heads):
    if type(heads) is not int or heads not in (1, 4, 12):
        raise ValueError("RN heads per block must be explicitly1,4,12")


def rn_local_statistics(queries, selections, row_table, out, *, group_start, page_tokens, layer,
                        key_bits, key_codec, heads_per_block):
    _controls(group_start, page_tokens)
    _format(key_bits, key_codec)
    _rn_heads(heads_per_block)
    extension().rn_local_statistics(queries, selections, row_table, out, group_start, page_tokens,
                                     layer, key_bits, key_codec, heads_per_block)
    return out


def rn_merge_statistics(gathered, row_table, out, *, group_start, page_tokens, layer):
    _controls(group_start, page_tokens)
    extension().rn_merge_statistics(gathered, row_table, out, group_start, page_tokens, layer)
    return out


def rn_probabilities(queries, selections, row_table, global_statistics, out, *, group_start,
                     page_tokens, layer, key_bits, key_codec, heads_per_block):
    _controls(group_start, page_tokens)
    _format(key_bits, key_codec)
    _rn_heads(heads_per_block)
    extension().rn_probabilities(queries, selections, row_table, global_statistics, out, group_start,
                                 page_tokens, layer, key_bits, key_codec, heads_per_block)
    return out


def rn_partial_values(probability, selections, row_table, out, *, page_tokens, layer, value_bits, value_codec):
    _controls(0, page_tokens)
    _format(value_bits, value_codec)
    extension().rn_partial_values(probability, selections, row_table, out, page_tokens, layer, value_bits, value_codec)
    return out


def rn_preparation_identity():
    """Startup source policy, not proof of actual PTX/SASS or quality bounds."""
    root = Path(__file__).parent
    sources = (root / "paged_attention.py", root / "paged_attention_bindings.cpp", root / "paged_attention.cu")
    return {"kernel_family": RN_KERNEL_FAMILY, "graph_family": RN_GRAPH_FAMILY,
            "algorithm": RN_ALGORITHM, "log2e_bits": RN_LOG2E_BITS, "max_selected": RN_MAX_SELECTED,
            "max_dense": RN_MAX_SELECTED, "qk_heads_per_block_choices": [1, 4, 12], "pv_columns_per_block": 64,
            "pv_warps_per_block": 4, "pv_accumulators_per_thread": 16,
            "requested_cuda_flags": list(RN_CUDA_FLAGS),
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            "qualification": "experimental numerical family; bounds/binary/quality/resource/P3 unqualified"}


def pack_owner_queries(queries, row_table, out, *, group_start, page_tokens, layer):
    _controls(group_start, page_tokens)
    extension().pack_owner_queries(queries, row_table, out, group_start, page_tokens, layer)
    _publish_owner_writes(out)
    return out


def gather_owner_q8(selections, row_table, keys, values, valid, *, first, group_start, page_tokens, layer):
    _controls(group_start, page_tokens)
    extension().gather_owner_q8(selections, row_table, keys, values, valid, first, group_start, page_tokens, layer)
    _publish_owner_writes(keys, values, valid)
    return keys, values, valid
def weight_owner_normalized(normalized, local_lse, global_statistics, row_table, out, *,
                            group_start, page_tokens, layer):
    _controls(group_start, page_tokens)
    extension().weight_owner_normalized(normalized, local_lse, global_statistics, row_table, out,
                                        group_start, page_tokens, layer)
    _publish_owner_writes(out)
    return out


def _publish_owner_writes(*outputs):
    import torch
    for value in outputs:
        if not value.is_inference():
            torch.autograd.graph.increment_version(value)
