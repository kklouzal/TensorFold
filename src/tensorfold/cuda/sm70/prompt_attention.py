"""Tensor-core (mma.m8n8k4) local phases for SM70 unique-history prompt attention.

Selected by ``history_owner`` for prefill owners of the RN family with ordinary
Q8 (bits 8, codec 0) caches unless ``TENSORFOLD_PRIVATE_SM70_PROMPT_ATTENTION=0``.
It replaces the per-row RN local statistics/probability/value phases with one
CTA per query row that shares every compacted local K/V tile across the twelve
GQA heads of the KV head. Descriptor/selection/status semantics (shared device
helpers in ``paged_attention_device.cuh``) and the statistics/value collective
formats are those of ``paged_attention``:

* ``fused_local``   -> (m, l) [rows,12,2] FP32 plus the UNNORMALIZED local sum
  exp(s - m) v [rows,12,256] FP32 in one pass.
* ``weight_values`` -> after the ordinary statistics gather and RN merge, the
  globally normalized [rows,12,256] local partial for the value gather.

Arithmetic: exact Q8 codes (c + 0.5) and FP16 queries in the tensor core, FP32
group scales applied after each 32-dim group dot (K) or folded into FP16 P with
an exact power-of-two range (V); RN ex2 exponentials. The caller retains every
descriptor-addressed tensor through stream completion, as for the RN family.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

from .paged_attention_descriptor import ABI_VERSION, SCHEMA_ID, WORDS

FAMILY = "sm70-paged-owner-tc884-q8-exact-code-v1"
CUDA_FLAGS = ("-O3", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources

    prepared = _native()
    root = Path(__file__).parent
    sources = [root / "prompt_attention_bindings.cpp", root / "prompt_attention.cu"]
    headers = [root / "paged_attention_descriptor.cuh", root / "paged_attention_device.cuh"]
    digest = hashlib.sha256(prepared.__name__.encode())
    for source in [*sources, *headers]:
        digest.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=headers, source_root=root.parent)
    module = build.load("tf_sm70_prompt_attention_" + digest.hexdigest()[:20], staged, need=(7, 0),
                        extra_cuda_cflags=list(CUDA_FLAGS), extra_cflags=["-O3"])
    if (getattr(module, "descriptor_abi", None), getattr(module, "descriptor_words", None),
            getattr(module, "descriptor_schema_id", None), getattr(module, "family", None)) != (
                ABI_VERSION, WORDS, SCHEMA_ID, FAMILY):
        raise RuntimeError("tensor-core prompt attention ABI/family differs from its caller")
    return module


def _controls(group_start, page_tokens, key_bits, key_codec):
    if type(group_start) is not int or group_start not in (0, 4):
        raise ValueError("a KV-head group starts at rank zero or four")
    if type(page_tokens) is not int or not 0 < page_tokens <= 2**31 - 1 or page_tokens % 4:
        raise ValueError("history pages require positive int32 complete pooling groups")
    if (key_bits, key_codec) != (8, 0) or type(key_bits) is not int or type(key_codec) is not int:
        raise ValueError("tensor-core prompt attention supports only ordinary Q8")


def fused_local(queries, selections, row_table, statistics, unnormalized, *, group_start, page_tokens,
                layer, key_bits=8, key_codec=0):
    _controls(group_start, page_tokens, key_bits, key_codec)
    extension().fused_local(queries, selections, row_table, statistics, unnormalized, group_start,
                            page_tokens, layer, key_bits, key_codec)
    return statistics, unnormalized


def weight_values(unnormalized, local_statistics, global_statistics, row_table, out, *, layer):
    """``local_statistics`` must be this rank's (m, l) as published by ``fused_local``."""
    extension().weight_values(unnormalized, local_statistics, global_statistics, row_table, out, layer)
    return out
