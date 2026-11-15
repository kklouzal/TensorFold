"""Native FP16 chain/replay for the existing Flash Next State/GDNScratch ABI."""

from functools import lru_cache
import hashlib
from pathlib import Path

SPLIT_MIN_ROWS = 64  # prompt-chunk sized GDN chains; see chain()


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources

    prepared = _native()  # Retain the selected SM70/toolkit/ABI admission and identity.
    root = Path(__file__).parent
    sources = [root / "gdn_bindings.cpp", root / "gdn.cu", root / "gdn_prefill_split.cu"]
    identity = hashlib.sha256(prepared.__name__.encode())
    headers = [root.parent / "kernels/gdn_recurrence.cuh", root.parent / "kernels/tensor_contracts.cuh", root / "gdn_common.cuh"]
    for source in [*sources, *headers]:
        identity.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=headers, source_root=root.parent)
    return build.load("tf_sm70_gdn_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cuda_cflags=["-O3", "--fmad=false"], extra_cflags=["-O3"])


def chain(p, conv, weight, initial, decay_parameter, bias, norm, eps, rows, scratch, final, output, sums,
          *, negative_decay=False):
    import torch

    # Prompt chunks run the multi-block form (gdn_prefill_split.cu: bitwise chain's results, ~8x faster per 1024-row
    # chunk on V100). Short (decode) chains keep the single-launch serial kernel: its three launches measured no
    # integrated gain there (C8 decode 114 vs 117-120 tok/s). The split form takes a caching-allocator workspace,
    # so graph captures keep the serial kernel too.
    native = extension()
    split = rows >= SPLIT_MIN_ROWS and not torch.cuda.is_current_stream_capturing()
    run = native.split_chain if split else native.chain
    return run(p, conv, weight, initial, decay_parameter, bias, norm, eps, rows, output, sums, final,
               scratch.k, scratch.v, scratch.g, scratch.b, getattr(scratch, "keep_rows", None), negative_decay)


def replay(initial, scratch, rows, final):
    return extension().replay(initial, scratch.k, scratch.v, scratch.g, scratch.b, rows, final)


def batch_chain(p, weight, decay, bias, norm, eps, tables, layer, output, sums, *, negative_decay=False):
    """One unchanged chain body per independent(State,value-head) CTA."""
    from tensorfold.families.qwen4_exp.cuda.gdn_multi import NativeTables

    if type(tables) is not NativeTables:
        raise TypeError("native GDN batch requires the current owned frame")
    pointers, starts = tables.operands(layer, p, output, sums)
    return extension().batch_chain(p, weight, decay, bias, norm, eps, pointers, starts, output, sums, negative_decay)


def prepare_batch(device, value_heads):
    """Resolve the exact kernel functions before physical startup admission."""
    import torch
    with torch.cuda.device(device):
        extension().prepare_batch(value_heads)


def batch_chain_prepared(p, weight, decay, bias, norm, eps, operands, output, sums, *, negative_decay=False):
    """Submit cold-validated exact-owner descriptor views, including on its capture stream.

    The decoder retains the static tables and registered pointees until graphs
    retire. Its live actor issues/validates the fresh frame before every replay.
    Native operand geometry/alias checks remain the same batch entry contract.
    """
    pointers, starts = operands
    return extension().batch_chain(p, weight, decay, bias, norm, eps, pointers, starts, output, sums, negative_decay)
