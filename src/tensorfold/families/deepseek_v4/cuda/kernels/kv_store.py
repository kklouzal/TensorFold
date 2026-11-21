# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
#   vllm/models/deepseek_v41/kv_store.py (blob 69d83fb56dac4fc1e2645d5c0911c50d64b3f61a)
# TensorFold changes (kernel arithmetic verbatim):
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - Only the two Triton kernels are vendored: _insert_c (pack complete groups into the hot ring and, on the
#   writing rank, the host rows) and _fill_selected (copy each decode token's selected 288-byte rows from the hot
#   ring or host rows into the `selected` buffer that attention_ops._gather_attention_packed reads).
# - Not vendored: the V41PackedKV owner (SharedHostArena memfd, socket sharing, rank roles, staging), which
#   TensorFold's cache owner replaces. Callers own all buffers: hot [HOT_ROWS, 288] u8, host [rows, 288] u8
#   (int32-aligned base), selected [<= 8, 512, 288] u8; positions are the forward's positions (newest last).
# - vllm.triton_utils / kv_pack imports -> package-relative.
"""V4.1 packed long-context KV (G4b) row kernels; the owner lives in TensorFold."""

from .kv_pack import pack_c_group
from .triton_utils import jit, tl


@jit(do_not_specialize=["HOT_ROWS"])
def _insert_c(
    values,
    positions,
    valid_tokens,
    hot,
    host,
    HOT_ROWS,
    RATIO: tl.constexpr,
    WRITE_HOST: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    position = tl.load(positions + token).to(tl.int64)
    valid = (
        tl.load(valid_tokens + token) & ((position + 1) % RATIO == 0) & (position >= 0)
    )
    if valid:
        c = position // RATIO
        x = values + token.to(tl.int64) * 512
        pack_c_group(x, hot + (c % HOT_ROWS) * 288, group)
        if WRITE_HOST:
            pack_c_group(x, host + c * 288, group)


@jit(do_not_specialize=["T", "HOT_ROWS"])
def _fill_selected(
    topk,
    positions,
    hot,
    host,
    out,
    T,
    HOT_ROWS,
    RATIO: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t, part = tl.program_id(0), tl.program_id(1)
    j = part * BLOCK + tl.arange(0, BLOCK)
    c = tl.load(topk + t * TOPK + j).to(tl.int64)
    newest = (tl.load(positions + T - 1).to(tl.int64) + 1) // RATIO - 1
    valid = c >= 0
    in_hot = valid & (c > newest - HOT_ROWS)
    w = tl.arange(0, 128)
    words = (w < 72)[None, :]
    hot_words = (hot + (c % HOT_ROWS) * 288).to(tl.pointer_type(tl.int32))
    host_words = (host + c * 288).to(tl.pointer_type(tl.int32))
    a = tl.load(hot_words[:, None] + w[None, :], in_hot[:, None] & words, 0)
    b = tl.load(host_words[:, None] + w[None, :], (valid & ~in_hot)[:, None] & words, 0)
    dst = (out + (t * TOPK + j).to(tl.int64) * 288).to(tl.pointer_type(tl.int32))
    tl.store(dst[:, None] + w[None, :], tl.where(in_hot[:, None], a, b), words)
