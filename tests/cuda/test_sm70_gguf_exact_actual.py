"""Real-source all-slot output and common six-State arithmetic recovery gates."""

from pathlib import Path
import gc
import numpy as np
import pytest
import torch

from functools import lru_cache
import os
import hashlib
from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint, FORMATS, decode_rows
from tensorfold.families.qwen4_exp.cuda.gguf_weights import (
    GGUFPacked,
    GGUFExperts,
    GGUFDense,
)
from test_sm70_gguf_exact import bits

MODEL = Path(
    os.environ.get(
        "TF_FLASHNEXT_GGUF",
        "/models/huihui-v2/unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    )
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not MODEL.is_file(),
    reason="optional SM70 actual Flash Next GGUF fixture",
)
dims, count = 2560, 10


@lru_cache(maxsize=1)
def checkpoint():
    return GGUFCheckpoint.open(MODEL)


def read(tensor, start, rows):
    offset, n = tensor.row_span(start, rows)
    with tensor.path.open("rb", buffering=0) as stream:
        stream.seek(offset)
        value = stream.read(n)
    if len(value) != n:
        raise IOError("short source span")
    return np.frombuffer(value, np.uint8).reshape(rows, tensor.row_bytes).copy()


def make(layer, world):
    width = 640 // world
    first = 3 * width
    base = f"blk.{layer}."
    packed = []
    metadata = []
    for part in ["gate", "up", "down"]:
        tensor = checkpoint().tensors[base + f"ffn_{part}_exps.weight"]
        if part == "down":
            raw = np.concatenate([read(tensor, e * dims, dims) for e in range(count)])
            quantum, stored = FORMATS[tensor.ggml_type][1:]
            lo = first // quantum * stored
            hi = -(-(first + width) // quantum) * stored
            data = np.ascontiguousarray(raw[:, lo:hi])
            columns = width
            offset = first % quantum
        else:
            raw = np.concatenate(
                [read(tensor, e * 640 + first, width) for e in range(count)]
            )
            data = raw
            columns = dims
            offset = 0
        packed.append(
            GGUFPacked(
                torch.tensor(data, device="cuda"),
                tensor.ggml_type,
                data.shape[0],
                columns,
                offset,
            )
        )
        metadata.append(
            {
                "name": tensor.name,
                "encoding": tensor.encoding,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "retained_bytes": data.nbytes,
            }
        )
    shared = []
    for part in ["gate", "up", "down"]:
        tensor = checkpoint().tensors[base + f"ffn_{part}_shexp.weight"]
        if part == "down":
            raw = read(tensor, 0, dims)
            value = decode_rows(tensor, raw).astype(np.float16)[
                :, first : first + width
            ]
        else:
            raw = read(tensor, first, width)
            value = decode_rows(tensor, raw).astype(np.float16)
        shared.append(
            GGUFDense(torch.from_numpy(np.ascontiguousarray(value)).to("cuda"))
        )
    return GGUFExperts(*packed, shared, count=count, width=width, dims=dims), metadata


@pytest.fixture(autouse=True)
def release_test_owners():
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[0, 2])
def actual_bank(request):
    bank, metadata = make(request.param, 8)
    bank.packed_rows = 32
    return bank, metadata


@pytest.mark.parametrize("rows", [1, 3, 6, 18, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_actual_source_complete_routed_shared_outputs(actual_bank, rows, dtype):
    bank, _ = actual_bank
    torch.manual_seed(9201 + rows)
    x = (torch.randn(rows, 2560, device="cuda") * 0.1).half()
    ids = (torch.arange(rows * 11, device="cuda").reshape(rows, 11) + rows).remainder(
        10
    )
    ids[:, -1] = 10
    expected = bank.forward_reference(x, ids).to(dtype)
    actual = torch.empty_like(expected)
    bank.forward_exact(x, ids, actual)
    bits(actual, expected)
    # Every source row uses the original path independently as well.
    serial = torch.cat(
        [
            bank.forward_reference(x[i : i + 1], ids[i : i + 1]).to(dtype)
            for i in range(rows)
        ]
    )
    bits(actual, serial)


def committed_equal(a, b):
    # State.clone() canonically restores the active recurrent bank. Its parity
    # can differ while the committed state is byte-identical.
    assert a.pos == b.pos
    for index, parity in enumerate(a.cur):
        bits(a.rec[parity, index], b.rec[b.cur[index], index])
    for name in ["conv", "ple_tail"]:
        bits(getattr(a, name), getattr(b, name))
    for left, right in zip(a.kc, b.kc):
        for name in ["k", "v", "ks", "vs"]:
            bits(getattr(left, name)[: a.pos], getattr(right, name)[: b.pos])
    for left, right in zip(a.ikc, b.ikc):
        bits(left[: a.pos], right[: b.pos])
    for left, right in zip(a.pooled, b.pooled):
        bits(left[: a.pos // a.ratio], right[: b.pos // b.ratio])
    np.testing.assert_array_equal(a.ple_history, b.ple_history)


@pytest.mark.parametrize("cache", ["fp16", "int8"])
@pytest.mark.parametrize("lanes", [1, 6])
@pytest.mark.parametrize("expert_world", [8, 4])
def test_common_full_state_actual_banks_match_original_with_graph_partial_commit_and_prefix(
    cache, lanes, expert_world, monkeypatch
):
    from test_flashnext_sm70_family import _small_weights
    from test_sm70_gguf_prepared import dense_fixture
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.families.qwen4_exp.cuda import padded_linears
    from tensorfold.families.qwen4_exp.cuda.decode import prefill
    from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, stage
    from tensorfold.families.qwen4_exp.cuda.multi import _slot
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    monkeypatch.setattr(padded_linears, "_ENABLED", True)
    w = dense_fixture(
        _small_weights(include_ple=True, hidden=2560, lowrank=320, head_dim=256)
    )
    w.cfg.experts, w.cfg.top_k = 10, 10
    w.cfg.moe_width = w.cfg.shared_width = 640 // expert_world
    for layer, source_layer in zip(w.layers, [0, 2]):
        layer.moe.experts = make(source_layer, expert_world)[0]
        layer.moe.experts.packed_rows = 32
        layer.moe.experts.implementation = "reference"
        layer.moe.router = (torch.randn(11, 2560, device="cuda") * 0.03).half().float()
    count = 3 if lanes == 1 else 4
    rows = lanes * count
    rb = Buffers(w, rows, 64, moe_prefill=True)
    nb = Buffers(w, rows, 64, moe_prefill=True)
    pb = Buffers(w, 12, 64, prefill=True)
    states = [State(w, 64, 4, cache) for _ in range(lanes)]
    for i, state in enumerate(states):
        prefill(
            _slot(w, state, nb, None, pb, 64, 12),
            [4, 17, 3, 99, 33, 2, 7, 5, 6],
            None,
            mtp=False,
        )
        if i % 2:
            state.rec[1 - state.cur[0]].copy_(state.rec[state.cur[0]])
            state.cur = [1 - x for x in state.cur]
    for iteration in range(2):
        copies = [state.clone() for state in states]
        tokens = [
            [(lane * 13 + i + iteration * 17) % w.cfg.vocab for i in range(count)]
            for lane in range(lanes)
        ]
        for layer in w.layers:
            layer.moe.experts.implementation = "reference"
        ref_segs = stage(w, rb, list(zip(copies, tokens)))
        expected = compute(w, ref_segs, rb, context=256).clone()
        stream_expected = rb.streams[:rows].clone()
        for layer in w.layers:
            layer.moe.experts.implementation = "packed_exact"
        segs = stage(w, nb, list(zip(states, tokens)))
        compute(w, segs, nb, context=256)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = compute(w, segs, nb, context=256)
        graph.replay()
        bits(actual, expected)
        bits(nb.streams[:rows], stream_expected)
        for lane, ((state, start, end), copied) in enumerate(zip(segs, copies)):
            keep = 1 + lane % count
            commit(w, state, nb, count, keep, at=start)
            commit(w, copied, rb, count, keep, at=start)
            committed_equal(state, copied)
        del graph
    original = states[0]
    restored = State(w, 64, 4, cache)
    restored.copy_prefix(original, original.pos, 0)
    restored.restore(original.snapshot())
    committed_equal(original, restored)
    for layer in w.layers:
        layer.moe.experts.implementation = "reference"
    ref_segs = stage(w, rb, [(original, [31])])
    expected = compute(w, ref_segs, rb, context=256).clone()
    for layer in w.layers:
        layer.moe.experts.implementation = "packed_exact"
    segs = stage(w, nb, [(restored, [31])])
    bits(compute(w, segs, nb, context=256), expected)
    commit(w, original, rb, 1, 1)
    commit(w, restored, nb, 1, 1)
    committed_equal(original, restored)
