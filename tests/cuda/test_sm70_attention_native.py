"""Qualified H3/KV1/D256 attention and explicit original-Torch fallback scope."""

from copy import copy
import pytest
import torch
from unittest.mock import patch

from tensorfold.cuda.sm70 import attention_native, family_ops as ops
from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 native attention")


def reference(*args, **kwargs):
    with patch.object(ops, "_NATIVE_ATTENTION", False):
        return ops.attention(*args, **kwargs)


def query_reference(q, key, value, position, scratch, rows, scale, ks, vs, kb, vb):
    """The independent authority is one ordinary Torch query at its true end."""
    outputs = []
    start = int(position.cpu()[0])
    for row in range(rows):
        view = copy(scratch)
        view.ids, view.nk, view.sparse = scratch.ids[row:row+1], scratch.nk[row:row+1], scratch.sparse[row:row+1]
        out = torch.empty_like(q[row:row+1])
        outputs.append(reference(q[row:row+1], key, value, position+row, view, 1, scale, out=out,
                                 context=start+row+1, ks=ks, vs=vs, k_bits=kb, k_codec=0,
                                 v_bits=vb, v_codec=0))
    return torch.cat(outputs)


def inputs(capacity, *, rows=3, dim=256, qheads=3, kvheads=1, kb=0, vb=0, sparse=False):
    torch.manual_seed(731)
    q = torch.randn((rows, qheads, dim), device="cuda", dtype=torch.float16)
    key = torch.randn((capacity, kvheads, dim), device="cuda", dtype=torch.float16)
    value = torch.randn_like(key) * 0.2
    ks = vs = None
    if kb:
        q = ops.h32(q).half()
        key, ks = ops.quantize_int8(key)
    if vb:
        value, vs = ops.quantize_int8(value)
    scratch = AttnScratch(rows, qheads, dim, capacity, "cuda", budget=min(2048, capacity // 2), ratio=4, dtype=torch.float16)
    position = torch.tensor([capacity - rows - 3], device="cuda", dtype=torch.int32)
    if sparse:
        for row in range(rows):
            end = int(position[0]) + row + 1
            count = scratch.budget + end % scratch.ratio
            scratch.sparse[row] = 1
            scratch.nk[row] = count
            scratch.ids[row, :scratch.budget] = torch.arange(scratch.budget, device="cuda", dtype=torch.int32)
            scratch.ids[row, scratch.budget:] = end // scratch.ratio * scratch.ratio + torch.arange(scratch.ratio, device="cuda")
    return q, key, value, position, scratch, ks, vs


@pytest.mark.parametrize("context", [128, 512, 8192])
@pytest.mark.parametrize("kb,vb", [(0, 0), (8, 8), (0, 8), (8, 0)])
@pytest.mark.parametrize("sparse", [False, True])
def test_native_attention_matches_existing_stored_basis_fp16_policy(context, kb, vb, sparse):
    q, key, value, position, scratch, ks, vs = inputs(context, kb=kb, vb=vb, sparse=sparse)
    rows, scale = q.shape[0], q.shape[-1] ** -0.5
    expected = query_reference(q, key, value, position, scratch, rows, scale, ks, vs, kb, vb)
    actual = attention_native.attention(q, key, value, position, scratch, rows, scale, scratch.out,
                                         context, ks, vs, kb, vb)
    assert torch.equal(actual, expected)
    assert scratch.native_workspace.probabilities.dtype == torch.float16
    assert scratch.native_workspace.partials.dtype == torch.float32


@pytest.mark.parametrize("kb,vb", [(0, 0), (8, 8), (0, 8), (8, 0)])
def test_native_eager_and_captured_bucket_are_bit_identical_at_advancing_positions(kb, vb):
    q, key, value, position, scratch, ks, vs = inputs(8192, kb=kb, vb=vb)
    scale, rows = q.shape[-1] ** -0.5, q.shape[0]
    position.fill_(125)

    def run(context):
        return attention_native.attention(q, key, value, position, scratch, rows, scale,
                                            scratch.out, context, ks, vs, kb, vb)

    run(8192)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(8192)
    try:
        for start in (125, 128, 257, 1025, 2049, 7000):
            position.fill_(start)
            eager = run(start + rows).clone()
            expected = query_reference(q, key, value, position, scratch, rows, scale, ks, vs, kb, vb)
            graph.replay()
            assert torch.equal(scratch.out, eager)
            assert torch.equal(scratch.out, expected)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("sparse", [False, True])
def test_masked_nonfinite_values_do_not_leak_between_query_rows(sparse):
    q, key, value, position, scratch, _, _ = inputs(256, rows=3)
    position.fill_(64)                 # First true end65 is the qualified lower boundary.
    q.zero_()
    value.fill_(1)
    if sparse:
        scratch.sparse.fill_(1)
        scratch.nk.fill_(4)
        scratch.ids[:, :4] = torch.tensor([0, 2, 4, 6], device="cuda", dtype=torch.int32)
        value[1].fill_(float("nan"))
    value[65:].fill_(float("nan"))
    attention_native.attention(q, key, value, position, scratch, 3, 0.0625, scratch.out, 256, None, None, 0, 0)
    assert scratch.out[0].isfinite().all()
    if sparse:
        assert scratch.out.isfinite().all()
    else:
        assert scratch.out[1:].isnan().all()


def test_unqualified_replicated_gqa_and_128_head_dimension_refuses_native_and_keeps_torch(monkeypatch):
    q, key, value, position, scratch, ks, vs = inputs(256, rows=2, dim=128, qheads=6, kvheads=2, kb=8, vb=8)
    expected = reference(q, key, value, position, scratch, 2, 128 ** -0.5, context=256,
                             ks=ks, vs=vs, bits=8).clone()
    with pytest.raises(ValueError, match="qualified"):
        attention_native.attention(q, key, value, position, scratch, 2, 128 ** -0.5,
                                   scratch.out, 256, ks, vs, 8, 8)
    assert not hasattr(scratch, "native_workspace")

    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified geometry entered native attention")

    monkeypatch.setattr(attention_native, "attention", forbidden)
    monkeypatch.setattr(ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(ops, "_NATIVE_ATTENTION", True)
    actual = ops.attention(q, key, value, position, scratch, 2, 128 ** -0.5, context=256,
                           ks=ks, vs=vs, bits=8)
    assert torch.equal(actual, expected)


def test_native_uniform_attention_matches_independent_probability_storage_oracle():
    q, key, value, position, scratch, _, _ = inputs(512, rows=1)
    q.zero_()
    key.zero_()
    position.fill_(256)
    expected_probability = torch.tensor(1 / 257, dtype=torch.float16).float()
    values = value[:257, 0].cpu().float()
    # Independent pinned small-N PV tree: four contiguous65-key partitions,
    # sequential FP32 products/additions, then ascending partial reduction.
    partials = []
    for part in range(4):
        acc = torch.zeros(256, dtype=torch.float32)
        for key_index in range(part*65, min(257, (part+1)*65)):
            acc += values[key_index]*expected_probability
        partials.append(acc)
    expected = (((partials[0]+partials[1])+partials[2])+partials[3]).half()
    result = attention_native.attention(q, key, value, position, scratch, 1, 0.0625,
                                         scratch.out, 512, None, None, 0, 0)
    torch.testing.assert_close(scratch.native_workspace.probabilities[0, :, :257],
                               torch.full((3, 257), float(expected_probability), device="cuda", dtype=torch.float16),
                               atol=0, rtol=0)
    assert torch.equal(result[0].cpu(), expected.expand(3, -1))
