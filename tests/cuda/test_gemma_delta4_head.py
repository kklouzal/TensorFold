"""Actual tied lookup/head/FP8-copy/graph boundaries; no serving speed claim."""
from types import SimpleNamespace

import torch

from tensorfold.cuda import b16_delta4 as codec
from tensorfold.cuda.requant import Head
from tensorfold.families.gemma4.cuda.dflash import _packed_fp8_head
from tensorfold.families.gemma4.cuda.forward import Gemma4Target
from tensorfold.families.gemma4.cuda.head import Delta4Table
from tensorfold.families.qwen3_5.cuda.b16 import matmul as raw_reference
from tensorfold.families.qwen3_5.cuda.weights import Plain


def bits_equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    view = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.view(view), expected.view(view))


def actual_table():
    generator = torch.Generator().manual_seed(635741)
    original = torch.randn((129, 64), generator=generator).to(torch.bfloat16)
    original.view(torch.int16)[0, 0] = -32768
    cpu = codec.pack(original)
    status = codec.TransferStatus()
    packed = codec.to_cuda(cpu, status=status)
    table = Delta4Table(packed, 16384, reuse_rows=16)
    raw = original.to('cuda')
    x = torch.randn((17, 64), generator=generator).to(device='cuda', dtype=torch.bfloat16)
    return table, status, raw, x


def test_actual_tied_table_preserves_raw_scaled_and_single_softcap_contracts():
    table, status, raw, x = actual_table()
    target = object.__new__(Gemma4Target)
    target.weights = SimpleNamespace(embed=table)
    target.config = SimpleNamespace(softcap=30.)
    target._embedding = table.embedding
    target._embed_scale = torch.tensor(8., dtype=torch.bfloat16, device='cuda')
    ids = torch.tensor([[0, 1], [127, 128]], device='cuda', dtype=torch.int32)
    bits_equal(target.raw_embed_tokens(ids), raw[ids.long()])
    bits_equal(target.embed_tokens(ids), raw[ids.long()] * target._embed_scale)
    for rows in (1, 5, 17):
        expected = torch.tanh(raw_reference(x[:rows], raw) / 30.) * 30.
        bits_equal(target.project_logits(x[:rows]), expected)
    torch.cuda.synchronize()
    assert status.safe_to_release and status.transferred and not hasattr(table, 'weight')


def test_actual_raw_lookup_fp8_copy_matches_existing_private_code_and_scale_words():
    table, status, raw, x = actual_table()
    actual = _packed_fp8_head(table, softcap=30.)
    expected = Head(Plain(raw), 'fp8', softcap=30.)
    assert torch.equal(actual.linear.w8, expected.linear.w8)
    assert torch.equal(actual.linear.cols.view(torch.int32), expected.linear.cols.view(torch.int32))
    bits_equal(actual(x[:5]), expected(x[:5]))
    torch.cuda.synchronize()
    assert status.safe_to_release


def test_lookup_and_ordered_head_graph_replay_keep_exact_immutable_storage():
    table, status, raw, x = actual_table()
    ids = torch.tensor([0, 1, 128], device='cuda', dtype=torch.int64)
    static_x = x[:5].clone()
    warm_lookup, warm_logits = table.embedding(ids), table(static_x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        lookup = table.embedding(ids)
        logits = table(static_x)
    graph.replay()
    bits_equal(lookup, raw[ids])
    bits_equal(logits, raw_reference(static_x, raw))
    old_lookup = raw[ids].clone()
    old_logits = raw_reference(static_x, raw)
    ids.copy_(torch.tensor([128, 1, 0], device='cuda'))
    static_x.mul_(-.75)
    assert not torch.equal(old_lookup.view(torch.int16), raw[ids].view(torch.int16))
    assert not torch.equal(old_logits.view(torch.int16), raw_reference(static_x, raw).view(torch.int16))
    graph.replay()
    bits_equal(lookup, raw[ids])
    bits_equal(logits, raw_reference(static_x, raw))
    torch.cuda.synchronize()
    del graph  # every static input/output/field remains live through successful destruction
    assert status.safe_to_release and warm_lookup is not None and warm_logits is not None
