"""Checkpoint decoder hierarchy over the native selector score boundary."""
import os
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels import attention_ops, index_shard, mu_packed, mu_reindex, mu_topk

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
                                reason="CUDA or Triton interpreter required")


def _reference(source, scores, positions, valid, blocks, block):
    picks = torch.full((len(source), 512), -1, dtype=torch.int32)
    candidates = torch.full((len(source), blocks), -1, dtype=torch.int32)
    for row, (position, live) in enumerate(zip(positions.tolist(), valid.tolist(), strict=True)):
        if not live:
            continue
        length = position + 1
        complete, last = divmod(length - 1, block)
        block_scores = [source[row, j * block:(j + 1) * block].max().item() for j in range(complete)]
        ranking = sorted(range(complete), key=lambda j: (-block_scores[j], j))
        retained = sorted([complete, *ranking[:blocks - 1]])
        candidates[row, :len(retained)] = torch.tensor(retained)
        visible = [i for j in retained for i in range(j * block, min((j + 1) * block, length))]
        wins = sorted(sorted(visible, key=lambda i: (-scores[row, i].item(), i))[:512])
        picks[row, :len(wins)] = torch.tensor(wins)
        assert 0 <= last < block
    return candidates.to(source.device), picks.to(source.device)


def test_official_pool_boundary_and_replay():
    blocks, block, width = 2048, 8, 16400
    positions = torch.tensor([16383, 16384, 16391, 16399], device=DEVICE)
    valid = torch.tensor([True, True, True, False], device=DEVICE)
    column = torch.arange(width, device=DEVICE)
    source = (column // block).half().expand(4, -1).contiguous()
    # The newest block would be the worst, and the padding after it has
    # poisonous values. Neither may change the previously visible block scores.
    source[1, 16384] = -1
    source[1, 16385:16392] = torch.nan
    later = -(column // block).half().expand(4, -1).contiguous()
    candidates = torch.empty(4, blocks, dtype=torch.int32, device=DEVICE)
    full = torch.empty(4, 512, dtype=torch.int32, device=DEVICE)
    restricted = torch.empty_like(full)

    def select():
        mu_reindex.select_ranked(full, source, positions, valid, 1, mu_topk.topk,
                                 candidates=candidates, block=block, publish=True)
        mu_reindex.select_ranked(restricted, later, positions, valid, 1, mu_topk.topk,
                                 candidates=candidates, block=block, publish=False)

    def check():
        expected, picks = _reference(source.cpu(), later.cpu(), positions.cpu(), valid.cpu(), blocks, block)
        assert torch.equal(candidates.cpu(), expected)
        assert torch.equal(restricted.cpu(), picks)

    select()
    check()
    assert candidates[1, -1].item() == 2048
    assert restricted[0, 0].item() == 0 and restricted[1, 0].item() == 8
    if DEVICE == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            select()
        positions[1] = 16383
        valid[0] = False
        graph.replay()
        check()
        assert restricted[1, 0].item() == 0


def test_mixed_selector_writes_candidate_rows_back_in_step_order(monkeypatch):
    blocks, block, width = 128, 8, 1032
    positions = torch.tensor([1030, 1022, 1023, 1024, 0], device=DEVICE)
    valid = torch.tensor([True, True, True, True, False], device=DEVICE)
    source = (torch.arange(width, device=DEVICE) // block).half().expand(5, -1).contiguous()
    later = -source
    q = torch.zeros(5, 4, 128, dtype=torch.float16, device=DEVICE)
    weights = torch.zeros(5, 4, dtype=q.dtype, device=DEVICE)
    req = torch.tensor([0, 1, 1, 1, 2], dtype=torch.int32, device=DEVICE)
    decode = torch.tensor([0, 4], device=DEVICE)
    step = SimpleNamespace(prefill=[(1, 1, 4, 1024)], decode_index=decode, valid=valid, token_req=req)
    candidates = torch.empty(5, blocks, dtype=torch.int32, device=DEVICE)
    values = source

    def fake_index(*args, out, **kwargs):
        out.copy_(values.index_select(0, decode))

    def fake_prefill(q, weights, keys, pos, valid, ratio, all_reduce, *, hierarchy):
        target = torch.empty(len(q), 512, dtype=torch.int32, device=q.device)
        cand, group, publish = hierarchy
        assert torch.equal(pos, positions[1:4])
        return mu_reindex.select_ranked(target, values[1:4, :1025].contiguous(), pos, valid, 1,
                                        mu_topk.topk, candidates=cand, block=group, publish=publish)

    monkeypatch.setattr(mu_packed, "index_scores", fake_index)
    monkeypatch.setattr(mu_packed, "unpack_request_index", lambda *args, **kw: None)
    monkeypatch.setattr(mu_packed, "select_compressed_positions", fake_prefill)
    for publish in (True, False):
        output = mu_packed.select_positions(
            q, weights, positions, step, None, [None] * 3, layout=None, source=20, ratio=1,
            width=width, all_reduce=lambda x: x, hierarchy=(candidates, block, publish),
        )
        values = later
    expected, picks = _reference(source.cpu(), later.cpu(), positions.cpu(), valid.cpu(), blocks, block)
    assert torch.equal(candidates.cpu(), expected)
    assert torch.equal(output.cpu(), picks)


def test_position_sharded_reindex_uses_globally_gathered_scores():
    blocks, block, width, ranks = 128, 8, 1536, 8
    from tensorfold.families.deepseek_v4.cuda.kernels.index_shard import BN
    positions = torch.tensor([1024, width - 1], device=DEVICE)
    valid = torch.ones(2, dtype=torch.bool, device=DEVICE)
    source = (torch.arange(width, device=DEVICE) // block).half().expand(2, -1).contiguous()
    later = -source
    candidates = torch.empty(2, blocks, dtype=torch.int32, device=DEVICE)
    for scores, publish in ((source, True), (later, False)):
        rows = scores.view(2, -1, ranks, BN).permute(2, 0, 1, 3).reshape(ranks, 2, -1)
        got = index_shard.select_gathered(rows, positions, valid, 1, hierarchy=(candidates, block, publish))
    expected, picks = _reference(source.cpu(), later.cpu(), positions.cpu(), valid.cpu(), blocks, block)
    assert torch.equal(candidates.cpu(), expected)
    assert torch.equal(got.cpu(), picks)


def test_row_split_prefill_restores_candidate_rows_in_request_order():
    blocks, block, width = 128, 8, 1056
    tokens, first, world = 33, 1023, 2
    positions = torch.arange(first, first + tokens, device=DEVICE)
    valid = torch.ones(tokens, dtype=torch.bool, device=DEVICE)
    valid[31] = False
    q = torch.zeros(tokens, 4, 128, dtype=torch.float16, device=DEVICE)
    q[:, 0, 0] = 1
    weights = torch.zeros(tokens, 4, dtype=q.dtype, device=DEVICE)
    weights[:, 0] = 0.5
    keys = torch.zeros(width, 128, dtype=q.dtype, device=DEVICE)
    source = (torch.arange(width, device=DEVICE) // block).half().expand(tokens, -1).contiguous()
    later = source.amax() - source
    expected, picks = _reference(source.cpu(), later.cpu(), positions.cpu(), valid.cpu(), blocks, block)
    _, full = _reference(source.cpu(), source.cpu(), positions.cpu(), valid.cpu(), width // block, block)
    # The oracle's layer-20 result ranks all visible positions.
    gathered_outputs = {blocks: expected, 512: full}

    for rank in range(world):
        for publish, scores, wanted in ((True, source, full), (False, later, picks)):
            candidates = torch.empty(tokens, blocks, dtype=torch.int32, device=DEVICE) if publish else expected.to(DEVICE)
            keys[:, 0].copy_(scores[0])
            calls = 0

            def all_gather(send, recv):
                nonlocal calls
                calls += 1
                if calls <= 2:
                    recv.copy_(send)
                    return
                expected_rows = gathered_outputs[send.shape[1]] if publish else picks
                framed = torch.full((world * 32, send.shape[1]), -1, dtype=send.dtype, device=send.device)
                framed[:tokens].copy_(expected_rows)
                framed = framed.view(world, 32, -1)
                assert torch.equal(send, framed[rank])
                recv.copy_(framed)

            actual = attention_ops.select_rows_split(q, weights, keys, positions, valid, 1, first_position=first,
                                                     rank=rank, world=world, all_gather=all_gather,
                                                     hierarchy=(candidates, block, publish))
            assert torch.equal(actual.cpu(), wanted)
            assert torch.equal(candidates.cpu(), expected)
            assert calls == (4 if publish else 3)


def test_target_publish_consume_and_reset_per_forward(monkeypatch):
    from tensorfold.families.deepseek_v4.cuda.kernels import rope_qdq
    from tensorfold.families.deepseek_v4.cuda.target import Target

    monkeypatch.setattr(rope_qdq, "rope_qdq", lambda q, *args, **kwargs: q)
    selected = []

    def selector(*args, hierarchy, **kwargs):
        selected.append(hierarchy)
        return None

    monkeypatch.setattr(mu_packed, "select_positions", selector)
    target = SimpleNamespace(
        g=SimpleNamespace(candidate_source=20, candidate_blocks=2048, candidate_block=8,
                          index_sources=(2, 8, 14, 20, 24, 28, 32, 36)),
        index_context=131072, pool=SimpleNamespace(cache=None, layout=None), _linear=lambda x, weight: x,
        c=SimpleNamespace(world=SimpleNamespace(all_gather=None)), w=SimpleNamespace(rank=0),
        _sum_half=lambda x: x,
    )
    q = torch.zeros(1, 512, device=DEVICE)
    step = SimpleNamespace(prefill=[], positions=None, block_table=None, decode_index=None)
    layer = SimpleNamespace(index=20, ratio=1, source=20, idx_wq_b=None, idx_proj=None)
    shared = {}
    Target._select(target, layer, q, q, step, None, shared)
    pool = shared["index_candidates"]
    assert selected.pop() == (pool, 8, True)
    for index in (24, 28, 32, 36):
        layer.index = index
        Target._select(target, layer, q, q, step, None, shared)
        assert selected.pop() == (pool, 8, False)
    assert "index_candidates" not in shared
    layer.index = 20
    Target._select(target, layer, q, q, step, None, {})
    assert selected.pop()[0] is not pool

    step.prefill = [(0, 0, 1, 16383)]
    Target._select(target, layer, q, q, step, None, {})
    assert selected.pop() is None
