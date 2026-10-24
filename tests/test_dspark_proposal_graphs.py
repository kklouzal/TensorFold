"""CPU arithmetic/policy controls for private captured greedy proposal execution."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda.dspark_proposal_graphs import (
    ProposalWorkspace, bucketed_hidden, capture_fits, context_bucket, graph_host_bytes, masked_gqa, policy_eligible, proposal_body,
)
from test_dspark_cuda import absorb, bits, owner, provider


class GreedyOracle:
    """CPU equations only; actual CUDA execution uses the audited native packet."""
    @staticmethod
    def local(logits, ids, norm, status, packet):
        maximum = logits.max(-1).values
        tied = torch.where(logits[0] == maximum, ids, torch.iinfo(torch.int64).max)
        packet[0] = maximum.view(torch.int32)[0]
        packet[1] = tied.min()
        packet[2] = norm.view(torch.int32)[0]
        packet[3:] = status

    @staticmethod
    def global_winner(frame, token, safe, vocab):
        scores = frame[:, 0].int().view(torch.float32)
        maximum = scores.max()
        tied = torch.where(scores == maximum, frame[:, 1], torch.iinfo(torch.int64).max)
        token.copy_(tied.min().view(1))
        safe.copy_(torch.where((token >= 0) & (token < vocab), token, 0))


def workspace(p, st, block=3, bucket=None):
    bucket = st.length if bucket is None else bucket
    width = p.shard.width
    storage = torch.empty(block * p.world * 5 + block, dtype=torch.int64)
    return NS(keys=st.keys, values=st.values, block=block, bucket=bucket,
              length=torch.tensor([st.length]), block_offsets=torch.arange(block), prefix_offsets=torch.arange(bucket),
              inputs=torch.tensor([4] + [p.config.mask] * (block - 1)), previous=torch.zeros(1, dtype=torch.int64),
              status=torch.zeros(2, dtype=torch.int32), corrected=torch.empty(block, width, dtype=torch.float16),
              logits=torch.empty(1, width), ids=torch.arange(width) + p.shard.start, norm=torch.zeros(1),
              send=torch.empty(5, dtype=torch.int64), frames=storage[:block * p.world * 5].view(block, p.world, 5),
              tokens=storage[-block:], safe_input=torch.zeros(1, dtype=torch.int32), native=GreedyOracle())


@pytest.mark.parametrize("length", (0, 1, 4, 7))
def test_unpadded_bucket_has_identical_backbone_and_markov_storage_to_original_provider(length):
    p = provider()
    st = p.make_state(owner(), capacity=32)
    if length:
        from test_dspark_cuda import features
        absorb(p, st, features(p.config, length))
    ws = workspace(p, st)
    before = st.keys.clone(), st.values.clone(), st.length, st.version
    expected = p.propose(st, 4, block=3)
    actual_hidden = bucketed_hidden(p, ws, p.embed(ws.inputs))
    bits(actual_hidden, p._hidden(st, p.embed(ws.inputs)))
    tokens = proposal_body(p, ws)
    bits(ws.corrected, expected.corrected_logits)
    assert tokens.tolist() == expected.ids.tolist()
    assert not ws.frames[:, :, 3].any()
    bits(st.keys, before[0])
    bits(st.values, before[1])
    assert (st.length, st.version) == before[2:]


@pytest.mark.parametrize("length", (0, 1, 4))
def test_bucket_mask_ignores_uncommitted_nan_cache_and_exposes_the_whole_bidirectional_block(length):
    gen = torch.Generator().manual_seed(3301)
    q, k, v = [torch.randn(shape, generator=gen).half() * 0.2
               for shape in ((3, 4, 8), (8, 2, 8), (8, 2, 8))]
    bk, bv = [torch.randn(3, 2, 8, generator=gen).half() * 0.2 for _ in range(2)]
    valid = torch.arange(8) < length
    k[length:].fill_(float("nan"))
    v[length:].fill_(float("nan"))
    actual = masked_gqa(q, k, v, bk, bv, valid)
    assert torch.isfinite(actual).all()
    # Independent scalar FP64 attention over precisely the allowed prefix+block.
    keys, values = torch.cat((k[:length], bk)).double(), torch.cat((v[:length], bv)).double()
    expected = torch.empty(3, 4, 8, dtype=torch.float64)
    for row in range(3):
        for head in range(4):
            scores = (keys[:, head // 2] * q[row, head].double()).sum(-1) / (8 ** 0.5)
            expected[row, head] = (scores.softmax(-1)[:, None] * values[:, head // 2]).sum(0)
    torch.testing.assert_close(actual.view_as(expected).double(), expected, rtol=0.001, atol=0.00015)
    bv[-1].add_(1)
    assert not torch.equal(actual[0], masked_gqa(q, k, v, bk, bv, valid)[0])


def test_masked_future_nan_cache_does_not_poison_a_complete_proposal_but_real_nonfinite_latches_and_clamps():
    p = provider()
    st = p.make_state(owner(), capacity=32)
    absorb(p, st)
    st.keys[:, st.length:].fill_(float("nan"))
    st.values[:, st.length:].fill_(float("nan"))
    ws = workspace(p, st, bucket=16)
    proposal_body(p, ws)
    assert not ws.frames[:, :, 3].any()
    assert ((ws.tokens >= 0) & (ws.tokens < p.config.vocab)).all()
    # A bad real prefix must not let a sentinel reach the next Markov lookup.
    st.keys[:, :st.length].fill_(float("nan"))
    proposal_body(p, ws)
    assert ws.frames[:, :, 3].all() and ws.previous.item() == 0
    assert (ws.tokens == torch.iinfo(torch.int64).max).all()


@pytest.mark.parametrize("sampling", (None, Sampling(1, temperature=0), Sampling(1, temperature=-1)))
def test_valid_greedy_policies_preserve_keyed_policy_validation(sampling):
    assert policy_eligible(128, 3, sampling, 0)
    assert not policy_eligible(128, 3, sampling, 0.5)


@pytest.mark.parametrize("sampling", (
    Sampling(1), Sampling(1, temperature=float("nan")), Sampling(1, temperature=float("-inf")),
    Sampling(1, temperature=0, top_p=2),
    Sampling(1, temperature=0, min_p=-0.1), NS(temperature=0), 0,
))
def test_invalid_or_stochastic_policy_declines_to_the_original_sampler(sampling):
    assert not policy_eligible(128, 3, sampling, 0)


@pytest.mark.parametrize("length,count", ((-1, 3), (2**64 - 1, 3), (1, 0), (1, True)))
def test_invalid_absolute_positions_or_counts_decline_before_policy_access(length, count):
    assert not policy_eligible(length, count, NS(temperature=0), 0)


@pytest.mark.parametrize("length,capacity,expected", ((0, 32, 32), (1, 8192, 256), (256, 8192, 256),
                                                     (257, 8192, 512), (8191, 8192, 8192)))
def test_context_bucket_respects_actual_admitted_cache_and_transition(length, capacity, expected):
    assert context_bucket(length, capacity) == expected


def test_resident_packet_plan_counts_each_owned_tensor_once():
    p = provider()
    st = p.make_state(owner(), capacity=32)
    ws = workspace(p, st, bucket=16)
    tensors = (ws.ids, ws.inputs, ws.length, ws.block_offsets, ws.prefix_offsets, ws.previous,
               ws.safe_input, ws.corrected, ws.logits, ws.norm, ws.status, ws.send)
    measured = sum(t.numel() * t.element_size() for t in tensors) + (3 * p.world * 5 + 3) * 8
    assert measured == ProposalWorkspace.device_bytes(p.shard.width, 3, 16, p.world)


def test_corrupted_negative_topk_declines_and_host_forecast_covers_all_live_context_buckets():
    sampling = Sampling(1, temperature=0)
    # Sampling normally clamps top_k during construction. Probe the existing
    # validator's immutable-policy invariant directly, rather than its constructor.
    object.__setattr__(sampling, "top_k", -1)
    assert not policy_eligible(128, 3, sampling, 0)
    assert graph_host_bytes(6, 7, 8, 237598) == 6 * 11 * (7 * 8 * 5 + 14 + 1) * 8
    assert graph_host_bytes(1, 3, 1, 32) == (3 * 5 + 6 + 1) * 8


def test_runtime_optional_owner_is_unselected_by_default_and_retirement_is_explicit(tmp_path, monkeypatch):
    from test_dspark_runtime import setup as make_setup
    from tensorfold.families.qwen4_exp.cuda import dspark_proposal_graphs as graph_module
    fixture = make_setup.__wrapped__(tmp_path, monkeypatch)
    _, _, runtime, _ = next(fixture)
    agreed = []
    bound = NS(owner=NS(), dec=NS(depth=3), _agree=lambda enabled: agreed.append(enabled) or enabled)
    try:
        assert not runtime.prepare_proposal_graphs(bound)
        assert runtime._proposal_graphs is None and agreed == [False]
        monkeypatch.setattr(graph_module, "_ENABLED", True)
        assert runtime.prepare_proposal_graphs(bound)
        with pytest.raises(ValueError, match="another target graph"):
            runtime.prepare_proposal_graphs(NS(_agree=lambda enabled: enabled))
        with pytest.raises(ValueError, match="must retire"):
            runtime.close()
        assert runtime.provider is not None and not runtime.closed
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
        runtime.retire_proposals(bound)
        assert runtime._proposal_graphs is None
    finally:
        try:
            next(fixture)
        except StopIteration:
            pass


def bare_graph_owner(p, st):
    from tensorfold.families.qwen4_exp.cuda.dspark_proposal_graphs import DSparkProposalGraphs
    graph = DSparkProposalGraphs.__new__(DSparkProposalGraphs)
    graph.provider, graph.block = p, 3
    graph.slots, graph.declined = {}, set()
    graph.closed = False
    graph.hits = graph.misses = graph.fallbacks = 0
    graph.runtime = NS(_state=lambda target: st)
    graph.owner = NS(room_bytes=0, budget=2**30, spare=None, _has_room=lambda: True)
    graph.bound = NS(_agree=lambda allowed: bool(allowed), _account=lambda previous: None)
    return graph


@pytest.mark.parametrize("fault", ("rounded_budget", "physical_spare", "allocator_change", "unproved"))
def test_optional_new_workspace_refusal_uses_allocator_bounds_and_never_retries_hot(fault, monkeypatch):
    p = provider()
    st = p.make_state(owner(), capacity=32)
    graph = bare_graph_owner(p, st)
    raw = ProposalWorkspace.device_bytes(p.shard.width, 3, 32, 1)
    votes, bounds_calls = [], []
    graph.owner.budget = raw + 2048
    graph.bound._agree = lambda allowed: votes.append(allowed) or bool(allowed)

    def bound(sizes):
        bounds_calls.append(tuple(sizes))
        return None if fault == "unproved" else (raw + 4096 if fault == "rounded_budget" else raw, raw + 8192)

    graph.allocator = NS(bounds=bound, unchanged=lambda torch: fault != "allocator_change")
    if fault == "physical_spare":
        graph.owner.spare = lambda: raw
    assert graph._slot(st, capture=True) is None
    assert graph._slot(st, capture=True) is None
    assert len(votes) == len(bounds_calls) == len(graph.declined) == 1 and not votes[0]
    assert not graph.slots
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    graph.invalidate(st.target)
    assert not graph.declined


@pytest.mark.parametrize("failure", (False, True))
def test_hot_replay_never_accounts_unchanged_memory_or_votes_and_releases_busy_after_failure(failure):
    p = provider()
    st = p.make_state(owner(), capacity=32)
    graph = bare_graph_owner(p, st)
    called = []

    def forbidden(*args):
        raise AssertionError("hot hit must not inspect allocator or collect an admission vote")

    def finish():
        called.append("one-readback")
        if failure:
            raise ValueError("nonfinite")
        return [4, 7, 19]

    graph.bound._account = graph.bound._agree = forbidden
    ws = NS(prepare=lambda bonus: called.append(("prepare", bonus)), busy=False, keys=st.keys, values=st.values,
            pieces=NS(graphs=[NS(replay=lambda: called.append("replay"))]), finish=finish)
    graph.slots[ProposalWorkspace.key(p, st, 3, 32)] = ws
    if failure:
        with pytest.raises(ValueError, match="nonfinite"):
            graph.draft(st.target, 4, 3, None, 0)
    else:
        assert graph.draft(st.target, 4, 3, None, 0) == [4, 7, 19]
    assert not ws.busy and called == [("prepare", 4), "replay", "one-readback"]
    assert st.length == st.target.pos == 0


def test_measured_graph_pool_growth_cannot_consume_required_future_service_memory():
    root = NS(room_bytes=100, budget=1000, spare=lambda: -1)
    # The historical budget-only condition admits this capture despite its
    # driver/private-pool growth consuming a promised future cache allocation.
    assert root.room_bytes + 50 <= root.budget
    assert not capture_fits(root, 50)
    root.spare = lambda: 0
    assert capture_fits(root, 50)
    root.spare = None
    assert capture_fits(root, 900)
    assert not capture_fits(root, 901)
