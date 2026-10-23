"""Small real FP8 graph/eager differentials; no full-model/performance inference."""
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')

from test_gemma_cuda_dflash import DEV, make_fixture, taps_for

from tensorfold.cuda.sampling import Sampling
from tensorfold.cuda.speculative import Chains
from tensorfold.families.gemma4.cuda import body_graphs, dflash
from tensorfold.families.gemma4.cuda.body_graphs import DraftBodyGraphs

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='real proposal capture requires CUDA')


@pytest.fixture
def real_lane(tmp_path, request, monkeypatch):
    if torch.cuda.memory.get_allocator_backend() != 'native':
        pytest.skip('this private graph investigation uses the native allocator')
    cfg, _arrays, target = make_fixture(tmp_path, window=8)
    # The fixture's original Torch embedding/head oracle starts on CPU. Move
    # those immutable inputs once before capture, never inside a graph.
    target.embed, target.head = target.embed.to(DEV), target.head.to(DEV)
    target.weights = NS(embed=NS(weight=target.head))
    target.config.softcap = 30.0
    model = dflash.DFlash(tmp_path, target, context_limit=2048, bits=8, block=5, rows=128)
    eager, staged = model.make_state(), model.make_state()
    taps = taps_for(cfg, 1032, 713).to(DEV)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=DEV)
    with torch.cuda.stream(stream):
        owner = DraftBodyGraphs(model, staged, stream)
    actual_logits, expected_miss = [], [None]
    ordinary_chains = body_graphs.Chains

    def observed_chains(logits, *args, **kwargs):
        # Functional observation of the actual selected first/hit result,
        # after capture has ended. Never inspect uninitialized static output
        # on a miss. Observe the actual selected source, without running an
        # extra warmup, forcing a replay or substituting an algorithm.
        assert not torch.cuda.is_current_stream_capturing()
        captured = kwargs['owners'].output
        if expected_miss[0]:
            assert logits is not captured and logits.data_ptr() != captured.data_ptr()
        else:
            assert logits is captured
        actual_logits.append(logits.cpu())
        return ordinary_chains(logits, *args, **kwargs)

    monkeypatch.setattr(body_graphs, 'Chains', observed_chains)

    def retire():
        # Pytest owns the independent teardown outcome. A failed drain never
        # authorizes graph/tensor/pool release.
        stream.synchronize()
        owner.close_after_drain()

    request.addfinalizer(retire)
    return NS(model=model, eager=eager, staged=staged, owner=owner, stream=stream,
              taps=taps, actual_logits=actual_logits, expected_miss=expected_miss)


def append_pair(lane, begin, end, *, prompt=True):
    for at in range(begin, end, 128):
        rows = lane.taps[at:min(end, at+128)]
        lane.model.append(lane.eager, rows, prompt=prompt)
        lane.model.append(lane.staged, rows, prompt=prompt)


def compare(lane, anchor, count, *, sampled=False):
    reference = lane.model.logits(lane.eager, anchor, count)
    kept = reference.clone()
    width_index = reference.shape[0]-1
    entry = lane.owner.entries[width_index]
    full_slots = (lane.staged.length + reference.shape[0]) // dflash.FULL_CHUNK + 1
    missing = entry is None or entry.full_slots != full_slots
    lane.expected_miss[0] = missing
    sampling = Sampling(temperature=1.0, top_k=8, top_p=1.0, min_p=0.0, seed=17) if sampled else None
    chains = lane.owner.propose(lane.staged, anchor, count, sampled=sampled, sampling=sampling)
    assert chains.host is not None and chains.owners is None and chains.words is None and chains._source is None
    rows = reference.shape[0]
    # The constructor observer copied the actual selected source to CPU.
    # Captured static storage is deliberately unread on a miss.
    got = lane.actual_logits.pop()
    assert torch.equal(reference.cpu(), got)
    assert lane.owner.entries[width_index].output_initialized is (not missing)
    assert torch.equal(reference, kept)  # the eager public result remains fresh
    assert lane.eager.length == lane.staged.length
    for left, right in zip(lane.eager.snapshot()[1], lane.staged.snapshot()[1], strict=True):
        assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
    if sampled:
        packet = chains.sample(0, count, sampling, lane.eager.length+1)
        expected = Chains(reference, [None], rows, sampled=True, sampling=sampling)
        expected = expected.sample(0, count, sampling, lane.eager.length+1)
        assert packet == expected
    else:
        assert chains.finish(0, count)[0] == reference.argmax(-1).tolist()


@cuda
@torch.inference_mode()
def test_fp8_body_graphs_keep_full_logits_kv_across_frontiers_restore_and_tails(real_lane):
    lane = real_lane
    with torch.cuda.stream(lane.stream):
        append_pair(lane, 0, 1018)
        earlier = lane.eager.snapshot(), lane.staged.snapshot()
        for position in (1018, 1019, 1020):  # ends1023,1024,1025 for actualwidth5
            if position > lane.eager.length:
                append_pair(lane, lane.eager.length, position)
            lane.owner.query.fill_(float('nan'))
            compare(lane, 2+(position % 3), 4)
        assert lane.owner.captures == 2  # exactgrid1 reused, then exactgrid2
        lane.eager.restore(earlier[0])
        lane.staged.restore(earlier[1])
        compare(lane, 5, 4)
        assert lane.owner.captures == 3 and lane.owner.entries[3].full_slots == 1
        for count in (1, 2, 3):
            compare(lane, 3, count)
            lane.owner.query.fill_(float('nan'))
            compare(lane, 4, count)  # a real replay, not only each width's first warm result
        assert len(lane.owner.entries) == 4 and all(entry is not None for entry in lane.owner.entries)


@cuda
@torch.inference_mode()
def test_fp8_graph_sampled_readback_and_new_state_have_distinct_owners(real_lane):
    lane = real_lane
    with torch.cuda.stream(lane.stream):
        append_pair(lane, 0, 37)
        compare(lane, 2, 4, sampled=True)
        compare(lane, 3, 4)
        compare(lane, 4, 4, sampled=True)
        clone_state = lane.model.make_state()
        clone_state.restore(lane.staged.snapshot())
        other = DraftBodyGraphs(lane.model, clone_state, lane.stream)
        with pytest.raises(ValueError, match='one model and DraftState'):
            lane.owner.propose(clone_state, 2, 4)
        assert other.ids.data_ptr() != lane.owner.ids.data_ptr()
        assert other.query.data_ptr() != lane.owner.query.data_ptr()
        lane.expected_miss[0] = True
        other.propose(clone_state, 2, 4)
        actual = lane.actual_logits.pop()
        expected = lane.model.logits(clone_state, 2, 4).cpu()
        assert torch.equal(actual, expected) and other.entries[3].output_initialized is False
        lane.stream.synchronize()
        other.close_after_drain()
        assert all(entry is None for entry in other.entries)  # retire a capture never replayed
