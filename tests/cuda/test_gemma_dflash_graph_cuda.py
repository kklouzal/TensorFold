"""Exact eager/graph FP8 DFlash logits, q, restore and fresh-return ownership."""

import json
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')

from test_gemma_cuda_dflash import make_fixture, taps_for  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.gemma4.cuda.dflash import DFlash  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='native CUDA graph qualification')


def test_fp8_logits_graph_replays_exactly_across_restore_geometry_and_context_tail(tmp_path):
    directory = tmp_path / 'draft'
    directory.mkdir()
    cfg, _, target = make_fixture(directory, window=32)
    target.embed, target.head = target.embed.to(target.device), target.head.to(target.device)
    target.weights, target.config.softcap = NS(embed=NS(weight=target.head)), 30.0
    eager = DFlash(directory, target, context_limit=2048, bits=8, rows=128, block=5)
    graph = DFlash(directory, target, context_limit=2048, bits=8, rows=128, block=5, dflash_graph=True)
    left, right = eager.make_state(), graph.make_state()
    taps = taps_for(cfg, 2048, 933).to(target.device)

    def append(end):
        assert left.length == right.length
        for begin in range(left.length, end, 128):
            eager.append(left, taps[begin:min(end, begin + 128)], prompt=True)
            graph.append(right, taps[begin:min(end, begin + 128)], prompt=True)

    def compare(anchor, count):
        before = right.length
        expected, got = eager.logits(left, anchor, count), graph.logits(right, anchor, count)
        assert right.length == left.length == before
        if expected is None:
            assert got is None
        else:
            assert torch.equal(expected, got)
            assert got.data_ptr() != graph._logits_graph.output.data_ptr()
        return got

    append(1015)
    saved_left, saved_right = left.snapshot(), right.snapshot()
    torch.cuda.synchronize()
    memory_before = {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}
    first = compare(2, 4)
    held = first.clone()
    compare(3, 4)  # same-key replay must leave the previous owning result intact
    assert torch.equal(first, held)
    same_graph = graph._logits_graph.graph
    append(1016)
    advanced = compare(2, 4)
    assert graph._logits_graph.graph is same_graph
    assert not torch.equal(advanced, held)  # a stale captured position must fail this oracle
    left.restore(saved_left)
    right.restore(saved_right)
    restored = compare(2, 4)
    assert graph._logits_graph.graph is same_graph
    assert torch.equal(restored, held) and torch.equal(first, held)
    del same_graph  # do not retain an obsolete pool across later shape transitions
    append(1019)
    saved_left, saved_right = left.snapshot(), right.snapshot()
    append(1020)
    compare(4, 4)  # exact full-attention slot transition1 ->2
    compare(5, 2)
    compare(6, 1)
    left.restore(saved_left)
    right.restore(saved_right)
    compare(2, 4)  # restore recaptures the old exact shape, without stale positions
    policy = Sampling(71, .7, 8, .95)
    expected = eager.propose(left, 2, 4, sampled=True, sampling=policy).sample(0, 4, policy, left.length + 1)
    launched = graph.propose(right, 2, 4, sampled=True, sampling=policy)
    compare(3, 2)  # readback/full-row repair may still own the previous logits
    assert launched.sample(0, 4, policy, right.length + 1) == expected
    # Rejected tentative noise cannot become committed context after restore.
    left.restore(saved_left)
    right.restore(saved_right)
    append(2043)
    for end in range(2043, 2048):
        append(end)
        compare(2, 4)  # actual context capacity yields L5,4,3,2,then no proposal
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(target.device)
    (tmp_path / 'graph-owner-readback.json').write_text(json.dumps({
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': properties.name, 'capability': [properties.major, properties.minor],
        'before_graph_allocated_reserved': memory_before,
        'after_graph_allocated_reserved': {'allocated': torch.cuda.memory_allocated(),
                                         'reserved': torch.cuda.memory_reserved()},
        'scope': 'tiny native FP8 component; no full-model or physical UMA residency claim',
    }, indent=2) + '\n')
    graph._logits_graph.close()  # actual caller stream is terminal above
    assert graph._logits_graph.graph is graph._logits_graph.pool is None
