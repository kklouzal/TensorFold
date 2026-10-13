"""The CUDA target sampler is invariant to the other rows in a verify window."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.cuda.sampling import sample_rows


@pytest.mark.parametrize("sampling", [None, Sampling(seed=7382, top_k=20, top_p=0.95),
                                      Sampling(seed=7382, top_k=20, top_p=0.95, min_p=0.1)])
def test_rows_match_single_row(sampling):
    torch.manual_seed(841)
    logits = torch.randn((128, 4096), device="cuda", dtype=torch.bfloat16)
    logits[:, 2:6] = 4.0  # tied leaders exercise token-ID tie handling
    positions = list(range(312, 440))
    wide = sample_rows(logits, positions, sampling)
    alone = [sample_rows(logits[i:i + 1], positions[i:i + 1], sampling)[0]
             for i in range(128)]
    assert wide == alone


def test_streams_match_their_own_calls():
    from tensorfold.cuda.sampling import sample_streams

    torch.manual_seed(5)
    vocab = 248320
    # bf16 logits rounded to a coarse grid: many ties at every candidate boundary
    logits = (torch.randn((520, vocab), device="cuda") * 2).mul(4).round().div(4).to(torch.bfloat16)
    logits[:, 100:140] = 6.0
    samplings = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(99, 0.7, 20, 1.0, 0.2), Sampling(5, 1.0, 40, 0.9),
                 None, Sampling(8, 1.0, 20, 0.95, 0.05)] * 5
    sizes = [16, 12, 1, 16, 3, 7] * 5
    starts = [0]
    for n in sizes[:len(samplings)]:
        starts.append(starts[-1] + n)
    positions = [list(range(50 + a, 50 + b)) for a, b in zip(starts, starts[1:])]
    got = sample_streams(logits, starts, positions, samplings)
    for s, smp in enumerate(samplings):
        want = sample_rows(logits[starts[s]:starts[s + 1]], positions[s], smp)
        assert got[s] == want, s


def test_queued_streams_match_sample_streams():
    """``launch_streams`` then ``collect_streams`` (one host wait, top_k-off streams drawn after it) choose what
    ``sample_streams`` does, ties and top_k-off rows included; a readback's check runs after the wait, first."""

    from tensorfold.cuda.sampling import collect_streams, launch_streams, sample_streams

    torch.manual_seed(6)
    vocab = 248320
    logits = (torch.randn((40, vocab), device="cuda") * 2).mul(4).round().div(4)
    logits[:, 100:140] = 6.0
    samplings = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(77, 1.0, 0, 0.9), Sampling(5, 1.0, 40, 0.9),
                 None, Sampling(8, 1.0, 20, 0.95, 0.05), Sampling(9, 0.8, 0, 1.0, 0.1)]
    sizes = [5, 4, 3, 1, 0, 7, 2]
    starts = [0]
    for n in sizes:
        starts.append(starts[-1] + n)
    positions = [list(range(50 + a, 50 + b)) for a, b in zip(starts, starts[1:])]
    order = []

    class Readback:
        def queue(self):
            order.append("queue")

        def check(self):
            order.append("check")

    want = sample_streams(logits, starts, positions, samplings)
    got = collect_streams(launch_streams(logits, starts, positions, samplings), Readback(),
                          lambda: order.append("waited"))
    assert got == want and order == ["queue", "waited", "check"]
    bad = logits.clone()
    bad[starts[5] + 2, 9] = float("nan")
    with pytest.raises(ValueError, match="NaN or positive infinity"):
        collect_streams(launch_streams(bad, starts, positions, samplings))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lean_flags_equal_score_flags_on_every_invalid_domain(dtype):
    """``score_flags_lean`` (one max pass) marks exactly the rows ``score_flags`` does: NaN anywhere (first, last,
    across reduction blocks, beside +inf), +inf, and never finite or all-masked (-inf) rows."""

    from tensorfold.cuda.sampling import score_flags, score_flags_lean

    torch.manual_seed(3)
    width = 248320
    rows = torch.randn((24, width), device="cuda").to(dtype)
    nan, inf = float("nan"), float("inf")
    rows[1, 0] = nan
    rows[2, width - 1] = nan
    rows[3, 131072] = nan
    rows[4, 7] = inf
    rows[5, width - 1] = inf
    rows[6, 9], rows[6, 200000] = inf, nan
    rows[7, 9], rows[7, 200000] = nan, inf
    rows[8] = -inf
    rows[9, :] = -inf
    rows[9, 5] = 3.0
    rows[10, ::2] = nan
    rows[11, 100:] = inf
    rows[12, 12345] = -inf
    rows[13] = 0.0
    rows[14, 77] = torch.finfo(dtype).max
    want = score_flags(rows)
    assert want[[1, 2, 3, 4, 5, 6, 7, 10, 11]].all() and not want[[0, 8, 9, 12, 13, 14]].any()
    assert torch.equal(score_flags_lean(rows), want)
    assert torch.equal(score_flags_lean(rows[3:12]), score_flags(rows[3:12]))


def test_lean_packets_equal_the_copied_ones():
    """B3: back-to-back members sample from a slice of the logits instead of a copy, with the one-pass flags: the
    same packets for contiguous, out-of-order and gapped (empty-stream) groups, and the same choices."""

    from tensorfold.cuda.sampling import _group_packet, _stream_groups, collect_streams, launch_streams, sample_streams

    torch.manual_seed(8)
    vocab = 248320
    logits = (torch.randn((30, vocab), device="cuda") * 2).mul(4).round().div(4)
    logits[:, 100:140] = 6.0
    logits[17, 3] = float("nan")
    keyed = Sampling(1234, 1.0, 20, 0.95)
    samplings = [keyed, keyed, None, keyed, Sampling(5, 1.0, 20, 0.9, 0.05), None, keyed]
    sizes = [5, 4, 3, 0, 2, 6, 5]
    starts = [0]
    for n in sizes:
        starts.append(starts[-1] + n)
    positions = [list(range(50 + a, 50 + b)) for a, b in zip(starts, starts[1:])]
    _, _, groups = _stream_groups(logits, starts, positions, samplings, draw=False)
    assert sorted(groups) == [0, 28] and groups[28] == [0, 1, 4, 6]
    for count in groups:                              # back to back, across the empty stream 3, apart, reversed
        for chosen in ([0, 1], [2, 4], [0, 1, 2], [1, 4, 6], [6, 1], [5]):
            old = _group_packet(logits, starts, count, chosen, False)
            new = _group_packet(logits, starts, count, chosen, True)
            assert torch.equal(old[3], new[3]) and torch.equal(old[2], new[2]), (count, chosen)
    clean = torch.cat([logits[:17], logits[18:19], logits[18:]])
    want = sample_streams(clean, starts, positions, samplings)
    assert sample_streams(clean, starts, positions, samplings, lean=True) == want
    assert collect_streams(launch_streams(clean, starts, positions, samplings, lean=True)) == want
    with pytest.raises(ValueError, match="NaN or positive infinity"):
        sample_streams(logits, starts, positions, samplings, lean=True)
