"""Teacher scoring combines vocabulary-shard packets into the full-vocabulary log-softmax and top alternatives."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import teacher_scoring  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="teacher scoring reads CUDA head logits")


class _Captured(Exception):
    pass


class _Comm:
    """all_gather over precomputed rank packets; ``capture`` records this rank's packet and stops."""

    def __init__(self, world, rank, packets=None):
        self.world, self.rank, self.packets, self.sent = world, rank, packets, None

    def all_gather(self, flat, got):
        if self.packets is None:
            self.sent = flat.clone()
            raise _Captured
        parts = list(self.packets)
        parts[self.rank] = flat
        got.copy_(torch.cat(parts))


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("top", [0, 5])
def test_shard_packets_combine_to_the_full_log_softmax(world, top):
    generator = torch.Generator(device="cuda").manual_seed(31 + world + top)
    rows, vocab = 6, 4096
    full = torch.randn(rows, vocab, device="cuda", generator=generator) * 4
    full[1, 7] = 40.0                                        # one dominant logit
    targets = [7, 4095, 0, 2048, 1234, 3000]
    width = vocab // world
    shards = [full[:, r * width:(r + 1) * width].contiguous() for r in range(world)]

    def weights(rank, comm):
        return SimpleNamespace(comm=comm, meta={"vocab_offset": rank * width})

    if world == 1:
        results = [teacher_scoring.shard_logprobs(SimpleNamespace(comm=None, meta={}), full, targets, top)]
    else:
        packets = []
        for rank in range(world):
            comm = _Comm(world, rank)
            with pytest.raises(_Captured):
                teacher_scoring.shard_logprobs(weights(rank, comm), shards[rank], targets, top)
            packets.append(comm.sent)
        results = [teacher_scoring.shard_logprobs(weights(rank, _Comm(world, rank, packets)), shards[rank],
                                                  targets, top) for rank in range(world)]
    assert all(result == results[0] for result in results)  # every rank returns the same values
    chosen, ids, scores = results[0]
    expected = torch.log_softmax(full.double(), dim=1)
    want = expected[torch.arange(rows), torch.tensor(targets)].tolist()
    assert max(abs(a - b) for a, b in zip(chosen, want)) < 1e-5
    if top:
        values, columns = expected.topk(top, dim=1)
        assert ids == columns.tolist()
        assert max(abs(a - b) for row, ref in zip(scores, values.tolist()) for a, b in zip(row, ref)) < 1e-5
    else:
        assert ids == [[] for _ in range(rows)] and scores == ids
