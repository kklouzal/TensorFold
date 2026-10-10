"""CPU replay of the unchanged CUDA nucleus/distributed arithmetic.

The private namespace replaces only CUDA device admission with explicit CPU
metadata admission. Real CUDA validation stays intact; native tests separately
exercise the production device path.
"""

from pathlib import Path
import threading
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import sampling as cuda_sampling  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling, choose_rows  # noqa: E402
from tensorfold.thread_work import ThreadWork  # noqa: E402


def _cpu_rows(logits, positions, sampling):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.is_cuda
            or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or logits.shape[1] <= 0 or len(positions) != logits.shape[0]):
        raise ValueError("CPU nucleus replay requires floating CPU rows and matching positions")
    cuda_sampling.validate_policy(positions, sampling)


cs = ModuleType("_cpu_nucleus_arithmetic_replay")
cs.__file__ = cuda_sampling.__file__
exec(compile(Path(cs.__file__).read_bytes(), cs.__file__, "exec"), vars(cs))
cs.validate_rows = _cpu_rows


class Ranks:
    """Two ranks as threads: each ``gather`` stacks both ranks' words, rank 0 first."""

    def __init__(self):
        self.slots, self.barrier = [None, None], threading.Barrier(2, timeout=60)

    def gather(self, rank):
        def gather(words):
            self.slots[rank] = words.clone()
            self.barrier.wait()
            both = torch.stack(self.slots)
            self.barrier.wait()
            return both

        return gather


def two_ranks(logits, positions, sampling, split, probs=None):
    ranks, out = Ranks(), [None, None]
    errors = [None, None]
    shards = (logits[:, :split], logits[:, split:])

    def run(r):
        try:
            out[r] = cs.nucleus_rows(shards[r], positions, sampling, offset=0 if r == 0 else split,
                                     gather=ranks.gather(r), probs=probs if r == 0 else None)
        except BaseException as error:
            errors[r] = error
            ranks.barrier.abort()

    # Publish every callback owner before launch. An interrupted native start
    # can later run only a cancelled empty controller, never borrowed tensors.
    work = [ThreadWork(lambda r=r: run(r)) for r in (0, 1)]
    threads = [threading.Thread(target=owner.run) for owner in work]
    primary = None
    cleanup = []
    try:
        for owner, thread in zip(work, threads):
            owner.launch(thread)
    except BaseException as error:
        primary = error
        ranks.barrier.abort()
    finally:
        for owner, thread in zip(work, threads):
            try:
                owner.drain(thread, timeout=60, cancel_unentered=primary is not None)
            except BaseException as error:
                cleanup.append(error)
    failures = [error for error in errors if error is not None]
    failures.extend(cleanup)
    if primary is not None:
        failures.insert(0, primary)
    if failures:
        raise BaseExceptionGroup("CPU nucleus rank failures", failures)
    assert out[0] == out[1]                                       # every rank draws the same tokens
    return out[0]


def _logits(seed, rows=6, vocab=6000, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, vocab, generator=g) * scale).to(torch.bfloat16)


def test_cpu_replay_keeps_production_cuda_validation_and_policy_validation():
    logits = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="floating CUDA logits"):
        cuda_sampling.nucleus_rows(logits, [1, 2], Sampling(3, 1.0, 0, 1.0))
    with pytest.raises(ValueError, match="unsigned64"):
        cs.nucleus_rows(logits, [True, 2], Sampling(3, 1.0, 0, 1.0))
    with pytest.raises(ValueError, match="floating CPU"):
        cs.nucleus_rows(logits.to(torch.int64), [1, 2], Sampling(3, 1.0, 0, 1.0))
    assert cuda_sampling.validate_rows is not cs.validate_rows


def test_rank_failures_are_drained_and_reported(monkeypatch):
    primary = OSError("rank zero failed before its collective")
    original = cs.nucleus_rows

    def fail(logits, positions, sampling, **kwargs):
        if kwargs["offset"] == 0:
            raise primary
        return original(logits, positions, sampling, **kwargs)

    monkeypatch.setattr(cs, "nucleus_rows", fail)
    with pytest.raises(BaseExceptionGroup) as caught:
        two_ranks(torch.zeros(2, 4), [1, 2], Sampling(3, 1.0, 0, 1.0), 2)
    assert any(error is primary for error in caught.value.exceptions)
    assert any(isinstance(error, threading.BrokenBarrierError) for error in caught.value.exceptions)


def test_partial_rank_start_failure_drains_the_started_rank(monkeypatch):
    primary = RuntimeError("rank one could not start")
    original = threading.Thread.start
    started = []

    def start(thread):
        if started:
            raise primary
        original(thread)
        started.append(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(BaseExceptionGroup) as caught:
        two_ranks(torch.zeros(2, 4), [1, 2], Sampling(3, 1.0, 0, 1.0), 2)
    assert caught.value.exceptions[0] is primary
    assert len(started) == 1 and not started[0].is_alive()


def test_rank_start_acceptance_interruption_still_joins_owned_rank(monkeypatch):
    primary = KeyboardInterrupt("accepted rank start interrupted before caller return")
    original = threading.Thread.start
    accepted = []

    def start(thread):
        original(thread)
        accepted.append(thread)
        raise primary

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(BaseExceptionGroup) as caught:
        two_ranks(torch.zeros(2, 4), [1, 2], Sampling(3, 1.0, 0, 1.0), 2)
    assert caught.value.exceptions[0] is primary
    assert len(accepted) == 1 and not accepted[0].is_alive()


@pytest.mark.parametrize("top_p", [0.8, 0.95, 1.0])
@pytest.mark.parametrize("min_p", [0.0, 0.05])
def test_two_ranks_draw_what_one_rank_draws(top_p, min_p):
    for seed in range(12):
        logits = _logits(seed)
        s = Sampling(seed * 7919 + 3, 0.9, 0, top_p, min_p)
        positions = [100 + 17 * seed + r for r in range(logits.shape[0])]
        one = cs.nucleus_rows(logits, positions, s)
        assert two_ranks(logits, positions, s, 3000) == one
        assert two_ranks(logits, positions, s, 1234) == one          # shards of any size


def test_rows_past_the_candidates_read_whole_shards_and_still_agree(monkeypatch):
    flat = torch.zeros(3, 5000, dtype=torch.bfloat16)                  # every token tied: the nucleus is all of them
    flat[1, ::7] = 0.5
    reads = []
    real = cs._shares
    monkeypatch.setattr(cs, "_shares", lambda *a: reads.append(a[3]) or real(*a))
    for top_p, min_p in ((0.95, 0.0), (1.0, 0.0), (1.0, 0.3), (0.5, 0.2)):
        s = Sampling(5, 1.0, 0, top_p, min_p)
        one = cs.nucleus_rows(flat, [7, 8, 9], s)
        assert two_ranks(flat, [7, 8, 9], s, 2500) == one
    assert 5000 in reads and 2500 in reads                             # whole rows and whole shards were read


def test_the_same_distribution_as_the_float_rule():
    agree = total = 0
    for seed in range(20):
        logits = _logits(seed, rows=8, vocab=4000)
        s = Sampling(seed + 1, 1.0, 0, 0.9, 0.0)
        positions = list(range(8))
        values = logits.float().numpy()
        ids = np.broadcast_to(np.arange(4000, dtype=np.int64), values.shape)
        want = choose_rows(values, ids, positions, s)
        got = cs.nucleus_rows(logits, positions, s)
        agree += sum(a == b for a, b in zip(got, want))
        total += len(want)
    assert agree >= 0.99 * total, (agree, total)                       # only a cut's last token can differ


def test_the_drawn_tokens_share_of_the_mass():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    probs = []
    token = cs.nucleus_rows(logits, [3], Sampling(9, 1.0, 0, 1.0, 0.0), probs=probs)[0]
    assert probs == pytest.approx([float(torch.softmax(logits[0].double(), -1)[token])], rel=1e-9)


def test_streams_with_top_k_off_draw_by_the_nucleus_rule():
    logits = _logits(3, rows=4, vocab=3000)
    s = Sampling(11, 0.8, 0, 0.9, 0.02)
    want = cs.nucleus_rows(logits, [1, 2, 3, 4], s)
    keyed = Sampling(4, 1.0, 20, 0.95)
    got = cs.sample_streams(logits, [0, 2, 3, 4], [[1, 2], [3], [4]], [s, keyed, s])
    assert got[0] == want[:2] and got[2] == want[3:]
