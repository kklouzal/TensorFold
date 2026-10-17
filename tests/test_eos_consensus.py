"""Exact EOS control protocol on CPU tensors; no claim of native multi-GPU execution."""
import threading

import pytest
import torch

from tensorfold.cuda.comm import agree_eos


def participants(stops):
    world = len(stops)
    barrier, values, outcomes, calls = threading.Barrier(world, timeout=3), [None] * world, [None] * world, [0] * world
    class Comm:
        def __init__(self, rank):
            self.rank = rank
        def all_gather(self, source, destination):
            calls[self.rank] += 1
            values[self.rank] = source.clone()
            barrier.wait()
            destination.copy_(torch.cat(values))
            barrier.wait()
    def run(rank):
        try:
            agree_eos(Comm(rank), stops[rank], world=world, device="cpu")
        except BaseException as error:
            outcomes[rank] = error
    threads = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert all(not thread.is_alive() for thread in threads), "EOS participants failed to converge"
    return outcomes, calls


@pytest.mark.parametrize("stops", [[(0,)], [(3, 1, 3), (1, 3)], [(0, 2)] * 4])
def test_exact_membership_ignores_order_and_duplicate_declarations(stops):
    outcomes, calls = participants(stops)
    assert outcomes == [None] * len(stops) and calls == [2] * len(stops)


@pytest.mark.parametrize("stops", [[(1,), (1, 2)], [(1, 2), (1, 3)], [(0,)] * 3 + [(1,)]])
def test_every_participant_refuses_different_effective_stop_sets(stops):
    outcomes, _ = participants(stops)
    assert all(isinstance(error, RuntimeError) and "different effective EOS" in str(error) for error in outcomes)


@pytest.mark.parametrize("invalid", [(), (False,), (-1,), (2**63,), ("1",)])
def test_invalid_native_ids_are_reported_by_all_ranks_before_any_payload(invalid):
    outcomes, calls = participants([(1,), invalid])
    assert all(isinstance(error, ValueError) and "invalid native EOS" in str(error) for error in outcomes)
    assert calls == [1, 1]
