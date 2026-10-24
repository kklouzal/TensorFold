"""Private candidate packet/math and integrated owner tests; CUDA is not used."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.device_greedy import GreedyHandoff
from tensorfold.cuda.sm70 import greedy_native



SENTINEL = np.iinfo(np.int64).max


def maximum(bits, ids):
    bits = np.asarray(bits, dtype=np.uint32)
    ids = np.asarray(ids, dtype=np.int64)
    nan = (bits & np.uint32(0x7fffffff)) > np.uint32(0x7f800000)
    if nan.any():
        return int(bits[np.flatnonzero(nan)[0]]), SENTINEL
    keys = np.where((bits & np.uint32(0x80000000)) != 0, ~bits, bits | np.uint32(0x80000000))
    keys[(bits & np.uint32(0x7fffffff)) == 0] = np.uint32(0x80000000)
    selected = keys == keys.max()
    return int(bits[np.flatnonzero(selected)[0]]), int(ids[selected].min())


def packet(scores, ids, norm, status):
    bits, token = maximum(scores.numpy().view(np.uint32).reshape(-1), ids.numpy())
    signed = int(np.asarray(bits, dtype=np.uint32).view(np.int32))
    return np.asarray([signed, token, int(norm.view(-1).view(torch.int32)[0]), *status.tolist()],
                      dtype=np.int64)


def winner(frame, vocabulary):
    _bits, token = maximum(frame[:, 0].astype(np.int32).view(np.uint32), frame[:, 1])
    return token, token if 0 <= token < vocabulary and token <= np.iinfo(np.int32).max else 0


def inputs(width, case):
    values = torch.from_numpy(np.random.default_rng(width).normal(size=(1, width)).astype(np.float32))
    if case == "ties":
        values.zero_()
    elif case == "zeros":
        values.view(torch.int32).view(-1).copy_(torch.from_numpy(
            np.resize(np.asarray([0, 0x80000000], dtype=np.uint32).view(np.int32), width)))
    elif case == "subnormal":
        values.view(torch.int32).view(-1).copy_(torch.from_numpy(np.resize(
            np.asarray([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff], dtype=np.uint32).view(np.int32), width)))
    elif case == "adjacent":
        values.fill_(1)
        values[0, -1] = torch.nextafter(torch.tensor(1.), torch.tensor(2.))
    elif case == "minus_inf":
        values.fill_(-float("inf"))
    elif case == "plus_inf":
        values[0, width//2] = float("inf")
    elif case == "nan":
        values.view(torch.int32)[0, 0] = 0x7fc00001
        values.view(torch.int32)[0, -1] = 0x7fc00002
    ids = torch.arange(400, 400+width, dtype=torch.int64).flip(0)
    return values, ids, torch.tensor([-3.125], dtype=torch.float32), torch.tensor([4, -1], dtype=torch.int32)


def original_packet(values, ids, norm, status):
    maximum = values.max(-1).values
    tied = torch.where(values[0] == maximum, ids, torch.iinfo(torch.int64).max).min()
    return np.asarray([int(maximum.view(torch.int32)[0]), int(tied), int(norm.view(torch.int32)[0]),
                       *status.tolist()], dtype=np.int64)


@pytest.mark.parametrize("width", [1, 31, 32, 33, 9948, 9949, 31040])
@pytest.mark.parametrize("case", ["random", "ties", "zeros", "subnormal", "adjacent", "minus_inf", "plus_inf", "nan"])
def test_integer_key_packet_preserves_all_original_cpu_words(width, case):
    values, ids, norm, status = inputs(width, case)
    np.testing.assert_array_equal(packet(values, ids, norm, status), original_packet(values, ids, norm, status))


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("case", ["ties", "zeros", "subnormal", "minus_inf", "plus_inf", "nan"])
def test_global_key_order_and_safe_staging_match_original(world, case):
    values, ids, norm, status = inputs(world, case)
    frame = np.asarray([original_packet(values[:, rank:rank+1], ids[rank:rank+1], norm, status)
                        for rank in range(world)])
    maxima = torch.from_numpy(frame[:, 0].astype(np.int32)).view(torch.float32)
    expected = int(torch.where(maxima == maxima.max(), torch.from_numpy(frame[:, 1]),
                               torch.iinfo(torch.int64).max).min())
    assert winner(frame, 1000) == (expected, expected if 0 <= expected < 1000 else 0)


class CPUKernels:
    def local(self, values, ids, norm, status, output):
        output.copy_(torch.from_numpy(packet(values, ids, norm, status)))

    def global_winner(self, frame, token, staged, vocabulary):
        selected, safe = winner(frame.numpy(), vocabulary)
        token.fill_(selected)
        staged.fill_(safe)


def test_opt_in_native_epilogue_reuses_owners_packets_and_collective_order(monkeypatch):
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: original_empty(*args, **{
        key: value for key, value in kwargs.items() if key != "pin_memory"}))
    monkeypatch.setattr(greedy_native, "extension", lambda: CPUKernels())
    values, ids, norm, status = inputs(71, "ties")
    calls = []
    remote = torch.from_numpy(np.stack([original_packet(torch.tensor([[0.]]), torch.tensor([token]), norm,
                                                        torch.tensor([0, -1], dtype=torch.int32))
                                       for token in (7, 19, 5)]))

    def gather(send, receive):
        calls.append(send.clone())
        receive.view(4, 5)[0].copy_(send)
        receive.view(4, 5)[1:].copy_(remote)

    comm = NS(world=4, rank=0, all_gather=gather)
    plan = NS(width=71, device_map=ids, check_model=lambda *args: None)
    candidates = torch.zeros(4*65)
    candidates[64] = norm[0]
    dec = NS(w=NS(device="cpu", cfg=NS(vocab=1000), comm=comm, meta={}), depth=3,
             _draft_plan=plan, mbuf=NS(cand_all=candidates))
    previous = GreedyHandoff(NS(dec=dec))
    dec.w.meta["device_greedy_native_ready"] = True
    actual = GreedyHandoff(NS(dec=dec))
    pointers = {name: value.data_ptr() for name, value in vars(actual).items() if isinstance(value, torch.Tensor)}
    for step in range(3):
        previous.gather(step, values, NS(kv_status=status))
        actual.gather(step, values, NS(kv_status=status))
        torch.testing.assert_close(actual.frames[step], previous.frames[step], rtol=0, atol=0)
        torch.testing.assert_close(actual.tokens[step], previous.tokens[step], rtol=0, atol=0)
        torch.testing.assert_close(actual.input_token, previous.input_token, rtol=0, atol=0)
    assert len(calls) == 6
    assert pointers == {name: value.data_ptr() for name, value in vars(actual).items() if isinstance(value, torch.Tensor)}
    assert actual.device_bytes == previous.device_bytes and actual.host_bytes == previous.host_bytes
