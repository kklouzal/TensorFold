"""Stored GGUF negative decay is a distinct, directly consumed representation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import gdn
from tensorfold.families.qwen4_exp.cuda.gguf_weights import ArrayPlan, _array
from tensorfold.families.qwen4_exp.cuda.weight_types import GDNW
from tensorfold.families.qwen4_exp.gguf import PackedTensor
from gguf_fixture import tiny_model


class RowReader:
    def read(self, path, offset, length, device):
        assert device == "cpu"
        with Path(path).open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(length)
        return torch.from_numpy(np.frombuffer(raw, np.uint8).copy())


def test_import_preserves_stored_fp32_negative_decay_and_head_permutation(tmp_path):
    source = np.array([-.1, -.12345, -.37, -1.1, -3.25, -17.3], dtype="<f4")
    path = tmp_path / "decay.raw"
    path.write_bytes(source.tobytes())
    descriptor = PackedTensor("blk.0.ssm_a", source.shape, 0, path, 0, source.nbytes)
    permutation = (5, 0, 4, 2, 1, 3)
    plan = ArrayPlan(descriptor, permutation, transform="negative_exp")
    result = _array(plan, RowReader(), "cpu").numpy()
    np.testing.assert_array_equal(result.view(np.uint32), source[list(permutation)].view(np.uint32))
    # Retain a witness of the removed lossy inverse transform; it is not an oracle.
    assert np.any(np.exp(np.log(-source)).view(np.uint32) != (-source).view(np.uint32))


@pytest.mark.parametrize("invalid", [0., -0., 1., np.nan, np.inf, -np.inf])
def test_invalid_stored_decay_is_rejected_at_import(tmp_path, invalid):
    values = np.array([-1., invalid], dtype="<f4")
    path = tmp_path / "bad.raw"
    path.write_bytes(values.tobytes())
    source = PackedTensor("ssm_a", values.shape, 0, path, 0, values.nbytes)
    with pytest.raises(ValueError, match="finite negative"):
        _array(ArrayPlan(source, transform="negative_exp"), RowReader(), "cpu")


@pytest.mark.parametrize("direct", [False, True])
def test_zero_key_chain_has_analytic_decay_and_exact_replay_for_every_prefix(direct):
    rows, nk, nv = 4, 2, 6
    channels, width = gdn.widths(nk, nv)
    projection = torch.zeros(rows, width, dtype=torch.float16)
    history = torch.zeros(3, channels, dtype=torch.float16)
    conv = torch.zeros(channels, 4, dtype=torch.float16)
    initial = torch.arange(nv * 128 * 128).reshape(nv, 128, 128).float() * 2.**-20
    rates = -torch.arange(1, nv + 1).float()
    logs = torch.log(-rates)
    scratch = gdn.GDNScratch(rows, "cpu", nk, nv, dtype=torch.float16)
    final = torch.empty_like(initial)
    out, sums = torch.empty(rows, nv * 128, dtype=torch.float16), torch.empty(rows, nv * 4)
    gdn.chain(projection, history, conv, initial, None if direct else logs, torch.zeros(nv),
              torch.ones(128, dtype=torch.float16), 1e-6, rows, scratch, final, out, sums,
              a_neg=rates if direct else None)
    # Zero alpha/bias => softplus(0)=ln2; zero keys make every delta update zero.
    ln2 = torch.tensor(np.float32(np.log(2.)))
    rate = rates if direct else -torch.exp(logs)
    expected_decay = torch.exp(rate * ln2)
    torch.testing.assert_close(scratch.g, expected_decay.expand(rows, nv), rtol=0, atol=0)
    expected = initial.clone()
    for kept in range(rows + 1):
        replayed = torch.empty_like(initial)
        gdn.replay(initial, scratch, kept, replayed)
        torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
        if kept < rows:
            expected = expected * expected_decay[:, None, None]
    torch.testing.assert_close(final, expected, rtol=0, atol=0)
    assert torch.count_nonzero(out) == 0


def test_weight_graph_requires_one_decay_variant_and_preserves_hf_field_contract():
    log, negative = torch.zeros(6), -torch.ones(6)
    hf = GDNW(None, None, log, None, None, None)
    gguf = GDNW(None, None, None, None, None, None, a_neg=negative)
    assert hf.a_log is log and hf.a_neg is None
    assert gguf.a_log is None and gguf.a_neg is negative
    for first, second in ((None, None), (log, negative)):
        with pytest.raises(ValueError, match="exactly one"):
            GDNW(None, None, first, None, None, None, a_neg=second)


def test_invalid_representation_cannot_enter_native_or_modern_extension():
    for log, negative, dtype in ((None, None, torch.float16), (torch.zeros(6), -torch.ones(6), torch.float16),
                                  (None, -torch.ones(6), torch.bfloat16)):
        with pytest.raises(ValueError, match="exactly one|FP16 provider"):
            gdn.chain(torch.empty(0, dtype=dtype), None, None, None, log, None, None, 1e-6, 0,
                      None, None, None, None, a_neg=negative)


@pytest.mark.parametrize("world", (1, 2, 4, 8))
def test_all_coupled_recurrent_parts_restore_three_value_heads_before_tp(tmp_path, world):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan

    path = tiny_model(tmp_path / "three.gguf", linear_heads=(16, 48), linear_head_dim=128)
    # Source order is [V subhead, K group], restored order [K group, V subhead].
    global_values = np.arange(48).reshape(3, 16).T.reshape(-1)

    def expanded(ranges):
        return [i for start, count in ranges for i in range(start, start + count)]

    for rank in range(world):
        weights, _ = model_plan(path, (rank, world))
        recurrent = weights.layers[0].gdn
        heads = global_values[rank * (48 // world):(rank + 1) * (48 // world)].tolist()
        qrows = list(range(rank * (16 // world) * 128, (rank + 1) * (16 // world) * 128))
        vrows = [head * 128 + offset for head in heads for offset in range(128)]
        channels = qrows + [16 * 128 + i for i in qrows] + [2 * 16 * 128 + i for i in vrows]
        qkv, gate, beta, alpha = recurrent.proj.parts
        assert expanded(qkv.row_ranges) == channels
        assert expanded(gate.row_ranges) == vrows
        assert expanded(beta.row_ranges) == heads == expanded(alpha.row_ranges)
        assert list(recurrent.a_neg.indices) == heads == list(recurrent.dt_bias.indices)
        assert list(recurrent.conv.indices) == channels
        assert recurrent.out.columns == tuple((head * 128, 128) for head in heads)
        assert recurrent.a_log is None
