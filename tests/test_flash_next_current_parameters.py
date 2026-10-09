"""Native current-descriptor projection/lookup contracts; requires actual Apple Metal."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("requires an actual Apple Metal runtime", allow_module_level=True)
import mlx.nn as nn  # noqa: E402

from tensorfold.families.qwen4_exp import decode  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import lane_qmm  # noqa: E402
from tensorfold.kernels.qwen.flash_next.v1 import embed  # noqa: E402


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_lane_reads_current_same_descriptor_and_replaced_scale_bias(bits):
    if not lane_qmm.reads(bits, 32):
        pytest.skip("the declared lane format is not available on this target")
    mx.random.seed(17)
    linear = nn.QuantizedLinear(256, 64, bias=False, group_size=32, bits=bits)
    x = (mx.random.normal((3, 256)) * .05).astype(mx.bfloat16)
    decode._lane_project(x, linear)
    original = linear.weight
    linear.weight[:, :] = mx.bitwise_xor(linear.weight, mx.array(17, dtype=mx.uint32))
    assert linear.weight is original
    linear.scales = (linear.scales.astype(mx.float32) * 1.25).astype(mx.bfloat16)
    linear.biases = (linear.biases.astype(mx.float32) + .01).astype(mx.bfloat16)
    actual = decode._lane_project(x, linear)
    # Independent current untiled input to the same declared numeric operator.
    sbt = lane_qmm.pack_scales(linear.scales, linear.biases)
    expected = lane_qmm.lane_matmul(x, linear.weight, sbt, tiled=False, nt=lane_qmm.NT, group=32)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_ple_lookup_tracks_current_shards_without_rewriting_source_layout(bits):
    mx.random.seed(19)
    shards = [nn.QuantizedEmbedding(2, 160, group_size=32, bits=bits) for _ in range(8)]
    for shard in shards:
        shard.scales, shard.biases = shard.scales.astype(mx.bfloat16), shard.biases.astype(mx.bfloat16)
    embedding = SimpleNamespace(dims=160, shards=shards, host=None,
                                quant_bits=bits, quant_group=32, table_scale=.5)
    plan = embed.PleTables(embedding)
    ids = np.array([list(range(0, 16, 2)), list(range(1, 16, 2))], dtype=np.uint32)
    original = [(sh.weight, sh.scales, sh.biases) for sh in shards]
    before = embed.ple_lookup(ids, plan)
    mx.eval(before)
    assert all(sh.weight is w and sh.scales is sc and sh.biases is bi
               for sh, (w, sc, bi) in zip(shards, original))
    shards[0].weight[:, :] = mx.bitwise_xor(shards[0].weight, mx.array(17, dtype=mx.uint32))
    shards[1].scales = (shards[1].scales.astype(mx.float32) * 1.5).astype(mx.bfloat16)
    current = embed.ple_lookup(ids, plan)
    rows = []
    for row in ids:
        selected = []
        for value in row:
            shard, local = shards[int(value) // 2], int(value) % 2
            dense = mx.dequantize(shard.weight[local:local + 1], shard.scales[local:local + 1],
                                  shard.biases[local:local + 1], group_size=32, bits=bits)
            selected.append(dense)
        rows.append(mx.concatenate(selected, axis=-1))
    expected = embed.scaled_rows(mx.concatenate(rows), .5)
    mx.eval(current, expected)
    assert mx.array_equal(current, expected).item()


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_cut_head_binds_current_rows_with_exact_packed_fields(bits):
    from tensorfold.families.qwen4_exp.draft_head import cut_head

    mx.random.seed(23)
    head = nn.QuantizedLinear(256, 128, bias=False, group_size=32, bits=bits)
    ids = np.arange(0, 128, 2, dtype=np.uint32)
    first = cut_head(head, ids)
    mx.eval(first.weight, first.scales, first.biases)
    head.weight[:, :] = mx.bitwise_xor(head.weight, mx.array(17, dtype=mx.uint32))
    head.scales = (head.scales * 1.25).astype(head.scales.dtype)
    head.biases = (head.biases + .01).astype(head.biases.dtype)
    second = cut_head(head, ids)
    expected = [mx.take(getattr(head, name), mx.array(ids), axis=0) for name in ("weight", "scales", "biases")]
    mx.eval(second.weight, second.scales, second.biases, *expected)
    assert all(mx.array_equal(actual, value).item()
               for actual, value in zip((second.weight, second.scales, second.biases), expected))
    assert second.bits == head.bits and second.group_size == head.group_size
