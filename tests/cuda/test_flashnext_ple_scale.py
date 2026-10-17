"""A global n-gram scale multiplies stored bf16 rows before their projections and sums."""

import pytest
import torch
from fractions import Fraction

from tensorfold.families.qwen4_exp.cuda import glue
from tensorfold.cuda.kernels.qmm import group_sums


def _exact_group_sums(values):
    """Independent exact sums for these finite BF16 fixtures, then one FP32 rounding.

    Fraction does not inherit either GPU reduction tree. The equality check
    proves the intermediate FP64 conversion is exact, excluding double rounding.
    """

    groups = values.cpu().float().reshape(-1, 32).tolist()
    exact = [sum(map(Fraction, group), Fraction()) for group in groups]
    intermediates = [float(value) for value in exact]
    assert all(Fraction(value) == reference for value, reference in zip(intermediates, exact))
    return torch.tensor(intermediates, dtype=torch.float32, device=values.device).reshape(values.shape[0], -1)


@pytest.mark.parametrize("value", [1.0, 0.000199])
@pytest.mark.parametrize("packed", [False, True])
def test_table_scale_follows_dequantization_and_updates_group_sums(value, packed):
    rows, heads, dims = 3, 4, 160
    g = torch.Generator(device="cuda").manual_seed(17)
    factor = float(torch.tensor(value, dtype=torch.bfloat16))
    out = torch.empty((rows, heads * dims), device="cuda", dtype=torch.bfloat16)
    sums = torch.empty((rows, heads * dims // 32), device="cuda", dtype=torch.float32)
    plain, plain_sums = torch.empty_like(out), torch.empty_like(sums)
    if packed:
        words = torch.randint(-(2**31), 2**31, (rows * heads, dims // 8), generator=g,
                              device="cuda", dtype=torch.int32)
        scales = torch.rand((rows * heads, dims // 32), generator=g, device="cuda").to(torch.bfloat16)
        biases = torch.randn(scales.shape, generator=g, device="cuda").to(torch.bfloat16)
        glue.ple_embed(rows, words, scales, biases, heads, dims, plain, plain_sums)
        glue.ple_embed(rows, words, scales, biases, heads, dims, out, sums, scale=factor)
    else:
        values = torch.randn((rows * heads, dims), generator=g, device="cuda").to(torch.bfloat16)
        glue.ple_embed_bf16(rows, values, heads, dims, plain, plain_sums)
        glue.ple_embed_bf16(rows, values, heads, dims, out, sums, scale=factor)
    expected = (plain.float() * factor).to(torch.bfloat16)
    assert torch.equal(out, expected)
    assert torch.equal(sums, _exact_group_sums(expected))
    # The projection's precomputed XS must have the ordinary matmul's bits.
    # Torch's reduction tree can round differently even on this small group.
    assert torch.equal(sums, group_sums(out, 32))
    if factor == 1:
        assert torch.equal(out, plain) and torch.equal(sums, plain_sums)


def test_forward_scale_equals_bf16_rows_scaled_before_staging(tmp_path):
    import numpy as np
    from test_flashnext_forward import _Rand, _bf16_table, _cfg, _model, _ple
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill

    config = _cfg(ple=True)
    table = _bf16_table(tmp_path / "table.safetensors", config.ngram().rows, config.ngram().dims)
    table.weight_scale = 3.0
    engine = Engine(_model(ple=_ple(config, table, _Rand(3))), capacity=1024, max_rows=8, prefill_rows=16)
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12]
    first = prefill(engine, prompt, None)
    actual = engine.forward([first, 73, 91]).clone()
    gather = table.gather

    def scaled(ids):
        values = torch.from_numpy(gather(ids).view(np.int16).copy()).view(torch.bfloat16)
        return (values.float() * 3.0).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)

    table.weight_scale, table.gather = 1.0, scaled
    expected_first = prefill(engine, prompt, None)
    assert first == expected_first
    assert torch.equal(actual, engine.forward([first, 73, 91]))
