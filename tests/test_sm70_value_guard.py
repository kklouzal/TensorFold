"""Explicit-output guard preserves original finite/status semantics."""
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.sm70 import family_ops as ops  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("live", [None, -1, 0, 1, 3, 4])
@pytest.mark.parametrize("initial", [(0, -2), (3, 7)])
def test_output_guard_matches_independent_torch_contract_and_preserves_input(dtype, live, initial):
    value = torch.tensor([[0., -0., float("nan"), float("inf")],
                          [float("-inf"), 2**30, -(2**30), 2**31],
                          [1., -2., 3., 4.]], dtype=dtype)
    before = value.view(torch.int16 if dtype == torch.float16 else torch.int32).clone()
    valid = torch.isfinite(value) & (value.abs() <= 2**30)
    mask = torch.ones_like(valid) if live is None else torch.arange(3)[:, None] < live
    bad = bool((~valid & mask).any())
    expected = torch.where(valid, value, 0)
    status = torch.tensor(initial, dtype=torch.int32)
    slab = torch.full((3, 12), 29, dtype=dtype)
    out = slab[:, 4:8]
    result = ops._guard(value, status, 16, -1,
                        None if live is None else torch.tensor([live], dtype=torch.int32), out=out)
    assert result is out
    assert torch.equal(result.view(torch.int16 if dtype == torch.float16 else torch.int32),
                       expected.view(torch.int16 if dtype == torch.float16 else torch.int32))
    assert status.tolist() == [initial[0] | (16 if bad else 0), -1 if bad and initial[1] == -2 else initial[1]]
    assert torch.equal(value.view(before.dtype), before)
    assert bool((slab[:, :4] == 29).all()) and bool((slab[:, 8:] == 29).all())


def test_none_status_passthrough_never_touches_output():
    value = torch.tensor([[float("nan"), -0.]])
    out = torch.full_like(value, 42)
    assert ops._guard(value, None, 16, 0, out=out) is value
    assert bool((out == 42).all())


def test_original_no_output_path_keeps_ordinary_torch_result():
    value = torch.tensor([[1., float("inf")]])
    status = torch.tensor([0, -2], dtype=torch.int32)
    result = ops._guard(value, status, 2, 3)
    assert torch.equal(result, torch.tensor([[1., 0.]])) and status.tolist() == [2, 3]

@pytest.mark.parametrize("with_status", [False, True])
def test_actual_owner_prepare_reuses_existing_outputs_without_changing_projection_or_iq(with_status):
    from types import SimpleNamespace as NS
    from tensorfold.families.qwen4_exp.cuda import history_owner

    rows, dims, heads, index_heads, index_dim = 3, 256, 3, 4, 128
    torch.manual_seed(941)
    projection = torch.randn((rows, 2 * heads * dims + 2 * dims + (index_heads + 1) * index_dim),
                              dtype=torch.float16) * .01
    cfg = NS(heads=heads, head_dim=dims, index_heads=index_heads, index_dim=index_dim, eps=1e-6,
             mrope_section=(11, 11, 10), rope_attention_factor=1.)
    attention = NS(q_scale=torch.ones(dims), k_scale=torch.ones(dims), iq_scale=torch.ones(index_dim))
    layer = NS(index=0, attn=attention)
    observed = []
    for selected in (False, True):
        written = []
        fmt = NS(quantized=True, bits=8, codec=0)
        def append(position, key, value, ks, vs):
            written.append((position, *(item.clone() for item in (key, value, ks, vs))))
        cache = NS(key_format=fmt, value_format=fmt, append=append)
        state = NS(history_placement=NS(target_layers=1, indexer_owner=lambda *a, **kw: 0),
                   att_index={0: 0}, kc=[cache], pos=0, kv_status=torch.tensor([0, -2], dtype=torch.int32)
                   if with_status else None, compact_indexer=object(), history_rank=1, history_slot=0)
        weights = NS(cfg=cfg, meta={"sm70_native_guard_trial": selected},
                     inv_freq=1. / (10000. ** (torch.arange(32) / 32)))
        buffers = NS(q=torch.full((rows, heads, dims), 31, dtype=torch.float16),
                     iq=torch.full((rows, index_heads, index_dim), 31, dtype=torch.float16),
                     gated=torch.full((rows, heads * dims), 31, dtype=torch.float16), prefill=False,
                     target_indexer_coordinates=torch.zeros((rows, 3), dtype=torch.int64),
                     target_indexer_position=None)
        before = projection.clone()
        history_owner.prepare(projection, state, layer, weights, buffers, 0, rows, mtp=False)
        assert torch.equal(projection, before)  # Includes the later attention gate's raw projection columns.
        observed.append((buffers.q.clone(), buffers.iq.clone(), written, state.kv_status,
                         state.target_tentative_length))
    original, candidate = observed
    assert torch.equal(original[0], candidate[0]) and torch.equal(original[1], candidate[1])
    assert original[2][0][0] == candidate[2][0][0] == 0 and original[4] == candidate[4] == rows
    for left, right in zip(original[2][0][1:], candidate[2][0][1:]):
        assert torch.equal(left, right)
    assert (original[3] is candidate[3] is None) if not with_status else torch.equal(original[3], candidate[3])
