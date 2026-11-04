"""Load-time requantization of BF16 weights (tensorfold.cuda.requant): FP8 (E4M3 codes, one scale an output row) and
NVFP4 (E2M1 codes, an E4M3 scale per 16 inputs, one global scale)."""

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
requant = pytest.importorskip("tensorfold.cuda.requant")
E4M3_MAX, requant_rows = requant.E4M3_MAX, requant.requant_rows

MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)                # E2M1 magnitudes by code (OCP MX), the oracles' own copy


def test_rows_scale_to_the_e4m3_range_and_reconstruct_within_its_rounding():
    gen = torch.Generator().manual_seed(3)
    w = (torch.randn(300, 128, generator=gen) * torch.logspace(-4, 1, 300)[:, None]).bfloat16()
    w[7] = 0                                                   # an all-zero row keeps scale 1 and zero codes
    codes, scales = requant_rows(w, chunk=64)                  # several chunks, one partial
    assert codes.dtype == torch.float8_e4m3fn and codes.shape == w.shape and scales.dtype == torch.float32
    top = w.float().abs().amax(1)
    expect = torch.where(top > 0, top / E4M3_MAX, torch.ones_like(top))
    assert torch.equal(scales, expect) and scales[7] == 1 and not codes[7].float().any()
    assert float(codes.float().abs().amax(1).max()) <= E4M3_MAX
    back = codes.float() * scales[:, None]
    # E4M3 keeps 3 mantissa bits: a normal value rounds within 2^-4 relative; subnormals within half a step
    err = (back - w.float()).abs()
    bound = torch.maximum(w.float().abs() * 2.0 ** -4, scales[:, None] * 2.0 ** -10)
    assert bool((err <= bound).all())


def test_codes_are_round_to_nearest_of_the_scaled_row():
    w = torch.tensor([[1.0, -0.3, 0.0, 448.0 / 3], [2.5, 2.5, -5.0, 1e-3]], dtype=torch.bfloat16)
    codes, scales = requant_rows(w)
    scaled = w.float() / scales[:, None]
    np.testing.assert_array_equal(codes.float().numpy(), scaled.to(torch.float8_e4m3fn).float().numpy())


@pytest.mark.parametrize("bad", [torch.zeros(4), torch.zeros(2, 3, dtype=torch.int32),
                                 torch.tensor([[1.0, float("nan")]]), torch.tensor([[1.0, 2.0], [float("inf"), 1.0]]),
                                 torch.tensor([[1.0, 2.0], [-float("inf"), 1.0]]),
                                 torch.tensor([[1.0, 2.0], [1e4, float("nan")]])])   # NaN beside a larger value
def test_malformed_weights_are_refused(bad):
    with pytest.raises(ValueError):
        requant_rows(bad, chunk=1)                             # the non-finite row in the second chunk


def test_a_row_scale_that_underflows_fp32_is_refused_before_dividing_by_zero():
    smallest = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    with pytest.raises(ValueError, match="representable FP32 row scales"):
        requant_rows(smallest.reshape(1, 1))


@pytest.mark.parametrize("quantize", [requant_rows, requant.nvfp4_rows])
@pytest.mark.parametrize("chunk", [-1, 0, True, 1.5, None])
def test_invalid_chunk_rows_are_refused_instead_of_returning_uninitialized_outputs(quantize, chunk):
    with pytest.raises(ValueError, match="chunk rows must be a positive integer"):
        quantize(torch.ones(2, 16, dtype=torch.bfloat16), chunk=chunk)


@pytest.mark.parametrize("source", [object(), type("Packed", (), {"weight": torch.zeros(2, 64, dtype=torch.uint8)})()])
def test_a_head_without_floating_weights_is_refused_before_quantization(source):
    with pytest.raises(ValueError, match="floating head"):
        requant.Head(source, "fp8")


def test_accounting_matches_the_copy_and_refuses_unknown_kinds():
    from tensorfold.cuda.requant import resident_bytes, staging_bytes

    assert resident_bytes("fp8", 1000, 256) == 1024 * 256 + 1024 * 4
    assert staging_bytes("fp8", 262144, 2816) > resident_bytes("fp8", 262144, 2816)
    for call in (resident_bytes, staging_bytes):
        with pytest.raises(ValueError):
            call("int4", 10, 64)


@pytest.mark.parametrize("n,k", [(1, 64), (3, 64), (7, 256), (127, 64), (129, 256), (4100, 256), (16385, 64)])
def test_fp8_staging_covers_padded_fragment_storage_for_small_and_non_aligned_projections(n, k):
    from tensorfold.cuda.nvfp4.linear import FRAGMENT_ROWS, fragment_index

    linear = requant.requantize(torch.ones(n, k, dtype=torch.bfloat16), "fp8")
    rows = min(linear.npad, FRAGMENT_ROWS)
    kk, nn = fragment_index(k, rows, "cpu")
    assert kk.stride() != nn.stride()
    # CUDA's advanced-index constructor materializes matching contiguous indices; their original backing storage
    # still belongs to kk/nn. Observe both independently of the admission formula. CPU indexing skips these copies.
    allocations = [linear.w8.numel(), linear.cols.numel() * 4, n * k, n * 4, rows * k, rows * k,
                   kk.numel() * kk.element_size(), nn.numel() * nn.element_size(),
                   kk.untyped_storage().nbytes(), nn.untyped_storage().nbytes()]
    required = sum(-(-size // 512) * 512 for size in allocations)
    if (n, k) == (1, 64):
        assert required == 165376  # exact live-storage peak observed on the pinned native CUDA allocator
    assert linear.nbytes() == requant.resident_bytes("fp8", n, k)
    assert required <= linear.nbytes() + requant.staging_bytes("fp8", n, k)


def _recipe_oracle(x):
    """The NVFP4 recipe as tensorfold.cuda.nvfp4.experts computed it before nvfp4_rows (an independent oracle: whole
    tensor at once, nearest magnitude by argmin over the eight E2M1 distances, ties to the first, i.e. smaller)."""
    n, k = x.shape
    x = x.float()
    g = (x.abs().amax() / (6.0 * 448.0)).clamp_min(1e-30)
    blocks = x.view(n, k // 16, 16)
    s = (blocks.abs().amax(-1) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn)
    step = (s.float() * g)[..., None]
    v = torch.where(step > 0, blocks / step.clamp_min(1e-30), torch.zeros_like(blocks))
    code = ((v.abs()[..., None] - torch.tensor(MAGS)).abs().argmin(-1) + 8 * (v < 0).to(torch.int64)).view(n, k)
    code = code.to(torch.uint8)
    return code[:, 0::2] | (code[:, 1::2] << 4), s.view(torch.uint8), g


def _nvfp4_values(codes, scales, g):
    """What an NVFP4 copy holds, exactly, in FP32: the checkpoint format's numpy dequantizer (CPU tensors)."""
    from tensorfold.cuda.nvfp4.format import dequant

    return torch.from_numpy(dequant("nvfp4", codes.numpy(), scales.numpy(), float(g)))


def test_nvfp4_rows_match_the_recipe_bit_for_bit_in_any_chunking():
    from tensorfold.cuda.requant import nvfp4_rows

    gen = torch.Generator().manual_seed(4)
    w = (torch.randn(200, 128, generator=gen) * torch.logspace(-3, 0.5, 200)[:, None]).bfloat16()
    w[5] = 0
    w[9, :16] = 0                                              # an all-zero block: scale 0, zero codes
    want = _recipe_oracle(w)
    for chunk in (7, 64, 200, 4096):
        codes, scales, g = nvfp4_rows(w, chunk=chunk)
        assert codes.dtype == scales.dtype == torch.uint8 and codes.shape == (200, 64) and scales.shape == (200, 8)
        assert torch.equal(codes, want[0]) and torch.equal(scales, want[1]) and torch.equal(g, want[2]), chunk
    assert not codes[5].any() and not scales[9, 0]


def test_nvfp4_ties_round_to_the_smaller_magnitude_like_the_recipe():
    """Global 2**-10 and a unit block scale make the quotients exact: every E2M1 midpoint, signs, and the grid."""
    from tensorfold.cuda.requant import nvfp4_rows

    mids = [(a + b) / 2 for a, b in itertools.pairwise(MAGS)]
    row = [6.0] + mids + [-m for m in mids] + [0.0]            # 16 values; the block max 6 sets its scale to 1
    grid = [6.0] + list(MAGS) + [-v for v in MAGS[:7]]
    w = torch.tensor([row, grid, [6.0 * 448.0] + [0.0] * 15], dtype=torch.float32) * 2.0 ** -10
    codes, scales, g = nvfp4_rows(w.bfloat16())
    want = _recipe_oracle(w.bfloat16())
    assert torch.equal(codes, want[0]) and torch.equal(scales, want[1])
    values = _nvfp4_values(codes, scales, g) / 2.0 ** -10
    assert values[0].tolist() == [6.0] + list(MAGS[:-1]) + [-v for v in MAGS[:-1]] + [0.0]
    assert values[1].tolist() == grid                          # the grid itself is exact


def test_nvfp4_values_stay_within_one_block_step():
    from tensorfold.cuda.requant import nvfp4_rows

    gen = torch.Generator().manual_seed(8)
    w = (torch.randn(64, 256, generator=gen) * torch.logspace(-2, 1, 64)[:, None]).bfloat16()
    codes, scales, g = nvfp4_rows(w)
    step = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1) * g
    # E2M1 magnitudes are at most 2 apart and the block scale rounds to E4M3 (quotients <= 6 / (1 - 2**-4)):
    # an element moves by at most one step of its block.
    assert bool(((_nvfp4_values(codes, scales, g) - w.float()).abs() <= step).all())


@pytest.mark.parametrize("bad", [torch.zeros(4, 24), torch.zeros(2, 32, dtype=torch.int32),
                                 torch.tensor([[1.0, float("inf")] * 8])])
def test_malformed_nvfp4_weights_are_refused(bad):
    from tensorfold.cuda.requant import nvfp4_rows

    with pytest.raises(ValueError):
        nvfp4_rows(bad)


def test_nvfp4_accounting():
    from tensorfold.cuda.requant import resident_bytes, staging_bytes

    assert resident_bytes("nvfp4", 1000, 256) == 1024 * 128 + 1024 * 16
    assert staging_bytes("nvfp4", 248320, 2048) > resident_bytes("nvfp4", 248320, 2048)


def _token_head_source(cap=30.0):
    linear = requant.requantize(torch.arange(128 * 64).reshape(128, 64).bfloat16() / 4096, 'fp8')
    return requant.Head(linear, 'fp8', softcap=cap)


def test_token_head_aliases_immutable_fp8_payload_without_mutating_or_repacking(monkeypatch):
    source = _token_head_source()
    codes, scales = source.linear.w8.clone(), source.linear.cols.clone()
    def forbidden(*args, **kwargs):
        pytest.fail('an existing private FP8 head must not be quantized or packed again')
    monkeypatch.setattr(requant, 'requantize', forbidden)
    monkeypatch.setattr(requant.Fp8Linear, 'from_checkpoint', forbidden)
    native = requant.TokenFp8Head(source)
    assert native.linear is not source.linear and native.linear.token and not source.linear.token
    assert native.linear.w8 is source.linear.w8 and native.linear.cols is source.linear.cols
    assert native.nbytes() == source.nbytes() and native.softcap == source.softcap == 30.0
    assert torch.equal(source.linear.w8, codes) and torch.equal(source.linear.cols.view(torch.int32), scales.view(torch.int32))
    assert requant.KINDS == ('fp8', 'nvfp4')


@pytest.mark.parametrize('cap', [None, 30.0])
def test_token_head_explicit_fp32_owning_output_and_single_softcap(monkeypatch, cap):
    from tensorfold.cuda.nvfp4 import checkpoint
    source = _token_head_source(cap)
    native = requant.TokenFp8Head(source)
    x = torch.ones((4, 64), dtype=torch.bfloat16)
    values = torch.linspace(-90, 90, 4 * 128).reshape(4, 128)
    calls = []
    def product(mode, rows, linear, out=None, f32=False):
        assert mode == checkpoint.A8 and rows is x and linear is native.linear
        assert f32 is True and out is None
        result = values.clone();calls.append(result)
        return result
    monkeypatch.setattr(checkpoint, 'matmul', product)
    first, second = native(x), native(x)
    expected = values if cap is None else torch.tanh(values / cap) * cap
    assert first.dtype == second.dtype == torch.float32 and first.shape == (4, 128)
    assert torch.equal(first.view(torch.int32), expected.view(torch.int32))
    assert first.data_ptr() != second.data_ptr()
    if cap is not None:
        assert first.data_ptr() != calls[0].data_ptr()
        assert not torch.equal(first, torch.tanh(expected / cap) * cap)


@pytest.mark.parametrize('change', ['token', 'act', 'groups', 'cols', 'scale'])
def test_token_head_refuses_an_incompatible_existing_copy(change):
    source = _token_head_source()
    setattr(source.linear, change, {'token': True, 'act': 1.0, 'groups': torch.ones(1),
                                   'cols': None, 'scale': 2.0}[change])
    with pytest.raises(ValueError, match='unchanged per-channel'):
        requant.TokenFp8Head(source)
    with pytest.raises(ValueError, match='existing private FP8 Head'):
        requant.TokenFp8Head(object())


@pytest.mark.parametrize('rows,width,dtype', [(0, 64, torch.bfloat16), (5, 64, torch.bfloat16),
                                            (1, 63, torch.bfloat16), (1, 64, torch.float32)])
def test_token_head_bad_input_refuses_before_native_work(monkeypatch, rows, width, dtype):
    from tensorfold.cuda.nvfp4 import checkpoint
    native = requant.TokenFp8Head(_token_head_source())
    monkeypatch.setattr(checkpoint, 'matmul', lambda *args, **kwargs: pytest.fail('invalid head input reached native work'))
    with pytest.raises(ValueError, match='matching BF16 rows'):
        native(torch.zeros((rows, width), dtype=dtype))


def test_token_head_does_not_clamp_nonfinite_logits_or_retry_native_failure(monkeypatch):
    from tensorfold.cuda.nvfp4 import checkpoint
    native = requant.TokenFp8Head(_token_head_source())
    x = torch.ones((1, 64), dtype=torch.bfloat16)
    monkeypatch.setattr(checkpoint, 'matmul', lambda *args, **kwargs: torch.full((1, 128), float('nan')))
    assert torch.isnan(native(x)).all()
    primary = RuntimeError('native A8 head failed');primary.__cause__ = ValueError('original native cause')
    calls = []
    def failed(*args, **kwargs):
        calls.append('native');raise primary
    monkeypatch.setattr(checkpoint, 'matmul', failed)
    with pytest.raises(RuntimeError) as caught:native(x)
    assert caught.value is primary and isinstance(primary.__cause__, ValueError) and calls == ['native']
