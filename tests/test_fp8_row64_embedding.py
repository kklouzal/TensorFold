"""Independent row-major reference and existing GEMM pack authority checks."""
import copy

import pytest
import torch

from tensorfold.cuda.nvfp4.embedding import Row64Embedding, Row64Spec, packed_offsets
from tensorfold.cuda.nvfp4.linear import _fragment_order


def table(n=137, k=128, chunk=3):
    # Every finite code including signed zero/subnormals/max finite, unequal scales.
    codes = torch.arange(n * k).remainder(254)
    codes[codes >= 127] += 1
    codes = codes.to(torch.uint8).reshape(n, k)
    scales = torch.arange(n * (k // 64), dtype=torch.float32).reshape(n, -1) / 97 + .125
    spec = Row64Spec("token_embedding", n, k)
    return Row64Embedding.from_rows(spec, codes, scales, chunk_rows=chunk), codes, scales


def test_requested_rows_match_independent_dense_reference_and_share_gemm():
    owner, codes, scales = table()
    ids = torch.tensor([[136, 0, 64, 1], [63, 128, 64, 7]])
    reference = (codes.view(torch.float8_e4m3fn).float()
                 * scales.repeat_interleave(64, 1)).to(torch.bfloat16)
    got = owner.lookup_rows(ids, max_rows=8)
    assert torch.equal(got.view(torch.int16), reference[ids].view(torch.int16))
    assert owner.operation.w8 is owner._owners[0]
    assert owner.operation.bs is owner._owners[1]
    assert owner.nbytes() == owner.operation.nbytes()
    assert not hasattr(owner, "weight")


def test_address_map_bijective_and_matches_existing_forward_pack():
    # Existing forward pack uses fragment_index; lookup independently inverts it.
    n, k = 256, 192
    code = torch.arange(n * k).remainder(256).to(torch.uint8).reshape(n, k)
    packed = _fragment_order(code, n)
    offsets, scales = packed_offsets(torch.arange(n), k)
    assert torch.equal(packed[offsets], code)
    assert torch.equal(offsets.flatten().sort().values, torch.arange(n * k))
    # Independently specify tiled [rowtile, inputgroup, rowwithin64] scales.
    expected = torch.empty((n, k), dtype=torch.int64)
    for r in range(n):
        for c in range(k):
            expected[r, c] = ((r // 64) * (k // 64) + c // 64) * 64 + r % 64
    assert torch.equal(scales, expected)


def test_empty_scalar_noncontiguous_ids_and_request_limit():
    owner, _, _ = table(n=2)
    assert owner.lookup_rows(torch.empty((2, 0), dtype=torch.int64), max_rows=0).shape == (2, 0, 128)
    assert owner.lookup_rows(torch.tensor(1), max_rows=1).shape == (128,)
    ids = torch.tensor([[0, 1], [1, 0]]).t()
    assert torch.equal(owner.lookup_rows(ids, max_rows=4), owner.lookup_rows(ids.contiguous(), max_rows=4))
    for ids, budget in [(torch.tensor([-1]), 1), (torch.tensor([2]), 1),
                        (torch.tensor([0, 1]), 1), (torch.tensor([True]), 1),
                        (torch.tensor([0.]), 1), (torch.tensor([0]), True)]:
        with pytest.raises(ValueError):
            owner.lookup_rows(ids, max_rows=budget)


def test_manifest_roundtrip_strict_distinct_format_and_roles():
    for role in ("token_embedding", "per_layer_embedding", "assistant_embedding"):
        spec = Row64Spec(role, 262144, 10752)
        assert Row64Spec.from_manifest(spec.to_manifest()) == spec
    good = Row64Spec("token_embedding", 3, 64).to_manifest()
    for key, bad in [("format", "fp8block"), ("scale_block", [128, 128]),
                     ("scale_block", [True, 64]), ("rows", True), ("width", 65),
                     ("role", "lm_head"), ("lookup_dtype", "float32")]:
        value = copy.deepcopy(good)
        value[key] = bad
        with pytest.raises(ValueError):
            Row64Spec.from_manifest(value)
    with pytest.raises(ValueError):
        Row64Spec.from_manifest({**good, "extra": 1})


@pytest.mark.parametrize("bad_code", [127, 255])
def test_nonfinite_e4m3_rejected(bad_code):
    codes = torch.zeros((1, 64), dtype=torch.uint8)
    codes[0, 45] = bad_code
    with pytest.raises(ValueError, match="NaN"):
        Row64Embedding.from_rows(Row64Spec("token_embedding", 1, 64), codes, torch.ones((1, 1)))


@pytest.mark.parametrize("scale", [0., -1., float("inf"), float("nan"), 3.4e38])
def test_invalid_scales_and_bf16_overflow_rejected(scale):
    codes = torch.full((1, 64), 126, dtype=torch.uint8)  # finite448
    with pytest.raises(ValueError):
        Row64Embedding.from_rows(Row64Spec("token_embedding", 1, 64), codes, torch.tensor([[scale]]))


def test_zero_blocks_and_scale_extremes_no_silent_clipping():
    codes = torch.zeros((1, 64), dtype=torch.uint8)
    owner = Row64Embedding.from_rows(Row64Spec("per_layer_embedding", 1, 64), codes,
                                     torch.tensor([[3.4e38]]))
    assert torch.equal(owner.lookup_rows(torch.tensor([0]), max_rows=1), torch.zeros((1, 64), dtype=torch.bfloat16))


def test_mutated_or_replaced_authority_rejected():
    owner, _, _ = table(n=1)
    owner.operation.bs[0, 0, 0] = 0
    with pytest.raises(RuntimeError, match="mutated"):
        owner.lookup_rows(torch.tensor([0]), max_rows=1)
    owner, _, _ = table(n=1)
    owner.operation.w8 = owner.operation.w8.clone()
    with pytest.raises(RuntimeError, match="replaced"):
        owner.lookup_rows(torch.tensor([0]), max_rows=1)


def test_borrowed_operation_rejects_wrong_geometry_layout_and_storage():
    owner, _, _ = table(n=2)
    with pytest.raises(ValueError, match="geometry"):
        Row64Embedding(Row64Spec("token_embedding", 1, 128), owner.operation)
    owner.operation.layout = "mxfp8"
    with pytest.raises(ValueError, match="layout"):
        Row64Embedding(owner.spec, owner.operation)
    owner, _, _ = table(n=2)
    owner.operation.bs = owner.operation.bs.view(torch.float32)
    with pytest.raises(ValueError, match="layout"):
        Row64Embedding(owner.spec, owner.operation)
    with pytest.raises(ValueError, match="source"):
        Row64Embedding.from_rows(Row64Spec("token_embedding", 1, 64),
                                  torch.zeros((1, 64), dtype=torch.uint8), torch.ones((1, 2)))


def test_inference_mode_loading_retains_real_versioned_authority():
    with torch.inference_mode():
        codes = torch.zeros((2, 64), dtype=torch.uint8)
        scales = torch.ones((2, 1))
        owner = Row64Embedding.from_rows(Row64Spec("assistant_embedding", 2, 64), codes, scales)
        assert not owner.operation.w8.is_inference()
        assert not owner.operation.bs.is_inference()
        assert owner.lookup_rows(torch.tensor([1, 0]), max_rows=2).dtype == torch.bfloat16
        owner.operation.w8[0] = 1
        with pytest.raises(RuntimeError, match="mutated"):
            owner.lookup_rows(torch.tensor([0]), max_rows=1)


def test_constructor_failure_clears_owned_packed_aliases_without_stripping_frames(monkeypatch):
    import weakref
    from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear
    original = Fp8BlockLinear.from_rows.__func__
    references = []

    def observe(cls, codes, scales):
        operation = original(cls, codes, scales)
        references.append(weakref.ref(operation))
        references.extend((weakref.ref(operation.w8), weakref.ref(operation.bs)))
        return operation

    monkeypatch.setattr(Fp8BlockLinear, "from_rows", classmethod(observe))
    codes = torch.zeros((1, 64), dtype=torch.uint8)
    codes[0, 1] = 127
    try:
        Row64Embedding.from_rows(Row64Spec("token_embedding", 1, 64), codes, torch.ones((1, 1)))
    except ValueError as primary:
        # Keep the actual live exception traceback, unlike pytest.raises clearing it.
        assert "NaN" in str(primary)
        assert all(ref() is None for ref in references)
        frames = []
        current = primary.__traceback__
        while current:
            frames.append(current.tb_frame.f_code.co_name)
            current = current.tb_next
        assert "__init__" in frames and "from_rows" in frames
    else:
        pytest.fail("invalid native bytes unexpectedly admitted")
