"""Existing native prep math: frame addresses separated from absolute RoPE positions."""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                               reason="requires actual SM70")


@pytest.mark.parametrize("rows", [1, 2, 3, 17, 37, 256])
@pytest.mark.parametrize("near_limit", [False, True])
@pytest.mark.parametrize("independent_rows", [False, True])
def test_existing_native_prep_frame_offsets_preserve_every_stored_byte(rows, near_limit, independent_rows):
    from tensorfold.cuda.sm70.family_native import extension

    native = extension()
    assert native.prepare_owner_prelude() == 1
    torch.manual_seed(3197)
    dim, heads, ih, di = 256, 3, 4, 128
    base = 262144 - rows if near_limit else 273
    projection = torch.randn((rows, heads * 2 * dim + 2 * dim + (ih + 1) * di),
                             dtype=torch.float16, device="cuda") * 3
    norm = [torch.randn(width, device="cuda") * .2 + 1 for width in (dim, dim, di)]
    frequency = torch.tensor([1e7 ** (-2 * i / 64) for i in range(32)], device="cuda")
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    absolute = torch.tensor([base], dtype=torch.int32, device="cuda")
    outputs = []
    for capacity, position, delta in ((base + rows, absolute, None), (rows, zero, absolute)):
        query = torch.empty((rows, heads, dim), dtype=torch.float16, device="cuda")
        iq = torch.empty((rows, ih, di), dtype=torch.float16, device="cuda")
        key, value = [torch.zeros((capacity, 1, dim), dtype=torch.int8, device="cuda") for _ in range(2)]
        ks, vs = [torch.zeros((capacity, 1, dim // 32), dtype=torch.float16, device="cuda") for _ in range(2)]
        ik = torch.zeros((capacity, di), dtype=torch.float16, device="cuda")
        status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
        native.prep(projection, position, *norm, frequency, query, key, value, iq, ik,
                    ks, vs, status, delta, None, heads, 1, dim, ih, di, 8, 8, 3,
                    1e-6, 1., independent_rows)
        assert status.tolist() == [0, -2]
        start = base if delta is None else 0
        outputs.append((query, iq, *(item[start:start + rows] for item in (key, value, ks, vs, ik))))
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert zero.tolist() == [0] and absolute.tolist() == [base]


@pytest.mark.parametrize("failure", ["overflow-scale", "nonfinite-query", "nonfinite-index", "mixed"])
def test_existing_frame_math_status_and_q8_failure_bytes_match_torch(failure):
    from unittest.mock import patch
    from tensorfold.cuda.sm70 import family_ops as ops
    from tensorfold.cuda.sm70.family_native import extension

    extension()
    dim, qh, ih, di, rows = 256, 3, 4, 128, 3
    projection = torch.ones((rows, 2 * qh * dim + 2 * dim + (ih + 1) * di),
                            dtype=torch.float16, device="cuda")
    if failure in ("overflow-scale", "mixed"):
        projection[:, 2 * qh * dim + dim:2 * qh * dim + 2 * dim].fill_(65504)
    if failure in ("nonfinite-query", "mixed"):
        projection[1, 3] = float("nan")
    if failure in ("nonfinite-index", "mixed"):
        projection[0, -1] = float("inf")
    norm = [torch.ones(width, device="cuda") for width in (dim, dim, di)]
    freq = torch.tensor([1e7 ** (-2 * i / 64) for i in range(32)], device="cuda")
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    absolute = torch.tensor([262140], dtype=torch.int32, device="cuda")
    results = []
    for fused in (False, True):
        q = torch.empty((rows, qh, dim), dtype=torch.float16, device="cuda")
        iq = torch.empty((rows, ih, di), dtype=torch.float16, device="cuda")
        key, value = [torch.empty((rows, 1, dim), dtype=torch.int8, device="cuda") for _ in range(2)]
        ks, vs = [torch.empty((rows, 1, dim // 32), dtype=torch.float16, device="cuda") for _ in range(2)]
        ik = torch.empty((rows, di), dtype=torch.float16, device="cuda")
        status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
        with patch.object(ops, "_NATIVE_FAMILY", fused):
            ops.attn_prep(projection, zero, *norm, freq, q, key, value, iq, ik, 1e-6,
                          q_heads=qh, kv_heads=1, head_dim=dim, index_heads=ih, index_dim=di,
                          ks=ks, vs=vs, bits=8, delta=absolute, status=status, layer=7, independent_rows=True)
        results.append((q, key, value, iq, ik, ks, vs, status))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert results[1][-1].tolist()[1] == 7
