"""Standalone new-profile correctness; no exact-profile/model-quality tolerance.

Integer codes/numerators and specified FP32 dot operations compare exactly.
SiLU keeps the original CUDA expression/cast (library primitive, no CPU expf
threshold invented); down oracle consumes that actual intermediate, checking
its independent packing/dot exactly. Analytic-zero and graph controls cover the
activation wiring. Status reads here are cold test diagnostics only.
"""

from pathlib import Path
from tensorfold.cleanup import rollback
import numpy as np
import pytest
import torch

from dp4a_profile_oracle import quantize, group_oracle, warp_sum
from tensorfold.cuda.sm70.dp4a_experts import extension, Prototype
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFExperts, GGUFPacked

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0), reason="actualSM70privateDP4Aprofile"
)
DATA = Path(__file__).parents[1] / "fixtures/dp4a_ggml_hot.npz"


def exact(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize("columns,prefix", [(2560, 0), (80, 0), (80, 16)])
def test_quantizer_zero_tie_subnormal_extreme_codes_match_spec(columns, prefix):
    x = np.zeros((4, columns), np.float16)
    x[0, :8] = [127, 0.5, -0.5, 1.5, -1.5, 0.0, -0.0, 1]
    x[1].fill(np.nextafter(np.float16(0), np.float16(1)))
    x[2].fill(np.float16(65504))
    expected, status = quantize(x, prefix)
    value = torch.tensor(x, device="cuda")
    q = torch.empty(expected.shape, dtype=torch.uint8, device="cuda")
    word = torch.zeros(1, dtype=torch.int32, device="cuda")
    extension().quantize(value, q, prefix, word)
    exact(q.cpu(), torch.from_numpy(expected))
    assert status == 0 and word.cpu().tolist() == [0]


@pytest.mark.parametrize("kind", [18, 22, 20])
def test_all_scalar_codebook_rows_exact_int64_numerators_and_fp32_terms(kind):
    with np.load(DATA) as data:
        rawrows = data["raw_" + str(kind)].copy()
        decodedrows = data["bits_" + str(kind)].view(np.float32).copy()
    columns, prefix = (80, 16) if kind == 20 else (2560, 0)
    x = np.linspace(-0.125, 0.125, columns, dtype=np.float16).reshape(1, -1)
    q, status = quantize(x, prefix)
    assert status == 0
    value = torch.tensor(x, device="cuda")
    packed = torch.empty(q.shape, dtype=torch.uint8, device="cuda")
    word = torch.zeros(1, dtype=torch.int32, device="cuda")
    module = extension()
    module.quantize(value, packed, prefix, word)
    integers = torch.empty(q.shape[1], dtype=torch.int32, device="cuda")
    terms = torch.empty(q.shape[1], device="cuda")
    for raw, decoded in zip(rawrows, decodedrows, strict=True):
        repetitions = 3 if kind == 20 else 10
        row = np.tile(raw, repetitions)
        weights = np.tile(decoded, repetitions)
        expected_i, expected_f = group_oracle(row, weights, q[0], kind)
        module.debug(packed, torch.tensor(row[None], device="cuda"), kind, integers, terms, word)
        exact(integers.cpu(), torch.from_numpy(expected_i.astype(np.int32)))
        exact(terms.cpu(), torch.from_numpy(expected_f))
    assert word.cpu().tolist() == [0]


def bank(gkind, ukind, offset):
    with np.load(DATA) as data:
        raws = {kind: data["raw_" + str(kind)][3 if kind == 22 else 7].copy() for kind in (18, 22, 20)}
        decoded = {
            kind: data["bits_" + str(kind)][3 if kind == 22 else 7].view(np.float32).copy() for kind in (18, 22, 20)
        }

    def packed(kind, outputs, columns, crop):
        row = np.tile(raws[kind], 3 if kind == 20 else 10)
        tensor = torch.tensor(np.broadcast_to(row, (512 * outputs, row.size)).copy(), device="cuda")
        return GGUFPacked(tensor, kind, tensor.shape[0], columns, crop)

    torch.manual_seed(711)
    shared = [
        GGUFDense(torch.randn(shape, device="cuda", dtype=torch.float16) * 0.001)
        for shape in [(80, 2560), (80, 2560), (2560, 80)]
    ]
    b = GGUFExperts(
        packed(gkind, 80, 2560, 0),
        packed(ukind, 80, 2560, 0),
        packed(20, 2560, 80, offset),
        shared,
        count=512,
        width=80,
        dims=2560,
    )
    return b, raws, decoded


@pytest.mark.parametrize("gkind,ukind,offset", [(18, 22, 0), (22, 18, 16)])
@pytest.mark.parametrize("rows", [1, 8, 32])
def test_mixed_hot_experts_dot_tree_crop_and_exact_shared(gkind, ukind, offset, rows):
    b, raws, decoded = bank(gkind, ukind, offset)
    rng = np.random.default_rng(81 + rows)
    x = (rng.normal(size=(rows, 2560)) * 0.03).astype(np.float16)
    picks = torch.tensor([[0, 511, 7, 3, 88, 91, 5, 12, 42, 21, 512]] * rows, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, 11, 2560), device="cuda")
    owner = Prototype(b, torch.tensor(x, device="cuda"), picks, out)
    try:
        owner.run()
        torch.cuda.synchronize()
        assert owner.status.cpu().tolist() == [0]
        q, _ = quantize(x)
        for row in range(rows):
            for kind, actual in [(gkind, owner.gate), (ukind, owner.up)]:
                _integer, terms = group_oracle(np.tile(raws[kind], 10), np.tile(decoded[kind], 10), q[row], kind)
                value = warp_sum(terms)
                exact(actual[row, :10].cpu(), torch.full((10, 80), float(value)))
        h = owner.hidden.cpu().numpy().reshape(rows * 11, 80)
        assert np.isfinite(h).all()
        qdown, _ = quantize(h, offset)
        exact(owner.qh.cpu(), torch.from_numpy(qdown))
        for row in range(rows):
            for slot in range(10):
                _integer, terms = group_oracle(np.tile(raws[20], 3), np.tile(decoded[20], 3), qdown[row * 11 + slot], 20)
                exact(out[row, slot].cpu(), torch.full((2560,), float(warp_sum(terms))))
        reference = torch.zeros_like(out)
        b._write_shared(owner.x, picks, reference)
        exact(out[:, 10], reference[:, 10])
    except BaseException as primary:
        rollback(owner, primary, owner.close)
    else:
        owner.close()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_latched_device_status_final_veto_and_explicit_cold_recovery(bad):
    b, _raw, _decoded = bank(18, 22, 16)
    x = torch.zeros((1, 2560), dtype=torch.float16, device="cuda")
    x[0, 0] = bad
    picks = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    out = torch.empty((1, 1, 2560), device="cuda")
    owner = Prototype(b, x, picks, out)
    try:
        owner.run()
        torch.cuda.synchronize()
        assert owner.status.cpu().tolist()[0] & 1 and torch.isnan(out).all()
        x.zero_()
        owner.run()
        torch.cuda.synchronize()
        assert owner.status.cpu().tolist()[0] & 1 and torch.isnan(out).all()
        owner.reset_after_fence()
        owner.run()
        torch.cuda.synchronize()
        assert owner.status.cpu().tolist() == [0] and torch.isfinite(out).all()
    except BaseException as primary:
        rollback(owner, primary, owner.close)
    else:
        owner.close()


def test_invalid_pick_and_coefficient_nonfinite_never_read_out_of_range_or_commit_finite():
    b, _raw, _decoded = bank(18, 22, 0)
    x = torch.zeros((1, 2560), dtype=torch.float16, device="cuda")
    picks = torch.tensor([[-1, 513]], dtype=torch.int32, device="cuda")
    out = torch.empty((1, 2, 2560), device="cuda")
    owner = Prototype(b, x, picks, out)
    try:
        owner.run()
        torch.cuda.synchronize()
        assert owner.status.cpu().tolist()[0] & 8 and torch.isnan(out).all()
        picks.zero_()
        owner.reset_after_fence()
        b.gate.data[:, 0] = 0
        b.gate.data[:, 1] = 0x7C
        owner.run()
        torch.cuda.synchronize()
        assert owner.status.cpu().tolist()[0] & 2 and torch.isnan(out).all()
    except BaseException as primary:
        rollback(owner, primary, owner.close)
    else:
        owner.close()


def test_same_stream_graph_keeps_bindings_and_close_order():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        b, _raw, _decoded = bank(18, 22, 16)
        x = torch.zeros((1, 2560), dtype=torch.float16, device="cuda")
        picks = torch.tensor([[0, 512]], dtype=torch.int32, device="cuda")
        out = torch.empty((1, 2, 2560), device="cuda")
        owner = Prototype(b, x, picks, out)
        graph = None
        try:
            graph = owner.capture()
            with pytest.raises(RuntimeError, match="borrower"):
                owner.close()
            x.fill_(0.03125)
            owner.run()
            expected = out.clone()
            graph.replay()
            stream.synchronize()
            exact(out, expected)
        except BaseException as primary:

            def close_all():
                if owner.graph_owner is not None:
                    owner.graph_owner.close()
                owner.close()

            rollback(owner, primary, close_all)
        else:
            if graph is not None:
                graph.close()
            owner.close()


def test_analytic_zero_silu_and_cpp_alignment_refusals():
    module = extension()
    g = torch.zeros((1, 2, 80), device="cuda")
    u = torch.ones_like(g)
    h = torch.empty_like(g, dtype=torch.float16)
    module.activate(g, u, h)
    exact(h, torch.zeros_like(h))
    x = torch.zeros((1, 2560), dtype=torch.float16, device="cuda")
    raw = torch.empty(80 * 36 + 1, dtype=torch.uint8, device="cuda")
    q = raw[1:].view(1, 80, 36)
    status = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="alignment"):
        module.quantize(x, q, 0, status)
    # Valid aligned native geometry, but status aliases an immutable source.
    # Nonfinite weight scale would atomicOr that source if dispatch were reached.
    packed = torch.zeros((1, 3, 36), dtype=torch.uint8, device="cuda")
    weight = torch.zeros((1, 54), dtype=torch.uint8, device="cuda")
    weight[0, 1] = 124  # FP16 +Inf scale, no float-to-int conversion in this test.
    integers = torch.empty(3, dtype=torch.int32, device="cuda")
    terms = torch.empty(3, dtype=torch.float32, device="cuda")
    saved_q, saved_weight = packed.clone(), weight.clone()
    for source in (packed, weight):
        alias = source.view(-1)[:4].view(torch.int32)
        with pytest.raises(RuntimeError, match="debug status must not alias source"):
            module.debug(packed, weight, 20, integers, terms, alias)
        exact(packed, saved_q)
        exact(weight, saved_weight)


@pytest.mark.parametrize("columns,prefix,bit", [(2560, 0, 1), (80, 16, 4)])
def test_direct_nonfinite_quantizer_never_converts_bad_block(columns, prefix, bit):
    values = torch.zeros((1, columns), dtype=torch.float16, device="cuda")
    values[0, 0] = float("nan")
    groups = (columns + prefix + 31) // 32
    packed = torch.full((1, groups, 36), 255, dtype=torch.uint8, device="cuda")
    status = torch.zeros(1, dtype=torch.int32, device="cuda")
    extension().quantize(values, packed, prefix, status)
    assert status.cpu().tolist() == [bit]
    assert not packed[0, 0].cpu().any()
