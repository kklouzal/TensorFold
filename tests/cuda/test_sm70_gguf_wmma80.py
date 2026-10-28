"""Independent error measurements and safety contracts for approximate K80 math.

No approximation tolerance here grants model-quality qualification. The exact
BMM oracles remain separate and unchanged.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gguf_fixture import blocks, scalar_reference  # noqa: E402
from tensorfold.cuda.sm70 import gguf_ops  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFExperts, GGUFPacked  # noqa: E402
from tensorfold.families.qwen4_exp.gguf import FORMATS, PackedTensor, decode_rows  # noqa: E402
from test_sm70_gguf_exact import bits, exact_fixture  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="SM70 WMMA80 candidate")


def source_rows(kind, rows, width, seed):
    if kind in (18, 21, 22, 23, 42):
        # Reuse pinned GGML payload patterns. Normalize their explicit Half
        # scale field exactly as the existing synthetic blocks() fixture does;
        # the original edge/overflow wire decoder tests remain untouched.
        with np.load(Path(__file__).parents[1] / "fixtures/gguf_mixed_ggml.npz", allow_pickle=False) as wire:
            raw = wire[f"raw_{kind}"]
        selected = (np.arange(rows) + seed) % raw.shape[0]
        block, size = FORMATS[kind][1:]
        source_width = raw.shape[1] // size * block
        repeats = -(-width // source_width)
        assert width % block == 0
        raw = np.ascontiguousarray(np.tile(raw[selected], (1, repeats))[:, :width // block * size])
        raw.reshape(rows, width // block, size)[:, :, :2] = np.frombuffer(np.float16(.001).tobytes(), np.uint8)
        tensor = PackedTensor("wmma80-fixture", (rows, width), kind, Path("not-opened"), 0, raw.nbytes)
        return raw, decode_rows(tensor, raw).astype(np.float16)
    raw = blocks(kind, rows, width, seed=seed)
    return raw, scalar_reference(raw, kind, width).astype(np.float16)


@pytest.fixture(scope="module", params=[(18, 20), (18, 42), (21, 20), (21, 42), (22, 20),
                                       (22, 42), (23, 20), (23, 42), (8, 8), (12, 7)])
def bank(request):
    gate_kind, down_kind = request.param
    width, hidden, experts, offset = 80, 2560, 2, 48
    block = FORMATS[down_kind][1]
    physical = -(-(offset + width) // block) * block
    source = [source_rows(gate_kind, experts * width, hidden, 821),
              source_rows(gate_kind, experts * width, hidden, 822),
              source_rows(down_kind, experts * hidden, physical, 823)]
    packed = [GGUFPacked(torch.from_numpy(raw).to("cuda"), kind, raw.shape[0], columns, crop)
              for (raw, _), kind, columns, crop in zip(source, (gate_kind, gate_kind, down_kind),
                                                     (hidden, hidden, width), (0, 0, offset))]
    generator = torch.Generator(device="cuda").manual_seed(824)
    shared = [GGUFDense(torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16) * .02)
              for shape in ((width, hidden), (width, hidden), (hidden, width))]
    value = GGUFExperts(*packed, shared, count=experts, width=width, dims=hidden)
    cpu = [torch.from_numpy(values).double() for _, values in source]
    cpu = (cpu[0].reshape(experts, width, hidden), cpu[1].reshape(experts, width, hidden),
           cpu[2][:, offset:offset + width].reshape(experts, hidden, width))
    amplitude = .1 / max(1., float(cpu[0].abs().max()), float(cpu[1].abs().max()))
    return value, cpu, amplitude, request.param


def error_record(actual, reference):
    actual, reference = actual.cpu().double(), reference.cpu().double()
    assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
    delta = actual - reference
    norm = float(torch.linalg.vector_norm(reference))
    return {"max_abs": float(delta.abs().max()), "rms": float(delta.square().mean().sqrt()),
            "relative_l2": float(torch.linalg.vector_norm(delta)) / norm if norm else None,
            "half_bin_mismatches": int((actual.half().view(torch.int16) != reference.half().view(torch.int16)).sum()),
            "values": actual.numel()}


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("method", ["forward_wmma80", "forward_blocks2w"])
def test_candidate_error_against_original_bmm_and_independent_fp64_is_recorded(
    bank, rows, method, record_property, tmp_path
):
    from test_sm70_gguf_experts import fp64_oracle

    value, cpu, amplitude, kinds = bank
    generator = torch.Generator(device="cuda").manual_seed(825)
    x = (torch.randn((rows, 2560), generator=generator, device="cuda") * amplitude).half()
    picks = torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]], device="cuda", dtype=torch.int64)[:rows].contiguous()
    run = getattr(value, method)
    actual = run(x, picks)
    reference = value.forward_reference(x, picks)
    ideal = fp64_oracle(value, x, picks, cpu)
    assert actual.shape == reference.shape and actual.dtype == torch.float32
    record_property("approximate_error", json.dumps({"method": method, "kinds": kinds, "rows": rows,
        "original_bmm": error_record(actual, reference), "independent_fp64": error_record(actual, ideal),
        "quality_status": "unqualified; metrics do not establish nonmaterial model loss"}, sort_keys=True))
    raw = tmp_path / "numeric-outputs.npz"
    np.savez(raw, input_fp16=x.cpu().numpy(), picks=picks.cpu().numpy(),
             candidate_fp32=actual.cpu().numpy(), original_bmm_fp32=reference.cpu().numpy(),
             independent_fp64=ideal.numpy())
    record_property("numeric_raw", str(raw))
    # Fixed per-pair math must be row invariant within its own distinct policy.
    serial = torch.cat([run(x[row:row + 1], picks[row:row + 1]) for row in range(rows)])
    bits(actual, serial)
    half = torch.empty_like(actual, dtype=torch.float16)
    run(x, picks.int(), half)
    bits(half, actual.half())
    shared = picks == value.count - 1
    if method == "forward_wmma80":
        bits(actual[shared], reference[shared])


def test_wmma80_core_shared_and_invalid_routes_are_defined_before_owner_overwrite():
    value, _, _ = exact_fixture(8)
    x = torch.randn((3, 2560), device="cuda", dtype=torch.float16) * .01
    picks = torch.tensor([[-1, 2, 2**40], [0, 1, 2], [2, 0, 1]], device="cuda")
    for part in value.shared:
        part.weight.fill_(float("nan"))
    output = torch.full((3, 3, 2560), float("nan"), device="cuda")
    gguf_ops.experts_wmma80(x, picks, value.gate, value.up, value.down, value.shared, output)
    bits(output[picks == 2], torch.zeros_like(output[picks == 2]))
    assert bool(torch.isnan(output[0, 0]).all()) and bool(torch.isnan(output[0, 2]).all())
    assert bool(torch.isfinite(output[1, :2]).all())


def test_wmma80_dispatch_fallback_capture_and_native_bounds():
    value, _, _ = exact_fixture(8)
    value.implementation = "packed_wmma80"
    x = torch.randn((3, 2560), device="cuda", dtype=torch.float16) * .01
    picks = torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]], device="cuda")
    value.forward(x, picks)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = value.forward(x, picks)
    x.mul_(.5)
    picks.copy_(picks.roll(1, 1))
    graph.replay()
    bits(captured, value.forward_wmma80(x, picks))
    full = x[:1].repeat(33, 1)
    routes = picks[:1].repeat(33, 1)
    bits(value.forward(full, routes), value.forward_reference(full, routes))
    with pytest.raises(RuntimeError, match="at most32"):
        gguf_ops.experts_wmma80(full, routes, value.gate, value.up, value.down, value.shared,
                                torch.empty((33, 3, 2560), device="cuda"))
    with pytest.raises(RuntimeError, match="overlap"):
        gguf_ops.experts_wmma80(x, picks[:, :1].contiguous(), value.gate, value.up, value.down, value.shared, x[:, None])
