"""Independent byte oracle and owned native-A4 authority boundary controls."""

import struct
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

from nvfp4_a4_inputs import PREFIX, Context, swiglu

from tensorfold.cuda.nvfp4.experts_a4 import pack_projection_a4, projection_a4
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4


class Cache:
    device = torch.device("cuda:0")

    def __init__(self):
        self.allocations, self._layers = [], {}

    def allocate_authority(self, shape, *, dtype, tag):
        value = torch.empty(shape, dtype=dtype)
        self.allocations.append((shape, dtype, tag))
        return value

    def register(self, key, fields):
        self._layers[key] = fields


def oracle_blob(weight, scales):
    """Scalar indexing of the foreign lane pack contract, not a tensor reshape."""
    n, k = weight.shape[0], weight.shape[1] * 2
    raw, sf = weight.numpy().tobytes(), scales.view(torch.uint8).numpy().tobytes()
    result = bytearray()
    for tile in range(n // 64):
        for step in range(k // 64):
            for j in range(8):
                for lane in range(32):
                    row, t = tile * 64 + j * 8 + lane // 4, lane % 4
                    for half in range(2):
                        at = row * (k // 2) + step * 32 + 4 * t + half * 16
                        result.extend(raw[at:at + 4])
    for tile in range(n // 64):
        for step in range(k // 64):
            for col in range(64):
                at = (tile * 64 + col) * (k // 16) + step * 4
                result.extend(sf[at:at + 4])
    return torch.tensor(list(result), dtype=torch.uint8)


@pytest.mark.parametrize("n,k", [(64, 64), (128, 192), (512, 2048)])
def test_cpu_pack_is_scalar_foreign_layout_permutation(n, k):
    w = (torch.arange(n * k // 2, dtype=torch.int64) * 17 % 256).to(torch.uint8).view(n, k // 2)
    s = (torch.arange(n * k // 16, dtype=torch.int64) * 23 % 256).to(torch.uint8).view(n, k // 16)
    got = pack_projection_a4(w, s.view(torch.float8_e4m3fn))
    assert torch.equal(got, oracle_blob(w, s))
    linear = projection_a4(got, n, k, .25, .125)
    assert linear.words.data_ptr() == got.data_ptr()
    assert linear.bs.data_ptr() == got.data_ptr() + n * k // 2
    assert linear.scale == .25 and linear.act == 8. and linear.act_inverse == .125


def test_owned_loader_reads_only_selected_original_fields_and_prepares_exact_calibrations():
    ctx, cache = Context(), Cache()
    bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, owned_ids=[2, 0],
                                layer_id=7, swiglu=swiglu)
    authority = cache._layers[7]
    assert bank.count == 2 and bank.width == 64 and bank.dims == 128 and not bank.capturable
    assert len(cache.allocations) == 6 and len(ctx.reads) == 24 and len(set(ctx.reads)) == 24
    assert not any(".experts.1." in name or ".shared_expert." in name for name in ctx.reads)
    assert ctx.used == {"nvfp4": True}
    for local, expert in enumerate((2, 0)):
        for p, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            base = f"{PREFIX}.experts.{expert}.{projection}"
            expected = oracle_blob(ctx.payload[base + ".weight_packed"], ctx.payload[base + ".weight_scale"])
            assert torch.equal(authority[0][local, p] if p < 2 else authority[1][local], expected)
            for source, target in (("weight_global_scale", authority[2] if p < 2 else authority[3]),
                                   ("input_global_scale", authority[4] if p < 2 else authority[5])):
                raw = float(ctx.payload[base + "." + source][0])
                expected_scale = 1.0 / raw if source == "weight_global_scale" else raw
                direct = struct.unpack("f", struct.pack("f", expected_scale))[0]
                assert float(target[local, p if p < 2 else 0]) == direct
            read = ctx.reads.index(base + ".weight_packed")
            assert ctx.reads.index(base + ".weight_global_scale") < read
            assert ctx.reads.index(base + ".input_global_scale") < read


@pytest.mark.parametrize("damage", ["shape", "missing", "metadata", "full", "hardware", "device"])
def test_bad_native_contract_stops_before_authority_or_payload(damage):
    ctx, cache = Context(), Cache()
    name = PREFIX + ".experts.1.gate_proj.weight_packed"
    if damage == "shape": ctx.info[name] = ("U8", [1, 1])
    elif damage == "missing": del ctx.info[name]
    elif damage == "metadata": ctx.metadata.policy.defaults[0]["input_activations"] = None
    elif damage == "full": ctx.checkpoint_math = False
    elif damage == "hardware": ctx.own["nvfp4"] = False
    else: cache.device = torch.device("cuda:1")
    with pytest.raises(ValueError):
        load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, swiglu=swiglu)
    assert not cache.allocations and not ctx.reads and not cache._layers and not ctx.used


@pytest.mark.parametrize("field,value", [("input_global_scale", 0.), ("input_global_scale", float("nan")),
    ("weight_global_scale", float("inf")), ("input_global_scale", 1e-39)])
def test_invalid_scalar_stops_before_projection_codes_and_never_publishes_bank(field, value):
    ctx, cache = Context(), Cache()
    base = PREFIX + ".experts.0.gate_proj"
    ctx.payload[base + "." + field][0] = value
    with pytest.raises(ValueError, match="calibration|native FP32"):
        load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, swiglu=swiglu)
    assert not any(name.endswith(("weight_packed", "weight_scale")) for name in ctx.reads)
    assert not cache._layers and not ctx.used


@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_grouped_candidate_windows_preserve_every_pair_and_both_resource_limits(capacity):
    from tensorfold.cuda.nvfp4.grouped_a4 import _windows

    groups = {3: [(r, 0) for r in range(140)], 0: [(r, 1) for r in range(19)],
              2: [(r, 2) for r in range(91)], 1: [(r, 3) for r in range(7)]}
    batches = list(_windows(groups, capacity))
    got = []
    for pairs, ids in batches:
        assert 0 < len(pairs) <= 128 and 0 < len(ids) <= capacity
        assert ids == sorted({e for _, _, e in pairs})
        got.extend((e, row, slot) for row, slot, e in pairs)
    expected = [(e, row, slot) for e in sorted(groups) for row, slot in groups[e]]
    assert got == expected


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
def test_native_scale_admission_exhausts_byte_domain_without_mutation(dtype):
    from tensorfold.cuda.nvfp4.format import validate_native_a4_scales

    for byte in range(256):
        original = torch.tensor([[0, byte, 0x7e]], dtype=torch.uint8)
        scale = original.view(dtype)
        if byte in set(range(127)):
            validate_native_a4_scales(scale, name="independent byte-domain oracle")
        else:
            with pytest.raises(ValueError, match="finite unsigned E4M3"):
                validate_native_a4_scales(scale, name="independent byte-domain oracle")
        assert torch.equal(scale.view(torch.uint8), original)


@pytest.mark.parametrize("byte", [0x7f, 0x80, 0xff])
def test_invalid_routed_scale_payload_stops_before_pack_or_publication(monkeypatch, byte):
    from tensorfold.cuda.nvfp4 import experts_a4

    ctx, cache = Context(), Cache()
    name = PREFIX + ".experts.0.gate_proj.weight_scale"
    ctx.payload[name].view(torch.uint8).reshape(-1)[-1] = byte
    monkeypatch.setattr(experts_a4, "pack_projection_a4", lambda *args: pytest.fail("packed invalid native scales"))
    with pytest.raises(ValueError, match=r"gate_proj.weight_scale: native A4 block scales"):
        load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, swiglu=swiglu)
    assert len(ctx.reads) == 4 and ctx.reads[-1] == name
    assert not cache._layers and not ctx.used


def test_valid_zero_max_scales_are_preserved_and_unowned_invalid_payload_is_never_read(monkeypatch):
    from tensorfold.cuda.nvfp4 import format as fmt

    ctx, cache = Context(), Cache()
    untouched = PREFIX + ".experts.1.gate_proj.weight_scale"
    ctx.payload[untouched].view(torch.uint8).fill_(0xff)
    calls = []
    actual = fmt.validate_native_a4_scales

    def recorded(value, *, name):
        calls.append(name)
        return actual(value, name=name)

    monkeypatch.setattr(fmt, "validate_native_a4_scales", recorded)
    base = PREFIX + ".experts.2.gate_proj"
    scale = ctx.payload[base + ".weight_scale"].view(torch.uint8)
    scale.reshape(-1)[0], scale.reshape(-1)[-1] = 0x00, 0x7e
    bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, owned_ids=[2], swiglu=swiglu)
    assert bank.count == 1 and len(calls) == 3
    assert untouched not in ctx.reads
    expected = oracle_blob(ctx.payload[base + ".weight_packed"], scale)
    assert torch.equal(cache._layers[0][0][0, 0], expected)



def test_raw_ct_multiplier_boundary_has_independent_encoded_scale_expectation():
    from nvfp4_a4_inputs import ct_quant_reference

    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    raw = struct.unpack("f", struct.pack("I", 0x3e63ffff))[0]
    reconstructed = f32(1.0 / f32(1.0 / raw))
    x = torch.full((1, 64), .5, dtype=torch.bfloat16)
    _, direct = ct_quant_reference(x, raw)
    _, roundtrip = ct_quant_reference(x, reconstructed)
    assert torch.equal(direct, torch.full_like(direct, 0x0a))
    assert torch.equal(roundtrip, torch.full_like(roundtrip, 0x09))
    assert not torch.equal(direct, roundtrip)


def test_ct_native_calibration_admits_finite_original_multiplier_without_roundtrip_overflow():
    from tensorfold.cuda.nvfp4 import checkpoint
    from tensorfold.cuda.nvfp4.format import native_calibration

    raw = float(torch.finfo(torch.float32).max)
    act = float(torch.tensor(1., dtype=torch.float32) / torch.tensor(raw, dtype=torch.float32))
    assert checkpoint._inv(act) == float("inf")
    assert native_calibration(act, 1., name="original CT multiplier", act_inverse=raw) == act
    with pytest.raises(ValueError, match="native FP32"):
        native_calibration(act, 1., name="legacy inverse boundary")



@pytest.mark.parametrize("prefill", [False, True])
def test_bank_prepares_exact_scalar_reciprocals_once_and_run_never_recomputes_them(monkeypatch, prefill):
    from tensorfold.cuda.nvfp4 import checkpoint

    ctx, cache = Context(), Cache()
    original_inv, calls = checkpoint._inv, []

    def record(value):
        calls.append(value)
        return original_inv(value)

    monkeypatch.setattr(checkpoint, "_inv", record)
    bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, owned_ids=[2, 0], swiglu=swiglu)
    assert len(calls) == 3 * bank.count
    assert sum(t.numel() * t.element_size() for t in bank.prepared_inputs) == 12 * bank.count
    for source, prepared in zip(bank.scalars[2:], bank.prepared_inputs):
        assert prepared.device.type == "cpu" and prepared.dtype == torch.float32
        for row in range(bank.count):
            for projection in range(source.shape[1]):
                assert float(prepared[row, projection]) == original_inv(float(source[row, projection]))
    fields = cache._layers[0]
    expected = []
    for expert in range(bank.count):
        for projection in range(3):
            raw = float(bank.scalars[2 if projection < 2 else 3][expert, projection if projection < 2 else 0])
            act = float(bank.prepared_inputs[0 if projection < 2 else 1][expert, projection if projection < 2 else 0])
            expected.append((act, raw))
    observed = []

    def fake_matmul(mode, x, linear, *, f32=False):
        observed.append((linear.act, linear.act_inverse))
        return torch.full((x.shape[0], linear.n), linear.act,
                          dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)

    @contextmanager
    def lease(expert):
        yield fields, expert

    # This CPU control executes the actual bank grouping/projection/run body;
    # native arithmetic/device admission are independently covered on CUDA.
    monkeypatch.setattr(bank, "_check", lambda *a, **k: None)
    monkeypatch.setattr(bank, "_expert", lease)
    monkeypatch.setattr(bank, "device", torch.device("cpu"))
    monkeypatch.setattr(checkpoint, "matmul", fake_matmul)
    monkeypatch.setattr(checkpoint, "prompt", fake_matmul)
    monkeypatch.setattr(checkpoint, "_inv", lambda value: pytest.fail("recomputed a static reciprocal during run"))
    x = torch.ones((2, ctx.cfg.hidden), dtype=torch.bfloat16)
    picks = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)
    out = torch.empty((2, 2, ctx.cfg.hidden), dtype=torch.bfloat16 if prefill else torch.float32)
    bank.run(x, picks, out, prefill=prefill)
    assert observed == expected
    assert torch.count_nonzero(out[0, 1]) == 0
    for row, slot, expert in ((0, 0, 0), (1, 0, 1), (1, 1, 0)):
        assert torch.equal(out[row, slot], torch.full_like(out[row, slot], float(bank.prepared_inputs[1][expert, 0])))
