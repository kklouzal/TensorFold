"""Header dispatch, exact direct calibration authority and independent byte math."""

import struct

import pytest

from tensorfold.cuda.nvfp4.modelopt_layout import a4_bank, plan_workspace_bytes, runtime_workspace_bytes

PREFIX = "model.language_model.layers.0.mlp"


def block(activation=True):
    return {"quant_method": "modelopt", "config_groups": {"experts": {
        "targets": [PREFIX + ".experts"], "weights": {"num_bits": 4, "type": "float", "group_size": 16},
        "input_activations": {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": False}
                             if activation else None}}}


def infos(count=3, width=64, dims=128):
    fields = {}
    for expert in range(count):
        for projection, n, k in (("gate_proj", width, dims), ("up_proj", width, dims), ("down_proj", dims, width)):
            base = f"{PREFIX}.experts.{expert}.{projection}"
            fields.update({base + ".weight": {"dtype": "U8", "shape": [n, k // 2]},
                           base + ".weight_scale": {"dtype": "F8_E4M3", "shape": [n, k // 16]},
                           base + ".weight_scale_2": {"dtype": "F32", "shape": []},
                           base + ".input_scale": {"dtype": "F32", "shape": []}})
    return fields


@pytest.mark.parametrize("native,activation,expected", [(True, True, True), (False, True, False),
                                                       (True, False, False), (False, False, False)])
def test_selection_follows_declaration_and_prepared_native_policy(native, activation, expected):
    assert a4_bank(infos(), PREFIX, 3, block(activation), native=native) is expected


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "dynamic", "group", "mixed"])
def test_complete_bank_header_or_declaration_damage_is_refused(damage):
    header, declaration = infos(), block()
    name = PREFIX + ".experts.2.down_proj.input_scale"
    if damage == "missing": del header[name]
    elif damage == "shape": header[name]["shape"] = [2]
    elif damage == "dtype": header[name]["dtype"] = "U8"
    elif damage == "dynamic": declaration["config_groups"]["experts"]["input_activations"]["dynamic"] = True
    elif damage == "group": declaration["config_groups"]["experts"]["input_activations"]["group_size"] = 32
    else:
        declaration["ignore"] = [PREFIX + ".experts.2.down_proj"]
    with pytest.raises(ValueError):
        a4_bank(header, PREFIX, 3, declaration, native=True)


def test_weight_only_does_not_inspect_or_require_an_unused_input_scale():
    header = infos()
    del header[PREFIX + ".experts.0.gate_proj.input_scale"]
    assert not a4_bank(header, PREFIX, 3, block(False), native=True)


def test_excluded_stacked_higher_precision_mtp_stays_a_separate_bank():
    header = {"mtp.layers.0.mlp.experts.gate_up_proj": {"dtype": "BF16", "shape": [3, 128, 128]}}
    assert not a4_bank(header, "mtp.layers.0.mlp", 3, block(), native=True)


def test_actual_projection_workspace_counts_every_live_payload_and_temporary():
    # D2560/W640: input code+scale planes2880, gate/up2560,
    # down code+scale360, SiLU+product2560 bytes per pair.
    assert plan_workspace_bytes(1, 1, 640, 2560) == 8360
    text = {"hidden_size": 2560, "moe_intermediate_size": 640, "num_experts_per_tok": 10}
    assert runtime_workspace_bytes(text, rows=8, prompt_rows=2048) == (8 + 2048) * (11 * 8360 + 20)
    assert runtime_workspace_bytes(text, rows=12, prompt_rows=2060, mtp_a4=True) == (12 + 2060 + 12) * (11 * 8360 + 20)
    assert runtime_workspace_bytes(text, rows=12, prompt_rows=2060, cached=True) == 2 * 128 * 8360


def test_host_authority_preserves_all_original_direct_scalars_and_payloads():
    torch = pytest.importorskip("torch")
    from modelopt_a4_inputs import Reader

    from tensorfold.cuda.nvfp4.experts_a4 import pack_projection_a4
    from tensorfold.cuda.nvfp4.modelopt_experts import load_modelopt_a4

    class Cache:
        device = torch.device("cuda:0")
        def __init__(self): self._layers, self.allocations = {}, []
        def allocate_authority(self, shape, *, dtype):
            self.allocations.append((tuple(shape), dtype))
            return torch.empty(shape, dtype=dtype)
        def register(self, layer, fields): self._layers[layer] = fields

    reader, cache = Reader(), Cache()
    bank = load_modelopt_a4(reader, PREFIX, reader.cfg, device="cuda:0", cache=cache, owned_ids=[2, 0], layer_id=7)
    assert bank.count == 2 and not bank.capturable and bank.direct_input
    fields = cache._layers[7]
    assert len(fields) == len(cache.allocations) == 6
    assert len(reader.reads) == len(set(reader.reads)) == 24
    assert not any(".experts.1." in name for name in reader.reads)
    for local, expert in enumerate((2, 0)):
        for p, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            base = f"{PREFIX}.experts.{expert}.{projection}"
            packed = pack_projection_a4(reader.payload[base + ".weight"], reader.payload[base + ".weight_scale"])
            assert torch.equal(fields[0][local, p] if p < 2 else fields[1][local], packed)
            for suffix, index in (("weight_scale_2", 2 if p < 2 else 3), ("input_scale", 4 if p < 2 else 5)):
                assert struct.pack("f", fields[index][local, p if p < 2 else 0]) == struct.pack("f", reader.payload[base + "." + suffix])
    assert not any(isinstance(value, torch.Tensor) for value in vars(bank).values())


@pytest.mark.parametrize("field,value", [("input_scale", 0.), ("input_scale", float("nan")),
                                         ("weight_scale_2", float("inf")), ("input_scale", 1e-39)])
def test_invalid_global_calibration_never_publishes_or_reads_projection_payload(field, value):
    torch = pytest.importorskip("torch")
    from modelopt_a4_inputs import Reader

    from tensorfold.cuda.nvfp4.modelopt_experts import load_modelopt_a4

    reader = Reader()
    name = PREFIX + ".experts.0.gate_proj."
    reader.payload[name + field] = torch.tensor(value)
    class Cache:
        device = torch.device("cuda:0")
        def __init__(self): self._layers = {}
        def allocate_authority(self, shape, *, dtype): return torch.empty(shape, dtype=dtype)
        def register(self, *_): pytest.fail("invalid bank published")
    with pytest.raises(ValueError, match="calibration|native FP32"):
        load_modelopt_a4(reader, PREFIX, reader.cfg, device="cuda:0", cache=Cache())
    assert not any(name.endswith((".weight", ".weight_scale")) for name in reader.reads)


def test_original_direct_scale_cannot_be_replaced_by_a_reciprocal_roundtrip():
    f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
    direct = struct.unpack("f", struct.pack("I", 0x3b350913))[0]
    assert direct != f32(1. / f32(1. / direct))


def test_real_cache_registration_admits_only_the_two_exact_complete_layouts():
    torch = pytest.importorskip("torch")
    import threading

    from tensorfold.cuda.expert_cache import HostExpertCache, _Policy, entry_bytes_for
    from tensorfold.cuda.nvfp4.host_layout import authority_fields_modelopt
    from tensorfold.cuda.nvfp4.modelopt_cache import ModelOptExpertCache, _formats

    def fields(native):
        dtypes = {"U8": torch.uint8, "I32": torch.int32, "F32": torch.float32}
        return tuple(torch.full(shape, 3 if dtype == "I32" else 7, dtype=dtypes[dtype])
                     for shape, dtype in authority_fields_modelopt(2, 64, 128, native_a4=native))
    a4, legacy = fields(True), fields(False)
    # Real register/replacement-view logic over CPU backing; no mocked layout
    # acceptance and no CUDA calls. Native lease ordering is tested on target.
    cache = object.__new__(ModelOptExpertCache)
    cache._formats = _formats(64, 128)
    cache.entry_bytes, cache.capacity = entry_bytes_for(a4), 1
    cache._pool = torch.full((cache.entry_bytes,), 0xa5, dtype=torch.uint8)
    cache._lock, cache._layers, cache._policy = threading.RLock(), {}, _Policy(1)
    cache._fields, cache.pin_policy = None, None
    cache._active = cache._closed = False
    cache._failure = None
    cache.register(0, a4)
    cache.register(1, legacy)
    cache.register(2, a4)
    assert [len(cache._layers[i].views) for i in range(3)] == [6, 4, 6]
    replacement = torch.empty(2 * cache.entry_bytes, dtype=torch.uint8)
    views, _ = cache._replacement_views(replacement, 2)
    assert [tuple(v.shape[0] for v in views[i]) for i in range(3)] == [(2,) * 6, (2,) * 4, (2,) * 6]
    wrong = (*a4[:4], a4[4][:, :1].contiguous(), a4[5])
    with pytest.raises(ValueError, match="only its admitted"):
        cache.register(3, wrong)
    assert 3 not in cache._layers and 3 not in cache._policy.frequency
    # The ordinary base cache still refuses a layout transition.
    with pytest.raises(ValueError, match="fixed hot-pool"):
        HostExpertCache._validate_layout(cache, entry_bytes_for(legacy), cache._layers[1].fields)


@pytest.mark.parametrize("policies,native,reason", [
    ([["checkpoint", True], ["checkpoint", True]], True, "all-native"),
    ([["checkpoint", True], ["checkpoint", False]], False, "mixed-capability-full"),
    ([["full", True], ["full", False]], False, "explicit-full"),
    ([["checkpoint", False]], False, "non-native-full")])
def test_expert_group_resolves_one_uniform_math_before_layout(policies, native, reason):
    from tensorfold.cuda.nvfp4.modelopt_layout import expert_group_policy
    assert expert_group_policy(policies) == (native, reason)


@pytest.mark.parametrize("policies", [[], [["checkpoint", True], ["full", True]], [["checkpoint", 1]],
                                       [["invalid", True]], [["checkpoint"]]])
def test_invalid_or_conflicting_rank_math_is_refused(policies):
    from tensorfold.cuda.nvfp4.modelopt_layout import expert_group_policy
    with pytest.raises(ValueError):
        expert_group_policy(policies)


def test_nonnative_checkpoint_declarations_are_validated_before_capability_dispatch():
    damaged = block()
    damaged["config_groups"]["experts"]["input_activations"]["group_size"] = 32
    with pytest.raises(ValueError, match="unsupported checkpoint activation"):
        a4_bank(infos(), PREFIX, 3, damaged, native=False, checkpoint=True)
    assert not a4_bank(infos(), PREFIX, 3, block(), native=False, checkpoint=True)
    assert not a4_bank(infos(), PREFIX, 3, damaged, native=False, checkpoint=False)


def test_mixed_bank_layout_uses_exact_allocation_order_and_native_host_transform_bound(monkeypatch):
    from tensorfold.cuda.read_budget import reader_host_bytes
    from tensorfold.families.qwen4_exp import ram_experts

    width, dims, count, layers = 640, 2560, 3, 12
    header = {}
    for layer in range(layers):
        header.update({key.replace("layers.0.", f"layers.{layer}."): value
                       for key, value in infos(count, width, dims).items()})
    header.update({"mtp.layers.0.mlp.experts.gate_up_proj": {"dtype": "BF16", "shape": [count, 2 * width, dims]},
                   "mtp.layers.0.mlp.experts.down_proj": {"dtype": "BF16", "shape": [count, dims, width]}})
    declaration = block()
    declaration["config_groups"]["experts"]["targets"] = ["Linear"]
    text = {"hidden_size": dims, "moe_intermediate_size": width, "num_experts": count,
            "num_experts_per_tok": 2, "num_hidden_layers": layers, "_quantization": declaration}
    monkeypatch.setattr(ram_experts, "headers", lambda _: header)
    plan = ram_experts._nvfp4_layout("unused", 1, text, True, native_a4=True, checkpoint=True)
    blob = 921600
    native_fields = (2 * count * blob, count * blob, 8 * count, 4 * count, 8 * count, 4 * count)
    assert plan.authority_allocations == native_fields * layers + native_fields[:4]
    assert plan.entry_bytes == 3 * blob + 64
    assert plan.host_bytes == layers * count * (3 * blob + 24) + count * (3 * blob + 12)
    assert len(plan.native_a4_banks) == layers and not any(b.startswith("mtp.") for b in plan.native_a4_banks)
    assert plan.loading_bytes >= reader_host_bytes(3 * blob)
    full = ram_experts._nvfp4_layout("unused", 1, text, True, native_a4=False, checkpoint=False)
    assert full.entry_bytes == 3 * blob + 32 and not full.native_a4_banks
    assert full.authority_allocations == native_fields[:4] * (layers + 1)


def test_independent_activation_oracle_saturates_finite_overflow_and_defines_nan_zero():
    torch = pytest.importorskip("torch")
    from modelopt_a4_inputs import quant_reference

    maximum = torch.finfo(torch.bfloat16).max
    x = torch.tensor([maximum, -maximum] * 8, dtype=torch.bfloat16).view(1, 16)
    codes, scales = quant_reference(x, .001964750699698925)
    assert torch.equal(codes, torch.full_like(codes, 0xf7))
    assert torch.equal(scales, torch.full_like(scales, 0x7e))
    zero = torch.tensor([0., -0.] * 8, dtype=torch.bfloat16).view(1, 16)
    codes, scales = quant_reference(zero, 1.)
    assert torch.count_nonzero(codes) == torch.count_nonzero(scales) == 0
    codes, scales = quant_reference(torch.full((1, 16), float("nan"), dtype=torch.bfloat16), 1.)
    assert torch.equal(codes, torch.full_like(codes, 0x77)) and torch.count_nonzero(scales) == 0


def test_common_input_proof_checks_every_expert_and_projection_by_exact_bits():
    torch = pytest.importorskip("torch")
    from tensorfold.cuda.nvfp4.modelopt_experts import uniform_input_bits

    values = torch.full((512, 2), .001964750699698925, dtype=torch.float32)
    assert uniform_input_bits(values)
    original = values.view(torch.int32)[-1, -1].item()
    values.view(torch.int32)[-1, -1] = original + 1
    assert not uniform_input_bits(values)
    values.view(torch.int32)[-1, -1] = original
    values.view(torch.int32)[0, 1] = original - 1
    assert not uniform_input_bits(values)
    with pytest.raises(ValueError, match="complete contiguous CPU"):
        uniform_input_bits(values[:, :1])
