"""Independent CT payload, reserved-workspace and transaction-peak oracles."""
import json
import math
import struct
import sys
from dataclasses import replace

import pytest

from tensorfold.cuda.capacity import SIZES
from tensorfold.families.qwen3_5_moe import ram_experts as ram


def checkpoint(path):
    cfg = {"hidden_size": 128, "moe_intermediate_size": 64, "num_experts": 3,
           "num_experts_per_tok": 2, "num_hidden_layers": 2}
    (path / "config.json").write_text(json.dumps({"text_config": cfg}))
    infos, offset = {}, 0

    def add(name, dtype, shape):
        nonlocal offset
        size = math.prod(shape) * SIZES[dtype]
        infos[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size

    for layer in range(2):
        base = f"model.language_model.layers.{layer}.mlp"
        for expert in (*[f"experts.{i}" for i in range(3)], "shared_expert"):
            for projection, n, k in (("gate_proj", 64, 128), ("up_proj", 64, 128), ("down_proj", 128, 64)):
                for field, dtype, shape in (("weight_packed", "U8", [n, k // 2]),
                    ("weight_scale", "F8_E4M3", [n, k // 16]), ("weight_global_scale", "F32", [1]),
                    ("input_global_scale", "F32", [1])):
                    add(f"{base}.{expert}.{projection}.{field}", dtype, shape)
        add(base + ".gate.weight", "BF16", [3, 128])
        add(base + ".shared_expert_gate.weight", "BF16", [1, 128])
    add("model.language_model.embed_tokens.weight", "BF16", [9, 128])
    encoded = json.dumps(infos).encode()
    with (path / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        stream.truncate(8 + len(encoded) + offset)
    return cfg, infos


def test_raw_codes_scales_and_six_fp32_globals_match_authority_and_aligned_pool(tmp_path):
    _, infos = checkpoint(tmp_path)
    # Per projection: one half-byte code and one E4M3 scale per16 weights,
    # plus two FP32 calibration values. Six independent aligned scalar fields
    # in the cache occupy64 bytes; raw scalar authority occupies24.
    matrix = 128 * 64
    payload = 3 * (matrix // 2 + matrix // 16) + 24
    entry = 3 * (matrix // 2 + matrix // 16) + 64
    plan = ram.layout(tmp_path, (2 * entry + 7) / ram.GIB)
    assert plan.host_bytes == 6 * entry == 83328
    assert plan.authority_padding_bytes == 6 * (entry - payload) == 240
    assert plan.entry_bytes == entry == 13888
    assert plan.slots == 2 and plan.gpu_bytes == 27776
    assert plan.authority_allocations == (3 * entry,) * 2
    assert plan.staging_bytes == 32768
    assert plan.loading_bytes == max(64 << 20, 8 * entry)  # shared estimator owns the physical Reader ring once
    assert plan.metadata_host_bytes > 5 * plan.largest_header_bytes
    assert plan.control_host_bytes > 6
    assert not plan.receipt()["capturable"]
    assert "prepared_activation_calibration_bytes" not in plan.receipt()
    assert plan.receipt()["upload_fields_per_expert_miss"] == 1
    resident = lambda name, info: (math.prod(info["shape"]) * SIZES[info["dtype"]], 0)
    transform = plan.transform(resident)
    routed = sum(resident(name, info)[0] for name, info in infos.items() if ".experts." in name)
    total = sum(resident(name, info)[0] for name, info in infos.items())
    assert sum(transform(name, info)[0] for name, info in infos.items()) == total - routed
    shared = "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight_packed"
    assert transform(shared, infos[shared]) == resident(shared, infos[shared])
    shared_input = "model.language_model.layers.0.mlp.shared_expert.gate_proj.input_global_scale"
    assert transform(shared_input, infos[shared_input]) == (4, 0)
    assert transform("draft.layers.0.mlp.experts.0.gate_proj.weight_packed", infos[shared]) == (4096, 0)


@pytest.mark.parametrize("budget", [0, -1, True, "1", float("nan"), float("inf")])
def test_bad_public_budget_rejects_before_checkpoint_io(budget):
    with pytest.raises(ValueError, match="finite positive"):
        ram.layout("/absent", budget)


def test_large_finite_integer_budget_caps_at_logical_experts_without_float_conversion(tmp_path):
    checkpoint(tmp_path)
    plan = ram.layout(tmp_path, 10 ** 1000)
    assert plan.slots == plan.expert_count == 6


@pytest.mark.parametrize("field,value", [("num_experts", True), ("num_hidden_layers", 0),
    ("hidden_size", 63), ("num_experts_per_tok", 4)])
def test_invalid_geometry_rejects_before_header_acquisition(tmp_path, monkeypatch, field, value):
    cfg, _ = checkpoint(tmp_path)
    cfg[field] = value
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(ram, "headers", lambda path: pytest.fail("invalid geometry opened payload headers"))
    with pytest.raises(ValueError):
        ram.layout(tmp_path, "auto")


def test_missing_global_or_incomplete_bank_cannot_remove_weights_from_gpu_estimate(tmp_path, monkeypatch):
    _, infos = checkpoint(tmp_path)
    missing = "model.language_model.layers.1.mlp.experts.2.up_proj.input_global_scale"
    del infos[missing]
    monkeypatch.setattr(ram, "headers", lambda path, **kwargs: infos)
    with pytest.raises(ValueError, match="incomplete"):
        ram.layout(tmp_path, "auto")


def pool(plan, free, *, old=0, kv=0, snapshots=0, target=0, draft=0, reserve=0, allocator=lambda n: n):
    return ram.plan_pool(plan, raw_free_bytes=free, existing_pool_bytes=old, future_kv_bytes=kv,
                         future_snapshot_bytes=snapshots, future_target_workspace_bytes=target,
                         future_draft_workspace_bytes=draft, device_reserve_bytes=reserve,
                         allocation_size=allocator)


def test_zero_optional_reserve_preserves_every_mandatory_future_allocation(tmp_path):
    checkpoint(tmp_path)
    plan = ram.layout(tmp_path, "auto")
    entry, native = plan.entry_bytes, plan.runtime_workspace_bytes
    result = pool(plan, native + 100 + 200 + 300 + 3 * entry + 9,
                  kv=100, snapshots=200, draft=300)
    assert result["slots"] == 3
    assert result["future_target_workspace_bytes"] == native
    assert result["reserved_bytes"] == native + 600
    assert result["device_reserve_bytes"] == 0
    assert result["transaction_free_after_reservations_bytes"] == 9
    assert not result["snapshot_is_reservation"]


def test_allocator_bound_and_old_live_pool_are_charged_at_transaction_peak(tmp_path):
    checkpoint(tmp_path)
    plan = ram.layout(tmp_path, "auto")
    native = plan.runtime_workspace_bytes
    # Independent small allocator oracle: the fifth cell crosses64KiB.
    rounded = lambda size: ((size + 65535) // 65536) * 65536
    result = pool(plan, native + 65536 + 17, old=plan.entry_bytes, allocator=rounded)
    assert result["slots"] == 4  # four13888-byte cells fit one64KiB block; five do not
    assert result["requested_pool_bytes"] == 55552
    assert result["new_allocation_bound_bytes"] == 65536
    assert result["transaction_free_after_reservations_bytes"] == 17
    assert result["final_free_after_reservations_bytes"] == 17 + plan.entry_bytes
    assert result["existing_pool_not_credited"]


def test_existing_pool_reuse_never_frees_live_storage_to_manufacture_transaction_room(tmp_path):
    checkpoint(tmp_path)
    plan = ram.layout(tmp_path, "auto")
    result = pool(plan, plan.runtime_workspace_bytes, old=plan.entry_bytes)
    assert not result["replace_pool"] and result["slots"] == 1
    assert result["new_allocation_bound_bytes"] == 0
    assert result["transaction_free_after_reservations_bytes"] == 0


def test_fixed_budget_never_silently_shrinks_when_workspace_does_not_fit(tmp_path):
    checkpoint(tmp_path)
    plan = ram.layout(tmp_path, 3 * 13888 / ram.GIB)
    with pytest.raises(ValueError, match="requested expert pool"):
        pool(plan, plan.runtime_workspace_bytes + 2 * plan.entry_bytes, old=plan.entry_bytes)
    with pytest.raises(ValueError, match="mandatory"):
        pool(plan, plan.runtime_workspace_bytes - 1, old=plan.entry_bytes)


@pytest.mark.parametrize("kwargs", [{"old": 1}, {"kv": True}, {"draft": -1}, {"allocator": None},
                                   {"allocator": lambda n: n - 1}, {"allocator": lambda n: float(n)}])
def test_malformed_budget_or_invalid_allocator_bound_rejects(tmp_path, kwargs):
    checkpoint(tmp_path)
    plan = ram.layout(tmp_path, "auto")
    with pytest.raises(ValueError):
        pool(plan, 1 << 30, **kwargs)


def test_full_heretic_geometry_is_not_a_gpu_residency_estimate(tmp_path, monkeypatch):
    cfg, infos = checkpoint(tmp_path)
    cfg.update(hidden_size=2048, moe_intermediate_size=512, num_hidden_layers=40, num_experts=256)
    monkeypatch.setattr(ram, "config", lambda path: cfg)
    monkeypatch.setattr(ram, "headers", lambda path, **kwargs: infos)
    # Header validator separately has a real124k-field gate; this oracle isolates
    # full-model arithmetic without creating payload or relying on that helper.
    monkeypatch.setattr(ram, "bank_bases_a4", lambda entries, text: tuple(f"model.layers.{i}.mlp" for i in range(40)))
    plan = ram.layout(tmp_path, "auto")
    assert plan.host_bytes == 18119639040 + 409600 == 18120048640 and plan.entry_bytes == 1769536
    assert plan.authority_padding_bytes == 40 * 40 * 256 == 409600
    assert plan.authority_allocations == (256 * 1769536,) * 40
    assert plan.expert_count == 10240
    assert plan.runtime_workspace_bytes == 5308416
    assert plan.gpu_bytes == plan.entry_bytes and plan.automatic
    assert replace(plan, automatic=False).host_bytes == plan.host_bytes


def test_header_budget_counts_utf8_padding_filtered_metadata_and_nested_extensions(tmp_path):
    _, infos = checkpoint(tmp_path)
    initial = ram.layout(tmp_path, "auto")
    payload = max(info["data_offsets"][1] for info in infos.values())
    nested = "x" * 10000
    next(iter(infos.values()))["extension"] = {"nested": [nested]}
    infos["__metadata__"] = {"owner": "\U00010000", "opaque": "y" * 10000}
    raw = json.dumps(infos, ensure_ascii=False).encode("utf-8") + b" " * 8192
    with (tmp_path / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)) + raw)
        stream.truncate(8 + len(raw) + payload)
    padded = ram.layout(tmp_path, "auto")
    assert padded.header_bytes == padded.largest_header_bytes == len(raw) + 8
    wire_growth = padded.header_bytes - initial.header_bytes
    assert padded.metadata_host_bytes - initial.metadata_host_bytes >= 5 * wire_growth + 3 * sys.getsizeof(nested)
    assert padded.host_bytes == initial.host_bytes and padded.entry_bytes == initial.entry_bytes


def test_header_budget_accumulates_all_validated_shards_without_rewalking(tmp_path):
    checkpoint(tmp_path)
    initial = ram.layout(tmp_path, "auto")
    extra = {"auxiliary": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
             "__metadata__": {"owner": "second"}}
    raw = json.dumps(extra).encode("utf-8")
    (tmp_path / "second.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"x")
    multiple = ram.layout(tmp_path, "auto")
    assert multiple.header_bytes == initial.header_bytes + len(raw) + 8
    assert multiple.largest_header_bytes == initial.largest_header_bytes
    assert multiple.metadata_host_bytes > initial.metadata_host_bytes
    assert multiple.host_bytes == initial.host_bytes


def runtime_text():
    return {"hidden_size": 128, "moe_intermediate_size": 64, "num_experts": 3,
            "num_experts_per_tok": 2, "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
            "vocab_size": 17, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
            "linear_key_head_dim": 4, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 3}


def test_single_runtime_keeps_all_prefix_states_and_every_retained_moe_shape():
    plan = ram.runtime_geometry(runtime_text(), streams=1, keep=4, max_rows=3, prompt_rows=512)
    small = plan.components(1024)
    # Recurrent matrix2*8*4*F32 + two convolution rows of24 BF16 =352B.
    # Four kept prefixes + current + commit =six independently retained states.
    assert small["future_snapshot_bytes"] == 6 * 352
    # One full-attention layer plus its growth copy, two heads of32 BF16 K/V.
    assert small["future_kv_bytes"] == 2 * 1024 * 256
    # Separate constructor-shape census: decode16=31652; prompt16..512
    # =19364+38664+77264+154464+308864+623832. This includes the
    # otherwise-unused grouped arrays and wide rank/hist for1536 pairs.
    # Dense record/operator bound=7867264; native A4 transient=458752.
    assert small["future_target_workspace_bytes"] == 7867264 + 31652 + 1222452 + 458752 + 17408
    assert plan.geometry.bytes_at(1024) == sum(small.values())
    assert plan.geometry.needed(1021) == sum(small.values())
    large = plan.components(4096)
    assert large["future_kv_bytes"] == 2 * 4096 * 256
    assert large["future_snapshot_bytes"] == small["future_snapshot_bytes"]
    assert large["future_target_workspace_bytes"] - small["future_target_workspace_bytes"] == 52224


def test_parallel_runtime_reserves_full_growth_of_private_and_kept_kv():
    plan = ram.runtime_geometry(runtime_text(), streams=4, keep=3, max_rows=3, prompt_rows=512)
    small, large = plan.components(1024), plan.components(4096)
    assert small["future_kv_bytes"] == (4 + 1) * 1024 * 256
    assert small["future_snapshot_bytes"] == (2 * 4 + 3 + 2) * 352 + (3 + 1) * 1024 * 256
    assert large["future_snapshot_bytes"] - small["future_snapshot_bytes"] == 4 * 3072 * 256
    # Calibration retains16,32,64-row decode buckets even for3-row requests.
    assert small["future_target_workspace_bytes"] == 28298752 + 221308 + 1222452 + 458752 + 69632
    assert plan.geometry.bytes_at(4096) == sum(large.values())
    assert large["future_kv_bytes"] + large["future_snapshot_bytes"] > 4 * small["future_kv_bytes"]


def test_runtime_freezes_header_inputs_and_accepts_inferred_head_dimension():
    text = runtime_text()
    del text["head_dim"]
    plan = ram.runtime_geometry(text, streams=1, keep=0, max_rows=1, prompt_rows=512)
    expected = plan.components(2048)
    text["hidden_size"] = 1
    text["layer_types"].clear()
    assert plan.components(2048) == expected
    assert plan.geometry.reserve == 1
    with pytest.raises(ValueError, match="cache_slots"):
        plan.components(-1)


@pytest.mark.parametrize("kwargs", [{"streams": True}, {"keep": -1}, {"max_rows": 129},
                                   {"prompt_rows": 511}, {"streams": 1 << 30}])
def test_runtime_rejects_invalid_policy_or_grouped_index_overflow(kwargs):
    policy = {"streams": 1, "keep": 4, "max_rows": 16, "prompt_rows": 512}
    policy.update(kwargs)
    with pytest.raises(ValueError):
        ram.runtime_geometry(runtime_text(), **policy)


@pytest.mark.parametrize("field,value", [("linear_conv_kernel_dim", 0), ("num_attention_heads", False),
                                       ("layer_types", ["linear_attention"]),
                                       ("layer_types", ["linear_attention", "unknown"])])
def test_runtime_rejects_bad_header_before_workspace_arithmetic(field, value):
    text = runtime_text()
    text[field] = value
    with pytest.raises(ValueError):
        ram.runtime_geometry(text, streams=1, keep=4, max_rows=16, prompt_rows=512)
