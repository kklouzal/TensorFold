"""One ordered scalar owner is charged in both real admission estimates."""

import builtins
from math import prod

import pytest

from tensorfold.cuda import capacity
from tensorfold.cuda.nvfp4.host_layout import bulk_host_bytes_a4, scalar_batch_host_bytes_a4
from tensorfold.cuda.read_budget import f32_scalar_batch_host_bytes, reader_host_bytes
from tensorfold.families.qwen3_5.cuda.nvfp4_memory import weight_estimate
from tensorfold.families.qwen3_5_moe.ram_experts import Layout


def headers(bank, count, *, shared=False):
    owners = [f"{bank}.experts.{i}" for i in range(count)]
    owners += [f"{bank}.shared_expert"] if shared else []
    result = {}
    for owner in owners:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_global_scale", "input_global_scale"):
                result[f"{owner}.{projection}.{suffix}"] = {"dtype": "F32", "shape": [1]}
    return result


def expected(infos):
    import sys

    return f32_scalar_batch_host_bytes(len(infos), sum(sys.getsizeof(name) for name in infos))


def test_header_charge_uses_peak_bank_and_counts_potential_shared_owner():
    a = headers("model.language_model.layers.0.mlp", 256, shared=True)
    b = headers("model.language_model.layers.15.mlp", 3)
    assert len(a) == 1542
    assert scalar_batch_host_bytes_a4(a | b) == expected(a)
    assert scalar_batch_host_bytes_a4(a) > scalar_batch_host_bytes_a4(headers(
        "model.language_model.layers.0.mlp", 256))
    # Generated names' actual Unicode storage is charged, never a guessed
    # average name length or a rounded 6-KiB payload allowance.
    long = headers("language_model.model.layers.123456789.mlp", 256, shared=True)
    assert scalar_batch_host_bytes_a4(long) == expected(long)


def test_only_header_qualified_native_bank_scalars_enter_the_bound():
    unrelated = {"model.layers.0.mlp.gate_proj.input_global_scale": {"dtype": "F32", "shape": [1]},
                 "mtp.layers.0.mlp.experts.0.gate_proj.input_global_scale": {"dtype": "F32", "shape": [1]},
                 "model.layers.0.mlp.experts.0.unknown.input_global_scale": {"dtype": "F32", "shape": [1]},
                 "model.layers.0.mlp.experts.0.gate_proj.weight_scale": {"dtype": "F32", "shape": [1]},
                 "model.layers.0.mlp.experts.0.gate_proj.input_global_scale": {"dtype": "F16", "shape": [1]}}
    assert scalar_batch_host_bytes_a4(unrelated) == 0
    assert scalar_batch_host_bytes_a4({}) == f32_scalar_batch_host_bytes(0, 0) == 0


def test_many_small_banks_charge_cumulative_estimator_map_in_its_own_phase():
    infos = {}
    largest_batch = 0
    for layer in range(1024):
        bank = headers(f"model.layers.{layer}.mlp", 1)
        infos.update(bank)
        largest_batch = max(largest_batch, expected(bank))
    # Loading is serial, but the header scan's bank map is cumulative. A
    # one-bank-only charge would incorrectly remain at largest_batch here.
    assert scalar_batch_host_bytes_a4(infos) > largest_batch


@pytest.mark.parametrize("count,names", [(True, 1), (-1, 1), (1, False), (1, -1), (1.0, 1)])
def test_invalid_resource_counts_fail_explicitly(count, names):
    with pytest.raises(ValueError, match="nonnegative integers"):
        f32_scalar_batch_host_bytes(count, names)


def test_real_resident_and_offload_estimators_charge_same_batch_once(monkeypatch):
    bank = "model.language_model.layers.0.mlp"
    entries = headers(bank, 1)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"{bank}.experts.0.{projection}"
        entries[prefix + ".weight_packed"] = {"dtype": "U8", "shape": [64, 32]}
        entries[prefix + ".weight_scale"] = {"dtype": "F8_E4M3", "shape": [64, 4]}
    entries.update({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [128, 64]},
                    "lm_head.weight": {"dtype": "BF16", "shape": [128, 64]},
                    "model.norm.weight": {"dtype": "BF16", "shape": [64]}})
    monkeypatch.setattr(capacity, "headers", lambda path: entries)

    def resident(name, info):
        return prod(info["shape"]) * capacity.SIZES[info["dtype"]], 0

    transformed = Layout.transform(type("Plan", (), {"banks": (bank,)})(), resident)
    charge = scalar_batch_host_bytes_a4(entries)
    assert charge > 6 * 16 + 3 * 4096
    ordinary = builtins.__import__
    for transform in (resident, transformed):
        generic = capacity._estimate_weights_from_headers(entries, transform)
        largest = max(prod(info["shape"]) * capacity.itemsize(info, name)
                      for name, info in entries.items() if transform(name, info)[0])
        prior = reader_host_bytes(largest, max(generic.staging, largest))
        ordinary_host = prior
        _raw, resident_phase, planning = bulk_host_bytes_a4(entries)
        _, host = weight_estimate("validated-header-fixture", transform, scalar_batches=True,
                                  bulk_resident=transform is resident)
        if transform is resident:
            prior = reader_host_bytes(largest, max(generic.staging, largest, resident_phase))
        assert host == prior + charge + planning
        # Default dense admission remains outside this capability: no native
        # scalar-plan helper call or full additional header scan is performed.
        def imports(name, *args, **kwargs):
            if name == "tensorfold.cuda.nvfp4.host_layout":
                pytest.fail("dense scalar-plan import/scan")
            return ordinary(name, *args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(builtins, "__import__", imports)
            _, dense_host = weight_estimate("validated-header-fixture", transform)
            assert dense_host == ordinary_host
