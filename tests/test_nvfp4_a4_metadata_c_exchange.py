"""Real file-bank composition: metadata still gates native preparation/authority."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from nvfp4_a4_inputs import PREFIX, swiglu
from test_nvfp4_a4_bulk import FileContext, expected_authority
from test_nvfp4_a4_cache_control import CPUCache

from tensorfold.cuda.nvfp4 import bank_cpu
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4


@pytest.mark.parametrize("damage", ["activation", "weight"])
def test_current_metadata_stops_before_native_discovery_authority_or_payload(tmp_path, monkeypatch, request, damage):
    context = FileContext(count=2, width=64, dims=128).materialize(tmp_path)
    request.addfinalizer(context.files.close)
    cache = CPUCache(64, 128)
    if damage == "activation":
        context.metadata.policy.defaults[0]["input_activations"] = None
    else:
        context.metadata.policy.defaults[0]["weights"]["num_bits"] = 8

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid current metadata must fail before native/authority/payload work")

    monkeypatch.setattr(bank_cpu, "_producer", forbidden)
    monkeypatch.setattr(cache, "allocate_authority", forbidden)
    monkeypatch.setattr(context, "read_f32_scalars", forbidden)
    monkeypatch.setattr(context, "prepare_cpu_run", forbidden)
    with pytest.raises(ValueError):
        load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
    assert not context.reads and not context.used and not cache._layers


def test_public_policy_override_remains_authoritative_through_real_c_exchange_bank(tmp_path, request):
    context = FileContext(count=2, width=64, dims=128).materialize(tmp_path)
    request.addfinalizer(context.files.close)
    cache = CPUCache(64, 128)
    public = context.metadata.policy.resolve
    names = []

    def resolve(name):
        names.append(name)
        return public(name)

    context.metadata.policy.resolve = resolve
    bank = load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache,
                               owned_ids=[1, 0], swiglu=swiglu)
    assert names == [f"{PREFIX}.experts.{expert}.{projection}" for expert in (1, 0)
                     for projection in ("gate_proj", "up_proj", "down_proj")]
    assert bank.count == 2 and not context.where
    fields = cache.source_views(cache._layers[0].source[0])
    expected_authority(context, fields, [1, 0])
    # Published native authority contains primitive validation outcomes and
    # packed bytes; later policy mutation cannot rewrite its existing payload.
    context.metadata.policy.defaults[0]["weights"]["num_bits"] = 8
    expected_authority(context, fields, [1, 0])
