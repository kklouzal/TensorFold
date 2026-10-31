"""Affine MoE startup refuses unsupported cache mappings before loading payloads (CPU provider controls)."""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda import capacity, geometry, kv_arena
from tensorfold.families.qwen3_5_moe.cuda import weights
from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine
from tests.test_qwen27_startup_metadata import startup  # noqa: F401


def test_actual_affine_moe_startup_checks_mapping_support_before_loading(startup, monkeypatch):  # noqa: F811
    directory, _state, events, _options, _gathers = startup
    options = []

    def cache_geometry(_text, _world, _reserve, **kwargs):
        options.append(kwargs)
        return object()

    def admit(_directory, _context, _explicit, _runtime, factory, *_args, **_kwargs):
        factory({})
        events.append("admit")
        return {"context_window": 32, "cache_slots": 33}

    def load(*_args, **_kwargs):
        events.append("load")
        return NS(config=NS(eos=(1, 3)))

    monkeypatch.setattr(geometry, "gdn_geometry", cache_geometry)
    monkeypatch.setattr(capacity, "admit", admit)
    monkeypatch.setattr(weights, "load", load)
    engine = Qwen36Engine(directory, depth=0)
    try:
        assert engine.cache.keep == 4 and options == [{"mtp": False, "kept": 4}]
        assert events.index("device") < events.index("vmm") < events.index("admit") < events.index("load")
    finally:
        engine.close()


@pytest.mark.parametrize("depth", [0, 3])
def test_affine_moe_refuses_missing_vmm_before_loading_target_or_mtp(startup, monkeypatch, depth):  # noqa: F811
    directory, _state, events, _options, _gathers = startup
    failure = RuntimeError("CUDA device lacks virtual memory management")

    def refused(device):
        assert device == 0 and events == ["device"]
        raise failure

    monkeypatch.setattr(kv_arena, "driver", refused)
    with pytest.raises(RuntimeError) as caught:
        Qwen36Engine(directory, depth=depth)
    assert caught.value is failure and events == ["device"]
