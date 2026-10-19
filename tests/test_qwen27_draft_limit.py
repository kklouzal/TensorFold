"""Companion position bounds are independent of target YaRN and precede payload loading."""

import json

import pytest
from cuda_27b_headers import _write

from tensorfold.cuda import capacity
from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
from tests import test_qwen27_startup_metadata as startup_controls

startup = startup_controls.startup


def companion(directory, limit):
    cfg = {"dflash_config": {"block_size": 8}}
    if limit != "absent":
        cfg["max_position_embeddings"] = limit
    return _write(directory / "draft", cfg, [("candidate_selector.hidden_projection.weight", "BF16", [16, 64])])


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "32"])
def test_invalid_explicit_companion_limit_refuses_before_device(startup, value):
    directory, _, events, options, _ = startup
    draft = companion(directory, value)
    with pytest.raises(ValueError, match="max_position_embeddings.*positive integer"):
        Qwen27Engine(directory, draft)
    assert events == options == []


def test_explicit_overbound_refuses_before_device_even_when_target_yarn_is_enabled(startup):
    directory, _, events, options, _ = startup
    config = json.loads((directory / "config.json").read_text())
    config["text_config"]["max_position_embeddings"] = 32
    (directory / "config.json").write_text(json.dumps(config))
    draft = companion(directory, 32)
    with pytest.raises(ValueError, match="DFlash2 declares at most 32.*requested context 33"):
        Qwen27Engine(directory, draft, context=33, context_explicit=True, yarn_factor=4)
    assert events == options == []


def test_automatic_admission_is_checked_before_any_target_payload(startup, monkeypatch):
    directory, _, events, options, _ = startup
    draft = companion(directory, 32)
    monkeypatch.setattr(capacity, "admit", lambda *a, **k: {"context_window": 33})
    with pytest.raises(ValueError, match="DFlash2 declares at most 32.*admitted context 33"):
        Qwen27Engine(directory, draft)
    assert options == [] and events == ["device", "vmm"]


@pytest.mark.parametrize("limit", [32, "absent"])
def test_exact_bound_and_existing_custom_config_without_bound_remain_supported(startup, monkeypatch, limit):
    from tensorfold.families.qwen3_5.cuda import dflash2

    directory, _, _, options, _ = startup
    draft = companion(directory, limit)
    monkeypatch.setattr(dflash2, "DFlash2", lambda *a, **k: object())
    made = Qwen27Engine(directory, draft, context=32, context_explicit=True)
    try:
        assert made.context_window == 32 and len(options) == 1
    finally:
        made.close()


def test_target_only_context_does_not_borrow_a_companion_bound(startup):
    directory, _, _, options, _ = startup
    companion(directory, 1)
    made = Qwen27Engine(directory, None, context=33, context_explicit=True)
    try:
        assert len(options) == 1
    finally:
        made.close()


@pytest.mark.parametrize("streams", [1, 4])
def test_workspace_uses_reachable_fixed_stochastic_rows_and_preserves_map_loading_overlap(startup, monkeypatch, streams):
    import torch

    from tensorfold.cuda import sampling, speculative
    from tensorfold.families.qwen3_5.cuda import dflash2

    directory, _, _, _, _ = startup
    draft = companion(directory, 32)
    text = {"hidden_size": 64, "intermediate_size": 128, "vocab_size": 64}
    monkeypatch.setattr(capacity, "config", lambda path: text)
    monkeypatch.setattr(dflash2, "DFlash2", lambda *a, **k: object())
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", lambda value: None)
    monkeypatch.setattr(Qwen27Engine, "_prepare_serving", lambda self, directory, **kw: setattr(self, "scheduler", None))
    target_calls, packet_calls, admissions = [], [], []
    effective, packet = sampling.effective_workspace, speculative.chain_workspace

    def target(rows, width, **kwargs):
        target_calls.append((rows, width, kwargs))
        return effective(rows, width, **kwargs)

    def proposed(rows, width, **kwargs):
        packet_calls.append((rows, width, kwargs))
        return packet(rows, width, **kwargs)

    original_admit = capacity.admit

    def admitted(*args, **kwargs):
        admissions.append(kwargs)
        return original_admit(*args, **kwargs)

    monkeypatch.setattr(sampling, "effective_workspace", target)
    monkeypatch.setattr(speculative, "chain_workspace", proposed)
    monkeypatch.setattr(capacity, "admit", admitted)
    made = Qwen27Engine(directory, draft, max_rows=128, tree_rows=16, streams=streams, context=32)
    try:
        assert target_calls == [(8 * streams, 64, {"world": 1})]
        assert packet_calls == [(7, 16, {"streams": streams})]
        assert admissions[0]["host_resident"] == 64 * 8
        assert admissions[0]["device_workspace_bytes"] >= effective(8 * streams, 64)["device_bytes"]
        assert admissions[0]["host_workspace_bytes"] >= effective(8 * streams, 64)["host_bytes"]
    finally:
        made.close()
