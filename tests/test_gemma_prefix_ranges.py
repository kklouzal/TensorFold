"""CPU range transport controls; actual SM121 state/serving gates remain separate."""
from types import SimpleNamespace

import pytest
import torch

from gemma_cuda_target_metadata_test import tiny_config
from tensorfold.families.gemma4.cuda.state import State
from tensorfold.families.gemma4.cuda.weights import Config
from tensorfold.families.gemma4.cuda.dflash import DraftState


def target(mode="none"):
    raw = tiny_config()
    raw["text_config"]["num_hidden_layers"] = 4
    raw["text_config"]["layer_types"] = ["sliding_attention", "full_attention"] * 2
    config = Config.from_dict(raw)
    return State(config, "cpu", 256, rows=16, global_cache=mode,
                 cache_owner=object() if mode != "none" else None)


def fill_target(state, end):
    for layer in range(2):
        for row in range(256):
            state.raw[layer, row].view(torch.int16).fill_(0x3F00 + row + layer * 1024)
            state.inv[layer, row].view(torch.int32).fill_(0x3F000000 + row + layer * 65536)
        for row in range(max(0, end - state.ring), end):
            state.keys[layer, row % state.ring].view(torch.int16).fill_(0x3F00 + row + layer * 1024)
            state.values[layer, row % state.ring].view(torch.int16).fill_(0xBF00 + row + layer * 1024 - 65536)
    state.position = state.high_water = end


@pytest.mark.parametrize("mode", ["none", "k", "kv"])
def test_target_interior_range_wrap_detached_bytes_and_fixed_addresses(mode):
    source, dest = target(mode), target(mode)
    fill_target(source, 133)
    expected_bytes = {"target_full": 16 * 2 * (128 * 2 + 4),
                      "target_window": 16 * 2 * 64 * 2 * 2}
    assert source.cache_group_bytes(16) == expected_bytes
    payload = source.cache_export_rows(117, 133)
    assert {g: sum(t.numel() * t.element_size() for t in ts) for g, ts in payload.items()} == expected_bytes
    for values in payload.values():
        for tensor in values:
            assert all(tensor.untyped_storage().data_ptr() != live.untyped_storage().data_ptr()
                       for live in (source.raw, source.inv, source.keys, source.values))
    # Source mutation cannot change the immutable archive payload.
    source.raw.zero_()
    source.inv.zero_()
    source.keys.zero_()
    source.values.zero_()
    arrays = (dest.raw, dest.inv, dest.keys, dest.values)
    for value in arrays:
        value.fill_(-777)
    pointers = tuple(value.data_ptr() for value in arrays)
    dest.poisoned = True
    for group, values in payload.items():
        dest.cache_import_rows(group, 119, values, 2, 10)
    dest.cache_finish_restore(129)
    assert not dest.poisoned and dest.position == dest.high_water == 129
    assert dest.global_cache_ready is (mode == "none")
    assert pointers == tuple(value.data_ptr() for value in arrays)
    for layer in range(2):
        for row in range(119, 129):
            assert torch.all(dest.raw[layer, row].view(torch.int16) == 0x3F00 + row + layer * 1024)
            assert torch.all(dest.inv[layer, row].view(torch.int32) == 0x3F000000 + row + layer * 65536)
            assert torch.all(dest.keys[layer, row % dest.ring].view(torch.int16) == 0x3F00 + row + layer * 1024)
            assert torch.all(dest.values[layer, row % dest.ring].view(torch.int16) == 0xBF00 + row + layer * 1024 - 65536)
    assert torch.all(dest.raw[:, :119] == -777) and torch.all(dest.raw[:, 129:] == -777)
    untouched = sorted(set(range(dest.ring)) - {row % dest.ring for row in range(119, 129)})
    assert torch.all(dest.keys[:, untouched] == -777) and torch.all(dest.values[:, untouched] == -777)


def test_target_refuses_overwritten_uncommitted_and_invalid_frontiers():
    state = target()
    fill_target(state, 133)
    for start, stop in ((0, 16), (132, 134), (116, 133), (120, 120)):
        with pytest.raises(ValueError):
            state.cache_export_rows(start, stop)
    payload = state.cache_export_rows(117, 133)
    with pytest.raises(ValueError):
        state.cache_import_rows("target_full", 0, payload["target_full"], 0, 1)
    state.poisoned = True
    for position in (-1, 257, True):
        with pytest.raises(ValueError):
            state.cache_finish_restore(position)
        assert state.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        state.check(1)


def test_partial_target_import_failure_never_publishes_destination(monkeypatch):
    source, dest = target(), target()
    fill_target(source, 133)
    payload = source.cache_export_rows(117, 133)
    dest.poisoned = True
    primary = RuntimeError("second native transport failed")
    original, copies = torch.Tensor.copy_, []
    def copy(value, *args, **kwargs):
        copies.append(value)
        if len(copies) == 2:
            raise primary
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    with pytest.raises(RuntimeError) as caught:
        dest.cache_import_rows("target_full", 117, payload["target_full"], 0, 16)
    assert caught.value is primary and dest.poisoned and dest.position == 0
    with pytest.raises(RuntimeError, match="poisoned"):
        dest.check(1)


def draft():
    cfg = SimpleNamespace(layer_types=("sliding_attention", "full_attention", "sliding_attention"),
                          sliding_window=8, kv_heads=1, head_dim=8)
    return DraftState(cfg, "cpu", 256, 16, 5)


def test_draft_interior_range_wrap_independent_rows_and_payload_bytes():
    source, dest = draft(), draft()
    source.length = 133
    for layer, ring in enumerate(source.rings):
        for row in range(max(0, 133 - ring), 133):
            source.keys[layer][row % ring].view(torch.int16).fill_(0x3F00 + row + layer * 1024)
            source.values[layer][row % ring].view(torch.int16).fill_(0xBF00 + row + layer * 1024 - 65536)
    expected = {"draft_full": 16 * 8 * 2 * 2, "draft_window": 16 * 2 * 8 * 2 * 2}
    assert source.cache_group_bytes(16) == expected
    payload = source.cache_export_rows(117, 133)
    assert {g: sum(t.numel() * t.element_size() for t in ts) for g, ts in payload.items()} == expected
    live = (*source.keys, *source.values)
    for ts in payload.values():
        for value in ts:
            assert all(value.untyped_storage().data_ptr() != old.untyped_storage().data_ptr() for old in live)
    for value in (*source.keys, *source.values):
        value.zero_()
    for value in (*dest.keys, *dest.values):
        value.fill_(-777)
    pointers = tuple(value.data_ptr() for value in (*dest.keys, *dest.values))
    for group, values in payload.items():
        dest.cache_import_rows(group, 119, values, 2, 10)
    # Pair-level publication, not an individual import, owns the new length.
    assert dest.length == 0
    for layer, ring in enumerate(dest.rings):
        for row in range(119, 129):
            assert torch.all(dest.keys[layer][row % ring].view(torch.int16) == 0x3F00 + row + layer * 1024)
            assert torch.all(dest.values[layer][row % ring].view(torch.int16) == 0xBF00 + row + layer * 1024 - 65536)
        untouched = sorted(set(range(ring)) - {row % ring for row in range(119, 129)})
        assert torch.all(dest.keys[layer][untouched] == -777)
        assert torch.all(dest.values[layer][untouched] == -777)
    assert pointers == tuple(value.data_ptr() for value in (*dest.keys, *dest.values))
    for start, stop in ((0, 16), (132, 134), (116, 133)):
        with pytest.raises(ValueError):
            source.cache_export_rows(start, stop)


@pytest.mark.parametrize("family", ["target", "draft"])
def test_full_only_exports_retained_old_global_rows_without_window_work(family, monkeypatch):
    state = target() if family == "target" else draft()
    if family == "target":
        fill_target(state, 133)
        full, window = "target_full", "target_window"
    else:
        state.length = 133
        for row in range(133):
            state.keys[1][row].view(torch.int16).fill_(0x3F00 + row)
            state.values[1][row].view(torch.int16).fill_(0xBF00 + row - 65536)
        full, window = "draft_full", "draft_window"
    def no_window(*args, **kwargs):
        pytest.fail("full-only export allocated or gathered a sliding-window index")
    monkeypatch.setattr(torch, "arange", no_window)
    monkeypatch.setattr(torch.Tensor, "index_select", no_window)
    saved = state.cache_export_rows(0, 16, groups=(full,))
    assert tuple(saved) == (full,)
    for row in range(16):
        actual = saved[full][0][:, row] if family == "target" else saved[full][0][row]
        expected = torch.tensor([0x3F00 + row, 0x3F00 + row + 1024], dtype=torch.int16) if family == "target" else 0x3F00 + row
        if family == "target":
            assert torch.equal(actual.view(torch.int16).reshape(2, -1)[:, 0], expected)
        else:
            assert torch.all(actual.view(torch.int16) == expected)
    with pytest.raises(ValueError):
        state.cache_export_rows(0, 16, groups=(window,))


@pytest.mark.parametrize("family", ["target", "draft"])
def test_window_only_does_not_clone_unrequested_full_payload(family, monkeypatch):
    state = target() if family == "target" else draft()
    if family == "target":
        fill_target(state, 133)
        window = "target_window"
    else:
        state.length = 133
        for value in (*state.keys, *state.values):
            value.fill_(3)
        window = "draft_window"
    def no_full_clone(*args, **kwargs):
        pytest.fail("window-only export cloned a redundant full-attention payload")
    monkeypatch.setattr(torch.Tensor, "clone", no_full_clone)
    saved = state.cache_export_rows(117, 133, groups=(window,))
    assert tuple(saved) == (window,)
    for invalid in ((), (window, window), ("unknown",)):
        with pytest.raises(ValueError):
            state.cache_export_rows(117, 133, groups=invalid)
