"""Owned physical M32 projection buffers and preload accounting share one layout."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.qwen4_exp.cuda import padded_linears as prepared  # noqa: E402


def metadata():
    return {"hidden_size": 2560, "num_attention_heads": 24, "num_key_value_heads": 2,
            "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "hc_count": 4, "hc_lowrank": 320, "vocab_size": 248320, "ple_embed_dim": 2560,
            "layer_types": ["linear_attention", "full_attention", "linear_attention"]}


def local_config(world):
    return SimpleNamespace(hidden=2560, heads=24 // world, kv_heads=max(1, 2 // world), head_dim=256,
                           nk=16 // world, nv=48 // world, streams=4, low=320, index_heads=4,
                           index_dim=128, ple_dim=2560, layer_types=["linear", "attention", "linear"])


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [1, 3, 8, 18, 32])
def test_preload_geometry_matches_local_buffers_and_owned_storage(world, rows):
    c = local_config(world)
    w = SimpleNamespace(cfg=c, device=torch.device("cpu"), head=SimpleNamespace(n=248320 // world),
                        meta={"world": world})
    buffers = prepared.PaddedLinears(w, rows)
    logical_bytes = 0
    for name, (shape, itemsize) in buffers.layout.items():
        dtype = torch.float32 if itemsize == 4 else torch.float16
        live = buffers.tensor(name, shape, dtype)
        logical_bytes += live.numel() * live.element_size()
        assert live.shape == shape
        owner = buffers.owners[name]
        assert live.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()
        assert owner.is_contiguous()
        assert all(part.is_contiguous() for part in owner.unbind(0)) if live.ndim == 3 else live.is_contiguous()
    owned = [*buffers.owners.values(), buffers.input_storage, *buffers.partial.values()]
    actual = sum(t.numel() * t.element_size() for t in owned) - logical_bytes
    forecast = prepared.workspace_bytes(c, rows, w.head.n, world, enabled=True)
    assert actual == forecast
    assert prepared.workspace_bytes_from_text(metadata(), rows, world, enabled=True) == actual
    assert prepared.workspace_bytes_from_text({"text_config": metadata()}, rows, world, enabled=True) == actual
    if world == 8:
        assert buffers.layout["pa"][0][1] == 2688
        assert buffers.layout["proj"][0] == (2, rows, 2060)


@pytest.mark.parametrize("rows", [0, 1, 18, 32, 33])
def test_disabled_prompt_and_unsupported_windows_add_no_storage(rows, monkeypatch):
    monkeypatch.setattr(prepared, "_ENABLED", False)
    assert prepared.workspace_bytes_from_text({}, rows, 8) == 0
    assert prepared.workspace_bytes_from_text({}, rows, 8, enabled=True, prefill=True) == 0
    if rows in (0, 33):
        assert prepared.workspace_bytes_from_text({}, rows, 8, enabled=True) == 0


def test_owned_matrix_views_refuse_offsets_and_never_extend_storage():
    owner = torch.zeros((32, 16), dtype=torch.float16)
    live = owner[:3]
    full = prepared.PaddedLinears._matrix_view(live, [owner])
    assert full.shape == owner.shape and full.data_ptr() == owner.data_ptr()
    assert prepared.PaddedLinears._matrix_view(owner[1:4], [owner]) is None
    assert prepared.PaddedLinears._matrix_view(torch.zeros_like(live), [owner]) is None
    assert prepared.PaddedLinears._matrix_view(owner[:3, :8], [owner]) is None
    small = torch.zeros((3, 16), dtype=torch.float16)
    assert prepared.PaddedLinears._matrix_view(small, [small]) is None
    # A reduced head's contiguous flat-prefix output stays within its full owner.
    reduced = owner.reshape(-1)[:24].view(3, 8)
    bounded = prepared.PaddedLinears._matrix_view(reduced, [owner])
    assert bounded.shape == (32, 8) and bounded.numel() <= owner.numel()


def test_unknown_fields_preserve_requested_shape_and_type_and_layout_disagreements_raise():
    c = local_config(8)
    w = SimpleNamespace(cfg=c, device=torch.device("cpu"), head=SimpleNamespace(n=64), meta={"world": 8})
    buffers = prepared.PaddedLinears(w, 3)
    other = buffers.tensor("other", (7, 9), torch.float64, zero=True)
    assert other.shape == (7, 9) and other.dtype == torch.float64 and torch.count_nonzero(other) == 0
    with pytest.raises(ValueError, match="forecast"):
        buffers.tensor("mixed", (2, c.hidden), torch.float16)
    with pytest.raises(ValueError, match="forecast"):
        buffers.tensor("mixed", (3, c.hidden), torch.float32)
