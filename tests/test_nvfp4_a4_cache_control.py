"""Native A4 source ownership and transactional view controls; no CUDA execution."""

import gc
import weakref
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nvfp4_a4_inputs import PREFIX, Context, swiglu
from test_nvfp4_a4_host_control import cpu_authority_packing as cpu_authority_packing  # noqa: PLC0414 - pytest fixture re-export.
from test_nvfp4_a4_host_control import oracle_blob

from tensorfold.cuda.expert_cache import HostExpertCache, _layout
from tensorfold.cuda.nvfp4 import cache_a4
from tensorfold.cuda.nvfp4.cache_a4 import A4ExpertCache, native_views
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4
from tensorfold.cuda.nvfp4.host_layout import entry_layout_a4


class CPUCache(A4ExpertCache):
    """Exercise real packing/bank preparation without creating a CUDA cache."""

    def __init__(self, width=64, dims=128):
        self.field_layout = entry_layout_a4(width, dims)
        self.entry_bytes, self.device = self.field_layout.entry_bytes, torch.device("cuda:0")
        self.allocations, self._layers = [], {}

    def allocate_authority(self, shape, *, dtype, tag):
        self.allocations.append((tuple(shape), dtype, tag))
        return torch.full(shape, 0xa5, dtype=dtype)

    def register(self, layer_id, payloads):
        assert len(payloads) == 1
        self._layers[layer_id] = SimpleNamespace(source=payloads, fields=_layout(payloads)[1])


def test_source_has_original_projection_bits_zero_padding_and_only_cache_ownership():
    context, cache = Context(), CPUCache()
    bank = load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache,
                                owned_ids=[2, 0], swiglu=swiglu)
    source, = cache._layers[0].source
    fields = cache.source_views(source)
    assert len(cache.allocations) == 1
    assert cache.host_bytes == source.numel() == bank.count * cache.entry_bytes
    assert source.numel() == sum(t.numel() * t.element_size() for t in fields) + 40 * bank.count
    expected = torch.zeros_like(source)
    for local, expert in enumerate((2, 0)):
        for p, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
            base = f"{PREFIX}.experts.{expert}.{projection}"
            packed = oracle_blob(context.payload[base + ".weight_packed"], context.payload[base + ".weight_scale"])
            spec = cache.field_layout.fields[0 if p < 2 else 1]
            offset = spec.offset + (p * packed.numel() if p < 2 else 0)
            expected[local, offset:offset + packed.numel()].copy_(packed)
        # Scalar bit/calibration expectations have an independent oracle in
        # the loader boundary suite; compare their exact stored bytes here.
        for field, spec in zip(fields[2:], cache.field_layout.fields[2:]):
            expected[local, spec.offset:spec.offset + spec.size].copy_(
                field[local].view(torch.uint8).reshape(-1))
    assert torch.equal(source, expected)
    assert not any(isinstance(value, torch.Tensor) for value in vars(bank).values())
    source_ref = weakref.ref(source)
    del source, fields, field
    cache._layers.clear()
    gc.collect()
    assert source_ref() is None


@pytest.mark.parametrize("byte", [0x7f, 0x80, 0xff])
def test_loader_preserves_native_payload_refusal_and_never_registers_partial_source(byte):
    context, cache = Context(), CPUCache()
    context.payload[PREFIX + ".experts.0.gate_proj.weight_scale"].view(torch.uint8)[0, 0] = byte
    with pytest.raises(ValueError, match="finite unsigned E4M3"):
        load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
    assert not cache._layers and not context.used and len(cache.allocations) == 1


def test_loader_rejects_mismatched_geometry_before_allocation_or_payload():
    context, cache = Context(), CPUCache(width=64, dims=64)
    with pytest.raises(ValueError, match="differs from the admitted expert field geometry"):
        load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
    assert not cache.allocations and not context.reads


@pytest.mark.parametrize("count", [1, 2, 4])
def test_native_views_keep_literal_entry_stride_at_singleton_and_larger_capacities(count):
    geometry = entry_layout_a4(64, 128)
    blob = torch.zeros((count, geometry.entry_bytes), dtype=torch.uint8)
    views = native_views(blob, geometry)
    for view, field in zip(views, geometry.fields):
        assert view.stride(0) * view.element_size() == geometry.entry_bytes
        assert view.data_ptr() == blob.data_ptr() + field.offset
        assert tuple(view.shape) == (count, *field.shape)
        assert view[0].is_contiguous()


def test_replacement_prepares_all_views_before_commit_and_commit_allocates_no_views(monkeypatch):
    cache = CPUCache()
    source = torch.zeros((2, cache.entry_bytes), dtype=torch.uint8)
    cache.register(3, (source,))
    old_pool = torch.zeros((cache.entry_bytes,), dtype=torch.uint8)
    old = native_views(old_pool.view(1, cache.entry_bytes), cache.field_layout)
    cache._native_views, cache._native_down = old, old[1].unsqueeze(1)
    new_pool = torch.zeros((4 * cache.entry_bytes,), dtype=torch.uint8)
    layer_views, controls = cache._replacement_views(new_pool, 4)
    assert cache._native_views is old
    assert len(layer_views[3]) == 1 and layer_views[3][0].shape == (4, cache.entry_bytes)
    prepared, down = controls
    assert all(t.stride(0) * t.element_size() == cache.entry_bytes for t in prepared)
    assert down.shape == (4, 1, prepared[1].shape[-1])
    assert down.data_ptr() == prepared[1].data_ptr()
    monkeypatch.setattr(cache_a4, "native_views", lambda *a: pytest.fail("allocated views after commit"))
    cache._commit_replacement(controls)
    assert cache._native_views is prepared and cache._native_down is down


def test_failed_replacement_and_retirement_retain_native_view_owners(monkeypatch):
    cache = CPUCache()
    source = torch.zeros((2, cache.entry_bytes), dtype=torch.uint8)
    cache.register(3, (source,))
    pool = torch.zeros((cache.entry_bytes,), dtype=torch.uint8)
    initial = native_views(pool.view(1, cache.entry_bytes), cache.field_layout)
    cache._native_views, cache._native_down = initial, initial[1].unsqueeze(1)
    failure = RuntimeError("private view preparation failed")

    def fail(*args):
        raise failure

    monkeypatch.setattr(cache_a4, "native_views", fail)
    with pytest.raises(RuntimeError) as caught:
        cache._replacement_views(torch.empty_like(pool), 1)
    assert caught.value is failure and cache._native_views is initial
    monkeypatch.setattr(HostExpertCache, "close", fail)
    with pytest.raises(RuntimeError) as caught:
        cache.close()
    assert caught.value is failure and cache._native_views is initial and cache._native_down is not None

    def retired(self):
        assert self._native_views is initial

    monkeypatch.setattr(HostExpertCache, "close", retired)
    cache.close()
    assert cache._native_views == () and cache._native_down is None
