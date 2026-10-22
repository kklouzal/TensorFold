"""SF-first loading retires raw borrows before its larger code window."""

import gc
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from nvfp4_a4_inputs import PREFIX, Context, swiglu
from test_nvfp4_a4_bulk import FileContext, expected_authority
from test_nvfp4_a4_cache_control import CPUCache

from tensorfold.cuda.nvfp4 import bank_cpu, experts_a4, host_layout
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4


@pytest.mark.parametrize("direct", [False, True])
def test_sf_owner_and_all_exports_retire_before_larger_custom_order_code_chunks(tmp_path, monkeypatch, direct):
    count, width, dims = 12, 64, 128
    cap = 3 * 4096 + 5 * 3 * width * dims // 2
    monkeypatch.setattr(host_layout, "READER_PIECE_BYTES", cap)
    context = FileContext(count=count, width=width, dims=dims)
    generator = torch.Generator().manual_seed(34029)
    for name, value in context.payload.items():
        if name.endswith("weight_packed"):
            value.copy_(torch.randint(0, 256, value.shape, dtype=torch.uint8, generator=generator))
        elif name.endswith("weight_scale"):
            raw = value.view(torch.uint8)
            raw.copy_(torch.randint(0, 127, raw.shape, dtype=torch.uint8, generator=generator))
            raw.reshape(-1)[:127] = torch.randperm(127, generator=generator).to(torch.uint8)
    context.materialize(tmp_path, direct=direct)
    cache = CPUCache(width, dims)
    selected = [2, 11, 0, 10, 3, 9, 4, 8, 5, 7, 1, 6]
    exports, events, map_pointer = [], [], []
    original_array, original_prepare = bank_cpu._array, context.prepare_cpu_run
    original_scales, original_words = experts_a4.pack_bank_scales_a4, experts_a4.pack_bank_words_a4

    def array(value):
        result = original_array(value)
        exports.append(weakref.ref(result))
        return result

    def scales(payload, up, down, indices, n, k):
        events.append(("scales", indices.numel()))
        map_pointer.append(indices.untyped_storage().data_ptr())
        original_scales(payload, up, down, indices, n, k)
        assert all(ref() is None for ref in exports)

    def words(payload, up, down, indices, n, k):
        events.append(("words", indices.numel()))
        assert indices.untyped_storage().data_ptr() == map_pointer[0]
        original_words(payload, up, down, indices, n, k)

    def prepare(names, **geometry):
        run = original_prepare(names, **geometry)
        if geometry["dtype"] != "U8" or run is None:
            return run

        def read(first, stop):
            # The actual Reader's returned Tensor and every _base, plus all
            # four ndarray/DLPack export chains, must be gone before I/O.
            assert events and all(ref() is None for ref in context.run_refs)
            assert all(ref() is None for ref in exports)
            return run(first, stop)

        return read

    monkeypatch.setattr(bank_cpu, "_array", array)
    monkeypatch.setattr(experts_a4, "pack_bank_scales_a4", scales)
    monkeypatch.setattr(experts_a4, "pack_bank_words_a4", words)
    context.prepare_cpu_run = prepare
    try:
        load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache,
                             owned_ids=selected, swiglu=swiglu)
        assert events == [("scales", 12), ("words", 5), ("words", 5), ("words", 2)]
        assert list(map(len, context.run_reads)) == [36, 15, 15, 6]
        expected_authority(context, cache.source_views(cache._layers[0].source[0]), selected)
        assert not context.where and all(ref() is None for ref in context.run_refs)
        assert all(ref() is None for ref in exports)
    finally:
        context.files.close()


def test_invalid_sf_keeps_raw_owner_through_original_order_and_never_enters_stages(tmp_path, monkeypatch):
    context = FileContext(count=12)
    context.payload[PREFIX + ".experts.2.gate_proj.weight_scale"].view(torch.uint8)[0, 0] = 0xff
    context.materialize(tmp_path)
    canonical = Context(count=12)
    canonical.payload = context.payload
    cache = CPUCache()
    original_read = context.read

    def read(name, device="cpu"):
        if name.endswith("weight_packed"):
            assert context.run_refs and any(ref() is not None for ref in context.run_refs)
        return original_read(name, device)

    def forbidden(*_args):
        raise AssertionError("invalid SF entered a private staged write")

    context.read = read
    monkeypatch.setattr(experts_a4, "pack_bank_scales_a4", forbidden)
    monkeypatch.setattr(experts_a4, "pack_bank_words_a4", forbidden)
    try:
        with pytest.raises(ValueError) as reference:
            load_host_experts_a4(canonical, PREFIX, canonical.cfg, "cuda:0", cache=CPUCache(), swiglu=swiglu)
        with pytest.raises(ValueError) as actual:
            load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        assert type(actual.value) is type(reference.value) and str(actual.value) == str(reference.value)
        assert len(context.run_reads) == 1 and not cache._layers and not context.used
        assert not any(".experts.3." in name and name.endswith("weight_packed") for name in context.reads)
        del actual, reference
        gc.collect()
        assert all(ref() is None for ref in context.run_refs)
    finally:
        context.files.close()


@pytest.mark.parametrize("after_success", [False, True])
def test_code_read_failure_after_sf_write_never_publishes_or_retains_raw_owner(tmp_path, monkeypatch, after_success):
    count, width, dims = 5, 64, 512
    monkeypatch.setattr(host_layout, "READER_PIECE_BYTES", 3 * 4096 + 2 * 3 * width * dims // 2)
    context = FileContext(count=count, width=width, dims=dims).materialize(tmp_path)
    original_prepare, failure = context.prepare_cpu_run, OSError("later code read failed")
    attempts = []

    def prepare(names, **geometry):
        run = original_prepare(names, **geometry)
        if geometry["dtype"] != "U8" or run is None:
            return run

        def read(first, stop):
            assert all(ref() is None for ref in context.run_refs)
            attempts.append((first, stop))
            if len(attempts) == 1 + int(after_success):
                raise failure
            return run(first, stop)

        return read

    context.prepare_cpu_run = prepare
    cache = CPUCache(width, dims)
    try:
        with pytest.raises(OSError) as caught:
            load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        assert caught.value is failure and not cache._layers and not context.used
        assert len(context.run_reads) == 1 + int(after_success)
        # Failed reads never consumed their declared names; earlier successful
        # windows and the entire already-completed SF interval were consumed.
        packed_names = [name for name in context.where if name.endswith("weight_packed")]
        assert len(packed_names) == 3 * (count - 2 * int(after_success))
        assert not any(name.endswith("weight_scale") for name in context.where)
        del caught
        failure.__traceback__ = None
        gc.collect()
        assert all(ref() is None for ref in context.run_refs)
    finally:
        context.files.close()
