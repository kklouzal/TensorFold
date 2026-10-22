"""Direct producer, state/error/owner contracts and independent complete byte oracle."""

import gc
import weakref
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from test_nvfp4_a4_native_cpu import inputs

from tensorfold.cuda.nvfp4 import bank_cpu, experts_a4

HEADERS = ("bank_exchange_core.h", "bank_exchange_kernel.h")


@pytest.fixture(scope="module", params=["cpython", "pybind"])
def frontend(request):
    assert bank_cpu._producer() is not None, "the pinned provider must expose public DLPack1.3"
    if request.param == "cpython":
        module = bank_cpu._load("bank_exchange.cpp", HEADERS)
        return lambda callback: module.prepare(torch.Tensor, torch.Tensor.__dlpack_c_exchange_api__, callback)
    module = bank_cpu._load("bank_exchange_pybind.cpp", HEADERS)
    return lambda callback: module.prepare(torch.Tensor, torch.Tensor.__dlpack_c_exchange_api__, callback,
                                           module._pack, module._Context, module)


def test_prepared_entry_exposes_the_actual_strong_module_owner_without_loading_another(frontend, monkeypatch):
    prepared = frontend(torch.autograd.graph.increment_version)
    owner = bank_cpu._native_module(prepared)
    assert type(owner) is ModuleType
    assert prepared.__module__ == owner.__name__
    assert owner.__file__.endswith(owner.__name__ + ".so")
    reference = weakref.ref(owner)
    del owner

    def forbidden(*_args, **_kwargs):
        raise AssertionError("provider diagnostics must not load another module")

    monkeypatch.setattr(bank_cpu, "_load", forbidden)
    gc.collect()
    assert bank_cpu._native_module(prepared) is reference()
    context = prepared.__self__
    if type(context) is tuple:
        assert len(context) == 4 and context[3] is reference()
    else:
        with pytest.raises(AttributeError):
            context.module_owner = ModuleType("unrelated")


def test_prepared_module_owner_survives_external_module_reference_retirement():
    module = bank_cpu._load("bank_exchange.cpp", HEADERS)
    reference = weakref.ref(module)
    prepared = module.prepare(torch.Tensor, torch.Tensor.__dlpack_c_exchange_api__, torch.autograd.graph.increment_version)
    del module
    gc.collect()
    codes, scales, fields, indices, expected, _, _ = inputs(64, 64)
    prepared(codes, scales, *fields, indices, 64, 64)
    assert bank_cpu._native_module(prepared) is reference()
    assert all(torch.equal(value, old) for value, old in zip(fields, expected, strict=True))


def test_pybind_cold_binding_rejects_an_unrelated_module_before_creating_a_context():
    module = bank_cpu._load("bank_exchange_pybind.cpp", HEADERS)
    with pytest.raises(ValueError, match="do not belong"):
        module.prepare(torch.Tensor, torch.Tensor.__dlpack_c_exchange_api__, torch.autograd.graph.increment_version,
                       module._pack, module._Context, ModuleType("unrelated"))


@pytest.mark.parametrize("member", ["_pack", "_Context"])
def test_pybind_cold_member_lookup_preserves_the_exact_python_failure(member):
    module = bank_cpu._load("bank_exchange_pybind.cpp", HEADERS)
    primary = MemoryError("injected cold member equality failure")
    armed = False

    class Key(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            if armed and str.__eq__(self, other):
                raise primary
            return str.__eq__(self, other)

    owner = ModuleType("cold_members")
    owner.__dict__[Key(member)] = getattr(module, member)
    other = "_Context" if member == "_pack" else "_pack"
    owner.__dict__[other] = getattr(module, other)
    armed = True
    with pytest.raises(MemoryError) as outcome:
        module.prepare(torch.Tensor, torch.Tensor.__dlpack_c_exchange_api__, torch.autograd.graph.increment_version,
                       module._pack, module._Context, owner)
    assert outcome.value is primary


@pytest.mark.parametrize("width,dims", [(64, 64), (64, 128), (192, 320)])
@pytest.mark.parametrize("strided", [False, True])
def test_complete_byte_oracle_capacity_version_source_and_owner_contracts(frontend, width, dims, strided):
    codes, scales, fields, indices, expected, storage, expected_storage = inputs(width, dims, strided=strided)
    owners = (codes, scales, *fields, indices)
    before = [(value.data_ptr(), value.untyped_storage().resizable(), value._version, value.clone()) for value in owners]
    events = []

    def increment(targets):
        assert tuple(map(id, targets)) == tuple(map(id, fields))
        assert all(torch.equal(field, old[3]) for field, old in zip(fields, before[2:4], strict=True))
        events.append("version")
        # Collection during the reentrant boundary must not release any owner.
        gc.collect()
        torch.autograd.graph.increment_version(targets)

    frontend(increment)(codes, scales, *fields, indices, width, dims)
    assert events == ["version"]
    assert all(torch.equal(field, reference) for field, reference in zip(fields, expected, strict=True))
    if strided:
        assert torch.equal(storage, expected_storage)
    for position, (value, (pointer, capacity, version, original)) in enumerate(zip(owners, before, strict=True)):
        assert value.data_ptr() == pointer and value.untyped_storage().resizable() == capacity
        if position in (2, 3):
            expected_bumps = 2 if strided else 1
            assert value._version == version + expected_bumps
        else:
            assert value._version == version and torch.equal(value, original)


def test_nonzero_storage_offsets_strides_and_physical_padding(frontend):
    width, dims = 192, 320
    codes, scales, _, indices, expected, _, _ = inputs(width, dims)
    blob = width * dims * 9 // 16
    code_backing = torch.full((codes.numel() + 3,), 0x5a, dtype=torch.uint8)
    scale_backing = torch.full((scales.numel() + 5,), 0x5a, dtype=torch.uint8)
    code_backing[3:].copy_(codes)
    scale_backing[5:].copy_(scales)
    up_stride, down_stride = 2 * blob + 12, blob + 20
    up_storage = torch.full((4 * up_stride + 16,), 0xa5, dtype=torch.uint8)
    down_storage = torch.full((4 * down_stride + 24,), 0xa5, dtype=torch.uint8)
    up = up_storage.as_strided((4, 2, blob), (up_stride, blob, 1), storage_offset=4)
    down = down_storage.as_strided((4, blob), (down_stride, 1), storage_offset=8)
    expected_up, expected_down = up_storage.clone(), down_storage.clone()
    expected_up.as_strided(up.shape, up.stride(), storage_offset=4).copy_(expected[0])
    expected_down.as_strided(down.shape, down.stride(), storage_offset=8).copy_(expected[1])
    source_before = code_backing.clone(), scale_backing.clone()
    frontend(torch.autograd.graph.increment_version)(code_backing[3:], scale_backing[5:], up, down, indices, width, dims)
    assert torch.equal(up_storage, expected_up) and torch.equal(down_storage, expected_down)
    assert torch.equal(code_backing, source_before[0]) and torch.equal(scale_backing, source_before[1])


@pytest.mark.parametrize("fault", ["negative", "large", "duplicate", "truncated", "wrong_dtype", "rank",
                                 "stride", "unaligned", "input_alias", "map_alias", "dimension"])
def test_native_invalid_geometry_never_writes_authority(frontend, fault):
    codes, scales, fields, indices, _, _, _ = inputs(64, 64)
    up, down = fields
    width, dims = 64, 64
    if fault == "negative":
        indices[0] = -1
    elif fault == "large":
        indices[0] = 4
    elif fault == "duplicate":
        indices[1] = indices[0]
    elif fault == "truncated":
        codes = codes[:-1]
    elif fault == "wrong_dtype":
        up = up.view(torch.int8)
    elif fault == "rank":
        up = up.reshape(4, -1)
    elif fault == "stride":
        up = torch.full((up.numel() * 2,), 0xa5, dtype=torch.uint8).as_strided(up.shape, (up.stride(0) * 2, up.stride(1) * 2, 2))
    elif fault == "unaligned":
        up = torch.full((up.numel() + 1,), 0xa5, dtype=torch.uint8)[1:].view(up.shape)
    elif fault == "input_alias":
        codes = up.reshape(-1)[:codes.numel()]
    elif fault == "map_alias":
        indices = up.reshape(-1)[:indices.numel() * 8].view(torch.int64)
    else:
        dims = 2**31
    arguments = (codes, scales, up, down, indices)
    before = [value.clone() for value in arguments]
    with pytest.raises((ValueError, TypeError)):
        frontend(torch.autograd.graph.increment_version)(*arguments, width, dims)
    assert all(torch.equal(value, old) for value, old in zip(arguments, before, strict=True))


def test_producer_error_precedes_version_callback_and_write(frontend):
    codes, scales, fields, indices, _, _, _ = inputs(64, 64)
    meta = torch.empty(codes.shape, dtype=torch.uint8, device="meta")
    before = [value.clone() for value in fields]
    events = []

    def increment(_targets):
        events.append("unexpected version callback")

    with pytest.raises(RuntimeError):
        frontend(increment)(meta, scales, *fields, indices, 64, 64)
    assert events == []
    assert all(torch.equal(value, old) for value, old in zip(fields, before, strict=True))


def test_version_callback_failure_retains_exact_primary_and_unwritten_authority(frontend):
    codes, scales, fields, indices, _, _, _ = inputs(64, 64)
    before = [value.clone() for value in fields]
    primary = MemoryError("injected version callback allocation failure")

    def increment(targets):
        assert targets[0] is fields[0] and targets[1] is fields[1]
        raise primary

    with pytest.raises(MemoryError) as outcome:
        frontend(increment)(codes, scales, *fields, indices, 64, 64)
    assert outcome.value is primary
    assert all(torch.equal(value, old) for value, old in zip(fields, before, strict=True))


@pytest.mark.parametrize("position", [0, 1])
def test_raw_geometry_rejects_reentry_before_export_or_version_update(frontend, position):
    codes, scales, fields, indices, _, _, _ = inputs(64, 64)
    original = [(field.shape, field.data_ptr(), field._version, field.clone()) for field in fields]
    events = []

    class MutatingIndex:
        def __index__(self):
            events.append("geometry mutation")
            fields[0].resize_(1)
            return 64

    def increment(_targets):
        events.append("unexpected version update")

    geometry = [64, 64]
    geometry[position] = MutatingIndex()
    # A Meta producer would fail if export ran first. The exact-int guard must
    # reject before either producer or either potentially reentrant callback.
    meta = torch.empty(codes.shape, dtype=torch.uint8, device="meta")
    with pytest.raises(TypeError, match="exact integer width and dims"):
        frontend(increment)(meta, scales, *fields, indices, *geometry)
    assert events == []
    for field, (shape, pointer, version, value) in zip(fields, original, strict=True):
        assert field.shape == shape and field.data_ptr() == pointer and field._version == version
        assert torch.equal(field, value)


def test_public_numpy_integer_geometry_keeps_original_provider_and_exact_bytes(monkeypatch):
    import numpy as np

    codes, scales, fields, indices, expected, _, _ = inputs(64, 64)

    def forbidden():
        raise AssertionError("int-like geometry must not compile or call direct producer")

    monkeypatch.setattr(bank_cpu, "_ext", forbidden)
    bank_cpu.pack(codes, scales, *fields, indices, np.int64(64), np.int64(64))
    assert all(torch.equal(field, value) for field, value in zip(fields, expected, strict=True))


def test_gc_can_retire_a_callback_binding_cycle(frontend):
    class Callback:
        def __call__(self, targets):
            torch.autograd.graph.increment_version(targets)

    callback = Callback()
    function = frontend(callback)
    callback.function = function
    reference = weakref.ref(callback)
    del callback, function
    gc.collect()
    assert reference() is None


def test_public_adapter_selects_direct_without_creating_numpy_aliases(monkeypatch):
    codes, scales, fields, indices, expected, _, _ = inputs(64, 64)

    def forbidden(*_args):
        raise AssertionError("plain C exchange must not build or export the NumPy route")

    monkeypatch.setattr(bank_cpu, "_numpy_ext", forbidden)
    monkeypatch.setattr(bank_cpu, "_array", forbidden)
    experts_a4.pack_bank_a4(codes, scales, *fields, indices, 64, 64)
    assert all(torch.equal(value, old) for value, old in zip(fields, expected, strict=True))


def test_subclass_keeps_existing_protocol_instead_of_calling_the_plain_producer(monkeypatch):
    class Subclass(torch.Tensor):
        pass

    codes, scales, fields, indices, expected, _, _ = inputs(64, 64)
    codes = codes.as_subclass(Subclass)

    def forbidden():
        raise AssertionError("subclass must not compile or call the plain producer")

    monkeypatch.setattr(bank_cpu, "_ext", forbidden)
    experts_a4.pack_bank_a4(codes, scales, *fields, indices, 64, 64)
    assert all(torch.equal(value, old) for value, old in zip(fields, expected, strict=True))
