"""CPU scale reduction preserves the validator's bytes, metadata and dispatch."""

import weakref

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.nvfp4.format import validate_native_a4_scales


def payload(dtype, layout):
    backing = (torch.arange(512, device="cpu") % 127).to(torch.uint8)
    if layout == "scalar":
        raw = backing[5]
    elif layout == "contiguous":
        raw = backing[:256].view(16, 16)
    elif layout == "transpose":
        raw = backing[:256].view(16, 16).t()
    elif layout == "offset":
        raw = backing[4:260].view(16, 16)
        backing[0] = 255  # bytes outside the selected view do not participate
    elif layout == "stride":
        raw = backing[3:259:3]
        backing[4] = 255
    else:
        raw = backing[7:8].view(1, 1).expand(8, 32)
    return raw.view(dtype)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("layout", ["scalar", "contiguous", "transpose", "offset", "stride", "broadcast"])
def test_cpu_reduction_preserves_logical_bytes_and_source_storage(dtype, layout):
    value = payload(dtype, layout)
    before = value.view(torch.uint8).clone()
    storage = value.untyped_storage()
    metadata = (value.shape, value.stride(), value.storage_offset(), value.data_ptr(), value._version,
                storage.data_ptr(), storage.nbytes(), storage.resizable())
    validate_native_a4_scales(value, name="owned CPU view")
    assert torch.equal(value.view(torch.uint8), before)
    assert (value.shape, value.stride(), value.storage_offset(), value.data_ptr(), value._version,
            storage.data_ptr(), storage.nbytes(), storage.resizable()) == metadata


def test_cpu_bridge_owns_private_scratch_and_retires_it_without_torch_max(monkeypatch):
    value = payload(torch.float8_e4m3fn, "offset")
    original_numpy = torch.Tensor.numpy
    seen = []

    def numpy(private, *args, **kwargs):
        assert private.dtype == torch.uint8 and private.device.type == "cpu"
        assert private.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
        seen.append(weakref.ref(private))
        return original_numpy(private, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "numpy", numpy)
    monkeypatch.setattr(torch.Tensor, "amax", lambda *args, **kwargs: pytest.fail("CPU Torch max was not removed"))
    validate_native_a4_scales(value, name="private bridge")
    assert len(seen) == 1 and seen[0]() is None
    assert value.untyped_storage().resizable()


def test_cpu_source_can_still_grow_after_validation():
    value = torch.full((256,), 0x7e, dtype=torch.uint8, device="cpu")
    validate_native_a4_scales(value, name="resizable source")
    value.resize_(257)
    assert value.numel() == 257 and value.untyped_storage().resizable()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("byte", [0x7e, 0x7f])
@pytest.mark.parametrize("broadcast", [False, True])
def test_torch_ranks_above_numpy_limit_keep_the_original_contract(dtype, byte, broadcast):
    value = torch.tensor([0, byte], dtype=torch.uint8, device="cpu").view((1,) * 65 + (2,))
    if broadcast:
        value = value.expand((2,) + (1,) * 64 + (2,))
    value = value.view(dtype)
    # Exercise the exact original reduction at the changed interop boundary.
    assert int(value.view(torch.uint8).amax()) == byte
    before = value._version, value.data_ptr(), value.stride(), value.untyped_storage().resizable()
    if byte <= 0x7e:
        validate_native_a4_scales(value, name="high-rank bytes")
    else:
        with pytest.raises(ValueError, match="finite unsigned E4M3"):
            validate_native_a4_scales(value, name="high-rank bytes")
    assert (value._version, value.data_ptr(), value.stride(), value.untyped_storage().resizable()) == before


def test_existing_nonresizable_borrowed_storage_stays_borrowed():
    value = torch.frombuffer(bytearray([0, 0x38, 0x7e]), dtype=torch.uint8)
    before = value.data_ptr(), value._version
    assert not value.untyped_storage().resizable()
    validate_native_a4_scales(value, name="borrowed bytes")
    assert not value.untyped_storage().resizable()
    assert (value.data_ptr(), value._version) == before


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("bad", [0x7f, 0x80, 0xff])
def test_rejected_cpu_payload_keeps_exact_error_and_source_metadata(dtype, bad):
    raw = torch.tensor([0, bad, 0x7e], dtype=torch.uint8, device="cpu")
    value = raw.view(dtype)
    before = value._version, value.data_ptr(), value.untyped_storage().resizable()
    with pytest.raises(ValueError) as caught:
        validate_native_a4_scales(value, name="bad payload")
    assert str(caught.value) == (
        "bad payload: native A4 block scales must be finite unsigned E4M3 bytes in 0x00..0x7e")
    assert (value._version, value.data_ptr(), value.untyped_storage().resizable()) == before
    assert raw.tolist() == [0, bad, 0x7e]


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("shape", [(0,), (0, 7), (5, 0)])
def test_empty_storage_does_not_enter_either_reduction(dtype, shape, monkeypatch):
    value = torch.empty(shape, dtype=dtype, device="cpu")
    monkeypatch.setattr(torch.Tensor, "clone", lambda *args, **kwargs: pytest.fail("empty clone"))
    monkeypatch.setattr(torch.Tensor, "amax", lambda *args, **kwargs: pytest.fail("empty max"))
    validate_native_a4_scales(value, name="empty")


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8, torch.int32, torch.bfloat16])
def test_dtype_error_is_unchanged_and_precedes_payload_access(dtype, monkeypatch):
    value = torch.empty((0,), dtype=dtype, device="cpu")
    monkeypatch.setattr(torch.Tensor, "numel", lambda *args: pytest.fail("bad dtype inspected its payload"))
    with pytest.raises(ValueError) as caught:
        validate_native_a4_scales(value, name="rejected")
    assert str(caught.value) == "rejected: native A4 block scales require U8 bytes or E4M3 checkpoint values"


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
def test_negative_bit_view_keeps_the_original_error_before_private_copy(dtype, monkeypatch):
    value = torch.ones((4,), dtype=torch.uint8, device="cpu").view(dtype)._neg_view()
    with pytest.raises(RuntimeError) as original:
        value.view(torch.uint8).amax()
    monkeypatch.setattr(torch.Tensor, "clone", lambda *args, **kwargs: pytest.fail("negative view cloned"))
    with pytest.raises(RuntimeError) as current:
        validate_native_a4_scales(value, name="negative view")
    assert str(current.value) == str(original.value)


def test_subclass_keeps_original_torch_dispatch(monkeypatch):
    class ScaleTensor(torch.Tensor):
        pass

    value = torch.tensor([0, 0x38, 0x7e], dtype=torch.uint8, device="cpu").as_subclass(ScaleTensor)
    monkeypatch.setattr(torch.Tensor, "numpy", lambda *args, **kwargs: pytest.fail("subclass NumPy bridge"))
    validate_native_a4_scales(value, name="subclass")


def test_private_bridge_failure_preserves_the_primary_and_source(monkeypatch):
    value = torch.tensor([0, 0x38, 0x7e], dtype=torch.uint8, device="cpu")
    before = value.clone()
    primary = MemoryError("private NumPy bridge failed")

    def failed(private, *args, **kwargs):
        assert private.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
        raise primary

    monkeypatch.setattr(torch.Tensor, "numpy", failed)
    with pytest.raises(MemoryError) as caught:
        validate_native_a4_scales(value, name="failed bridge")
    assert caught.value is primary and value.untyped_storage().resizable()
    assert torch.equal(value, before)


def test_bridge_failure_does_not_pack_or_publish_a_host_bank(monkeypatch):
    pytest.importorskip("triton")
    from nvfp4_a4_inputs import PREFIX, Context, swiglu
    from test_nvfp4_a4_host_control import Cache

    from tensorfold.cuda.nvfp4 import experts_a4
    from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4

    ctx, cache = Context(), Cache()
    primary = MemoryError("validator bridge failed before publication")

    def failed(*args, **kwargs):
        raise primary

    monkeypatch.setattr(torch.Tensor, "numpy", failed)
    monkeypatch.setattr(experts_a4, "pack_projection_a4", lambda *args, **kwargs:
                        pytest.fail("failed validator entered packing"))
    with pytest.raises(MemoryError) as caught:
        load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", cache=cache, swiglu=swiglu)
    assert caught.value is primary
    assert len(cache.allocations) == 1 and not cache._layers and not ctx.used
