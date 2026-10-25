"""Vision unit limits and partial ownership use actual library-independent boundaries."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.mmproj_vision import SM70VisionEncoder, _grids, vision_arithmetic  # noqa: E402


def test_complete_attention_units_are_bounded_without_spatial_splitting():
    layout = SimpleNamespace(merge=2)
    grids, count = _grids(torch.tensor([[1, 4, 6], [3, 2, 2]]), layout, 36)
    assert grids == ((1, 4, 6), (3, 2, 2)) and count == 36
    with pytest.raises(ValueError, match="complete vision"):
        _grids(torch.tensor([[1, 4, 6], [3, 2, 2]]), layout, 35)
    for value in (torch.tensor([[1., 4., 6.]]), torch.tensor([[1, 3, 6]]),
                  torch.tensor([[0, 4, 6]]), torch.empty(0, 3, dtype=torch.int64)):
        with pytest.raises(ValueError):
            _grids(value, layout, 36)
    with pytest.raises(ValueError):
        _grids(torch.tensor([[1, 2, 2]]), layout, 2**31)


@pytest.fixture
def strict_vision_gemm():
    matmul = torch.backends.cuda.matmul
    accumulation = matmul.allow_fp16_accumulation
    reduction = (matmul.allow_fp16_reduced_precision_reduction,
                 matmul.allow_fp16_reduced_precision_reduction_split_k)
    try:
        matmul.allow_fp16_accumulation = False
        matmul.allow_fp16_reduced_precision_reduction = (False, reduction[1])
        yield matmul
    finally:
        matmul.allow_fp16_accumulation = accumulation
        matmul.allow_fp16_reduced_precision_reduction = reduction


@pytest.mark.parametrize("flag", ("allow_fp16_accumulation", "allow_fp16_reduced_precision_reduction"))
def test_strict_vision_arithmetic_refuses_unadmitted_reduced_precision_without_changing_flags(strict_vision_gemm, flag):
    assert not vision_arithmetic()["linear_fp16_accumulation"]
    setattr(strict_vision_gemm, flag, True)
    with pytest.raises(ValueError, match="FP32 GEMM"):
        vision_arithmetic()
    assert getattr(torch.backends.cuda.matmul, flag) is True


def test_constructor_journals_partial_owner_before_any_import_model_or_device_use():
    seen = []
    primary = MemoryError("owner journal publication failed")

    def register(owner):
        seen.append(owner)
        raise primary

    with pytest.raises(MemoryError) as caught:
        SM70VisionEncoder.create(None, None, staging_bytes=256, device="cpu", max_unit_patches=4,
                                  software_binding=None, register=register)
    assert caught.value is primary and len(seen) == 1
    assert seen[0].model is None and not seen[0].gpu_started and not seen[0].closed


def test_fp32_split_k_policy_remains_permitted_and_is_part_of_numeric_identity(strict_vision_gemm):
    strict_vision_gemm.allow_fp16_reduced_precision_reduction = (False, False)
    first = vision_arithmetic()
    strict_vision_gemm.allow_fp16_reduced_precision_reduction = (False, True)
    second = vision_arithmetic()
    assert first != second
    assert first["linear_fp16_split_k"] is False and second["linear_fp16_split_k"] is True


def test_failed_input_retirement_and_close_retain_owners(monkeypatch):
    primary = RuntimeError("event wait failed")

    class Done:
        def synchronize(self):
            raise primary

    owner = SM70VisionEncoder.__new__(SM70VisionEncoder)
    payload = [object()]
    owner.inputs, owner.done, owner.failed = payload, Done(), False
    with pytest.raises(RuntimeError) as caught:
        owner._retire_inputs()
    assert caught.value is primary and owner.failed and owner.inputs is payload and owner.done is not None
    model = object()
    owner.closed, owner.gpu_started, owner.device, owner.model = False, True, "cuda:0", model
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: (_ for _ in ()).throw(primary))
    with pytest.raises(RuntimeError) as caught:
        owner.close()
    assert caught.value is primary and not owner.closed and owner.model is model and owner.inputs is payload
