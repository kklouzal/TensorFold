"""Actual pinned Torch flag controls; no GPU allocation or kernel qualification."""
import itertools
import json

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import media_numerics  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.media_numerics import MediaNumericScope  # noqa: E402


def _actual():
    return (torch._C._get_cublas_allow_fp16_accumulation(),
            *torch._C._get_cublas_allow_fp16_reduced_precision_reduction())


def _unrelated():
    return (torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
            torch._C._get_cublas_allow_tf32(), torch._C._get_cublaslt_prefer_grouped_gemm())


@pytest.fixture(autouse=True)
def restore_actual_torch_flags():
    initial = _actual()
    setters = (torch._C._set_cublas_allow_fp16_accumulation,
               torch._C._set_cublas_allow_fp16_reduced_precision_reduction)
    try:
        yield setters
    finally:
        # Direct saved setters restore even when this test injected a failure
        # into the public Torch wrapper's underlying native setter entry.
        setters[1](*initial[1:])
        setters[0](initial[0])
        assert _actual() == initial


def _configure(values):
    torch._C._set_cublas_allow_fp16_accumulation(values[0])
    torch._C._set_cublas_allow_fp16_reduced_precision_reduction(*values[1:])


@pytest.mark.parametrize("accumulation", (False, True))
def test_native_library_refuses_reduced_precision_without_split_k(accumulation):
    _configure((accumulation, False, False))
    with pytest.raises(RuntimeError, match="allow_splitk=False"):
        torch._C._set_cublas_allow_fp16_reduced_precision_reduction(True, False)
    assert _actual() == (accumulation, False, False)


@pytest.mark.parametrize("initial", [values for values in itertools.product((False, True), repeat=3)
                                        if not values[1] or values[2]])
def test_actual_native_flags_are_strict_split_k_independent_and_exactly_restored(initial, monkeypatch):
    _configure(initial)
    unrelated = _unrelated()
    initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("numeric scope allocated a tensor"))
    journal = []

    def register(owner):
        assert _actual() == initial
        assert owner.original is None and not owner.active
        journal.append(owner)

    owner = MediaNumericScope.create(register=register)
    assert journal == [owner]
    assert owner.original == initial
    assert _actual() == (False, False, initial[2])
    record = owner.record()
    assert record["torch_version"] == torch.__version__
    assert type(record["torch_version"]) is str
    assert record["arithmetic"]["linear_fp16_accumulation"] is False
    assert record["arithmetic"]["linear_fp16_reduced_precision"] is False
    assert record["arithmetic"]["linear_fp16_split_k"] is initial[2]
    assert json.loads(json.dumps(record))["scope"] == "one-model-process-global-drained-lifecycle"
    from tensorfold.engine.model_identity import canonical_runtime_json
    canonical_runtime_json(record)
    assert _unrelated() == unrelated and torch.cuda.is_initialized() == initialized
    retired = []

    def retire():
        assert _actual() == (False, False, initial[2])
        assert owner.active and not owner.retired and not owner.closed
        retired.append(True)

    owner.close(retire=retire)
    assert retired == [True] and owner.closed and owner.retired and not owner.active
    assert _actual() == initial and _unrelated() == unrelated
    owner.close(retire=lambda: pytest.fail("closed scope repeated its retirement"))
    with pytest.raises(ValueError, match="not active"):
        owner.record()


def test_failed_fence_preserves_active_policy_and_journal_until_successful_retry():
    _configure((True, True, True))
    journal = []
    owner = MediaNumericScope.create(register=journal.append)
    primary = RuntimeError("GPU consumers did not retire")

    def failed_retire():
        raise primary

    with pytest.raises(RuntimeError) as caught:
        owner.close(retire=failed_retire)
    assert caught.value is primary
    assert journal == [owner] and owner.active and not owner.closed and not owner.retired
    assert _actual() == (False, False, True)
    assert owner.verify()["linear_fp16_split_k"] is True
    owner.close(retire=lambda: None)
    assert _actual() == (True, True, True)


@pytest.mark.parametrize("setter_name,after", [
    ("_set_cublas_allow_fp16_accumulation", False),
    ("_set_cublas_allow_fp16_accumulation", True),
    ("_set_cublas_allow_fp16_reduced_precision_reduction", False),
    ("_set_cublas_allow_fp16_reduced_precision_reduction", True),
])
def test_partial_native_setter_failure_restores_verified_changes_and_preserves_primary(setter_name, after,
                                                                                     monkeypatch):
    initial = (True, True, True)
    _configure(initial)
    unrelated = _unrelated()
    original_setter = getattr(torch._C, setter_name)
    primary, calls, journal = RuntimeError("native setter interrupted"), [], []

    def fail_once(*values):
        calls.append(values)
        if len(calls) == 1:
            if after:
                original_setter(*values)
            raise primary
        return original_setter(*values)

    monkeypatch.setattr(torch._C, setter_name, fail_once)
    with pytest.raises(RuntimeError) as caught:
        MediaNumericScope.create(register=journal.append)
    assert caught.value is primary and len(journal) == 1
    assert journal[0].closed and journal[0].retired and not journal[0].active
    assert _actual() == initial and _unrelated() == unrelated
    assert len(calls) == (2 if after else 1)


def test_constructor_restoration_failure_retains_actual_owner_and_primary_for_retry(monkeypatch):
    _configure((True, True, True))
    original_setter = torch._C._set_cublas_allow_fp16_accumulation
    primary, cleanup = RuntimeError("constructor setter"), RuntimeError("rollback setter")
    calls, journal = [], []

    def setter(*values):
        calls.append(values)
        if len(calls) == 1:
            original_setter(*values)
            raise primary
        raise cleanup

    monkeypatch.setattr(torch._C, "_set_cublas_allow_fp16_accumulation", setter)
    with pytest.raises(RuntimeError) as caught:
        MediaNumericScope.create(register=journal.append)
    assert caught.value is primary and caught.value.__cause__ is cleanup
    owner = journal[0]
    assert owner in primary._tensorfold_retained_owners
    assert owner.retired and owner.active and not owner.closed
    monkeypatch.setattr(torch._C, "_set_cublas_allow_fp16_accumulation", original_setter)
    owner.close(retire=lambda: pytest.fail("failed constructor had no model consumers"))
    assert owner.closed and _actual() == (True, True, True)


def test_failed_initial_adoption_restores_flags_before_propagating_primary(monkeypatch):
    initial = (True, True, True)
    _configure(initial)
    primary, journal = RuntimeError("initial policy verification"), []

    def fail_verification():
        assert _actual() == (False, False, True)
        raise primary

    monkeypatch.setattr(media_numerics, "vision_arithmetic", fail_verification)
    with pytest.raises(RuntimeError) as caught:
        MediaNumericScope.create(register=journal.append)
    assert caught.value is primary and journal[0].closed
    assert _actual() == initial


def test_post_fence_restore_failure_does_not_repeat_completed_retirement(monkeypatch):
    _configure((True, True, True))
    owner = MediaNumericScope.create(register=lambda owner: None)
    original_setter = torch._C._set_cublas_allow_fp16_reduced_precision_reduction
    primary, retired = RuntimeError("restoration refused"), []

    def failed_setter(*values):
        raise primary

    monkeypatch.setattr(torch._C, "_set_cublas_allow_fp16_reduced_precision_reduction", failed_setter)
    with pytest.raises(RuntimeError) as caught:
        owner.close(retire=lambda: retired.append(True))
    assert caught.value is primary and retired == [True]
    assert owner.retired and owner.active and not owner.closed
    with pytest.raises(ValueError, match="not active"):
        owner.verify()
    monkeypatch.setattr(torch._C, "_set_cublas_allow_fp16_reduced_precision_reduction", original_setter)
    owner.close(retire=lambda: pytest.fail("completed fence repeated"))
    assert owner.closed and _actual() == (True, True, True)


def test_failed_journal_does_not_change_any_flags(monkeypatch):
    initial, journal = _actual(), []
    primary = RuntimeError("journal publication")

    def forbidden_setter(*values):
        pytest.fail("unregistered scope changed flags")

    def register(owner):
        journal.append(owner)
        raise primary

    monkeypatch.setattr(torch._C, "_set_cublas_allow_fp16_accumulation", forbidden_setter)
    monkeypatch.setattr(torch._C, "_set_cublas_allow_fp16_reduced_precision_reduction", forbidden_setter)
    with pytest.raises(RuntimeError) as caught:
        MediaNumericScope.create(register=register)
    assert caught.value is primary and journal[0].closed
    assert _actual() == initial


@pytest.mark.parametrize("flag", ["accumulation", "reduction", "split_k"])
def test_adoption_rechecks_all_three_actual_flags(flag):
    _configure((True, True, True))
    owner = MediaNumericScope.create(register=lambda owner: None)
    if flag == "accumulation":
        torch._C._set_cublas_allow_fp16_accumulation(True)
    elif flag == "reduction":
        torch._C._set_cublas_allow_fp16_reduced_precision_reduction(True, True)
    else:
        torch._C._set_cublas_allow_fp16_reduced_precision_reduction(False, False)
    with pytest.raises(ValueError, match="FP32|changed"):
        owner.verify()
    if flag == "split_k":
        with pytest.raises(ValueError, match="outside"):
            owner.close(retire=lambda: None)
        assert owner.retired and not owner.closed
        torch._C._set_cublas_allow_fp16_reduced_precision_reduction(False, True)
    owner.close(retire=lambda: None)
    assert _actual() == (True, True, True)


def test_other_process_cannot_verify_or_restore_parent_arithmetic(monkeypatch):
    _configure((True, True, True))
    owner = MediaNumericScope.create(register=lambda owner: None)
    pid = owner.pid
    with monkeypatch.context() as isolated:
        isolated.setattr(media_numerics.os, "getpid", lambda: pid + 1)
        with pytest.raises(ValueError, match="another process"):
            owner.verify()
        with pytest.raises(ValueError, match="another process"):
            owner.close(retire=lambda: pytest.fail("child attempted parent fence"))
    assert _actual() == (False, False, True)
    owner.close(retire=lambda: None)


def test_restore_does_not_overwrite_an_unowned_accumulation_change():
    initial = (False, False, True)
    _configure(initial)
    owner = MediaNumericScope.create(register=lambda owner: None)
    torch._C._set_cublas_allow_fp16_accumulation(True)
    with pytest.raises(ValueError, match="outside"):
        owner.close(retire=lambda: None)
    assert _actual() == (True, False, True) and owner.retired and not owner.closed
    torch._C._set_cublas_allow_fp16_accumulation(False)
    owner.close(retire=lambda: pytest.fail("retirement repeated after detected unowned change"))
    assert owner.closed and _actual() == initial


def test_owner_journal_and_retirement_callback_are_explicit():
    with pytest.raises(ValueError, match="owner journal"):
        MediaNumericScope.create(register=None)
    owner = MediaNumericScope.create(register=lambda owner: None)
    with pytest.raises(ValueError, match="fence/drain"):
        owner.close(retire=None)
    assert owner.active and not owner.retired
    owner.close(retire=lambda: None)
