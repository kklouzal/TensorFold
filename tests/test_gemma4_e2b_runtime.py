"""Qualified descriptors fail before checkpoint allocation; no CUDA is required."""
from types import SimpleNamespace
import pytest

from tensorfold.families.gemma4.cuda.e2b_runtime import (
    require_runtime, TORCH_COMMIT, TRANSFORMERS_VERSION, TRITON_VERSION,
)


def runtime(monkeypatch, tmp_path):
    state = tmp_path / "externally-verified-foundation.json"
    state.write_text("{}")  # This boundary never substitutes for launcher seal verification.
    monkeypatch.setenv("TF_FOUNDATION_STATE", str(state))
    monkeypatch.setenv("TF_FOUNDATION_RUNTIME_PHASE", "cuda")
    return SimpleNamespace(version=SimpleNamespace(git_version=TORCH_COMMIT, cuda="13.4"),
                           cuda=SimpleNamespace(is_available=lambda: True,
                                                get_device_capability=lambda _: (12, 1)),
                           backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(
                               allow_bf16_reduced_precision_reduction=False,
                               allow_bf16_reduced_precision_reduction_split_k=False))))


def test_qualified_descriptor_is_admitted(monkeypatch, tmp_path):
    require_runtime(runtime(monkeypatch, tmp_path), TRANSFORMERS_VERSION, TRITON_VERSION)


@pytest.mark.parametrize("part", ["phase", "state", "torch", "toolkit", "device", "cuda", "controls", "hf", "triton"])
def test_unqualified_runtime_has_actionable_error(monkeypatch, tmp_path, part):
    torch = runtime(monkeypatch, tmp_path)
    hf, triton = TRANSFORMERS_VERSION, TRITON_VERSION
    if part == "phase":
        monkeypatch.setenv("TF_FOUNDATION_RUNTIME_PHASE", "cpu")
    elif part == "state":
        monkeypatch.delenv("TF_FOUNDATION_STATE")
    elif part == "torch":
        torch.version.git_version = "unqualified"
    elif part == "toolkit":
        torch.version.cuda = "13.3"
    elif part == "device":
        torch.cuda.get_device_capability = lambda _: (8, 9)
    elif part == "cuda":
        torch.cuda.is_available = lambda: False
    elif part == "controls":
        del torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k
    elif part == "hf":
        hf = "5.20.0"
    else:
        triton = "other"
    with pytest.raises(RuntimeError, match="cuda_entrypoint"):
        require_runtime(torch, hf, triton)
