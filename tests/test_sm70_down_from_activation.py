"""CPU transport/metadata controls; real native data checks live in CUDA tests."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import gguf_ops


def frame(module):
    x = torch.zeros((1, 2560), dtype=torch.float16)
    picks = torch.arange(11, dtype=torch.int32)[None]
    picks[:, -1] = 512
    packed = SimpleNamespace(data=torch.zeros((1, 18), dtype=torch.uint8), ggml_type=42, column_offset=160)
    args = dict(first=256, shared=torch.zeros((1, 2560), dtype=torch.float32),
                local=torch.empty((1, 11), dtype=torch.int64), activation=torch.zeros((11, 160), dtype=torch.float16),
                out=torch.empty((1, 11, 2560), dtype=torch.float32), grouping=torch.empty(100, dtype=torch.int32))
    return gguf_ops._OwnedDownFromActivation(module, x.device), (x, picks, packed, packed, packed), args


def test_supplied_activation_delegates_once_without_routed_gu_and_versions():
    calls = []
    native = SimpleNamespace(experts_owned_from_activation=lambda *args: calls.append(args))
    prepared, inputs, outputs = frame(native)
    versions = {name: value._version for name, value in outputs.items() if isinstance(value, torch.Tensor)}
    assert prepared.run(*inputs, **outputs) is outputs["out"]
    assert len(calls) == 1
    actual = calls[0]
    assert actual[11:14] == (160, 256, 512)
    assert actual[14] is outputs["shared"] and actual[16] is outputs["activation"]
    for name in ("local", "grouping", "out"):
        assert outputs[name]._version == versions[name] + 1
    assert outputs["activation"]._version == versions["activation"]
    assert outputs["shared"]._version == versions["shared"]


def test_authentic_native_failure_preserved_without_metadata_or_retry():
    primary = RuntimeError("native dispatch failed")
    calls = []

    def fail(*args):
        calls.append(args)
        raise primary

    prepared, inputs, outputs = frame(SimpleNamespace(experts_owned_from_activation=fail))
    before = outputs["out"]._version
    with pytest.raises(RuntimeError) as caught:
        prepared.run(*inputs, **outputs)
    assert caught.value is primary and len(calls) == 1
    assert outputs["out"]._version == before


@pytest.mark.parametrize("first", [True, -1, 128, 512])
def test_first_boundary_refused_before_native(first):
    calls = []
    prepared, inputs, outputs = frame(SimpleNamespace(experts_owned_from_activation=lambda *a: calls.append(a)))
    outputs["first"] = first
    with pytest.raises(ValueError):
        prepared.run(*inputs, **outputs)
    assert calls == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prepare_rejects_unindexed_or_noncuda_without_extension(monkeypatch, device):
    calls = []
    monkeypatch.setattr(gguf_ops, "extension", lambda: calls.append("extension"))
    with pytest.raises(ValueError):
        gguf_ops.prepare_owned_from_activation(device)
    assert calls == []
