"""Fixed binding/status/graph outcome controls without author GPU execution."""

from types import SimpleNamespace as NS
import pytest
import torch
from tensorfold.cuda.sm70 import dp4a_experts as module


def setup(monkeypatch, role="target"):
    log = []
    stream = NS(cuda_stream=17, synchronize=lambda: log.append("fence"))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _self: True))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    shared = [
        NS(weight=torch.ones((2, 2), dtype=torch.float16), n=2, k=2, implementation="reference") for _ in range(3)
    ]
    bank = NS(
        count=513,
        width=80,
        dims=2560,
        limit=0.0,
        swiglu=True,
        shared=shared,
        gate=NS(data=torch.zeros((2, 3), dtype=torch.uint8), ggml_type=18, column_offset=0),
        up=NS(data=torch.zeros((2, 3), dtype=torch.uint8), ggml_type=22, column_offset=0),
        down=NS(data=torch.zeros((2, 3), dtype=torch.uint8), ggml_type=20, column_offset=16),
    )
    x = torch.zeros((1, 2560), dtype=torch.float16)
    picks = torch.zeros((1, 2), dtype=torch.int32)
    out = torch.zeros((1, 2, 2560))
    bank.forward_exact = lambda *_args: log.append("exact") or out
    bank._write_shared = lambda *_args: log.append("shared") or out
    backend = NS(
        quantize=lambda *_: log.append("quantize"),
        project=lambda *_: log.append("project"),
        activate=lambda *_: log.append("activate"),
        mask=lambda *_: log.append("mask"),
    )
    monkeypatch.setattr(module, "extension", lambda: backend)
    return module.Prototype(bank, x, picks, out, role=role), log, stream, backend


def test_hot_run_keeps_device_status_latched_without_readback(monkeypatch):
    owner, log, _stream, _backend = setup(monkeypatch)
    owner.status.fill_(8)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_a, **_k: pytest.fail("hotCPUreadback"))
    monkeypatch.setattr(torch.Tensor, "item", lambda *_a, **_k: pytest.fail("hotitem"))
    assert owner.run() is owner.out
    assert log == ["quantize", "project", "project", "activate", "quantize", "project", "shared", "mask"]
    assert owner.status.tolist() == [8]


def test_exact_mtp_fallback_never_quantizes(monkeypatch):
    owner, log, _stream, _backend = setup(monkeypatch, role="mtp")
    assert owner.exact_only and owner.status is None and owner.run() is owner.out
    assert log == ["exact"]


def test_foreign_stream_and_rebound_storage_rejected(monkeypatch):
    owner, log, stream, _backend = setup(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: NS(cuda_stream=18))
    with pytest.raises(RuntimeError, match="serialized"):
        owner.run()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    owner.qx = owner.qx.clone()
    with pytest.raises(RuntimeError, match="bindings"):
        owner.run()
    assert log == []


def test_graph_borrower_retains_original_owners_until_positive_close(monkeypatch):
    owner, log, _stream, _backend = setup(monkeypatch)
    graph = NS(
        capture_begin=lambda **_: log.append("begin"),
        capture_end=lambda: log.append("end"),
        replay=lambda: log.append("replay"),
        reset=lambda: log.append("reset"),
    )
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    captured = owner.capture()
    with pytest.raises(RuntimeError, match="borrower"):
        owner.close()
    captured.replay()
    captured.close()
    owner.close()
    assert owner.closed and log[-3:] == ["fence", "reset", "fence"]


def test_capture_body_and_end_failures_preserve_primary(monkeypatch):
    owner, _log, _stream, backend = setup(monkeypatch)
    primary = ValueError("body")
    secondary = OSError("captureend")
    calls = []

    def project(*_args):
        calls.append(1)
        if len(calls) > 3:
            raise primary

    backend.project = project

    def end():
        raise secondary

    graph = NS(capture_begin=lambda **_: None, capture_end=end, reset=lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    with pytest.raises(ValueError) as caught:
        owner.capture()
    assert caught.value is primary and owner.graph_owner is not None
    cause = caught.value.__cause__
    assert cause is secondary or isinstance(cause, BaseExceptionGroup) and secondary in cause.exceptions
    owner.graph_owner.close()
    owner.close()
