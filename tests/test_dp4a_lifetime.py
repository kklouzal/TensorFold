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
        gate=NS(data=torch.zeros((2, 3), dtype=torch.uint8), ggml_type=18, column_offset=0, n=40960, k=2560),
        up=NS(data=torch.zeros((2, 3), dtype=torch.uint8), ggml_type=22, column_offset=0, n=40960, k=2560),
        down=NS(data=torch.zeros((2, 3), dtype=torch.uint8), ggml_type=20, column_offset=16, n=1310720, k=80),
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


def test_graph_math_policy_change_refuses_replay(monkeypatch):
    owner,_log,_stream,_backend=setup(monkeypatch)
    previous=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=not previous
        with pytest.raises(RuntimeError,match="math policy"):
            owner.run()
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=previous
    owner.close()


def test_retained_packed_metadata_change_refuses_reuse(monkeypatch):
    owner,_log,_stream,_backend=setup(monkeypatch)
    owner.bank.down.k=79
    with pytest.raises(RuntimeError,match="layout"):
        owner.run()
    owner.close()


def fake_graph(monkeypatch, log):
    graph = NS(capture_begin=lambda **_: log.append("begin"),
               capture_end=lambda: log.append("end"), replay=lambda: log.append("replay"),
               reset=lambda: log.append("reset"))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    return graph


def fail(error):
    def operation(*_args, **_kwargs):
        raise error
    return operation


@pytest.mark.parametrize("operation", ["capture_begin", "capture_end"])
def test_prepare_native_failure_latches_both_owners(monkeypatch, operation):
    owner, log, _stream, _backend = setup(monkeypatch)
    graph = fake_graph(monkeypatch, log)
    error = OSError(operation)
    setattr(graph, operation, fail(error))
    with pytest.raises(OSError) as caught:
        owner.capture()
    assert caught.value is error
    borrower = owner.graph_owner
    assert borrower.state == "failed" and borrower.owner is owner and owner.failed is error
    with pytest.raises(RuntimeError, match="not prepared"):
        borrower.replay()
    with pytest.raises(RuntimeError, match="uncertain"):
        owner.run()
    with pytest.raises(RuntimeError, match="new borrower"):
        borrower.prepare()
    borrower.close()
    owner.close()


@pytest.mark.parametrize("operation", ["fence", "reset"])
def test_graph_failed_retirement_retains_and_poisons(monkeypatch, operation):
    owner, log, stream, _backend = setup(monkeypatch)
    graph = fake_graph(monkeypatch, log)
    borrower = owner.capture()
    error = OSError(operation)
    def retire_fault():
        assert borrower.state == "retiring"
        with pytest.raises(RuntimeError, match="uncertain"):
            owner.run()
        with pytest.raises(RuntimeError, match="not prepared"):
            borrower.replay()
        raise error

    if operation == "fence":
        stream.synchronize = retire_fault
    else:
        graph.reset = retire_fault
    with pytest.raises(OSError) as caught:
        borrower.close()
    assert caught.value is error and borrower.owner is owner and borrower.graph is graph
    assert owner.graph_owner is borrower and borrower.state == "failed" and not borrower.closed
    with pytest.raises(RuntimeError, match="not prepared"):
        borrower.replay()
    with pytest.raises(RuntimeError, match="uncertain"):
        owner.run()
    stream.synchronize = lambda: log.append("recovered-fence")
    graph.reset = lambda: log.append("recovered-reset")
    borrower.close()
    owner.close()
    assert borrower.closed and owner.closed


@pytest.mark.parametrize("operation", ["close", "reset_after_fence"])
def test_owner_failed_fence_blocks_reuse_and_retains(monkeypatch, operation):
    owner, log, stream, _backend = setup(monkeypatch)
    original = owner.bindings
    error = OSError(operation)
    stream.synchronize = fail(error)
    with pytest.raises(OSError) as caught:
        getattr(owner, operation)()
    assert caught.value is error and owner.bindings is original and not owner.closed
    with pytest.raises(RuntimeError, match="uncertain"):
        owner.run()
    with pytest.raises(RuntimeError, match="uncertain"):
        owner.capture()
    stream.synchronize = lambda: log.append("recovered-fence")
    owner.close()
    assert owner.closed


def test_foreign_capture_refused_before_borrower_publication(monkeypatch):
    owner, log, _stream, _backend = setup(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="borrower"):
        owner.capture()
    assert owner.graph_owner is None and log == []
    with pytest.raises(RuntimeError, match="explicit graph"):
        owner.run()
    with pytest.raises(RuntimeError, match="capture"):
        owner.close()


def test_only_owned_active_capture_accepts_run(monkeypatch):
    owner, log, _stream, _backend = setup(monkeypatch)
    graph = fake_graph(monkeypatch, log)
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    graph.capture_begin = lambda **_: capturing.__setitem__(0, True)
    graph.capture_end = lambda: capturing.__setitem__(0, False)
    borrower = owner.capture()
    assert borrower.state == "prepared" and owner.failed is None
    capturing[0] = True
    with pytest.raises(RuntimeError, match="foreign"):
        borrower.replay()
    with pytest.raises(RuntimeError, match="explicit graph"):
        owner.run()
    with pytest.raises(RuntimeError, match="borrower"):
        owner.capture()
    with pytest.raises(RuntimeError, match="capture"):
        borrower.close()
    assert borrower.state == "prepared"
    capturing[0] = False
    borrower.replay()
    borrower.close()
    owner.close()


def test_replay_failure_poison_blocks_owner_run(monkeypatch):
    owner, log, _stream, _backend = setup(monkeypatch)
    graph = fake_graph(monkeypatch, log)
    borrower = owner.capture()
    error = OSError("replay")
    graph.replay = fail(error)
    with pytest.raises(OSError) as caught:
        borrower.replay()
    assert caught.value is error and borrower.failed is error and owner.failed is error
    with pytest.raises(RuntimeError, match="uncertain"):
        owner.run()
    borrower.close()
    owner.close()
