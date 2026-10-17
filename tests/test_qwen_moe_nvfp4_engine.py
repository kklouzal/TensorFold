"""NVFP4 MoE format dispatch and expert-owner retirement without a GPU."""
import json
import threading
from types import SimpleNamespace as NS

import pytest

from tensorfold.families import qwen3_5_moe as family
from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
from tensorfold.families.qwen3_5_moe.cuda.nvfp4_engine import QwenNvfp4MoeEngine


@pytest.mark.parametrize("bits", [False, True, 0.0, 4.0, "4", None, -1, 8])
def test_invalid_draft_precision_refused_before_startup(monkeypatch, bits):
    monkeypatch.setattr(QwenNvfp4MoeEngine, "_start", lambda *a, **kw: pytest.fail("invalid precision reached startup"))
    with pytest.raises(ValueError, match="draft precision"):
        QwenNvfp4MoeEngine("missing", drafter_bits=bits)


def _engine(monkeypatch, *, cache=None, scheduler=None):
    events = []

    def start(self, *args, **kwargs):
        self.scheduler, self.expert_cache = scheduler, cache
        self.concurrent = scheduler is not None
        self.torch = NS(cuda=NS(synchronize=lambda: events.append("fence")))

    monkeypatch.setattr(QwenNvfp4MoeEngine, "_start", start)
    return QwenNvfp4MoeEngine("target"), events


def test_accepted_parallel_calls_retire_before_expert_authority(monkeypatch):
    entered, release, closing = threading.Barrier(3), threading.Event(), threading.Event()
    cache_events = []
    scheduler = NS(close=closing.set)
    engine, events = _engine(monkeypatch, cache=NS(close=lambda: cache_events.append("unpin")), scheduler=scheduler)

    def generate(self, marker, *args, **kwargs):
        entered.wait(timeout=5)
        if not release.wait(5):
            raise AssertionError("test failed to release the accepted operation")
        return marker

    monkeypatch.setattr(Qwen27Engine, "generate", generate)
    results, errors = [], []

    def caller(marker):
        try:
            results.append(engine.generate(marker, 2, None, lambda _: None))
        except BaseException as error:  # noqa: BLE001 -- The parent assertion must observe every worker failure.
            errors.append(error)

    calls = [threading.Thread(target=caller, args=(i,)) for i in (1, 2)]
    for thread in calls:
        thread.start()
    entered.wait(timeout=5)
    closer = threading.Thread(target=engine.close)
    closer.start()
    try:
        assert closing.wait(5)
        assert not events and not cache_events
        with pytest.raises(RuntimeError, match="closing"):
            engine.generate(3, 2, None, lambda _: None)
    finally:
        release.set()
        for thread in [*calls, closer]:
            thread.join(timeout=5)
            assert not thread.is_alive()
    assert not errors and sorted(results) == [1, 2]
    assert events == ["fence"] and cache_events == ["unpin"]
    engine.close()
    assert events == ["fence"]


def test_failed_retirement_retains_ownership_and_can_be_retried(monkeypatch):
    failure = RuntimeError("unregister refused")
    attempts = []

    def retire():
        attempts.append(1)
        if len(attempts) == 1:
            raise failure

    cache = NS(close=retire)
    engine, events = _engine(monkeypatch, cache=cache)
    with pytest.raises(RuntimeError) as caught:
        engine.close()
    assert caught.value is failure and engine.expert_cache is cache and not engine._closed
    with pytest.raises(RuntimeError, match="closing"):
        engine.generate([], 2, None, lambda _: None)
    engine.close()
    assert engine._closed and len(events) == 2 and len(attempts) == 2


def test_reentrant_close_does_not_deadlock_or_release_active_owners(monkeypatch):
    engine, events = _engine(monkeypatch)
    monkeypatch.setattr(Qwen27Engine, "generate", lambda self, *a, **kw: self.close())
    with pytest.raises(RuntimeError, match="active call"):
        engine.generate([], 2, None, lambda _: None)
    assert not engine._closing and not engine._calls and not events
    engine.close()
    assert events == ["fence"]


def test_partial_startup_failure_drains_factory_published_authority(monkeypatch):
    failure = ValueError("later layer failed")
    events = []

    def start(self, *args, **kwargs):
        self.expert_cache = NS(close=lambda: events.append("unpin"))
        self.torch = NS(cuda=NS(synchronize=lambda: events.append("fence")))
        raise failure

    monkeypatch.setattr(QwenNvfp4MoeEngine, "_start", start)
    with pytest.raises(ValueError) as caught:
        QwenNvfp4MoeEngine("target")
    assert caught.value is failure and events == ["fence", "unpin"]


def _config(path, *, method="compressed-tensors", format="nvfp4-pack-quantized"):
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe", "quantization_config": {
        "quant_method": method, "format": format}}))
    return path


def test_ct_factory_binds_separate_draft_flags_and_model_overrides(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("tensorfold.families.qwen3_5_moe.cuda.nvfp4_engine.QwenNvfp4MoeEngine",
                        lambda *args, **kw: calls.append((args, kw)) or "nv4")
    target = _config(tmp_path)
    assert family.cuda_drafter(target) == family.DRAFTER
    assert family.cuda_engine(target, drafter="secondary", parallel=4, vram_experts="auto",
                              drafter_bits=4, draft_rows=8, eos_token_ids=[0, 3]) == "nv4"
    args, options = calls[0]
    assert args == (target, target.__class__("secondary"))
    assert options["streams"] == 4 and options["drafter_bits"] == 4 and options["max_rows"] == 8
    assert options["eos_token_ids"] == [0, 3] and options["allow_copy"] is True
    family.cuda_engine(target, no_drafts=True, drafter="secondary")
    assert calls[-1][0][1] is None and calls[-1][1]["allow_copy"] is False


@pytest.mark.parametrize("format", ["pack-quantized", "float-quantized", "unknown"])
def test_ram_cache_refuses_other_storage_before_payload_download(tmp_path, format):
    with pytest.raises(ValueError, match="original compressed-tensors NVFP4"):
        family.check_vram_experts(_config(tmp_path, format=format), "auto")
