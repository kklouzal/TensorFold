"""Optional MTP producer failure retirement on labeled default-stream seams; no native MLX execution."""

import builtins
from pathlib import Path
import sys
from types import ModuleType
import weakref

import pytest

from tensorfold.families import qwen3_5, qwen3_5_moe
from tests.mlx_host_protocol_fakes import cli_protocol


class Model:
    def close(self):
        pytest.fail("MLX models have no declared generic close contract")


def failed_load(monkeypatch, stage, *, fail_gpu=False, fail_cpu=False, fail_lookup=False):
    primary = RuntimeError("optional MTP " + stage + " failed")
    gpu, cpu = OSError("GPU fence failed"), LookupError("CPU fence failed")
    model = Model()
    reference = weakref.ref(model)
    monkeypatch.setattr(qwen3_5, "load_lane_model", lambda path, **options: (model, "tokenizer"))
    package = ModuleType("tensorfold.families.qwen3_5_moe.family")

    class Family:
        def __init__(self, *args, **options):
            if stage == "factory":
                raise primary

        def probe(self):
            if stage == "probe":
                raise primary

    package.Qwen36Family = Family
    monkeypatch.setitem(sys.modules, package.__name__, package)

    def wrap(owner, **options):
        assert owner is model
        if stage == "wrapper":
            raise primary
        result = options["make"]()
        result.probe()
        return result

    def select(path):
        if stage == "selection":
            raise primary
        return Path("head.safetensors")

    monkeypatch.setattr(qwen3_5, "lane_family", wrap)
    monkeypatch.setattr(qwen3_5_moe, "mtp_file", select)
    original_import = builtins.__import__

    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        if stage == "import" and name == "family" and level == 1:
            raise primary
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", importing)
    attempts = []
    with cli_protocol(monkeypatch) as core:
        def stream(device):
            label = "gpu" if device is core.gpu else "cpu"
            attempts.append(("lookup", label))
            if fail_lookup and label == "gpu":
                raise gpu
            return label

        def fence(label):
            attempts.append(("fence", label))
            if label == "gpu" and fail_gpu:
                raise gpu
            if label == "cpu" and fail_cpu:
                raise cpu

        core.default_stream, core.synchronize = stream, fence
        with pytest.raises(RuntimeError) as caught:
            qwen3_5_moe.load(Path("checkpoint"), mtp_drafts=4)
    return caught.value, primary, gpu, cpu, attempts, reference


@pytest.mark.parametrize("stage", ["import", "selection", "wrapper", "factory", "probe"])
def test_every_new_positive_mtp_failure_attempts_both_default_fences(monkeypatch, stage):
    actual, primary, _, _, attempts, _ = failed_load(monkeypatch, stage)
    assert actual is primary and actual.__cause__ is None
    assert attempts == [("lookup", "gpu"), ("fence", "gpu"), ("lookup", "cpu"), ("fence", "cpu")]


@pytest.mark.parametrize("fail_gpu,fail_cpu", [(True, False), (False, True), (True, True)])
def test_fence_failure_preserves_primary_and_every_secondary(monkeypatch, fail_gpu, fail_cpu):
    actual, primary, gpu, cpu, attempts, reference = failed_load(
        monkeypatch, "probe", fail_gpu=fail_gpu, fail_cpu=fail_cpu)
    assert actual is primary
    causes = actual.__cause__.exceptions if isinstance(actual.__cause__, BaseExceptionGroup) else (actual.__cause__,)
    assert causes == tuple(error for enabled, error in ((fail_gpu, gpu), (fail_cpu, cpu)) if enabled)
    assert attempts[-2:] == [("lookup", "cpu"), ("fence", "cpu")]
    # The producer traceback retains the acquired owner if either fence failed.
    frames, trace = [], actual.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code is qwen3_5_moe.load.__code__:
            frames.append(trace.tb_frame)
        trace = trace.tb_next
    assert frames and all(frame.f_locals["model"] is reference() for frame in frames)


def test_gpu_stream_lookup_failure_still_attempts_cpu_completion(monkeypatch):
    actual, primary, gpu, _, attempts, _ = failed_load(monkeypatch, "selection", fail_lookup=True)
    assert actual is primary and actual.__cause__ is gpu
    assert attempts == [("lookup", "gpu"), ("lookup", "cpu"), ("fence", "cpu")]


def test_success_does_not_add_failure_fences(monkeypatch):
    package = ModuleType("tensorfold.families.qwen3_5_moe.family")
    package.Qwen36Family = type("Family", (), {})
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setattr(qwen3_5, "load_lane_model", lambda *args, **options: (Model(), "tokenizer"))
    monkeypatch.setattr(qwen3_5, "lane_family", lambda *args, **options: "wrapped")
    with cli_protocol(monkeypatch) as core:
        core.synchronize = lambda *args: pytest.fail("successful load acquired new failure fence work")
        assert qwen3_5_moe.load(Path("checkpoint"), mtp_drafts=4, mtp_head="bound.safetensors") == ("wrapped", "tokenizer")


@pytest.mark.parametrize("cap,drafter", [(None, ""), (0, ""), (None, "selected-dflash"), (4, "selected-dflash")])
def test_baseline_failure_routes_do_not_acquire_new_mtp_fences(monkeypatch, cap, drafter):
    primary = RuntimeError("baseline factory failed")
    monkeypatch.setattr(qwen3_5, "load_lane_model", lambda *args, **options: (Model(), "tokenizer"))

    def wrapper(*args, **options):
        raise primary

    monkeypatch.setattr(qwen3_5, "lane_family", wrapper)
    with cli_protocol(monkeypatch) as core:
        core.synchronize = lambda *args: pytest.fail("baseline route acquired optional-MTP fences")
        with pytest.raises(RuntimeError) as caught:
            qwen3_5_moe.load(Path("checkpoint"), mtp_drafts=cap, drafter=drafter)
    assert caught.value is primary and primary.__cause__ is None
