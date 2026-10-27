"""Finite caller/argument/retirement controls; no model, GPU or replaced host gate."""
from pathlib import Path
import hashlib
import json
import os
import signal
import time
import weakref
from types import SimpleNamespace

import pytest

from tools import v100_backend_probe as probe
from tensorfold.server.request_limits import RequestLimit


def cli(tmp_path):
    files = {}
    for name in ("model", "mtp", "wheel", "source-hold"):
        path = tmp_path / name
        path.write_bytes(b"argument fixture only; never used as a model")
        files[name] = path
    return [item for name, path in files.items() for item in ("--" + name, str(path))] + [
        "--image-identity", "sha256:" + "a" * 64, "--out", str(tmp_path / "results")]


def test_default_arguments_preserve_native_profile_without_override_flags(tmp_path, monkeypatch):
    monkeypatch.delenv("TENSORFOLD_SM70_PROFILE", raising=False)
    args = probe.arguments(cli(tmp_path))
    assert args.tokens == 128 and args.prompt == probe.DEFAULT_PROMPT and not args.serial_reference
    assert args.repetitions == 1 and args.instrumented_departure_grace_seconds is None
    assert args.execution_profile_manifest is None and args.execution_manifest_identity is None
    assert args.devices == [str(index) for index in range(8)]
    assert args.requested_sm70_profile is None and "TENSORFOLD_SM70_PROFILE" not in os.environ
    assert not hasattr(args, "context") and not hasattr(args, "precision") and not hasattr(args, "position_profile_manifest")
    assert (probe.CONTEXT, probe.WORLD, probe.DEPTH, probe.CONFIDENCE, probe.INPUT_CREDIT) == (
        262144, 8, 2, .7, 16 << 20)


@pytest.mark.parametrize("extra", [
    ["--tokens", "0"], ["--tokens", "513"], ["--prompt", ""], ["--prompt", "\U0001f308" * 1025],
    ["--gpus", "0,1,2,3,4,5,6,6"], ["--gpus", "0,1,2,3,4,5,6"],
    ["--timeout-seconds", "nan"], ["--timeout-seconds", "inf"], ["--timeout-seconds", "0"],
    ["--master-port", "0"], ["--rank", "8"], ["--base-commit", "dirty"],
    ["--image-identity", "not-an-image"], ["--context", "4096"], ["--no-drafts"],
    ["--precision", "full"], ["--vram-experts", "1"], ["--position-profile-manifest", "override.json"],
    ["--repetitions", "0"], ["--repetitions", "17"], ["--repetitions", "1.5"],
    ["--instrumented-departure-grace-seconds", "nan"], ["--instrumented-departure-grace-seconds", "0"],
    ["--instrumented-departure-grace-seconds", "1801"],
])
def test_invalid_or_contract_changing_arguments_refuse_before_worker_admission(tmp_path, extra):
    with pytest.raises(SystemExit):
        probe.arguments([*cli(tmp_path), *extra])
    assert not (tmp_path / "results").exists()


def test_explicit_execution_manifest_uses_strict_bounded_reader_and_records_exact_bytes(tmp_path):
    from tensorfold.families.qwen4_exp.cuda.execution_identity import SCHEMA

    path = tmp_path / "manifest.json"
    raw = json.dumps({"schema": SCHEMA, "version": 1, "artifacts": {}}).encode() + b"\n"
    path.write_bytes(raw)
    args = probe.arguments([*cli(tmp_path), "--execution-profile-manifest", str(path)])
    assert args.execution_profile_manifest == path.resolve()
    assert args.execution_manifest_identity == {"path": str(path.resolve()), "bytes": len(raw),
                                                "sha256": hashlib.sha256(raw).hexdigest()}
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("raw", [b"", b" " * ((4 << 20) + 1),
                                     b'{"schema":"tensorfold.execution-profile-manifest","version":1,"version":1}',
                                     b'{"schema":"unknown","version":1}'])
def test_invalid_execution_manifest_refuses_before_workers_or_result_creation(tmp_path, raw):
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        probe.arguments([*cli(tmp_path), "--execution-profile-manifest", str(path)])
    assert not (tmp_path / "results").exists()


def test_execution_manifest_directory_refuses_before_workers(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        probe.arguments([*cli(tmp_path), "--execution-profile-manifest", str(tmp_path)])


def test_missing_input_refuses_before_result_directory_or_model_acquisition(tmp_path):
    args = cli(tmp_path)
    Path(args[1]).unlink()
    with pytest.raises(FileNotFoundError):
        probe.arguments(args)
    assert not (tmp_path / "results").exists()


def test_callback_bounds_validate_before_retaining_and_close_is_terminal():
    callback = probe.TokenCollector(3, 10)
    assert callback([]) is False
    assert callback.first_nonempty_monotonic_ns is None
    assert callback([0, 9]) is False
    first = callback.first_nonempty_monotonic_ns
    for malformed in ([1, 2], [True], [-1], [10], [1.0], (1,)):
        with pytest.raises(ValueError, match="finite output/token"):
            callback(malformed)
        assert callback.ids == [0, 9]
        assert callback.first_nonempty_monotonic_ns == first
    callback([1])
    assert callback.ids == [0, 9, 1]
    callback.close()
    callback.close()
    with pytest.raises(RuntimeError, match="after retirement"):
        callback([])
    assert callback.first_nonempty_monotonic_ns == first


def test_first_nonempty_callback_timestamp_is_once_and_invalid_chunks_do_not_start_it():
    callback = probe.TokenCollector(2, 10)
    with pytest.raises(ValueError):
        callback([10])
    assert callback.first_nonempty_monotonic_ns is None
    before = time.monotonic_ns()
    callback([])
    callback([1])
    first = callback.first_nonempty_monotonic_ns
    assert before <= first <= time.monotonic_ns()
    callback([])
    callback([2])
    assert callback.first_nonempty_monotonic_ns == first


class Generation:
    """Only the execution boundary is substituted; RequestLimit is the real owner."""

    def __init__(self, limit, *, failure=None, chunk=None):
        self.limit, self.failure, self.chunk, self.callback = limit, failure, chunk, None
        self.calls = []

    def generate(self, prompt, count, sampling, callback, *, draft, stop_eos):
        assert self.limit.used == 1
        assert prompt == [0] and count == 3 and stop_eos
        assert sampling.temperature == 0 and sampling.seed == 17
        self.callback = callback
        self.calls.append(draft)
        callback([1] if self.chunk is None else self.chunk)
        if self.failure is not None:
            raise self.failure
        return {"generated_tokens": len(callback.ids)}


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, models
    return Tokenizer(models.WordLevel({"prompt": 0, "answer": 1}, unk_token="prompt"))


@pytest.fixture
def nvtx_markers(monkeypatch):
    pytest.importorskip("torch")
    from torch.cuda import nvtx

    events = []
    monkeypatch.setattr(nvtx, "range_push", lambda label: events.append(("push", label)))
    monkeypatch.setattr(nvtx, "range_pop", lambda: events.append(("pop", None)))
    return SimpleNamespace(api=nvtx, events=events)


def test_real_logical_owner_covers_generation_and_decode_then_retires(tokenizer, nvtx_markers):
    limit = RequestLimit(1)
    engine = Generation(limit)
    result = probe.generate_reply(engine, tokenizer, [0], count=3, draft=True, limit=limit, result={})
    assert result["text"] == "answer" and result["token_ids"] == [1]
    assert result["label"] == "mtp-target" and result["draft"] is True
    assert 0 <= result["first_committed_token_elapsed_ns"] <= result["elapsed_ns"]
    assert engine.calls == [True] and limit.used == 0 and engine.callback.closed
    with pytest.raises(RuntimeError, match="after retirement"):
        engine.callback([1])
    reference = probe.generate_reply(engine, tokenizer, [0], count=3, draft=False, limit=limit, result={})
    assert reference["label"] == "serial-reference" and reference["draft"] is False
    assert engine.calls == [True, False] and limit.used == 0
    assert nvtx_markers.events == [("push", "tensorfold/mtp-target"), ("pop", None),
                                   ("push", "tensorfold/serial-reference"), ("pop", None)]


def test_generation_failure_preserved_with_retired_owner_and_callback(tokenizer, nvtx_markers):
    limit = RequestLimit(1)
    primary = ValueError("original generation failure")
    engine = Generation(limit, failure=primary)
    result = {}
    with pytest.raises(ValueError) as caught:
        probe.generate_reply(engine, tokenizer, [0], count=3, draft=True, limit=limit, result=result)
    assert caught.value is primary and limit.used == 0 and engine.callback.closed
    assert engine.callback.ids == result["token_ids"] == [1] and result["complete"] is False


def test_output_overrun_refuses_without_partial_append_and_retires(tokenizer, nvtx_markers):
    limit = RequestLimit(1)
    engine = Generation(limit, chunk=[1] * 4)
    with pytest.raises(ValueError, match="finite output/token"):
        probe.generate_reply(engine, tokenizer, [0], count=3, draft=True, limit=limit, result={})
    assert engine.callback.ids == [] and engine.callback.closed and limit.used == 0


def test_receipt_limit_refuses_retained_payload_and_retires(tokenizer, nvtx_markers):
    class OversizedReceipt(Generation):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            result["oversized"] = "x" * probe.MAX_RECEIPT_BYTES
            return result

    limit = RequestLimit(1)
    engine = OversizedReceipt(limit)
    with pytest.raises(ValueError, match="receipt bound"):
        probe.generate_reply(engine, tokenizer, [0], count=3, draft=True, limit=limit, result={})
    assert engine.callback.closed and limit.used == 0


def test_changed_template_refuses_before_encoding_or_generation():
    class WrongTemplate:
        def provenance(self):
            return {"sha256": "0" * 64}

        def render(self, *_args, **_kwargs):
            pytest.fail("changed template was rendered")

    args = SimpleNamespace(prompt=probe.DEFAULT_PROMPT, thinking=False)
    with pytest.raises(ValueError, match="unchanged pinned SC117"):
        probe.prepared_prompt(SimpleNamespace(chat_template=WrongTemplate()), None, args)


def test_aggregate_report_budget_includes_every_rank_and_supervisor(tmp_path):
    assert (probe.LEADER_RECEIPT_BYTES + 7 * probe.FOLLOWER_RECEIPT_BYTES
            + probe.SUPERVISOR_RECEIPT_BYTES) == probe.MAX_RECEIPT_BYTES
    path = tmp_path / "report.json"
    value = {"complete": False, "failure": "original failure", "raw_ids": [1, 2]}
    probe.write_report(path, value, allowance=512)
    import json
    assert json.loads(path.read_text()) == value
    with pytest.raises(ValueError, match="complete overflow evidence saved"):
        probe.write_report(path, value, allowance=1)
    assert json.loads(path.with_suffix(".overflow.json").read_text()) == value
    assert json.loads(path.read_text()) == value


def test_report_budget_counts_ascii_escaped_unicode_wire_bytes(tmp_path):
    path = tmp_path / "unicode.json"
    value = {"text": "\U0001f308" * 10}
    with pytest.raises(ValueError, match="report slice exceeded"):
        probe.write_report(path, value, allowance=100)
    assert not path.exists() and path.with_suffix(".overflow.json").exists()


def rendering_fault(monkeypatch, kind):
    diagnostic = MemoryError("diagnostic rendering failed")

    class Primary(ValueError):
        def __str__(self):
            raise diagnostic

    if kind == "message":
        return Primary(), diagnostic

    def failed_traceback():
        raise diagnostic

    monkeypatch.setattr(probe.traceback, "format_exc", failed_traceback)
    return ValueError("original operation failed"), diagnostic


def retained_failure(error, target):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is target:
            return True
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(getattr(current, "exceptions", ()))
        pending.extend((current.__cause__, current.__context__))
    return False


@pytest.mark.parametrize("kind", ["message", "traceback"])
@pytest.mark.parametrize("manifest", [False, True])
def test_worker_rendering_fault_preserves_primary_and_unconditionally_closes_engine(tmp_path, monkeypatch, kind, manifest):
    from tensorfold import hub
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tensorfold.families import qwen4_exp
    from tensorfold import serve_options

    primary, diagnostic = rendering_fault(monkeypatch, kind)
    events = []

    class OwnedEngine:
        capacity_plan = {}
        sm70_profile = "reference"

        def follow(self):
            events.append("follow")
            raise primary

        def close(self):
            events.append("close")

    # Only external model/import boundaries are substituted. The real worker's
    # report, failure transport and an actual borrowed pipe run unchanged.
    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(hub, "resolve", lambda *_args, **_kw: tmp_path)
    monkeypatch.setattr(serve_options, "drafter_metadata_source", lambda _source: tmp_path)
    options = {}
    def construct(*_args, **kwargs):
        options.update(kwargs)
        return OwnedEngine()
    monkeypatch.setattr(qwen4_exp, "cuda_engine", construct)
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(FAILURE_FD_ENV, str(write_fd))
    args = SimpleNamespace(rank=1, wheel=tmp_path, model=tmp_path, mtp=tmp_path,
                           master_port=29607, tokens=128, out=tmp_path, requested_sm70_profile=None, repetitions=1)
    if manifest:
        args.execution_profile_manifest = tmp_path / "explicit.json"
    try:
        with pytest.raises(ValueError) as caught:
            probe.worker(args)
        assert caught.value is primary and events == ["follow", "close"]
        assert options.get("execution_profile_manifest") == (args.execution_profile_manifest if manifest else None)
        assert ("execution_profile_manifest" in options) == manifest
        assert retained_failure(primary, diagnostic)
    finally:
        os.close(write_fd)
        os.close(read_fd)


@pytest.mark.parametrize("kind", ["message", "traceback"])
def test_launcher_rendering_fault_preserves_primary_restores_handlers_and_closes_pipes(tmp_path, monkeypatch, kind):
    from tensorfold.cuda import expert_launch

    args = probe.arguments(cli(tmp_path))
    primary, diagnostic = rendering_fault(monkeypatch, kind)
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    descriptors = []
    actual_pipe = os.pipe2

    def owned_pipe(flags):
        pair = actual_pipe(flags)
        descriptors.extend(pair)
        return pair

    def failed_supervision(*_args, **_kwargs):
        for signum, handler in handlers.items():
            assert signal.getsignal(signum) is not handler
        raise primary

    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(probe.os, "pipe2", owned_pipe)
    monkeypatch.setattr(expert_launch, "supervise", failed_supervision)
    with pytest.raises(ValueError) as caught:
        probe.launch(args, [])
    assert caught.value is primary and retained_failure(primary, diagnostic)
    for signum, handler in handlers.items():
        assert signal.getsignal(signum) is handler
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_launcher_retirement_allocates_no_new_enumeration_after_supervision(tmp_path, monkeypatch):
    from tensorfold.cuda import expert_launch

    args = probe.arguments(cli(tmp_path))
    primary = ValueError("original supervision failure")
    retired = False
    descriptors = []
    actual_pipe = os.pipe2
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def owned_pipe(flags):
        pair = actual_pipe(flags)
        descriptors.extend(pair)
        return pair

    def prepared_enumeration(*args, **kwargs):
        if retired:
            raise MemoryError("cleanup iteration allocation must not occur")
        return enumerate(*args, **kwargs)

    def failed_supervision(*_args, **_kwargs):
        nonlocal retired
        retired = True
        raise primary

    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(probe, "enumerate", prepared_enumeration, raising=False)
    monkeypatch.setattr(probe.os, "pipe2", owned_pipe)
    monkeypatch.setattr(expert_launch, "supervise", failed_supervision)
    with pytest.raises(ValueError) as caught:
        probe.launch(args, [])
    assert caught.value is primary
    for signum, handler in handlers.items():
        assert signal.getsignal(signum) is handler
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_fatal_notification_is_latched_and_allows_bounded_report_retirement(tmp_path, monkeypatch):
    import json
    from tensorfold.cuda import expert_launch

    args = probe.arguments(cli(tmp_path))
    now = [100.0]
    monkeypatch.setattr(probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})

    def fatal_supervision(_commands, _environments, *, failure_probe, pass_fds, grace):
        assert grace == 30
        os.write(pass_fds[0], b"F")
        assert failure_probe() is False
        first = json.loads((args.out / "fatal-notification.json").read_text())
        assert first["fatal_notification"]["bytes_observed"] == 1
        now[0] += probe.FATAL_REPORT_GRACE_SECONDS - .5
        assert failure_probe() is False
        os.write(pass_fds[0], b"F")
        assert failure_probe() is False  # A later byte must not reset the deadline.
        now[0] += .5
        assert failure_probe() is True
        return 1

    monkeypatch.setattr(expert_launch, "supervise", fatal_supervision)
    assert probe.launch(args, []) == 1
    final = json.loads((args.out / "launch-final.json").read_text())
    assert final["complete"] is False and final["timed_out"] is False
    assert final["fatal_notification"]["bytes_observed"] == 2
    assert final["fatal_notification"]["first_elapsed_s"] == 0


@pytest.mark.parametrize("interruption", ["stop", "deadline"])
def test_stop_and_process_deadline_override_fatal_report_grace(tmp_path, monkeypatch, interruption):
    import json
    from tensorfold.cuda import expert_launch

    args = probe.arguments(cli(tmp_path))
    args.timeout_seconds = 1
    now = [100.0]
    monkeypatch.setattr(probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})

    def interrupted_supervision(_commands, _environments, *, failure_probe, pass_fds, **_kwargs):
        os.write(pass_fds[0], b"F")
        assert failure_probe() is False
        if interruption == "stop":
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        else:
            now[0] += 1
        assert failure_probe() is True
        return 1

    monkeypatch.setattr(expert_launch, "supervise", interrupted_supervision)
    assert probe.launch(args, []) == 1
    final = json.loads((args.out / "launch-final.json").read_text())
    assert final["complete"] is False
    assert final["stopped"] is (interruption == "stop")
    assert final["timed_out"] is (interruption == "deadline")


def test_final_fatal_byte_cannot_be_reported_as_success(tmp_path, monkeypatch):
    import json
    from tensorfold.cuda import expert_launch

    args = probe.arguments(cli(tmp_path))
    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})

    def late_notification(_commands, _environments, *, pass_fds, **_kwargs):
        os.write(pass_fds[0], b"FF")
        return 0

    monkeypatch.setattr(expert_launch, "supervise", late_notification)
    assert probe.launch(args, []) == 1
    final = json.loads((args.out / "launch-final.json").read_text())
    assert final["exit_code"] == 1 and final["complete"] is False
    assert final["fatal_notification"]["bytes_observed"] == 2


def test_unset_profile_uses_actual_reference_constructor_policy_and_records_native_flags(tmp_path, monkeypatch):
    from tensorfold.cuda.sm70 import execution_policy, family_ops, routing_native
    from tensorfold.families.qwen4_exp.cuda.prefix_policy import identity as prefix_identity
    from tensorfold.families.qwen4_exp.cuda import (
        dspark_context_graphs, dspark_proposal_graphs, gguf_prefill, gguf_weights, padded_linears,
    )

    # Use the actual supported reference selection without querying CUDA or
    # altering qualification. Restore every process-fixed policy owner afterward.
    for owner, name in (
        (execution_policy, "_SELECTED"), (family_ops, "_NATIVE_FAMILY"),
        (family_ops, "_NATIVE_ATTENTION"), (routing_native, "_NATIVE_LOGITS"),
        (padded_linears, "_ENABLED"), (gguf_weights.GGUFExperts, "implementation"),
        (gguf_weights.GGUFConcat, "implementation"), (gguf_prefill, "WORKSPACE_BUDGET_BYTES"),
        (dspark_context_graphs, "_ENABLED"), (dspark_proposal_graphs, "_ENABLED"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.delenv(execution_policy.PROFILE_ENV, raising=False)
    args = probe.arguments(cli(tmp_path))
    selected = execution_policy.configure({}, tp=1, torch=None)
    assert selected == "reference"
    record = {}
    probe.record_execution_profile(SimpleNamespace(sm70_profile=selected), record,
                                   requested=args.requested_sm70_profile)
    assert record == {
        "requested_sm70_profile": None, "sm70_profile": "reference",
        "prefix_policy": {"identity": prefix_identity("reference"), "state_epoch": None},
        "native_flags": {"family": False, "private_attention": False, "routing": False,
            "padded": False, "gguf_dense_implementation": gguf_weights.GGUFDense.implementation,
            "gguf_concat_implementation": "reference", "gguf_experts_implementation": "reference",
            "gguf_experts_packed_rows": gguf_weights.GGUFExperts.packed_rows},
        "owner_execution": {"prepared_identity": None, "roles": {}},
    }
    assert execution_policy.PROFILE_ENV not in os.environ


def test_execution_record_reads_exact_flags_and_rejects_requested_actual_mismatch(monkeypatch):
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.families.qwen4_exp.cuda import gguf_weights

    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", True)
    monkeypatch.setattr(gguf_weights.GGUFDense, "implementation", "native32")
    record = {}
    with pytest.raises(ValueError, match="actual constructor selection"):
        probe.record_execution_profile(SimpleNamespace(sm70_profile="reference"), record, requested="optimized")
    assert record["sm70_profile"] == "reference" and record["requested_sm70_profile"] == "optimized"
    assert record["native_flags"]["private_attention"] is True
    assert record["native_flags"]["gguf_dense_implementation"] == "native32"
    assert family_ops._NATIVE_ATTENTION is True and gguf_weights.GGUFDense.implementation == "native32"


@pytest.mark.parametrize("prepared", [False, True])
def test_owner_observation_uses_adopted_buffers_and_binds_actual_prepared_identity(prepared):
    from tensorfold.engine.model_identity import canonical_runtime_json
    from tensorfold.cuda.sm70 import family_ops

    private_attention = family_ops._NATIVE_ATTENTION
    identity = {"specification": {"kernel_family": "intent-only-native",
                                   "graph_family": "intent-only-graph"},
                "roles": {"decode": 32, "prefill": 256}}
    preparation = SimpleNamespace(identity=lambda: identity, prepared=True, closed=False,
                                  specification={"trial": True}) if prepared else None
    decode = SimpleNamespace(owner_kernel_family="observed-native", owner_graph_family="uncaptured",
                             history_owner=SimpleNamespace(native_collective="observed-full8"))
    prefill = SimpleNamespace(owner_kernel_family="observed-reference", owner_graph_family="uncaptured",
                              history_owner=SimpleNamespace(native_collective="observed-prefill-full8"))
    engine = SimpleNamespace(sm70_profile="reference", _owner_preparation=preparation,
                              multi=SimpleNamespace(buf=decode, mbuf=None, pbuf=prefill))
    report = {}
    probe.record_execution_profile(engine, report, requested=None)
    observed = report["owner_execution"]
    assert observed["roles"] == {
        "decode": {"kernel_family": "observed-native", "graph_family": "uncaptured",
                   "collective_family": "observed-full8"},
        "prefill": {"kernel_family": "observed-reference", "graph_family": "uncaptured",
                    "collective_family": "observed-prefill-full8"}}
    if prepared:
        assert observed["prepared_identity"] == {
            "sha256": hashlib.sha256(canonical_runtime_json(identity).encode()).hexdigest(),
            "prepared": True, "closed": False, "trial": True}
    else:
        assert observed["prepared_identity"] is None
    # The bounded report keeps neither the preparation object nor its potentially
    # large allocation graph, and changing intent cannot rewrite observations.
    identity["roles"]["decode"] = 99
    assert "intent-only" not in json.dumps(observed) and "allocations" not in observed
    assert report["native_flags"]["private_attention"] is private_attention
    assert family_ops._NATIVE_ATTENTION is private_attention


def test_requested_profile_validation_is_deferred_to_existing_constructor(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "constructor-must-refuse-this")
    args = probe.arguments(cli(tmp_path))
    assert args.requested_sm70_profile == "constructor-must-refuse-this"
    assert os.environ["TENSORFOLD_SM70_PROFILE"] == "constructor-must-refuse-this"


def test_nvtx_push_failure_prevents_generation_and_retires_collector_owner(tokenizer, nvtx_markers, monkeypatch):
    primary = RuntimeError("NVTX initialization failed")
    limit = RequestLimit(1)
    engine = Generation(limit)
    closed = []
    actual_close = probe.TokenCollector.close

    def failed_push(_label):
        assert limit.used == 1
        raise primary

    def close_collector(collector):
        actual_close(collector)
        closed.append(collector)

    monkeypatch.setattr(nvtx_markers.api, "range_push", failed_push)
    monkeypatch.setattr(probe.TokenCollector, "close", close_collector)
    with pytest.raises(RuntimeError) as caught:
        probe.generate_reply(engine, tokenizer, [0], count=3, draft=True, limit=limit, result={})
    assert caught.value is primary and engine.calls == [] and limit.used == 0
    assert len(closed) == 1 and closed[0].closed and nvtx_markers.events == []


def test_nvtx_pop_failure_preserves_generation_primary_and_all_retirement(tokenizer, nvtx_markers, monkeypatch):
    primary = ValueError("generation failed")
    cleanup = RuntimeError("NVTX pop failed")
    limit = RequestLimit(1)
    engine = Generation(limit, failure=primary)

    def failed_pop():
        nvtx_markers.events.append(("pop", None))
        raise cleanup

    monkeypatch.setattr(nvtx_markers.api, "range_pop", failed_pop)
    with pytest.raises(ValueError) as caught:
        probe.generate_reply(engine, tokenizer, [0], count=3, draft=True, limit=limit, result={})
    assert caught.value is primary and retained_failure(primary, cleanup)
    assert limit.used == 0 and engine.callback.closed
    assert nvtx_markers.events == [("push", "tensorfold/mtp-target"), ("pop", None)]


def test_nvtx_pop_failure_after_success_is_reported_and_retires(tokenizer, nvtx_markers, monkeypatch):
    cleanup = RuntimeError("NVTX pop failed")
    limit = RequestLimit(1)
    engine = Generation(limit)

    def failed_pop():
        raise cleanup

    monkeypatch.setattr(nvtx_markers.api, "range_pop", failed_pop)
    with pytest.raises(RuntimeError) as caught:
        probe.generate_reply(engine, tokenizer, [0], count=3, draft=True, limit=limit, result={})
    assert caught.value is cleanup and engine.callback.closed and limit.used == 0


def test_nvtx_range_includes_final_decode_failure_and_unwinds(tokenizer, nvtx_markers, monkeypatch):
    from tensorfold import token_metadata

    primary = ValueError("final decode failed")
    limit = RequestLimit(1)
    engine = Generation(limit)

    def failed_decode(_tokenizer, ids):
        assert ids == [1] and limit.used == 1
        assert nvtx_markers.events == [("push", "tensorfold/mtp-target")]
        raise primary

    monkeypatch.setattr(token_metadata, "decode_presentation", failed_decode)
    with pytest.raises(ValueError) as caught:
        probe.generate_reply(engine, tokenizer, [0], count=3, draft=True, limit=limit, result={})
    assert caught.value is primary and engine.callback.closed and limit.used == 0
    assert nvtx_markers.events == [("push", "tensorfold/mtp-target"), ("pop", None)]


def test_repetitions_use_one_engine_owner_and_release_previous_reply_pair(tmp_path, tokenizer, nvtx_markers):
    import json

    args = SimpleNamespace(repetitions=3, serial_reference=True, tokens=3, out=tmp_path)
    report = {"replies": []}
    limit = RequestLimit(1)
    lifetimes = []

    class Stats(dict):
        pass

    class Repeated(Generation):
        def generate(self, *arguments, **kwargs):
            iteration = report["iteration"]
            assert len(report["replies"]) <= 2
            assert all(row["iteration"] == iteration for row in report["replies"])
            if kwargs["draft"]:
                assert all(reference() is None for reference in lifetimes)
            stats = Stats(super().generate(*arguments, **kwargs))
            stats["cached"] = 0 if iteration == 1 else 1
            lifetimes.append(weakref.ref(stats))
            return stats

    engine = Repeated(limit)
    probe.repeat_replies(engine, tokenizer, [0], args, report, limit)
    assert engine.calls == [True, False] * 3 and limit.used == 0
    assert report["completed_iterations"] == 3 and len(report["replies"]) == 2
    assert sum(reference() is not None for reference in lifetimes) == 2
    files = sorted(tmp_path.glob("iteration-*.json"))
    assert len(files) == 6 and not (tmp_path / "mtp-target.json").exists()
    for path in files:
        row = json.loads(path.read_text())
        assert row["complete"] and row["token_ids"] == [1]
        assert row["stats"]["cached"] == (0 if row["iteration"] == 1 else 1)


def test_default_repetition_keeps_existing_reply_filenames(tmp_path, tokenizer, nvtx_markers):
    args = SimpleNamespace(repetitions=1, serial_reference=True, tokens=3, out=tmp_path)
    report = {"replies": []}
    limit = RequestLimit(1)
    probe.repeat_replies(Generation(limit), tokenizer, [0], args, report, limit)
    assert (tmp_path / "mtp-target.json").exists() and (tmp_path / "serial-reference.json").exists()
    assert not list(tmp_path.glob("iteration-*.json"))
    assert report["completed_iterations"] == 1


def test_prefix_expectation_requires_explicit_profile_and_repeated_case(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "source-family-prefix-trial")
    with pytest.raises(SystemExit):
        probe.arguments(cli(tmp_path) + ["--expect-prefix-reuse"])
    args = probe.arguments(cli(tmp_path) + ["--expect-prefix-reuse", "--repetitions", "10"])
    assert args.expect_prefix_reuse and args.repetitions == 10
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "reference")
    with pytest.raises(SystemExit):
        probe.arguments(cli(tmp_path) + ["--expect-prefix-reuse", "--repetitions", "10"])


@pytest.mark.parametrize("profile", ["source-packed-trial", "source-packed-graphs-trial"])
def test_prefix_expectation_accepts_each_explicit_distinct_packed_epoch(tmp_path, monkeypatch, profile):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", profile)
    args = probe.arguments(cli(tmp_path) + ["--expect-prefix-reuse", "--repetitions", "10"])
    assert args.expect_prefix_reuse and args.requested_sm70_profile == profile


@pytest.mark.parametrize("profile,expected", [(None, False), ("source-family-trial", False),
    ("source-packed-trial", False), ("source-packed-graphs-trial", True)])
def test_worker_requests_graphs_only_for_explicit_graph_profile(tmp_path, monkeypatch, profile, expected):
    from tensorfold import hub, serve_options
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tensorfold.families import qwen4_exp

    primary = ValueError("stop at the real constructor boundary")
    seen = []

    def constructor(*args, **kwargs):
        seen.append(kwargs["graphs"])
        raise primary

    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(hub, "resolve", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(serve_options, "drafter_metadata_source", lambda _source: tmp_path)
    monkeypatch.setattr(qwen4_exp, "cuda_engine", constructor)
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(FAILURE_FD_ENV, str(write_fd))
    args = SimpleNamespace(rank=1, wheel=tmp_path, model=tmp_path, mtp=tmp_path,
        master_port=29607, tokens=128, out=tmp_path, requested_sm70_profile=profile, repetitions=1)
    try:
        with pytest.raises(ValueError) as failure:
            probe.worker(args)
        assert failure.value is primary and seen == [expected]
    finally:
        os.close(write_fd)
        os.close(read_fd)


@pytest.mark.parametrize("captured,keys,allowed", [(False, True, False), (True, False, False), (True, True, True)])
def test_graph_trial_record_requires_actual_admitted_owner_and_reports_its_budget(captured, keys, allowed):
    from tensorfold.cuda.sm70.execution_policy import PACKED_GRAPH_TRIAL
    from tensorfold.families.qwen4_exp.cuda.multi_graphs import UNIQUE_CAPTURE_CONTRACT

    admission = {"planned_bytes": 4096, "room_bytes": 128}
    graphs = SimpleNamespace(verify={1: object()} if keys else {}, mtp={1: object()} if keys else {})
    engine = SimpleNamespace(sm70_profile=PACKED_GRAPH_TRIAL, graphs_enabled=True,
        multi=SimpleNamespace(graphs=graphs, graph_record={"captured": captured}),
        w=SimpleNamespace(meta={"unique_capture_contract": UNIQUE_CAPTURE_CONTRACT}),
        capacity_plan={"multi_graphs": admission})
    report = {}
    if allowed:
        probe.record_execution_profile(engine, report, requested=PACKED_GRAPH_TRIAL)
        assert report["unique_graph_trial"]["admission"] is admission
        assert report["unique_graph_trial"]["contract"] == UNIQUE_CAPTURE_CONTRACT
    else:
        with pytest.raises(ValueError, match="admitted unique-history graph owner"):
            probe.record_execution_profile(engine, report, requested=PACKED_GRAPH_TRIAL)


@pytest.mark.parametrize("wrong", [False, True])
def test_private_prefix_adapter_preserves_raw_result_before_check_failure(tmp_path, tokenizer, nvtx_markers, wrong):
    import json

    args = SimpleNamespace(repetitions=2, serial_reference=True, tokens=3, out=tmp_path, expect_prefix_reuse=True)
    report, limit = {"replies": []}, RequestLimit(1)

    class Cached:
        def generate(self, prompt, count, sampling, callback, *, draft, stop_eos):
            assert prompt == [0, 1] and count == 3 and stop_eos and limit.used == 1
            callback([1])
            return {"cached": 0 if report["iteration"] == 1 else (0 if wrong else 1)}

    engine = Cached()
    if wrong:
        with pytest.raises(ValueError, match="declared cold/full-prefix"):
            probe.repeat_replies(engine, tokenizer, [0, 1], args, report, limit)
    else:
        probe.repeat_replies(engine, tokenizer, [0, 1], args, report, limit)
    row = json.loads((tmp_path / "iteration-0002-mtp-target.json").read_text())
    assert row["token_ids"] == [1] and row["prefix_reuse_check"]["passed"] is not wrong
    assert row["prefix_reuse_check"]["expected"] == 1 and limit.used == 0


def test_private_prefix_adapter_refuses_foreign_state_before_host_or_request_publication(monkeypatch):
    monkeypatch.setattr(probe, "check_engine", lambda engine: None)
    engine = SimpleNamespace(sm70_profile="source-family-prefix-trial",
                             retention_catalog=SimpleNamespace(states=[SimpleNamespace(kv_identity=())]))
    with pytest.raises(ValueError, match="State owners lack"):
        probe.run_leader(engine, None, SimpleNamespace(expect_prefix_reuse=True), {})


@pytest.mark.parametrize("write_failure", [False, True])
def test_later_iteration_failure_preserves_prior_outputs_partial_ids_and_original_error(
        tmp_path, tokenizer, nvtx_markers, monkeypatch, write_failure):
    import json

    args = SimpleNamespace(repetitions=3, serial_reference=True, tokens=3, out=tmp_path)
    report = {"replies": []}
    limit = RequestLimit(1)
    primary, secondary = ValueError("second iteration generation failed"), OSError("failed row write")

    class Failing(Generation):
        def generate(self, *arguments, **kwargs):
            if report["iteration"] == 2:
                self.failure = primary
            return super().generate(*arguments, **kwargs)

    if write_failure:
        actual_write = probe.write_json

        def failed_write(path, value):
            if Path(path).name == "iteration-0002-mtp-target.json":
                raise secondary
            return actual_write(path, value)

        monkeypatch.setattr(probe, "write_json", failed_write)
    engine = Failing(limit)
    with pytest.raises(ValueError) as caught:
        probe.repeat_replies(engine, tokenizer, [0], args, report, limit)
    assert caught.value is primary and limit.used == 0 and engine.callback.closed
    assert report["completed_iterations"] == 1 and report["iteration"] == 2
    assert report["replies"][0]["token_ids"] == [1] and not report["replies"][0]["complete"]
    assert (tmp_path / "iteration-0001-mtp-target.json").exists()
    assert (tmp_path / "iteration-0001-serial-reference.json").exists()
    assert not (tmp_path / "iteration-0002-serial-reference.json").exists()
    if write_failure:
        assert retained_failure(primary, secondary)
    else:
        row = json.loads((tmp_path / "iteration-0002-mtp-target.json").read_text())
        assert row["iteration"] == 2 and row["token_ids"] == [1] and not row["complete"]


def test_legacy_installed_supervisor_keeps_default_call_without_new_keywords(tmp_path, monkeypatch):
    import json
    from tensorfold.cuda import expert_launch

    args = probe.arguments(cli(tmp_path))
    calls = []

    def legacy(commands, environments, *, grace, failure_probe, pass_fds):
        calls.append((len(commands), grace))
        return 0

    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(expert_launch, "supervise", legacy)
    assert probe.launch(args, []) == 0 and calls == [(8, 30)]
    record = json.loads((args.out / "launch-final.json").read_text())
    assert record["supervision"]["available"] is False


def test_requested_departure_override_requires_installed_capability_before_launch(tmp_path, monkeypatch):
    from tensorfold.cuda import expert_launch

    args = probe.arguments([*cli(tmp_path), "--instrumented-departure-grace-seconds", "120"])

    def legacy(*_args, **_kwargs):
        pytest.fail("unsupported override reached process launch")

    monkeypatch.setattr(expert_launch, "supervise", legacy)
    with pytest.raises(ValueError, match="does not support instrumented departure grace"):
        probe.launch(args, [])
    assert not args.out.exists()


def test_instrumented_departure_and_exit_record_map_to_rank_order(tmp_path, monkeypatch):
    import json
    from tensorfold.cuda import expert_launch

    args = probe.arguments([*cli(tmp_path), "--instrumented-departure-grace-seconds", "120"])

    def observed(_commands, _environments, *, grace, failure_probe, pass_fds, departure_grace=None, on_exit=None):
        assert grace == 30 and departure_grace == 120
        on_exit({"reason": "all-children-exited-zero", "children": [{"command_index": 0, "pid": 42,
                  "pre_containment_status": 0, "final_returncode": 0}]})
        return 0

    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(expert_launch, "supervise", observed)
    assert probe.launch(args, []) == 0
    record = json.loads((args.out / "launch-final.json").read_text())
    assert record["supervision"]["available"] is True
    assert record["supervision"]["rank_order"] == [1, 2, 3, 4, 5, 6, 7, 0]
    assert record["supervision"]["children"][0]["pid"] == 42


def finite_engine_capacity(*, ceiling=None, resident=26, materialized=True, commitments=None):
    return SimpleNamespace(capacity_plan={"session_capacity": {"context_tokens": probe.CONTEXT,
        "configured_ceiling": ceiling, "resident_capacity": resident,
        "full_context_commitments": resident if commitments is None else commitments},
        "kv_full_windows": {"materialized": materialized}}, tp=8, depth=2, confidence=.7,
        context_window=262144, indexer_history="gpu", kv_pair=SimpleNamespace(key_dtype="int8", value_dtype="int8"),
        w=SimpleNamespace(mtp=object(), meta={"execution_provider": "sm70", "execution_dtype": "float16",
            "full": SimpleNamespace(hidden=2560, layers=48, vocab=248320)}),
        scheduler=SimpleNamespace(_request_limit=None), retention_catalog=object())


def test_all_eight_ready_reports_fit_the_funded_slices_and_overflow_remains_a_failure(tmp_path):
    import json

    assert probe.FOLLOWER_RECEIPT_BYTES == 128 << 10
    assert probe.MAX_RECEIPT_BYTES == 1507328
    total = 0
    for rank in range(probe.WORLD):
        # The actual failed reports were 89,341..89,349 bytes. Preserve an
        # independently bounded payload above that old 64 KiB refusal boundary.
        value = {"rank": rank, "capacity_plan": {"graph_manifest": "x" * 90000}, "complete": False}
        path = tmp_path / f"rank-{rank}-ready.json"
        allowance = probe.LEADER_RECEIPT_BYTES if rank == 0 else probe.FOLLOWER_RECEIPT_BYTES
        probe.write_report(path, value, allowance=allowance)
        assert json.loads(path.read_text()) == value
        total += path.stat().st_size
    assert total + probe.SUPERVISOR_RECEIPT_BYTES < probe.MAX_RECEIPT_BYTES
    value = {"too_large": "x" * probe.FOLLOWER_RECEIPT_BYTES}
    with pytest.raises(ValueError, match="report slice exceeded"):
        probe.write_report(tmp_path / "rank-overflow.json", value, allowance=probe.FOLLOWER_RECEIPT_BYTES)
    assert not (tmp_path / "rank-overflow.json").exists()
    assert json.loads((tmp_path / "rank-overflow.overflow.json").read_text()) == value


def test_actual_finite_caller_receives_full_eight_rank_report_commitment(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import retention_resources

    engine = finite_engine_capacity(ceiling=24, resident=24)
    captured = []
    engine.scheduler.configure_requests = lambda limit: captured.append(("configure", limit.used))
    args = SimpleNamespace(session_ceiling=24, serial_reference=True, tokens=128)
    monkeypatch.setattr(probe, "prepared_prompt", lambda *_args: ([0], {"prepared": True}))
    monkeypatch.setattr(probe, "write_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(probe, "repeat_replies", lambda *_args: None)

    def finalize(catalog, *, tokenizer, scheduler, request_limit, caller):
        assert type(caller) is retention_resources.FiniteBackendCaller
        assert caller.max_receipt_bytes == (512 + 7 * 128 + 64) << 10
        assert caller.retained_results == 2 and request_limit.maximum == 1 and request_limit.used == 0
        captured.append(("admit", caller.max_receipt_bytes))
        return {"admitted": True}

    monkeypatch.setattr(retention_resources, "finalize_backend_resources", finalize)
    args.out = Path("unused")
    report = {}
    probe.run_leader(engine, None, args, report)
    assert captured == [("admit", probe.MAX_RECEIPT_BYTES), ("configure", 0)]
    assert report["host_admission"] == {"admitted": True}


def test_ready_publication_failure_reaches_parent_before_blocked_cleanup(tmp_path, monkeypatch):
    import json
    import threading
    from tensorfold import hub, serve_options
    from tensorfold.cuda import expert_launch
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tensorfold.families import qwen4_exp

    args = probe.arguments(cli(tmp_path))
    now = [100.0]
    closed, release, retired = threading.Event(), threading.Event(), threading.Event()
    outcomes = []

    class Engine:
        capacity_plan = {"oversized_manifest": "x" * probe.FOLLOWER_RECEIPT_BYTES}
        def follow(self):
            pytest.fail("failed ready publication must not enter follow")
        def close(self):
            closed.set()
            assert release.wait(3), "test owner failed to release its blocked cleanup"

    monkeypatch.setattr(probe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(probe, "record_execution_profile", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hub, "resolve", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(serve_options, "drafter_metadata_source", lambda _source: tmp_path)
    monkeypatch.setattr(qwen4_exp, "cuda_engine", lambda *_args, **_kwargs: Engine())

    def supervised(_commands, _environments, *, failure_probe, pass_fds, **_kwargs):
        monkeypatch.setenv(FAILURE_FD_ENV, str(pass_fds[0]))
        worker_args = SimpleNamespace(**{**vars(args), "rank": 1})
        def run():
            try:
                probe.worker(worker_args)
            except BaseException as error:
                outcomes.append(error)
            finally:
                retired.set()
        thread = threading.Thread(target=run)
        thread.start()
        try:
            assert closed.wait(2) and not retired.is_set()
            assert failure_probe() is False
            notification = json.loads((args.out / "fatal-notification.json").read_text())
            assert notification["fatal_notification"]["bytes_observed"] == 1
            now[0] += probe.FATAL_REPORT_GRACE_SECONDS
            assert failure_probe() is True and not retired.is_set()
        finally:
            release.set()
            thread.join(2)
        assert not thread.is_alive() and retired.is_set()
        return 1

    monkeypatch.setattr(expert_launch, "supervise", supervised)
    assert probe.launch(args, []) == 1
    assert len(outcomes) == 1 and isinstance(outcomes[0], ValueError)
    assert "report slice exceeded" in str(outcomes[0])
    assert (args.out / "rank-1-ready.overflow.json").exists()
    assert (args.out / "rank-1-failure.overflow.json").exists()
    final = json.loads((args.out / "launch-final.json").read_text())
    assert not final["complete"] and not final["timed_out"]


def test_notification_failure_is_secondary_and_does_not_skip_close(tmp_path, monkeypatch):
    from tensorfold import hub, serve_options
    from tensorfold.cuda import cohort
    from tensorfold.families import qwen4_exp

    primary, secondary = ValueError("original follow failure"), OSError("notification write failed")
    events = []
    class Engine:
        capacity_plan = {}
        def follow(self):
            raise primary
        def close(self):
            events.append("close")
    def notify():
        events.append("notify")
        raise secondary
    monkeypatch.setattr(cohort, "notify_failure", notify)
    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(probe, "record_execution_profile", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hub, "resolve", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(serve_options, "drafter_metadata_source", lambda _source: tmp_path)
    monkeypatch.setattr(qwen4_exp, "cuda_engine", lambda *_args, **_kwargs: Engine())
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(cohort.FAILURE_FD_ENV, str(write_fd))
    args = SimpleNamespace(rank=1, wheel=tmp_path, model=tmp_path, mtp=tmp_path, master_port=29607,
        tokens=128, out=tmp_path, requested_sm70_profile=None, repetitions=1, session_ceiling=24)
    try:
        with pytest.raises(ValueError) as caught:
            probe.worker(args)
        assert caught.value is primary and retained_failure(primary, secondary)
        assert events == ["notify", "close"]
    finally:
        os.close(write_fd)
        os.close(read_fd)


def test_graph_execution_snapshot_distinguishes_key_availability_from_request_cache_hits():
    bound = SimpleNamespace(hits=3, misses=2, lane_declines=1, verify_hits=1, mtp_hits=2)
    owner = SimpleNamespace(bound=bound, verify={1: object(), 2: object()}, mtp={1: object()}, eager_declines=4)
    engine = SimpleNamespace(multi=SimpleNamespace(graphs=owner))
    before = probe.graph_execution_snapshot(engine)
    bound.hits += 17
    bound.verify_hits += 7
    bound.mtp_hits += 10
    bound.misses += 2
    after = probe.graph_execution_snapshot(engine)
    assert before["captured_keys"] == after["captured_keys"] == {"verify": 2, "mtp": 1}
    assert before["counters"]["bound.hits"] == 3 and after["counters"]["bound.hits"] == 20
    assert probe.graph_execution_delta(before, after) == {
        "owner.eager_declines": 0, "bound.hits": 17, "bound.misses": 2,
        "bound.lane_declines": 0, "bound.verify_hits": 7, "bound.mtp_hits": 10}
    assert probe.graph_execution_snapshot(SimpleNamespace()) is None
    assert probe.graph_execution_snapshot(SimpleNamespace(multi=SimpleNamespace(graphs=None))) is None
    assert probe.graph_execution_delta(None, after) is None


def test_graph_snapshot_never_calls_device_methods_descriptors_or_unbounded_container_iteration():
    class Device:
        def item(self):
            pytest.fail("device reads are not part of the snapshot")
    class Owner:
        @property
        def hits(self):
            pytest.fail("descriptor must not execute")
    owner = Owner()
    owner.bound = SimpleNamespace(hits=Device(), misses=True, lane_declines=5)
    owner.verify = {1: Device()}
    owner.mtp = {1: Device(), 2: Device()}
    owner.unrelated = {str(n): n for n in range(10000)}
    snapshot = probe.graph_execution_snapshot(SimpleNamespace(multi=SimpleNamespace(graphs=owner)))
    assert snapshot == {"counters": {"bound.lane_declines": 5}, "captured_keys": {"verify": 1, "mtp": 2}}
    # A counter first exposed after the initial snapshot has no invented zero baseline.
    owner.bound.mtp_hits = 100
    after = probe.graph_execution_snapshot(SimpleNamespace(multi=SimpleNamespace(graphs=owner)))
    assert probe.graph_execution_delta(snapshot, after) == {"bound.lane_declines": 0}


@pytest.mark.parametrize("rank", [0, 1])
def test_worker_brackets_all_requests_after_constructor_warmup_and_before_cleanup(tmp_path, monkeypatch, rank):
    import json
    from tensorfold import hub, serve_options, token_metadata
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tensorfold.families import qwen4_exp

    bound = SimpleNamespace(hits=3, misses=2, lane_declines=1)
    owner = SimpleNamespace(bound=bound, verify={1: object()}, mtp={1: object()})
    events = []
    class Engine:
        capacity_plan = {}
        multi = SimpleNamespace(graphs=owner)
        def __init__(self):
            # Real engines own multi directly, as their actual constructor does.
            self.multi = self.__class__.multi
        def follow(self):
            events.append("requests")
            bound.hits += 11
        def close(self):
            events.append("close")
            bound.hits = 0  # Snapshot must precede owner retirement/reset.
    engine = Engine()
    monkeypatch.setattr(probe, "installed_identity", lambda _wheel: {})
    monkeypatch.setattr(probe, "record_execution_profile", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hub, "resolve", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(serve_options, "drafter_metadata_source", lambda _source: tmp_path)
    monkeypatch.setattr(qwen4_exp, "cuda_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(token_metadata, "load_tokenizer", lambda *_args: None)
    monkeypatch.setattr(token_metadata, "tokenizer_identity", lambda *_args: "test-tokenizer")
    monkeypatch.setattr(probe, "run_leader", lambda *_args: engine.follow())
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(FAILURE_FD_ENV, str(write_fd))
    args = SimpleNamespace(rank=rank, wheel=tmp_path, model=tmp_path, mtp=tmp_path, master_port=29607,
        tokens=128, out=tmp_path, requested_sm70_profile=None, repetitions=1, session_ceiling=24)
    try:
        assert probe.worker(args) == 0
    finally:
        os.close(write_fd)
        os.close(read_fd)
    ready = json.loads((tmp_path / f"rank-{rank}-ready.json").read_text())
    final = json.loads((tmp_path / f"rank-{rank}-final.json").read_text())
    assert ready["graph_execution"]["before_requests"]["counters"]["bound.hits"] == 3
    assert ready["graph_execution"]["after_requests"] is None
    assert final["graph_execution"]["after_requests"]["counters"]["bound.hits"] == 14
    assert final["graph_execution"]["delta"] == {"bound.hits": 11, "bound.misses": 0, "bound.lane_declines": 0}
    assert final["complete"] and events == ["requests", "close"] and bound.hits == 0
