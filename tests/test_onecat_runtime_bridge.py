"""Cold credit/loader/adoption controls, not substitute GPU/model evidence."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition
from tensorfold.families.qwen4_exp.cuda import gguf_weights as weights
from tensorfold.families.qwen4_exp.cuda import owned_experts
from tensorfold.families.qwen4_exp.cuda.onecat_experts import OneCatGU, OneCatOwner, _Sink, _key


def prototype(rank=0):
    partition = ExpertPartition(8, rank, 512, 640, 4)
    rows = tuple((expert * 640 + partition.column, 160) for expert in range(partition.first, partition.stop))
    kinds = [22] * 20 + [21] * 10 + [18] * 17 + [23]
    layers = []
    for index, kind in enumerate(kinds):
        row_bytes = 10 * {18: 98, 21: 110, 22: 82, 23: 136}[kind]

        def matrix(role):
            source = SimpleNamespace(path=Path("/immutable/model.gguf"), offset=(index * 2 + role) * 1000000,
                                     ggml_type=kind, row_bytes=row_bytes, rows=512 * 640)
            return weights.MatrixPlan(source, rows, ((0, 2560),), True)

        bank = weights.ExpertPlan(matrix(0), matrix(1), None, (), 512, 160, 2560, partition)
        layers.append(SimpleNamespace(moe=SimpleNamespace(experts=bank)))
    cfg = SimpleNamespace(hidden=2560, experts=512, top_k=10, moe_width=160, shared_width=80, execution_dtype="float16")
    return SimpleNamespace(cfg=cfg, layers=layers), partition


def owner():
    return OneCatOwner(OneCatGU("tensorfold-fp32-epilogue", 4_056_689_664, 1 << 30), "cpu")


def test_forecast_exact_census_before_any_plane_allocation():
    proto, _ = prototype(7)
    prepared = owner()
    result = prepared.forecast(proto, weights.STAGING_BYTES)
    assert result["additional_device_bytes"] == 4_056_689_664
    assert result["max_host_chunk_payload_bytes_per_rank"] == 39_168_000
    assert result["format_banks"] == {"18": 17, "21": 10, "22": 20, "23": 1}
    assert result["global_host_credit_bytes"] == 1 << 30
    assert len(prepared.sinks) == 96 and all(s.codes is None and s.scales is None for s in prepared.sinks)
    assert prepared.gu is None and not prepared.started and not prepared.complete
    with pytest.raises(RuntimeError, match="incomplete"):
        prepared.profile()


@pytest.mark.parametrize("credit", ["device", "host"])
def test_underfunded_header_fails_before_allocation(credit):
    proto, _ = prototype()
    spec = OneCatGU("onecat-half-epilogue", 1 if credit == "device" else 4_056_689_664,
                   1 if credit == "host" else 1 << 30)
    prepared = OneCatOwner(spec, "cpu")
    with pytest.raises(MemoryError):
        prepared.forecast(proto, weights.STAGING_BYTES)
    assert prepared.sinks is None and not prepared.started


@pytest.mark.parametrize("change", ["kind", "rows", "partition", "dtype"])
def test_nonmatching_model_refused_before_gpu_retention(change):
    proto, _ = prototype()
    bank = proto.layers[1].moe.experts
    if change == "kind":
        bank.up.source.ggml_type = 23
    elif change == "rows":
        proto.layers[1].moe.experts = replace(bank, gate=replace(bank.gate, row_ranges=((0, 40960),)))
    elif change == "partition":
        proto.layers[1].moe.experts = replace(bank, partition=ExpertPartition(8, 1, 512, 640, 4))
    else:
        proto.cfg.execution_dtype = "bfloat16"
    prepared = owner()
    with pytest.raises(ValueError):
        prepared.forecast(proto, weights.STAGING_BYTES)
    assert prepared.sinks is None and not prepared.started


@pytest.mark.parametrize("method,device,host", [(True, 1, 1), ("invalid", 1, 1),
    ("onecat-half-epilogue", True, 1), ("onecat-half-epilogue", 1, 0)])
def test_selector_is_closed_and_requires_explicit_credits(method, device, host):
    with pytest.raises(ValueError):
        OneCatGU(method, device, host)


def test_existing_loader_sink_rounds_whole_experts_and_preserves_raw_bytes():
    kind, row_bytes = 22, 820
    source = SimpleNamespace(path=Path("/read-lease.gguf"), offset=0, ggml_type=kind,
                             row_bytes=row_bytes, rows=40960)
    plan = weights.MatrixPlan(source, ((0, 40960),), ((0, 2560),), True)

    class Reader:
        def __init__(self):
            self.rows = []

        def read(self, path, offset, size, device):
            assert path == source.path and device == "cpu"
            first, count = offset // row_bytes, size // row_bytes
            self.rows.append((first, count))
            raw = np.zeros((count, row_bytes), np.uint8)
            ids = np.arange(first, first + count, dtype=np.uint16)
            raw[:, :2] = ids.astype("<u2").view(np.uint8).reshape(count, 2)
            return torch.from_numpy(raw).reshape(-1)

    sink = _Sink(_key(plan), kind, 40800, torch.device("cpu"))

    class Cold:
        def sink(self, actual):
            assert actual is plan
            sink.start()
            return sink

    # The typed owner journals and starts its sink before returning the lease.
    direct = Reader()
    expected = weights._packed(plan, direct, "cpu")
    through = Reader()
    actual = weights._packed(plan, through, "cpu", _onecat_owner=Cold())
    assert direct.rows == [(0, 40920), (40920, 40)]
    assert through.rows == [(0, 40800), (40800, 160)]
    assert torch.equal(actual.data, expected.data)
    assert sink.written == 40960 and sink.receipt["experts"] == 256


def test_decode_dispatch_and_original_grouped4_prefill_fallback(monkeypatch):
    monkeypatch.setattr(owned_experts, "_GROUPED_DOWN", "grouped-v1")
    _, partition = prototype()
    spec = OneCatGU("onecat-half-epilogue", 4_056_689_664, 1 << 30)
    plan = owned_experts.Plan(partition, 2560, 11, 32, 4096, False, _gate_up=spec)
    assert plan._onecat and plan._reference_gate_up.rows == 4
    calls = []
    bank = SimpleNamespace(_owned_expert_plan=plan, implementation=owned_experts.POLICY,
                           _owned_expert_role="onecat-target")
    bank._onecat_provider = SimpleNamespace(run_slots=lambda *a, **k: calls.append(("candidate", k)))
    bank.forward_partitioned_native_candidate = lambda *a, **k: calls.append(("original", k))
    for prefill, count in ((False, 24), (True, 24), (True, 64)):
        x = torch.empty((count, 2560), dtype=torch.float16)
        picks = torch.empty((count, 11), dtype=torch.int32)
        out = torch.empty((count, 11, 2560), dtype=torch.float32)
        local = torch.empty((count, 11), dtype=torch.int64)
        act = torch.empty((count, 11, 160), dtype=torch.float16)
        grouping = torch.empty(100000, dtype=torch.int32)
        arena = owned_experts.Arena(plan, local, act, out, grouping, prefill=prefill)
        arena.execute(bank, x, picks, out)
    assert [name for name, _ in calls] == ["candidate", "original", "original"]
    assert all(kwargs["_gate_up"].rows == 4 for name, kwargs in calls if name == "original")


def test_mtp4_verification_tiles_without_losing_rows_and_mtp_keeps_original_provider(monkeypatch):
    monkeypatch.setattr(owned_experts, "_GROUPED_DOWN", "grouped-v1")
    _, partition = prototype()
    spec = OneCatGU("tensorfold-fp32-epilogue", 117_976_064, 1 << 20)
    # Twenty-four MTP4 requests need120 target verification rows, not32.
    plan = owned_experts.Plan(partition, 2560, 11, 120, 4096, True, _gate_up=spec)
    calls = []
    x = torch.arange(120, dtype=torch.float16)[:, None].expand(120, 2560).contiguous()
    picks = torch.zeros((120, 11), dtype=torch.int64)
    output = torch.full((120, 11, 2560), float('nan'), dtype=torch.float32)
    arena = owned_experts.Arena(plan, torch.empty((120, 11), dtype=torch.int64),
        torch.empty((120, 11, 160), dtype=torch.float16), output,
        torch.empty(100000, dtype=torch.int32))

    def candidate(bank, values, selected, **kwargs):
        calls.append(values[:, 0].tolist())
        kwargs['out'].copy_(values[:, None, :])

    bank = SimpleNamespace(_owned_expert_plan=plan, implementation=owned_experts.POLICY,
        _owned_expert_role="onecat-target", _onecat_provider=SimpleNamespace(run_slots=candidate))
    assert arena.execute(bank, x, picks, output) is output
    assert [len(rows) for rows in calls] == [32, 32, 32, 24]
    assert [row for panel in calls for row in panel] == list(range(120))
    assert torch.equal(output[:, 0, :], x.float())
    assert plan.resources(120)['mtp_expert_implementation'] == 'tensorfold-fp32'
    calls.clear()
    bank._owned_expert_role = 'tensorfold-fp32-mtp'

    def original(values, selected, **kwargs):
        calls.append((values.shape[0], kwargs['_gate_up'].rows))
        return kwargs['out']

    bank.forward_partitioned_native_candidate = original
    assert arena.execute(bank, x, picks, output) is output
    assert calls == [(120, 4)]


@pytest.mark.parametrize('prefill_method', ['tensorfold', 'candidate'])
def test_q8_panel_scratch_is_paid_once_per_selected_role_and_never_for_mtp(monkeypatch, prefill_method):
    monkeypatch.setattr(owned_experts, "_GROUPED_DOWN", "grouped-v1")
    _, partition = prototype()
    spec = OneCatGU("tensorfold", 1, 1, down_method="onecat-q8-slots", prefill_method=prefill_method)
    plan = owned_experts.Plan(partition, 2560, 11, 120, 64, True, _gate_up=spec)

    def buffers(rows, role):
        b = SimpleNamespace(rows=rows, prefill=role == 'prefill', owner_role=role,
            moe=SimpleNamespace(act=torch.empty((rows, 11, 160), dtype=torch.float16),
                y=torch.empty((rows, 11, 2560), dtype=torch.float32),
                pick=torch.empty((rows, 11), dtype=torch.int64)))
        b.owned_experts = plan.arena(b)
        return b

    target, mtp, prompt = buffers(120, 'decode'), buffers(120, 'mtp'), buffers(64, 'prefill')
    assert target.owned_experts.q8.shape == (32 * 11, 5, 36)
    assert target.owned_experts.status.shape == (32,)
    assert mtp.owned_experts.q8 is None and mtp.owned_experts.status is None
    if prefill_method == 'tensorfold':
        assert prompt.owned_experts.q8 is None and prompt.owned_experts.status is None
    else:
        assert prompt.owned_experts.q8.shape == (32 * 11, 5, 36)
        assert prompt.owned_experts.status.shape == (32,)
    owner = SimpleNamespace(spec=spec, lease=SimpleNamespace(enabled=True), profile=lambda: {})
    dec = SimpleNamespace(buf=target, mbuf=mtp, pbuf=prompt,
                          w=SimpleNamespace(meta={'onecat_experts': owner}))
    result = plan.reconcile(dec)
    expected = 63_488 * (1 + int(prefill_method == 'candidate'))
    assert result['additional_device_bytes'] == result['materialized_onecat_down_scratch_bytes'] == expected
    calls = []

    def candidate(bank, values, selected, **kwargs):
        calls.append(values.shape[0])
        assert kwargs['q8'].shape == (values.shape[0] * 11, 5, 36)
        kwargs['out'].zero_()

    bank = SimpleNamespace(_owned_expert_plan=plan, implementation=owned_experts.POLICY,
        _owned_expert_role='onecat-target', _onecat_provider=SimpleNamespace(
            run_slots=candidate, prepare_prefill_frame=lambda bank, picks: False),
        forward_partitioned_native_candidate=lambda *a, **k: calls.append('tensorfold-prefill'))
    prompt.owned_experts.execute(bank, torch.empty((64, 2560), dtype=torch.float16),
                                prompt.moe.pick, prompt.moe.y)
    assert calls == ([32, 32] if prefill_method == 'candidate' else ['tensorfold-prefill'])


def test_prepare_failure_keeps_authentic_primary_through_transport_and_notify(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import execution_identity

    prepared = owner()
    primary = ValueError("native preparation primary")
    transport = MemoryError("phase publication failure")
    notification = MemoryError("notification failure")

    def operation():
        raise primary

    def phase(engine, channel, stage, callback, **kwargs):
        try:
            callback()
        except BaseException:
            raise transport

    def fatal(*args):
        raise notification

    monkeypatch.setattr(execution_identity, "_phase", phase)
    monkeypatch.setattr(execution_identity, "_fatal", fatal)
    with pytest.raises(ValueError) as caught:
        prepared.prepare_cohort(SimpleNamespace(), "owned-operator", operation)
    assert caught.value is primary
    assert prepared._status == [transport, notification]


def test_failed_retirement_retains_fixed_owners_and_disables_execution(monkeypatch):
    prepared = owner()
    prepared.started = True
    prepared.complete = True
    prepared.lease.enabled = True
    retained = torch.ones(5)
    prepared.tables["iq2"] = retained
    primary = RuntimeError("device fence failed")

    def fail(*args):
        raise primary

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError) as caught:
        prepared.close()
    assert caught.value is primary and not prepared.lease.closed and not prepared.lease.enabled
    assert prepared.tables["iq2"] is retained
    assert prepared.lease.retiring and prepared.retirement_error is primary
    with pytest.raises(RuntimeError) as re_adoption:
        prepared.adopt(prepared.spec)
    assert re_adoption.value is primary
    assert not prepared.lease.enabled and prepared.tables["iq2"] is retained


def test_retirement_journal_refuses_adoption_before_close_side_effects():
    prepared = owner()
    prepared.complete = True
    prepared.lease.retiring = True
    with pytest.raises(RuntimeError, match="retiring"):
        prepared.adopt(prepared.spec)
    assert not prepared.lease.enabled


@pytest.mark.parametrize("reader_failure,table_failure,transport_failure", [
    (False, False, False), (True, False, False), (False, True, False),
    (True, True, False), (True, True, True),
])
def test_actual_loader_failure_independently_retires_and_retains_owners(
        monkeypatch, reader_failure, table_failure, transport_failure):
    import gc
    import traceback
    import weakref

    primary = ValueError("packing primary")
    prior = RuntimeError("original native cause")
    primary.__cause__ = prior
    publication = MemoryError("owner publication exhausted")
    read_error = RuntimeError("Reader retirement failed")
    table_error = RuntimeError("PLE retirement failed")
    transport_error = MemoryError("neutral transport frame allocation exhausted")
    events, references = [], []

    class Reader:
        def __init__(self):
            references.append(weakref.ref(self))

        def close(self):
            events.append("reader-close")
            if reader_failure:
                raise read_error

    class Tables:
        def __init__(self):
            references.append(weakref.ref(self))

        def close(self, retire):
            events.append("tables-close")
            if table_failure:
                raise table_error
            retire()

    def retain(error, label, value):
        events.append("retain")
        try:
            raise publication
        except MemoryError as failure:
            raise error from failure

    def materializer(*args, **kwargs):
        def materialize(value):
            raise primary
        return materialize

    rope = SimpleNamespace(inverse_frequencies=lambda framework: torch.ones(1), metadata=lambda: {})
    proto = SimpleNamespace(cfg=SimpleNamespace(rope=rope), embed=None, layers=[object()], meta={})
    monkeypatch.setattr(weights, "model_plan", lambda *a, **k: (proto, None))
    monkeypatch.setattr(weights, "Reader", Reader)
    monkeypatch.setattr(weights, "PLETables", Tables)
    monkeypatch.setattr(weights, "materializer", materializer)
    monkeypatch.setattr(weights, "_retain_failure", retain)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: events.append("device-fence"))
    if transport_failure:
        def failed_transport(*args):
            raise transport_error
        monkeypatch.setattr(weights, "raise_failures", failed_transport)
    with pytest.raises(ValueError) as caught:
        weights.load_gguf("unused", device="cpu", mtp=False)
    assert caught.value is primary
    assert events[:3] == ["retain", "reader-close", "tables-close"]
    # Include the preallocated native-cause journal in the diagnostic graph;
    # transport/retention exhaustion must not lose prior cause or cleanup errors.
    pending, seen = [primary], []
    while pending:
        error = pending.pop()
        if error is None or any(error is previous for previous in seen):
            continue
        seen.append(error)
        pending.extend((error.__cause__, error.__context__))
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
        namespace = BaseException.__dict__["__dict__"].__get__(error)
        pending.extend(namespace.get("cleanup_status", ()))
        pending.extend(namespace.get(key) for key in ("original_cause", "original_context", "transport_error"))
    assert any(error is publication for error in seen)
    assert any(error is prior for error in seen)
    if reader_failure:
        assert any(error is read_error for error in seen)
    if table_failure:
        assert any(error is table_error for error in seen)
    if transport_failure:
        assert any(error is transport_error for error in seen)
    if reader_failure or table_failure:
        for error in seen:
            traceback.clear_frames(error.__traceback__)
        gc.collect()
        assert all(reference() is not None for reference in references)
