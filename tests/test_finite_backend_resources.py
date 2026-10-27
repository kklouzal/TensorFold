"""Finite caller admission retains real model/input/output and physical guards."""
from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.server.request_limits import RequestLimit
from tensorfold.families.qwen4_exp.cuda.retention_resources import (
    FiniteBackendCaller, finalize_backend_resources,
)
from tests.test_cuda_retention_resources import catalog, selected_tokenizer


def owners():
    selected = catalog(None)
    decoder = selected._cohort.multi
    decoder.states = selected.states
    decoder.w.cfg.vocab = selected_tokenizer().get_vocab_size()
    scheduler = Scheduler(decoder, max_streams=1, admission_policy="fifo", max_waiting_input_bytes=16384)
    caller = FiniteBackendCaller(max_source_bytes=512, max_input_tokens=32,
        max_output_tokens=16, retained_results=2, max_receipt_bytes=4096)
    return selected, scheduler, RequestLimit(1), caller


def test_finite_caller_has_distinct_real_admitted_role_without_http_zero_limits(monkeypatch):
    from tensorfold.cuda import capacity
    selected, scheduler, limit, caller = owners()
    calls = []
    def admit(total, *, materialized_owned_bytes):
        calls.append((total, materialized_owned_bytes))
        return {"real_boundary_reached": True}
    monkeypatch.setattr(capacity, "admit_host_commitment", admit)
    record = finalize_backend_resources(selected, tokenizer=selected_tokenizer(),
        scheduler=scheduler, request_limit=limit, caller=caller)
    assert calls == [(record["global_host_commitment_bytes"], record["materialized_owned_bytes"])]
    assert record["authority"] == {"real_boundary_reached": True}
    assert record["context_tokens"] == 128 and selected.states[0].limit == 131
    assert record["caller"] == caller.__dict__
    assert "max_http_connections" not in record and "logical_api" not in record
    assert record["global_waiting_input_bytes"] == 16384
    assert all(amount > 0 for amount in record["caller_payload"].values())
    assert scheduler._request_limit is None and limit.used == 0
    assert selected.host_resource_envelope is record
    scheduler.configure_requests(limit)
    assert scheduler._request_limit is limit


@pytest.mark.parametrize("kind", ["already-submitted", "other-owner", "context-overflow", "multiple-callers"])
def test_finite_caller_refuses_wrong_lifetime_or_geometry_before_physical_admission(monkeypatch, kind):
    from tensorfold.cuda import capacity
    selected, scheduler, limit, caller = owners()
    if kind == "already-submitted":
        scheduler._submitted = True
    elif kind == "other-owner":
        scheduler._request_limit = RequestLimit(1)
    elif kind == "multiple-callers":
        limit = RequestLimit(2)
    else:
        caller = replace(caller, max_output_tokens=128)
    monkeypatch.setattr(capacity, "admit_host_commitment",
                        lambda *_a, **_k: pytest.fail("invalid caller reached physical admission"))
    with pytest.raises(ValueError):
        finalize_backend_resources(selected, tokenizer=selected_tokenizer(),
            scheduler=scheduler, request_limit=limit, caller=caller)
    assert not hasattr(selected, "host_resource_envelope")


def test_finite_caller_keeps_the_real_physical_refusal_and_unpublished_catalog(monkeypatch):
    from tensorfold.cuda import capacity
    selected, scheduler, limit, caller = owners()
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 8 << 30, "MemAvailable": 2 << 30})
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda **_kwargs: None)
    with pytest.raises(ValueError, match="remaining host commitment needs"):
        finalize_backend_resources(selected, tokenizer=selected_tokenizer(),
            scheduler=scheduler, request_limit=limit, caller=caller)
    assert not hasattr(selected, "host_resource_envelope") and not scheduler._submitted


@pytest.mark.parametrize("value", [0, -1, True, None])
def test_finite_contract_requires_actual_positive_caller_bounds(value):
    with pytest.raises(ValueError):
        FiniteBackendCaller(value, 32, 16, 1, 4096)


def test_json_runtime_authority_rejects_changed_captured_native_content(monkeypatch):
    from tensorfold.engine import model_identity
    from tensorfold.families.qwen4_exp.cuda.retention_resources import _json_payload_policy
    capture = model_identity.capture_model_identity
    def altered(*args, **kwargs):
        receipt = capture(*args, **kwargs)
        files = tuple(replace(item, sha256="0" * 64) if item.logical_name == "json-native" else item
                      for item in receipt.files)
        return SimpleNamespace(files=files, verify_unchanged=receipt.verify_unchanged,
                               model_prefix=receipt.model_prefix)
    monkeypatch.setattr(model_identity, "capture_model_identity", altered)
    with pytest.raises(ValueError, match="actual runtime/header content"):
        _json_payload_policy()


@pytest.mark.parametrize("count", [0, 1, 5, 6, 10, 11, 21, 22, 85, 86, 170, 171])
def test_selected_fresh_table_bound_covers_actual_unicode_and_general_tables(count):
    from tensorfold.families.qwen4_exp.cuda.retention_resources import _fresh_dict_payload
    strings = {str(n): None for n in range(count)}
    integers = {n: None for n in range(count)}
    assert _fresh_dict_payload(count, 16)[0] >= sys.getsizeof(strings)
    assert _fresh_dict_payload(count, 24)[0] >= sys.getsizeof(integers)


def test_selected_parser_has_separate_c_depth_codec_error_copy_and_complete_sum():
    from tensorfold.families.qwen4_exp.cuda.retention_resources import _json_parse_payload, _json_payload_policy
    policy = _json_payload_policy()
    encoded = 96 << 20
    projection = _json_parse_payload(encoded, policy)
    assert projection["c_depth_bound"] == 10000
    assert policy.recursion_limit == sys.getrecursionlimit()
    terms = projection["terms"]
    assert terms["raw_input"] == terms["unicode_error_input_copy"]
    assert terms["decoded_document"] >= 8 * encoded
    assert projection["total_bytes"] == sum(terms.values())


def concurrent_owners(count=2):
    selected, scheduler, _, caller = owners()
    selected.states.extend(SimpleNamespace(limit=131, logical_context=128) for _ in range(count - 1))
    scheduler.max_streams = count
    floor = selected._cohort.multi.input_reservation_bytes()
    scheduler.max_waiting_input_bytes = count * floor
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
    from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController
    decoder = selected._cohort.multi
    decoder.logical_context, decoder.full_resident = 128, True
    decoder.streams, decoder.filling, decoder._kept = {}, [], []
    decoder.controller = AdaptiveController(slots=count, row_budget=count * 3, proposal_limit=2, prefill_tokens=16)
    adapter = TensorParallelDecoder.__new__(TensorParallelDecoder)
    adapter.decoder, adapter.requests, adapter.admission_policy = decoder, {}, 'fifo'
    adapter.engine = SimpleNamespace(serving_profiles=None)
    adapter._finite_contract = None
    # Resource-unit seam only; the separate real ordered-peer controls prove
    # cohort activation/ACK/rejection and partial-install containment.
    adapter.configure_finite_caller = lambda contract: setattr(adapter, '_finite_contract', contract)
    scheduler.decoder = adapter
    caller = replace(caller, max_concurrent_requests=count, retained_results=count,
                     worker_stack_bytes=64 << 20, control_payload_bytes=16384)
    return selected, scheduler, RequestLimit(count), caller


def test_concurrent_caller_funds_each_input_native_result_stack_before_real_publication(monkeypatch):
    from tensorfold.cuda import capacity
    calls = []
    monkeypatch.setattr(capacity, 'admit_host_commitment',
                        lambda total, **kw: calls.append((total, kw)) or {'actual_boundary': True})
    selected, scheduler, limit, caller = concurrent_owners()
    record = finalize_backend_resources(selected, tokenizer=selected_tokenizer(),
        scheduler=scheduler, request_limit=limit, caller=caller)
    terms = record['caller_payload']
    assert record['max_pending_requests'] == record['generation_owner_bound']['owners'] == 2
    assert record['caller']['max_concurrent_requests'] == 2
    assert terms['worker_stacks'] == 2 * (64 << 20)
    assert terms['bounded_caller_control_payload'] == 16384
    assert terms['additional_native_decode_and_conversion'] == terms['one_native_decode_and_conversion']
    assert terms['prepared_source_strings_and_buffers'] == 4 * (80 + 5 * caller.max_source_bytes)
    assert len(calls) == 1 and calls[0][0] == record['global_host_commitment_bytes']
    assert scheduler._request_limit is None and limit.used == 0
    assert scheduler.decoder._finite_contract.record() == record['finite_protocol']['role']
    assert record['finite_protocol']['native_logical_context_unchanged'] == 128
    assert record['finite_protocol']['cpu_token_span'] == 54
    scheduler.configure_requests(limit)
    assert scheduler._request_limit is limit


@pytest.mark.parametrize('kind', ['slots', 'streams', 'input-credit', 'results', 'owner-mismatch'])
def test_concurrent_caller_rejects_unfunded_capacity_before_physical_admission(monkeypatch, kind):
    from tensorfold.cuda import capacity
    selected, scheduler, limit, caller = concurrent_owners()
    if kind == 'slots':
        selected.states.pop()
    elif kind == 'streams':
        scheduler.max_streams = 1
    elif kind == 'input-credit':
        scheduler.max_waiting_input_bytes -= 1
    elif kind == 'results':
        caller = replace(caller, retained_results=1)
    else:
        limit = RequestLimit(1)
    monkeypatch.setattr(capacity, 'admit_host_commitment',
                        lambda *_a, **_kw: pytest.fail('unfunded concurrency reached physical admission'))
    with pytest.raises(ValueError):
        finalize_backend_resources(selected, tokenizer=selected_tokenizer(),
            scheduler=scheduler, request_limit=limit, caller=caller)
    assert not hasattr(selected, 'host_resource_envelope')


def test_default_single_numeric_terms_have_no_new_stack_control_or_concurrent_cost(monkeypatch):
    from tensorfold.cuda import capacity
    from tensorfold.families.qwen4_exp.cuda.retention_resources import _append_list_payload, _python_layout, _json_payload_policy
    monkeypatch.setattr(capacity, 'admit_host_commitment', lambda *_a, **_kw: {})
    selected, scheduler, limit, caller = owners()
    record = finalize_backend_resources(selected, tokenizer=selected_tokenizer(),
        scheduler=scheduler, request_limit=limit, caller=caller)
    layout = _python_layout()
    integer = max(layout['integer'], sys.getsizeof(record['tokenizer']['vocab_size'] - 1))
    native = record['native_decode']
    assert record['caller_payload'] == {
        'prepared_source_strings_and_buffers': 2 * (80 + 5 * caller.max_source_bytes),
        'original_input_ids': _append_list_payload(caller.max_input_tokens) + caller.max_input_tokens * integer,
        'retained_result_ids': caller.retained_results * (
            _append_list_payload(caller.max_output_tokens) + caller.max_output_tokens * integer),
        'retained_decoded_text': caller.retained_results * (80 + 5 * native['decoded_utf8_bytes']),
        'one_native_decode_and_conversion': max(native['native_heap_requested_bytes'], native['result_to_python_payload_bytes']),
        'bounded_report_graph': caller.max_receipt_bytes * layout['dict_entry'],
        'bounded_report_encoding_and_wire': 5 * caller.max_receipt_bytes + _json_payload_policy().frame_work,
    }
    assert caller.max_concurrent_requests == 1 and caller.worker_stack_bytes == caller.control_payload_bytes == 0


@pytest.mark.parametrize('change', [
    {'max_concurrent_requests': 0}, {'max_concurrent_requests': True},
    {'max_concurrent_requests': 2}, {'max_concurrent_requests': 2, 'worker_stack_bytes': 1},
    {'worker_stack_bytes': -1}, {'control_payload_bytes': True},
])
def test_concurrent_contract_requires_explicit_positive_counts_and_owned_payload(change):
    with pytest.raises(ValueError):
        replace(owners()[3], **change)


def test_concurrent_envelope_is_published_only_after_physical_admission_and_role_ack(monkeypatch):
    from tensorfold.cuda import capacity
    selected, scheduler, limit, caller = concurrent_owners()
    events = []
    primary = RuntimeError('peer role installation failed')
    def admit(*_args, **_kwargs):
        events.append('physical-admission')
        return {'test_boundary': True}
    def fail_role(_contract):
        assert events == ['physical-admission']
        assert not hasattr(selected, 'host_resource_envelope') and not scheduler._request_limit
        events.append('role-failed')
        raise primary
    monkeypatch.setattr(capacity, 'admit_host_commitment', admit)
    scheduler.decoder.configure_finite_caller = fail_role
    with pytest.raises(RuntimeError) as caught:
        finalize_backend_resources(selected, tokenizer=selected_tokenizer(),
            scheduler=scheduler, request_limit=limit, caller=caller)
    assert caught.value is primary
    assert events == ['physical-admission', 'role-failed']
    assert not hasattr(selected, 'host_resource_envelope') and limit.used == 0


def test_concurrent_actual_physical_refusal_precedes_role_installation(monkeypatch):
    from tensorfold.cuda import capacity
    selected, scheduler, limit, caller = concurrent_owners()
    primary = ValueError('real physical refusal')
    def refuse(*_args, **_kwargs):
        raise primary
    monkeypatch.setattr(capacity, 'admit_host_commitment', refuse)
    scheduler.decoder.configure_finite_caller = lambda _: pytest.fail('unbacked role activation')
    with pytest.raises(ValueError) as caught:
        finalize_backend_resources(selected, tokenizer=selected_tokenizer(),
            scheduler=scheduler, request_limit=limit, caller=caller)
    assert caught.value is primary and not hasattr(selected, 'host_resource_envelope')


def test_v2_real_admission_funds_2048_outputs_and_preserves_24_native_windows(monkeypatch):
    from tensorfold.cuda import capacity
    from tools import finite_concurrency
    receipts=[]
    monkeypatch.setattr(capacity,"admit_host_commitment",
        lambda total,**kwargs:receipts.append((total,kwargs)) or {"real_boundary":True})
    results=[]
    for output in (512,2048):
        selected,scheduler,limit,caller=concurrent_owners(24)
        decoder=selected._cohort.multi
        decoder.logical_context,decoder.capacity=262144,262147
        for state in selected.states:
            state.logical_context,state.limit=262144,262147
        scheduler.max_waiting_input_bytes=24*decoder.input_reservation_bytes()
        scheduler.output_queue_tokens=output
        caller=replace(caller,max_input_tokens=4096,max_output_tokens=output,
            max_receipt_bytes=25*2424832,
            control_payload_bytes=finite_concurrency.control_payload_bytes(output)
                +finite_concurrency.screen_control_payload_bytes(output))
        record=finalize_backend_resources(selected,tokenizer=selected_tokenizer(),
            scheduler=scheduler,request_limit=limit,caller=caller)
        assert len(selected.states)==24 and all(state.limit==262147 for state in selected.states)
        assert record["context_tokens"]==record["finite_protocol"]["native_logical_context_unchanged"]==262144
        assert record["finite_protocol"]["role"]["outputs"]==output
        assert scheduler._request_limit is None and not limit.used
        results.append(record)
    assert len(receipts)==2 and all(total==row["global_host_commitment_bytes"] for (total,_kwargs),row in zip(receipts,results))
    old,new=results
    assert new["global_host_commitment_bytes"]>old["global_host_commitment_bytes"]
    assert new["caller_payload"]["retained_result_ids"]>old["caller_payload"]["retained_result_ids"]
    assert new["caller_payload"]["retained_decoded_text"]>old["caller_payload"]["retained_decoded_text"]
    assert new["caller_payload"]["bounded_caller_control_payload"]>old["caller_payload"]["bounded_caller_control_payload"]
    assert new["finite_protocol"]["cpu_token_span"]==6150
