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
