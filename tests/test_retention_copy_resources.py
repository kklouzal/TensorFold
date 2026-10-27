"""Actual prepared Multi copy policy controls only its source-owned CPU roles."""
from types import SimpleNamespace as NS
import weakref

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from tensorfold.families.qwen4_exp.cuda import multi
from tensorfold.families.qwen4_exp.cuda.retained_catalog import make_retention_catalog
from tensorfold.families.qwen4_exp.cuda.retention_resources import (
    HostBaseline, continuation_payload_bytes, finalize_retention_resources,
)
from tests.test_retention_resources import tokenizer

pytest_plugins = ("test_state_retention",)
COPY_ROLES = {"copy.query", "copy.starts", "copy.best", "copy.context_slice"}


def test_disabled_copy_removes_only_its_index_and_four_list_roles():
    options = dict(context=128, vocab_size=248320, presentation=768, max_body_bytes=96 << 20)
    enabled = continuation_payload_bytes(**options, copy_enabled=True)
    disabled = continuation_payload_bytes(**options, copy_enabled=False)
    assert disabled["copy_index"] == 0 < enabled["copy_index"]
    assert set(enabled["token_list_roles"]) - set(disabled["token_list_roles"]) == COPY_ROLES
    assert tuple(name for name in enabled["token_list_roles"] if name not in COPY_ROLES) == disabled["token_list_roles"]
    assert disabled["continuation_lists"] < enabled["continuation_lists"]
    changed = {"continuation_lists", "copy_index", "token_list_roles", "list_partition", "project_payload_bytes"}
    assert all(disabled[name] == value for name, value in enabled.items() if name not in changed)
    assert enabled["project_payload_bytes"] - disabled["project_payload_bytes"] == (
        enabled["copy_index"] + enabled["continuation_lists"] - disabled["continuation_lists"])


@pytest.mark.parametrize("invalid", [None, 0, 1, "1"])
def test_copy_projection_requires_an_actual_boolean(invalid):
    with pytest.raises(ValueError, match="actual prepared decoder boolean"):
        continuation_payload_bytes(context=128, vocab_size=248320, presentation=768,
                                   max_body_bytes=96 << 20, copy_enabled=invalid)


@pytest.mark.parametrize("configured,dflash,expected", [(None, False, False), ("1", True, False), ("1", False, True)])
def test_actual_prepared_policy_and_validated_catalog_binding_freeze_copy_resources(
        materialized, monkeypatch, configured, dflash, expected):
    if configured is None:
        monkeypatch.delenv("TENSORFOLD_MTP_COPY", raising=False)
    else:
        monkeypatch.setenv("TENSORFOLD_MTP_COPY", configured)
    weights = NS(comm=None, meta={"rank": 0}, cfg=NS(eos=(), vocab=248320), layers=[],
                 dflash2=object() if dflash else None)
    decoder = multi.MultiDecoder.__new__(multi.MultiDecoder)
    boundary = RuntimeError("next buffer acquisition")
    def next_buffer(*_args, **_kwargs):
        raise boundary
    monkeypatch.setattr(multi, "Buffers", next_buffer)
    # Run the real constructor through policy preparation, stopping before
    # device/model resources. This does not qualify DFlash model execution.
    with pytest.raises(RuntimeError) as stopped:
        multi.MultiDecoder.__init__(decoder, weights, slots=1, capacity=materialized.limit,
                                   depth=0, prefill_rows=1)
    assert stopped.value is boundary and decoder.copy is expected
    catalog = make_retention_catalog(None, weights, [materialized], prefix_limit=0, recovery_versions=0)
    try:
        decoder.states = catalog.states
        catalog.bind_decoder(decoder)
        assert isinstance(catalog._cohort.multi, weakref.ProxyTypes)
        catalog.host_resources = HostBaseline(0, 0)
        admissions = []
        monkeypatch.setattr("tensorfold.cuda.capacity.admit_host_commitment",
            lambda total, **_kwargs: admissions.append(total) or {"release_qualified": False})
        # Environment changes after preparation cannot change actual policy.
        monkeypatch.setenv("TENSORFOLD_MTP_COPY", "0" if expected else "1")
        options = dict(tokenizer=tokenizer(), max_pending_requests=2,
                       max_waiting_input_bytes=4096, max_http_connections=2)
        receipt = finalize_retention_resources(catalog, **options)
        assert receipt["copy_enabled"] is expected
        assert bool(receipt["continuation"]["copy_index"]) is expected
        assert any(name in COPY_ROLES for name in receipt["continuation"]["token_list_roles"]) is expected
        assert len(admissions) == 1
        decoder.copy = not expected
        with pytest.raises(ValueError, match="immutable"):
            finalize_retention_resources(catalog, **options)
        assert len(admissions) == 1 and catalog.host_resource_envelope is receipt
    finally:
        catalog.close(timeout=3)


def test_finalization_refuses_an_unbound_decoder_before_host_admission(materialized, monkeypatch):
    catalog = make_retention_catalog(None, NS(comm=None, meta={"rank": 0}), [materialized],
                                     prefix_limit=0, recovery_versions=0)
    try:
        catalog.host_resources = HostBaseline(0, 0)
        monkeypatch.setattr("tensorfold.cuda.capacity.admit_host_commitment",
                            lambda *_args, **_kwargs: pytest.fail("unbound decoder reached host admission"))
        with pytest.raises(ValueError, match="actual bound decoder"):
            finalize_retention_resources(catalog, tokenizer=tokenizer(), max_pending_requests=2,
                max_waiting_input_bytes=4096, max_http_connections=2)
    finally:
        catalog.close(timeout=3)
