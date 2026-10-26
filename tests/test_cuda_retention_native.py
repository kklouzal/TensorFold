"""Opt-in native CPU controls using the actual prepared SC117 frontend domain."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")

import torch

from tensorfold.cuda import server
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.server.api_records import ApiRecordStore
from tensorfold.server.errors import RequestError
from tensorfold.server.token_routes import count_chat
from tensorfold.token_metadata import load_tokenizer, tokenizer_vocab_size
from tests.test_api_records import limits
from tests.test_cuda_retention_resources import catalog


@pytest.fixture(scope="module")
def target():
    pytest.importorskip("xgrammar")
    path = os.environ.get("TENSORFOLD_TEST_TARGET_FRONTEND")
    if path is None:
        pytest.skip("actual prepared SC117 frontend is an explicit qualification input")
    directory = Path(path)
    assert hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest() == "7b6b3e0e8caa705198c91d4925d545bb4067e72aa46ada04c6e3bb71ae498919"
    tok = load_tokenizer(directory)
    assert tokenizer_vocab_size(tok) == 248320 and tok.get_vocab_size() == 248077
    assert len(tok.control_token_ids) == 27 and len(tok.unused_token_ids) == 243
    return directory, tok


class NativeEngine:
    limit = 4096

    def __init__(self, tok):
        self.eos = (tok.token_to_id("<|im_end|>"),)
        self.vocab = tokenizer_vocab_size(tok)
        self.answer = tok.encode("a", add_special_tokens=False).ids
        self.constraints = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
        assert constraint is not None
        self.constraints.append(constraint)
        for desired in self.answer[:max_tokens]:
            scores = torch.zeros(1, self.vocab)
            scores[0, desired] = 1
            constraint.mask(scores)
            token = int(scores.argmax())
            assert token == desired
            constraint.advance([token])
            if on_tokens([token]):
                break
        constraint.retire_device_staging()
        return {}


def app_for_target(target, monkeypatch, policy=None):
    directory, tok = target
    engine = NativeEngine(tok)
    if policy is not None:
        engine.retention_catalog = catalog(policy)
        engine.retention_catalog.states[0].limit = engine.limit
        engine.scheduler = Scheduler(NS(), max_streams=1, admission_policy="fifo", max_waiting_input_bytes=1 << 20)
        # CPU source test observes the actual projection; it cannot grant a
        # process/V100 memory entitlement on this host.
        monkeypatch.setattr("tensorfold.cuda.capacity.admit_host_commitment",
                            lambda *_args, **_kwargs: {"release_qualified": False})
    return server.App(engine, directory, "native-target", max_tokens=1, context_window=4096,
                      max_pending_requests=2 if policy is not None else None,
                      max_http_connections=2 if policy is not None else None), engine


def test_actual_full_domain_native_compile_mask_and_cpu_staging_retirement(target, monkeypatch):
    app, engine = app_for_target(target, monkeypatch)
    body = {"messages": [{"role": "user", "content": "a"}], "guided_choice": ["a"], "max_tokens": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    prepared = app.prepare(body, True)
    grammars = app._grammars()
    assert grammars.vocab_size == 248320
    spec, compiled = prepared.grammar
    assert spec.kind == "choice"
    constraint = grammars.constraint(compiled, spec=spec)
    scores = torch.zeros(1, 248320)
    constraint.mask(scores)
    unused = torch.tensor(sorted(app.tok.unused_token_ids))
    assert torch.isneginf(scores[0, unused]).all()
    constraint.retire_device_staging()
    emitted = []
    def emit(delta):
        emitted.append(delta)
        return True
    result = app.run(body, True, emit, prepared=prepared)
    assert result["finish"] in ("stop", "length") and engine.constraints
    assert "a" in result["content"]
    prepared.close()


def test_actual_native_count_compile_then_retained_run_refuses_without_matcher(target, tmp_path, monkeypatch):
    app, engine = app_for_target(target, monkeypatch, RetentionPolicy("gpu", 1, 10))
    body = {"messages": [{"role": "user", "content": "a"}], "guided_choice": ["a"], "max_tokens": 1}
    count = count_chat(app, body)
    assert count > 0
    prepared = app.prepare_for_count(body, True)
    assert app._grammars().vocab_size == 248320 and prepared.grammar is not None
    store = ApiRecordStore(tmp_path / "logical.sqlite", limits=limits())
    try:
        with store.response_operation("trusted", "resp_native", previous_response_id=None, conversation_id=None,
            added=[{"role": "user", "content": "a"}], stored=True) as operation, operation.execution_scope():
            with pytest.raises(RequestError, match="native matcher CPU-memory envelope"):
                app.run(body, True, lambda _delta: None, prepared=prepared)
        assert engine.constraints == []
    finally:
        prepared.close()
        store.close()
