"""Exact wire specialization and ownership; no protocol waits are removed."""
import gc
import json
from types import SimpleNamespace as NS
import weakref

import pytest

from tensorfold.engine.model_identity import canonical_runtime_json
from tensorfold.families.qwen4_exp.cuda import input_preparation as module
from tensorfold.families.qwen4_exp.cuda import execution_identity as protocol


@pytest.mark.parametrize("label", tuple(module._SUCCESS_PREFIXES))
@pytest.mark.parametrize("ordinal", [1, 2, 10**18, (1 << 63) - 1])
@pytest.mark.parametrize("value", [None, "0123456789abcdef" * 4])
def test_success_wire_matches_actual_general_serializer_exactly(label, ordinal, value):
    outcome = {"ok": True, "value": value}
    record = {"version": 1, "ordinal": ordinal, "label": label, "outcome": outcome}
    expected = canonical_runtime_json(record, max_bytes=8192)
    actual = module._success_envelope(ordinal, label, outcome, 8192)
    assert actual == expected and actual.encode() == expected.encode()
    assert module._success_envelope(ordinal, label, outcome, len(expected)) == expected
    with pytest.raises(ValueError, match="byte bound"):
        module._success_envelope(ordinal, label, outcome, len(expected) - 1)


@pytest.mark.parametrize("ordinal,label,outcome", [
    (True, "decode-window", {"ok": True, "value": None}),
    (0, "decode-window", {"ok": True, "value": None}),
    (1, "other-supported-label", {"ok": True, "value": None}),
    (1, "decode-window", {"ok": 1, "value": None}),
    (1, "decode-window", {"ok": True, "value": []}),
    (1, "decode-window", {"ok": True, "value": "F" * 64}),
    (1, "decode-window", {"ok": True, "value": None, "extra": 1}),
    (1, "decode-window", {"ok": False, "error": "bad", "invalidates_binding": False}),
])
def test_other_schemas_keep_general_encoder(ordinal, label, outcome):
    assert module._success_envelope(ordinal, label, outcome, 8192) is None


class Store:
    """Single-rank driver supplies immutable copies from the ordered eight slots."""
    def __init__(self):
        self.values, self.calls, self.adds, self.transform = {}, [], 0, None
    def set(self, key, value):
        self.calls.append(("set", key))
        self.values[key] = value
    def add(self, key, value):
        self.calls.append(("add", key, value))
        self.adds += 1
        return self.adds * 8
    def wait(self, keys, timeout):
        self.calls.append(("wait", tuple(keys)))
        assert all(key in self.values for key in keys)
    def delete_key(self, key):
        self.calls.append(("delete", key))
        return self.values.pop(key, None) is not None
    def multi_get(self, keys):
        self.calls.append(("multi_get", tuple(keys)))
        own = self.values[keys[0]].encode()
        values = [own for _ in keys]
        if self.transform is not None:
            values[1] = self.transform(own)
        return values


def context():
    store = Store()
    weights = NS(meta={"coordinate_profile_identity": "a" * 64, "rank": 0, "world": 8},
                 positions=object(), comm=NS(rank=0, world=8, store=store))
    return module.InputPreparationContext(weights), store


def fatal(monkeypatch):
    notifications = []
    def fail(engine, error):
        engine._execution_manifest_uncertain = True
        notifications.append(error)
    monkeypatch.setattr(protocol, "_fatal", fail)
    monkeypatch.setattr(module, "_fatal", fail)
    return notifications


@pytest.mark.parametrize("contract", [None, "a" * 64])
def test_matching_success_validates_fields_and_projects_only_immutable_values(monkeypatch, contract):
    ctx, store = context()
    fields = []
    ctx.cohort.store_fields = NS(value=lambda value: fields.append(("value", value)),
                                 received=lambda value: fields.append(("received", value)))
    monkeypatch.setattr(module.json, "loads", lambda *_: pytest.fail("matching canonical success parsed again"))
    box = ctx.cohort._phase_mailbox
    channel = "channel"
    box.begin(1, "decode-window", channel)
    values = protocol._phase(ctx.cohort, channel, "prepare", lambda: contract, may_refuse=False)
    assert values == [contract] * 8 and all(value is None or type(value) is str for value in values)
    assert [kind for kind, _ in fields] == ["value"] + ["received"] * 8
    box.retire(channel, ("prepare",))
    assert box.active is None
    assert [row[0] for row in store.calls] == [
        "set", "add", "set", "wait", "multi_get", "add", "set", "wait", "add", "set", "wait",
        "delete", "delete", "delete", "delete"]


@pytest.mark.parametrize("mode,accepted", [
    ("noncanonical", True), ("duplicate", True), ("refusal", False), ("stale", False),
    ("extra", False), ("malformed", False), ("wrong-contract", False), ("wrong-outcome", False),
])
def test_nonmatching_packets_retain_old_parser_and_failure_contract(monkeypatch, mode, accepted):
    ctx, store = context()
    loads, parsed = json.loads, []
    def transform(raw):
        if mode == "malformed":
            return b"["
        if mode == "duplicate":
            return raw[:-1] + b',"version":1}'  # The existing parser accepts the final duplicate value.
        row = loads(raw)
        if mode == "stale":
            row["ordinal"] += 1
        elif mode == "extra":
            row["extra"] = True
        elif mode == "refusal":
            row["outcome"] = {"ok": False, "error": "bad", "invalidates_binding": False}
        elif mode == "wrong-contract":
            row["outcome"]["value"] = "b" * 64
        elif mode == "wrong-outcome":
            row["outcome"]["extra"] = True
        return json.dumps(row).encode()
    store.transform = transform
    def parse(value):
        parsed.append(value)
        return loads(value)
    monkeypatch.setattr(module.json, "loads", parse)
    notifications = fatal(monkeypatch)
    if accepted:
        result = object()
        assert ctx.prepare("decode-window", lambda: result) is result
        assert not notifications
    else:
        with pytest.raises((RuntimeError, ValueError)):
            ctx.prepare("decode-window", lambda: object())
        if mode != "refusal":
            assert notifications
    assert len(parsed) == 1


def test_result_cell_preserves_none_and_failure_traceback_lifetime(monkeypatch):
    ctx, _store = context()
    assert ctx.prepare("decode-window", lambda: None) is None
    ctx, _store = context()
    fatal(monkeypatch)
    error = OSError("retirement failure after completed operation")
    def retire(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(module, "_retire_store", retire)
    refs = []
    class Result:
        pass
    def operation():
        result = Result()
        refs.append(weakref.ref(result))
        return result
    with pytest.raises(OSError) as caught:
        ctx.prepare("decode-window", operation)
    assert caught.value is error and ctx._active is None
    gc.collect()
    assert refs[0]() is not None  # The failure's operation frame owns its result until error retirement.
    error.__traceback__ = None
    caught.value.__traceback__ = None
    del caught
    gc.collect()
    assert refs[0]() is None


def test_generic_labels_still_use_general_parser(monkeypatch):
    ctx, _store = context()
    loads, parsed = json.loads, []
    def parse(value):
        parsed.append(value)
        return loads(value)
    monkeypatch.setattr(module.json, "loads", parse)
    result = object()
    assert ctx.prepare("other-supported-label", lambda: result) is result
    assert len(parsed) == 8


def test_nonexact_key_types_cannot_bypass_general_json_validation():
    class Key(str):
        pass
    outcome = {Key("ok"): True, "value": None}
    assert module._success_envelope(1, "decode-window", outcome, 8192) is None
    with pytest.raises(ValueError, match="keys must be strings"):
        canonical_runtime_json({"ordinal": 1, "label": "decode-window", "outcome": outcome, "version": 1})


def test_real_eight_rank_tcpstore_fast_success_keeps_all_waits_and_key_retirement(monkeypatch):
    torch = pytest.importorskip("torch")
    from concurrent.futures import ThreadPoolExecutor
    from datetime import timedelta
    server = torch.distributed.TCPStore("127.0.0.1", 0, 8, True, timedelta(seconds=10),
                                        wait_for_workers=False, use_libuv=False)
    contexts = [module.InputPreparationContext(NS(positions=object(),
        comm=NS(rank=rank, world=8, store=server.clone()),
        meta={"rank": rank, "world": 8, "coordinate_profile_identity": "e" * 64})) for rank in range(8)]
    baseline = server.num_keys()
    monkeypatch.setattr(module.json, "loads", lambda *_: pytest.fail("canonical success reparsed"))
    for label, contract in [("decode-window", None), ("mtp-refresh-window", "a" * 64), ("mtp-proposal-window", None)]:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda pair: pair[1].prepare(label, lambda: pair[0], contract=contract), enumerate(contexts)))
        assert results == list(range(8))
        assert server.num_keys() == baseline + 8
    assert all(ctx.ordinal == 3 and ctx._active is None for ctx in contexts)
    for ctx in contexts:
        ctx.close()
    assert server.num_keys() == baseline
