"""Optional cohort field boundaries preserve old callers and validate before IO."""

import json
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import execution_identity
from tensorfold.families.qwen4_exp.cuda.retention_resources import StoreFieldGuard


class Store:
    def __init__(self):
        self.values, self.calls = {}, []

    def set(self, key, value):
        self.calls.append(("set", key))
        self.values[key] = value.encode()

    def wait(self, keys, _timeout):
        self.calls.append(("wait", tuple(keys)))
        assert all(key in self.values for key in keys)

    def get(self, key):
        self.calls.append(("get", key))
        return self.values[key]

    def delete_key(self, key):
        self.calls.append(("delete", key))
        return self.values.pop(key, None) is not None


def owner(fields=None):
    failures = []
    result = SimpleNamespace(comm=SimpleNamespace(store=Store()), rank=0, tp=1,
        multi=SimpleNamespace(poison_control=failures.append))
    if fields is not None:
        result.store_fields = fields
    return result, failures


def test_omitted_guard_preserves_existing_long_key_and_retirement_protocol():
    value, failures = owner()
    channel = "x" * 8192  # Nonretention Store may have a different key contract.
    assert execution_identity._phase(value, channel, "s", lambda: {"value": 7}) == [{"value": 7}]
    execution_identity._retire_store(value, channel, stages=("s",))
    assert value.comm.store.values == {} and failures == []
    assert [kind for kind, _key in value.comm.store.calls] == ["set", "wait", "get", "set", "wait", "set", "wait",
                                                            "delete", "delete", "delete"]


def test_immutable_phase_keys_are_validated_once_before_set_wait_get():
    class Spy:
        def __init__(self):
            self.keys = []
            self.actual = StoreFieldGuard(1024)
        def key(self, key):
            self.keys.append(key)
            self.actual.key(key)
        def value(self, value):
            self.actual.value(value)
        def received(self, value):
            self.actual.received(value)
    fields = Spy()
    value, failures = owner(fields)
    assert execution_identity._phase(value, "c", "s", lambda: True) == [True]
    assert fields.keys == ["c/phase/s/0"] and failures == []
    fields.keys.clear()
    execution_identity._retire_store(value, "c", stages=(stage for stage in ("s",)))
    assert fields.keys == ["c/phase/s/0", "c/ready/retirement/0", "c/retired/0"]
    assert value.comm.store.values == {}


@pytest.mark.parametrize("surface", ["phase", "retire"])
def test_invalid_keys_fail_before_callback_or_first_native_operation(surface):
    value, failures = owner(StoreFieldGuard(1024))
    entered = []
    with pytest.raises(ValueError, match="protocol limit"):
        if surface == "phase":
            execution_identity._phase(value, "x" * 8192, "s", lambda: entered.append(True))
        else:
            execution_identity._retire_store(value, "c", stages=("s" * 8192,))
    assert not entered and value.comm.store.calls == [] and len(failures) == 1


def test_encoded_value_limit_is_checked_before_publication():
    value, failures = owner(StoreFieldGuard(24))
    with pytest.raises(ValueError, match="native field"):
        execution_identity._phase(value, "c", "s", lambda: "\u00e9" * 20)
    assert value.comm.store.calls == [] and len(failures) == 1


def test_received_bytes_are_bounded_before_json_expansion(monkeypatch):
    value, failures = owner(StoreFieldGuard(64))
    value.comm.store.get = lambda _key: b" " * 65
    monkeypatch.setattr(execution_identity.json, "loads", lambda *_args: pytest.fail("oversized input was decoded"))
    with pytest.raises(ValueError, match="received cohort"):
        execution_identity._phase(value, "c", "s", lambda: True)
    assert len(failures) == 1


def test_native_field_measures_utf8_bytes_not_unicode_character_count():
    guard = StoreFieldGuard(8)
    guard.value("\u00e9" * 4)
    with pytest.raises(ValueError, match="native field"):
        guard.value("\u00e9" * 5)
    guard.received(json.dumps(0).encode())
