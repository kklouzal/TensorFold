"""Immutable RAM object publication, references and corruption; no device claims."""

import pytest

from tensorfold.families.qwen4_exp.cuda.ram_backing import RamObjects


IDENTITY = "a" * 64


def put(store, data):
    builder = store.begin(IDENTITY, len(data))
    builder.write(data[:2])
    builder.write(data[2:])
    return builder.finish()


def test_incomplete_ram_destination_never_replaces_a_published_object():
    store = RamObjects(16, max_objects=3, max_readers=2)
    key = put(store, b"abcd")
    partial = store.begin(IDENTITY, 4)
    partial.write(b"xy")
    with pytest.raises(ValueError, match="incomplete"):
        partial.finish()
    with store.acquire(key) as reader:
        assert reader.read(0, 4) == b"abcd"
    partial.close()
    store.release(key)
    assert store.record()["payload_bytes"] == 0


def test_referenced_or_read_objects_are_not_evicted_by_another_consumer():
    store = RamObjects(8, max_objects=2, max_readers=2)
    key = put(store, b"abcd")
    reader = store.acquire(key)
    store.release(key)
    assert store.record()["payload_bytes"] == 4 and reader.read(1, 2) == b"bc"
    with pytest.raises(RuntimeError, match="readers"):
        store.close()
    reader.close()
    store.close()
    assert store.record()["payload_bytes"] == 0


def test_immutable_content_is_reused_and_each_manifest_has_its_own_reference():
    store = RamObjects(8, max_objects=3, max_readers=2)
    first, second = put(store, b"abcd"), put(store, b"abcd")
    assert first == second and store.record()["payload_bytes"] == 4
    store.release(first)
    with store.acquire(second) as reader:
        assert reader.read(0, 4) == b"abcd"
    store.release(second)
    assert store.record()["objects"] == 0


def test_exhaustion_and_corruption_are_explicit_and_do_not_erase_source():
    store = RamObjects(4, max_objects=1, max_readers=1)
    key = put(store, b"abcd")
    with pytest.raises(MemoryError):
        store.begin(IDENTITY, 1)
    store._objects[key].data[0] ^= 1  # Controlled corruption; no production writable view is exposed.
    with pytest.raises(ValueError, match="checksum"):
        store.acquire(key)
    assert store.record()["objects"] == 1 and store.record()["readers"] == 0
    store.release(key)


def test_reader_constructor_failure_does_not_publish_or_leak_a_claim(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import ram_backing

    store = RamObjects(8, max_objects=2, max_readers=1)
    key = put(store, b"abcd")
    primary = MemoryError("injected reader allocation failure")

    def fail(*_args):
        raise primary

    with monkeypatch.context() as faults:
        faults.setattr(ram_backing, "RamReader", fail)
        with pytest.raises(MemoryError) as caught:
            store.acquire(key)
    assert caught.value is primary
    assert store.record()["readers"] == 0 and store._objects[key].readers == 0
    with store.acquire(key) as reader:
        assert reader.read(0, 4) == b"abcd"
    store.release(key)
    store.close()


def test_failed_verification_keeps_original_error_and_failed_close_owner(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import ram_backing
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired

    store = RamObjects(8, max_objects=2, max_readers=1)
    key = put(store, b"abcd")
    primary, cleanup = ValueError("checksum failure"), RuntimeError("close failure")
    def verify(_self):
        raise primary

    def close(_self):
        raise cleanup

    with monkeypatch.context() as faults:
        faults.setattr(ram_backing.RamReader, "verify", verify)
        faults.setattr(ram_backing.RamReader, "close", close)
        with pytest.raises(ValueError) as caught:
            store.acquire(key)
    assert caught.value is primary and _unretired(primary)
    assert store._failure is primary and isinstance(store._retained_reader, ram_backing.RamReader)
    assert store._retained_reader.key == key and store._readers == 1
    with pytest.raises(RuntimeError, match="poisoned") as unhealthy:
        store.check_health()
    assert unhealthy.value.__cause__ is primary


@pytest.mark.parametrize("existing", [False, True])
def test_finished_reference_return_cut_is_owned_or_rolled_back_before_caller_publication(monkeypatch, existing):
    from tensorfold.families.qwen4_exp.cuda import ram_backing

    store = RamObjects(16, max_objects=3, max_readers=1)
    old = put(store, b"abcd") if existing else None
    builder = store.begin(IDENTITY, 4)
    builder.write(b"abcd")
    original = ram_backing.RamBuilder.finish
    primary = MemoryError("interrupted finished-key return")

    def interrupted(value):
        original(value)
        raise primary

    journal = [None]
    with monkeypatch.context() as faults:
        faults.setattr(ram_backing.RamBuilder, "finish", interrupted)
        with pytest.raises(MemoryError) as caught:
            builder.finish_into(journal, 0)
    assert caught.value is primary and journal == [None]
    assert store.record()["payload_bytes"] == (4 if existing else 0)
    if old is not None:
        with store.acquire(old) as reader:
            assert reader.read(0, 4) == b"abcd"
        store.release(old)
    store.close()


def test_backing_allocation_failure_preserves_primary_and_unretired_builder(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import ram_backing
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired

    store = RamObjects(8, max_objects=2, max_readers=1)
    primary = MemoryError("backing allocation failure")
    original = ram_backing.RamBuilder.close

    def allocate(_size):
        raise primary

    def close(_owner):
        raise RuntimeError("builder close failure")

    with monkeypatch.context() as faults:
        faults.setattr(ram_backing, "bytearray", allocate, raising=False)
        faults.setattr(ram_backing.RamBuilder, "close", close)
        with pytest.raises(MemoryError) as caught:
            store.begin(IDENTITY, 4)
    assert caught.value is primary and _unretired(primary)
    for owner in primary.__dict__["_tensorfold_retained_owners"]:
        if isinstance(owner, ram_backing.RamBuilder):
            original(owner)
    store.close()


def test_reuse_collision_and_reader_close_failure_preserve_both_owned_outcomes(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import ram_backing
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired

    store = RamObjects(16, max_objects=3, max_readers=1)
    key = put(store, b"abcd")
    store._objects[key].identity = "b" * 64  # Actual metadata collision fault.
    builder = store.begin(IDENTITY, 4)
    builder.write(b"abcd")
    def fail(_owner):
        raise RuntimeError("reuse reader close failed")

    with monkeypatch.context() as faults:
        faults.setattr(ram_backing.RamReader, "close", fail)
        with pytest.raises(ValueError, match="collision") as caught:
            builder.finish()
    assert _unretired(caught.value)
    assert store._failure is caught.value and store._retained_reader.key == key
    assert store._readers == 1 and builder in store._builders
    with pytest.raises(RuntimeError, match="poisoned"):
        store.check_health()


def test_builder_constructor_failure_publishes_no_reservation(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import ram_backing

    store = RamObjects(8, max_objects=2, max_readers=1)
    primary = MemoryError("builder construction failed")

    def construct(*_args):
        raise primary

    with monkeypatch.context() as faults:
        faults.setattr(ram_backing, "RamBuilder", construct)
        with pytest.raises(MemoryError) as caught:
            store.begin(IDENTITY, 4)
    assert caught.value is primary and store.record()["payload_bytes"] == 0
    assert not store._builders
    builder = store.begin(IDENTITY, 4)
    builder.close()
    store.close()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_builder_registration_after_effect_is_rolled_back_or_terminally_owned(cleanup_fails):
    store = RamObjects(8, max_objects=2, max_readers=1)
    primary, secondary = MemoryError("after builder registration"), RuntimeError("builder rollback failed")

    class Builders(set):
        def add(self, value):
            super().add(value)
            raise primary

        def discard(self, value):
            if cleanup_fails:
                raise secondary
            return super().discard(value)

    store._builders = Builders()
    with pytest.raises(MemoryError) as caught:
        store.begin(IDENTITY, 4)
    assert caught.value is primary
    if cleanup_fails:
        assert store._failure is primary and len(store._builders) == 1
        assert store._retained_reader is next(iter(store._builders))
        with pytest.raises(RuntimeError, match="poisoned"):
            store.begin(IDENTITY, 4)
    else:
        assert not store._builders and store.record()["payload_bytes"] == 0
        store._builders = set()
        store.begin(IDENTITY, 4).close()
        store.close()


def test_builder_retirement_arithmetic_failure_keeps_original_retryable_owner():
    store = RamObjects(8, max_objects=2, max_readers=1)
    builder = store.begin(IDENTITY, 4)
    builder.write(b"abcd")
    primary = MemoryError("retirement scalar allocation failed")

    class Used(int):
        def __sub__(self, _size):
            raise primary

    store._used = Used(4)  # Inject the otherwise native integer-allocation cut.
    with pytest.raises(MemoryError) as caught:
        builder.close()
    assert caught.value is primary and builder in store._builders
    assert not builder.finished and builder.data == b"abcd" and store._failure is None
    store._used = 4
    builder.close()
    assert store.record()["payload_bytes"] == 0
    store.close()


def test_builder_after_removal_failure_poison_retains_payload_before_reuse():
    store = RamObjects(8, max_objects=2, max_readers=1)
    builder = store.begin(IDENTITY, 4)
    builder.write(b"abcd")
    primary = MemoryError("after builder removal")

    class Builders(set):
        def remove(self, value):
            super().remove(value)
            raise primary

    store._builders = Builders(store._builders)
    with pytest.raises(MemoryError) as caught:
        builder.close()
    assert caught.value is primary and store._failure is primary
    assert store._retained_reader is builder and builder.data == b"abcd"
    with pytest.raises(RuntimeError, match="poisoned"):
        store.begin(IDENTITY, 4)


def test_builder_checks_failure_again_after_actor_poison_before_lock_acquisition(monkeypatch):
    import threading

    store = RamObjects(8, max_objects=2, max_readers=1)
    checked, resume = threading.Event(), threading.Event()
    original, errors = store._check, []
    primary = ValueError("actor retirement failed")

    def check():
        original()
        if threading.current_thread() is worker and not checked.is_set():
            checked.set()
            assert resume.wait(5)

    def produce():
        try:
            store.begin(IDENTITY, 4)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=produce)
    monkeypatch.setattr(store, "_check", check)
    worker.start()
    try:
        assert checked.wait(5)
        with store._lock, pytest.raises(ValueError) as caught:
            store._poison(primary)
        assert caught.value is primary
    finally:
        resume.set()
        worker.join(5)
    assert not worker.is_alive() and len(errors) == 1
    assert isinstance(errors[0], RuntimeError) and errors[0].__cause__ is primary
    assert not store._builders and store._used == 0


@pytest.mark.parametrize("path", ["acquire", "context", "finish"])
def test_failed_reader_cleanup_has_durable_owner_when_diagnostic_marker_cannot_allocate(monkeypatch, path):
    import tensorfold.cleanup as cleanup
    from tensorfold.families.qwen4_exp.cuda import ram_backing
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired

    store = RamObjects(16, max_objects=3, max_readers=2)
    key = put(store, b"abcd")
    primary, secondary = ValueError("original reader operation"), MemoryError("reader retirement failed")
    if path == "context":
        reader = store.acquire(key)
    elif path == "finish":
        store._objects[key].identity = "b" * 64
        builder = store.begin(IDENTITY, 4)
        builder.write(b"abcd")

    class FailedMarker(dict):
        @staticmethod
        def __setitem__(mapping, name, value):
            if name == "_tensorfold_retained_owners":
                raise MemoryError("diagnostic ownership marker failed")
            dict.__setitem__(mapping, name, value)

    def verify(_reader):
        raise primary

    def close(_reader):
        raise secondary

    with monkeypatch.context() as faults:
        faults.setattr(cleanup, "dict", FailedMarker, raising=False)
        faults.setattr(ram_backing.RamReader, "close", close)
        if path == "acquire":
            faults.setattr(ram_backing.RamReader, "verify", verify)
        with pytest.raises(ValueError) as reported:
            if path == "acquire":
                store.acquire(key)
            elif path == "context":
                with reader:
                    raise primary
            else:
                builder.finish()
    error = reported.value
    assert path == "finish" or error is primary
    assert store._failure is error and not _unretired(error)
    assert isinstance(store._retained_reader, ram_backing.RamReader)
    assert store._retained_reader.key == key and store._readers == store._objects[key].readers == 1
    seen, pending = set(), [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend((current.__cause__, current.__context__))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    assert id(secondary) in seen
    with pytest.raises(RuntimeError, match="poisoned") as unhealthy:
        store.check_health()
    assert unhealthy.value.__cause__ is error
    with pytest.raises(RuntimeError, match="poisoned"):
        store.begin(IDENTITY, 1)
