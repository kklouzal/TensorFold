"""Failure containment against maintained lease methods, without a GPU import.

Legal stream/event doubles expose scheduling and lifetime fences. Payload math
and CUDA execution remain the responsibility of the maintained CUDA gates.
"""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
import threading
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def maintained_class(path, name, methods, namespace):
    tree = ast.parse((ROOT / path).read_bytes())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    selected = ast.ClassDef(name=name, bases=[], keywords=[], decorator_list=[],
                            body=[node for node in original.body
                                  if isinstance(node, ast.FunctionDef) and node.name in methods])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              selected], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


class OpaqueFailure(Exception):
    formatting_calls = 0

    def __repr__(self):
        type(self).formatting_calls += 1
        raise LookupError("foreign repr must not run")

    def __str__(self):
        type(self).formatting_calls += 1
        raise LookupError("foreign str must not run")


class Event:
    def __init__(self):
        self.error = None
        self.records = 0
        self.waits = 0

    def record(self, stream):
        self.records += 1
        if self.error is not None:
            raise self.error

    def synchronize(self):
        self.waits += 1


class Stream:
    def __init__(self):
        self.error = None
        self.waits = 0
        self.synchronizations = 0

    def wait_event(self, event):
        self.waits += 1
        if self.error is not None:
            raise self.error

    def synchronize(self):
        self.synchronizations += 1


class Policy:
    def __init__(self):
        self.keys = [None, None]
        self.resident = {}
        self.recency = [0, 0]
        self.tick = 0
        self._logical = None

    def touch(self, keys):
        self.tick += 1

    def victim(self, protected):
        return next(slot for slot in range(2) if slot not in protected)

    def remove(self, slot):
        old = self.keys[slot]
        if old is not None:
            del self.resident[old]
        self.keys[slot] = None

    def install(self, slot, key):
        self.keys[slot] = key
        self.resident[key] = slot


def cache_fixture():
    OpaqueFailure.formatting_calls = 0
    stream = Stream()
    torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda device: nullcontext(),
                            is_current_stream_capturing=lambda: False, current_stream=lambda device: stream),
                            is_inference=lambda value: False)
    namespace = {"contextmanager": contextmanager, "torch": torch,
                 "_integer": lambda value, name, minimum: value}
    cleanup = {}
    exec(compile((ROOT / "src/tensorfold/cleanup.py").read_text(), "actual_cleanup", "exec"), cleanup)
    namespace["raise_failures"] = cleanup["raise_failures"]
    cls = maintained_class("src/tensorfold/cuda/expert_cache.py", "HostExpertCache",
                           {"_usable", "lease", "close"}, namespace)
    cache = cls()
    cache.device = "controlled-cuda"
    cache._lock = threading.RLock()
    cache._closed = cache._active = False
    cache._failure = cache._last_stream = None
    cache._last_use = Event()
    cache._policy = Policy()
    cache.capacity = 2
    cache._layers = {0: SimpleNamespace(count=2, source=(SimpleNamespace(_version=0),), versions=(0,),
                                       views=("borrowed-views",))}
    cache._pool = cache._staging = object()
    cache.hits = cache.misses = cache.evictions = cache.copied_bytes = 0
    cache.copy_error = None
    cache.copy_calls = 0

    def copy(layer, expert, slot, current):
        cache.copy_calls += 1
        if cache.copy_error is not None and cache.copy_calls == 2:
            raise cache.copy_error
        return 16

    cache._copy = copy
    return cache, stream, namespace


def rejected(cache):
    try:
        with cache.lease(0, [0]):
            raise AssertionError("poisoned cache accepted a consumer")
    except RuntimeError as error:
        assert "unusable after scheduling failure" in str(error)
    else:
        raise AssertionError("poisoned cache remained usable")
    assert type(cache._failure) is str
    assert not any(isinstance(value, BaseException) for name, value in vars(cache).items()
                   if name not in {"copy_error"})
    assert OpaqueFailure.formatting_calls == 0


def test_partial_fill_poison_preserves_opaque_primary_and_last_use_fence():
    cache, stream, _ = cache_fixture()
    primary = cache.copy_error = OpaqueFailure()
    try:
        with cache.lease(0, [0, 1]):
            raise AssertionError("incomplete payload reached consumer")
    except OpaqueFailure as error:
        assert error is primary
    assert cache.misses == 1 and cache.copied_bytes == 16 and cache._policy.resident == {(0, 0): 0}
    assert cache._last_use.records == 1 and cache._last_stream is stream and not cache._active
    rejected(cache)
    cache.copy_error = None
    cache.close()
    assert stream.synchronizations == 1 and cache._last_use.waits == 0 and cache._closed


def test_failed_stream_dependency_poison_preserves_opaque_primary():
    cache, stream, _ = cache_fixture()
    cache._last_stream = Stream()
    primary = stream.error = OpaqueFailure()
    try:
        with cache.lease(0, [0]):
            raise AssertionError("unordered consumer reached payload")
    except OpaqueFailure as error:
        assert error is primary
    assert stream.waits == 1 and cache.copy_calls == 0 and cache._last_use.records == 1
    rejected(cache)


def test_last_use_failure_poison_preserves_opaque_primary():
    cache, stream, _ = cache_fixture()
    primary = cache._last_use.error = OpaqueFailure()
    try:
        with cache.lease(0, [0]):
            pass
    except OpaqueFailure as error:
        assert error is primary
    assert cache.misses == 1 and cache._last_stream is stream and not cache._active
    rejected(cache)
    cache.close()
    assert stream.synchronizations == 1


def test_valid_consumer_failure_keeps_cache_usable():
    cache, _, _ = cache_fixture()
    primary = OpaqueFailure()
    try:
        with cache.lease(0, [0]):
            raise primary
    except OpaqueFailure as error:
        assert error is primary
    assert cache._failure is None and not cache._active
    with cache.lease(0, [0]):
        pass
    assert cache.hits == 1 and cache._last_use.records == 2 and OpaqueFailure.formatting_calls == 0


def test_consumer_primary_survives_failed_last_use_with_cleanup_cause():
    cache, _, _ = cache_fixture()
    primary = OpaqueFailure()
    cleanup = cache._last_use.error = OpaqueFailure()
    try:
        with cache.lease(0, [0]):
            raise primary
    except OpaqueFailure as error:
        assert error is primary and error.__cause__ is cleanup
    rejected(cache)


def test_partial_pointer_publication_poison_preserves_opaque_primary():
    cache, _, namespace = cache_fixture()
    cls = maintained_class("src/tensorfold/cuda/exl3/host_experts.py", "CachedExl3Experts", {"lease"}, namespace)
    adapter = cls()
    adapter._immutable, adapter._versions = (), ()
    adapter.cache, adapter.layer_id, adapter._tables = cache, 0, object()
    primary = OpaqueFailure()
    publication = []

    def publish(layer, tables, ids, mapping):
        publication.append((layer, ids, mapping))
        raise primary

    cache.publish = publish
    try:
        with adapter.lease([0]):
            raise AssertionError("partially published pointers reached consumer")
    except OpaqueFailure as error:
        assert error is primary
    assert len(publication) == 1 and cache._last_use.records == 1 and not cache._active
    rejected(cache)
