"""CPU lifecycle/dispatch controls for the private FP16 State batch owner."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.qwen4_exp.cuda import gdn_state_batch as batch  # noqa: E402
from tensorfold.cuda.sm70 import gdn_native  # noqa: E402


class TensorAddress:
    device = torch.device("cuda:0")

    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class State:
    def __init__(self, pointer):
        self.rec, self.conv = TensorAddress(pointer), TensorAddress(pointer + 100)
        self.scratch = []
        self.slot_generation = self.history_generation = 0


class Backend:
    def __init__(self):
        self.fills = []
        self.calls = 0

    def fill_batch(self, host, frame, segments, layers):
        self.fills.append(tuple(int(frame[i, 3]) for i in range(len(segments))))

    def batch(self, *args):
        self.calls += 1


@pytest.fixture
def owner(monkeypatch):
    log = []
    stream = SimpleNamespace(cuda_stream=17)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    event = SimpleNamespace(record=lambda *_: log.append("record"), synchronize=lambda: log.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    backend = Backend()
    monkeypatch.setattr(gdn_native, "extension", lambda: backend)
    result = batch.GDNStateBatch(1, "cuda:0")
    return result, backend, log, stream


def test_funding_has_no_device_descriptor_and_bounds_native_stack():
    assert batch.workspace_bytes(36) == (0, 36 * 24 * 10 * 8 + 24 * 4 * 8 + 32768 + (128 << 10))
    for invalid in (0, 49, True):
        with pytest.raises(ValueError):
            batch.workspace_bytes(invalid)


def test_static_catalogue_survives_reorder_and_refreshes_only_changed_generation(owner):
    obj, backend, _, _ = owner
    a, b = State(1000), State(2000)
    assert obj.bind([(a, 0, 1), (b, 1, 3)])
    assert backend.fills[-1] == (1, 1)
    assert obj.bind([(b, 0, 2), (a, 2, 3)])
    assert backend.fills[-1] == (0, 0)
    a.slot_generation += 1
    assert obj.bind([(b, 0, 2), (a, 2, 3)])
    assert backend.fills[-1] == (0, 1)


def test_one_frame_event_retains_states_until_consumer_retirement(owner):
    obj, backend, log, _ = owner
    a, b = State(1000), State(2000)
    obj.bind([(a, 0, 1), (b, 1, 2)])
    for layer in range(3):
        assert obj.submit(layer, None, None, None, None, None, 1e-6, None, None, negative_decay=True)
    assert backend.calls == 3 and log == []
    obj.seal()
    assert log == ["record"] and obj.owners[0] is a
    obj.retire()
    assert log == ["record", "synchronize"] and all(item is None for item in obj.owners)


@pytest.mark.parametrize("size", [1, 25])
def test_n1_and_over_capacity_keep_original_path(owner, size):
    obj, backend, _, _ = owner
    states = [State(1000 * i) for i in range(size)]
    assert obj.bind([(state, i, i + 1) for i, state in enumerate(states)]) is False
    assert backend.fills == []


def test_general_long_chain_keeps_original_path(owner):
    obj, backend, _, _ = owner
    assert obj.bind([(State(1), 0, 4), (State(2), 4, 5)]) is False
    assert backend.fills == []


def test_duplicate_states_rejected(owner):
    obj, _, _, _ = owner
    state = State(1)
    with pytest.raises(ValueError, match="independent"):
        obj.bind([(state, 0, 1), (state, 1, 2)])


def test_storage_or_generation_mutation_before_launch_rejected(owner):
    obj, backend, _, _ = owner
    a, b = State(1000), State(2000)
    obj.bind([(a, 0, 1), (b, 1, 2)])
    a.rec.pointer += 8
    with pytest.raises(ValueError, match="storage changed"):
        obj.submit(0, None, None, None, None, None, 1e-6, None, None, negative_decay=True)
    assert backend.calls == 0


def test_cross_stream_reuse_rejected(owner):
    obj, _, _, stream = owner
    stream.cuda_stream = 18
    with pytest.raises(ValueError, match="cross"):
        obj.bind([(State(1), 0, 1), (State(2), 1, 2)])


def test_weak_catalogue_does_not_retain_completed_state_or_kv(owner):
    import gc
    import weakref

    obj, _, _, _ = owner
    a, b = State(1000), State(2000)
    reference = weakref.ref(a)
    obj.bind([(a, 0, 1), (b, 1, 2)])
    obj.retire()
    del a
    gc.collect()
    assert reference() is None


def test_graph_binding_refuses_and_model_capture_keeps_original(owner, monkeypatch):
    obj, backend, _, _ = owner
    obj.bind([(State(1), 0, 1), (State(2), 1, 2)])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert obj.submit(0, None, None, None, None, None, 1e-6, None, None, negative_decay=True) is False
    with pytest.raises(ValueError, match="capture"):
        obj.bind([])
    assert backend.calls == 0
