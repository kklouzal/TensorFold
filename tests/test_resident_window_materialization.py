"""Run the actual allocation/gate constructor block and normal/forced lifecycle on CPU."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.memory_gate import MemoryGate
from tensorfold.families.qwen4_exp.cuda.service_memory import context_reservation, future_service
from tests import test_flashnext_absolute_grow_host as _absolute_growth

GIB = 1 << 30


def allocation_block():
    root = os.environ.get("TENSORFOLD_PRIVATE_SOURCE_ROOT")
    if root is not None:
        file = Path(root) / "src/tensorfold/families/qwen4_exp/cuda/multi.py"
    else:
        import inspect
        from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

        file = Path(inspect.getsourcefile(MultiDecoder))
    cls = next(n for n in ast.parse(file.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "MultiDecoder")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")

    def assigned(n, attribute):
        return isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == attribute for t in n.targets
        )

    start = next(i for i, n in enumerate(init.body) if assigned(n, "full_resident"))
    end = next(i for i, n in enumerate(init.body) if assigned(n, "held") and i > start)
    names = [
        "self",
        "w",
        "slots",
        "capacity",
        "depth",
        "kv_dtype",
        "kv_pair",
        "kv_key_dtype",
        "kv_value_dtype",
        "workspace_bytes",
        "full_windows",
    ]
    f = ast.FunctionDef(
        name="prepare",
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg=n) for n in names], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=init.body[start : end + 1],
        decorator_list=[],
    )
    return file, ast.Module(body=[f], type_ignores=[])


class Array:
    def __init__(self, nbytes):
        self.nbytes = nbytes

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1


@pytest.mark.parametrize("slots", range(2, 9))
@pytest.mark.parametrize("selection", ["default", "shared", "equal", "larger"])
@pytest.mark.parametrize("aux", [0, 2560])
def test_production_constructor_owns_every_promised_cache_and_counts_it_once(slots, selection, aux):
    calls = []

    class State:
        def __init__(self, w, rows, max_rows, kv_dtype, *, limit, **kwargs):
            self.capacity, self.limit = rows, limit
            self.dspark_runtime = NS(plan=NS(cache_row_bytes=aux)) if aux else None
            self.array = Array(self.cache_bytes())
            calls.append((rows, limit, max_rows))

        def cache_bytes(self, rows=None):
            return (self.capacity if rows is None else rows) * (100 + aux)

        def layer_bytes(self, rows):
            return rows * 10

    full = {"default": None, "shared": 1, "equal": slots, "larger": slots + 1}[selection]
    file, module = allocation_block()
    ns = {
        "State": State,
        "_tensors": lambda state: [state.array],
        "FIRST": 256,
        "GIB": GIB,
        "context_reservation": context_reservation,
        "MemoryGate": MemoryGate,
        "torch": NS(cuda=NS(is_available=lambda: False)),
    }
    exec(compile(ast.fix_missing_locations(module), str(file), "exec"), ns)
    dec = NS()
    ns["prepare"](dec, NS(), slots, 262148, 3, "int8", None, None, None, 0, full)
    materialized = selection in ("equal", "larger")
    rows = 262148 if materialized else 256
    assert calls == [(rows, 262148, 4)] * slots
    assert dec.full_resident == materialized
    assert dec.slot_bytes == rows * (100 + aux)
    assert sum(st.array.nbytes for st in dec.free) == slots * rows * (100 + aux)
    assert dec.window_bytes == 262148 * (100 + aux)
    expected_held = slots * (rows - 256) * (100 + aux) if full is not None else 0
    assert dec.memory_gate.held == expected_held
    if materialized:
        # All full KV bytes are already allocated. Only replacement and
        # workspace remain future allocations; they must not count KV twice.
        p = context_reservation(dec.free, full, preserve_overlap=True)
        assert p["growth_bytes"] == 0
        assert p["copy_overlap_bytes"] == 262148 * max(10, aux)
        assert dec.memory_gate.room == expected_held + p["copy_overlap_bytes"] + 2 * GIB
        assert dec.memory_gate.fits(p["copy_overlap_bytes"])
        assert not dec.memory_gate.fits(p["copy_overlap_bytes"] + 1)


@pytest.mark.parametrize("aux", [0, 2560])
def test_materialized_graph_future_excludes_owned_kv_but_protects_recovery_copy(aux):
    class State:
        limit = capacity = 262148
        rec = conv = ple_tail = NS(numel=lambda: 8, element_size=lambda: 4)

        def cache_bytes(self, rows=None):
            return (self.capacity if rows is None else rows) * (100 + aux)

        def layer_bytes(self, rows):
            return rows * 10

        dspark_runtime = NS(plan=NS(cache_row_bytes=aux)) if aux else None

    class Rec:
        def __getitem__(self, index):
            return State.rec

        def element_size(self):
            return 4

    states = [State() for _ in range(3)]
    for st in states:
        st.rec = Rec()
    p = context_reservation(states, 3, preserve_overlap=True)
    w = NS(cfg=NS(streams=4, hidden=2560), head=NS(n=31040), mtp=None)
    future = future_service(
        w,
        states,
        streams=3,
        retained=20,
        growth_bytes=0,
        copy_overlap_bytes=p["copy_overlap_bytes"],
        minimum_workspace_bytes=2 * GIB,
    )
    assert future["growth_bytes"] == 0
    assert future["copy_overlap_bytes"] == 262148 * max(10, aux)
    assert future["future_bytes"] == future["copy_overlap_bytes"] + future["snapshot_bytes"] + 2 * GIB


@pytest.fixture
def growth():
    case = _absolute_growth.AbsoluteGrowthTests()
    case.setUp()
    from tests.test_flashnext_absolute_grow_host import source_class

    file, _ = allocation_block()
    cls = source_class(str(file), "MultiDecoder", {"_shrink"}, case.dec._grow.__func__.__globals__.copy())
    case.dec._shrink = cls._shrink.__get__(case.dec)
    for st in case.dec.free:
        st.reset = lambda w: None
    try:
        yield case
    finally:
        case.doCleanups()


def test_normal_shrink_preserves_full_cache_allocations_and_gate_holding(growth):
    dec, allocator = growth.dec, growth.alloc
    st = dec.free[0]
    assert dec._grow(st, st.limit, alone=True)
    dec.full_resident = True
    before = (allocator.allocated, dec.memory_gate.held, st.capacity)
    dec._shrink(st)
    assert (allocator.allocated, dec.memory_gate.held, st.capacity) == before
    assert st.capacity == st.limit


def test_forced_retirement_restores_full_window_before_republishing_free_slot(growth):
    dec, allocator = growth.dec, growth.alloc
    st = dec.free.pop(0)
    assert dec._grow(st, st.limit, alone=True)
    dec.full_resident = True
    before = (allocator.allocated, dec.memory_gate.held)
    dec.failed_slots = [st]
    dec._retire_failed_slot(st)
    assert st.capacity == st.limit
    assert dec.free.count(st) == 1 and st not in dec.failed_slots
    assert (allocator.allocated, dec.memory_gate.held) == before


def test_failed_full_window_restoration_never_publishes_undersized_slot(growth):
    dec = growth.dec
    st = dec.free.pop(0)
    dec.full_resident = True
    dec.failed_slots = [st]
    dec._grow = lambda *args, **kwargs: False
    with pytest.raises(RuntimeError, match="promised full context"):
        dec._retire_failed_slot(st)
    assert st not in dec.free and st in dec.failed_slots
