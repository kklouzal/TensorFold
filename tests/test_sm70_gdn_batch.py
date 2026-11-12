"""Owned GDN frame/descriptor contracts. CPU doubles do not qualify kernels."""
import ast
import gc
import importlib
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from tensorfold.cuda.geometry import multi_static_domains
from tensorfold.families.qwen4_exp.cuda.gdn_multi import RoundTables, NativeTables

DEVICE = 'owned-cuda-like-device'


class Array:
    def __init__(self, shape=None, dtype=torch.float16, data=None):
        self.data = torch.zeros(shape, dtype=dtype) if data is None else data
        self.device, self.requires_grad = DEVICE, False
        self.fail = False
    @property
    def shape(self):
        return self.data.shape
    @property
    def dtype(self):
        return self.data.dtype
    def data_ptr(self):
        return self.data.data_ptr()
    def numel(self):
        return self.data.numel()
    def element_size(self):
        return self.data.element_size()
    def stride(self, axis=None):
        return self.data.stride() if axis is None else self.data.stride(axis)
    def is_contiguous(self):
        return self.data.is_contiguous()
    def is_neg(self):
        return False
    def is_conj(self):
        return False
    def __getitem__(self, key):
        return Array(data=self.data[key])
    def view(self, *shape):
        return Array(data=self.data.view(*shape))
    def copy_(self, value, **kwargs):
        if self.fail:
            raise OSError('controlled descriptor copy failure')
        self.data.copy_(value.data if isinstance(value, Array) else value)


class State:
    pass


class Event:
    def __init__(self):
        self.waits, self.records = 0, 0
    def synchronize(self):
        self.waits += 1
    def record(self):
        self.records += 1


def fixture(monkeypatch, *, slots=4, width=3, layers=2, keep=False):
    stream = object()
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: stream)
    static = RoundTables.__new__(RoundTables)
    static.native_half, static.native_states, static.native_current = True, (), None
    static.slots, static.width, static.rows, static.lin = slots, width, slots*width, layers
    static.at = {'starts': 8*static.rows}
    static.i64 = Array((8*layers*slots,), torch.int64)
    static.h64 = torch.zeros(8*layers*slots, dtype=torch.int64)
    static.i32 = Array((8*static.rows+slots+1,), torch.int32)
    static.h32 = torch.zeros(8*static.rows+slots+1, dtype=torch.int32)
    static.queued, static.copied = False, Event()
    b = NS(h=Array((static.rows, 10240)), proj=Array((layers, static.rows, 2060)),
           gout=Array((static.rows, 768)), gxs=Array((static.rows, 24), torch.float32),
           rows=static.rows, prefill=False, split=None, owner_graph_family='uncaptured')
    states = []
    for _ in range(slots):
        st = State()
        st.conv, st.rec = Array((layers, 3, 1280)), Array((2, layers, 6, 128, 128), torch.float32)
        st.cur = [0]*layers
        st.scratch = [NS(k=Array((width, 2, 128), torch.float32), v=Array((width, 6, 128)),
                         g=Array((width, 6), torch.float32), b=Array((width, 6), torch.float32)) for _ in range(layers)]
        if keep:
            for sc in st.scratch:
                sc.keep_rows = Array((1,), torch.int32)
        states.append(st)
    static.bind_native(states, b)
    return static, b, states, stream


def test_current_parity_scratch_and_masks_are_published_before_submission(monkeypatch):
    static, b, states, _ = fixture(monkeypatch, keep=True)
    states[2].cur = [1, 0]
    frame = static.native_frame([(states[2], 0, 2), (states[0], 2, 3)], b)
    assert type(frame) is NativeTables and frame.rows == 3 and static.copied.records == 1
    ptrs, starts = frame.operands(0, b.proj[0, :3], b.gout[:3], b.gxs[:3])
    values = ptrs.data
    st = states[2]
    stride = st.rec.stride(0)*4
    assert values[1, 0] == st.rec.data_ptr()+stride
    assert values[2, 0] == st.rec.data_ptr()
    assert values[3, 0] == st.scratch[0].k.data_ptr()
    assert values[4, 0] == st.scratch[0].v.data_ptr()
    assert values[7, 0] == st.scratch[0].keep_rows.data_ptr()
    assert starts.data.tolist() == [0, 2, 3]
    assert ptrs.stride(0) == static.lin*static.slots
    states[2].cur = [0, 1]  # A new generation/parity uses the same live storage.
    newer = static.native_frame([(states[0], 0, 1), (states[2], 1, 2)], b)
    assert static.copied.waits == 1
    assert newer.pointers.data[1, 0, 1] == st.rec.data_ptr()
    with pytest.raises(ValueError, match='current'):
        frame.operands(0, b.proj[0, :3], b.gout[:3], b.gxs[:3])


def test_C1_prefill_capture_and_replaced_State_storage_keep_scalar(monkeypatch):
    static, b, states, _ = fixture(monkeypatch)
    assert static.native_frame([(states[0], 0, 1)], b) is None
    for attr, value in [('prefill', True), ('split', object()), ('owner_graph_family', 'captured')]:
        original = getattr(b, attr)
        setattr(b, attr, value)
        assert static.native_frame([(states[0], 0, 1), (states[1], 1, 2)], b) is None
        setattr(b, attr, original)
    states[0].rec = Array(states[0].rec.shape, torch.float32)
    assert static.native_frame([(states[0], 0, 1), (states[1], 1, 2)], b) is None
    assert static.copied.records == 0


def test_only_explicit_exact_preparation_admits_single_state_and_original_buffer(monkeypatch):
    static, b, states, _ = fixture(monkeypatch, width=5)
    segs = [(states[0], 0, 5)]
    assert static.native_frame(segs, b) is None
    frame = static.native_frame(segs, b, allow_single=True)
    assert type(frame) is NativeTables and frame.rows == 5
    pointers, starts = frame.operands(0, b.proj[0, :5], b.gout[:5], b.gxs[:5])
    assert pointers.shape == (8, 1) and starts.data.tolist() == [0, 5]
    with pytest.raises(ValueError, match='explicit boolean'):
        static.native_frame(segs, b, allow_single=1)
    assert static.native_frame(segs, NS(**vars(b)), allow_single=True) is None


@pytest.mark.parametrize('changed', ['parity', 'storage', 'segments', 'stream'])
def test_exact_current_frame_refuses_changed_preparation_without_republishing(monkeypatch, changed):
    static, b, states, stream = fixture(monkeypatch, width=5)
    segs = [(states[0], 0, 5)]
    frame = static.native_frame(segs, b, allow_single=True)
    frame.validate_current(segs, b)
    copies = static.copied.records
    if changed == 'parity':
        states[0].cur[0] ^= 1
    elif changed == 'storage':
        states[0].rec = Array(states[0].rec.shape, torch.float32)
    elif changed == 'segments':
        segs = [(states[0], 0, 4)]
    else:
        monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: object())
    with pytest.raises(ValueError, match='current prepared'):
        frame.validate_current(segs, b)
    assert static.copied.records == copies


@pytest.mark.parametrize('windows', [[(0, 0, 1), (0, 1, 2)], [(0, 0, 1), (1, 2, 3)],
                                    [(0, 0, 4), (1, 4, 5)], [(0, 0, 1), (1, 1, 1)]])
def test_invalid_alias_or_window_fails_before_DMA(monkeypatch, windows):
    static, b, states, _ = fixture(monkeypatch)
    with pytest.raises(ValueError, match='bounded'):
        static.native_frame([(states[s], a, z) for s, a, z in windows], b)
    assert static.copied.records == 0


def test_registration_rejects_cross_State_mutable_alias(monkeypatch):
    static, b, states, _ = fixture(monkeypatch)
    states[1].rec = states[0].rec
    with pytest.raises(ValueError, match='overlaps'):
        static.bind_native(states, b)


def test_stream_mismatch_and_failed_copy_invalidate_old_frame(monkeypatch):
    static, b, states, stream = fixture(monkeypatch)
    segs = [(states[0], 0, 1), (states[1], 1, 2)]
    old = static.native_frame(segs, b)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: object())
    with pytest.raises(ValueError, match='current'):
        old.operands(0, b.proj[0, :2], b.gout[:2], b.gxs[:2])
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: stream)
    static.i64.fail = True
    with pytest.raises(OSError, match='descriptor'):
        static.native_frame(segs, b)
    assert static.native_current is None
    with pytest.raises(ValueError, match='current'):
        old.operands(0, b.proj[0, :2], b.gout[:2], b.gxs[:2])


def test_frame_has_no_static_current_reference_cycle(monkeypatch):
    static, b, states, _ = fixture(monkeypatch)
    gc.disable()
    try:
        frame = static.native_frame([(states[0], 0, 1), (states[1], 1, 2)], b)
        ref = weakref.ref(frame)
        del frame
        assert ref() is None and static.native_current() is None
    finally:
        gc.enable()


def test_exact_additional_pointer_words_are_charged_in_both_domains():
    config = {'num_hidden_layers': 48, 'layer_types': ['linear_attention']*36+['full_attention']*12}
    ordinary = multi_static_domains(config, 24, 1, False)
    native = multi_static_domains(config, 24, 1, False, gdn_pointer_fields=8)
    assert tuple(a-b for a, b in zip(native, ordinary)) == (6*36*24*8,)*2
    with pytest.raises(ValueError):
        multi_static_domains(config, 24, 1, False, gdn_pointer_fields=7)


def test_native_extension_declares_its_owned_source_root_with_the_real_stager(monkeypatch, tmp_path):
    from tensorfold.cuda import build
    from tensorfold.cuda.sm70 import gdn_native, provider, source_cache

    root = Path(gdn_native.__file__).parent.parent
    # Normal installed execution is unchanged: both roots are the same CUDA
    # package directory. An explicitly loaded component declares its own root.
    assert root == Path(source_cache.__file__).parent.parent
    actual_stage, calls = source_cache.stage_sources, []
    def stage(sources, *, headers, source_root):
        assert source_root == root
        assert all(Path(path).is_relative_to(root) for path in [*sources, *headers])
        result = actual_stage(sources, headers=headers, source_root=source_root, cache_root=tmp_path)
        calls.append(result)
        return result
    native = object()
    monkeypatch.setattr(provider, '_native', lambda: NS(__name__='owned_gdn_source_closure_probe'))
    monkeypatch.setattr(source_cache, 'stage_sources', stage)
    monkeypatch.setattr(build, 'load', lambda *args, **kwargs: native)
    gdn_native.extension.cache_clear()
    try:
        assert gdn_native.extension() is native and len(calls) == 1
        assert [Path(path).read_bytes() for path in calls[0]] == [
            (root/'sm70/gdn_bindings.cpp').read_bytes(), (root/'sm70/gdn.cu').read_bytes()]
    finally:
        gdn_native.extension.cache_clear()


@pytest.mark.parametrize('key', ['moe_selection_prepared', 'half_moe_partial_prepared'])
@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_loaded_weights_are_owned_before_fallible_capability_publication(key, cleanup_fails):
    from tests.test_flashnext_t47_lifecycle_source import subject, roots

    source = Path(__file__).resolve().parents[1]/'src/tensorfold/families/qwen4_exp/cuda/engine.py'
    cls_ast = next(n for n in ast.parse(source.read_bytes()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'FlashNextEngine')
    initialize = next(n for n in cls_ast.body if isinstance(n, ast.FunctionDef) and n.name == '_initialize')
    names = {'moe_selection_prepared', 'half_moe_partial_prepared'}
    def selected(node):
        if not isinstance(node, ast.Assign):
            return False
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id == 'self' and target.attr == 'w':
                    return True
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
                if (isinstance(target.value.value, ast.Name) and target.value.value.id == 'w'
                        and target.value.attr == 'meta' and isinstance(target.slice, ast.Constant)
                        and target.slice.value in names):
                    return True
        return False
    statements = [node for node in initialize.body if selected(node)]
    assert len(statements) == 3
    code = compile(ast.Module(body=statements, type_ignores=[]), str(source), 'exec')
    primary, secondary, events, owners = MemoryError('capability publication'), OSError('cache retirement'), [], []
    class Metadata(dict):
        def __setitem__(self, name, value):
            assert owners[0].w is weights, 'actual loaded weights escaped the Engine journal'
            events.append(('publish', name))
            if name == key:
                raise primary
            return super().__setitem__(name, value)
    def retire():
        assert owners[0].w is weights
        events.append('retired-loaded-cache')
        if cleanup_fails:
            raise secondary
    weights = NS(device='owned-device', meta=Metadata(expert_cache=NS(close=retire)))
    engine = subject()
    def run(self, *args, **kwargs):
        owners.append(self)
        exec(code, {'self': self, 'w': weights, 'moe_selection_prepared': object(), 'half_partial_prepared': object()})
    engine._initialize = run
    runtime = ModuleType('torch')
    def fence(device):
        assert owners[0].w is weights and device == weights.device
        events.append('fenced-loaded-weights')
    runtime.cuda = NS(synchronize=fence)
    # Keep the real constructor diagnostic module outside the temporary module
    # map: patch.dict would otherwise remove its sys.modules entry while the
    # parent package retained a different, stale module for later fatal tests.
    importlib.import_module('tensorfold.cuda.cohort')
    with patch.dict(sys.modules, {'torch': runtime}):
        with pytest.raises(MemoryError) as caught:
            engine(Path('opaque-loaded-checkpoint'))
    assert caught.value is primary and owners[0].w is weights
    assert 'fenced-loaded-weights' in events and 'retired-loaded-cache' in events
    if cleanup_fails:
        assert id(secondary) in roots(primary) and not owners[0]._closed
    else:
        assert owners[0]._closed
