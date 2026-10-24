"""Native pending-safe target features, without a DSpark checkpoint or engine hook."""
from types import SimpleNamespace
import gc
import weakref

import pytest
import torch

from test_flashnext_sm70_family import _small_weights
from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda import forward as target, glue
from tensorfold.families.qwen4_exp.cuda.dspark_taps import TapWorkspace
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 target tap gate")


@pytest.fixture(autouse=True)
def native_family(monkeypatch):
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", True)


def config(hidden=2560, before=(2, 13, 24, 35, 46)):
    return SimpleNamespace(hidden=hidden, taps=tuple(i - 1 for i in before), capture_before=before)


def exact(actual, expected):
    torch.testing.assert_close(actual.contiguous().view(torch.int16 if actual.dtype == torch.float16 else torch.int32),
                               expected.contiguous().view(torch.int16 if expected.dtype == torch.float16 else torch.int32),
                               rtol=0, atol=0)


def oracle(h, pending):
    if pending is None:
        return h.view(h.shape[0], 4, -1).float().mean(1).half()
    mode, branch, weights, inject = pending
    materialized = torch.empty_like(h)
    pss = torch.empty((h.shape[0], h.shape[1] // 4 // 256, 4), device=h.device)
    glue.hc_writeback(h, materialized, pss, 4, mode, branch=branch if mode != 2 else None,
                      inject=inject, y=branch if mode == 2 else None, wts=weights)
    return materialized.view(h.shape[0], 4, -1).float().mean(1).half()


@pytest.mark.parametrize("rows", [1, 3, 18])
@pytest.mark.parametrize("mode", [None, 1, 2, 3, 4])
def test_native_pending_features_match_explicit_hc_and_hf_float_mean_without_target_mutation(rows, mode):
    torch.manual_seed(424)
    h = torch.randn((rows, 10240), device="cuda", dtype=torch.float16) * .7
    inject = torch.rand((rows, 4), device="cuda", dtype=torch.float16)
    pending = None
    if mode is not None:
        shape = (rows, 11, 2560) if mode == 2 else ((8, rows, 2560) if mode in (3, 4) else (rows, 2560))
        value = torch.randn(shape, device="cuda") * .2
        pending = mode, value, torch.rand((rows, 11), device="cuda") if mode == 2 else None, inject
    originals = [value.clone() for value in ([h] + ([] if pending is None else [v for v in pending[1:] if v is not None]))]
    expected = oracle(h, pending)
    owner = TapWorkspace(config(), rows, h.device)
    capture = owner.begin(rows)
    for index in owner.before:
        capture.before_layer(index, h, pending)
    batch = capture.finish()
    for slot in range(5):
        exact(batch.features.view(rows, 5, 2560)[:, slot], expected)
    for actual, previous in zip([h] + ([] if pending is None else [v for v in pending[1:] if v is not None]), originals):
        exact(actual, previous)
    batch.release()
    owner.close()


def test_full_2048_row_bound_hf_mean_adversaries_and_owned_allocation_forecast():
    rows = 2048
    h = torch.zeros((rows, 4, 2560), device="cuda", dtype=torch.float16)
    h[:, 0].fill_(65504)
    h[:, 1].fill_(2**-12)
    h[:, 2].fill_(-65504)
    h[:, 3].fill_(2**-12)
    owner = TapWorkspace(config(), rows, h.device)
    expected = h.float().mean(1).half()
    # Warm the operation before measuring its allocation envelope.
    warm = owner.begin(rows)
    for index in owner.before:
        warm.before_layer(index, h)
    warm.finish().release()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    capture = owner.begin(rows)
    for index in owner.before:
        capture.before_layer(index, h)
    batch = capture.finish()
    torch.cuda.synchronize()
    temporary = torch.cuda.max_memory_allocated() - before
    assert temporary <= owner.plan.transient_bytes
    assert owner.nbytes == owner.plan.device_bytes == 115671040
    for slot in range(5):
        exact(batch.features.view(rows, 5, 2560)[:, slot], expected)
    print({"tap_rows": rows, "owned_device_bytes": owner.nbytes, "measured_operator_temporary_bytes": temporary,
           "conservative_transient_bytes": owner.plan.transient_bytes}, flush=True)
    batch.release()
    owner.close()


def test_captured_taps_replay_live_inputs_and_enforce_graph_lifetime():
    h = torch.randn((3, 10240), dtype=torch.float16, device="cuda")
    owner = TapWorkspace(config(), 3, h.device)
    warm = owner.begin(3)
    for index in owner.before:
        warm.before_layer(index, h)
    warm.finish().release()
    capture = owner.begin(3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in owner.before:
            capture.before_layer(index, h)
    plan = capture.finish_capture(graph)
    for scale in (.5, -.125):
        h.mul_(scale)
        batch = plan.replay()
        expected = h.view(3, 4, 2560).float().mean(1).half()
        for slot in range(5):
            exact(batch.features.view(3, 5, 2560)[:, slot], expected)
        batch.release()
    with pytest.raises(ValueError, match="graph plan"):
        owner.close()
    plan.retire()
    owner.close()
    with pytest.raises(ValueError, match="retired"):
        plan.replay()


def test_cross_stream_feature_consumer_release_precedes_next_producer_overwrite():
    h = torch.randn((3, 10240), dtype=torch.float16, device="cuda")
    original = h.view(3, 4, 2560).float().mean(1).half()
    owner = TapWorkspace(config(before=(2,)), 3, h.device)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        capture = owner.begin(3)
        capture.before_layer(2, h)
        batch = capture.finish()
    with torch.cuda.stream(consumer):
        copied = batch.features.clone()
        batch.release()
    with torch.cuda.stream(producer):
        h.add_(1)
        capture = owner.begin(3)
        capture.before_layer(2, h)
        next_batch = capture.finish()
        next_batch.release()
    torch.cuda.synchronize()
    exact(copied, original)
    owner.close()


def test_failed_or_reset_target_lease_can_release_and_retire_its_actual_device_storage():
    h = torch.randn((3, 10240), dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    owner = TapWorkspace(config(), 3, h.device)
    storage = weakref.ref(owner.features)
    state = SimpleNamespace(pos=0, version=0, kv_identity=("fp16", "int8"), cur=[0],
                            _kv_error=None, _resize_error=None)
    capture = owner.begin(3, [(state, 0, 3)])
    for index in owner.before:
        capture.before_layer(index, h)
    batch = capture.finish()
    state.cur = [0]
    state._kv_error = RuntimeError("target failed after feature production")
    with pytest.raises(ValueError, match="target storage"):
        _ = batch.features
    batch.release()
    owner.close()
    gc.collect()
    torch.cuda.synchronize()
    assert storage() is None and owner.nbytes == 0
    assert torch.cuda.memory_allocated() <= before


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("cache", ["fp16", "int8"])
def test_native_target_forward_hook_preserves_logits_and_committed_state(prefill, cache, monkeypatch):
    w = _small_weights(True, hidden=2560, lowrank=320, head_dim=256)
    w.layers[1].ple, w.layers[0].ple = w.layers[0].ple, None
    w.cfg.ple_layers = [1]
    rows = 3 if prefill else 6
    states = [State(w, 32, 8, cache) for _ in range(2)]
    buffers = [Buffers(w, 8, 32, prefill=prefill) for _ in range(2)]
    owner = TapWorkspace(config(before=(1,)), rows, w.device)
    original_layer = target.layer_forward
    outputs = []
    for enabled, state, buf in zip((False, True), states, buffers):
        buf.ids[:rows].copy_(torch.arange(rows, dtype=torch.int32, device=w.device) + 3)
        buf.ple_v.fill_(.125)
        capture = owner.begin(rows, [(state, 0, rows)]) if enabled else None

        def hook(layer, weights, segs, b, count, pending, **kwargs):
            if enabled:
                before, sums = b.h[:count].clone(), b.pss[:count].clone()
                capture.before_layer(layer.index, b.h[:count], pending)
                exact(b.h[:count], before)
                exact(b.pss[:count], sums)
            return original_layer(layer, weights, segs, b, count, pending, **kwargs)

        with monkeypatch.context() as hooks:
            if enabled:
                hooks.setattr(target, "layer_forward", hook)
            outputs.append(target.compute(w, [(state, 0, rows)], buf).clone())
        if enabled:
            batch = capture.finish()
            assert batch.features.shape == (rows, 2560)
            batch.release()
        target.commit(w, state, buf, rows, rows if prefill else 2)
    exact(*outputs)
    exact(buffers[0].streams[:rows], buffers[1].streams[:rows])
    for name in ("rec", "conv", "ple_tail"):
        exact(getattr(states[0], name), getattr(states[1], name))
    assert states[0].pos == states[1].pos and states[0].cur == states[1].cur
    for ka, kb in zip(states[0].kc, states[1].kc):
        for name in ("k", "v", "ks", "vs"):
            a, b = getattr(ka, name)[:states[0].pos], getattr(kb, name)[:states[1].pos]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    owner.close()


@pytest.mark.parametrize('alias', ['cuda', torch.device('cuda')])
def test_unindexed_cuda_workspace_device_matches_allocated_hc_storage(alias):
    owner = TapWorkspace(config(hidden=256, before=(1,)), 1, alias)
    h = torch.ones((1, 1024), dtype=torch.float16, device=alias)
    assert owner.device == h.device == owner.features.device
    capture = owner.begin(1)
    capture.before_layer(1, h)
    batch = capture.finish()
    exact(batch.features, torch.ones((1, 256), dtype=torch.float16, device=h.device))
    batch.release()
    owner.close()
