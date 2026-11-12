"""Value8 registered-batch ownership/geometry controls, CPU metadata only."""
import gc
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from tensorfold.cuda.sm70 import gdn_value8
from tensorfold.families.qwen4_exp.cuda.gdn import GDNScratch
from tensorfold.families.qwen4_exp.cuda.gdn_multi import RoundTables


def setup(monkeypatch, *, slots=2, width=5, layers=1, inference=False):
    actor = object()
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: actor)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    with torch.inference_mode(inference):
        static = RoundTables(NS(device='cpu'), NS(lin=layers), slots, width, native_half=True)
        b = NS(rows=slots*width, h=torch.empty((slots*width, 10240), dtype=torch.float16),
            proj=torch.empty((layers, slots*width, 2060), dtype=torch.float16),
            gout=torch.empty((slots*width, 768), dtype=torch.float16),
            gxs=torch.empty((slots*width, 24)), prefill=False, split=None, owner_graph_family='uncaptured')
        states = [NS(conv=torch.zeros((layers, 3, 1280), dtype=torch.float16),
            rec=torch.zeros((2, layers, 6, 128, 128)), cur=[index % 2]*layers,
            scratch=[GDNScratch(width, 'cpu', 2, 6, dtype=torch.float16) for _ in range(layers)])
            for index in range(slots)]
        static.bind_native(states, b)
        workspace = gdn_value8.allocate_workspace(b.rows, 'cpu', device_budget_bytes=1024*b.rows)
    calls = []
    kernel = gdn_value8.Prepared(NS(batch_chain=lambda *args: calls.append(args)), torch.device('cpu'), True)
    return static, b, states, workspace, kernel, calls


def plan(static, b, states, workspace, kernel, *, segs=None):
    segs = [(states[0], 0, 5)] if segs is None else segs
    frame = static.native_frame(segs, b, allow_single=True)
    rows = frame.rows
    return kernel.batch_prepare(frame, 0, b.proj[0, :rows], b.gout[:rows], b.gxs[:rows], workspace), frame


@pytest.mark.parametrize('rows,count,width,bound,preload', [
    (1, 1, 5, 1, 1), (5, 1, 5, 5, 5), (6, 2, 5, 5, 5),
    (5, 2, 3, 3, 4), (24, 24, 5, 1, 1), (120, 24, 5, 5, 5)])
def test_launch_bound_depends_only_on_exact_key_and_registered_width(rows, count, width, bound, preload):
    facts = gdn_value8.batch_layout(rows, count, width)
    assert facts['max_window_rows'] == bound and facts['preload_rows'] == preload
    assert facts['core_grid'] == [16, 6, count]
    assert facts['front_grid'] == [2, bound, count] and facts['back_grid'] == [6, bound, count]
    assert facts['minimum_active_q_bytes'] == rows*1024 and facts['q_bytes_per_capacity_row'] == 1024


def test_new_batch_resource_facts_are_directly_publishable_on_strict_control_channel():
    import json
    from tensorfold.engine.model_identity import canonical_runtime_json

    facts = gdn_value8.batch_layout(120, 24, 5)
    assert json.loads(canonical_runtime_json(facts, max_bytes=4096)) == facts


@pytest.mark.parametrize('args', [(0, 0, 5), (1, 0, 5), (121, 24, 5), (25, 25, 1),
                                  (1, 1, 6), (1, True, 5), (1., 1, 5)])
def test_empty_and_unsupported_batch_geometry_refused(args):
    with pytest.raises(ValueError, match='nonempty bounded'):
        gdn_value8.batch_layout(*args)


def test_fresh_frame_changes_owners_parity_and_ragged_starts_without_pinning_cold_frame(monkeypatch):
    static, b, states, workspace, kernel, _ = setup(monkeypatch)
    initial = [(states[0], 0, 1), (states[1], 1, 6)]
    bound, frame = plan(static, b, states, workspace, kernel, segs=initial)
    assert bound.max_window_rows == 5
    ref = weakref.ref(frame)
    for state in states:
        state.cur[0] ^= 1
    fresh = static.native_frame([(states[1], 0, 3), (states[0], 3, 6)], b)
    del frame
    gc.collect()
    assert ref() is None
    bound.validate_frame(fresh)
    assert bound.starts.tolist() == [0, 3, 6]
    expected = states[1].rec[states[1].cur[0], 0].data_ptr()
    assert bound.pointers[1, 0].item() == expected
    assert bound.static is static and bound.buffers is b and bound.workspace is workspace


@pytest.mark.parametrize('changed', ['parity', 'width', 'R', 'State-storage', 'Q-storage'])
def test_replay_validation_refuses_stale_or_rebound_storage(monkeypatch, changed):
    static, b, states, workspace, kernel, _ = setup(monkeypatch)
    bound, frame = plan(static, b, states, workspace, kernel)
    if changed == 'parity':
        states[0].cur[0] ^= 1
    elif changed == 'width':
        static.width = 4
    elif changed == 'R':
        frame = static.native_frame([(states[0], 0, 4)], b, allow_single=True)
    elif changed == 'State-storage':
        states[0].rec = states[0].rec.clone()
    else:
        workspace.q.set_(torch.empty_like(workspace.q))
    with pytest.raises(ValueError):
        bound.validate_frame(frame)


@pytest.mark.parametrize('alias', ['unselected-State-save', 'projection', 'residual', 'descriptor'])
def test_extra_Q_credit_never_authorizes_aliasing_registered_inference_storage(monkeypatch, alias):
    static, b, states, workspace, kernel, calls = setup(monkeypatch, slots=4, layers=4)
    rows = 1
    if alias == 'unselected-State-save':
        q = states[1].scratch[0].k[:1]
    elif alias == 'projection':
        q = b.proj.view(torch.float32).reshape(-1)[:256].view(1, 2, 128)
    elif alias == 'residual':
        q = b.h.view(torch.float32).reshape(-1)[:256].view(1, 2, 128)
    else:
        q = static.i64.view(torch.float32).reshape(1, 2, 128)
    frame = static.native_frame([(states[0], 0, rows)], b, allow_single=True)
    with pytest.raises(ValueError, match='Q workspace overlaps'):
        kernel.batch_prepare(frame, 0, b.proj[0, :rows], b.gout[:rows], b.gxs[:rows],
                             gdn_value8.Workspace(q, 1))
    assert not calls


def test_cold_binding_requires_prior_residency_and_current_frame(monkeypatch):
    static, b, states, workspace, kernel, _ = setup(monkeypatch)
    frame = static.native_frame([(states[0], 0, 5)], b, allow_single=True)
    unprepared = gdn_value8.Prepared(kernel.module, kernel.device)
    with pytest.raises(RuntimeError, match='resident before admission'):
        unprepared.batch_prepare(frame, 0, b.proj[0, :5], b.gout[:5], b.gxs[:5], workspace)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='outside capture'):
        kernel.batch_prepare(frame, 0, b.proj[0, :5], b.gout[:5], b.gxs[:5], workspace)


def test_cold_binding_retains_already_funded_workspace_without_allocating_or_republishing(monkeypatch):
    static, b, states, workspace, kernel, _ = setup(monkeypatch)
    frame = static.native_frame([(states[0], 0, 5)], b, allow_single=True)
    b.owned_extra_q = workspace.q
    pointer_words, starts = static.i64.clone(), static.i32.clone()
    monkeypatch.setattr(torch, 'empty', lambda *args, **kwargs: pytest.fail('binding allocation'))
    bound = kernel.batch_prepare(frame, 0, b.proj[0, :5], b.gout[:5], b.gxs[:5], workspace)
    assert bound.workspace is workspace
    assert torch.equal(static.i64, pointer_words) and torch.equal(static.i32, starts)


@pytest.mark.parametrize('negative', [False, True])
def test_side_submission_uses_static_views_with_no_actor_check_or_workspace_allocation(monkeypatch, negative):
    static, b, states, workspace, kernel, calls = setup(monkeypatch)
    bound, frame = plan(static, b, states, workspace, kernel)
    weights = (torch.empty((1280, 4), dtype=torch.float16), torch.empty(6),
               torch.empty(6), torch.empty(128, dtype=torch.float16))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: pytest.fail('side submission actor check'))
    monkeypatch.setattr(torch, 'empty', lambda *args, **kwargs: pytest.fail('execution allocation'))
    assert bound.chain_prepared(*weights, 1e-6, negative_decay=negative) is bound.output
    assert len(calls) == 1 and calls[0][6] is bound.pointers and calls[0][7] is bound.starts
    assert calls[0][11] is workspace.q and calls[0][12] is negative


def test_inference_mutation_domain_is_preserved_for_indirect_registered_outputs(monkeypatch):
    static, b, states, workspace, kernel, calls = setup(monkeypatch, inference=True)
    with torch.inference_mode():
        bound, frame = plan(static, b, states, workspace, kernel)
    weights = (torch.empty((1280, 4), dtype=torch.float16), torch.empty(6), torch.empty(6), torch.empty(128, dtype=torch.float16))
    with pytest.raises(RuntimeError, match='outside inference mode'):
        bound.chain_prepared(*weights, 1e-6)
    assert not calls
    with torch.inference_mode():
        bound.chain_prepared(*weights, 1e-6)
    assert len(calls) == 1
