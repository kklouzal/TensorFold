"""Value8 descriptor batch against unchanged native batch and scalar oracles.

This component gate requires a genuine installed SDK with the current
NativeTables ABI. It does not qualify an Engine/ExactStateGraphs integration.
"""
from types import SimpleNamespace as NS
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.sm70 import gdn_native, gdn_value8
from tests.cuda import test_sm70_gdn_batch as incumbent_controls

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='actual SM70 and genuine current descriptor ABI required')


@pytest.fixture(scope='module')
def component():
    old = gdn_native.extension()
    old.prepare_batch(6)
    candidate = gdn_value8.prepare('cuda', batch=True)
    assert hashlib.sha256(Path(candidate.module.__file__).read_bytes()).digest() != hashlib.sha256(Path(old.__file__).read_bytes()).digest()
    return old, candidate


def identity(old, candidate):
    return {'scope': 'worktree component only; no main Source16/model/P3 claim',
        'candidate': candidate.module.__file__, 'candidate_sha256': hashlib.sha256(Path(candidate.module.__file__).read_bytes()).hexdigest(),
        'incumbent': old.__file__, 'incumbent_sha256': hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),
        'candidate_source': gdn_value8.__file__}


def frame(static, b, states, rows, count, iteration):
    chosen = list(reversed(states))[:count] if iteration else states[:count]
    lengths = ([1, 5] if iteration == 0 else [3, 3]) if (rows, count) == (6, 2) else [rows//count]*count
    end, segs = 0, []
    for state, size in zip(chosen, lengths):
        segs.append((state, end, end+size))
        end += size
    assert end == rows
    return static.native_frame(segs, b, allow_single=True)


def plans(candidate, table, b, workspace):
    return [candidate.batch_prepare(table, layer, b.proj[layer, :table.rows],
                b.gout[:table.rows], b.gxs[:table.rows], workspace) for layer in range(table.static.lin)]


@pytest.mark.parametrize('rows,count', [(1, 1), (5, 1), (6, 2), (120, 24)])
@pytest.mark.parametrize('pattern', ['random', 'masked'])
@pytest.mark.parametrize('negative', [False, True])
def test_actual_batch_and_scalar_bits_saved_state_ragged_owners_parity_masks(component, rows, count, pattern, negative, record_property):
    old, candidate = component
    static, states, b, weight, norm, bias, log, neg = incumbent_controls.setup(6, pattern, width=5)
    workspace = gdn_value8.allocate_workspace(b.rows, 'cuda', device_budget_bytes=1024*b.rows)
    workspace.q.fill_(-91)
    parameter = neg if negative else log
    original_conv = [state.conv.clone() for state in states]
    table = frame(static, b, states, rows, count, 0)
    bound = plans(candidate, table, b, workspace)
    for iteration in range(2):
        if iteration:
            for state in states:
                state.cur = [1-value for value in state.cur]
            b.proj.mul_(-1)
            if pattern == 'masked':
                for index, state in enumerate(states):
                    for saved in state.scratch:
                        saved.keep_rows.fill_((index+iteration)%6)
            table = frame(static, b, states, rows, count, iteration)
        for layer, prepared in enumerate(bound):
            prepared.validate_frame(table)
            expected = incumbent_controls.oracle(old, table, b, layer, weight, norm, bias, parameter, negative)
            pointers, starts = table.operands(layer, b.proj[layer, :rows], b.gout[:rows], b.gxs[:rows])
            old.batch_chain(b.proj[layer, :rows], weight, parameter, bias, norm, 1e-6,
                            pointers, starts, b.gout[:rows], b.gxs[:rows], negative)
            incumbent_controls.compare(table, b, layer, expected)
            prepared.chain_prepared(weight, parameter, bias, norm, 1e-6, negative_decay=negative)
            incumbent_controls.compare(table, b, layer, expected)
        incumbent_controls.exact(workspace.q[rows:], torch.full_like(workspace.q[rows:], -91))
    for state, before in zip(states, original_conv):
        incumbent_controls.exact(state.conv, before)
    record_property('value8_batch_component', json.dumps({**identity(old, candidate),
        'rows': rows, 'participants': count, 'pattern': pattern, 'negative_decay': negative,
        'preload': bound[0].max_window_rows, 'Q_capacity_bytes': workspace.q.numel()*workspace.q.element_size()}))


@pytest.mark.parametrize('count', [1, 24])
@pytest.mark.parametrize('keep', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('negative', [False, True])
def test_side_capture_changed_actor_frames_and_every_MTP4_accepted_prefix(component, count, keep, negative, record_property):
    from tensorfold.families.qwen4_exp.cuda.forward import commit

    old, candidate = component
    static, states, b, weight, norm, bias, log, neg = incumbent_controls.setup(6, 'random', width=5)
    for state in states:
        state.kv_check = lambda: None
        state.ple_last, state.pos = None, 0
        state.set_pos = lambda value, state=state: setattr(state, 'pos', value)
    weights = NS(cfg=NS(conv_dim=1280))
    rows = 5*count
    workspace = gdn_value8.allocate_workspace(b.rows, 'cuda', device_budget_bytes=1024*b.rows)
    workspace.q.fill_(-91)
    table = frame(static, b, states, rows, count, 0)
    bound = plans(candidate, table, b, workspace)
    parameter = neg if negative else log
    for prepared in bound:
        prepared.chain_prepared(weight, parameter, bias, norm, 1e-6, negative_decay=negative)
    torch.cuda.synchronize()
    side, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    primary, cleanup = None, [None, None]
    try:
        with torch.cuda.graph(graph, stream=side):
            for prepared in bound:
                prepared.chain_prepared(weight, parameter, bias, norm, 1e-6, negative_decay=negative)
        torch.cuda.synchronize()
        for iteration in range(2):
            table = frame(static, b, states, rows, count, iteration)
            for prepared in bound:
                prepared.validate_frame(table)
            expected = [incumbent_controls.oracle(old, table, b, layer, weight, norm, bias, parameter, negative)
                        for layer in range(static.lin)]
            conv = {id(state): state.conv.clone() for state, _, _ in table.segs}
            prefix = {}
            for state, _, _ in table.segs:
                values = []
                for layer in range(static.lin):
                    final = torch.empty_like(state.rec[0, layer])
                    if keep == 5:
                        final.copy_(expected[layer][0][id(state)][1-state.cur[layer], layer])
                    else:
                        saved = expected[layer][1][id(state)]
                        old.replay(state.rec[state.cur[layer], layer], saved.k, saved.v, saved.g, saved.b, keep, final)
                    values.append(final)
                prefix[id(state)] = values
            graph.replay()
            incumbent_controls.exact(b.gout, expected[-1][2])
            incumbent_controls.exact(b.gxs, expected[-1][3])
            for state, begin, _ in table.segs:
                before = state.cur[:]
                for layer in range(static.lin):
                    incumbent_controls.exact(state.rec[:, layer], expected[layer][0][id(state)][:, layer])
                    for name in ('k', 'v', 'g', 'b'):
                        incumbent_controls.exact(getattr(state.scratch[layer], name), getattr(expected[layer][1][id(state)], name))
                commit(weights, state, b, 5, keep, at=begin)
                assert state.cur == [1-value for value in before]
                for layer in range(static.lin):
                    incumbent_controls.exact(state.rec[state.cur[layer], layer], prefix[id(state)][layer])
                expected_conv = torch.cat((conv[id(state)], b.proj[:, begin:begin+keep, :1280]), dim=1)[:, -3:]
                incumbent_controls.exact(state.conv, expected_conv)
            incumbent_controls.exact(workspace.q[rows:], torch.full_like(workspace.q[rows:], -91))
            b.proj.mul_(-1)
    except BaseException as error:
        primary = error
    finally:
        try:
            torch.cuda.synchronize()
        except BaseException as error:
            cleanup[0] = error
        if cleanup[0] is None:
            try:
                graph.reset()
            except BaseException as error:
                cleanup[1] = error
        raise_failures(primary, [error for error in cleanup if error is not None])
    record_property('value8_batch_component', json.dumps({**identity(old, candidate),
        'participants': count, 'keep': keep, 'negative_decay': negative, 'changed_replays': 2}))


@pytest.mark.parametrize('operand', ['weight', 'norm'])
def test_cpp_rejects_Q_weight_read_alias_before_any_kernel_mutation(component, operand):
    old, candidate = component
    static, states, b, weight, norm, bias, _, neg = incumbent_controls.setup(6, 'random', width=5)
    workspace = gdn_value8.allocate_workspace(b.rows, 'cuda', device_budget_bytes=1024*b.rows)
    workspace.q.fill_(-91)
    table = frame(static, b, states, 120, 24, 0)
    bound = plans(candidate, table, b, workspace)[0]
    if operand == 'weight':
        weight = workspace.q.view(torch.float16).reshape(-1)[:5120].view(1280, 4)
    else:
        norm = workspace.q.view(torch.float16).reshape(-1)[:128]
    before = workspace.q.clone()
    with pytest.raises(RuntimeError, match='Q aliases'):
        bound.chain_prepared(weight, neg, bias, norm, 1e-6, negative_decay=True)
    incumbent_controls.exact(workspace.q, before)
