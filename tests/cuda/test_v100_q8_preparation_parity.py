"""Synthetic actual-CUDA live-input witness controls; not a model/P3 run."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tools import v100_owned_qualification as owned
from tensorfold.cuda.sm70 import family_ops as ops
from tests.cuda.test_sm70_owned_q8_preparation import fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='Q8 witness requires actual SM70')


def setup(monkeypatch, tmp_path, *, corrupt=False):
    operands = [fixture(64 + index % 4, owned_pages={0}) for index in range(24)]
    weights, attn = operands[0][3], operands[0][2].attn
    weights.layers = [NS(index=i, attn=attn if i % 4 == 3 else None) for i in range(48)]
    states = [item[1] for item in operands]
    for index, state in enumerate(states):
        state.history_slot = index
    projection = torch.cat([item[0] for item in operands])
    coordinates = torch.cat([item[4].target_indexer_coordinates for item in operands])
    weights.device = projection.device
    native = ops.prepare_owned_q8(projection.device)
    buf = NS(pa=projection, target_indexer_coordinates=coordinates,
             q=torch.zeros((24, 3, 256), device=projection.device, dtype=torch.float16),
             iq=torch.zeros((24, 4, 128), device=projection.device, dtype=torch.float16),
             history_owner=NS(owned_q8_preparation=native),
             gdn_tables=NS(segs=[(state, i, i+1) for i, state in enumerate(states)]))
    streams = {i: NS(st=state, done=False, out=[1, 2], index=i) for i, state in enumerate(states)}
    decoder = NS(buf=buf, states=states, streams=streams, filling=[],
                 gdn=NS(q=torch.zeros((24, 2, 128), device=projection.device)))
    engine = NS(rank=0, w=weights, multi=decoder, _closed=False,
                tp_decoder=NS(wires={id(stream): i for i, stream in streams.items()}))
    def close():
        engine._closed = True
        return 'actual-synthetic-retirement'
    engine.close = close
    requests = [{'index': i, 'prompt_sha256': f'{i:064x}'} for i in range(24)]
    cohort = {'id': 6, 'level': 24, 'warmup': False, 'input_tokens': 64, 'tokens': 32, 'requests': requests}
    plan = {'profile': 'q8', 'cohorts': [cohort]}
    resources = owned._q8_parity_resources(plan, tmp_path)
    resources['input_sha256'] = 'b'*64
    observer = NS(current=6, finished=True, failed=None,
                  _identity=lambda stream: (cohort, requests[stream.index]))
    monkeypatch.setattr(owned, '_q8_witness_region', lambda engine: engine.multi)
    notices = []
    monkeypatch.setattr(owned, '_observer_failure', lambda engine, error: notices.append(error))
    module = None
    original = ops.owned_q8_prepare
    if corrupt:
        def wrong(*args, **kwargs):
            result = original(*args, **kwargs)
            args[7].view(torch.int16).view(-1)[0].bitwise_xor_(1)
            return result
        module = NS(owned_q8_prepare=wrong, __file__=__file__)
    witness = owned._Q8PreparationParity(engine, resources, tmp_path, module=module)
    witness.observer = observer
    witness.install()
    def call(index, layer):
        state = states[index]
        function = ops.owned_q8_prepare if module is None else module.owned_q8_prepare
        return function(native, projection[index:index+1], coordinates[index:index+1], weights,
            state.kc[0], state, layer, buf.q[index:index+1], buf.iq[index:index+1],
            frequency=weights.inv_freq, sections=(11, 11, 10), amplitude=1., start=state.pos)
    return NS(witness=witness, engine=engine, observer=observer, call=call, notices=notices,
              original=original, states=states, buf=buf)


@torch.no_grad()
def test_actual_cuda_reference_before_native_one24state_round_all12layers_and_retirement(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    before = [state.history_generations.clone() for state in f.states]
    f.witness.begin_round(f.observer)
    for layer in f.engine.w.layers:
        if layer.attn is not None:
            for index in range(24):
                result = f.call(index, layer)
                assert result.dtype == torch.float16 and result.shape == (1, 128)
    f.witness.end_round()
    assert f.witness.calls == f.witness.matches == 288 and f.witness.completed
    for state, generation in zip(f.states, before):
        assert torch.equal(state.history_generations, generation)
        assert state.kv_status.cpu().tolist() == [0, -2]
    assert f.engine.close() == 'actual-synthetic-retirement'
    summary = json.loads((f.witness.directory/'summary.json').read_text())
    assert summary['closed'] and not summary['failed'] and summary['matched'] == 288
    assert not summary['P3_valid'] and not f.witness.views and not f.witness.slabs
    assert ops.owned_q8_prepare is f.original and not f.notices


@torch.no_grad()
def test_actual_cuda_bit_mismatch_preserves_initial_status_and_bounded_raw_npy_before_abort(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path, corrupt=True)
    f.states[0].kv_status.copy_(torch.tensor([8, 2], device=f.engine.w.device, dtype=torch.int32))
    f.witness.begin_round(f.observer)
    with pytest.raises(RuntimeError) as caught:
        f.call(0, f.engine.w.layers[3])
    assert caught.value is f.witness.mismatch and f.notices == [f.witness.mismatch]
    record = json.loads((f.witness.directory/'first-mismatch.json').read_text())
    assert record['rank'] == 0 and record['layer'] == 3 and record['position'] == 64
    assert record['wire'] == 0 and record['owned'] and not record['P3_valid']
    assert np.load(f.witness.directory/'initial_status.npy').tolist() == [8, 2]
    actual = np.load(f.witness.directory/'actual_q.npy').view(np.uint16)
    expected = np.load(f.witness.directory/'expected_q.npy').view(np.uint16)
    assert np.count_nonzero(actual != expected) == 1
    assert set(record['files']) == {p.stem for p in f.witness.directory.glob('*.npy')}
    f.witness.end_round(f.witness.mismatch)
    with pytest.raises(RuntimeError) as caught:
        f.engine.close()
    assert caught.value is f.witness.mismatch
    assert not f.witness.views and not f.witness.slabs
