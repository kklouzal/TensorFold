"""Real CUDA current-module witness controls, not full-model quality/P3."""
import json

from tensorfold.cleanup import raise_failures
from types import SimpleNamespace as NS

import pytest
import torch

from tests.cuda.test_sm70_gdn_batch import setup
from tensorfold.cuda.sm70 import gdn_native
from tools import v100_owned_qualification as owned

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='actual SM70 diagnostic witness required')


@pytest.mark.parametrize('wrong_bit', [False, True])
def test_live_scalar_before_native_batch_raw_state_and_evidence(tmp_path, monkeypatch, wrong_bit):
    gdn_native.extension().prepare_batch(6)
    static, states, b, weight, norm, bias, _, decay = setup(6, 'masked')
    for i, state in enumerate(states):
        state.history_slot, state.slot_generation, state.history_generation, state.pos = i, 1, 2, 64
    # Explicit synthetic caller-owned unused scratch; no model/State mirror.
    shared = NS(k=torch.empty(6*128*128, device='cuda'), q=torch.empty(280, device='cuda'),
                v=torch.empty(1536, device='cuda', dtype=torch.float16),
                g=torch.empty(6, device='cuda'), beta=torch.empty(6, device='cuda'), lin=2)
    dec = NS(gdn=shared, buf=b, held={}, states=states, full_resident=True, round_tables=static,
             speculation=NS(method='none', maximum_proposals=0))
    engine = NS(rank=0, w=NS(device=b.h.device, mtp=None, cfg=NS(nk=2, nv=6)),
                multi=dec, _closed=False, _abort_comm=False, capacity_plan={'native_gdn_batch': {'enabled': True}})
    def close():
        torch.cuda.synchronize()
        engine._closed = True
        return 'retired'
    engine.close = close
    failures = []
    monkeypatch.setattr(owned, '_observer_failure', lambda e, error: failures.append(error))
    original = gdn_native.batch_chain
    def faulty(*args, **kwargs):
        result = original(*args, **kwargs)
        args[8].view(torch.int16)[0, 0] ^= 1
        return result
    if wrong_bit:
        monkeypatch.setattr(gdn_native, 'batch_chain', faulty)
    resources = owned._gdn_batch_resources({'cohorts': [{'id': 6, 'level': 24, 'warmup': False,
        'input_tokens': 64, 'tokens': 2}]}, tmp_path)
    witness = owned._GDNBatchParity(engine, resources, tmp_path)
    streams = [NS(st=s) for s in states]
    dec.streams = {i: stream for i, stream in enumerate(streams)}
    engine.tp_decoder = NS(wires={id(stream): i for i, stream in enumerate(streams)})
    witness.observer = NS(finished=True, _identity=lambda stream: ({'id': 6},
        {'index': stream.st.history_slot, 'prompt_sha256': 'a'*64}))
    witness.install()
    segs = [(s, i, i+1) for i, s in enumerate(states)]
    table = static.native_frame(segs, b)
    inputs = [s.rec[s.cur[0], 0].clone() for s in states]
    args = (b.proj[0, :24], weight, decay, bias, norm, 1e-6, table, 0, b.gout[:24], b.gxs[:24])
    primary, cleanup = None, [None]
    try:
        if wrong_bit:
            with pytest.raises(RuntimeError, match='raw-bit mismatch') as caught:
                gdn_native.batch_chain(*args, negative_decay=True)
            assert failures == [caught.value]
            proof = json.loads((witness.directory/'mismatch.json').read_text())
            assert len(proof['files']) == 23 and proof['rows'] == 24
            actual = __import__('numpy').load(witness.directory/'actual_out.npy', allow_pickle=False)
            expected = __import__('numpy').load(witness.directory/'expected_out.npy', allow_pickle=False)
            assert actual.view('uint16')[0, 0] != expected.view('uint16')[0, 0]
            assert witness.failed
        else:
            assert gdn_native.batch_chain(*args, negative_decay=True) is None
            second = (b.proj[1, :24], weight, decay, bias, norm, 1e-6, table, 1, b.gout[:24], b.gxs[:24])
            assert gdn_native.batch_chain(*second, negative_decay=True) is None
            assert witness.counts[0]['matched'] == witness.counts[1]['matched'] == 24 and not failures
        for state, before in zip(states, inputs):
            assert torch.equal(state.rec[state.cur[0], 0].view(torch.int32), before.view(torch.int32))
    except BaseException as error:
        primary = error
    finally:
        try:
            engine.close()
        except BaseException as error:
            if error is not witness.primary:
                cleanup[0] = error
        if cleanup[0] is not None:
            raise_failures(primary, cleanup)
        if primary is not None:
            raise_failures(primary, ())
    assert witness.closed and witness.engine is None and not witness.slabs
