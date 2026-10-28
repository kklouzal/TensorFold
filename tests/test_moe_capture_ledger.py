"""Cold raw evidence remains durable without inflating bounded ready replies."""
import hashlib
import json
from types import SimpleNamespace as N

import pytest

from tools import v100_backend_probe as probe


def test_complete_phase_ledger_is_external_and_ready_descriptor_small(tmp_path):
    gates=[]
    rows=[{'key':['moe','target',30,i],'charged_bytes':i,'physical_delta_bytes_raw':i-1} for i in range(145)]
    catalogue=N(ready=True,measured=rows,plan={'future_host_bytes_per_rank':1<<30},
                host_before={'VmRSS':1,'VmHWM':2},host_after={'VmRSS':3,'VmHWM':4},
                _host_gate=lambda:gates.append('checked'))
    engine=N(multi=N(graphs=N(moe_regions=catalogue,private_pool_records=[{'family':'n1'}])))
    descriptor=probe.write_moe_capture_ledger(engine,tmp_path,0)
    path=tmp_path/'rank-0-moe-capture-ledger.json'
    raw=path.read_bytes()
    assert json.loads(raw)['capture_phases']==rows
    assert descriptor['bytes']==len(raw) and descriptor['sha256']==hashlib.sha256(raw).hexdigest()
    assert len(json.dumps(descriptor))<1024 and descriptor['phase_records']==145 and gates==['checked']
    with pytest.raises(FileExistsError):
        probe.write_moe_capture_ledger(engine,tmp_path,0)  # Preserve historical evidence.


def test_incomplete_cold_ledger_refuses_without_ready_descriptor(tmp_path):
    engine=N(multi=N(graphs=N(moe_regions=N(ready=True,measured=[]))))
    with pytest.raises(ValueError,match='complete145-phase'):
        probe.write_moe_capture_ledger(engine,tmp_path,7)
    assert not list(tmp_path.iterdir())
    assert probe.write_moe_capture_ledger(N(multi=N(graphs=None)),tmp_path,1) is None


def test_actual_export_write_and_close_fault_preserve_named_primary(tmp_path,monkeypatch):
    primary,secondary=OSError('ledger-write'),OSError('ledger-close')
    prior=ValueError('native-write-root')
    primary.__cause__=prior
    log=[]
    class File:
        def write(self,encoded):
            log.append('write')
            raise primary
        def close(self):
            log.append('close')
            raise secondary
    def opened(path,mode):
        assert mode=='xb'
        return File()
    monkeypatch.setattr(probe.Path,'open',opened)
    catalogue=N(ready=True,measured=[{'key':i} for i in range(145)],plan={},host_before={},host_after={},
                _host_gate=lambda:log.append('unexpected-gate'))
    engine=N(multi=N(graphs=N(moe_regions=catalogue,private_pool_records=[])))
    with pytest.raises(OSError) as caught:
        probe.write_moe_capture_ledger(engine,tmp_path,0)
    assert caught.value is primary
    def leaves(error):
        return sum((leaves(item) for item in error.exceptions),[]) if isinstance(error,BaseExceptionGroup) else [error]
    assert {id(value) for value in leaves(primary.__cause__)}=={id(prior),id(secondary)}
    assert log==['write','close']


def test_candidate_ledger_is_separate64_phase_artifact(tmp_path):
    rows=[{'key':['candidate',role,r],'charged_bytes':0} for role in ('target','mtp') for r in range(1,33)]
    owner=N(ready=True,measured=rows,plan={},host_before={},host_after={},_host_gate=lambda:None)
    result=probe.write_candidate_capture_ledger(N(multi=N(graphs=N(candidate_regions=owner))),tmp_path,2)
    assert result['phase_records']==64
    raw=json.loads((tmp_path/'rank-2-candidate-capture-ledger.json').read_text())
    assert raw['schema']=='tensorfold.cold-candidate-capture-ledger-v1'
    assert raw['capture_phases']==rows
