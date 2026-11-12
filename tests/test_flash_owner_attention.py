"""Paid fixed-P owner scheduling and bounded native ownership, CPU mocks only."""
from types import SimpleNamespace as NS
import threading
import pytest
import torch
from tensorfold.cuda.sm70 import attention_backends as backend, flash_v100_forward as primitive, paged_attention as native


def test_complete_role_forecast_counts_single_private_workspace():
    f = backend.flash_owner_requirements(5, 3)
    assert f['native_private_bytes'] == primitive.compacted_workspace_bytes(3)
    assert f['persistent_bytes'] == sum(v for k, v in f.items() if k.endswith('_bytes') and k not in
                                       ('persistent_bytes', 'operator_temporary_bytes'))
    assert f['operator_temporary_bytes'] == 0 and f['panel_rows'] == 3


def frame_owner(rows=5):
    queries = torch.zeros(8 * rows * 3 * 256).half()
    owner = NS(rank=0, queries=queries, native_page_tokens=128, native_layer=torch.tensor([3], dtype=torch.int32),
               native_buckets=(1, 2, 5))
    def views(n):
        return dict(q_all=queries[:8*n*3*256].view(8,n,3,256), row_table=torch.zeros(n,16,dtype=torch.int64),
            selected_all=torch.zeros(8,n,12,dtype=torch.int32), local_stats=torch.empty(n,12,2),
            all_stats=torch.empty(8,n,12,2), global_stats=torch.empty(n,12,2),
            local_values=torch.empty(n,12,256), all_values=torch.empty(8,n,12,256), out=torch.empty(n,3,256).half())
    owner.native_views = views
    return owner


def test_one_fixed_panel_owner_active_prefix_and_ghosts_are_cold(monkeypatch):
    calls, native_owners = [], []
    class Artifact:
        module = object()
        def loan(self): calls.append('loan'); return object()
        def release(self, token): calls.append('release')
        def record(self): return {'profile_sha256': 'a'*64}
    class Prepared:
        def __init__(self, q,k,v,valid,out,lse): self.args=(q,k,v,valid,out,lse)
        def settings(self): return {'compacted': True}
        def run(self,q,k,v,valid,out,lse):
            assert tuple(q.shape)==(36,1,256) and tuple(valid.shape)==(3,1,2051)
            active=valid.any(-1)[:,0]
            assert torch.equal(q.view(3,12,256)[~active], torch.zeros_like(q.view(3,12,256)[~active]))
            assert torch.equal(q.view(3,12,256)[active], torch.full_like(q.view(3,12,256)[active],7))
            out.view(3,12,256).copy_(active[:,None,None].expand(3,12,256).half())
            lse.view(3,12).copy_(torch.where(active[:,None], 0., -torch.inf).expand(3,12))
            calls.append(('run',int(active.sum())))
        def close(self): calls.append('close')
        def handoff_to_current_thread(self): calls.append('handoff')
    def prepare(module,*args,**kw):
        result=Prepared(*args); native_owners.append(result); return result
    monkeypatch.setattr(primitive,'prepare_compacted',prepare)
    monkeypatch.setattr(native,'pack_owner_queries',lambda q,table,out,**kw: out.fill_(7))
    def gather(selected,table,key,value,valid,**kw):
        valid.zero_(); valid[:,:,0]=True; key.zero_(); value.fill_(1)
        calls.append(('gather',kw['first'],len(key)))
    monkeypatch.setattr(native,'gather_owner_q8',gather)
    monkeypatch.setattr(backend,'_merge_normalized_owner',lambda owner,views,norm,lse: norm)
    owner=frame_owner(); artifact=Artifact()
    needed=backend.flash_owner_requirements(5,3)['persistent_bytes']
    with pytest.raises(MemoryError):
        backend.FlashOwnerTiles(5,3,'cpu',artifact,tensor_budget_bytes=needed-1,native_owner=owner)
    assert calls==[]
    tile=backend.FlashOwnerTiles(5,3,'cpu',artifact,tensor_budget_bytes=needed,native_owner=owner)
    # Every runtime view was sealed before serving; one native geometry only.
    owner.native_views=lambda n: pytest.fail('runtime frame construction')
    for rows in (5,2,1,5): assert torch.equal(tile.run(owner,rows),torch.ones(rows,12,256).half())
    assert len(native_owners)==1 and [c for c in calls if isinstance(c,tuple) and c[0]=='gather'][:2]==[
        ('gather',0,3),('gather',3,2)]
    errors=[]
    def other():
        try: tile.run(owner,1)
        except BaseException as error: errors.append(error)
    thread=threading.Thread(target=other);thread.start();thread.join()
    assert len(errors)==1 and 'Thread' in str(errors[0])
    tile.close(); assert calls[-2:]==['close','release']


@pytest.mark.parametrize('operation', ['handoff', 'close'])
def test_uncertain_retirement_blocks_every_later_submission(monkeypatch, operation):
    tile = object.__new__(backend.FlashOwnerTiles)
    tile.closed = tile.failed = tile._retirement_failed = False
    tile.thread = threading.current_thread()
    marker, token, calls = object(), object(), []
    tile.query, tile.frames, tile._loan = marker, {1: marker}, token
    class Native:
        def handoff_to_current_thread(self):
            calls.append('handoff')
            raise RuntimeError('uncertain handoff fence')
        def close(self): calls.append('close')
    class Artifact:
        module = object()
        def release(self, value):
            calls.append('release')
            raise RuntimeError('uncertain release')
    tile.primitive, tile.artifact = Native(), Artifact()
    for name in ('pack_owner_queries', 'gather_owner_q8'):
        monkeypatch.setattr(native, name, lambda *a, **k: pytest.fail('submission after uncertain retirement'))
    with pytest.raises(RuntimeError, match='uncertain'):
        if operation == 'handoff': tile.handoff_to_current_thread()
        else: tile.close()
    assert tile._retirement_failed and tile.query is marker and tile._loan is token
    with pytest.raises(RuntimeError, match='uncertain'): tile.run(NS(), 1)
    with pytest.raises(RuntimeError, match='uncertain'): tile.handoff_to_current_thread()
    with pytest.raises(RuntimeError, match='retry'): tile.close()
    assert calls == (['handoff'] if operation == 'handoff' else ['close', 'release'])
