"""Diagnostic boundary/protocol checks; actual costs and profile math require SM70."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from tools import dp4a_actual_cost as tool
from tensorfold.cuda.sm70 import dp4a_experts as original


def test_routed_sequence_is_original_native_recipe_without_only_shared_assembly():
    source=ast.parse(inspect.getsource(original.Prototype._run).strip())
    wanted=source.body[0].body[1:]  # Remove the preexisting exact-only branch.
    wanted=[n for n in wanted if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='_write_shared')]
    actual=ast.parse(inspect.getsource(tool.RoutedBurst.routed_once).strip()).body[0].body[1:]
    assert ast.dump(ast.Module(body=actual,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=wanted,type_ignores=[]),include_attributes=False)
    assert tool.ROWS==(1,8,32) and (tool.SAMPLES,tool.REPLAYS,tool.UNROLL)==(3,8,4)


def test_unsupported_actual_bank_cannot_time_exact_fallback():
    owner=object.__new__(tool.RoutedBurst)
    owner.exact_only=True
    with pytest.raises(RuntimeError,match='not eligible'):
        owner.routed_once()


def test_event_batches_are_fixed_and_record_full_raw_duration(monkeypatch):
    log=[]
    class Event:
        def __init__(self,enable_timing):
            assert enable_timing
        def record(self,stream):
            log.append(('record',stream))
        def synchronize(self):
            log.append('sync')
        def elapsed_time(self,other):
            return 12.5
    monkeypatch.setattr(tool.torch.cuda,'Event',Event)
    assert tool.timed(lambda:log.append('call'),'stream')==12.5
    assert log==[('record','stream')]+['call']*8+[('record','stream'),'sync']


def test_baseline_borrower_refuses_release_until_graph_retires():
    log=[]
    owner=object.__new__(tool.PackedBurst)
    owner.closed=False
    owner.graph_owner=object()
    owner.stream=N(synchronize=lambda:log.append('fence'))
    with pytest.raises(RuntimeError,match='retire first'):
        owner.close()
    assert not log and not owner.closed
    owner.graph_owner=None
    owner.close()
    assert log==['fence'] and owner.closed and owner.anchors is None


def test_actual512_loader_is_exact_owned_supplier():
    root=Path(__file__).resolve().parents[1]
    import hashlib
    # Source supplier identity is recorded separately in HOLD; no model subset
    # change or installed-module monkeypatch is part of this diagnostic.
    assert tool.fixture is not None
    assert hashlib.sha256((root/'tools/ncu_experts_actual.py').read_bytes()).hexdigest()=='3eed9c7b32e69f029652ffe854f5befd29542acf5c99e44a577a3967ae0d8c57'


def test_actual_cleanup_observes_published_graph_and_preserves_original_with_close_errors(monkeypatch):
    primary=RuntimeError('capture operation')
    prior=ValueError('native prior root')
    primary.__cause__=prior
    close_error=RuntimeError('packed reset uncertain')
    log=[]
    class Owner:
        stream=object()
        def __init__(self,name,fail=False):
            self.name,self.fail=name,fail
            self.graph_owner=N(close=self.close_graph)
        def close_graph(self):
            log.append(self.name+' graph')
            if self.fail:
                raise close_error
            self.graph_owner=None
        def close(self):
            log.append(self.name+' owner')
            if self.graph_owner is not None:
                raise RuntimeError('borrower not retired')
    owners=[Owner('packed',True),Owner('dp')]
    errors=[None]*5
    body=ast.parse(inspect.getsource(tool.actual_case)).body[0].body
    cleanup=body[-5:-1]
    assert [type(n) for n in cleanup]==[ast.For,ast.For,ast.Try,ast.Expr]
    scope={'owners':owners,'errors':errors,'primary':primary,'original':'original',
           'cleanup_native_roots':[[None,None,False] for _ in range(5)],
           'torch':N(cuda=N(set_stream=lambda value:log.append(value))),
           'finish_failures':tool.finish_failures}
    with pytest.raises(RuntimeError) as caught:
        exec(compile(ast.fix_missing_locations(ast.Module(body=cleanup,type_ignores=[])),'actual cleanup','exec'),scope)
    assert caught.value is primary
    assert log==['packed graph','dp graph','packed owner','dp owner','original']
    assert errors[0] is close_error and errors[2] is not None
    assert owners[0].graph_owner is not None  # No release after uncertain reset.


def test_actual_case_compaction_exhaustion_retains_preexisting_native_roots(monkeypatch):
    primary=RuntimeError('case operation')
    cause,context=ValueError('prior explicit root'),LookupError('prior native context')
    primary.__cause__=cause
    primary.__context__=context
    primary.__suppress_context__=True
    original=object()
    side=N(wait_stream=lambda stream:None)
    monkeypatch.setattr(tool.torch.cuda,'current_stream',lambda:original)
    monkeypatch.setattr(tool.torch.cuda,'Stream',lambda:side)
    def select(stream):
        if stream is side:
            raise primary
    monkeypatch.setattr(tool.torch.cuda,'set_stream',select)
    real_finish=tool.finish_failures
    class Exhausted:
        def __iter__(self):
            raise MemoryError('outcome compaction')
    monkeypatch.setattr(tool,'finish_failures',lambda error,slots:real_finish(error,Exhausted()))
    with pytest.raises(RuntimeError) as caught:
        tool.actual_case(N(),1)
    assert caught.value is primary and isinstance(primary.__cause__,MemoryError)
    frame=primary.__traceback__
    while frame is not None and frame.tb_frame.f_code.co_name!='actual_case':
        frame=frame.tb_next
    assert frame is not None
    roots=frame.tb_frame.f_locals['native_roots']
    assert roots[0] is cause and roots[1] is context and roots[2] is True


def test_actual_cleanup_only_compaction_exhaustion_retains_first_cleanup_native_roots():
    primary=RuntimeError('first cleanup')
    cause,context=ValueError('cleanup explicit cause'),LookupError('cleanup native context')
    primary.__cause__=cause
    primary.__context__=context
    primary.__suppress_context__=True
    class Graph:
        def close(self):
            raise primary
    class Owner:
        graph_owner=Graph()
        stream=object()
        def close(self):
            raise RuntimeError('dependent owner remains borrowed')
    class Exhausted:
        def __init__(self):
            self.first=True
            self.values=(primary,)
        def __iter__(self):
            if self.first:
                self.first=False
                raise MemoryError('outcome compaction')
            return iter(self.values)
    real_finish=tool.finish_failures
    roots=[[None,None,False] for _ in range(5)]
    scope={'owners':[Owner(),None],'errors':[None]*5,'primary':None,'original':'original',
           'cleanup_native_roots':roots,'torch':N(cuda=N(set_stream=lambda value:None)),
           'finish_failures':lambda error,slots:real_finish(error,Exhausted())}
    body=ast.parse(inspect.getsource(tool.actual_case)).body[0].body
    with pytest.raises(RuntimeError) as caught:
        exec(compile(ast.fix_missing_locations(ast.Module(body=body[-5:-1],type_ignores=[])),'actual cleanup-only outcome','exec'),scope)
    assert caught.value is primary and isinstance(primary.__cause__,MemoryError)
    assert roots[0][0] is cause and roots[0][1] is context and roots[0][2] is True
