"""Required common owners reserve their shared budget before optional keys."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def warm(events):
    path=Path(__file__).parents[1]/'src/tensorfold/families/qwen4_exp/cuda/bound_graphs.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='warm')
    env={'stage':lambda *args,**kwargs: events.append('stage') or [],'mtp_stage':lambda *args: []}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env['warm']


def test_workspace_reservation_precedes_every_optional_key():
    events=[]
    owner=NS(unique=False,dec=NS(depth=3,w=None,buf=NS(rows=8),mbuf=None),_prepare_greedy=lambda:events.append('reserve'),
             forward=lambda *args:events.append('capture'))
    warm(events)(owner,[1,2])
    assert events==['reserve','stage','capture','stage','capture']


def test_required_owner_failure_precedes_any_optional_staging_or_capture():
    events=[]

    def failed():
        raise MemoryError('retained partial workspace')

    owner=NS(unique=False,dec=NS(depth=3),_prepare_greedy=failed)
    with pytest.raises(MemoryError,match='retained partial'):
        warm(events)(owner,[1,2])
    assert events==[]


def test_declined_optional_optimization_preserves_all_existing_key_attempts():
    events=[]
    owner=NS(unique=False,dec=NS(depth=0,w=None,buf=NS(rows=3),mbuf=None),_prepare_greedy=lambda:None,
             forward=lambda *args:events.append('capture'))
    warm(events)(owner,[1,2,3])
    assert events==['stage','capture']*3
