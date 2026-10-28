"""Issued HC-cut ownership and original HC suppression at the real caller."""
import ast
from pathlib import Path
from types import SimpleNamespace as N

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import moe_region_catalogue as module
from test_moe_region_catalogue import owner as catalogue_owner

owner = catalogue_owner  # Register the unchanged shared fixture under its original name.


def hc_owner(owner, rows=30):
    cat, w, b, log = owner
    cat.hc = True
    branch, injection = torch.zeros(8, rows, 2560), torch.zeros(rows, 4)
    for layer in w.layers:
        region = cat.regions[rows, layer.index]
        region.branch_signature = module._span(branch)
        region.injection_signature = module._span(injection)
        region.replay_prevalidated = lambda i=layer.index: log.append(('hc-moe', i))
    return cat, w, b, log, branch, injection


def test_full_ordered_cycle_issues_one_hc_cut_per_one_combined_replay(owner):
    cat, w, b, log, branch, injection = hc_owner(owner)
    assert cat.begin(w, b, 30)
    for layer in w.layers:
        assert cat.defer_hc(layer, b, 30, 3, branch.view_as(branch), injection[:])
        assert cat.pending_hc is layer
        assert cat.run(layer, b, 30)
        assert cat.pending_hc is None
    cat.finish()
    assert [i for kind, i in log if kind == 'hc-moe'] == list(range(48))
    assert cat.frames == 1 and cat.hits == 48 and not cat.lease


@pytest.mark.parametrize('cut', ['missing', 'duplicate', 'mode', 'foreign-branch', 'foreign-injection'])
def test_missing_duplicate_or_rebound_hc_cut_is_fatal_not_eager_fallback(owner, cut):
    cat, w, b, _, branch, injection = hc_owner(owner)
    cat.begin(w, b, 30)
    layer = w.layers[0]
    with pytest.raises(RuntimeError) as caught:
        if cut == 'missing':
            cat.run(layer, b, 30)
        else:
            cat.defer_hc(layer, b, 30, 3, branch, injection)
            if cut == 'duplicate':
                cat.defer_hc(layer, b, 30, 3, branch, injection)
            else:
                # Clear only the test's issued cut to isolate the next cut's
                # mode/storage guard from the separate duplicate-cut guard.
                cat.pending_hc = None
                cat.defer_hc(layer, b, 30, 1 if cut == 'mode' else 3,
                    branch.clone() if cut == 'foreign-branch' else branch,
                    injection.clone() if cut == 'foreign-injection' else injection)
    with pytest.raises(RuntimeError) as retired:
        cat.finish(caught.value)
    assert retired.value is caught.value and cat.failure is caught.value
    assert not cat.lease and not cat.frame_lock.locked()


def test_combined_replay_error_preserves_issued_owner_until_poison_retirement(owner):
    cat, w, b, _, branch, injection = hc_owner(owner)
    primary = RuntimeError('combined replay')
    cat.regions[30, 0].replay_prevalidated = lambda: (_ for _ in ()).throw(primary)
    cat.begin(w, b, 30)
    cat.defer_hc(w.layers[0], b, 30, 3, branch, injection)
    with pytest.raises(RuntimeError) as caught:
        cat.run(w.layers[0], b, 30)
    assert caught.value is primary and cat.pending_hc is w.layers[0]
    with pytest.raises(RuntimeError) as retired:
        cat.finish(primary)
    assert retired.value is primary and not cat.lease and cat.retiring


@pytest.mark.parametrize('hc,active,expected', [(False, True, 'eager'), (True, False, 'eager'), (True, True, 'issued')])
def test_actual_pre_moe_cut_runs_hc_exactly_once_or_defers_it(owner, hc, active, expected):
    root = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/qwen4_exp/cuda'
    tree = ast.parse((root / 'forward.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_pre_moe')
    statements = function.body[-2:]
    assert isinstance(statements[0], ast.Assign) and isinstance(statements[1], ast.If)
    log = []
    def issue(*args):
        log.append('issued')
        return True
    region = N(hc=hc, active=active, defer_hc=issue)
    b = N(moe_region_graphs=region, inj_a=torch.zeros(32, 4), inj_m=object())
    scope = {'getattr': getattr, 'b': b, 'layer': N(mlp_hc=object()), 'w': object(), 'R': 30,
        'c': N(eps=1e-6, streams=4, low=320), 'mode': 3, 'branch': object(), 'h': object(),
        'hc_block': lambda *args, **kwargs: log.append('eager')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), 'actual-pre-moe-cut', 'exec'), scope)
    assert log == [expected]


def test_hc_optin_requires_qualified_parent_catalogue(monkeypatch):
    monkeypatch.setenv(module.HC_ENV, '1')
    monkeypatch.setenv(module.ENV, '0')
    with pytest.raises(ValueError, match='admitted MoE'):
        module.requested_hc(module.PROFILE)
    monkeypatch.setenv(module.ENV, '1')
    assert all(module.requested_hc(profile) for profile in module.PROFILES)
    with pytest.raises(ValueError):
        module.requested_hc('reference')
