"""Nemotron group validation before weight loads, allocations, or native execution."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def _source(name, symbol):
    path = ROOT / 'src/tensorfold/families/nemotron_h/cuda' / name
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == symbol)
    return ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[]))


def _config():
    return {'hidden_size': 64, 'vocab_size': 128, 'hybrid_override_pattern': 'M', 'num_attention_heads': 1,
            'num_key_value_heads': 1, 'head_dim': 128, 'mamba_num_heads': 8, 'mamba_head_dim': 32, 'n_groups': 2,
            'ssm_state_size': 128, 'conv_kernel': 4, 'n_routed_experts': 2, 'num_experts_per_tok': 1,
            'moe_intermediate_size': 64, 'moe_shared_expert_intermediate_size': 128}


@pytest.mark.parametrize('heads,groups', [(0, 1), (-1, 1), (8, 0), (8, -1), (8, 3), (8, 9)])
def test_config_refuses_invalid_groups_before_loading(tmp_path, heads, groups):
    namespace = {'__name__': __name__, 'dataclass': dataclass, 'json': json, 'Path': Path}
    exec(compile(_source('weights.py', 'Config'), '<config-source>', 'exec'), namespace)
    config = _config()
    config.update(mamba_num_heads=heads, n_groups=groups)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='positive heads and groups'):
        namespace['Config'].read(tmp_path)


@pytest.mark.parametrize('heads,groups', [(8, 1), (8, 2), (8, 4), (8, 8), (64, 8)])
def test_config_preserves_positive_divisible_groups(tmp_path, heads, groups):
    namespace = {'__name__': __name__, 'dataclass': dataclass, 'json': json, 'Path': Path}
    exec(compile(_source('weights.py', 'Config'), '<config-source>', 'exec'), namespace)
    config = _config()
    config.update(mamba_num_heads=heads, n_groups=groups)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    got = namespace['Config'].read(tmp_path)
    assert (got.m_heads, got.m_groups) == (heads, groups)


@pytest.mark.parametrize('heads,head_dim,groups', [(0, 32, 1), (-1, 32, 1), (8, 32, 0),
                                                (8, 32, -1), (8, 32, 3), (8, 32, 9), (8, 0, 2)])
def test_python_scan_refuses_before_allocating_or_calling_native(heads, head_dim, groups):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid groups reached allocation or native execution')
    namespace = {'torch': SimpleNamespace(empty=forbidden), '_ext': forbidden}
    exec(compile(_source('mamba.py', 'scan_rows'), '<scan-source>', 'exec'), namespace)
    with pytest.raises(ValueError):
        namespace['scan_rows'](None, None, None, None, None, None, 1, heads=heads, head_dim=head_dim,
                               groups=groups, state_dim=128, lo=0.0, hi=float('inf'))
