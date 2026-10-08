"""Pure spare-cache ownership contract independent of an array runtime."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from tensorfold.engine import family_common


def test_retained_cache_returns_original_identity_and_runs_optional_hooks_in_order():
    events = []

    class Spare:
        def __init__(self, identity):
            self.identity = identity

        def drop_spare(self):
            events.append(self.identity)

    for cache in ([], [object()], [type('Plain', (), {'drop_spare': None})()]):
        assert family_common.drop_spares(cache) is cache
    cache = [Spare(0), object(), Spare(2)]
    assert family_common.drop_spares(cache) is cache
    assert events == [0, 2]


def test_spare_failure_preserves_primary_exception_and_stops_later_mutation():
    events = []
    primary = RuntimeError('owned cache failure')

    class Spare:
        def __init__(self, identity):
            self.identity = identity

        def drop_spare(self):
            events.append(self.identity)
            if self.identity == 1:
                raise primary

    with pytest.raises(RuntimeError) as failed:
        family_common.drop_spares([Spare(0), Spare(1), Spare(2)])
    assert failed.value is primary
    assert events == [0, 1]


def test_spare_property_failure_propagates():
    class Spare:
        @property
        def drop_spare(self):
            raise ValueError('lookup failed')

    with pytest.raises(ValueError, match='lookup failed'):
        family_common.drop_spares([Spare()])


def test_shared_cache_owner_imports_and_runs_without_mlx():
    # Fresh interpreter tests dependency ownership even when this suite has
    # already imported MLX on a machine that provides it.
    code = '''
import builtins, importlib.util, sys
original = builtins.__import__
def reject(name, *args, **kwargs):
    if name.split('.')[0] in ('mlx', 'mlx_lm'):
        raise AssertionError('pure helper imported array runtime: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = reject
spec = importlib.util.spec_from_file_location('pure_owner', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cache = [object()]
assert module.drop_spares(cache) is cache
'''
    subprocess.run([sys.executable, '-B', '-W', 'error', '-c', code, family_common.__file__], check=True, timeout=10)


def test_alternating_cache_public_name_reexports_same_callable():
    # Loading the actual module with only its foreign base-class boundary
    # supplied exercises the public alias without claiming MLX cache behavior.
    mlx, core = ModuleType('mlx'), ModuleType('mlx.core')
    mlx.core = core
    models, cache = ModuleType('mlx_lm.models'), ModuleType('mlx_lm.models.cache')
    cache.KVCache = type('KVCache', (), {})
    models.cache = cache
    mlx_lm = ModuleType('mlx_lm')
    mlx_lm.models = models
    path = Path(family_common.__file__).with_name('alternating_kv.py')
    spec = importlib.util.spec_from_file_location('test_owned_alternating', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': core, 'mlx_lm': mlx_lm,
                                 'mlx_lm.models': models, 'mlx_lm.models.cache': cache}):
        spec.loader.exec_module(module)
    assert module.drop_spares is family_common.drop_spares
