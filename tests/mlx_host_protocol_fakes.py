"""Explicit host-only MLX boundaries for tests that perform no array math."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
import sys
from types import ModuleType

import pytest


def _install_cli_protocol(monkeypatch, version='test-host-protocol'):
    """Startup metadata/residency calls only; unsupported array calls stay absent."""
    core = ModuleType('mlx.core')
    core.__version__ = version
    core.__file__ = __file__  # metadata path only; this fixture supplies no MLX headers or compiler
    core.int32 = object()  # opaque default-argument metadata; array operations stay absent
    core.set_cache_limit = core.set_memory_limit = lambda value: None
    core.synchronize = core.clear_cache = lambda: None
    core.get_active_memory = lambda: 0
    core.set_wired_limit = lambda value: 0
    core.device_info = lambda: {'max_recommended_working_set_size': 64 << 30, 'memory_size': 128 << 30}
    mlx = ModuleType('mlx')
    mlx.core = core
    nn = ModuleType('mlx.nn')  # metadata imports only; no numerical methods or fake layers

    class QuantizedLinearMetadata:
        def __new__(cls, *args, **kwargs):
            raise AssertionError('host metadata fixture cannot construct numerical layers')

    nn.QuantizedLinear = QuantizedLinearMetadata
    mlx.nn = nn
    monkeypatch.setitem(sys.modules, 'mlx', mlx)
    monkeypatch.setitem(sys.modules, 'mlx.core', core)
    monkeypatch.setitem(sys.modules, 'mlx.nn', nn)
    previous = metadata.version
    monkeypatch.setattr(metadata, 'version', lambda name: version if name == 'mlx-lm' else previous(name))
    return core


@contextmanager
def cli_protocol(monkeypatch, version='test-host-protocol'):
    """Sequential host-only scope; no numerical work may outlive its teardown.

    Remove only newly imported owned modules retaining this exact fake foreign
    boundary before the real module bindings are restored. Existing production
    modules and independently replaced parent attributes remain untouched.
    """
    before = {name: module for name, module in sys.modules.items() if name.startswith('tensorfold.')}
    with monkeypatch.context() as scoped:
        core = _install_cli_protocol(scoped, version)
        protocol_objects = (core, sys.modules['mlx'], sys.modules['mlx.nn'], metadata.version,
                            *(value for value in vars(core).values() if callable(value)),
                            *(value for value in vars(sys.modules['mlx.nn']).values() if callable(value)))
        try:
            yield core
        finally:
            owned = [(name, module) for name, module in list(sys.modules.items())
                     if name.startswith('tensorfold.') and isinstance(module, ModuleType) and module is not before.get(name)
                     and any(value is protocol for value in vars(module).values() for protocol in protocol_objects)]
            for name, module in sorted(owned, key=lambda item: item[0].count('.'), reverse=True):
                if sys.modules.get(name) is module:
                    parent_name, _, child = name.rpartition('.')
                    parent = sys.modules.get(parent_name)
                    if parent is not None and getattr(parent, child, None) is module:
                        delattr(parent, child)
                    del sys.modules[name]


@pytest.fixture(name="mlx_host_protocol")
def mlx_host_protocol(monkeypatch):
    with cli_protocol(monkeypatch) as core:
        yield core


def owned_release_class(relative_source, class_name):
    """Execute the actual pure release method, with no foreign class imports.

    This fixture qualifies host ownership only. The numerical MLX family and
    its foreign base classes require their separate platform execution tests.
    """
    import tensorfold

    source_root = Path(tensorfold.__file__).resolve().parents[1]
    path = source_root / relative_source
    if not path.resolve(strict=True).is_relative_to(source_root.resolve(strict=True)):
        raise ValueError('owned source fixture escaped source root')
    tree = ast.parse(path.read_bytes())
    source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    release = next(n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name == 'release_rounds')
    cls = ast.ClassDef(name=class_name, bases=[], keywords=[], body=[release], decorator_list=[])
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[class_name]
