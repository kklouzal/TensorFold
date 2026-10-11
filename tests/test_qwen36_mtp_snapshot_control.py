"""New Qwen3.6 cache source/schema controls without MLX math or foreign imports."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from tensorfold.engine.snapshot_builtin import Profile
from tensorfold.engine.snapshot_payload import capture_snapshot

ROOT = Path(__file__).resolve().parents[1]


class Array:
    def __init__(self, shape):
        self.shape, self.dtype = shape, "F32"


def describe(value):
    return {"shape": list(value.shape), "dtype": value.dtype} if type(value) is Array else None


def setup_cache(monkeypatch):
    path = ROOT / "src/tensorfold/families/qwen3_5_moe/mtp.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "MTPCache")

    class KVCache:
        step = 4

        def __init__(self):
            self.keys = self.values = None
            self.offset = 0

    module = ModuleType("tensorfold.families.qwen3_5_moe.mtp")
    vars(module).update(KVCache=KVCache, Any=object)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), vars(module))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    prototype, cache = module.MTPCache(), module.MTPCache()
    cache.keys, cache.values, cache.offset = Array((1, 2, 8, 8)), Array((1, 2, 8, 8)), 3
    profile = Profile([prototype], token_limit=8, token_id_limit=100, tensor_byte_limit=4096,
                      max_draft=4, sizes={"F32": 4}, describe_tensor=describe, main_layers=0)
    profile.observe([cache])
    return profile.registry(), cache


def capture(registry, cache):
    return capture_snapshot(registry, "current", [1, 2, 3, 4], [cache], describe_tensor=describe,
                            tensor_kind=lambda value: "array" if type(value) is Array else None)


def test_new_head_snapshot_retains_absorbed_rows_and_drops_chain_state(monkeypatch):
    registry, cache = setup_cache(monkeypatch)
    cache.side = (Array((1, 2, 2, 8)), Array((1, 2, 2, 8)))
    cache.speculation = (object(), 3, False)
    payload = capture(registry, cache)
    entry = json.loads(payload.metadata["layers"])[0]
    assert set(payload.arrays) == {"0.keys", "0.values"}
    assert entry["plain"] == {"offset": 3}
    assert not {"side", "speculation", "drafted"} & set(registry.layers[0].fields())
    assert payload.metadata["format"] == "2"
    restored = copy.copy(registry.layers[0].prototype)
    assert restored.side is None and restored.speculation is None


def test_empty_head_context_is_valid_but_partial_absorbed_context_is_refused(monkeypatch):
    registry, cache = setup_cache(monkeypatch)
    cache.keys = cache.values = None
    cache.offset = 0
    assert capture(registry, cache).arrays == {}
    for offset in (0, 1):
        cache.offset, cache.keys = offset, Array((1, 2, 4, 8))
        with pytest.raises(ValueError):
            capture(registry, cache)


def test_auxiliary_offset_cannot_pass_the_authoritative_prefix(monkeypatch):
    registry, cache = setup_cache(monkeypatch)
    cache.offset = 5
    with pytest.raises(ValueError, match="auxiliary cache"):
        capture(registry, cache)


def test_mtp_weight_names_preserve_declared_prefix_and_refuse_ambiguous_aliases():
    path = ROOT / "src/tensorfold/families/qwen3_5_moe/mtp.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_weight_names")
    namespace = {"Any": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    normalize = namespace["_weight_names"]
    first, second = object(), object()
    assert normalize({"language_model.mtp.fc.weight": first, "norm.weight": second}) == {
        "fc.weight": first, "norm.weight": second,
    }
    with pytest.raises(ValueError, match="duplicate"):
        normalize({"fc.weight": first, "language_model.mtp.fc.weight": second})
    # Preserve the upstream suffix mapping; the strict module load validates the resulting field.
    assert normalize({"other.mtp.fc.weight": first}) == {"fc.weight": first}
