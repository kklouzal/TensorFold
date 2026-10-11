"""Four-rank host protocol controls; these do not qualify multi-device CUDA kernels."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.families.glm5_next.cuda.split import RankReader, split_bytes, split_file


def methods(*names):
    path = Path(__file__).resolve().parents[1] / "src/tensorfold/families/glm5_next/cuda/engine.py"
    tree = ast.parse(path.read_bytes())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GlmEngine")
    selected = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {}
    module = ast.Module([ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *selected], [])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


def test_each_follower_consumes_only_its_own_bell_key():
    class Store:
        def __init__(self):
            self.keys = set()
            self.deleted = []

        def set(self, key, value):
            self.keys.add(key)

        def wait(self, keys, timeout):
            assert set(keys) <= self.keys

        def delete_key(self, key):
            self.keys.remove(key)
            self.deleted.append(key)

    store = Store()
    scope = methods("_ring", "_await_bell")
    ranks = [SimpleNamespace(rank=rank, world=4, _store=lambda: store) for rank in range(4)]
    for sequence in range(1, 4):
        scope["_ring"](ranks[0])
        for rank in (2, 1, 3):  # a fast follower cannot delete a slower follower's readiness signal
            scope["_await_bell"](ranks[rank])
            assert ranks[rank]._bell == sequence
        assert not store.keys
    assert len(store.deleted) == 9


@pytest.mark.parametrize("kind,shape,axis", [("row", [8, 12], 0), ("col", [8, 12], 1),
                                             ("dim1", [3, 8, 5], 1)])
def test_four_rank_splits_reassemble_the_exact_stored_bytes(kind, shape, axis):
    itemsize = 2
    raw = np.arange(np.prod(shape) * itemsize, dtype=np.uint8)
    full = raw.reshape(*shape, itemsize)
    shares = []
    for rank in range(4):
        part, local_shape = split_bytes(raw, shape, itemsize, kind, rank, 4)
        shares.append(part.reshape(*local_shape, itemsize))
    assert np.array_equal(np.concatenate(shares, axis=axis), full)


@pytest.mark.parametrize("rank,ranks", [(2, 2), (4, 4), (-1, 4), (True, 4), (0, 0), (0, True)])
def test_split_file_invalid_rank_refuses_before_checkpoint_reads(tmp_path, rank, ranks):
    with pytest.raises(ValueError, match="rank"):
        split_file(tmp_path / "missing.safetensors", tmp_path / "out", rank, ranks)
    assert not (tmp_path / "out").exists()


def test_a_share_outside_the_requested_world_is_not_misread_as_a_full_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "model.rank3.safetensors").touch()
    closed = []
    providers = SimpleNamespace(Reader=lambda: object(), ReadAhead=lambda *args: SimpleNamespace(
                                close=lambda: closed.append("read-ahead")), SafeTensors=None)
    monkeypatch.setitem(sys.modules, "tensorfold.cuda.direct_read", providers)
    with pytest.raises(ValueError, match="another rank's share"):
        RankReader(tmp_path, rank=0, ranks=2)
    assert closed == ["read-ahead"]
