"""Standalone foreign input closure validation; no upstream algorithm tests."""
import hashlib
from types import MappingProxyType

import pytest

from tools import build_flash_v100_forward as tool


@pytest.fixture
def tree(tmp_path, monkeypatch):
    entries = {'include/header.h': b'// fixture header\n', 'kernel/forward.cu': b'// fixture source\n'}
    monkeypatch.setattr(tool, 'INPUTS', MappingProxyType({name: (len(raw), hashlib.sha256(raw).hexdigest())
        for name, raw in entries.items()}))
    for name, raw in entries.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return tmp_path


def test_exact_inputs_accepted_but_additional_include_cannot_shadow_sdk(tree):
    tool.verify(tree)
    (tree / 'include/cuda_runtime.h').write_text('// unbound input')
    with pytest.raises(ValueError, match='undeclared'):
        tool.verify(tree)


def test_kernel_local_quoted_include_shadow_refused(tree):
    shadow = tree / 'kernel/header.h'
    shadow.write_text('// unexpected first-search header')
    with pytest.raises(ValueError, match='shadows'):
        tool.verify(tree)
    shadow.unlink()
    shadow.symlink_to(tree / 'not-yet-published.h')
    with pytest.raises(ValueError, match='shadows'):
        tool.verify(tree)


def test_changed_or_symlinked_source_not_adopted(tree):
    source = tree / 'kernel/forward.cu'
    original = source.read_bytes()
    source.write_bytes(b'?' * len(original))
    with pytest.raises(ValueError, match='digest'):
        tool.verify(tree)
    actual = tree / 'foreign-copy.cu'
    actual.write_bytes(original)
    source.unlink()
    source.symlink_to(actual)
    with pytest.raises(ValueError, match='type/size'):
        tool.verify(tree)
