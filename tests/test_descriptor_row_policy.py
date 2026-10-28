"""Exact chosen policy files and rejection of broken inheritance; no GPU work."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import runpy

import pytest

from tools.descriptor_row_policy import validate


@pytest.fixture
def package(tmp_path):
    original = Path(os.environ['TENSORFOLD_DESCRIPTOR_POLICY_PACKAGE'])
    for name in ('candidate.json', 'source-hold.json', 'row-policy.json',
                 'wheel-provenance.json', 'row-policy-wheel-provenance.json'):
        shutil.copy(original / name, tmp_path / name)
    return tmp_path


def _change(package, name, transform, *, resign=None):
    path = package / name
    value = json.loads(path.read_text())
    transform(value)
    path.write_text(json.dumps(value))
    if resign:
        _change(package, 'candidate.json', lambda c: c.update({resign: hashlib.sha256(path.read_bytes()).hexdigest()}))


def test_exact_chosen_current_and_inherited_files(package):
    result = validate(package)
    assert result['current_HOLD_sha256'] == 'b5c78eb23e5d4c5111a707506a8afb5623cef81c0dd6412c70c1376786560f3a'
    assert result['row_policy_sha256'] == '8021154e213e1383b2a26290d14cb4d5318dfb0c80eee7cb003e3f9fc5e2979d'
    assert result['rows'] == [4, 5, 6, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 30, 31, 32]


def test_current_hold_cannot_masquerade_as_parent(package):
    _change(package, 'candidate.json', lambda c: c.update(row_policy_file='source-hold.json'))
    with pytest.raises(ValueError, match='explicit package document'):
        validate(package)


def test_wrong_original_policy_hash_is_rejected(package):
    _change(package, 'candidate.json', lambda c: c.update(row_policy_sha256='0' * 64))
    with pytest.raises(ValueError, match='authority hash differs'):
        validate(package)


def test_changed_graph_source_cannot_borrow_original_rows(package):
    def mutate(provenance):
        next(row for row in provenance['source_files'] if row['path'].endswith('/moe_region_catalogue.py'))['sha256'] = '0' * 64
    _change(package, 'wheel-provenance.json', mutate)
    with pytest.raises(ValueError, match='history-owner-only'):
        validate(package)


def test_changed_current_row_policy_cannot_borrow_parent_authority(package):
    _change(package, 'source-hold.json', lambda h: h.update(rows=[4] * 16), resign='source_hold_sha256')
    with pytest.raises(ValueError, match='inherit this exact row policy'):
        validate(package)


def test_exact_unchanged_checker_accepts_parent_and_rejects_current_schema(monkeypatch):
    original = Path(os.environ['TENSORFOLD_DESCRIPTOR_POLICY_PACKAGE'])
    monkeypatch.syspath_prepend(str(original / 'tools'))
    checker = runpy.run_path(str(original / 'tools/moe_screen_coverage.py'))['held_hot_rows']
    binding = validate(original)
    assert checker(original / binding['row_policy_file'], binding['row_policy_sha256']) == tuple(binding['rows'])
    with pytest.raises(ValueError, match='held hot16 policy'):
        checker(original / 'source-hold.json', binding['current_HOLD_sha256'])
