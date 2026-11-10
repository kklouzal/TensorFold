"""CPU contract/origin controls; these do not qualify native CUDA execution."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tensorfold.cuda.sm70 import family_native

ROOT = Path(__file__).parent.parent
SPEC = importlib.util.spec_from_file_location('candidate_qsa_test_oracle', ROOT / 'tests/cuda/test_sm70_qsa_select512.py')
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


def test_numeric_zero_ties_keep_lower_logical_group_instead_of_raw_bit_order():
    scores = np.zeros((1, 1025), dtype=np.float32)
    scores[0, ::2] = -0.
    ids, status = fixture.oracle(scores, [1025])
    assert np.array_equal(ids[0], np.arange(512)) and status[0] == 0


def test_independent_oracle_matches_pairwise_numeric_rank_for_adverse_finite_values():
    values = np.array([0xff7fffff, 0x80800000, 0x80000001, 0x80000000,
                       0, 1, 0x00800000, 0x7f7fffff], dtype=np.uint32).view(np.float32)
    scores = np.resize(values, 521)
    expected = [index for index, value in enumerate(scores)
                if sum(float(other) > float(value) or other == value and peer < index
                       for peer, other in enumerate(scores)) < 512]
    ids, status = fixture.oracle(scores[None], [521])
    assert ids[0].tolist() == expected and status[0] == 0


def test_oracle_contains_nonfinite_only_in_causal_prefix_and_invalid_lengths():
    scores = np.zeros((5, 1025), dtype=np.float32)
    scores[0, 512:] = np.nan
    scores[1, 0] = np.inf
    scores[2, 0] = -np.inf
    ids, status = fixture.oracle(scores, [512, 513, 1, -1, 1026])
    assert status.tolist() == [0, 1, 1, 2, 2]
    assert ids[0].tolist() == list(range(512)) and np.all(ids[1:] == -1)


def test_candidate_header_joins_existing_verified_family_source_identity_without_compile(monkeypatch):
    from tensorfold.cuda import build
    from tensorfold.cuda.sm70 import provider, source_cache, rotorquant_q4_tables
    before = torch.cuda.is_initialized()
    captured = {}
    monkeypatch.setattr(provider, '_native', lambda: SimpleNamespace(__name__='verified_sm70_provider'))
    monkeypatch.setattr(rotorquant_q4_tables, 'verify', lambda: None)
    def stage(sources, *, headers):
        captured.update(sources=sources, headers=headers)
        return sources
    def load(name, sources, **kwargs):
        captured.update(name=name, options=kwargs)
        return 'source-only-control'
    monkeypatch.setattr(source_cache, 'stage_sources', stage)
    monkeypatch.setattr(build, 'load', load)
    assert family_native.extension.__wrapped__() == 'source-only-control'
    assert 'qsa_select512.cuh' in [path.name for path in captured['headers']]
    identity = hashlib.sha256(b'verified_sm70_provider')
    for path in [*captured['sources'], *captured['headers']]:
        identity.update(path.name.encode() + b'\0' + path.read_bytes())
    assert captured['name'] == 'tf_sm70_family_' + identity.hexdigest()[:20]
    assert captured['options']['need'] == (7, 0)
    assert '--fmad=false' in captured['options']['extra_cuda_cflags']
    assert torch.cuda.is_initialized() is before
