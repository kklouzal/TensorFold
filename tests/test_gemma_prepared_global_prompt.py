"""Private prepared-global-KV ownership/routing boundaries, without CUDA work."""
from types import SimpleNamespace

import pytest

from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from test_gemma_cuda_modelopt_integration import _capacity_boundary
from tensorfold.families.gemma4.cuda import capacity


@pytest.mark.parametrize('value', [1, None, 'true'])
def test_private_engine_candidate_flag_is_strict_before_runtime(value, tmp_path):
    with pytest.raises(ValueError, match='bool'):
        Gemma4Engine(tmp_path, prepare_global_prompt=value)


def test_one_scratch_pair_not_multiplied_by_prefix_snapshots(monkeypatch, tmp_path):
    _capacity_boundary(monkeypatch, uma=True)
    baseline = capacity.admit(tmp_path, 128, True, max_rows=1, keep=3)
    candidate = capacity.admit(tmp_path, 128, True, max_rows=1, keep=3, prepare_global_prompt=True)
    # The synthetic profile has one global KV head, width128. One BF16 K+V
    # pair spans the admitted context; neither layers nor snapshots multiply it.
    assert candidate['geometry'].needed(128) - baseline['geometry'].needed(128) == 128 * 1 * 128 * 4
    assert candidate['weight'] == baseline['weight'] and candidate['host'] == baseline['host']


def test_target_refuses_unadmitted_or_wrong_arithmetic_scratch_before_tensors():
    from tensorfold.families.gemma4.cuda.forward import Gemma4Target
    weights = SimpleNamespace(numerics='mlx', config=SimpleNamespace(max_position=128))
    with pytest.raises(ValueError, match='admitted context'):
        Gemma4Target(weights, prepare_global_prompt=True, prompt_context=128)
    weights.numerics = 'modelopt_nvfp4'
    for limit in (None, 0, 129, True):
        with pytest.raises(ValueError, match='admitted context'):
            Gemma4Target(weights, prepare_global_prompt=True, prompt_context=limit)


def test_preparation_precedes_consumption_and_reuses_fixed_pair(monkeypatch):
    import torch
    from tensorfold.families.gemma4.cuda import attention
    raw = torch.empty((33, 2, 512), dtype=torch.bfloat16)
    keys, values = torch.empty_like(raw), torch.empty_like(raw)
    inv = torch.empty((33, 2), dtype=torch.float32)
    norm = torch.empty(512, dtype=torch.bfloat16)
    freq = torch.empty(256, dtype=torch.float32)
    pos = torch.zeros((), dtype=torch.int32)
    q = torch.empty((17, 16, 512), dtype=torch.bfloat16)
    calls = []
    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                assert args[4] is keys and args[5] is values and args[6] == 33
                calls.append(('prepare', grid))
            return launch
    monkeypatch.setattr(attention, '_prepare_global', Kernel())
    def consume(*args, **kwargs):
        assert kwargs['prepared'][0] is keys and kwargs['prepared'][1] is values
        calls.append(('consume', kwargs['chunks']))
        return q
    monkeypatch.setattr(attention, 'full', consume)
    for _ in range(2):
        assert attention.full_prepared(q, raw, inv, norm, freq, pos, chunks=1, length=33,
                                       round_normalized=False, scratch=(keys, values)) is q
    assert calls == [('prepare', (2, 2)), ('consume', 1)] * 2
    with pytest.raises(ValueError, match='scratch'):
        attention.full_prepared(q, raw, inv, norm, freq, pos, chunks=1, length=34,
                                round_normalized=False, scratch=(keys, values))
