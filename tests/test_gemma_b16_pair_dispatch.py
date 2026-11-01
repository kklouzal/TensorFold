"""B01 explicit projection roles; no tensor kernels or model loading."""
from types import SimpleNamespace

import pytest

from tensorfold.families.gemma4.cuda import forward
from tensorfold.families.gemma4.cuda.weights import Config


def target(policy):
    import torch
    c = Config(hidden=64, layers=2, vocab=128, heads=16, kv_heads=8, global_kv_heads=2,
               head_dim=256, global_head_dim=512, intermediate=128, moe_width=64, experts=4, top_k=2,
               window=8, max_position=4096, eps=1e-6, softcap=30.,
               layer_types=('sliding_attention', 'full_attention'), rope_theta=10000.,
               global_rope_theta=1000000., global_rotary_fraction=.25, vision_bidirectional=True)
    embed = SimpleNamespace(weight=object(), embedding=lambda ids: ids)
    return forward.Gemma4Target(SimpleNamespace(config=c, numerics=policy, norm=torch.ones(64), embed=embed))


@pytest.mark.parametrize('policy', ['mlx', 'ct_nvfp4'])
def test_legacy_projection_roles_keep_original_bound_function(policy):
    model = target(policy)
    assert model._project_qkv.__func__ is forward.Gemma4Target._project
    assert model._project_gate_up.__func__ is forward.Gemma4Target._project


@pytest.mark.parametrize('prompt', [False, True])
def test_only_modelopt_equal_roles_pair_with_original_output_order(monkeypatch, prompt):
    model = target('modelopt_nvfp4')
    calls = []
    class Plain:
        def __init__(self, name):
            self.weight = name
        def __call__(self, x):
            calls.append(('decode-independent', self.weight, x))
            return self.weight
        def prefill(self, x):
            calls.append(('prompt-independent', self.weight, x))
            return self.weight
    def pair(x, w0, w1):
        calls.append(('pair', w0, w1, x))
        return [w0, w1]
    monkeypatch.setattr(forward.b16, 'prompt_pair' if prompt else 'matmul_pair', pair)
    def forbidden(*args, **kwargs):
        pytest.fail('wrong pair arithmetic selected')
    monkeypatch.setattr(forward.b16, 'matmul_pair' if prompt else 'prompt_pair', forbidden)
    q, k, v = Plain('q'), Plain('k'), Plain('v')
    assert model._project_qkv('x', (q, k, v), prompt) == ('q', 'k', 'v')
    assert model._project_gate_up('x', (Plain('gate'), Plain('up')), prompt) == ['gate', 'up']
    assert calls == [('prompt-independent' if prompt else 'decode-independent', 'q', 'x'),
                     ('pair', 'k', 'v', 'x'), ('pair', 'gate', 'up', 'x')]
    calls.clear()
    # Full-attention unequal Q/K and singleton O/down still take the original
    # role, even though they too are BF16 Plain projections.
    assert model._project('x', (q, k), prompt) == ['q', 'k']
    assert all(call[0] != 'pair' for call in calls)
