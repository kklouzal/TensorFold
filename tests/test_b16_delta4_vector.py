"""CPU-only adapter boundaries over the maintained owned Delta4 descriptor.

These controls neither compile a provider nor qualify CUDA numerics/performance.
The protected packer remains the sole encoder; actual GPU oracles are separate.
"""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda import b16_delta4_vector as vector
from tensorfold.cuda import b16_delta4 as codec


@pytest.mark.parametrize('rows', [True, False, 0, 3, 32, 1.0, None])
def test_invalid_reuse_is_refused_before_dependency_or_provider_work(monkeypatch, rows):
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid reuse must not enter packed validation or provider')

    monkeypatch.setattr(codec, '_validate', forbidden)
    monkeypatch.setattr(vector, '_ext', forbidden)
    with pytest.raises(ValueError, match='1/2/4/8/16'):
        vector.matmul_rows(None, None, rows=rows)


@pytest.mark.parametrize('operation', ['unpack', 'matmul'])
def test_cpu_encoding_does_not_reach_cuda_provider(monkeypatch, operation):
    packed = codec.pack(torch.ones((2, 64), dtype=torch.bfloat16))

    def forbidden():
        raise AssertionError('CPU encoding must fail before CUDA provider preparation')

    monkeypatch.setattr(vector, '_ext', forbidden)
    with pytest.raises(ValueError, match='same-device schema'):
        if operation == 'unpack':
            vector.unpack(packed)
        else:
            vector.matmul_rows(torch.ones((1, 64), dtype=torch.bfloat16), packed)


@pytest.mark.parametrize('reuse', [1, 2, 4, 8, 16])
def test_adapter_reuses_validator_and_forwards_exact_original_field_owners(monkeypatch, reuse):
    fields = [object() for _ in range(4)]
    packed = SimpleNamespace(low=fields[0], delta=fields[1], metadata=fields[2], raw=fields[3], shape=(17, 2816))
    x, output = object(), object()
    calls = []

    def validate(actual, *, cuda):
        assert actual is packed and cuda is True
        calls.append('validate')

    def matmul(*arguments):
        assert arguments == (x, *fields, 17, 2816, reuse)
        calls.append('matmul')
        return output

    monkeypatch.setattr(codec, '_validate', validate)
    monkeypatch.setattr(vector, '_ext', lambda: SimpleNamespace(matmul_rows=matmul))
    assert vector.matmul_rows(x, packed, rows=reuse) is output
    assert calls == ['validate', 'matmul']


def test_unpack_forwards_same_encoded_field_authority(monkeypatch):
    fields = [object() for _ in range(4)]
    packed = SimpleNamespace(low=fields[0], delta=fields[1], metadata=fields[2], raw=fields[3], shape=(3, 64))
    output = object()
    seen = []
    monkeypatch.setattr(codec, '_validate', lambda actual, **kwargs: seen.append((actual, kwargs)))

    def unpack(*arguments):
        assert arguments == (*fields, 3, 64)
        return output

    monkeypatch.setattr(vector, '_ext', lambda: SimpleNamespace(unpack=unpack))
    assert vector.unpack(packed) is output
    assert seen == [(packed, {'cuda': True})]
