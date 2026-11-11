"""Prepared selection metadata/dispatch controls; CPU spies do not qualify CUDA math."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda import moe
from tensorfold.cuda.sm70 import family_ops, routing_native


def buffers(rows=3):
    return NS(rows=rows, act=torch.empty((), dtype=torch.float16), logits=torch.randn(rows, 513),
              pick=torch.empty(rows, 11, dtype=torch.int32), wts=torch.empty(rows, 11))


@pytest.mark.parametrize('inference', [False, True])
def test_prepared_raw_writes_publish_two_original_mutations_per_output(inference):
    with torch.inference_mode(inference):
        buf, calls = buffers(), []
        def raw(logits, picks, weights):
            calls.append((logits, picks, weights))
            picks.numpy().fill(1)
            weights.numpy().fill(.5)
        prepared = routing_native.PreparedMoESelection(NS(route=raw), 'cpu')
        owned = prepared.bind(buf)
        before = None if inference else (buf.pick._version, buf.wts._version)
        assert owned.run(buf.logits[:2], buf, 10, 512) is True
        assert len(calls) == 1 and calls[0][1].shape == (2, 11)
        if not inference:
            assert (buf.pick._version, buf.wts._version) == (before[0] + 2, before[1] + 2)


def test_cpu_public_dispatch_preserves_explicit_torch_oracle_even_with_private_capability(monkeypatch):
    buf, expected = buffers(), buffers()
    expected.logits.copy_(buf.logits)
    def forbidden(*args):
        pytest.fail('CPU selection entered prepared CUDA operation')
    buf.selection = routing_native.PreparedMoESelection(NS(route=forbidden), 'cpu').bind(buf)
    monkeypatch.setattr(family_ops, '_NATIVE_FAMILY', False)
    moe._torch_select_rows(expected.logits, expected, 10, 512)
    before = buf.pick._version, buf.wts._version
    moe.select_rows(buf.logits, buf, 10, 512)
    assert torch.equal(buf.pick, expected.pick)
    assert torch.equal(buf.wts.view(torch.int32), expected.wts.view(torch.int32))
    assert (buf.pick._version, buf.wts._version) == (before[0] + 2, before[1] + 2)


@pytest.mark.parametrize('fault', ['topk', 'experts', 'foreign', 'offset', 'stride', 'grad',
                                  'subclass', 'output-grad', 'output-replaced', 'relocated'])
def test_prepared_domains_decline_without_native_effects(fault):
    buf, calls = buffers(), []
    owned = routing_native.PreparedMoESelection(NS(route=lambda *_: calls.append(True)), 'cpu').bind(buf)
    logits, topk, experts = buf.logits, 10, 512
    if fault == 'topk':
        topk = 9
    elif fault == 'experts':
        experts = 511
    elif fault == 'foreign':
        logits = logits.clone()
    elif fault == 'offset':
        logits = logits[1:]
    elif fault == 'stride':
        logits = logits[::2]
    elif fault == 'grad':
        logits.requires_grad_()
    elif fault == 'subclass':
        class Subclass(torch.Tensor):
            pass
        logits = torch.Tensor._make_subclass(Subclass, logits)
    elif fault == 'output-grad':
        buf.wts.requires_grad_()
    elif fault == 'output-replaced':
        buf.pick = buf.pick.clone()
    else:
        buf.pick.set_(torch.empty_like(buf.pick))
    assert owned.run(logits, buf, topk, experts) is False
    assert calls == []


def test_binding_rejects_alias_and_nonordinary_output():
    buf = buffers()
    buf.wts = buf.logits.reshape(-1)[:33].view(3, 11)
    with pytest.raises(ValueError, match='disjoint'):
        routing_native.PreparedMoESelection(NS(route=lambda *_: None), 'cpu').bind(buf)
    buf = buffers(257)
    with pytest.raises(ValueError, match='1..256'):
        routing_native.PreparedMoESelection(NS(route=lambda *_: None), 'cpu').bind(buf)


def test_preparation_resolves_function_before_return_without_global_family_enable(monkeypatch):
    from contextlib import nullcontext
    calls = []
    module = NS(route=lambda *_: None, prepare_route=lambda: calls.append('resident'))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (7, 0))
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(routing_native, 'extension', lambda: module)
    original = family_ops._NATIVE_FAMILY
    prepared = routing_native.prepare_owned_selection('cuda:0')
    assert calls == ['resident'] and prepared.route is module.route
    assert family_ops._NATIVE_FAMILY is original
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 0))
    with pytest.raises(ValueError, match='SM70'):
        routing_native.prepare_owned_selection('cuda:0')
    with pytest.raises(ValueError, match='SM70'):
        routing_native.prepare_owned_selection('cpu')


def test_inference_outputs_outside_mode_decline_then_preserve_reference_refusal(monkeypatch):
    calls = []
    with torch.inference_mode():
        buf = buffers()
        buf.selection = routing_native.PreparedMoESelection(NS(route=lambda *_: calls.append(True)), 'cpu').bind(buf)
    assert buf.selection.run(buf.logits, buf, 10, 512) is False
    monkeypatch.setattr(family_ops, '_NATIVE_FAMILY', False)
    with pytest.raises(RuntimeError, match='inference tensor'):
        moe.select_rows(buf.logits, buf, 10, 512)
    assert calls == []


def test_normal_output_versions_inside_inference_mode_match_original_copy():
    candidate, reference = buffers(), buffers()
    def raw(logits, picks, weights):
        picks.numpy().fill(1)
        weights.numpy().fill(.5)
    prepared = routing_native.PreparedMoESelection(NS(route=raw), 'cpu').bind(candidate)
    before = candidate.pick._version, candidate.wts._version
    expected_before = reference.pick._version, reference.wts._version
    with torch.inference_mode():
        moe._torch_select_rows(candidate.logits, reference, 10, 512)
        assert prepared.run(candidate.logits, candidate, 10, 512) is True
    assert (candidate.pick._version - before[0], candidate.wts._version - before[1]) == (
        reference.pick._version - expected_before[0], reference.wts._version - expected_before[1])
