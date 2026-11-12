"""CPU execution-owner contracts; native arithmetic/graphs are separate gates."""
from types import SimpleNamespace
import threading
import weakref

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import dense_projections as dense
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense


class Buffers(SimpleNamespace):
    pass


@pytest.fixture
def native(monkeypatch):
    calls, prepared, contexts = [], [], []
    class Context:
        def __init__(self, device, *, backend, max_plans):
            self.thread = threading.current_thread()
            self.closed = False
            self.capacity = max_plans
            self._native = self  # Explicit CPU fixture provenance in owner.record().
            contexts.append(self)
        def _check(self):
            if self.closed or threading.current_thread() is not self.thread:
                raise RuntimeError('context closed or wrong owner thread')
        def handoff_to_current_thread(self):
            self.thread = threading.current_thread()
        def close(self):
            self.closed = True

    class Plan:
        def __init__(self, context, x, weight, out):
            assert len(prepared) < context.capacity
            self.context = context
            self.shape = tuple(x.shape), tuple(weight.shape), tuple(out.shape), tuple(out.stride()), out.dtype
            prepared.append(self)
            self.run(x, weight, out)
        def run(self, x, weight, out):
            self.context._check()
            assert self.shape == (tuple(x.shape), tuple(weight.shape), tuple(out.shape), tuple(out.stride()), out.dtype)
            calls.append((tuple(x.shape), out.data_ptr()))
            out.copy_(x.float() @ weight.float().T)

    monkeypatch.setattr(dense.dense_backends, 'Context', Context)
    monkeypatch.setattr(dense.dense_backends, 'prepare', lambda *args: Plan(*args))
    monkeypatch.setattr(dense.Owner, '_memory', lambda self: {
        'physical_free_bytes': 1 << 30, 'torch_allocated_bytes': 0, 'torch_reserved_bytes': 0})
    return calls, prepared, contexts


def fixtures(rows=5, prefill_rows=40):
    def matrix(n, k):
        return GGUFDense(torch.arange(n * k, dtype=torch.float16).view(n, k) % 3 - 1)
    def hc(n):
        return SimpleNamespace(down=matrix(n, 16), up=matrix(16, 2), prefill_up=None)
    layer = SimpleNamespace(index=0, linear=False, attn_hc=hc(4), mlp_hc=hc(2),
        attn=SimpleNamespace(proj=matrix(16, 8), o=matrix(8, 8)), ple=None)
    weights = SimpleNamespace(device=torch.device('cpu'), meta={'world': 8, 'execution_provider': 'sm70'},
        layers=[layer], mixer=hc(2), head=matrix(4, 8))
    def buffers(count, role):
        def value(width, dtype=torch.float16):
            return torch.zeros(count, width, dtype=dtype)
        return Buffers(rows=count, owner_role=role, prefill=role == 'prefill', world=8,
            execution_dtype=torch.float16, normed=value(16), act=value(2), up=value(16), dn=value(4), dn_mix=value(2),
            mixed=value(8), pa=value(16), gated=value(8), part_branch=value(8, torch.float32), branch=value(8),
            xs_act=torch.zeros(count, 1),
            gout=value(128), proj=torch.zeros(1, count, 514).half(), ple_emb=value(8), ple_keys=value(16),
            ple_vals=value(8), logits=torch.zeros(min(count, 8) if role == 'prefill' else count, 4))
    decoder = SimpleNamespace(buf=buffers(rows, 'decode'), pbuf=buffers(prefill_rows, 'prefill'),
        mbuf=SimpleNamespace(dense_linears='independent-reference'))
    text = {'hidden_size': 8, 'hc_count': 2, 'hc_lowrank': 2, 'num_attention_heads': 8,
        'num_key_value_heads': 1, 'head_dim': 8, 'linear_num_key_heads': 8, 'linear_num_value_heads': 8,
        'vocab_size': 32, 'num_hidden_layers': 1, 'ple_layer_ids': []}
    return weights, decoder, text


def prepare(native, *, rows=5, prefill_rows=40, backend='cublas', geometry='fixed-m32', panel_rows=32):
    weights, decoder, text = fixtures(rows, prefill_rows)
    spec = dense.Selection(backend, geometry, ('decode', 'prefill'), panel_rows)
    journal = []
    owner = dense.Owner.prepare(spec, text, {'decode': rows, 'prefill': min(32, prefill_rows)}, 'cpu', register=journal.append)
    assert journal == [owner]
    owner.bind_weights(weights)
    before = len(native[0]), len(native[1])
    owner.adopt(weights, decoder)
    assert (len(native[0]), len(native[1])) == before  # Adoption neither launches nor discovers native work.
    return owner, weights, decoder


@pytest.mark.parametrize('backend', ['cublas', 'cublaslt', 'cutlass'])
def test_fixed_geometry_cold_catalogue_roles_and_funded_storage(native, backend):
    calls, plans, contexts = native
    owner, weights, decoder = prepare(native, backend=backend)
    try:
        actual = sum(t.numel() * t.element_size() for arena in owner._arenas.values() for t in arena.values())
        assert actual == owner.resources['tensor_bytes']
        assert len(contexts) == 1 and len(plans) <= owner.resources['max_plans']
        assert decoder.mbuf.dense_linears == 'independent-reference'
        frame, matrix = decoder.pbuf.dense_linears, weights.layers[0].attn_hc.up
        x = torch.arange(80, dtype=torch.float16).view(40, 2) % 3 - 1
        before = len(plans)
        out = decoder.pbuf.up
        frame.matmul(x, matrix, out)
        assert torch.equal(out, (x.float() @ matrix.weight.float().T).half())
        assert all(shape[0] == 32 for shape, _ in calls)
        assert len(plans) == before  # No shape discovery/preparation on runtime calls.
    finally:
        frame = decoder.buf.dense_linears
        pad = weakref.ref(owner._arenas['decode']['input'])
        owner.close()
        assert decoder.buf.dense_linears is decoder.pbuf.dense_linears is None
        assert pad() is None and frame.closed


def test_serial_verify_regroup_uses_one_m32_provider_and_float_hc_partials(native):
    calls, plans, _ = native
    owner, weights, decoder = prepare(native, rows=32)
    try:
        frame, matrix = decoder.buf.dense_linears, weights.layers[0].attn_hc.up
        anchor = torch.tensor([[1., -1.]], dtype=torch.float16)
        frame.matmul(anchor, matrix, decoder.buf.up[:1])
        serial = decoder.buf.up[:1].clone()
        x = torch.zeros(4, 2).half()
        x[3].copy_(anchor[0])
        frame.matmul(x, matrix, decoder.buf.up[:4])
        assert torch.equal(serial[0].view(torch.int16), decoder.buf.up[3].view(torch.int16))
        source = torch.ones(4, 16).half()
        for hc in (weights.layers[0].attn_hc.down, weights.layers[0].mlp_hc.down):
            result = frame.partials(hc, source)
            assert result.shape == (1, 4, hc.n) and result.dtype == torch.float32 and result.is_contiguous()
            assert torch.equal(result[0], source.float() @ hc.weight.float().T)
        assert all(shape[0] == 32 for shape, _ in calls)
    finally:
        owner.close()


def test_malformed_original_inputs_rejected_before_pad_or_native_effect(native):
    calls, _, _ = native
    owner, weights, decoder = prepare(native)
    try:
        frame, matrix = decoder.buf.dense_linears, weights.layers[0].attn_hc.up
        before, pad = len(calls), owner._arenas['decode']['input'].clone()
        with pytest.raises(ValueError, match='domain'):
            frame.matmul(torch.ones(1, 2), matrix, decoder.buf.up[:1])
        with pytest.raises(ValueError, match='overlaps'):
            frame.matmul(torch.ones(1, 2).half(), matrix, matrix.weight.flatten()[:16].view(1, 16))
        assert len(calls) == before and torch.equal(pad, owner._arenas['decode']['input'])
        assert frame.matmul(torch.ones(1, 2).half(), {}, decoder.buf.up[:1]) is None
        with pytest.raises(ValueError, match='cold call catalogue'):
            frame.matmul(torch.ones(1, 2).half(), GGUFDense(torch.ones(16, 2).half()), decoder.buf.up[:1])
    finally:
        owner.close()


def test_owner_thread_handoff_precedes_any_pad_and_control_close_detaches(native):
    calls, _, _ = native
    owner, weights, decoder = prepare(native)
    frame, matrix = decoder.buf.dense_linears, weights.layers[0].attn_hc.up
    errors = []
    def worker():
        try:
            owner.handoff_to_current_thread()
            frame.matmul(torch.ones(1, 2).half(), matrix, decoder.buf.up[:1])
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors == []
    before = len(calls)
    with pytest.raises(RuntimeError, match='owner thread'):
        frame.matmul(torch.ones(1, 2).half(), matrix, decoder.buf.up[:1])
    assert len(calls) == before
    owner.close()
    assert decoder.buf.dense_linears is None
    with pytest.raises(RuntimeError, match='closed'):
        frame.matmul(torch.ones(1, 2).half(), matrix, decoder.buf.up[:1])


def test_direct_cutlass_contracts_bounded_and_no_vendor_direct_mtp_selection(native):
    owner, weights, decoder = prepare(native, backend='cutlass', geometry='direct', panel_rows=4)
    try:
        assert owner._arenas['decode']['input'].numel() == owner._arenas['decode']['output'].numel() == 0
        matrix = weights.layers[0].attn_hc.up
        x = torch.ones(5, 2).half()
        decoder.buf.dense_linears.matmul(x, matrix, decoder.buf.up)
        assert torch.equal(decoder.buf.up, (x.float() @ matrix.weight.float().T).half())
    finally:
        owner.close()
    with pytest.raises(ValueError, match='fixed M32'):
        dense.Selection('cublas', 'direct', ('decode',), 4)
    with pytest.raises(ValueError, match='MTP'):
        dense.Selection('cutlass', 'direct', ('mtp',), 4)


@pytest.mark.parametrize('role', [[], {}, True])
def test_untrusted_role_elements_refused_as_manifest_values_before_native_acquisition(native, role):
    with pytest.raises(ValueError, match='roles'):
        dense.Selection.parse({'backend': 'cublas', 'geometry': 'fixed-m32',
                               'roles': ['decode', role], 'panel_rows': 32})
    assert native[2] == []


def test_failed_cold_adoption_not_published_or_retried_and_closes_journal(native):
    weights, decoder, text = fixtures()
    selection = dense.Selection('cublas', 'fixed-m32', ('decode',), 32)
    owner = dense.Owner.prepare(selection, text, {'decode': 5}, 'cpu', register=lambda value: None)
    owner.bind_weights(weights)
    weights.layers[0].attn_hc.up = GGUFDense(torch.ones(16, 3).half())
    with pytest.raises(ValueError, match='pre-State weight binding'):
        owner.adopt(weights, decoder)
    assert not hasattr(decoder.buf, 'dense_linears')
    with pytest.raises(RuntimeError, match='one matching cold adoption'):
        owner.adopt(weights, decoder)
    owner.close()
    assert owner.closed


def test_forward_and_hc_hooks_preserve_role_output_dtype_and_reference_exclusion(native, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import forward

    owner, weights, decoder = prepare(native)
    activation = []
    monkeypatch.setattr(forward.glue, 'hc_act', lambda value, *args: activation.append(('half', value)))
    monkeypatch.setattr(forward.glue, 'hc_reduce_act', lambda value, *args: activation.append(('partial', value)))
    try:
        for buffers in (decoder.buf, decoder.pbuf):
            hc = weights.layers[0].attn_hc
            buffers.normed[:3].fill_(1)
            forward._down_act(hc, buffers, 3, 2, 2, torch.zeros(3, 2).half())
            kind, value = activation[-1]
            assert value.is_contiguous()
            assert value.dtype == (torch.float16 if buffers.prefill else torch.float32)
            assert kind == ('half' if buffers.prefill else 'partial')
            result = forward._mm(torch.ones(3, 2).half(), hc.up, None, buffers.up[:3], buffers)
            assert result.shape == (3, 16) and result.dtype == torch.float16
        assert decoder.mbuf.dense_linears == 'independent-reference'
    finally:
        owner.close()
