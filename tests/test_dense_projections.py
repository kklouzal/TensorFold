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


def _prefill_only(native, *, rows=40, method=False, method_hidden=8):
    from copy import deepcopy
    weights, decoder, text = fixtures(prefill_rows=rows)
    if method:
        weights.mtp = SimpleNamespace(fc_e=GGUFDense(torch.ones(method_hidden, method_hidden).half()),
            fc_h=GGUFDense(torch.ones(method_hidden, method_hidden).half()), layer=deepcopy(weights.layers[0]), mixer=deepcopy(weights.mixer))
    selection = dense.Selection('cublas', 'fixed-m32', ('prefill',), 32)
    owner = dense.Owner.prepare(selection, text, {'prefill': rows}, 'cpu', register=lambda owner: None)
    owner.bind_weights(weights)
    owner.adopt(weights, decoder)
    return owner, weights, decoder


def _aligned_half(shape, alignment=256):
    count = 1
    for size in shape:
        count *= size
    storage = torch.empty(count + alignment, dtype=torch.float16)
    shift = ((-storage.data_ptr()) % alignment) // storage.element_size()
    return storage[shift:shift + count].view(*shape)


def test_prefill_full_m32_source_borrows_sealed_plan_and_tail_still_stages(native):
    owner, weights, decoder = _prefill_only(native)
    try:
        x = _aligned_half((40, 2)); x.fill_(1)
        arena = owner._arenas['prefill']['input']; arena.fill_(123)
        count = len(native[1])
        result = decoder.pbuf.dense_linears.matmul(x, weights.layers[0].attn_hc.up, decoder.pbuf.up)
        expected = (x.float() @ weights.layers[0].attn_hc.up.weight.float().T).half()
        assert torch.equal(result, expected) and len(native[1]) == count
        pad = arena[:32 * 2].view(32, 2)
        assert torch.all(pad[:8] == 1) and torch.all(pad[8:] == 123)
        assert not hasattr(decoder.buf, 'dense_linears')
        assert decoder.mbuf.dense_linears == 'independent-reference'
    finally:
        owner.close()
    assert decoder.pbuf.dense_linears is None


def test_prefill_declines_actual_distinct_method_faces_but_unknown_faces_still_fail(native):
    owner, weights, decoder = _prefill_only(native, method=True)
    try:
        from tensorfold.families.qwen4_exp.cuda import forward
        b, method = decoder.pbuf, weights.mtp
        calls = len(native[0])
        x, out = torch.ones(3, 8).half(), torch.empty(3, 8).half()
        result = forward._mm(x, method.fc_e, None, out, b)
        assert torch.equal(result, x @ method.fc_e.weight.T) and len(native[0]) == calls
        assert b.dense_linears.matmul(x, method.layer.attn.proj, torch.empty(3, 16).half()) is None
        with pytest.raises(ValueError, match='cold call catalogue'):
            b.dense_linears.matmul(x, GGUFDense(torch.ones(8, 8).half()), out)
        # A target alias is not accidentally excluded by the method catalogue.
        assert weights.layers[0].attn.proj in b.dense_linears.faces
    finally:
        frame = decoder.pbuf.dense_linears
        owner.close()
    assert frame.excluded == frozenset()


def test_actual_mtp_compute_absorption_uses_existing_provider_on_prefill_buffers(native, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import mtp
    owner, weights, decoder = _prefill_only(native, method=True, method_hidden=32)
    b, m, n = decoder.pbuf, weights.mtp, 3
    weights.cfg = SimpleNamespace(hidden=32, streams=2, eps=1e-6)
    weights.comm = None
    for name, shape in {'mtp_e':(40,32), 'mixed':(40,32), 'mtp_eo':(40,32), 'mtp_in':(40,64),
        'mtp_hn':(40,64), 'mtp_hs':(80,32), 'h':(40,64)}.items():
        setattr(b, name, torch.ones(shape).half())
    b.ids = torch.zeros(40,dtype=torch.int32)
    b.xs_mixed, b.mtp_xh = torch.zeros(40,1), torch.zeros(40,2)
    b.split = None
    m.norm_e, m.norm_h = torch.ones(32), torch.ones(64)
    monkeypatch.setattr(mtp, '_embed', lambda w, ids, copies, out: out.fill_(1))
    monkeypatch.setattr(mtp, '_kv_begin', lambda segs: None)
    monkeypatch.setattr(mtp, '_kv_check', lambda *a, **k: None)
    monkeypatch.setattr(mtp, 'layer_forward', lambda *a, **k: None)
    monkeypatch.setattr(mtp, 'finish', lambda *a, **k: None)
    monkeypatch.setattr(mtp._family(b), 'rmsnorm', lambda x, w, eps, out, xs: (out.copy_(x), xs))
    monkeypatch.setattr(mtp._family(b), 'add_streams', lambda *a: None)
    before = len(native[0])
    try:
        assert mtp.mtp_compute(weights, [(SimpleNamespace(),0,n)], b, logits=False) is None
        assert len(native[0]) == before and torch.all(b.mtp_eo[:n] == 32)
    finally:
        owner.close()


def test_composed_adoption_handoff_and_retirement_keep_all_buffer_roles_disjoint(native, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import owner_preparation as preparation
    from tensorfold.cuda.sm70 import foundation_headers
    from tests.test_owner_preparation_choices import composed_spec
    weights, decoder, text = fixtures()
    vendor = dense.Owner.prepare(dense.Selection('cublas','fixed-m32',('prefill',),32),
        text, {'prefill':40}, 'cpu', register=lambda owner:None)
    vendor.bind_weights(weights)
    selected = preparation.OwnerPreparation(composed_spec(),
        role_rows={'decode':120,'mtp':120,'prefill':40}, manifest_sha256='a'*64)
    selected.prepared = True
    selected.source_foundation = {'id':'CPU-fixture'}
    selected.dense_owner = vendor
    decode_plan, mtp_plan = object(), object()
    retired = []
    selected.prepared_m32 = SimpleNamespace(role_rows={'decode':120,'mtp':120},
        buffers={'decode':decode_plan,'mtp':mtp_plan},close=lambda:retired.append('m32'))
    decoder.buf.padded_linears = decode_plan
    decoder.mbuf.padded_linears = mtp_plan
    decoder.mbuf.dense_linears = None
    decoder.pbuf.padded_linears = None
    for role,b in (('decode',decoder.buf),('mtp',decoder.mbuf),('prefill',decoder.pbuf)):
        selected.workspaces[role] = workspace = SimpleNamespace(native_kernel_family=preparation.RN_KERNEL_FAMILY,
            native_heads_per_block=12,bind_native_states=lambda states:None,close=lambda:None)
        b.history_owner = workspace
    decoder.states=[]
    decoder.controller=SimpleNamespace(configure_execution_families=lambda **kwargs:None)
    decoder.apply_execution_family=lambda:None
    weights.meta['owner_workspaces']=selected.workspaces
    monkeypatch.setattr(foundation_headers,'selected_identity',lambda:selected.source_foundation)
    selected.adopt(weights,decoder)
    assert decoder.pbuf.dense_linears is vendor._frames['prefill']
    assert not hasattr(decoder.buf,'dense_linears') and decoder.mbuf.dense_linears is None
    assert decoder.buf.padded_linears is decode_plan and decoder.mbuf.padded_linears is mtp_plan
    assert decoder.pbuf.padded_linears is None
    assert selected.requires_worker_handoff
    selected.handoff_to_current_thread()
    selected.close()
    assert vendor.closed and retired==['m32'] and decoder.pbuf.dense_linears is None


def test_prefill_misaligned_full_panel_retains_paid_input_staging(native):
    owner, weights, decoder = _prefill_only(native)
    try:
        x = _aligned_half((33,2))[1:]; x.fill_(1)
        arena = owner._arenas['prefill']['input']; arena.fill_(123)
        assert dense._alignment(x) < dense._alignment(arena)
        owner._frames['prefill'].matmul(x,weights.layers[0].attn_hc.up,decoder.pbuf.up[:32])
        assert torch.all(arena[:64] == 1)
    finally:
        owner.close()


def test_prefill_concat_uses_existing_joined_face_without_copying_original_weight_bank(native):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFConcat
    weights, decoder, text = fixtures()
    matrix = GGUFConcat((GGUFDense(torch.ones(32,8).half()), GGUFDense(torch.full((32,8),2).half())))
    assert matrix.fused is not None
    joined = matrix.fused.weight
    assert all(part.weight.untyped_storage().data_ptr()==joined.untyped_storage().data_ptr()for part in matrix.parts)
    weights.layers[0].attn.proj=matrix
    decoder.pbuf.pa=torch.zeros(40,64).half()
    owner=dense.Owner.prepare(dense.Selection('cublas','fixed-m32',('prefill',),32),
        text,{'prefill':40},'cpu',register=lambda owner:None)
    try:
        owner.bind_weights(weights);owner.adopt(weights,decoder)
        assert owner._bound['prefill'][matrix] is joined
        source=_aligned_half((40,8));source.fill_(1)
        result=decoder.pbuf.dense_linears.matmul(source,matrix,decoder.pbuf.pa)
        assert torch.all(result[:,:32]==8)and torch.all(result[:,32:]==16)
        assert owner._bound['prefill'][matrix] is joined
    finally:
        owner.close()
