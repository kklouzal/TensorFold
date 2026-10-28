"""HC-only selection, borrowed-weight and fallback contracts; no GPU math claim."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import hc_smallrow_vendor as module
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat
from tensorfold.families.qwen4_exp.cuda.prefix_policy import (
    HC_SMALLROW_VENDOR_PROFILE, SHARED_FUSION_PROFILE, identity, state_epoch,
)
from tensorfold.families.qwen4_exp.cuda.weight_types import HC


def hc(inject=True):
    down = GGUFDense(torch.empty((320, 10240), dtype=torch.float16, device='meta'))
    if inject:
        down = GGUFConcat((down, GGUFDense(torch.empty((4, 10240), dtype=torch.float16, device='meta'))))
    up = GGUFDense(torch.empty((10240, 320), dtype=torch.float16, device='meta'))
    return HC(down, up, torch.empty(10240, device='meta'), inject, down, up)


def test_hc_binding_keeps_original_weights_and_prefill_owners():
    original = hc()
    selected = module.bind_hc(original)
    assert selected.down.original is original.down and selected.up.original is original.up
    assert selected.down.parts is original.down.parts and selected.down.fused is original.down.fused
    assert selected.up.weight is original.up.weight and selected.scale is original.scale
    assert selected.prefill_down is original.down and selected.prefill_up is original.up
    assert selected.down.device_tensors() == original.down.device_tensors()


def test_numerical_profile_prefix_and_state_epochs_are_distinct():
    assert identity(HC_SMALLROW_VENDOR_PROFILE) != identity(SHARED_FUSION_PROFILE)
    assert state_epoch(HC_SMALLROW_VENDOR_PROFILE) != state_epoch(SHARED_FUSION_PROFILE)


def test_enrollment_changes_only_explicit_hc_objects_before_publication():
    original, mixer = hc(), hc(False)
    layers = [SimpleNamespace(attn_hc=original, mlp_hc=original, gdn=object(), moe=object()) for _ in range(48)]
    embed, head, fc = object(), object(), object()
    mtp = SimpleNamespace(layer=SimpleNamespace(attn_hc=original, mlp_hc=original), mixer=mixer, fc_h=fc)
    weights = SimpleNamespace(meta={'sm70_profile': HC_SMALLROW_VENDOR_PROFILE}, layers=layers,
                              mixer=mixer, mtp=mtp, embed=embed, head=head)
    gdn = layers[0].gdn
    module.enroll(weights)
    assert weights.embed is embed and weights.head is head and weights.mtp.fc_h is fc
    assert layers[0].gdn is gdn and weights.meta['hc_smallrow_vendor']['hc_objects'] == 100
    assert weights.meta['hc_smallrow_vendor']['controlled_metadata_bytes_observed'] <= module.HOST_ALLOWANCE
    assert weights.mtp.layer.attn_hc.prefill_down is original.down


def test_late_geometry_refusal_does_not_partially_publish_hc_adapters():
    original, mixer = hc(), hc(False)
    layers = [SimpleNamespace(attn_hc=original, mlp_hc=original) for _ in range(48)]
    layers[-1].mlp_hc = SimpleNamespace(down=original.down, up=original.down,
        prefill_down=original.down, prefill_up=original.down, inject=True)
    weights = SimpleNamespace(meta={'sm70_profile': HC_SMALLROW_VENDOR_PROFILE}, layers=layers, mixer=mixer, mtp=None)
    with pytest.raises(ValueError, match='up geometry'):
        module.enroll(weights)
    assert layers[0].attn_hc is original and weights.mixer is mixer


class Tensor:
    layout = torch.strided
    is_cuda = True
    requires_grad = False
    contiguous = True
    negative = False
    conjugate = False
    inference = False
    alias = None

    def __init__(self, shape, dtype=torch.float16, device='cuda:0'):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim, self._version = len(shape), 0

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugate

    def is_inference(self):
        return self.inference


@pytest.fixture
def matrices(monkeypatch):
    calls = []
    def increment(out):
        out._version += 1
    fake = SimpleNamespace(Tensor=Tensor, strided=torch.strided, float16=torch.float16, float32=torch.float32,
        is_inference_mode_enabled=lambda: True,
        empty=lambda shape, dtype, device: Tensor(shape, dtype, device),
        _C=SimpleNamespace(_overlaps=lambda output, value: output.alias is value),
        autograd=SimpleNamespace(graph=SimpleNamespace(increment_version=increment)))
    monkeypatch.setattr(module, 'torch', fake)
    monkeypatch.setattr(module.gguf_ops, 'hc_smallrow', lambda x, weights, out: calls.append(('native', weights, out)))
    monkeypatch.setattr(GGUFDense, 'implementation', 'native32')
    monkeypatch.setattr(GGUFConcat, 'implementation', 'native32')
    down = GGUFDense(Tensor((320, 10240)))
    inject = GGUFDense(Tensor((4, 10240)))
    up = GGUFDense(Tensor((10240, 320)))
    concat = object.__new__(GGUFConcat)
    concat.parts, concat.n, concat.k, concat.fused = (down, inject), 324, 10240, None
    for original in (down, up, concat):
        original.matmul = lambda x, **kw: calls.append(('original', x, kw)) or 'original'
        original.prefill = lambda x, **kw: calls.append(('prefill', x, kw)) or 'prefill'
    return module.Dense(up, down=False), module.Dense(down, down=True), module.Concat(concat), calls


@pytest.mark.parametrize('rows', (1, 2, 3))
def test_only_small_hc_rows_use_borrowed_original_weight_tensors(matrices, rows):
    up, down, concat, calls = matrices
    assert up.matmul(Tensor((rows, 320))).dtype == torch.float16
    assert down.matmul(Tensor((rows, 10240)), f32=True).dtype == torch.float32
    assert concat.matmul(Tensor((rows, 10240)), f32=True).dtype == torch.float32
    assert len(calls) == 3 and all(call[0] == 'native' for call in calls)
    assert calls[0][1][0] is up.original.weight
    assert calls[-1][1][1] is concat.original.parts[1].weight


@pytest.mark.parametrize('case', ('row4', 'row32', 'row0', 'stride', 'grad', 'neg', 'dtype', 'out-alias', 'out-grad'))
def test_required_generic_contracts_keep_original_matrix_path(matrices, case):
    up, _, _, calls = matrices
    x, out = Tensor((3, 320)), Tensor((3, 10240))
    if case.startswith('row'):
        x.shape = (int(case[3:]), 320)
    elif case == 'stride':
        x.contiguous = False
    elif case == 'grad':
        x.requires_grad = True
    elif case == 'neg':
        x.negative = True
    elif case == 'dtype':
        x.dtype = torch.float32
    elif case == 'out-alias':
        out.alias = x
    else:
        out.requires_grad = True
    assert up.matmul(x, out=out) == 'original' and calls[0][0] == 'original'


def test_dense_and_concat_publish_raw_output_mutation(matrices):
    up, _, concat, _ = matrices
    output = Tensor((1, 10240))
    up.matmul(Tensor((1, 320)), out=output)
    assert output._version == 1
    output = Tensor((1, 324), dtype=torch.float32)
    concat.matmul(Tensor((1, 10240)), out=output)
    assert output._version == 1
    assert up.prefill(Tensor((1, 320))) == 'prefill'


def test_raw_concat_write_invalidates_saved_autograd_input(monkeypatch):
    selected = module.bind_hc(hc())
    monkeypatch.setattr(selected.down, '_eligible', lambda *args: True)
    def opaque_copy(x, weights, out):
        out.numpy()[:] = 2  # Like a raw native write, NumPy does not bump Torch's version.
    monkeypatch.setattr(module.gguf_ops, 'hc_smallrow', opaque_copy)
    output = torch.ones((1, 324))
    parameter = torch.ones_like(output, requires_grad=True)
    saved = (parameter * output).sum()
    selected.down.matmul(torch.zeros((1, 10240), dtype=torch.float16), out=output, f32=True)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        saved.backward()


def test_generic_failure_propagates_original_identity(matrices):
    up, _, _, _ = matrices
    primary = ValueError('original matrix rejection')
    up.original.matmul = lambda *a, **kw: (_ for _ in ()).throw(primary)
    with pytest.raises(ValueError) as caught:
        up.matmul(Tensor((4, 320)))
    assert caught.value is primary


@pytest.mark.parametrize('inject', (True, False))
def test_prefill_down_uses_original_owner_even_for_one_row(monkeypatch, inject):
    from tensorfold.families.qwen4_exp.cuda import forward

    original = hc(inject)
    selected = module.bind_hc(original)
    got = torch.zeros((1, original.down.n))
    original.down.matmul = lambda *args, **kw: got
    selected.down.matmul = lambda *args, **kw: (_ for _ in ()).throw(AssertionError('private decode in prefill'))
    observed = []
    monkeypatch.setattr(forward.glue, 'hc_act', lambda value, *args: observed.append(value))
    buffers = SimpleNamespace(prefill=True, dn=torch.empty((1, 324)), dn_mix=torch.empty((1, 320)),
        normed=torch.empty((1, 10240)), act=torch.empty((1, 320)), xs_act=torch.empty((1, 10)), padded_linears=None)
    forward._down_act(selected, buffers, 1, 4, 320, None)
    assert observed == [got]


def test_host_metadata_funding_precedes_admission():
    import ast
    from pathlib import Path

    source = Path(__file__).parents[1] / 'src/tensorfold/families/qwen4_exp/cuda/engine.py'
    tree = ast.parse(source.read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
        and ast.unparse(n.test) == 'self.sm70_profile == HC_SMALLROW_VENDOR_TRIAL')
    options = {'host_workspace_bytes': 19, 'device_workspace_bytes': 23}
    scope = {'self': SimpleNamespace(sm70_profile=HC_SMALLROW_VENDOR_PROFILE),
        'HC_SMALLROW_VENDOR_TRIAL': HC_SMALLROW_VENDOR_PROFILE, 'options': options, 'tp': 8,
        '__package__': 'tensorfold.families.qwen4_exp.cuda'}
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), 'exec'), scope)
    assert options == {'host_workspace_bytes': 19 + 8 * module.HOST_ALLOWANCE, 'device_workspace_bytes': 23}
    admission = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'admit')
    assert block.end_lineno < admission.lineno


def test_selected_copy_variants_are_prepared_in_order_before_return(monkeypatch):
    from tensorfold.cuda.sm70 import execution_policy as policy, family_native

    log = []
    monkeypatch.setattr(policy, '_family_runtime', lambda torch: None)
    monkeypatch.setattr(policy, '_family_software', lambda: {'protected': {},
        'source_specification_sha256': 'same', 'recorded_components': {}})
    monkeypatch.setattr(family_native, 'extension', lambda: object())
    operations = SimpleNamespace(prepare_dense32=lambda: log.append('dense') or 4,
        prepare_shared_fusion=lambda: log.append('shared') or 32,
        prepare_hc_smallrow=lambda: log.append('hc') or 3)
    monkeypatch.setattr(module.gguf_ops, 'extension', lambda: operations)
    for name in ('TENSORFOLD_SM70_GDN_STATE_BATCH', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF',
                 'PYTORCH_HIP_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING'):
        monkeypatch.delenv(name, raising=False)
    fake = SimpleNamespace(cuda=SimpleNamespace(memory=SimpleNamespace(get_allocator_backend=lambda: 'native')))
    text = dict(policy._FAMILY_GEOMETRY, dtype='float16', mamba_ssm_dtype='float32', output_gate_type='sigmoid')
    policy._prepare_family_trial(text, 8, fake, packed=True, dense=True, shared_fusion=True, hc_smallrow=True)
    assert log == ['dense', 'shared', 'hc']


def test_private_tool_requires_explicit_smallrow_and_parent_intents(tmp_path, monkeypatch):
    from tools import v100_backend_probe as probe
    from tests.test_v100_backend_probe import cli

    monkeypatch.setenv('TENSORFOLD_SM70_PROFILE', HC_SMALLROW_VENDOR_PROFILE)
    monkeypatch.setenv('TENSORFOLD_SM70_GDN_STATE_BATCH', '1')
    args = cli(tmp_path) + ['--native-dense-trial', '--owner-batch-prelude-trial', '--router-blas-trial',
        '--gdn-state-batch-trial', '--packed-shared-fusion-trial', '--session-ceiling', '24', '--concurrency-grid',
        '--fast-grid', '--fast-grid-protocol', 'fast-grid-v2', '--tokens', '2048', '--quality-spotchecks']
    with pytest.raises(SystemExit):
        probe.arguments(args)
    parsed = probe.arguments(args + ['--hc-smallrow-vendor-trial'])
    assert parsed.hc_smallrow_vendor_trial and parsed.packed_shared_fusion_trial
    monkeypatch.setenv('TENSORFOLD_SM70_PROFILE', SHARED_FUSION_PROFILE)
    with pytest.raises(SystemExit):
        probe.arguments(args + ['--hc-smallrow-vendor-trial'])
