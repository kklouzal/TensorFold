"""HC catalogue/copy/funding contracts; CPU kernels here are scheduling mocks."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from tensorfold.families.qwen4_exp.cuda import bitblas_hc as hc

TEXT = {'hidden_size': 2560, 'num_hidden_layers': 48, 'hc_count': 4, 'hc_lowrank': 320}


def selection(roles=('decode', 'prefill')):
    return hc.Selection('/sealed/manifest.json', 'a'*64, roles)


def test_forecast_pays_full_inject_copies_and_role_arenas():
    resources = hc.forecast(selection(), TEXT, {'decode': 120, 'prefill': 256})
    assert resources['padded_weight_bytes'] == 15*(1 << 20)
    assert resources['bound_HC_faces'] == 194
    assert resources['tensor_bytes'] == 15*(1 << 20) + 2*(32*10240*2+32*10240*4) + 120*324*4
    for role, rows in [('decode', 120), ('prefill', 256)]:
        value = resources['roles'][role]
        assert value['program_step_reference_bound'] == 3*sum((r+31)//32 for r in range(1,rows+1))
        assert value['unique_step_bound'] == 3*rows


@pytest.mark.parametrize('roles', [('mtp',), (), ('decode', 'decode'), ['decode']])
def test_selection_rejects_implicit_or_method_roles(roles):
    with pytest.raises(ValueError, match='roles'):
        selection(roles)


@pytest.mark.parametrize('field,value', [('hidden_size', 4096), ('hc_lowrank', 160), ('hc_count', 8), ('num_hidden_layers', 24)])
def test_forecast_does_not_reinterpret_other_model_geometries(field, value):
    with pytest.raises(ValueError, match='actual SC117'):
        hc.forecast(selection(), {**TEXT,field:value}, {'decode': 40, 'prefill': 256})


def prepared(role='decode', rows=40):
    resource = hc.forecast(selection((role,)), TEXT, {role:rows})
    arena = resource['roles'][role]
    owner = SimpleNamespace(device=torch.device('cpu'), resources=resource, poisoned=False,
        _kernels={'hc-down':'down','hc-inject':'inject','hc-up':'up'}, _check=lambda:None)
    owner._arenas = {role:{'input':torch.zeros(arena['input_half_elements'],dtype=torch.float16),
                          'output':torch.empty(arena['output_float_elements'],dtype=torch.float32),
                          'partial':torch.empty(arena['partial_float_elements'],dtype=torch.float32)}}
    keys = [object(),object(),object()]
    factors = [torch.tensor([1.],dtype=torch.float16),torch.tensor([3.],dtype=torch.float16),
               torch.tensor([2.],dtype=torch.float16)]
    owner._bindings = {keys[0]:hc._Binding('down',324,tuple(factors[:2])),
                       keys[1]:hc._Binding('down',320,(factors[0],)),
                       keys[2]:hc._Binding('up',10240,(factors[2],))}
    values = torch.arange(1,rows+1,dtype=torch.float16).view(rows,1)/16
    buffers = SimpleNamespace(owner_role=role, execution_dtype=torch.float16, rows=rows,
        normed=values.expand(rows,10240).clone(),act=values.expand(rows,320).clone(),
        up=torch.full((rows,10240),-3.,dtype=torch.float16),dn=torch.full((rows,324),-3.,dtype=torch.float16),
        dn_mix=torch.full((rows,320),-3.,dtype=torch.float16))
    return owner,buffers,keys,hc.Frame(owner,buffers,role)


@pytest.mark.parametrize('role', ['decode','prefill'])
@pytest.mark.parametrize('rows', [1,5,32,40])
def test_precreated_catalogue_preserves_partial_panel_rows_cast_and_crops(role, rows, monkeypatch):
    owner,b,keys,frame=prepared(role)
    def schedule_mock(kernel,a,w,out,_torch):
        assert a.shape[0] == out.shape[0] == 32
        out.copy_(a[:,:1].float().expand_as(out)*w[0])
    monkeypatch.setattr(hc.library,'call_library',schedule_mock)
    if role == 'decode':
        actual=frame.partials(keys[0],b.normed[:rows])[0]
    else:
        actual=frame.matmul(b.normed[:rows],keys[0],b.dn[:rows])
        assert torch.all(b.dn[rows:]==-3)
    expected=b.normed[:rows,:1].float()
    torch.testing.assert_close(actual[:,:320],expected.expand(rows,320).to(actual.dtype),rtol=0,atol=0)
    torch.testing.assert_close(actual[:,320:],(expected*3).expand(rows,4).to(actual.dtype),rtol=0,atol=0)
    result=frame.matmul(b.act[:rows],keys[2],b.up[:rows])
    assert result.dtype == torch.float16
    torch.testing.assert_close(result,(b.act[:rows,:1]*2).expand_as(result),rtol=0,atol=0)
    assert torch.all(b.up[rows:]==-3)
    assert not owner.poisoned


def test_runtime_uses_no_new_tensor_views_and_rejects_unsealed_storage(monkeypatch):
    owner,b,keys,frame=prepared()
    x=b.normed[:5]
    monkeypatch.setattr(hc.library,'call_library',lambda kernel,a,w,out,t:out.fill_(1.))
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,'__getitem__',lambda *args:pytest.fail('new runtime Tensor view'))
        result=frame.partials(keys[0],x)
        assert result.shape == (1,5,324)
    with pytest.raises(ValueError,match='sealed Buffer'):
        frame.partials(keys[0],x.clone())
    assert frame.matmul(b.normed,object(),b.dn) is None
    assert not owner.poisoned


def test_launch_failure_is_propagated_and_poisoned_without_hidden_fallback(monkeypatch):
    owner,b,keys,frame=prepared()
    def fail(*args):
        raise RuntimeError('native launch failed')
    monkeypatch.setattr(hc.library,'call_library',fail)
    with pytest.raises(RuntimeError,match='native launch failed'):
        frame.partials(keys[0],b.normed[:1])
    assert owner.poisoned


def test_hc_parse_preserves_explicit_scope_and_rejects_generic_dense_schema():
    value=selection().as_dict()
    assert hc.Selection.parse(value)==selection()
    assert hc.Selection.parse(None) is None
    with pytest.raises(ValueError,match='schema'):
        hc.Selection.parse({'backend':'bitblas','roles':['decode']})


def test_failed_retirement_retains_all_owners_and_forbids_retry(monkeypatch):
    from contextlib import nullcontext
    owner=hc.Owner()
    owner.closed=False;owner.failed_retirement=False;owner._device_resources=True;owner.device=torch.device('cuda',0)
    retained=object();owner._kernels=retained
    monkeypatch.setattr(torch.cuda,'device',lambda *a:nullcontext())
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    def fail(*args):
        raise RuntimeError('device retirement failed')
    monkeypatch.setattr(torch.cuda,'synchronize',fail)
    with pytest.raises(RuntimeError,match='device retirement failed'):
        owner.close()
    assert not owner.closed and owner.failed_retirement and owner._kernels is retained
    with pytest.raises(RuntimeError,match='retry is forbidden'):
        owner.close()


def test_bad_sealed_manifest_is_journalled_before_any_cuda_effect(monkeypatch):
    captured=[]
    def fail(*args):
        raise ValueError('manifest refused')
    monkeypatch.setattr(hc.library,'hc_manifest',fail)
    monkeypatch.setattr(torch.cuda,'device',lambda *a:pytest.fail('CUDA effect before manifest validation'))
    with pytest.raises(ValueError,match='manifest refused'):
        hc.Owner.prepare(selection(('decode',)),TEXT,{'decode':40},'cuda:0',register=captured.append)
    assert len(captured)==1 and captured[0]._kernels=={} and captured[0]._arenas=={}
    assert captured[0].record()['quality_acceptance'] is False
    captured[0].close()
    assert captured[0].closed


def test_owner_receipt_uses_strict_json_lists_for_native_geometry():
    from tensorfold.engine.model_identity import canonical_runtime_json
    owner=hc.Owner()
    owner.selection=selection(('decode',));owner.resources=hc.forecast(owner.selection,TEXT,{'decode':40})
    owner.bound_weights=False;owner.adopted=False;owner._frames={}
    owner.preparation_memory=owner.binding_memory=owner.adoption_memory=None
    owner._kernels={'hc-down':SimpleNamespace(dimensions=(32,320,10240),native_provider={},runtime_provider={})}
    record=owner.record()
    assert record['native_libraries']['hc-down']['dimensions']==[32,320,10240]
    assert canonical_runtime_json(record)
