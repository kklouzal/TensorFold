"""NativeCT Gemma metadata/packing boundaries without importing CUDA runtimes."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'src/tensorfold'


def _definitions(path, names, namespace):
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


class Tensor:
    def __init__(self, shape, *, dtype='bf16', device='cuda:0'):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)

    def numel(self):
        from math import prod

        return prod(self.shape)

    def element_size(self):
        return 1 if self.dtype == 'u8' else 4

    def contiguous(self):return self

    def data_ptr(self):return id(self)

    def stride(self):
        from math import prod
        return tuple(prod(self.shape[i+1:]) for i in range(self.ndim))

    def __getitem__(self,key):
        shape=(len(range(*key.indices(self.shape[0]))),*self.shape[1:])
        return Tensor(shape,dtype=self.dtype,device=self.device)


class Bank:
    def __init__(self, swiglu, *, direct=False, capturable=True):
        self.swiglu, self.direct_input, self.capturable = swiglu, direct, capturable
        self.count, self.width, self.dims, self.device = 128, 704, 2816, 'cuda:0'

    def device_tensors(self):
        return Tensor((128, 2, 1115136), dtype='u8'), Tensor((128, 1115136), dtype='u8'), Tensor((768,), dtype='f32')


def _namespace():
    calls = []

    def allocate(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('invalid Gemma NVFP4 boundary reached an allocation')

    torch = SimpleNamespace(Tensor=Tensor, bfloat16='bf16', int32='i32', float32='f32', empty=allocate)
    ns = _definitions(ROOT / 'families/gemma4/cuda/nvfp4_experts.py', {'Experts', 'Workspace'},
                      {'torch': torch, 'ExpertsA4': Bank, 'PlanBuffers': object, 'geglu': object()})
    return ns, calls


def test_adapter_rejects_modelopt_direct_scales_and_wrong_activation_contract():
    ns, calls = _namespace()
    for bank in (Bank(ns['geglu'], direct=True), Bank(object()), Bank(ns['geglu'], capturable=False), object()):
        with pytest.raises(ValueError, match='original CT'):
            ns['Experts'](bank)
    assert not calls


def test_native_fields_are_counted_once_without_authoritative_cpu_copy():
    ns, calls = _namespace()
    adapter = ns['Experts'](Bank(ns['geglu']))
    assert adapter.nbytes() == 128 * 3 * 1115136 + 768 * 4
    assert not calls


@pytest.mark.parametrize('x', [None, Tensor((0, 2816)), Tensor((1, 2815)), Tensor((1, 2816), dtype='f32'),
                              Tensor((1, 2816), device='cpu'), Tensor((1, 1, 2816))])
def test_activation_metadata_rejected_before_native_or_cuda_work(x):
    ns, calls = _namespace()
    with pytest.raises(ValueError):
        ns['Experts'](Bank(ns['geglu'])).forward(x, Tensor((1, 8), dtype='i32'), Tensor((1, 8), dtype='f32'))
    assert not calls


@pytest.mark.parametrize('picks', [None, Tensor((1, 8), dtype='i64'), Tensor((2, 8), dtype='i32'),
                                  Tensor((1, 0), dtype='i32'), Tensor((1, 129), dtype='i32'),
                                  Tensor((1, 8), dtype='i32', device='cuda:1')])
def test_pick_metadata_prevents_implicit_integer_narrowing_and_wrong_geometry(picks):
    ns, calls = _namespace()
    with pytest.raises(ValueError):
        ns['Experts'](Bank(ns['geglu'])).forward(Tensor((1, 2816)), picks, Tensor((1, 8), dtype='f32'))
    assert not calls


@pytest.mark.parametrize('weights', [None, Tensor((1, 8), dtype='bf16'), Tensor((1, 7), dtype='f32'),
                                    Tensor((1, 8), dtype='f32', device='cpu')])
def test_routing_weights_retain_fp32_and_exact_row_slot_shape(weights):
    ns, calls = _namespace()
    with pytest.raises(ValueError):
        ns['Experts'](Bank(ns['geglu'])).forward(Tensor((1, 2816)), Tensor((1, 8), dtype='i32'), weights)
    assert not calls


@pytest.mark.parametrize('weight_device,scale_device,weight_dtype,scale_dtype,shape,scale_shape', [
    ('meta', 'meta', 'u8', 'f8', (64, 32), (64, 4)),
    ('cpu', 'cuda', 'u8', 'f8', (64, 32), (64, 4)),
    ('cuda', 'cuda', 'i32', 'f8', (64, 32), (64, 4)),
    ('cpu', 'cpu', 'u8', 'u8', (64, 32), (64, 4)),
    ('cpu', 'cpu', 'u8', 'f8', (63, 32), (63, 4)),
    ('cpu', 'cpu', 'u8', 'f8', (64, 31), (64, 4)),
    ('cpu', 'cpu', 'u8', 'f8', (64, 32), (64, 3)),
])
def test_shared_packer_rejects_bad_devices_dtype_and_partial_tiles_before_kernel(
        weight_device, scale_device, weight_dtype, scale_dtype, shape, scale_shape):
    devices = {name: SimpleNamespace(type=name) for name in ('meta', 'cpu', 'cuda')}
    torch = SimpleNamespace(uint8='u8', float8_e4m3fn='f8')
    ns = _definitions(ROOT / 'cuda/nvfp4/experts_a4.py', {'pack_projection_a4'}, {'torch': torch})
    weight = Tensor(shape, dtype=weight_dtype, device=devices[weight_device])
    scale = Tensor(scale_shape, dtype=scale_dtype, device=devices[scale_device])
    with pytest.raises(ValueError):
        ns['pack_projection_a4'](weight, scale)


@pytest.mark.parametrize('members,grouped,original', [
    ([1, 2, 0], ('expert0-row0', 'expert1-row0', 'expert2-row0'),
     ('expert2-row0', 'expert0-row0', 'expert1-row0')),
    ([1, 4, 2, 5, 0, 3], ('expert0-row0', 'expert0-row1', 'expert1-row0',
                        'expert1-row1', 'expert2-row0', 'expert2-row1'),
     ('expert2-row0', 'expert0-row0', 'expert1-row0', 'expert2-row1', 'expert0-row1', 'expert1-row1')),
])
def test_plan_order_lookup_matches_independent_original_row_slots(members, grouped, original):
    lookup = _definitions(Path(__file__).parent / 'nvfp4_a4_inputs.py', {'plan_order_positions'}, {})['plan_order_positions']
    positions = lookup(members)
    assert tuple(grouped[positions[original_slot]] for original_slot in range(len(members))) == original
    # The failed oracle indexed grouped scratch by original row/slot directly.
    assert grouped != original
    assert all(members[positions[original_slot]] == original_slot for original_slot in range(len(members)))


@pytest.mark.parametrize('members', [[], [0, 0, 2], [0, 2], [-1, 0, 1], [0, 1, 3], [0, True, 2]])
def test_plan_order_lookup_rejects_missing_duplicate_foreign_or_noninteger_slots(members):
    lookup = _definitions(Path(__file__).parent / 'nvfp4_a4_inputs.py', {'plan_order_positions'}, {})['plan_order_positions']
    with pytest.raises(ValueError, match='permute'):
        lookup(members)


@pytest.mark.parametrize('selected',[None,True,False])
def test_actual_forward_keeps_default_kwargs_and_explicit_decode_without_wrapper(selected):
    events=[];launches=[]
    def allocate(shape,**kwargs):
        events.append('out');return Tensor(shape,dtype=kwargs['dtype'],device=kwargs['device'])
    class Combine:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):launches.append((grid,args,kwargs))
            return launch
    torch=SimpleNamespace(Tensor=Tensor,bfloat16='bf16',int32='i32',float32='f32',empty=allocate)
    ns=_definitions(ROOT/'families/gemma4/cuda/nvfp4_experts.py',{'Experts','Workspace'},
        {'torch':torch,'ExpertsA4':Bank,'PlanBuffers':object,'geglu':object(),
         '_combine':Combine(),'triton':SimpleNamespace(cdiv=lambda n,d:(n+d-1)//d,next_power_of_2=lambda n:1<<(n-1).bit_length())})
    bank=Bank(ns['geglu']);calls=[]
    bank.run_plan=lambda *args,**kwargs:calls.append((args,kwargs))
    rows=2;buf=SimpleNamespace(pick=Tensor((rows,8),dtype='i32'),y=Tensor((rows,8,2816)),route=lambda n:events.append(('route',n)))
    workspace=SimpleNamespace(buffers=lambda *args:buf)
    adapter=ns['Experts'](bank,workspace)
    x=Tensor((rows,2816));weights=Tensor((rows,8),dtype='f32')
    options={} if selected is None else {'prompt':selected}
    result=adapter.forward(x,buf.pick,weights,**options)
    assert calls==[((x,buf),{'prefill':True,'decode':True} if selected is False else {'prefill':True})]
    assert events==[('route',rows),'out'] and len(launches)==1
    grid,args,kwargs=launches[0]
    assert grid==(rows,11) and args==(buf.y,buf.pick,weights,result)
    assert kwargs==dict(N=2816,R=8,RS=8,BLOCK=256,num_warps=4,enable_fp_fusion=False)


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_invalid_prompt_role_precedes_workspace_native_and_allocation(value):
    ns,calls=_namespace()
    adapter=ns['Experts'](Bank(ns['geglu']))
    with pytest.raises(ValueError,match='prompt role'):
        adapter.forward(Tensor((1,2816)),Tensor((1,8),dtype='i32'),Tensor((1,8),dtype='f32'),prompt=value)
    assert calls==[]
