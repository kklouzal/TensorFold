"""Explicitly provisioned prebuilt HC libraries: real SM70 adapter gates."""
import os
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import bitblas_hc as candidate
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFConcat, GGUFDense

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='actual SM70 HC reference component')


class Buffers:
    def __init__(self, role, rows, device):
        self.rows,self.owner_role,self.execution_dtype=rows,role,torch.float16
        values=torch.arange(1,rows+1,device=device,dtype=torch.float16).view(-1,1)/16
        self.normed=values.expand(rows,10240).clone()
        self.act=values.expand(rows,320).clone()
        self.up=torch.empty((rows,10240),device=device,dtype=torch.float16)
        self.dn=torch.empty((rows,324),device=device,dtype=torch.float16)
        self.dn_mix=torch.empty((rows,320),device=device,dtype=torch.float16)


@pytest.fixture(scope='module')
def prepared():
    path=os.environ.get('TENSORFOLD_BITBLAS_HC_MANIFEST')
    sha=os.environ.get('TENSORFOLD_BITBLAS_HC_SHA256')
    if not path or not sha:
        pytest.skip('explicit producer-bound HC manifest/SHA required; no implicit cache/generation')
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,0):
        pytest.skip('one actual SM70 device required')
    with torch.inference_mode():
        device=torch.device('cuda',0)
        registered=[]
        owner=candidate.Owner.prepare(candidate.Selection(path,sha),
            {'hidden_size':2560,'hc_lowrank':320,'hc_count':4,'num_hidden_layers':48},
            {'decode':40,'prefill':40},device,register=registered.append)
        down=torch.full((320,10240),2**-10,dtype=torch.float16,device=device)
        inject=torch.full((4,10240),2**-9,dtype=torch.float16,device=device)
        up=torch.full((10240,320),2**-8,dtype=torch.float16,device=device)
        def hc(injecting=True):
            d=GGUFConcat([GGUFDense(down),GGUFDense(inject)]) if injecting else GGUFDense(down)
            return SimpleNamespace(down=d,up=GGUFDense(up),inject=injecting)
        weights=SimpleNamespace(device=device,meta={'world':8,'execution_provider':'sm70'},
            layers=[SimpleNamespace(attn_hc=hc(),mlp_hc=hc()) for _ in range(48)],mixer=hc(False))
        owner.bind_weights(weights)
        decoder=SimpleNamespace(buf=Buffers('decode',40,device),pbuf=Buffers('prefill',40,device))
        owner.adopt(weights,decoder)
        try:
            yield owner,weights,decoder
        finally:
            owner.close()


@pytest.mark.parametrize('role',['decode','prefill'])
@pytest.mark.parametrize('rows',[1,5,32,40])
def test_actual_hc_known_answers_row_regroup_live_graph_and_funding(prepared,role,rows):
    owner,w,decoder=prepared
    b=decoder.buf if role=='decode' else decoder.pbuf
    frame=b.dense_linears;hc=w.layers[0].attn_hc
    assert owner.resources['padded_weight_bytes']==15*(1<<20)
    assert owner.adoption_memory['before']['torch_allocated_bytes']==owner.adoption_memory['after']['torch_allocated_bytes']
    def run():
        if role=='decode':
            down=frame.partials(hc.down,b.normed[:rows])[0]
        else:
            down=frame.matmul(b.normed[:rows],hc.down,b.dn[:rows])
        frame.matmul(b.act[:rows],hc.up,b.up[:rows])
        return down
    initial=run().clone();initial_up=b.up[:rows].clone()
    values=torch.arange(1,rows+1,device=b.normed.device,dtype=torch.float32).view(-1,1)
    assert torch.equal(initial[:,:320],(values*.625).expand(rows,320).to(initial.dtype))
    assert torch.equal(initial[:,320:],(values*1.25).expand(rows,4).to(initial.dtype))
    assert torch.equal(initial_up,(values*.078125).expand(rows,10240).half())
    # The existing physical-M32 proof is exercised through this complete staged
    # publication pipeline at both smaller logical prefixes and a32+8 boundary.
    full=(frame.partials(hc.down,b.normed)[0] if role=='decode'
          else frame.matmul(b.normed,hc.down,b.dn)).clone()
    assert torch.equal(initial,full[:rows])
    frame.matmul(b.act,hc.up,b.up)
    assert torch.equal(initial_up,b.up[:rows])
    allocated=torch.cuda.memory_allocated()
    run();torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==allocated
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):run()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph,stream=stream):
            captured=run()
        b.normed.mul_(2);b.act.mul_(2)
        graph.replay();torch.cuda.synchronize()
        assert torch.equal(captured,initial*2)
        assert torch.equal(b.up[:rows],initial_up*2)
    finally:
        torch.cuda.synchronize();graph.reset()
        b.normed.div_(2);b.act.div_(2)
        torch.cuda.synchronize()


@pytest.mark.parametrize('pattern',['random','gguf'])
def test_random_and_actual_hc_matrices_keep_row_law_and_live_graph(pattern, record_property):
    """Numerical differences are recorded, never accepted by a fitted tolerance."""
    import json
    from pathlib import Path

    import numpy as np

    from tensorfold.families.qwen4_exp.gguf import decode_rows
    from tensorfold.gguf_file import GGUFCheckpoint

    path=os.environ.get('TENSORFOLD_BITBLAS_HC_MANIFEST')
    sha=os.environ.get('TENSORFOLD_BITBLAS_HC_SHA256')
    if not path or not sha:
        pytest.skip('explicit producer-bound HC manifest/SHA required')
    gguf=os.environ.get('TENSORFOLD_BITBLAS_HC_GGUF')
    if pattern=='gguf' and not gguf:
        pytest.skip('explicit actual HC GGUF coefficients required for asset gate')
    device=torch.device('cuda',0)
    owner=None;graph=None
    primary=None
    with torch.inference_mode():
        generator=torch.Generator(device=device).manual_seed(31781)
        if pattern=='gguf':
            checkpoint=GGUFCheckpoint.open(gguf)
            def read(name):
                tensor=checkpoint.tensors[name]
                if tensor.nbytes > (32<<20):
                    raise ValueError('actual HC matrix exceeds the established bounded reader')
                with Path(tensor.path).open('rb') as stream:
                    stream.seek(tensor.offset);raw=stream.read(tensor.nbytes)
                if len(raw)!=tensor.nbytes:
                    raise ValueError('actual HC coefficient read truncated')
                decoded=decode_rows(tensor,np.frombuffer(raw,np.uint8).reshape(tensor.rows,tensor.row_bytes))
                half=decoded.astype(np.float16)
                if not np.all(np.isfinite(half)):
                    raise ValueError('actual HC coefficients exceed finite Half contract')
                return torch.from_numpy(half).to(device)
            down=read('blk.0.hc_attn_down.weight')
            inject=read('blk.0.hc_attn_inject.weight')
            up=read('blk.0.hc_attn_up.weight')
        else:
            down=torch.randn((320,10240),device=device,dtype=torch.float16,generator=generator)*.02
            inject=torch.randn((4,10240),device=device,dtype=torch.float16,generator=generator)*.02
            up=torch.randn((10240,320),device=device,dtype=torch.float16,generator=generator)*.02
        def hc(injecting=True):
            d=GGUFConcat([GGUFDense(down),GGUFDense(inject)]) if injecting else GGUFDense(down)
            return SimpleNamespace(down=d,up=GGUFDense(up),inject=injecting)
        weights=SimpleNamespace(device=device,meta={'world':8,'execution_provider':'sm70'},
            layers=[SimpleNamespace(attn_hc=hc(),mlp_hc=hc()) for _ in range(48)],mixer=hc(False))
        registered=[]
        try:
            owner=candidate.Owner.prepare(candidate.Selection(path,sha),
                {'hidden_size':2560,'hc_lowrank':320,'hc_count':4,'num_hidden_layers':48},
                {'decode':40,'prefill':40},device,register=registered.append)
            owner.bind_weights(weights)
            decoder=SimpleNamespace(buf=Buffers('decode',40,device),pbuf=Buffers('prefill',40,device))
            owner.adopt(weights,decoder)
            observations={}
            for role,b in [('decode',decoder.buf),('prefill',decoder.pbuf)]:
                b.normed.copy_(torch.randn(b.normed.shape,device=device,dtype=torch.float16,generator=generator)*.02)
                b.act.copy_(torch.randn(b.act.shape,device=device,dtype=torch.float16,generator=generator)*.02)
                frame=b.dense_linears;h=weights.layers[0].attn_hc
                def run(rows,frame=frame,h=h,b=b,role=role):
                    result=(frame.partials(h.down,b.normed[:rows])[0] if role=='decode'
                            else frame.matmul(b.normed[:rows],h.down,b.dn[:rows]))
                    frame.matmul(b.act[:rows],h.up,b.up[:rows])
                    return result
                baseline=run(40).clone();baseline_up=b.up.clone()
                expected=h.down.matmul_reference(b.normed,f32=role=='decode')
                expected_up=h.up.matmul_reference(b.act)
                for name,actual,reference in [('down',baseline,expected),('up',baseline_up,expected_up)]:
                    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
                    observations[role+'-'+name]={'elements':actual.numel(),
                        'unequal_elements_by_bits':int((actual.view(torch.int32 if actual.dtype==torch.float32 else torch.int16)
                            !=reference.view(torch.int32 if reference.dtype==torch.float32 else torch.int16)).sum().item()),
                        'max_absolute':float((actual.float()-reference.float()).abs().max().item()),
                        'quality_acceptance':False}
                for rows in (1,5,32):
                    result=run(rows)
                    assert torch.equal(result.view(torch.uint8),baseline[:rows].view(torch.uint8))
                    assert torch.equal(b.up[:rows].view(torch.uint8),baseline_up[:rows].view(torch.uint8))
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):run(40)
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):captured=run(40)
                b.normed.mul_(.5);b.act.mul_(1.5)
                graph.replay();torch.cuda.synchronize()
                changed=captured.clone();changed_up=b.up.clone()
                eager=run(40);torch.cuda.synchronize()
                assert torch.equal(changed.view(torch.uint8),eager.view(torch.uint8))
                assert torch.equal(changed_up.view(torch.uint8),b.up.view(torch.uint8))
                graph.reset();graph=None
            record_property('unqualified-numerical-observations',json.dumps(observations,sort_keys=True))
        except BaseException as error:  # noqa: BLE001 - retain the primary through all GPU retirements.
            primary=error
        errors=[]
        try:torch.cuda.synchronize()
        except BaseException as error:  # noqa: BLE001 - preserve cleanup failures with the primary.
            errors.append(error)
        if graph is not None:
            try:graph.reset()
            except BaseException as error:  # noqa: BLE001 - preserve independent retirement failures.
                errors.append(error)
        for registered_owner in registered:
            try:registered_owner.close()
            except BaseException as error:  # noqa: BLE001 - preserve independent retirement failures.
                errors.append(error)
        from tensorfold.cleanup import raise_failures
        raise_failures(primary,errors)
