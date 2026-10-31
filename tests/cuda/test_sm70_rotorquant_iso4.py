"""Actual-SM70 ISO4 gates; compilation/CPU checks do not qualify these tests."""
from pathlib import Path
from types import SimpleNamespace
import json
import struct

import pytest
import torch

from tensorfold.cuda.sm70 import rotorquant_compact as compact
from tensorfold.cuda.sm70 import rotorquant_q4_native as native

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7,0),
    reason='requires actual SM70 ISO4 native execution')
GOLDENS = json.loads((Path(__file__).parents[1] / 'fixtures/rotorquant_iso4_sm70.json').read_text())


@pytest.mark.parametrize('case', GOLDENS['fixtures'], ids=lambda case: case['name'])
def test_iso4_native_direct_wire_and_decoded_working_basis_match_independent_source(case):
    raw = bytes.fromhex(case['input_fp16_le_hex'])
    values = struct.unpack('<'+'e'*(len(raw)//2),raw)
    dims = len(values)
    source = torch.tensor(values,device='cuda').half().reshape(1,1,dims)
    codes = torch.empty(1,1,dims//2,dtype=torch.uint8,device='cuda')
    words = torch.empty(1,1,dims//128,device='cuda')
    status = torch.tensor([0,-2],dtype=torch.int32,device='cuda')
    native.encode(source,codes,words,codec=10,status=status)
    wire = b''.join(bytes(word.tolist())+bytes(code.tolist()) for word,code in zip(
        words.cpu().view(torch.uint8).reshape(-1,4),codes.cpu().reshape(-1,64)))
    assert wire == bytes.fromhex(case['output_blocks_hex'])
    assert wire == compact.encode_oracle(values,codec=10)
    out = torch.empty_like(source,dtype=torch.float32)
    native.decode(codes,words,out,codec=10,status=status)
    expected = compact.dequant_ref(codes.cpu(),words.cpu(),codec=10)
    assert torch.equal(out.cpu().view(torch.int32),expected.view(torch.int32))
    native.decode(codes,words,out,codec=10,inverse=True,status=status)
    expected = compact.dequant_ref(codes.cpu(),words.cpu(),codec=10,inverse=True)
    assert torch.equal(out.cpu().view(torch.int32),expected.view(torch.int32))
    assert status.cpu().tolist()==[0,-2]


@pytest.mark.parametrize('dims',[128,256])
@pytest.mark.parametrize('inverse',[False,True])
def test_iso4_native_rotation_exercises_every_quaternion_component_in_each_group(dims,inverse):
    source = (torch.arange(3*2*dims,dtype=torch.float32).remainder(129)/32-2).reshape(3,2,dims)
    out = torch.empty_like(source,device='cuda')
    status = torch.tensor([0,-2],dtype=torch.int32,device='cuda')
    native.rotate(source.cuda(),out,codec=10,inverse=inverse,status=status)
    expected = compact.rotate_ref(source,codec=10,inverse=inverse)
    assert torch.equal(out.cpu().view(torch.int32),expected.view(torch.int32))
    assert status.cpu().tolist()==[0,-2]


def test_iso4_native_graph_live_prefix_reserved_metadata_and_sticky_failure():
    native.extension()
    source = torch.ones(3,1,256,device='cuda').half()
    codes = torch.full((5,1,128),91,dtype=torch.uint8,device='cuda')
    words = torch.zeros(5,1,2,device='cuda')
    status = torch.tensor([0,-2],dtype=torch.int32,device='cuda')
    position = torch.tensor([1],dtype=torch.int64,device='cuda')
    live = torch.tensor([1],dtype=torch.int32,device='cuda')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.encode(source,codes,words,codec=10,position=position,status=status,valid_rows=live)
    graph.replay()
    assert codes[0].eq(91).all() and codes[2:].eq(91).all()
    before = codes.clone(),words.clone()
    live.zero_()
    graph.replay()
    assert torch.equal(codes,before[0]) and torch.equal(words.view(torch.int32),before[1].view(torch.int32))
    words.view(torch.int32)[1,0,0] |= 1 << 16
    out = torch.empty(5,1,256,device='cuda').half()
    native.decode(codes,words,out,codec=10,status=status,layer=7)
    assert status.cpu().tolist()==[8,7]
    native.rotate(source,torch.empty_like(source),codec=10,status=status,layer=2)
    assert status.cpu().tolist()==[8,7]


@pytest.mark.parametrize('rows,base',[(1,3),(3,3),(17,3),(3,262136)])
def test_iso4_native_fused_preparation_preserves_raw_boundaries_and_exact_standalone_codec(rows,base):
    from tests.cuda.test_sm70_rotorquant_prep import fixture,raw_reference,invoke

    case = fixture(dim=256,rows=rows,capacity=base+rows+5,base=base,kcodec=10,vcodec=10)
    raw = raw_reference(case)
    invoke(case)
    query = compact.rotate_ref(raw[0].cpu(),codec=10).half()
    assert torch.equal(case.query.cpu().view(torch.int16),query.view(torch.int16))
    assert torch.equal(case.iq,raw[3])
    assert torch.equal(case.index[base:base+rows],raw[4][base:base+rows])
    for original,data,words in ((raw[1],case.key,case.ks),(raw[2],case.value,case.vs)):
        expected = torch.full_like(data,7)
        expected_words = torch.full_like(words,7)
        native.encode(original[base:base+rows],expected,expected_words,position=case.position,codec=10)
        assert torch.equal(data,expected)
        assert torch.equal(words.view(torch.int32),expected_words.view(torch.int32))
        assert data[:base].eq(7).all() and data[base+rows:].eq(7).all()
    assert case.status.cpu().tolist()==[0,-2]


def test_iso4_native_attention_and_cpp_boundary_refuse_unsupported_mixed_profile():
    from tensorfold.cuda.sm70.attention_native import attention,extension
    torch.manual_seed(21)
    keys,rows,heads,dims = 67,3,3,256
    value = torch.randn(keys,1,dims).half()
    payload,metadata = compact.quantize_ref(value,codec=10)
    codes,words = payload.cuda(),metadata.cuda()
    query = compact.rotate_ref(torch.randn(rows,heads,dims).half(),codec=10).half().cuda()
    position = torch.tensor([64],dtype=torch.int32,device='cuda')
    scratch = SimpleNamespace(out=torch.empty_like(query),sparse=torch.ones(rows,dtype=torch.int32,device='cuda'),
                              nk=torch.ones(rows,dtype=torch.int32,device='cuda'),
                              ids=torch.arange(rows,dtype=torch.int32,device='cuda')[:,None],qsa=True,capacity=keys)
    output = attention(query,codes,codes,position,scratch,rows,dims**-.5,scratch.out,
                       keys,words,words,4,4,kcodec=10,vcodec=10)
    stored = compact.dequant_ref(payload[:rows],metadata[:rows],codec=10).half()
    expected = compact.rotate_ref(stored.float(),codec=10,inverse=True).half()
    assert torch.equal(output.cpu(),expected.expand(rows,heads,dims))
    workspace = scratch.native_workspace
    operands = (query,codes,codes,position,scratch.sparse,scratch.nk,scratch.ids,
                workspace.scores,workspace.probabilities,workspace.partials,output,words,words,
                rows,keys,4,4,True,dims**-.5)
    before = output.clone()
    for kcodec,vcodec in ((10,9),(9,10)):
        with pytest.raises(RuntimeError,match='symmetric'):
            extension().run_rotor(*operands,kcodec,vcodec)
    assert torch.equal(output,before)
