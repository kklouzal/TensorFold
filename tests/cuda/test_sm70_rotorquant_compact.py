"""Actual-SM70 checks for the pinned compact Planar4 pack/unpack ABI."""
import pytest
import torch
import json
from pathlib import Path
import struct
import itertools

from tensorfold.cuda.sm70 import rotorquant_compact as compact
from tensorfold.cuda.sm70 import rotorquant_q4_native as native
from tensorfold.families.qwen4_exp.kv_formats import KVPairFormat, get
from tensorfold.families.qwen4_exp.cuda.sm70_kv import FORMATS as SM70_FORMATS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason='requires the actual SM70 native codec')


GOLDENS = json.loads((Path(__file__).parents[1] / 'fixtures/rotorquant_planar4_sm70.json').read_text())
COMPACT_PAIRS = tuple(KVPairFormat(get(key), get(value)) for key, value in itertools.product(SM70_FORMATS, repeat=2)
                      if compact.NAME in (key, value) and compact.ISO_NAME not in (key, value))


@pytest.mark.parametrize('case', GOLDENS['fixtures'], ids=lambda case: case['name'])
def test_compact_native_direct_pack_and_decode_against_independent_block_oracle(case):
    raw = bytes.fromhex(case['input_fp16_le_hex'])
    values = struct.unpack('<' + 'e' * (len(raw)//2), raw)
    dims = len(values)
    source = torch.tensor(values, device='cuda').half().reshape(1,1,dims)
    codes = torch.empty(1,1,dims//2, dtype=torch.uint8, device='cuda')
    metadata = torch.empty(1,1,dims//128, device='cuda')
    status = torch.tensor([0,-2], dtype=torch.int32, device='cuda')
    native.encode(source, codes, metadata, codec=9, status=status)
    actual_codes = codes.cpu().reshape(-1,64)
    actual_words = metadata.cpu().view(torch.uint8).reshape(-1,4)
    wire = b''.join(bytes(word.tolist())+bytes(payload.tolist()) for word,payload in zip(actual_words,actual_codes))
    assert wire == bytes.fromhex(case['output_blocks_hex'])
    assert wire == compact.encode_oracle(source.cpu().flatten().tolist())
    assert status.cpu().tolist() == [0,-2]
    out = torch.empty_like(source)
    native.decode(codes,metadata,out,codec=9,inverse=True,status=status)
    assert torch.equal(out.cpu().flatten(), torch.tensor(compact.decode_oracle(wire)).half())
    assert status.cpu().tolist() == [0,-2]


def test_compact_native_graph_live_mask_and_reserved_metadata_validation():
    native.extension()
    source = torch.ones(2,1,128,device='cuda').half()
    codes = torch.full((4,1,64),91,dtype=torch.uint8,device='cuda')
    metadata = torch.zeros(4,1,1,device='cuda')
    status = torch.tensor([0,-2],dtype=torch.int32,device='cuda')
    position = torch.tensor([1],dtype=torch.int32,device='cuda')
    live = torch.tensor([1],dtype=torch.int32,device='cuda')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.encode(source,codes,metadata,codec=9,position=position,status=status,valid_rows=live)
    graph.replay()
    assert codes[0].eq(91).all() and codes[2:].eq(91).all()
    before = codes.clone()
    live.zero_()
    graph.replay()
    assert torch.equal(codes,before)
    metadata.view(torch.int32)[1,0,0] |= 1 << 16
    out = torch.empty(4,1,128,device='cuda').half()
    native.decode(codes,metadata,out,codec=9,status=status)
    assert status.cpu().tolist() == [8,0]


@pytest.mark.parametrize('pair', COMPACT_PAIRS,
                         ids=lambda pair: f'K-{pair.key_dtype}__V-{pair.value_dtype}')
def test_compact_native_attention_consumes_packed_metadata_and_inverse_value_basis(pair):
    from types import SimpleNamespace
    from tensorfold.cuda.sm70.attention_native import attention
    from tensorfold.cuda.sm70 import family_ops as ops
    from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as ref

    torch.manual_seed(14)
    keys, rows, dim, heads = 67, 3, 256, 3
    key_source = torch.randn(keys,1,dim).half()
    value_source = torch.randn_like(key_source)

    def encode(source, format):
        if format.compact is not None:
            payload, metadata = compact.quantize_ref(source)
            stored = compact.dequant_ref(payload, metadata).half()
        elif format.codec:
            payload, metadata = ref.quantize_ref(source, 4, format.rotor.variant)
            stored = ref.dequant_ref(payload, metadata, 4, format.rotor.variant).half()
        elif format.bits == 8:
            payload, metadata = ops.quantize_int8(source)
            stored = ops.dequantize_int8(payload, metadata)
        else:
            payload, metadata, stored = source, None, source
        return payload.cuda(), None if metadata is None else metadata.cuda(), stored

    kc, ks, _ = encode(key_source, pair.key)
    vc, vs, values = encode(value_source, pair.value)
    selected = torch.arange(rows,device='cuda',dtype=torch.int32)
    scratch = SimpleNamespace(out=torch.empty(rows,heads,dim,device='cuda').half(),
                              sparse=torch.ones(rows,dtype=torch.int32,device='cuda'),
                              nk=torch.ones(rows,dtype=torch.int32,device='cuda'),
                              ids=selected[:,None].contiguous(),qsa=True,capacity=keys)
    query = torch.randn(rows,heads,dim).half()
    if pair.key.compact is not None:
        query = compact.rotate_ref(query).half()
    elif pair.key.codec:
        query = ref.rotate_ref(query, pair.key.rotor.variant).half()
    elif pair.key.quantized:
        query = ops.h32(query).half()
    actual = attention(query.cuda(),kc,vc,torch.tensor([64],dtype=torch.int32,device='cuda'),
                       scratch,rows,dim**-.5,scratch.out,keys,ks,vs,
                       0 if not pair.key.quantized else pair.key.bits,
                       0 if not pair.value.quantized else pair.value.bits,
                       kcodec=pair.key.codec,vcodec=pair.value.codec)
    # Each row selects exactly one causal key: its global softmax probability
    # is exactly one. This independently checks mixed dispatch, wire decoding
    # and the final value-side inverse without a fitted GEMM error tolerance.
    expected = values[:rows].float()
    if pair.value.compact is not None:
        expected = compact.rotate_ref(expected, inverse=True)
    elif pair.value.codec:
        expected = ref.rotate_ref(expected, pair.value.rotor.variant, inverse=True)
    elif pair.value.quantized:
        expected = ops.h32(expected)
    assert torch.equal(actual.cpu(),expected.half().expand(rows,heads,dim))
