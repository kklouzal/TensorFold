"""Pinned ISO4 wire semantics and SM70-only initial profile; no hardware claim."""
from pathlib import Path
import json
import math
import struct

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.cuda.sm70 import rotorquant_compact as compact
from tensorfold.families.qwen4_exp.kv_formats import get, get_pair
from tensorfold.families.qwen4_exp.cuda.sm70_kv import check_pair, check_kernel_fields
from tests.test_sm70_rotorquant_compact import _source_wire

GOLDENS = json.loads((Path(__file__).parent / 'fixtures/rotorquant_iso4_sm70.json').read_text())


def test_iso4_is_a_distinct_pinned_compact_68byte_format_not_custom_iso_or_planar():
    format = get(compact.ISO_NAME)
    assert (format.bits, format.codec, format.group, format.scale_bytes) == (4, 10, 128, 4)
    assert format.compact.variant == 'cuda-iso4-compact'
    assert format.compact.norm_policy == 'fp16-corrected-l2-reserved-zero'
    assert format.compact.vector_bytes_per_group == 68
    assert format.value_bytes(256) == 136 and get_pair(compact.ISO_NAME).row_bytes(1, 256) == 272
    assert format.storage_uid not in {get(name).storage_uid for name in
                                     ('rotorquant-iso4', compact.NAME, 'rotorquant-signed128-4')}
    assert check_pair(get_pair(compact.ISO_NAME), head_dim=256)
    assert check_kernel_fields(4, 10, 4, 10) == (4, 4)


def test_iso4_reference_provenance_pins_writer_reader_and_actual_quaternion_tables():
    files = {record['file']: record['sha256'] for record in GOLDENS['reference_files']}
    assert GOLDENS['reference_revision'] == compact.SOURCE_REVISION
    assert files['ggml/src/ggml-cuda/set-rows-planar-iso.cuh'] == compact.SOURCE_SET_ROWS_SHA256
    assert files['ggml/src/ggml-cuda/fattn-common.cuh'] == compact.SOURCE_ATTENTION_SHA256
    assert files['ggml/src/ggml-cuda/planar-iso-constants.cuh'] == compact.SOURCE_CONSTANTS_SHA256
    assert len({(w,x,y,z) for w,x,y,z in zip(compact.QW,compact.QX,compact.QY,compact.QZ)}) == 32


@pytest.mark.parametrize('case', GOLDENS['fixtures'], ids=lambda case: case['name'])
def test_iso4_encoding_and_scalar_decoder_match_unmodified_original_source(case):
    raw = bytes.fromhex(case['input_fp16_le_hex'])
    values = struct.unpack('<' + 'e' * (len(raw)//2), raw)
    source = torch.tensor(values).reshape(1,1,-1)
    payload, metadata = compact.quantize_ref(source, codec=10)
    wire = _source_wire(payload, metadata)
    expected = bytes.fromhex(case['output_blocks_hex'])
    assert wire == expected == compact.encode_oracle(values, codec=10)
    actual = compact.decode_oracle(wire, codec=10)
    assert struct.pack('<'+'f'*len(actual), *actual) == bytes.fromhex(case['decoded_fp32_le_hex'])
    assert all(wire[start+2:start+4] == b'\0\0' for start in range(0,len(wire),68))


def _source_staging_bound(wire):
    """A priori bound for scale-before/after inverse-dot FP32 staging.

    Each evaluated path has at most eight rounded operations on a contribution
    (four products, three adds, and scale). Both are within gamma8 times the
    absolute dot contributions; their difference is bounded by twice that.
    The absolute term covers gradual-underflow rounding, and binary64 bound
    evaluation gets its own outward envelope. No empirical tolerance is fitted.
    """
    unit = 2.**-24
    gamma8 = 8*unit/(1-8*unit)
    bounds = []
    for start in range(0,len(wire),68):
        norm = struct.unpack('<e',wire[start:start+2])[0]
        codes = [value for byte in wire[start+4:start+68] for value in (byte&15,byte>>4)]
        for group in range(32):
            w,x,y,z = (table[group] for table in (compact.QW,compact.QX,compact.QY,compact.QZ))
            matrix = ((w,x,y,z),(-x,w,z,-y),(-y,-z,w,x),(-z,y,-x,w))
            values = [compact.CENTROIDS[code] for code in codes[4*group:4*group+4]]
            for row in matrix:
                absolute = norm*math.fsum(abs(a*b) for a,b in zip(row,values))
                bound = 2*gamma8*absolute + 32*2.**-150*(norm+1) + 32*2.**-53*absolute
                bounds.append(math.nextafter(bound, math.inf))
    return torch.tensor(bounds,dtype=torch.float64)


@pytest.mark.parametrize('case', GOLDENS['fixtures'], ids=lambda case: case['name'])
def test_iso4_vector_reference_has_the_declared_source_staging_error_bound(case):
    raw = bytes.fromhex(case['input_fp16_le_hex'])
    source = torch.tensor(struct.unpack('<'+'e'*(len(raw)//2),raw)).reshape(1,1,-1)
    payload, metadata = compact.quantize_ref(source,codec=10)
    actual = compact.dequant_ref(payload,metadata,codec=10,inverse=True).flatten().double()
    expected = torch.tensor(struct.unpack('<'+'f'*(len(raw)//2),bytes.fromhex(case['decoded_fp32_le_hex'])),
                            dtype=torch.float64)
    assert ((actual-expected).abs() <= _source_staging_bound(bytes.fromhex(case['output_blocks_hex']))).all()


@pytest.mark.parametrize('other', ['fp16','int8','rotorquant-signed128-4','rotorquant-signed64-norm4',compact.NAME])
@pytest.mark.parametrize('key_side', [True,False])
def test_initial_iso4_profile_rejects_unsupported_mixed_variants(other,key_side):
    key,value = (compact.ISO_NAME,other) if key_side else (other,compact.ISO_NAME)
    pair = get_pair('int8',key,value)
    with pytest.raises(ValueError,match='Iso4|codec10'):
        check_pair(pair)
    fields = [field for side in (pair.key,pair.value) for field in
              (0 if side.bits==16 else side.bits,side.codec)]
    with pytest.raises(ValueError,match='Iso4|codec10'):
        check_kernel_fields(*fields)


def test_iso4_cpu_family_has_live_prefix_control_and_sticky_failure_without_requantization():
    codes = torch.full((5,1,128),91,dtype=torch.uint8)
    metadata = torch.zeros(5,1,2)
    source = torch.ones(3,1,256).half()
    status = torch.tensor([0,-2],dtype=torch.int32)
    ops._rotor_encode(source,codes,metadata,torch.tensor([2]),codec=10,
                      valid_rows=torch.tensor([1]),status=status,layer=7)
    expected,words = compact.quantize_ref(source[:1],codec=10)
    assert torch.equal(codes[2:3],expected)
    assert torch.equal(metadata[2:3].view(torch.int32),words.view(torch.int32))
    assert codes[:2].eq(91).all() and codes[3:].eq(91).all()
    assert status.tolist()==[0,-2]
    ops._rotor_encode(torch.full((1,1,256),60000.),codes,metadata,torch.tensor([0]),
                      codec=10,status=status,layer=7)
    assert status.tolist()==[8,7]
    assert metadata[0].view(torch.int32).eq(0).all()
