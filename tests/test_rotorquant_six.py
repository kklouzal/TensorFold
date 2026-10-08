"""Independent six-bit plane interpretation, Gaussian tables and encoder bounds."""
import bisect
import math
import struct

import pytest

from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as ref
from tensorfold.families.qwen4_exp.kv_formats import get
from test_rotorquant_wide import certified_source, scale_interval, wide_index_envelope


def unpack_six_independent(payload):
    # Each entire plane is one little-endian bit integer. This avoids the
    # optimized decoder's byte/stride/address implementation.
    assert len(payload)%96 == 0
    output=[]
    for start in range(0,len(payload),96):
        low=int.from_bytes(payload[start:start+64],"little")
        high=int.from_bytes(payload[start+64:start+96],"little")
        output.extend(((low>>(4*i))&15) | (((high>>(2*i))&3)<<4) for i in range(128))
    return output


def test_six_format_tables_and_original_norm_identity():
    book,thresholds=ref.centroids(6),ref.thresholds(6)
    assert len(book)==64 and len(thresholds)==63
    assert tuple(-x for x in reversed(book))==book
    assert all(x==struct.unpack("<f",struct.pack("<f",x))[0] for x in book+thresholds)
    assert thresholds[31]==0 and list(book)==sorted(book)
    fmt=get("rotorquant6")
    assert fmt.bits==6 and fmt.codec==4 and fmt.group==128 and fmt.scale_bytes==4
    assert fmt.value_bytes(256)==200 and fmt.rotor.norm_policy=="original-rms"
    assert fmt.rotor.vector_bytes_per_group==100
    assert fmt.rotor != ref.descriptor(4,"signed-iso128")
    edges=[-math.inf,*[(a+b)/2 for a,b in zip(book,book[1:])],math.inf]
    def phi(x):
        return 0. if math.isinf(x) else math.exp(-x*x/2)/math.sqrt(2*math.pi)
    def cdf(x):
        return math.erfc(-x/math.sqrt(2))/2
    for value,a,b in zip(book,edges,edges[1:]):
        conditional=(phi(a)-phi(b))/(cdf(b)-cdf(a))
        assert abs(value-conditional)<2e-6


@pytest.mark.parametrize("offset",[0,63,64,127,128,255])
def test_six_plane_integer_oracle_covers_all_codes_at_boundaries(offset):
    for code in range(64):
        values=[31]*256
        values[offset]=code
        payload=ref.pack_indices_ref(values,6)
        assert len(payload)==192 and unpack_six_independent(payload)==values
        assert ref.unpack_indices_ref(payload,6)==values


def test_six_midpoints_zero_and_staged_certified_payload():
    torch=pytest.importorskip("torch")
    for index,threshold in enumerate(ref.thresholds(6)):
        assert bisect.bisect_left(ref.thresholds(6),threshold)==index
    source,expected=certified_source(4,ref.centroids(6),ref.thresholds(6))
    lower,upper=wide_index_envelope(source,4,ref.thresholds(6))
    assert lower==upper==expected
    x=torch.tensor(source,dtype=torch.bfloat16).repeat(4).reshape(2,256)
    payload,metadata=ref.quantize_ref(x,6,"signed-iso128")
    for row,packed,norm in zip(x.float().tolist(),payload.tolist(),metadata.tolist()):
        codes=unpack_six_independent(bytes(packed))
        assert codes==expected*2
        low,high=ref.index_envelope_oracle(row,6,"signed-iso128",rms=norm)
        assert low==high==codes
        for group in range(2):
            interval=scale_interval(row[group*128:(group+1)*128],codes[group*128:(group+1)*128],ref.centroids(6),False)
            assert interval[0]<=norm[group]<=interval[1]
    packed,norm=ref.quantize_ref(torch.zeros((1,128),dtype=torch.bfloat16),6,"signed-iso128")
    assert packed.flatten().tolist()==[255]*64+[85]*32 and norm.item()==0
    assert unpack_six_independent(bytes(packed.flatten().tolist()))==[31]*128
    assert not torch.cuda.is_initialized()
