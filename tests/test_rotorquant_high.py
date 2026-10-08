"""Independent original-RMS7/8 wire, Gaussian-book and FP32 envelope checks."""
import bisect
import math
import struct

import pytest

from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as ref
from tensorfold.families.qwen4_exp.kv_formats import get
from test_rotorquant_wide import certified_source,scale_interval,wide_index_envelope


def unpack_independent(payload,bits):
    if bits == 8:
        return list(payload)
    if len(payload)%112:
        raise ValueError("seven-bit integer oracle requires complete112-byte groups")
    values = []
    for start in range(0,len(payload),112):
        low = int.from_bytes(payload[start:start+64],"little")
        mid = int.from_bytes(payload[start+64:start+96],"little")
        high = int.from_bytes(payload[start+96:start+112],"little")
        values.extend(((low>>(4*i))&15)|(((mid>>(2*i))&3)<<4)|(((high>>i)&1)<<6) for i in range(128))
    return values


@pytest.mark.parametrize("bits",[7,8])
def test_high_precision_format_binary32_gaussian_tables_and_original_rms_identity(bits):
    book,thresholds = ref.centroids(bits),ref.thresholds(bits)
    assert len(book) == 1 << bits and len(thresholds) == (1 << bits)-1
    assert tuple(-x for x in reversed(book)) == book and list(book) == sorted(book)
    assert thresholds[(1 << (bits-1))-1] == 0
    assert all(x == struct.unpack("<f",struct.pack("<f",x))[0] for x in book+thresholds)
    fmt = get(f"rotorquant{bits}")
    assert fmt.codec == 4 and fmt.bits == bits and fmt.group == 128 and fmt.scale_bytes == 4
    assert fmt.value_bytes(256) == 2*(128*bits//8+4)
    assert fmt.rotor.norm_policy == "original-rms" and fmt.rotor != get("rotorquant6").rotor
    edges = [-math.inf,*[(a+b)/2 for a,b in zip(book,book[1:])],math.inf]
    for value,a,b in zip(book,edges,edges[1:]):
        numerator = ((0 if math.isinf(a) else math.exp(-a*a/2))
                     - (0 if math.isinf(b) else math.exp(-b*b/2)))/math.sqrt(2*math.pi)
        mass = (math.erfc(-b/math.sqrt(2))-math.erfc(-a/math.sqrt(2)))/2
        assert abs(value-numerator/mass) < 2e-6
    for index,threshold in enumerate(thresholds):
        assert bisect.bisect_left(thresholds,threshold) == index
    for variant in ("planar","isofast","iso64-norm","signed-iso64-norm","signed-iso128-norm","signed-iso128-outlier-norm"):
        with pytest.raises(ValueError):
            ref.descriptor(bits,variant)


@pytest.mark.parametrize("bits",[7,8])
@pytest.mark.parametrize("offset",[0,63,64,127,128,255])
def test_high_precision_integer_planes_cover_every_code_at_group_and_plane_boundaries(bits,offset):
    for code in range(1 << bits):
        values = [(1 << (bits-1))-1]*256
        values[offset] = code
        payload = ref.pack_indices_ref(values,bits)
        assert len(payload) == 256*bits//8
        assert unpack_independent(payload,bits) == values == ref.unpack_indices_ref(payload,bits)


@pytest.mark.parametrize("bits",[7,8])
def test_high_precision_certified_inverse_payload_and_original_rms_staged_bounds(bits):
    torch = pytest.importorskip("torch")
    source,expected = certified_source(4,ref.centroids(bits),ref.thresholds(bits))
    lower,upper = wide_index_envelope(source,4,ref.thresholds(bits))
    assert lower == upper == expected
    x = torch.tensor(source,dtype=torch.bfloat16).repeat(2).reshape(1,256)
    packed,metadata = ref.quantize_ref(x,bits,"signed-iso128")
    assert unpack_independent(bytes(packed.flatten().tolist()),bits) == expected*2
    low,high = ref.index_envelope_oracle(x.float().flatten().tolist(),bits,"signed-iso128",rms=metadata.flatten().tolist())
    assert low == high == expected*2
    raw = math.hypot(*source)/math.sqrt(128)
    for norm in metadata.flatten().tolist():
        interval = scale_interval(source.tolist(),expected,ref.centroids(bits),False)
        assert interval[0] <= norm <= interval[1] and math.isclose(norm,raw,rel_tol=1e-6)
    packed,metadata = ref.quantize_ref(torch.zeros((1,256),dtype=torch.bfloat16),bits,"signed-iso128")
    assert metadata.eq(0).all()
    zero = (1 << (bits-1))-1
    assert unpack_independent(bytes(packed.flatten().tolist()),bits) == [zero]*256
    if bits == 7:
        assert packed.flatten().tolist() == ([255]*96+[0]*16)*2
    else:
        assert packed.flatten().tolist() == [127]*256
    assert ref.dequant_ref(packed,metadata,bits,"signed-iso128",inverse=True).eq(0).all()
    assert not torch.cuda.is_initialized()
