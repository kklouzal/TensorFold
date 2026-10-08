"""Independent six-bit corrected/outlier IEEE754 index and metadata contracts."""
from __future__ import annotations

import bisect
import math

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as ref
from tensorfold.families.qwen4_exp.kv_formats import get
from test_rotorquant_wide import MASK, COS, certified_source, scale_interval, wide_index_envelope, wide_matrix

CASES = ((6, "signed-iso128-norm", "rotorquant6-norm"),
         (7, "signed-iso128-outlier-norm", "rotorquant6-outlier-norm"))


def rn_sum(values, tree):
    """Alternative legal RN trees, independent of Torch/Triton reduction."""
    terms = np.asarray(values, dtype=np.float32)
    if tree in ("forward", "reverse"):
        if tree == "reverse":
            terms = terms[::-1]
        total = terms[0]
        for term in terms[1:]:
            total = np.float32(total + term)
        return total
    while len(terms) > 1:
        terms = np.add(terms[::2], terms[1::2], dtype=np.float32)
    return terms[0]


def independently_staged(values, codec, tree):
    """Interpret the declared linear stages via signed matrix rows and RN trees.

    The affine stage rows, metadata energy and original RMS are evaluated by
    separate loops; no optimized/staged reference helper supplies intermediates.
    Returned coordinates are those used by scalar midpoint selection.
    """
    values = np.asarray(values, dtype=np.float32)
    maximum = np.max(np.abs(values))
    z = np.divide(values, maximum if maximum else np.float32(1), dtype=np.float32)
    squares = np.multiply(z, z, dtype=np.float32)
    t = np.float32(math.sqrt(float(np.float32(rn_sum(squares, tree) * np.float32(1/128)))))
    raw_rms = np.float32(maximum * t)
    coordinates = np.divide(z, t if t else np.float32(1), dtype=np.float32)
    coordinates *= np.array([1 - 2*((MASK >> (i//2)) & 1) for i in range(128)], dtype=np.float32)
    rows = np.array([[1,-1,-1,-1], [1,1,-1,1], [1,1,1,-1], [1,-1,1,1]], dtype=np.float32)
    for low, high in ((0,1), (2,3), (4,5)):
        before = coordinates.copy()
        bitmask = (1 << low) | (1 << high)
        for base in range(128):
            if base & bitmask:
                continue
            positions = [base, base|(1<<low), base|(1<<high), base|bitmask]
            for position,row in zip(positions,rows):
                terms = row * before[positions]
                coordinates[position] = np.float32(rn_sum(terms, tree) * np.float32(.5))
    before = coordinates.copy()
    for base in range(64):
        for position,signs in ((base,(1,-1)), (base+64,(1,1))):
            terms = np.multiply(before[[base,base+64]], np.array(signs, dtype=np.float32)*np.float32(COS), dtype=np.float32)
            coordinates[position] = rn_sum(terms, tree)
    alpha = np.float32(1.)
    if codec == 7:
        alpha = max(alpha,np.float32(np.max(np.abs(coordinates))/np.float32(ref.CENTROIDS_6[-1])))
        coordinates = np.divide(coordinates, alpha, dtype=np.float32)
    codes = [bisect.bisect_left(ref.THRESHOLDS_6, float(x)) for x in coordinates]
    centers = np.asarray([ref.CENTROIDS_6[code] for code in codes], dtype=np.float32)
    energy = rn_sum(np.multiply(centers, centers, dtype=np.float32), tree)
    factor = np.float32(math.sqrt(float(np.float32(np.float32(128)/energy))))
    stored = np.float32(raw_rms*factor)
    return coordinates.astype(np.float64), codes, float(stored), float(alpha)


def source_fixture(kind):
    if kind == "zero":
        return np.zeros(128,dtype=np.float32)
    if kind == "random":
        return np.random.default_rng(192837).normal(size=128).astype(np.float32)
    if kind == "large":
        return (np.random.default_rng(73).normal(size=128)*2**20).astype(np.float32)
    rotated = np.full(128,math.sqrt((128-5**2)/127))
    rotated[0] = 5
    if kind in ("alpha-below", "alpha-above"):
        rotated[0] = ref.CENTROIDS_6[-1] * (1 + (-1 if kind == "alpha-below" else 1)*2**-20)
        rotated[1:] = math.sqrt((128-rotated[0]**2)/127)
    matrix = wide_matrix(4)
    return (rotated @ matrix / (2*COS*COS)).astype(np.float32)


@pytest.mark.parametrize("codec,variant,name",CASES)
def test_corrected_six_format_has_distinct_full_identity_and_identical_rotation(codec,variant,name):
    fmt = get(name)
    assert ref.variant_id(variant) == codec == fmt.codec
    assert ref.norm_corrected(variant) and ref.outlier_scaled(variant) == (codec == 7)
    assert fmt.bits == 6 and fmt.value_bytes(256) == 200
    assert fmt.rotor.norm_policy == "reconstruction-norm"
    assert fmt.rotor != get("rotorquant6").rotor
    assert get("rotorquant6-norm").rotor != get("rotorquant6-outlier-norm").rotor
    source = source_fixture("random")
    assert ref.rotate_oracle(source.tolist(),variant) == ref.rotate_oracle(source.tolist(),"signed-iso128")
    assert ref.rotate_oracle(source.tolist(),variant,inverse=True) == ref.rotate_oracle(source.tolist(),"signed-iso128",inverse=True)
    for bits in (3,4):
        with pytest.raises(ValueError,match="six-bit"):
            ref.descriptor(bits,variant)
        with pytest.raises(ValueError,match="six-bit"):
            ref.rounding_envelope_oracle(source.tolist(),variant,bits=bits)


@pytest.mark.parametrize("codec,variant,name",CASES)
@pytest.mark.parametrize("tree",("forward","reverse","pairwise"))
@pytest.mark.parametrize("kind",("zero","random","large","outlier","alpha-below","alpha-above"))
def test_corrected_six_independent_rn_trees_satisfy_coordinate_bin_and_scale_bounds(codec,variant,name,tree,kind):
    del name
    source = source_fixture(kind)
    actual,codes,scale,alpha = independently_staged(source,codec,tree)
    ideal,error,nominal,norm_error = ref.rounding_envelope_oracle(source.tolist(),variant,bits=6,rms=[scale])
    assert np.all(np.abs(actual-np.asarray(ideal)) <= np.asarray(error))
    low,high = ref.index_envelope_oracle(source.tolist(),6,variant,rms=[scale])
    independent_low,independent_high = wide_index_envelope(source,codec,ref.THRESHOLDS_6)
    assert low == independent_low and high == independent_high
    assert all(a <= code <= b and b-a <= 1 for a,code,b in zip(low,codes,high))
    assert abs(scale-nominal[0]) <= norm_error[0]
    interval = scale_interval(source.tolist(),codes,ref.CENTROIDS_6,True,max_scale_factor=32)
    assert interval[0] <= scale <= interval[1]
    if codec == 7:
        assert alpha >= 1
        # RN alpha division can put the maximum just beyond the outer centroid;
        # a one-ULP bound is arithmetic coverage, not permission to clamp it.
        assert np.max(np.abs(actual)) <= ref.CENTROIDS_6[-1]*(1+2*2**-24)
        if kind == "outlier":
            assert alpha > 1.3


@pytest.mark.parametrize("codec,variant,name",CASES)
def test_corrected_six_certified_bytes_and_metadata_use_c6_energy(codec,variant,name):
    del name
    torch = pytest.importorskip("torch")
    source,expected = certified_source(codec,ref.CENTROIDS_6,ref.THRESHOLDS_6)
    x = torch.tensor(source,dtype=torch.bfloat16).repeat(2).reshape(1,256)
    packed,metadata = ref.quantize_ref(x,6,variant)
    assert ref.unpack_indices_ref(bytes(packed.flatten().tolist()),6) == expected*2
    lower,upper = ref.index_envelope_oracle(x.float().flatten().tolist(),6,variant,rms=metadata.flatten().tolist())
    assert lower == upper == expected*2
    for group,stored in enumerate(metadata.flatten().tolist()):
        raw = math.hypot(*source)/math.sqrt(128)
        energy = math.fsum(ref.CENTROIDS_6[index]**2 for index in expected)
        c4_wrong = math.fsum(ref.CENTROIDS_4[index%16]**2 for index in expected)
        interval = scale_interval(source.tolist(),expected,ref.CENTROIDS_6,True,max_scale_factor=32)
        assert interval[0] <= stored <= interval[1]
        assert math.isclose(stored,raw*math.sqrt(128/energy),rel_tol=1e-6)
        assert not math.isclose(stored,raw*math.sqrt(128/c4_wrong),rel_tol=1e-3)
    with pytest.raises(ValueError,match="derived wide"):
        ref.rounding_envelope_oracle(source.tolist(),variant,bits=6,rms=[metadata[0,0].item()*2])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("codec,variant,name",CASES)
def test_corrected_six_zero_exact_and_unproved_subnormal_envelope_rejected(codec,variant,name):
    del codec,name
    torch = pytest.importorskip("torch")
    packed,metadata = ref.quantize_ref(torch.zeros((2,256),dtype=torch.bfloat16),6,variant)
    assert packed[0].tolist() == ([255]*64+[85]*32)*2
    assert metadata.eq(0).all() and ref.dequant_ref(packed,metadata,6,variant,inverse=True).eq(0).all()
    with pytest.raises(ValueError,match="normal RMS"):
        ref.rounding_envelope_oracle([2**-140]*128,variant,bits=6)
    with pytest.raises(ValueError,match="zero scale"):
        ref.rounding_envelope_oracle([0.]*128,variant,bits=6,rms=[1.])
