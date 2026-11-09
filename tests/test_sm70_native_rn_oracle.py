"""Independent contracts of exact arithmetic tools; no target/library math oracle."""
from fractions import Fraction

import pytest

from tests.sm70_native_rn_oracle import (
    Interval, LOG2E_BITS, Unresolved, bits_to_fraction, exp_interval,
    exp2_interval, exp2_round, gamma, half_probability_set,
    ln2_rational, log2e_certificate, native_exp_enclosure,
    pv_enclosure, qk_enclosure, round_fraction, round_interval,
)


@pytest.mark.parametrize("width,patterns", [
    (16, [0, 1, 0x03FF, 0x0400, 0x3555, 0x3C00, 0x7BFF, 0x8001, 0xBC00]),
    (32, [0, 1, 0x007FFFFF, 0x00800000, 0x3F800000, 0x3FB8AA3B, 0x7F7FFFFF, 0x80000001]),
])
def test_exact_finite_decode_roundtrip(width, patterns):
    for bits in patterns:
        assert round_fraction(bits_to_fraction(bits, width), width) == bits


@pytest.mark.parametrize("width,one,sign", [(16, 0x3C00, 0x8000), (32, 0x3F800000, 0x80000000)])
def test_zero_sign_and_midpoint_ties_are_explicit(width, one, sign):
    assert bits_to_fraction(sign, width) == 0
    assert round_fraction(0, width, negative_zero=True) == sign
    assert round_fraction(0, width) == 0
    a, b, c = (bits_to_fraction(one+i, width) for i in range(3))
    assert round_fraction((a+b)/2, width) == one
    assert round_fraction((b+c)/2, width) == one+2
    assert round_fraction(-(a+b)/2, width) == sign | one


@pytest.mark.parametrize("width,smallest,infinity,maxfinite", [
    (16, Fraction(1, 2**24), 0x7C00, 0x7BFF),
    (32, Fraction(1, 2**149), 0x7F800000, 0x7F7FFFFF),
])
def test_subnormal_underflow_and_overflow_cells(width, smallest, infinity, maxfinite):
    assert round_fraction(smallest/2, width) == 0
    assert round_fraction(3*smallest/2, width) == 2
    assert round_fraction(-smallest/4, width) == 1 << (width-1)
    maximum = bits_to_fraction(maxfinite, width)
    previous = bits_to_fraction(maxfinite-1, width)
    midpoint = maximum+(maximum-previous)/2
    assert round_fraction(midpoint, width) == infinity
    assert round_fraction(midpoint-(maximum-previous)/4, width) == maxfinite


@pytest.mark.parametrize("bits,width", [(0x7C00, 16), (0x7E00, 16), (0x7F800000, 32), (True, 32), (-1, 16)])
def test_invalid_bits_refuse(bits, width):
    with pytest.raises(ValueError):
        bits_to_fraction(bits, width)


def test_log2e_exact_independent_96_term_enclosure_and_rn_cell():
    cert = log2e_certificate()
    low96, high96 = ln2_rational(96)
    assert cert["lower"] < 1/high96 < 1/low96 < cert["upper"]
    assert cert["constant"] == Fraction(12102203, 8388608)
    assert round_fraction(cert["lower"]) == round_fraction(cert["upper"]) == LOG2E_BITS
    lo, hi = cert["absolute_delta"]
    assert 0 < lo < hi < Fraction(1, 2**24)


def test_directed_interval_signed_arithmetic_and_reciprocal():
    a = Interval.enclose(Fraction(-1, 3), Fraction(1, 7), precision=128)
    b = Interval.enclose(Fraction(2, 5), Fraction(3, 4), precision=128)
    low, high = (a*b).fractions()
    assert low <= Fraction(-1, 4) <= high
    low, high = b.reciprocal().fractions()
    assert low <= Fraction(4, 3) and high >= Fraction(5, 2)
    with pytest.raises(Unresolved):
        a.reciprocal()


@pytest.mark.parametrize("value", [Fraction(0), Fraction(-1), Fraction(-32), Fraction(-149), Fraction(-150), Fraction(-151)])
def test_exp2_exact_power_rounding(value):
    expected = Fraction(1, 2**-int(value))
    lo, hi = exp2_interval(value, precision=256).fractions()
    assert lo <= expected <= hi
    assert exp2_round(value) == round_fraction(expected)


def test_exponential_taylor_independent_sqrt_identity_and_nested_bounds():
    lo, hi = exp2_interval(Fraction(-1, 2), precision=256, terms=128).fractions()
    # 2^-1/2 has square1/2. No logarithm/float/same-algorithm oracle.
    assert 0 < lo < hi and lo*lo <= Fraction(1, 2) <= hi*hi
    assert round_fraction(lo) == round_fraction(hi) == exp2_round(Fraction(-1, 2))
    for value in (Fraction(-1, 4), Fraction(-1), Fraction(-4)):
        low, high = exp_interval(value, precision=256, terms=128).fractions()
        coarse_low, coarse_high = exp_interval(value, precision=128).fractions()
        assert 0 < coarse_low <= low <= high <= coarse_high < 1
    # exp(-1) reciprocal contains e=sum 1/j!, bounded by3/(m+1)!.
    from math import factorial
    exact_lower = sum((Fraction(1, factorial(j)) for j in range(65)), Fraction())
    exact_upper = exact_lower+Fraction(3, factorial(65))
    low, high = exp_interval(Fraction(-1), precision=256, terms=128).fractions()
    assert low <= 1/exact_upper <= 1/exact_lower <= high


def test_unresolved_refinement_midpoint_and_explicit_work_limits(monkeypatch):
    midpoint = (bits_to_fraction(0x3C00, 16)+bits_to_fraction(0x3C01, 16))/2
    with pytest.raises(Unresolved):
        round_interval(Interval.enclose(midpoint-Fraction(1, 2**50), midpoint+Fraction(1, 2**50)), 16)
    import tests.sm70_native_rn_oracle as oracle
    attempts = []

    def ambiguous(value, *, precision, terms):
        attempts.append(precision)
        return Interval.enclose(0, 1, precision=precision)

    with monkeypatch.context() as patch:
        patch.setattr(oracle, "exp2_interval", ambiguous)
        with pytest.raises(Unresolved, match="refinement exhausted"):
            oracle.exp2_round(Fraction(-1, 2))
    assert attempts == [128, 256, 512, 1024]
    for kwargs in ({"precision": 2048}, {"terms": 257}):
        with pytest.raises(ValueError):
            exp2_interval(Fraction(-1, 2), **kwargs)
    with pytest.raises(Unresolved):
        round_fraction(Fraction(1, 2**4097))
    with pytest.raises(ValueError):
        exp2_round(Fraction(-1), precisions=(256, 128))


def test_gamma_dot_certificates_cover_exact_cancellation_and_maximum_geometry():
    assert gamma(13) == Fraction(13, 2**24-13)
    assert gamma(70) == Fraction(70, 2**24-70)
    q = [0x3C00]*256
    k = [0x3C00, 0xBC00]*128
    lo, hi = qk_enclosure(q, k)
    assert lo < 0 < hi
    assert hi == gamma(13)*16
    lo, hi = pv_enclosure([0x3C00]*2051, [0x3C00]*2051)
    assert lo < 2051 < hi and hi-2051 == gamma(70)*2051
    with pytest.raises(ValueError):
        pv_enclosure([0x3C00]*2052, [0x3C00]*2052)
    with pytest.raises(ValueError):
        pv_enclosure([0xBC00], [0x3C00])


def test_native_exponential_contract_and_bounded_half_probability_ambiguity():
    assert native_exp_enclosure(0) == (1, 1)
    assert native_exp_enclosure(0xFF800000) == (0, 0)
    assert native_exp_enclosure(0xFF7FFFFF) == (0, 0)
    assert half_probability_set(0, 0, 0x3F800000) == (0x3C00,)
    low, high = native_exp_enclosure(0xBF800000)
    ideal_low, ideal_high = exp_interval(-1, precision=256, terms=128).fractions()
    assert low < ideal_low < ideal_high < high
    with pytest.raises(ValueError):
        native_exp_enclosure(0x3F800000)
    with pytest.raises(ValueError):
        half_probability_set(0x3F800000, 0, 0x3F800000)
    with pytest.raises(ValueError):
        half_probability_set(0, 0, 0)
    # denominator1024 and score0 yields a valid Half exactly; helper does not
    # claim probabilities globally normalized when supplied impossible M/L.
    assert half_probability_set(0, 0, 0x44800000) == (0x1400,)


def test_half_probability_midpoint_ambiguity_is_not_reported_as_unique():
    # Independently construct a normalization input near a Half midpoint from
    # its exact representable endpoints; finite precision must not hide choices.
    midpoint = (bits_to_fraction(0x3555, 16)+bits_to_fraction(0x3556, 16))/2
    lower, upper = native_exp_enclosure(0xBF800000)
    denominator = round_fraction((lower+upper)/(2*midpoint))
    permitted = half_probability_set(0xBF800000, 0, denominator)
    assert permitted == (0x3555, 0x3556)
    with pytest.raises(Unresolved, match="ambiguity"):
        half_probability_set(0xBF800000, 0, denominator, maximum_values=1)


def test_ex2_underflow_retains_negative_uncertainty_and_both_zero_signs():
    from tests.sm70_native_rn_oracle import half_candidates, native_exp_range, Bounds
    low, high = native_exp_enclosure(0xC3000000)  # -128 -> mathematical exp below normal.
    assert low < 0 < high
    permitted = half_probability_set(0xC3000000, 0, 0x3F800000)
    assert permitted == (0x8000, 0)
    assert half_candidates(-Fraction(1, 2**100), Fraction(1, 2**100)) == (0x8000, 0)
    with pytest.raises(Unresolved):
        half_candidates(-1, 1)
    interval = native_exp_range(Bounds(-128, -127))
    assert interval.lo < 0 < interval.hi
    with pytest.raises(Unresolved):
        native_exp_range(Bounds(-1, 1))


def test_whole_uniform_33_rows_online_tiles_and_four_peer_analytic_mass():
    from tests.sm70_native_rn_oracle import attention_head_enclosure
    query = [0]*256
    keys = [[0x3C00]*256 for _ in range(33)]
    values = [[0x3C00]*256 for _ in range(33)]
    owners = [i % 4 for i in range(33)]
    report = attention_head_enclosure(query, keys, values, owners)
    maximum, denominator = report["global_statistics"]
    assert maximum.lo == maximum.hi == 0
    assert denominator.lo == denominator.hi == 33
    assert report["ideal"]["denominator"].lo == report["ideal"]["denominator"].hi == 33
    assert all(v.lo <= 0 <= v.hi for v in report["candidate_minus_ideal"])
    for peer, (maximum, mass) in enumerate(report["local_statistics"]):
        assert maximum.lo == maximum.hi == 0
        assert mass.lo == mass.hi == owners.count(peer)
    p = bits_to_fraction(round_fraction(bits_to_fraction(round_fraction(Fraction(1, 33))), 16), 16)
    assert all(v.lo == v.hi == p for v in report["half_probabilities"])
    for value in report["peer_fold"]:
        assert value.lo <= 33*p <= value.hi
    # Exact linear Hadamard operator: each constant32 block has one32*p*n
    # coordinate and31 zeros; final cast authority is exact dyadic conversion.
    factor = bits_to_fraction(round_fraction(Fraction("0.1767766952966369")))
    for col, value in enumerate(report["finished_half_values"]):
        expected = bits_to_fraction(round_fraction(32*factor*(33*p) if col % 32 == 0 else 0, 16), 16)
        assert value.lo <= expected <= value.hi


def test_whole_empty_shards_masks_and_global_empty_failure():
    from tests.sm70_native_rn_oracle import attention_head_enclosure
    # Masked payloads are intentionally invalid and must never be decoded.
    report = attention_head_enclosure([0]*256, [[0]*256, [0x7E00]*256],
                                     [[0]*256, [0x7E00]*256], [2, None])
    assert report["local_statistics"][0][0] is None
    assert report["local_statistics"][1][0] is None
    assert report["local_statistics"][3][0] is None
    assert report["global_statistics"][1].lo == report["global_statistics"][1].hi == 1
    assert all(v.lo == v.hi == 0 for v in report["finished_half_values"])
    with pytest.raises(ValueError, match="globally empty"):
        attention_head_enclosure([0]*256, [[0x7E00]*256], [[0x7E00]*256], [None])


def test_whole_v2_dense_bound_and_compact_coefficient_authority():
    from tests.sm70_native_rn_oracle import attention_head_enclosure, source_binding
    import hashlib
    snapshots = {"codec-authority": b"explicit independent identity coefficients"}
    receipt = source_binding(snapshots, {n: hashlib.sha256(b).hexdigest() for n, b in snapshots.items()})
    # Independent identity linear operator is a fixture, NOT actual codec10.
    rows = tuple(((col, 0x3F800000),) for col in range(256))
    report = attention_head_enclosure([0]*256, [[0]*256], [[0x3C00]*256], [0],
                                     finish="compact-linear", codec_rows=rows, source_receipt=receipt)
    assert all(v.lo <= 1 <= v.hi for v in report["finished_half_values"])
    assert report["source_receipt"] == receipt
    with pytest.raises(Unresolved, match="finish"):
        attention_head_enclosure([0]*256, [[0]*256], [[0]*256], [0],
                                 finish="compact-linear", codec_rows=rows)
    with pytest.raises(ValueError, match="identity mismatch"):
        source_binding({"codec-authority": b"changed"}, receipt)
    with pytest.raises(ValueError, match="domain"):
        attention_head_enclosure([0]*256, [[0]*256]*2052, [[0]*256]*2052, [0]*2052)


def test_whole_exp_domain_uncertainty_and_nonfinite_are_not_passes():
    from tests.sm70_native_rn_oracle import Bounds, rn_div, rn_mul
    with pytest.raises(Unresolved, match="positive"):
        rn_div(Bounds(1, 1), Bounds(-1, 1))
    maximum = bits_to_fraction(0x7F7FFFFF)
    with pytest.raises(Unresolved, match="overflow"):
        rn_mul(Bounds(maximum, maximum), Bounds(2, 2))


@pytest.mark.parametrize("codec", [9, 10])
def test_authenticated_actual_compact_finish_matrix_and_both_128_blocks(codec):
    import hashlib
    from pathlib import Path
    from tests.sm70_native_rn_oracle import compact_finish_authority, attention_head_enclosure

    root = Path(__file__).resolve().parents[1]
    names = ["rotorquant_compact.py", "rotorquant_q4_tables.py",
             "rotorquant_q4_tables.cuh", "rotorquant_q4.cuh"]
    snapshots = {"src/tensorfold/cuda/sm70/"+name:
                 (root/"src/tensorfold/cuda/sm70"/name).read_bytes() for name in names}
    expected = {name: hashlib.sha256(payload).hexdigest() for name, payload in snapshots.items()}
    authority = compact_finish_authority(codec, snapshots, expected)
    rows = authority["coefficient_rows"]
    assert authority["arithmetic_depth"] == (2 if codec == 9 else 4)
    assert len(rows) == 256 and authority["descriptor"].startswith("rq-")
    assert all(tuple((index-128, bits) for index, bits in rows[col+128]) == rows[col] for col in range(128))
    # The real matrix is an authority, not assumed orthonormal after rounding.
    # A basis vector gives independent exact coefficient outputs in both blocks.
    values = [[0x3C00 if col in (0, 128) else 0 for col in range(256)]]
    report = attention_head_enclosure([0]*256, [[0]*256], values, [0],
                                     finish="compact-linear", codec_rows=rows,
                                     source_receipt=authority["source_receipt"])
    for col, row in enumerate(rows):
        exact = sum((bits_to_fraction(bits) for index, bits in row if index in (0, 128)), Fraction())
        expected_half = bits_to_fraction(round_fraction(exact, 16), 16)
        assert report["ideal"]["finished_half_values"][col].lo == expected_half
        assert report["ideal"]["finished_half_values"][col].hi == expected_half
        bound = report["finished_half_values"][col]
        assert bound.lo <= expected_half <= bound.hi
    tampered = dict(snapshots)
    header = "src/tensorfold/cuda/sm70/rotorquant_q4_tables.cuh"
    tampered[header] += b"// changed\n"
    with pytest.raises(ValueError, match="identity mismatch"):
        compact_finish_authority(codec, tampered, expected)


def test_whole_nonuniform_peer_rescaling_and_independent_real_softmax_authority():
    from tests.sm70_native_rn_oracle import attention_head_enclosure
    # Exact dyadic QK logits are1,0,-1 on three different owners. This forces
    # nontrivial cross-peer exponential rescaling rather than a uniform-only gate.
    query = [0x4C00]+[0]*255  # 16 -> exact scale1/16 cancels.
    keys = [[first]+[0]*255 for first in (0x3C00, 0, 0xBC00)]
    values = [[first]+[0]*255 for first in (0x3C00, 0xBC00, 0)]
    report = attention_head_enclosure(query, keys, values, [0, 1, 2])
    assert report["ideal"]["scores"] == (1, 0, -1)
    ideal = report["ideal"]["denominator"]
    # 2<e<3 gives13/9 < 1+e^-1+e^-2 < 7/4, without target math.
    assert Fraction(13, 9) < ideal.lo <= ideal.hi < Fraction(7, 4)
    maximum, mass = report["global_statistics"]
    assert maximum.lo <= 1 <= maximum.hi and mass.lo > 0
    assert report["local_statistics"][3][0] is None
    assert all(v.lo <= 0 <= v.hi for v in report["candidate_minus_ideal"])


def test_preregistered_whole_work_bound_and_prework_exhaustion():
    from tests.sm70_native_rn_oracle import (
        WHOLE_FRACTION_PRODUCT_CAP, attention_head_enclosure, whole_work_bound,
    )
    maximum = whole_work_bound(2051, compact_terms=1024)
    assert maximum["operand_endpoint_products"] == 8*2051*256
    assert maximum["nonempty_peer_tiles_upper"] == 259
    assert maximum["native_exponential_transfers_upper"] == 4361
    assert maximum["fraction_multiplications_upper"] == WHOLE_FRACTION_PRODUCT_CAP
    assert whole_work_bound(33, 1)["nonempty_peer_tiles_upper"] == 1
    assert whole_work_bound(33, 33)["nonempty_peer_tiles_upper"] == 5
    masked = whole_work_bound(2051, 1)
    assert masked["fraction_multiplications_upper"] == (
        14*256+13+2*2051+34*256+2+17*3+10+8)
    assert masked["ordered_RN_interval_transfers_upper"] == 5+2051+3+2+11*256
    # A small declared budget refuses before invalid payloads are evaluated.
    with pytest.raises(Unresolved, match="budget exhausted before work"):
        attention_head_enclosure([0x7E00]*256, [[0x7E00]*256], [[0x7E00]*256], [0],
                                 fraction_product_limit=1)
    for geometry in (0, 2052, True):
        with pytest.raises(ValueError):
            whole_work_bound(geometry)
