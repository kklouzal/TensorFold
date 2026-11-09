"""Independent bounded arithmetic certificates for the private NativeRN candidate.

No CUDA/Torch/candidate imports. Fractions model mathematical values, not signed
zero; round_fraction requires an explicit negative_zero choice for exact zero.
Dot certificates bound values only and do not assert signed-zero/bit equality.
The fixed-point exponential is an independently bounded Taylor oracle, not a
simulation of ex2 or a Float64 sum. Exhaustion raises Unresolved, never passes.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import factorial

MAX_INTEGER_BITS = 4096
MAX_SELECTED = 2051
PRECISIONS = (128, 256, 512, 1024)
LOG2E_BITS = 0x3FB8AA3B


class Unresolved(ArithmeticError):
    """The declared oracle work/precision budget did not certify a decision."""


def _fraction(value):
    if type(value) not in (int, Fraction):
        raise TypeError("oracle values require exact int or Fraction")
    value = Fraction(value)
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_INTEGER_BITS:
        raise Unresolved("exact rational exceeds oracle integer budget")
    return value


def _format(width):
    if type(width) is not int or width not in (16, 32):
        raise ValueError("oracle supports binary16 or binary32")
    return (10, 5, 15) if width == 16 else (23, 8, 127)


def _pow2(exponent):
    if abs(exponent) > 2048:
        raise Unresolved("dyadic exponent exceeds oracle budget")
    return Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**-exponent)


def bits_to_fraction(bits, width=32):
    """Decode a finite bit pattern exactly; ±zero both have mathematical value0."""
    fraction_bits, exponent_bits, bias = _format(width)
    if type(bits) is not int or not 0 <= bits < 2**width:
        raise ValueError("invalid floating bit pattern")
    exponent = bits >> fraction_bits & (2**exponent_bits - 1)
    significand = bits & (2**fraction_bits - 1)
    if exponent == 2**exponent_bits - 1:
        raise ValueError("nonfinite value has no rational oracle value")
    if exponent:
        significand += 2**fraction_bits
    power = (exponent - bias if exponent else 1 - bias) - fraction_bits
    value = significand * _pow2(power)
    return -value if bits >> (width - 1) else value


def _nearest_integer(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    twice = 2 * remainder
    return quotient + (twice > value.denominator or twice == value.denominator and quotient % 2)


def _log2_floor(value):
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    return exponent - (value < _pow2(exponent))


def round_fraction(value, width=32, *, negative_zero=False):
    """Exact IEEE nearest/even rounding, including subnormals and overflow.

    Exact zero sign is explicit; negative nonzero values that underflow preserve
    their sign. This is a conversion primitive, not an FMA zero-sign simulator.
    """
    fraction_bits, exponent_bits, bias = _format(width)
    value = _fraction(value)
    if type(negative_zero) is not bool:
        raise TypeError("negative_zero must be bool")
    sign = int(value < 0 or not value and negative_zero) << (width - 1)
    if not value:
        return sign
    value = abs(value)
    emin, emax = 1 - bias, bias
    exponent = _log2_floor(value)
    infinity = (2**exponent_bits - 1) << fraction_bits
    if exponent > emax:
        return sign | infinity
    power = max(exponent, emin) - fraction_bits
    significand = _nearest_integer(value / _pow2(power))
    if exponent < emin:
        return sign | significand  # Rounding can reach the minimum normal.
    if significand == 2**(fraction_bits + 1):
        significand //= 2
        exponent += 1
    if exponent > emax:
        return sign | infinity
    return sign | ((exponent + bias) << fraction_bits) | (significand - 2**fraction_bits)


def _ceil_ratio(numerator, denominator):
    return -((-numerator) // denominator)


@dataclass(frozen=True)
class Interval:
    """Closed directed fixed-point interval, endpoints integer *2^-precision."""

    lo: int
    hi: int
    precision: int

    def __post_init__(self):
        if (type(self.precision) is not int or not 1 <= self.precision <= 1024
                or type(self.lo) is not int or type(self.hi) is not int or self.lo > self.hi):
            raise ValueError("invalid fixed-point interval")
        if max(self.lo.bit_length(), self.hi.bit_length()) > MAX_INTEGER_BITS:
            raise Unresolved("fixed-point integer budget exhausted")

    @classmethod
    def enclose(cls, lower, upper=None, *, precision=128):
        lower = _fraction(lower)
        upper = lower if upper is None else _fraction(upper)
        if lower > upper:
            raise ValueError("reversed exact interval")
        scale = 2**precision if type(precision) is int and 1 <= precision <= 1024 else None
        if scale is None:
            raise ValueError("invalid interval precision")
        return cls(lower.numerator * scale // lower.denominator,
                   _ceil_ratio(upper.numerator * scale, upper.denominator), precision)

    def fractions(self):
        return Fraction(self.lo, 2**self.precision), Fraction(self.hi, 2**self.precision)

    def _same(self, other):
        if not isinstance(other, Interval) or self.precision != other.precision:
            raise ValueError("interval precisions must match")

    def __add__(self, other):
        self._same(other)
        return Interval(self.lo + other.lo, self.hi + other.hi, self.precision)

    def __mul__(self, other):
        self._same(other)
        products = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        if max(abs(v).bit_length() for v in products) > MAX_INTEGER_BITS:
            raise Unresolved("interval multiplication budget exhausted")
        scale = 2**self.precision
        return Interval(min(products)//scale, _ceil_ratio(max(products), scale), self.precision)

    def divide_integer(self, value):
        if type(value) is not int or value < 1:
            raise ValueError("positive exact integer divisor required")
        return Interval(self.lo//value, _ceil_ratio(self.hi, value), self.precision)

    def reciprocal(self):
        if self.lo <= 0:
            raise Unresolved("positive denominator enclosure required")
        scale_squared = 2**(2*self.precision)
        return Interval(scale_squared//self.hi,
                        _ceil_ratio(scale_squared, self.lo), self.precision)


def ln2_rational(terms=64):
    """Positive 2atanh(1/3) series and rigorous geometric tail enclosure."""
    if type(terms) is not int or not 1 <= terms <= 256:
        raise ValueError("ln2 terms outside bounded oracle")
    lower = 2*sum((Fraction(1, (2*j+1)*3**(2*j+1)) for j in range(terms)), Fraction())
    tail = Fraction(9, 4*(2*terms+1)*3**(2*terms+1))
    return _fraction(lower), _fraction(lower+tail)


def log2e_certificate():
    """Fresh-checkout proof; no dependency on private certificate artifacts."""
    lo, hi = ln2_rational(64)
    lower, upper = 1/hi, 1/lo
    constant = bits_to_fraction(LOG2E_BITS)
    half_ulp = Fraction(1, 2**24)
    if not constant-half_ulp < lower < upper < constant+half_ulp:
        raise ArithmeticError("LOG2E constant not proved inside RN cell")
    return {"bits": LOG2E_BITS, "constant": constant, "lower": lower, "upper": upper,
            "absolute_delta": (lower-constant, upper-constant)}


def _exp_positive_unit(argument, terms):
    if not 0 <= argument.lo <= argument.hi <= 2**argument.precision:
        raise ValueError("Taylor oracle requires [0,1] argument")
    one = Interval.enclose(1, precision=argument.precision)
    term = total = one
    for j in range(1, terms+1):
        term = (term*argument).divide_integer(j)
        total = total+term
    # exp(b)<=e<3 for 0<=b<=1 gives a rigorous Lagrange remainder.
    tail = Interval.enclose(0, Fraction(3, factorial(terms+1)), precision=argument.precision)
    return total+tail


def exp_interval(value, *, precision=128, terms=64):
    """Directed enclosure of mathematical exp(x), x<=0, bounded work.

    Range-reduce independently by powers of two, reciprocal, then square.
    Very negative inputs use exp(x)<2^x<=2^-150, not an unbounded series.
    """
    value = _fraction(value)
    if value > 0 or type(terms) is not int or not 1 <= terms <= 256:
        raise ValueError("negative exponential/bounded series required")
    if value <= -150:
        return Interval.enclose(0, Fraction(1, 2**150), precision=precision)
    if value == 0:
        return Interval.enclose(1, precision=precision)
    magnitude = -value
    shifts = max(0, _log2_floor(magnitude)+1)
    reduced = Interval.enclose(magnitude/_pow2(shifts), precision=precision)
    result = _exp_positive_unit(reduced, terms).reciprocal()
    for _ in range(shifts):
        result = result*result
    return result


def exp2_interval(value, *, precision=128, terms=64):
    """Mathematical 2^t enclosure independent of target ex2 implementation."""
    value = _fraction(value)
    if value > 0 or type(terms) is not int or not 1 <= terms <= 256:
        raise ValueError("nonpositive exponent/bounded series required")
    if value < -150:
        return Interval.enclose(0, Fraction(1, 2**150), precision=precision)
    integer = value.numerator//value.denominator
    fractional = value-integer
    if not fractional:
        return Interval.enclose(_pow2(integer), precision=precision)
    lo, hi = ln2_rational(min(256, terms))
    argument = (Interval.enclose(lo, hi, precision=precision)
                * Interval.enclose(fractional, precision=precision))
    result = _exp_positive_unit(argument, terms)
    scale = Interval.enclose(_pow2(integer), precision=precision)
    return result*scale


def round_interval(interval, width=32):
    """Return unique RN bits only if every enclosed value has that rounding."""
    lo, hi = interval.fractions()
    lower, upper = round_fraction(lo, width), round_fraction(hi, width)
    if lower != upper:
        raise Unresolved("rounding interval crosses a midpoint or signed zero")
    return lower


def exp2_round(value, width=32, *, precisions=PRECISIONS):
    """Bounded refinement; exhaustion is explicitly UNRESOLVED, never a pass."""
    _format(width)
    if (not isinstance(precisions, tuple) or not precisions or len(precisions) > 4
            or any(type(p) is not int or p not in PRECISIONS for p in precisions)
            or tuple(sorted(set(precisions))) != precisions):
        raise ValueError("invalid bounded refinement catalogue")
    for precision in precisions:
        interval = exp2_interval(value, precision=precision, terms=min(256, precision//2))
        try:
            return round_interval(interval, width)
        except Unresolved:
            pass
    raise Unresolved("exponential oracle refinement exhausted")


def gamma(depth):
    if type(depth) is not int or not 0 <= depth < 2**24:
        raise ValueError("invalid RN error depth")
    return Fraction(depth, 2**24-depth)


def dot_enclosure(left_bits, right_bits, *, depth, scale=Fraction(1)):
    """Exact Half dot plus analytical bound; no candidate tree simulation."""
    if (not isinstance(left_bits, (list, tuple)) or not isinstance(right_bits, (list, tuple))
            or not 1 <= len(left_bits) == len(right_bits) <= MAX_SELECTED):
        raise ValueError("bounded equal nonempty Half vectors required")
    products = [bits_to_fraction(a, 16)*bits_to_fraction(b, 16)
                for a, b in zip(left_bits, right_bits)]
    scale = _fraction(scale)
    exact = sum(products, Fraction())*scale
    bound = gamma(depth)*sum(map(abs, products), Fraction())*abs(scale)
    return exact-bound, exact+bound


def qk_enclosure(query_bits, key_bits):
    if len(query_bits) != 256 or len(key_bits) != 256:
        raise ValueError("NativeRN QK requires D256")
    return dot_enclosure(query_bits, key_bits, depth=13, scale=Fraction(1, 16))


def pv_enclosure(probability_bits, value_bits):
    if any(bits_to_fraction(bits, 16) < 0 for bits in probability_bits):
        raise ValueError("negative Half probability")
    return dot_enclosure(probability_bits, value_bits,
                         depth=(len(probability_bits)+31)//32+5)


def native_exp_enclosure(argument_bits):
    """Enclose the SPECIFIED RN(x*c)->nonFTZ ex2 result, not libdevice expf.

    PTX's two-ULP statement can straddle a binade. Four maximum adjacent
    spacings about the correctly rounded reference cover that union, including
    either endpoint's ULP convention. This is a conservative specification
    enclosure, not an empirical four-ULP acceptance tolerance. Special x=0
    follows the candidate's exact identity contract. No candidate imports.
    """
    if type(argument_bits) is not int or not 0 <= argument_bits < 2**32:
        raise ValueError("invalid binary32 exponential argument")
    if argument_bits == 0xFF800000:
        return Fraction(), Fraction()
    value = bits_to_fraction(argument_bits)
    if value > 0:
        raise ValueError("native exponential requires nonpositive argument")
    if not value:
        return Fraction(1), Fraction(1)
    exponent_bits = round_fraction(value*bits_to_fraction(LOG2E_BITS))
    if exponent_bits == 0xFF800000:
        return Fraction(), Fraction()  # RN product overflow: ex2(-inf) is exactly zero.
    exponent = bits_to_fraction(exponent_bits)
    reference_bits = exp2_round(exponent)
    reference = bits_to_fraction(reference_bits)
    next_value = bits_to_fraction(reference_bits+1)
    previous = bits_to_fraction(reference_bits-1) if reference_bits else Fraction()
    spacing = max(next_value-reference, reference-previous)
    radius = 4*spacing
    return reference-radius, reference+radius


def half_probability_set(score_bits, maximum_bits, denominator_bits, *, maximum_values=8):
    """Certified possible Half casts for fixed actual scalar normalization input.

    This does NOT certify the provenance/error of M/L or whole attention.
    Their separate interval/semantic certificate remains necessary. A midpoint
    can legitimately yield multiple allowed bits; excessive ambiguity refuses.
    """
    if type(maximum_values) is not int or not 1 <= maximum_values <= 64:
        raise ValueError("bounded probability ambiguity required")
    score = bits_to_fraction(score_bits)
    maximum = bits_to_fraction(maximum_bits)
    denominator = bits_to_fraction(denominator_bits)
    if denominator <= 0 or score > maximum:
        raise ValueError("invalid positive-mass normalization domain")
    argument = round_fraction(score-maximum)
    lower, upper = native_exp_enclosure(argument)
    low_bits = round_fraction(lower/denominator)
    high_bits = round_fraction(upper/denominator)
    low_value, high_value = bits_to_fraction(low_bits), bits_to_fraction(high_bits)
    return half_candidates(low_value, high_value, maximum_values=maximum_values,
                           negative_zero=low_bits == 0x80000000)


def half_candidates(lower, upper, *, maximum_values=8, negative_zero=False):
    """Finite Half bit set in signed numeric order; retain -0/+0 separately."""
    if type(maximum_values) is not int or not 1 <= maximum_values <= 64:
        raise ValueError("bounded Half ambiguity required")
    lower, upper = _fraction(lower), _fraction(upper)
    if lower > upper or type(negative_zero) is not bool:
        raise ValueError("invalid Half enclosure")
    first = round_fraction(lower, 16, negative_zero=negative_zero)
    last = round_fraction(upper, 16)
    if first & 0x7FFF >= 0x7C00 or last & 0x7FFF >= 0x7C00:
        raise Unresolved("Half enclosure may overflow")
    def key(bits):
        return 0xFFFF-bits if bits & 0x8000 else bits+0x8000

    start, stop = key(first), key(last)
    if stop < start or stop-start+1 > maximum_values:
        raise Unresolved("Half probability ambiguity exceeds budget")
    return tuple(0xFFFF-k if k < 0x8000 else k-0x8000 for k in range(start, stop+1))


@dataclass(frozen=True)
class Bounds:
    """Exact rational endpoints for a numeric certificate, not zero-sign proof."""

    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        object.__setattr__(self, "lo", _fraction(self.lo))
        object.__setattr__(self, "hi", _fraction(self.hi))
        if self.lo > self.hi:
            raise ValueError("reversed numeric bounds")


def _rn_bounds(lower, upper, width=32):
    try:
        return Bounds(bits_to_fraction(round_fraction(lower, width), width),
                      bits_to_fraction(round_fraction(upper, width), width))
    except ValueError as error:
        raise Unresolved("numeric interval can overflow/nonfinite") from error


def rn_add(left, right):
    return _rn_bounds(left.lo+right.lo, left.hi+right.hi)


def rn_sub(left, right):
    return _rn_bounds(left.lo-right.hi, left.hi-right.lo)


def rn_mul(left, right):
    products = [a*b for a in (left.lo, left.hi) for b in (right.lo, right.hi)]
    return _rn_bounds(min(products), max(products))


def rn_div(left, right):
    if right.lo <= 0:
        raise Unresolved("normalization denominator not proved positive")
    quotients = [a/b for a in (left.lo, left.hi) for b in (right.lo, right.hi)]
    return _rn_bounds(min(quotients), max(quotients))


def _maximum(bounds):
    return Bounds(max(v.lo for v in bounds), max(v.hi for v in bounds))


def native_exp_range(argument, *, max_dominates=False):
    """Union of ex2's pointwise contract; NEVER assume approximate monotonicity.

    Exact mathematical 2^t and RN conversion are monotone. Their extrema bound
    reference values; a maximum-binade spacing then encloses every approximate
    output. Negative underflow uncertainty is retained. max_dominates permits
    intersecting x<=0 ONLY for proven score-minus-own/global-max dependence.
    """
    if type(max_dominates) is not bool:
        raise TypeError("max correlation flag must be explicit bool")
    lo, hi = argument.lo, argument.hi
    if max_dominates:
        hi = min(hi, Fraction())
    if lo > hi or hi > 0:
        raise Unresolved("nonpositive exponential domain not established")
    if lo == hi == 0:
        return Bounds(1, 1)
    constant = bits_to_fraction(LOG2E_BITS)
    low_t = round_fraction(lo*constant)
    high_t = round_fraction(hi*constant)
    low_r = 0 if low_t == 0xFF800000 else exp2_round(bits_to_fraction(low_t))
    high_r = 0 if high_t == 0xFF800000 else exp2_round(bits_to_fraction(high_t))
    lower, upper = bits_to_fraction(low_r), bits_to_fraction(high_r)
    spacing = bits_to_fraction(high_r+1)-upper
    # Four spacings cover the possible one-binade ULP convention transition.
    radius = 4*spacing
    return Bounds(lower-radius, upper+radius)


def source_binding(snapshots, expected):
    """Hash immutable caller-captured source bytes against reviewed identities.

    Existing loader owns filesystem snapshot/TOCTOU rules. This helper performs
    no I/O or candidate imports; its receipt proves the supplied bytes only.
    """
    import hashlib
    from pathlib import PurePosixPath

    if (not isinstance(expected, dict) or not 1 <= len(expected) <= 32
            or not isinstance(snapshots, dict) or set(snapshots) != set(expected)):
        raise ValueError("bounded complete explicit source identity required")
    result = {}
    for name, digest in expected.items():
        if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid source identity")
        payload = snapshots[name]
        if type(payload) is not bytes or len(payload) > 2**20:
            raise ValueError("source snapshot must be bounded immutable bytes")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != digest:
            raise ValueError("source identity mismatch: "+name)
        result[name] = actual
    return result


def _local_statistics(scores, owner, owners):
    """Abstract transfer of ordered TILE32 online program; -inf is None."""
    maximum, denominator = None, Bounds(0, 0)
    for first in range(0, len(scores), 32):
        active = [scores[i] for i in range(first, min(first+32, len(scores))) if owners[i] == owner]
        if not active:
            continue
        tile_max = _maximum(active)
        next_max = tile_max if maximum is None else _maximum((maximum, tile_max))
        factor = (Bounds(0, 0) if maximum is None else
                  native_exp_range(rn_sub(maximum, next_max), max_dominates=True))
        mass = Bounds(0, 0)
        for score in active:
            mass = rn_add(mass, native_exp_range(rn_sub(score, next_max), max_dominates=True))
        denominator = rn_add(rn_mul(denominator, factor), mass)
        maximum = next_max
    if maximum is not None and denominator.lo <= 0:
        raise Unresolved("local online mass positivity not established")
    return maximum, denominator


def _merge_statistics(locals_):
    active = [maximum for maximum, _ in locals_ if maximum is not None]
    if not active:
        raise ValueError("globally empty attention is invalid")
    maximum = _maximum(active)
    denominator = Bounds(0, 0)
    for local_max, mass in locals_:  # Exact four-peer order, no tree substitute.
        if local_max is not None:
            if mass.lo <= 0:
                raise Unresolved("local statistics may violate mass invariant")
            factor = native_exp_range(rn_sub(local_max, maximum), max_dominates=True)
            denominator = rn_add(denominator, rn_mul(factor, mass))
    if denominator.lo <= 0:
        raise Unresolved("four-peer normalization mass not proved positive")
    return maximum, denominator


def _half_bounds(value):
    return _rn_bounds(value.lo, value.hi, 16)


def _pv_bounds(probabilities, values, owner, owners):
    exact_lower = exact_upper = absolute_sum = Fraction()
    for index, probability in enumerate(probabilities):
        if owners[index] != owner:
            continue
        if probability.lo < 0:
            raise Unresolved("possible negative Half probability: candidate would fail")
        value = bits_to_fraction(values[index], 16)
        products = (probability.lo*value, probability.hi*value)
        exact_lower += min(products)
        exact_upper += max(products)
        absolute_sum += max(map(abs, products))
    radius = gamma((len(probabilities)+31)//32+5)*absolute_sum
    return Bounds(exact_lower-radius, exact_upper+radius)


def _finish_h32(values, *, rounded=True):
    """Interval transfer for existing five-butterfly Q8 finish, then Half cast."""
    add, subtract, multiply = ((rn_add, rn_sub, rn_mul) if rounded
                               else (real_add, real_sub, real_mul))
    result = values
    for offset in (1, 2, 4, 8, 16):
        previous = result
        result = [subtract(previous[col ^ offset], previous[col]) if col & offset
                  else add(previous[col], previous[col ^ offset]) for col in range(256)]
    factor_bits = round_fraction(Fraction("0.1767766952966369"))
    factor = bits_to_fraction(factor_bits)
    return tuple(_half_bounds(multiply(value, Bounds(factor, factor))) for value in result)


def _finish_linear(values, rows, *, rounded=True):
    """Independent coefficient-map certificate for compact codec9/10 finish.

    Rows MUST come from separately authenticated codec/math authority. At most
    four explicit FP32-coefficient products plus ordered adds -> depth<=4.
    This transfer bounds every such order; it does not import/copy codec code.
    """
    if not isinstance(rows, tuple) or len(rows) != 256:
        raise ValueError("authenticated complete compact-codec coefficient map required")
    output = []
    for row in rows:
        if not isinstance(row, tuple) or not 1 <= len(row) <= 4:
            raise ValueError("compact codec row requires one to four coefficient terms")
        low = high = absolute = Fraction()
        for index, coefficient_bits in row:
            if type(index) is not int or not 0 <= index < 256:
                raise ValueError("invalid codec source coordinate")
            coefficient = bits_to_fraction(coefficient_bits)
            products = (coefficient*values[index].lo, coefficient*values[index].hi)
            low += min(products)
            high += max(products)
            absolute += max(map(abs, products))
        count = len(row)
        radius = ((gamma(count)*absolute
                   + Fraction(2*count-1, 2**150)*(1+Fraction(1, 2**24))**count)
                  if rounded else Fraction())
        output.append(_half_bounds(Bounds(low-radius, high+radius)))
    return tuple(output)


def attention_head_enclosure(query, keys, values, owners, *, sparse=False,
                             finish="q8-h32", codec_rows=None, source_receipt=None,
                             fraction_product_limit=None):
    """One head's whole candidate-value certificate, from working Half bits.

    V2 legal dense/sparse count<=2051. owners gives independent canonical
    ownership/masks (0..3 or None), never candidate-derived routing. Local
    statistics None represents exactly (-inf,0). Global invalidity refuses.
    Query/keys are snapshotted and reused, establishing QK-pass identity and
    max-dominance correlation. Returned bounds are numeric, not signed-zero or
    descriptor/state/mask correctness proof. Full output is 256 coordinates.
    """
    if type(sparse) is not bool or not 1 <= len(keys) <= MAX_SELECTED:
        raise ValueError("V2 NativeRN selected-count domain exceeded")
    if len(query) != 256 or len(values) != len(keys) or len(owners) != len(keys):
        raise ValueError("inconsistent D256 attention operands")
    query, keys = tuple(query), tuple(tuple(row) for row in keys)
    values, owners = tuple(tuple(row) for row in values), tuple(owners)
    if any(len(row) != 256 for row in values):
        raise ValueError("value working operands must have D256")
    if any(owner is not None and (type(owner) is not int or not 0 <= owner < 4) for owner in owners):
        raise ValueError("independent owner map requires four peers or masked None")
    if source_receipt is not None and (not isinstance(source_receipt, dict) or not source_receipt):
        raise ValueError("invalid source provenance receipt")
    terms = 0
    if finish == "compact-linear":
        if (not isinstance(codec_rows, tuple) or len(codec_rows) != 256
                or any(not isinstance(row, tuple) or not 1 <= len(row) <= 4 for row in codec_rows)):
            raise ValueError("bounded complete compact coefficient rows required")
        terms = sum(map(len, codec_rows))
    elif finish != "q8-h32" or codec_rows is not None:
        raise Unresolved("unsupported finish work geometry")
    work = whole_work_bound(len(keys), sum(owner is not None for owner in owners),
                            compact_terms=terms)
    if fraction_product_limit is not None and (
            type(fraction_product_limit) is not int or fraction_product_limit < 1):
        raise ValueError("positive exact Fraction-product budget required")
    limit = (WHOLE_FRACTION_PRODUCT_CAP if fraction_product_limit is None
             else min(fraction_product_limit, WHOLE_FRACTION_PRODUCT_CAP))
    if work["fraction_multiplications_upper"] > limit:
        raise Unresolved("whole oracle static Fraction-product budget exhausted before work")
    scores = tuple(Bounds(*qk_enclosure(query, key)) if owner is not None else None
                   for key, owner in zip(keys, owners))
    locals_ = tuple(_local_statistics(scores, owner, owners) for owner in range(4))
    maximum, denominator = _merge_statistics(locals_)
    probabilities = []
    for score in scores:
        if score is None:
            probabilities.append(Bounds(0, 0))
        else:
            exponent = native_exp_range(rn_sub(score, maximum), max_dominates=True)
            probabilities.append(_half_bounds(rn_div(exponent, denominator)))
    local_values = tuple(tuple(_pv_bounds(probabilities, [row[col] for row in values], peer, owners)
                               for col in range(256)) for peer in range(4))
    folded = []
    for col in range(256):
        result = local_values[0][col]
        for peer in (1, 2, 3):
            result = rn_add(result, local_values[peer][col])
        folded.append(result)
    if finish == "q8-h32" and codec_rows is None:
        finished = _finish_h32(folded)
    elif finish == "compact-linear" and source_receipt is not None:
        finished = _finish_linear(folded, codec_rows)
    else:
        raise Unresolved("finish codec/authority not specified or supported")
    ideal = ideal_head_enclosure(query, keys, values, owners, finish=finish, codec_rows=codec_rows)
    differences = tuple(real_sub(candidate, original)
                        for candidate, original in zip(finished, ideal["finished_half_values"]))
    return {"local_statistics": locals_, "global_statistics": (maximum, denominator),
            "half_probabilities": tuple(probabilities), "local_values": local_values,
            "peer_fold": tuple(folded), "finished_half_values": finished,
            "source_receipt": None if source_receipt is None else dict(source_receipt),
            "ideal": ideal, "candidate_minus_ideal": differences, "work_bound": work,
            "scope": "candidate-value and exact-decoded-math error intervals; not cuBLAS/model/mask proof",
            "signed_zero": "numeric intervals only: separately unresolved where raw zero sign matters"}


def real_add(left, right):
    return Bounds(left.lo+right.lo, left.hi+right.hi)


def real_sub(left, right):
    return Bounds(left.lo-right.hi, left.hi-right.lo)


def real_mul(left, right):
    products = [a*b for a in (left.lo, left.hi) for b in (right.lo, right.hi)]
    return Bounds(min(products), max(products))


def ideal_head_enclosure(query, keys, values, owners, *, finish="q8-h32", codec_rows=None):
    """Independent exact-decoded mathematical attention with declared Half P.

    Exact dyadic QK, real exp/normalization, then RN_H probabilities; exact PV
    and exact pinned-coefficient rotation before final RN_H. Candidate FP32
    approximations/rounding (including double rounding) are NOT the authority.
    Exp uses directed 256-bit/128-term independent series, not target ex2.
    A crossing Half midpoint remains an interval, never a unique-bit claim.
    This function's input geometry is validated by attention_head_enclosure.
    """
    scores = []
    for key, owner in zip(keys, owners):
        scores.append(None if owner is None else sum(
            (bits_to_fraction(a, 16)*bits_to_fraction(b, 16) for a, b in zip(query, key)),
            Fraction())/16)
    maximum = max(score for score in scores if score is not None)
    exponents = [Bounds(0, 0) if score is None else Bounds(*exp_interval(
        score-maximum, precision=256, terms=128).fractions()) for score in scores]
    denominator = Bounds(sum(v.lo for v in exponents), sum(v.hi for v in exponents))
    if denominator.lo <= 0:
        raise Unresolved("ideal denominator enclosure not positive")
    probabilities = tuple(_half_bounds(Bounds(value.lo/denominator.hi,
                                             value.hi/denominator.lo)) for value in exponents)
    summed = []
    for col in range(256):
        result = Bounds(0, 0)
        for index, probability in enumerate(probabilities):
            if owners[index] is not None:
                value = bits_to_fraction(values[index][col], 16)
                result = real_add(result, real_mul(probability, Bounds(value, value)))
        summed.append(result)
    if finish == "q8-h32":
        finished = _finish_h32(summed, rounded=False)
    elif finish == "compact-linear":
        finished = _finish_linear(summed, codec_rows, rounded=False)
    else:
        raise Unresolved("ideal finish authority missing")
    return {"scores": tuple(scores), "maximum": maximum, "denominator": denominator,
            "half_probabilities": probabilities, "exact_pv": tuple(summed),
            "finished_half_values": finished,
            "authority": "exact decoded Half operands/real softmax/Half P/exact coefficient map/Half output"}


def compact_finish_authority(codec, snapshots, expected):
    """Authenticate/project compact authority DATA, never its arithmetic code.

    Existing owned generator owns the header/descriptor format. We verify its
    compact table words against stdlib struct's same FP32 serialization of the
    pinned authority literals; no Float64 arithmetic serves as a math oracle.
    Signed128 tables/quality are deliberately not adopted.
    """
    import ast
    import re
    import struct

    prefix = "src/tensorfold/cuda/sm70/"
    authority_name = prefix+"rotorquant_compact.py"
    generator_name = prefix+"rotorquant_q4_tables.py"
    header_name = prefix+"rotorquant_q4_tables.cuh"
    operator_name = prefix+"rotorquant_q4.cuh"
    required = {authority_name, generator_name, header_name, operator_name}
    if type(codec) is not int or codec not in (9, 10) or not required <= set(snapshots):
        raise ValueError("complete pinned compact codec authority required")
    receipt = source_binding(snapshots, expected)
    wanted = {"SOURCE_REVISION", "SOURCE_CONSTANTS_SHA256", "SOURCE_CUDA_SHA256",
              "SOURCE_SET_ROWS_SHA256", "SOURCE_ATTENTION_SHA256", "CENTROIDS",
              "COS", "SIN", "QW", "QX", "QY", "QZ"}
    projected = {}
    for node in ast.parse(snapshots[authority_name]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                projected[name] = ast.literal_eval(node.value)
    if set(projected) != wanted or projected["SOURCE_REVISION"] != "08e025c06ab521e4fa9e5c08b80af57614543e53":
        raise ValueError("unsupported compact authority revision/data")
    header = snapshots[header_name].decode("ascii")
    tables = {}
    table_sizes = {"CENTROIDS": 16, "COS": 64, "SIN": 64, "QW": 32, "QX": 32, "QY": 32, "QZ": 32}
    for name, size in table_sizes.items():
        values = projected[name]
        if type(values) is not tuple or len(values) != size or any(type(v) is not float for v in values):
            raise ValueError("malformed compact authority table")
        bits = tuple(struct.unpack("<I", struct.pack("<f", v))[0] for v in values)
        for value in bits:
            bits_to_fraction(value)
        suffix = "centroid" if name == "CENTROIDS" else name.lower()
        declaration = re.findall(r"\bcompact_"+suffix+r"_bits\["+str(size)+r"\]\s*=\s*\{([^}]+)\}", header)
        if len(declaration) != 1:
            raise ValueError("missing/duplicate generated compact table")
        emitted = tuple(int(word.strip().removesuffix("u"), 16) for word in declaration[0].split(","))
        if emitted != bits:
            raise ValueError("generated compact data differs from authority")
        tables[name] = bits
    variant = "planar4" if codec == 9 else "iso4"
    labels = re.findall(r"// (rq-"+variant+r"-compact-v1-[0-9a-f]{64})\n", header)
    if len(labels) != 1:
        raise ValueError("generated compact descriptor absent/ambiguous")
    rows = []
    if codec == 9:
        for group in range(128):
            base = group*2
            c, s = tables["COS"][group % 64], tables["SIN"][group % 64]
            rows.extend((((base, c), (base+1, s)), ((base, s ^ 0x80000000), (base+1, c))))
    else:
        for group in range(64):
            base = group*4
            w, x, y, z = (tables[name][group % 32] for name in ("QW", "QX", "QY", "QZ"))
            # Inverse quaternion multiplication matrix, independently derived.
            coefficients = ((w, x, y, z), (x ^ 0x80000000, w, z, y ^ 0x80000000),
                            (y ^ 0x80000000, z ^ 0x80000000, w, x),
                            (z ^ 0x80000000, y, x ^ 0x80000000, w))
            rows.extend(tuple((base+i, coefficient) for i, coefficient in enumerate(row))
                        for row in coefficients)
    return {"codec": codec, "descriptor": labels[0], "source_receipt": receipt,
            "authority_revision": projected["SOURCE_REVISION"],
            "upstream_source_hashes": {k: v for k, v in projected.items() if k.endswith("SHA256")},
            "coefficient_rows": tuple(rows), "arithmetic_depth": 2 if codec == 9 else 4,
            "scope": "authenticated compact inverse matrix; no exact-unit-norm or quality assumption"}


# Preregistered work-only V3 correction: counts owned high-level Fraction
# multiplications, not undocumented CPython GCD/integer-machine instructions.
# Integer sizes/series/factorial inputs remain independently bounded above.
WHOLE_FRACTION_PRODUCT_CAP = (14*2051*256 + 15*2051 + 40*256
                              + 17*(2*2051+259) + 10*259 + 8*4)


def whole_work_bound(selected, active=None, *, compact_terms=0):
    """Static upper bounds for THIS helper, candidate plus separate ideal.

    8*A*D operand products: two QK evaluations, candidate PV's two interval
    endpoints and ideal PV's four endpoints. Decode/scalar/finish products are
    additional. No timing estimate or target instruction count is implied.
    Compact terms is the sum of256 authenticated row lengths (1..4).
    """
    if (type(selected) is not int or not 1 <= selected <= MAX_SELECTED
            or active is not None and (type(active) is not int or not 0 <= active <= selected)
            or type(compact_terms) is not int
            or compact_terms != 0 and not 256 <= compact_terms <= 1024):
        raise ValueError("invalid bounded whole-oracle geometry")
    active = selected if active is None else active
    quotient, remainder = divmod(selected, 32)
    nonempty_tiles = min(active, 4*quotient+min(4, remainder))
    peers = min(4, active)
    exponentials = 2*active+nonempty_tiles
    finish_products = (6*compact_terms+16*256) if compact_terms else (34*256+2)
    fraction_products = (14*active*256+13*active+2*selected+finish_products
                         + 17*exponentials+10*nonempty_tiles+8*peers)
    return {
        "selected": selected, "active": active, "heads": 1, "dimension": 256,
        "nonempty_peer_tiles_upper": nonempty_tiles, "nonempty_peers_upper": peers,
        "native_exponential_transfers_upper": exponentials,
        "operand_endpoint_products": 8*active*256,
        "fraction_multiplications_upper": fraction_products,
        "fixedpoint_endpoint_products_upper": 5776*exponentials+550*active,
        "native_Taylor_terms_upper": 1408*exponentials,
        "native_ln2_terms_upper": 1408*exponentials,
        "ideal_Taylor_terms_upper": 128*active,
        "ideal_squarings_upper": 8*active,
        "fixedpoint_ex2_refinements_upper": 8*exponentials,
        "ordered_RN_interval_transfers_upper":
            5*active+selected+3*nonempty_tiles+2*peers+(5 if compact_terms else 11)*256,
        "small_rational_powers_upper": 256 if compact_terms else 0,
        "small_rational_power_exponent_max": 4 if compact_terms else 0,
        "integer_bit_cap": MAX_INTEGER_BITS,
        "scope": "owned high-level operations; exact library internals not machine-operation counts",
    }
