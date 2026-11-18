"""Exact cache formats and working arithmetic admitted by the SM70 provider.

Storage remains owned by the common registered KV descriptors. Equal bit
counts or codec numbers alone never authorize another Rotor variant.
"""

from ..kv_formats import KVPairFormat, get
from tensorfold.cuda.sm70.rotorquant_compact import NAME as COMPACT_Q4, ISO_NAME as COMPACT_ISO4


ROTOR_Q4 = "rotorquant-signed128-4"
ROTOR_Q4_NORM = "rotorquant-signed64-norm4"
ROTOR_Q8 = "rotorquant8"
ROTOR_Q8_NORM = "rotorquant8-norm"
ROTOR_FORMATS = (ROTOR_Q4, ROTOR_Q4_NORM, COMPACT_Q4, COMPACT_ISO4, ROTOR_Q8, ROTOR_Q8_NORM)
# Eight-bit signed Iso128 sides run only on the unique-history owned path (owned writer + RN paged readers).
EIGHT_BIT = (ROTOR_Q8, ROTOR_Q8_NORM)
_DESCRIPTOR = {ROTOR_Q4: (4, 4), ROTOR_Q4_NORM: (4, 5), COMPACT_Q4: (4, 9), COMPACT_ISO4: (4, 10),
               ROTOR_Q8: (8, 4), ROTOR_Q8_NORM: (8, 8)}


def kernel_codec(side) -> int:
    """Native SM70 codec id of a registered side: its descriptor codec, except descriptor codec 4 at eight
    bits (rotorquant8, original RMS), which the native kernels call 12 to keep it distinct from Q4 codec 4."""
    return 12 if (side.bits, side.codec) == (8, 4) else side.codec
FORMATS = ("fp16", "int8", *ROTOR_FORMATS)
RAW_POLICY = "stored-basis-fp16-fp32acc-v1"
ROTOR_POLICY = "stored-rotor-basis-fp16-fp32acc-v1"


def check_pair(pair: KVPairFormat, *, head_dim: int | None = None) -> str:
    """Validate registered ordered sides and return their FP16 working policy."""

    if type(pair) is not KVPairFormat:
        raise ValueError("SM70 KV requires a registered ordered KVPairFormat")
    for side in (pair.key, pair.value):
        if side.name not in FORMATS or side != get(side.name):
            raise ValueError(
                "SM70 KV supports fp16, int8, signed Rotor Q4, pinned compact Planar4/Iso4 or rotorquant8(-norm)"
            )
        if side.name in ROTOR_FORMATS and (
            (side.bits, side.codec, side.group, side.scale_bytes) != (*_DESCRIPTOR[side.name], 128, 4)
            or side.payload_dtype != "uint8"
            or side.scale_dtype != "float32"
        ):
            raise ValueError("SM70 RotorQuant requires its unchanged registered descriptor and four-byte metadata")
    if any(side.name in EIGHT_BIT for side in (pair.key, pair.value)) and not (
            pair.symmetric and pair.key.name in EIGHT_BIT):
        raise ValueError("SM70 eight-bit RotorQuant requires the same rotorquant8 or rotorquant8-norm on K and V")
    if any(side.name == COMPACT_ISO4 for side in (pair.key, pair.value)) and not (
            pair.symmetric and pair.key.name == COMPACT_ISO4):
        raise ValueError("initial SM70 compact Iso4 profile requires K=Iso4 and V=Iso4; mixed variants are unsupported")
    if head_dim is not None:
        pair.row_bytes(1, head_dim)  # Group, packing and power-of-two geometry, before allocation.
    return ROTOR_POLICY if pair.key.codec or pair.value.codec else RAW_POLICY


def check_kernel_fields(k_bits: int, k_codec: int, v_bits: int, v_codec: int) -> tuple[int, int]:
    """Match descriptor preflight to the operator ABI; zero bits means raw FP16."""

    fields = (k_bits, k_codec, v_bits, v_codec)
    if any(type(value) is not int for value in fields):
        raise ValueError("SM70 KV requires one complete exact integer key/value plan")
    supported = ((0, 0), (8, 0), (4, 4), (4, 5), (4, 9), (4, 10))
    if (k_bits, k_codec) not in supported or (v_bits, v_codec) not in supported:
        raise ValueError("SM70 KV admits raw fp16, native int8, signed128 or signed64-norm4 Rotor Q4 only")
    if 10 in (k_codec, v_codec) and fields != (4, 10, 4, 10):
        raise ValueError("initial SM70 compact Iso4 operators require symmetric K/V codec10")
    return k_bits, v_bits
