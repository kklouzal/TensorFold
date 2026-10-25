"""Exact cache formats and working arithmetic admitted by the SM70 provider.

Storage remains owned by the common registered KV descriptors. Equal bit
counts or codec numbers alone never authorize another Rotor variant.
"""

from ..kv_formats import KVPairFormat, get
from tensorfold.cuda.sm70.rotorquant_compact import NAME as COMPACT_Q4


ROTOR_Q4 = "rotorquant-signed128-4"
ROTOR_Q4_NORM = "rotorquant-signed64-norm4"
ROTOR_FORMATS = (ROTOR_Q4, ROTOR_Q4_NORM, COMPACT_Q4)
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
                "SM70 KV supports fp16, int8, signed Rotor Q4 or pinned compact Planar4 on each cache side"
            )
        if side.name in ROTOR_FORMATS and (
            (side.bits, side.codec, side.group, side.scale_bytes)
            != (4, 4 if side.name == ROTOR_Q4 else 5 if side.name == ROTOR_Q4_NORM else 9, 128, 4)
            or side.payload_dtype != "uint8"
            or side.scale_dtype != "float32"
        ):
            raise ValueError("SM70 Rotor Q4 requires its unchanged registered descriptor and four-byte metadata")
    if head_dim is not None:
        pair.row_bytes(1, head_dim)  # Group, packing and power-of-two geometry, before allocation.
    return ROTOR_POLICY if pair.key.codec or pair.value.codec else RAW_POLICY


def check_kernel_fields(k_bits: int, k_codec: int, v_bits: int, v_codec: int) -> tuple[int, int]:
    """Match descriptor preflight to the operator ABI; zero bits means raw FP16."""

    fields = (k_bits, k_codec, v_bits, v_codec)
    if any(type(value) is not int for value in fields):
        raise ValueError("SM70 KV requires one complete exact integer key/value plan")
    supported = ((0, 0), (8, 0), (4, 4), (4, 5), (4, 9))
    if (k_bits, k_codec) not in supported or (v_bits, v_codec) not in supported:
        raise ValueError("SM70 KV admits raw fp16, native int8, signed128 or signed64-norm4 Rotor Q4 only")
    return k_bits, v_bits
