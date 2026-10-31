"""Allocation-free SM70 primitives for explicitly registered Q4 wire formats.

Owners retain all operands, controls, status, and outputs through stream/graph
completion. Prepare the extension before capture. Encoding uses fixed FP32
128-coordinate reductions, original RMS (CODEC4), exact existing staged basis
and lower-midpoint indices. Selected reconstructions must be representable as
FP16 for the SM70 working policy; errors are contained before cache publication.
CODEC9 is pinned CUDA Planar4 and CODEC10 is pinned CUDA Iso4: corrected FP16
L2 norm plus reserved FP16 zero in each opaque four-byte metadata word. Neither
uses FP32 RMS or inherits Signed128 format or quality evidence.
No inactive encoding row writes any cache byte or scale. Decode and rotation
zero inactive scratch rows, while preserving output rows beyond ``rows``.

Status is the existing int32 [bits, first_layer] with an initial layer sentinel
of -2. Bad source/rotation values latch ``error``; invalid RMS or FP16 decoded
operands latch8; invalid device indices/live-row counts latch16. Status remains
sticky. Omitting it is only valid for previously validated finite/range-safe
operands and device controls. Model encoding/query preparation supplies status;
readers may omit it only under their validated cache and working-range contract.
These primitives do not admit a format: the existing ordered descriptor policy
selects supported model formats independently.
"""
from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources
    from .rotorquant_q4_tables import verify

    verify()
    prepared = _native()
    root = Path(__file__).parent
    sources = [root / "rotorquant_q4_bindings.cpp", root / "rotorquant_q4.cu"]
    headers = [root / "rotorquant_q4.cuh", root / "rotorquant_q4_tables.cuh"]
    identity = hashlib.sha256(prepared.__name__.encode())
    for source in [*sources, *headers]:
        identity.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load("tf_sm70_rotorquant_q4_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cuda_cflags=["-O3", "--fmad=false", "--ftz=false"], extra_cflags=["-O3"])


def encode(x, codes, scales, *, codec=4, position=None, status=None, error=2, layer=0,
           valid_rows=None, rows=None):
    """[R,H,D] FP16/FP32 -> uint8 [C,H,D/2], four-byte metadata [C,H,D/128].

    The metadata tensor dtype is float32. CODEC1..5 store numerical FP32
    scales; CODEC9/10 store the bits of FP16 corrected norm and reserved zero.

    Source head rows may have a larger leading stride. A nullable CUDA int32 or
    int64 ``position[1]`` selects the destination base; bounds are checked on
    device without a host readback. A nullable CUDA int32 ``valid_rows[1]``
    limits active source rows. An invalid active cache index latches16 and does
    not write. No allocations occur after preparation.
    """
    extension().encode(x, codes, scales, position, status, valid_rows,
                       codec, x.shape[0] if rows is None else rows, error, layer)
    return codes, scales


def rotate(x, out, *, codec=4, inverse=False, status=None, error=1, layer=0,
           valid_rows=None, rows=None):
    """Staged FP32 basis transform, rounded to the caller's FP16/FP32 output."""
    extension().rotate(x, out, status, valid_rows, codec, x.shape[0] if rows is None else rows,
                       inverse, error, layer)
    return out


def decode(codes, scales, out, *, codec=4, inverse=False, status=None, error=8, layer=0,
           valid_rows=None, rows=None):
    """Decode to rotated FP16/FP32, or inverse-rotate FP32 values before output rounding."""
    extension().decode(codes, scales, out, status, valid_rows, codec,
                       out.shape[0] if rows is None else rows, inverse, error, layer)
    return out


def copy_rows(x, target, position, *, status=None, valid_rows=None, layer=0):
    """Bitwise rank2/rank3 row copy at a dynamic position, with inactive rows untouched.

    Source leading stride may include projection gaps; trailing dimensions must
    be contiguous and match target. Dtypes match exactly. This does no numerical
    conversion or finite check: the producing frame owns those guarantees.
    Device position/live-count faults latch16 and cannot access outside target.
    """
    extension().copy_rows(x, target, position, status, valid_rows, layer)
    return target
