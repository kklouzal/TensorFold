"""Pinned CUDA Planar4 compact cache format; distinct from TensorFold Signed128.

The format authority is llama-cpp-turboquant commit
08e025c06ab521e4fa9e5c08b80af57614543e53, CUDA planar-iso-constants.cuh and
cpy-planar-iso.cu, not its different CPU Planar4 rotation table. Each 128 values
use little-endian FP16 corrected L2 norm, reserved FP16 zero, then 64 low-even
nibble bytes. In a TensorFold arena payload and metadata are separate arrays;
the metadata uses an opaque four-byte float32 container, never a float32 scale.
No encoding or decode operation needs an expanded full-history allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct

SOURCE_REVISION = "08e025c06ab521e4fa9e5c08b80af57614543e53"
SOURCE_URL = f"https://github.com/johndpope/llama-cpp-turboquant/tree/{SOURCE_REVISION}"
SOURCE_CONSTANTS_SHA256 = "c116f9a23ef1966689ddac7446d3e0ba1652b2cf2d5b35b597d3ef360aa226a7"
SOURCE_CUDA_SHA256 = "9652235bf4237205917989a0dd71a85f07f5ff3bc87b02f2f09902197c75a86b"
CODEC = 9
NAME = "rotorquant-planar4-sm70-v1"
GROUP = 128
BLOCK_BYTES = 68

CENTROIDS = (
    -0.1739259958267212, -0.1171950027346611, -0.08952700346708298, -0.06875599920749664,
    -0.05126199871301651, -0.035597000271081924, -0.02098900079727173, -0.006938000209629536,
    0.006938000209629536, 0.02098900079727173, 0.035597000271081924, 0.05126199871301651,
    0.06875599920749664, 0.08952700346708298, 0.1171950027346611, 0.1739259958267212,
)

COS = (
    -0.9095053672790527, 0.15355785191059113, -0.8537489175796509, -0.6827217936515808,
    -0.42493879795074463, 0.9864510297775269, 0.9906674027442932, 0.5752363204956055,
    -0.9866458773612976, 0.9878848195075989, -0.6215683817863464, -0.983559787273407,
    0.8777263760566711, -0.46246400475502014, 0.2843135893344879, -0.7739960551261902,
    0.2385234236717224, 0.9121915102005005, -0.8815004229545593, -0.26396995782852173,
    -0.5517087578773499, -0.9035294651985168, -0.8520542979240417, -0.5600636005401611,
    -0.7667286396026611, -0.9877949357032776, -0.9781949520111084, -0.9953373074531555,
    -0.8622053861618042, -0.7382118105888367, 0.9136037826538086, -0.2558504641056061,
    -0.8541000485420227, -0.6159335374832153, 0.986125648021698, -0.6758560538291931,
    0.42495715618133545, -0.6219545006752014, 0.9130573272705078, -0.5948160886764526,
    0.5759782791137695, 0.972990095615387, 0.6535998582839966, 0.9222195744514465,
    -0.7668083906173706, 0.5116178393363953, -0.784878671169281, 0.9902111291885376,
    0.19971677660942078, 0.717300295829773, -0.9999998211860657, -0.9557868838310242,
    0.5594852566719055, -0.9980111718177795, 0.9782398343086243, -0.9150004386901855,
    -0.40847542881965637, 0.007154918275773525, 0.9558482766151428, -0.09719216823577881,
    -0.946933388710022, 0.9999492168426514, 0.6100589036941528, 0.0350818932056427,
)

SIN = (
    -0.4156922399997711, 0.9881396889686584, 0.5206848978996277, -0.7306784391403198,
    -0.9052220582962036, 0.16405613720417023, 0.13630154728889465, 0.8179872632026672,
    0.16287989914417267, 0.1551889330148697, 0.7833598852157593, -0.18058288097381592,
    -0.47916218638420105, 0.8866380453109741, -0.9587313532829285, 0.633190393447876,
    -0.971136748790741, 0.4097641706466675, 0.4721832871437073, -0.964530885219574,
    0.8340368270874023, 0.42852598428726196, 0.5234533548355103, 0.8284496068954468,
    0.6419713497161865, -0.15575994551181793, -0.20768867433071136, 0.0964556559920311,
    0.5065588355064392, -0.6745690107345581, -0.40660566091537476, -0.9667163491249084,
    0.5201087594032288, -0.7877981066703796, 0.16600050032138824, -0.7370336651802063,
    0.905213475227356, 0.7830533981323242, -0.40783119201660156, -0.8038617968559265,
    0.8174650073051453, -0.23084676265716553, -0.7568402886390686, -0.3866666555404663,
    0.6418760418891907, -0.85921311378479, 0.6196494698524475, 0.13957782089710236,
    0.9798536896705627, 0.6967641115188599, -0.000631460512522608, 0.29406028985977173,
    0.8288403153419495, -0.06303712725639343, 0.20747719705104828, 0.4034528434276581,
    0.9127693176269531, -0.9999744296073914, 0.29386064410209656, 0.9952656626701355,
    0.3214298188686371, 0.0100754015147686, -0.7923560738563538, -0.9993844628334045,
)


@dataclass(frozen=True, slots=True)
class CompactDescriptor:
    """Exact decode identity; the float32 tensor stores opaque two-half words."""
    codec_id: str
    variant: str = "cuda-planar4-compact"
    bits: int = 4
    group: int = GROUP
    scale_dtype: str = "float32"
    norm_policy: str = "fp16-corrected-l2-reserved-zero"
    version: int = 1

    @property
    def vector_bytes_per_group(self):
        return BLOCK_BYTES


def descriptor():
    authority = ("tf-sm70-planar4-compact-v1|" + SOURCE_REVISION +
                 "|cuda-rotation|fp16-corrected-l2|reserved-zero|even-low-nibble")
    tables = struct.pack("<144f", *(CENTROIDS + COS + SIN))
    return CompactDescriptor("rq-planar4-compact-v1-" + hashlib.sha256(authority.encode() + tables).hexdigest())


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _width(width):
    if type(width) is not int or width <= 0 or width % GROUP:
        raise ValueError("compact Planar4 needs complete positive 128-value groups")


def rotate_ref(value, *, inverse=False):
    """Staged binary32 adjacent-pair rotation, with no runtime-global tensors."""
    import torch
    _width(value.shape[-1])
    shape = value.shape
    pairs = value.float().reshape(*shape[:-1], shape[-1] // 128, 64, 2)
    c = torch.tensor(COS, dtype=torch.float32, device=value.device)
    s = torch.tensor(SIN, dtype=torch.float32, device=value.device)
    a, b = pairs[..., 0], pairs[..., 1]
    result = (torch.stack((c*a+s*b, -s*a+c*b), dim=-1) if inverse else
              torch.stack((c*a-s*b, s*a+c*b), dim=-1))
    return result.reshape(shape)


def quantize_ref(value):
    """Bounded-frame reference encoding into split payload and opaque metadata."""
    import torch
    _width(value.shape[-1])
    if value.dtype not in (torch.float16, torch.float32) or not bool(torch.isfinite(value).all()):
        raise ValueError("compact Planar4 source needs finite FP16/FP32 values")
    groups = value.float().reshape(*value.shape[:-1], value.shape[-1] // GROUP, GROUP)
    norm = groups.square().sum(-1).sqrt()
    inv = torch.where(norm > 1e-10, norm.reciprocal(), 0.)
    rotated = rotate_ref((groups * inv[..., None]).reshape(value.shape))
    book = torch.tensor(CENTROIDS, dtype=torch.float32, device=value.device)
    # Strictly closer selection fixes lower-code ties like the pinned CUDA path.
    distance = (rotated[..., None] - book).abs()
    codes = distance.argmin(-1).to(torch.uint8)
    energy = book[codes.long()].reshape(groups.shape).square().sum(-1).sqrt()
    corrected = torch.where(energy > 1e-10, norm / energy, norm).half()
    if not bool(torch.isfinite(corrected).all()) or bool(((norm > 1e-10) & (corrected == 0)).any()):
        raise ValueError("compact Planar4 corrected FP16 norm is outside its finite nonzero domain")
    pair = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    payload = pair[..., 0] | (pair[..., 1] << 4)
    words = torch.zeros((*corrected.shape, 2), dtype=torch.float16, device=value.device)
    words[..., 0] = corrected
    metadata = words.contiguous().view(torch.float32).reshape(corrected.shape)
    return payload, metadata


def dequant_ref(payload, metadata, *, inverse=False):
    """Decode a caller-bounded tile; metadata floats are reinterpreted, never converted."""
    import torch
    if payload.dtype != torch.uint8 or payload.shape[-1] <= 0 or payload.shape[-1] % 64:
        raise ValueError("compact Planar4 payload needs complete uint8 64-byte groups")
    expected = (*payload.shape[:-1], payload.shape[-1] // 64)
    if (metadata.dtype != torch.float32 or tuple(metadata.shape) != expected
            or metadata.device != payload.device or not metadata.is_contiguous()):
        raise ValueError("compact Planar4 needs matching opaque four-byte metadata groups")
    halves = metadata.view(torch.float16).reshape(*expected, 2)
    norm = halves[..., 0]
    reserved_zero = (metadata.view(torch.int32) >> 16) == 0
    if not bool((torch.isfinite(norm) & (norm >= 0) & reserved_zero).all()):
        raise ValueError("compact Planar4 requires finite nonnegative norm and reserved zero")
    codes = torch.stack((payload & 15, payload >> 4), -1).reshape(*payload.shape[:-1], payload.shape[-1] * 2)
    book = torch.tensor(CENTROIDS, dtype=torch.float32, device=payload.device)
    result = (book[codes.long()].reshape(*expected, GROUP) * norm.float()[..., None]).reshape(codes.shape)
    return rotate_ref(result, inverse=True) if inverse else result


def encode_oracle(values):
    """Independent scalar wire oracle; returns source-compatible 68-byte blocks.

    Scalar FP32 arithmetic deliberately has its own reduction order. Exact
    encoding identity is asserted only on inputs clear of quantizer boundaries.
    """
    _width(len(values))
    if any(not math.isfinite(v) for v in values):
        raise ValueError("compact Planar4 source values must be finite")
    result = bytearray()
    for start in range(0, len(values), GROUP):
        block = [_f32(v) for v in values[start:start+GROUP]]
        energy = 0.
        for v in block:
            energy = _f32(energy + _f32(v*v))
        norm = _f32(math.sqrt(energy))
        inv = _f32(1./norm) if norm > 1e-10 else 0.
        codes = []
        for p, (c, s) in enumerate(zip(COS, SIN)):
            a,b = (_f32(block[2*p+i]*inv) for i in (0,1))
            for v in (_f32(_f32(c*a)-_f32(s*b)), _f32(_f32(s*a)+_f32(c*b))):
                codes.append(min(range(16), key=lambda k: abs(_f32(v-CENTROIDS[k]))))
        energy = 0.
        for code in codes:
            energy = _f32(energy+_f32(CENTROIDS[code]*CENTROIDS[code]))
        reconstructed = _f32(math.sqrt(energy))
        corrected = _f32(norm/reconstructed) if reconstructed > 1e-10 else norm
        try:
            header = struct.pack('<ee', corrected, 0.)
        except (OverflowError, struct.error) as exc:
            raise ValueError("compact Planar4 norm exceeds FP16") from exc
        if norm > 1e-10 and struct.unpack('<e', header[:2])[0] == 0:
            raise ValueError("compact Planar4 norm underflows FP16")
        result.extend(header)
        result.extend(codes[j] | (codes[j+1] << 4) for j in range(0,GROUP,2))
    return bytes(result)


def decode_oracle(data):
    """Independent source-block decoder, including inverse rotation after scale."""
    if not data or len(data) % BLOCK_BYTES:
        raise ValueError("compact Planar4 wire data needs complete 68-byte blocks")
    output = []
    for start in range(0,len(data),BLOCK_BYTES):
        norm,reserved = struct.unpack('<ee',data[start:start+4])
        if not math.isfinite(norm) or norm < 0 or data[start+2:start+4] != b'\0\0':
            raise ValueError("compact Planar4 wire metadata is invalid")
        for p, byte in enumerate(data[start+4:start+BLOCK_BYTES]):
            a,b = CENTROIDS[byte&15],CENTROIDS[byte>>4]
            output.extend((_f32(_f32(_f32(COS[p]*a)+_f32(SIN[p]*b))*norm),
                           _f32(_f32(_f32(-SIN[p]*a)+_f32(COS[p]*b))*norm)))
    return output
