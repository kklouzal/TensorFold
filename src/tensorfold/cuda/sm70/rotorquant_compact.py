"""Pinned CUDA Iso4 and Planar4 formats; distinct from TensorFold Signed128.

The format authority is llama-cpp-turboquant commit
08e025c06ab521e4fa9e5c08b80af57614543e53: CUDA set-rows-planar-iso.cuh,
fattn-common.cuh and planar-iso-constants.cuh, with the cpy Planar4 comparison
writer. Its CPU Planar4 rotation table is different. Each 128 values
use little-endian FP16 corrected L2 norm, reserved FP16 zero, then 64 low-even
nibble bytes. In a TensorFold arena payload and metadata are separate arrays;
the metadata uses an opaque four-byte float32 container, never a float32 scale.
Iso4 uses the exact 32 quaternion rows of the same pinned CUDA authority.
Planar4 remains the explicit equal-memory comparison. Neither format inherits
another format's quality/runtime qualification. No encoding or decode operation
needs an expanded full-history allocation.
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
SOURCE_SET_ROWS_SHA256 = "bf8e7a20b9321bdb8840ff29505c315678935f6215f033accadc721b40c29679"
SOURCE_ATTENTION_SHA256 = "bca07237947f5b6562573447b3f9715bf9f89f1953226354d363c9efa35873c9"
CODEC = 9
NAME = "rotorquant-planar4-sm70-v1"
ISO_CODEC = 10
ISO_NAME = "rotorquant-iso4-sm70-v1"
CODECS = (CODEC, ISO_CODEC)
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


QW = (
    0.8350809812545776, -0.16484981775283813, 0.12837521731853485, 0.28976985812187195,
    -0.18205493688583374, 0.9549587368965149, -0.8741137385368347, 0.8988990187644958,
    -0.13125844299793243, -0.39905983209609985, -0.26948168873786926, -0.11818988621234894,
    0.13633954524993896, 0.2665117681026459, -0.8263269662857056, -0.18341895937919617,
    0.30982473492622375, 0.28046974539756775, -0.5655074715614319, -0.1627507507801056,
    0.8684155941009521, 0.2233296036720276, -0.12916718423366547, 0.6606932878494263,
    -0.5694432258605957, -0.27827608585357666, 0.5113853812217712, -0.5139024257659912,
    0.7489815354347229, -0.303739994764328, -0.4143463373184204, -0.35240501165390015,
)

QX = (
    0.35471028089523315, -0.5782636404037476, -0.8299785256385803, 0.5694668292999268,
    -0.8199930191040039, 0.12595438957214355, -0.30908143520355225, -0.26135966181755066,
    -0.1660282462835312, -0.5143862962722778, 0.5898610353469849, -0.8277072310447693,
    -0.6826571226119995, -0.17406293749809265, 0.14161993563175201, 0.4648889899253845,
    0.34856218099594116, 0.8982698917388916, -0.30152493715286255, 0.4990116357803345,
    0.23989422619342804, -0.744769811630249, 0.4783197045326233, 0.07358556240797043,
    -0.2975912094116211, -0.07007047533988953, 0.29756274819374084, -0.2652103304862976,
    -0.15397650003433228, 0.08499941229820251, -0.10698036849498749, -0.5753474831581116,
)

QY = (
    0.24168501794338226, -0.44881996512413025, 0.3478420674800873, 0.5024775266647339,
    0.16965430974960327, 0.17604760825634003, 0.025450540706515312, 0.2389279007911682,
    -0.9429193735122681, 0.3925755023956299, -0.27574583888053894, -0.14852671325206757,
    0.5530825853347778, -0.8936085104942322, 0.29537156224250793, -0.5285226702690125,
    0.7939327955245972, 0.013978931121528149, -0.25557103753089905, 0.4543992280960083,
    -0.2698826789855957, -0.47369682788848877, 0.4361720681190491, -0.34612220525741577,
    0.07921162247657776, 0.8827795386314392, 0.7416539788246155, -0.38263994455337524,
    -0.3534849286079407, -0.8696597814559937, -0.6908422708511353, 0.20827366411685944,
)

QZ = (
    0.30386945605278015, 0.4734756052494049, -0.38788437843322754, 0.5831694602966309,
    -0.5054479241371155, -0.17316944897174835, -0.3737666606903076, 0.23287048935890198,
    0.2621760964393616, 0.6239953637123108, -0.7082104682922363, 0.5308507680892944,
    -0.4413037896156311, -0.2802782654762268, -0.45223671197891235, -0.6698107123374939,
    -0.37524569034576416, -0.3359423279762268, 0.7181019186973572, 0.7106907367706299,
    0.31000736355781555, 0.4016827940940857, 0.7350437641143799, -0.6607965230941772,
    0.7619289755821228, 0.3648703992366791, -0.3040413558483124, 0.7213236689567566,
    0.5280022621154785, -0.3742936849594116, -0.5760775208473206, 0.7015634775161743,
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


def _codec(codec):
    if type(codec) is not int or codec not in CODECS:
        raise ValueError("compact RotorQuant requires pinned Planar4 codec9 or Iso4 codec10")
    return codec


def variant_codec(variant):
    if variant == "cuda-planar4-compact":
        return CODEC
    if variant == "cuda-iso4-compact":
        return ISO_CODEC
    raise ValueError("unknown pinned compact RotorQuant variant")


def descriptor(codec=CODEC):
    codec = _codec(codec)
    if codec == CODEC:
        authority = ("tf-sm70-planar4-compact-v1|" + SOURCE_REVISION +
                     "|cuda-rotation|fp16-corrected-l2|reserved-zero|even-low-nibble")
        tables = struct.pack("<144f", *(CENTROIDS + COS + SIN))
        return CompactDescriptor("rq-planar4-compact-v1-" + hashlib.sha256(authority.encode() + tables).hexdigest())
    authority = ("tf-sm70-iso4-compact-v1|" + SOURCE_REVISION +
                 "|cuda-quaternion32|fp16-corrected-l2|reserved-zero|even-low-nibble")
    tables = struct.pack("<144f", *(CENTROIDS + QW + QX + QY + QZ))
    return CompactDescriptor("rq-iso4-compact-v1-" + hashlib.sha256(authority.encode() + tables).hexdigest(),
                             variant="cuda-iso4-compact")


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _width(width):
    if type(width) is not int or width <= 0 or width % GROUP:
        raise ValueError("compact RotorQuant needs complete positive 128-value groups")


def rotate_ref(value, *, codec=CODEC, inverse=False):
    """Pinned staged binary32 rotation; no runtime-global mutable tensors."""
    import torch
    codec = _codec(codec)
    _width(value.shape[-1])
    shape = value.shape
    if codec == ISO_CODEC:
        groups = value.float().reshape(*shape[:-1], shape[-1] // 128, 32, 4)
        w, x, y, z = (torch.tensor(table, dtype=torch.float32, device=value.device)
                       for table in (QW, QX, QY, QZ))
        if inverse:
            x, y, z = -x, -y, -z  # Exact pinned conjugation, not coefficient renormalization.
        a, b, c, d = (groups[..., component] for component in range(4))
        return torch.stack((w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                            w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a), -1).reshape(shape)
    pairs = value.float().reshape(*shape[:-1], shape[-1] // 128, 64, 2)
    c = torch.tensor(COS, dtype=torch.float32, device=value.device)
    s = torch.tensor(SIN, dtype=torch.float32, device=value.device)
    a, b = pairs[..., 0], pairs[..., 1]
    result = (torch.stack((c*a+s*b, -s*a+c*b), dim=-1) if inverse else
              torch.stack((c*a-s*b, s*a+c*b), dim=-1))
    return result.reshape(shape)


def quantize_ref(value, *, codec=CODEC):
    """Bounded-frame reference encoding into split payload and opaque metadata."""
    import torch
    codec = _codec(codec)
    _width(value.shape[-1])
    if value.dtype not in (torch.float16, torch.float32) or not bool(torch.isfinite(value).all()):
        raise ValueError("compact RotorQuant source needs finite FP16/FP32 values")
    groups = value.float().reshape(*value.shape[:-1], value.shape[-1] // GROUP, GROUP)
    norm = groups.square().sum(-1).sqrt()
    inv = torch.where(norm > 1e-10, norm.reciprocal(), 0.)
    rotated = rotate_ref((groups * inv[..., None]).reshape(value.shape), codec=codec)
    book = torch.tensor(CENTROIDS, dtype=torch.float32, device=value.device)
    # Strictly closer selection fixes lower-code ties like the pinned CUDA path.
    distance = (rotated[..., None] - book).abs()
    codes = distance.argmin(-1).to(torch.uint8)
    energy = book[codes.long()].reshape(groups.shape).square().sum(-1).sqrt()
    corrected = torch.where(energy > 1e-10, norm / energy, norm).half()
    if not bool(torch.isfinite(corrected).all()) or bool(((norm > 1e-10) & (corrected == 0)).any()):
        raise ValueError("compact RotorQuant corrected FP16 norm is outside its finite nonzero domain")
    pair = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    payload = pair[..., 0] | (pair[..., 1] << 4)
    words = torch.zeros((*corrected.shape, 2), dtype=torch.float16, device=value.device)
    words[..., 0] = corrected
    metadata = words.contiguous().view(torch.float32).reshape(corrected.shape)
    return payload, metadata


def dequant_ref(payload, metadata, *, codec=CODEC, inverse=False):
    """Decode a caller-bounded tile; metadata floats are reinterpreted, never converted."""
    import torch
    codec = _codec(codec)
    if payload.dtype != torch.uint8 or payload.shape[-1] <= 0 or payload.shape[-1] % 64:
        raise ValueError("compact RotorQuant payload needs complete uint8 64-byte groups")
    expected = (*payload.shape[:-1], payload.shape[-1] // 64)
    if (metadata.dtype != torch.float32 or tuple(metadata.shape) != expected
            or metadata.device != payload.device or not metadata.is_contiguous()):
        raise ValueError("compact RotorQuant needs matching opaque four-byte metadata groups")
    halves = metadata.view(torch.float16).reshape(*expected, 2)
    norm = halves[..., 0]
    reserved_zero = (metadata.view(torch.int32) >> 16) == 0
    if not bool((torch.isfinite(norm) & (norm >= 0) & reserved_zero).all()):
        raise ValueError("compact RotorQuant requires finite nonnegative norm and reserved zero")
    codes = torch.stack((payload & 15, payload >> 4), -1).reshape(*payload.shape[:-1], payload.shape[-1] * 2)
    book = torch.tensor(CENTROIDS, dtype=torch.float32, device=payload.device)
    result = (book[codes.long()].reshape(*expected, GROUP) * norm.float()[..., None]).reshape(codes.shape)
    return rotate_ref(result, codec=codec, inverse=True) if inverse else result


def _iso_scalar(values, group, inverse):
    w, x, y, z = (table[group] for table in (QW, QX, QY, QZ))
    if inverse:
        x, y, z = -x, -y, -z
    a, b, c, d = values
    def mul(left, right):
        return _f32(left * right)
    def add(left, right):
        return _f32(left + right)
    def sub(left, right):
        return _f32(left - right)
    return (sub(sub(sub(mul(w, a), mul(x, b)), mul(y, c)), mul(z, d)),
            sub(add(add(mul(w, b), mul(x, a)), mul(y, d)), mul(z, c)),
            add(add(sub(mul(w, c), mul(x, d)), mul(y, a)), mul(z, b)),
            add(sub(add(mul(w, d), mul(x, c)), mul(y, b)), mul(z, a)))


def encode_oracle(values, *, codec=CODEC):
    """Independent scalar wire oracle; returns source-compatible 68-byte blocks.

    Scalar FP32 arithmetic deliberately has its own reduction order. Exact
    encoding identity is asserted only on inputs clear of quantizer boundaries.
    """
    codec = _codec(codec)
    _width(len(values))
    if any(not math.isfinite(v) for v in values):
        raise ValueError("compact RotorQuant source values must be finite")
    result = bytearray()
    for start in range(0, len(values), GROUP):
        block = [_f32(v) for v in values[start:start+GROUP]]
        energy = 0.
        for v in block:
            energy = _f32(energy + _f32(v*v))
        norm = _f32(math.sqrt(energy))
        inv = _f32(1./norm) if norm > 1e-10 else 0.
        codes = []
        if codec == ISO_CODEC:
            rotated = []
            for group in range(32):
                unit = [_f32(block[4*group+i]*inv) for i in range(4)]
                rotated.extend(_iso_scalar(unit, group, False))
            codes = [min(range(16), key=lambda k: abs(_f32(v-CENTROIDS[k]))) for v in rotated]
        else:
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
            raise ValueError("compact RotorQuant norm exceeds FP16") from exc
        if norm > 1e-10 and struct.unpack('<e', header[:2])[0] == 0:
            raise ValueError("compact RotorQuant norm underflows FP16")
        result.extend(header)
        result.extend(codes[j] | (codes[j+1] << 4) for j in range(0,GROUP,2))
    return bytes(result)


def decode_oracle(data, *, codec=CODEC):
    """Independent source-block decoder, including inverse rotation after scale."""
    codec = _codec(codec)
    if not data or len(data) % BLOCK_BYTES:
        raise ValueError("compact RotorQuant wire data needs complete 68-byte blocks")
    output = []
    for start in range(0,len(data),BLOCK_BYTES):
        norm,reserved = struct.unpack('<ee',data[start:start+4])
        if not math.isfinite(norm) or norm < 0 or data[start+2:start+4] != b'\0\0':
            raise ValueError("compact RotorQuant wire metadata is invalid")
        if codec == ISO_CODEC:
            codes = [component for byte in data[start+4:start+BLOCK_BYTES] for component in (byte & 15, byte >> 4)]
            for group in range(32):
                values = [CENTROIDS[code] for code in codes[4*group:4*group+4]]
                output.extend(_f32(value * norm) for value in _iso_scalar(values, group, True))
        else:
            for p, byte in enumerate(data[start+4:start+BLOCK_BYTES]):
                a,b = CENTROIDS[byte&15],CENTROIDS[byte>>4]
                output.extend((_f32(_f32(_f32(COS[p]*a)+_f32(SIN[p]*b))*norm),
                               _f32(_f32(_f32(-SIN[p]*a)+_f32(COS[p]*b))*norm)))
    return output
