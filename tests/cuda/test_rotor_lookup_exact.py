"""Maintained native gate for the selected 64x256 decoder primitive.

ROOT runs this against the actual integrated installed source. Local stdlib
source checks must not import this module. Full model quality/P3 remain separate.
"""
import struct

import pytest
import torch
import triton
import triton.language as tl

from tensorfold.families.qwen4_exp.cuda import rotorquant_kernel as kernel
from tensorfold.families.qwen4_exp.cuda.rotorquant_ref import CENTROIDS_7

pytestmark = pytest.mark.torch


@triton.jit
def _decode(LOW, MIDDLE, HIGH, SCALE, OUT, LOOKUP: tl.constexpr):
    row = tl.arange(0, 64)
    lo = tl.load(LOW + row[:, None] * 128 + tl.arange(0, 128)[None, :])
    scale = tl.load(SCALE + row[:, None] * 2 + tl.arange(0, 2)[None, :])
    middle = tl.load(MIDDLE + row[:, None] * 64 + tl.arange(0, 64)[None, :])
    high = tl.load(HIGH + row[:, None] * 32 + tl.arange(0, 32)[None, :])
    if LOOKUP:
        out = kernel.dequant_group_7_lookup(lo, middle, high, scale, M=64, W=256)
    else:
        out = kernel.dequant_group_7(lo, middle, high, scale, M=64, W=256)
    tl.store(OUT + row[:, None] * 256 + tl.arange(0, 256)[None, :], out)


def bf16_word(centroid, scale):
    # Each operand is an exact binary32 value. Their <=48-bit product is
    # exactly represented in binary64, then rounded once to binary32 by pack.
    word, = struct.unpack('<I', struct.pack('<f', centroid * scale))
    upper, lower = word >> 16, word & 65535
    return (upper + int(lower > 32768 or lower == 32768 and upper & 1)) & 65535


def test_all_seven_bit_codes_original_and_lookup_match_independent_raw_word_oracle():
    if not torch.cuda.is_available():
        pytest.skip('native CUDA gate; a skip gives no qualification credit')
    book = CENTROIDS_7
    values = [0.0, 2.0**-80, .12345670163631439, .37, 13.0, 2.0**96]
    values = [struct.unpack('<f', struct.pack('<f', value))[0] for value in values]
    codes = [[(row * 256 + column) % len(book) for column in range(256)] for row in range(64)]
    scales = [[values[(row * 2 + group) % len(values)] for group in range(2)] for row in range(64)]
    low = [[line[i] & 15 | ((line[i + 1] & 15) << 4) for i in range(0, 256, 2)] for line in codes]
    middle = [[sum(((line[i + j] >> 4) & 3) << (j * 2) for j in range(4))
               for i in range(0, 256, 4)] for line in codes]
    high = [[sum(((line[i + j] >> 6) & 1) << j for j in range(8))
             for i in range(0, 256, 8)] for line in codes]
    low, middle, high = [torch.tensor(value, dtype=torch.uint8, device='cuda') for value in (low, middle, high)]
    scale = torch.tensor(scales, dtype=torch.float32, device='cuda')
    expected = [bf16_word(book[code], scales[row][column // 128])
                for row, line in enumerate(codes) for column, code in enumerate(line)]
    for lookup in (False, True):
        output = torch.empty((64, 256), dtype=torch.bfloat16, device='cuda')
        _decode[(1,)](low, middle, high, scale, output, LOOKUP=lookup, num_warps=4)
        actual = [word & 65535 for word in output.view(torch.int16).cpu().reshape(-1).tolist()]
        assert actual == expected
