"""Execute the actual shared C++/CUDA decoder against an independent wire oracle."""

from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires the declared native host C++ compiler")
    directory = tmp_path_factory.mktemp("nvfp4-host-decode")
    source = directory / "probe.cpp"
    source.write_text(r'''
#include "nvfp4_decode.cuh"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
int main() {
    unsigned char code, scale, bytes[4];
    while (std::cin.read(reinterpret_cast<char*>(&code),1)) {
        if (!std::cin.read(reinterpret_cast<char*>(&scale),1) || !std::cin.read(reinterpret_cast<char*>(bytes),4)) return 2;
        uint32_t bits=uint32_t(bytes[0])|uint32_t(bytes[1])<<8|uint32_t(bytes[2])<<16|uint32_t(bytes[3])<<24;
        float factor; std::memcpy(&factor,&bits,4);
        float value=tf_sm70_nvfp4::value(code,scale,factor); std::memcpy(&bits,&value,4);
        for (unsigned i=0;i<4;++i) bytes[i]=static_cast<unsigned char>(bits>>(8*i));
        std::cout.write(reinterpret_cast<char*>(bytes),4);
    }
    return std::cin.eof()?0:3;
}
''')
    executable = directory / "probe"
    root = Path(__file__).resolve().parents[1] / "src/tensorfold/cuda/sm70"
    subprocess.run([compiler, "-std=c++17", "-O3", "-ffp-contract=off", "-fno-fast-math", "-I", str(root),
                    str(source), "-o", str(executable)], check=True, capture_output=True)
    return executable


def test_every_finite_e4m3_and_e2m1_code_preserves_declared_fp32_operation_order(probe):
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
    records, expected = [], []
    for factor in (np.float32(.125), np.float32(1e-37), np.float32(65519)):
        for block in range(256):
            if block & 127 == 127:
                continue  # Reserved NaNs are rejected by immutable import.
            exponent, fraction = (block >> 3) & 15, block & 7
            magnitude = fraction / 512 if not exponent else (1 + fraction / 8) * 2.0 ** (exponent - 7)
            scale = np.float32(-magnitude if block & 128 else magnitude)
            for code in range(16):
                records.append(bytes((code, block)) + struct.pack("<f", factor))
                first = np.float32(levels[code] * scale)
                expected.append(np.float32(first * factor))
    result = subprocess.run([str(probe)], input=b"".join(records), check=True, capture_output=True)
    actual = np.frombuffer(result.stdout, dtype="<u4")
    assert np.array_equal(actual, np.asarray(expected, dtype=np.float32).view(np.uint32))
