"""Immutable launch options for the explicit, process-lifetime SM70 SDK profile.

Triton3.2 exposes ``ptx_version`` but its automatic CUDA-version table stops at
12.6. Its pinned LLVM backend documents PTX8.3 as its maximum feature set.
Requesting that ISA is independent of the real CUDA12.9 assembler identity;
the protected SDK auditor continues to verify the actual compiler/libdevice.
No vendor table, environment version, kernel math or modern profile is changed.
"""
import importlib.metadata
import os
from types import MappingProxyType


def compiler_options():
    profile = os.environ.get("TENSORFOLD_SM70_TRITON_PROFILE")
    if profile is None:
        return MappingProxyType({})
    if profile != "sm70-triton32-v1":
        raise ValueError("unsupported explicit SM70 Triton compiler profile")
    import triton

    if triton.__version__ != "3.2.0" or importlib.metadata.version("triton") != "3.2.0":
        raise ValueError("explicit PTX8.3 selection requires the actual Triton3.2 provider")
    return MappingProxyType({"ptx_version": 83})


# The SDK selection is immutable for the process lifetime. Bind once before
# cold compilation; warmup, live launches and conformance use this same mapping.
COMPILER_OPTIONS = compiler_options()
