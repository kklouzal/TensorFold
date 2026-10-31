"""Capture-safe fused FP16 family operations behind the shared family ABI.

Staged indexer entrypoints are internal owner-mediated operations. History,
State and Weights construct their writable stage/ring/pooled tensors in
separate allocations from position/status/live/origin and model norm/frequency
reads, then retain those storages through stream/graph completion. These
immutable ownership facts are established at construction, not re-proved on
every native launch; direct internal callers must preserve the same contract.
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
    sources = [root / "family_bindings.cpp", root / "family.cu"]
    identity = hashlib.sha256(prepared.__name__.encode())
    headers = [root / "torch_reduce.cuh", root.parent / "kernels/tensor_contracts.cuh",
               root / "rotorquant_q4.cuh", root / "rotorquant_q4_tables.cuh"]
    for source in [*sources, *headers]:
        identity.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load("tf_sm70_family_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cuda_cflags=["-O3", "--fmad=false"], extra_cflags=["-O3"])
