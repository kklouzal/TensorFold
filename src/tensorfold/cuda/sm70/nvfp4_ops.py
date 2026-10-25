"""Bounded NVFP4 software decode on Volta; no native FP4/FP8 matrix instruction."""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

from .nvfp4_layout import EXPERT_ABI, EXPERT_WORDS, PROFILE


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .source_cache import stage_sources

    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("NVFP4 software tile execution requires an actual SM70 device")
    sources = [Path(__file__).with_suffix(suffix) for suffix in (".cpp", ".cu")]
    headers = [Path(__file__).with_name(name) for name in ("nvfp4_layout.cuh", "nvfp4_decode.cuh")]
    identity = hashlib.sha256(b"".join(path.read_bytes() for path in [*sources, *headers])).hexdigest()
    module = build.load("tf_sm70_nvfp4_" + identity[:20], stage_sources(sources, headers=headers), need=(7, 0),
                        extra_cuda_cflags=["-O3", "--fmad=false"], extra_cflags=["-O3"])
    if (module.numerical_profile != PROFILE or module.expert_abi != EXPERT_ABI
            or module.expert_descriptor_words != EXPERT_WORDS):
        raise RuntimeError("compiled SM70 NVFP4 numerical interface differs from the declared source")
    return module


def decode_tile(codes, scales, global_scale, out, *, row, column, rows, columns):
    """Decode an admitted view; pad the complete caller-owned tile with zeros.

    Payload finite/range checks belong to immutable import preparation. This
    operator validates shapes, ownership and view bounds before device reads.
    """
    extension().decode_tile(codes, scales, global_scale, out, row, column, rows, columns)
    return out


def matmul(x, codes, scales, global_scale, out):
    """Maintained fused SM70 WMMA candidate; selection requires a named manifest."""
    extension().matmul(x, codes, scales, global_scale, out)
    return out


def experts(x, picks, descriptor, workspace, out, *, width):
    """GPU grouping/WMMA trial over immutable import-owned descriptors.

    Counts, stable members, actual M32 group tiles and activation storage are
    caller-owned. No selector/count readback or serving-time allocation occurs.
    Corrupted private picks/descriptors trigger the declared fatal cohort trap.
    """
    extension().experts(x, picks, descriptor, workspace.counts_device, workspace.starts_device,
        workspace.members_device, workspace.tile_expert, workspace.tile_begin, workspace.work_count,
        workspace.activation_device, out, width)
    return out
