"""Lossless original IQ4_NL block crop into caller storage; no new planes.

Input stable raw512x2560x360 bytes (GGUF K640); first0|256 and lane0..3
select E256/160 original columns. One expert230400B is staged at a time.
This performs NO activation or coefficient requantization. Q8_1 is a separate
explicit GPU activation method. Caller owns/retains source/output and heap.
"""

import hashlib
import numpy as np


def requirements(*, ggml_type):
    if type(ggml_type) is not int or ggml_type != 20:
        raise ValueError("original IQ4_NL only; Q2_0 remains incumbent")
    return {
        "output_shape": (256 * 2560, 90),
        "output_bytes": 256 * 2560 * 90,
        "expert_staging_bytes": 2560 * 90,
        "max_experts_in_flight": 1,
        "packing": "lossless-whole32-block-crop",
        "column_offset_after_crop": 0,
        "managed_heap_RSS_bound": "not-certified",
    }


def pack_owned_into(raw, output, *, first, lane, ggml_type):
    spec = requirements(ggml_type=ggml_type)
    if type(first) is not int or first not in (0, 256) or type(lane) is not int or lane not in range(4):
        raise ValueError("owned E256/ETP4 lane required")
    if (
        type(raw) not in (np.ndarray, np.memmap)
        or raw.dtype != np.uint8
        or raw.shape != (512, 2560, 360)
        or raw.strides[2] != 1
        or raw.strides[1] < 360
        or raw.strides[0] < 0
        or raw.strides[0]
        and raw.strides[0] < 2559 * raw.strides[1] + 360
    ):
        raise ValueError("stable original512x2560x360 IQ4_NL bank required")
    if (
        type(output) is not np.ndarray
        or output.dtype != np.uint8
        or output.shape != spec["output_shape"]
        or not output.flags.c_contiguous
        or not output.flags.writeable
        or np.may_share_memory(raw, output)
    ):
        raise ValueError("exclusive exact caller-owned raw crop output required")
    digest = hashlib.sha256()
    for expert in range(256):
        stage = np.array(
            raw[first + expert, :, lane * 90 : (lane + 1) * 90], dtype=np.uint8, copy=True, order="C", subok=False
        )
        digest.update(memoryview(stage))
        output[expert * 2560 : (expert + 1) * 2560] = stage
        del stage  # Retire the sole expert staging allocation before the next RHS.
    return {
        "schema": "tensorfold.onecat-down-original-crop-v1",
        "first": first,
        "lane": lane,
        "ggml_type": 20,
        "completed_experts": 256,
        "source_selected_payload_sha256": digest.hexdigest(),
        "output_sha256": hashlib.sha256(memoryview(output)).hexdigest(),
        "resources": spec,
        "coefficient_transform": "NONE; original18-byte blocks retained",
    }
