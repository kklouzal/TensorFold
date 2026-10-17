"""Bounded native-A4 grouped candidate; selection requires native/integrated evidence.

Private preparation consumes the same admitted immutable six-field authority as
ExpertsA4. It batches existing checkpoint quantization/MMA contracts across
experts, preserving each projection's calibration and native K reduction order.
"""

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch

from .experts_a4 import PAIRS


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda import precision
    from tensorfold.cuda.build import MIN_CAPABILITY, load

    here = Path(__file__).parent
    return load(name="tensorfold_nvfp4_experts_a4_v1", sources=[str(here / "experts_a4.cpp"),
                str(here / "experts_a4.cu")], need=MIN_CAPABILITY,
                arch_specific=precision.own_math(torch.cuda.get_device_capability())["nvfp4"],
                extra_include_paths=[str(here)], extra_cuda_cflags=["-O3"], verbose=False)


@contextmanager
def _lease(bank, ids):
    if hasattr(bank, "cache"):
        with bank.cache.lease(bank.layer_id, ids) as result:
            yield result
    else:
        yield bank.fields, {expert: expert for expert in ids}


def _windows(groups, capacity):
    """At most128pairs and capacity distinct experts; same-expert items <=16."""
    batch, ids = [], []
    for expert in sorted(groups):
        for row, slot in groups[expert]:
            if len(batch) == PAIRS or (expert not in ids and len(ids) == capacity):
                yield batch, ids
                batch, ids = [], []
            if expert not in ids:
                ids.append(expert)
            batch.append((row, slot, expert))
    if batch:
        yield batch, ids


def run(bank, x, picks, out, *, prefill):
    """Candidate run with the same admitted public bank contract as ExpertsA4."""
    from tensorfold.cuda.kernels import qmm

    bank._check(x, picks, out, prefill=prefill)
    out.zero_()
    capacity = bank.cache.capacity if hasattr(bank, "cache") else bank.count
    ext = _ext()
    up_sk = 1 if prefill else qmm.split_k(bank.width, bank.dims)
    down_sk = 1 if prefill else qmm.split_k(bank.dims, bank.width)
    for lo in range(0, x.shape[0], PAIRS):
        groups = {}
        for row, values in enumerate(picks[lo:lo + PAIRS].cpu().tolist()):
            for slot, expert in enumerate(values):
                if 0 <= expert < bank.count:
                    groups.setdefault(expert, []).append((lo + row, slot))
        for pairs, ids in _windows(groups, capacity):
            with _lease(bank, ids) as (fields, mapping):
                members = [(row, slot, mapping[expert]) for row, slot, expert in pairs]
                items = []
                for first in range(len(members)):
                    if (first and members[first][2] == members[first - 1][2]
                            and first - items[-1][1] < 16):
                        cell, at, count = items[-1]
                        items[-1] = (cell, at, count + 1)
                    else:
                        items.append((members[first][2], first, 1))
                plan = torch.tensor(members, dtype=torch.int32, device=x.device)
                item = torch.tensor(items, dtype=torch.int32, device=x.device)
                count = len(members)
                codes = torch.empty((2, count, bank.dims // 2), dtype=torch.uint8, device=x.device)
                scales = torch.empty((2, count, bank.dims // 16), dtype=torch.uint8, device=x.device)
                gu = torch.empty((2, count, bank.width), dtype=torch.bfloat16, device=x.device)
                ext.quantize(x, plan, fields[4], codes, scales)
                ext.gemm(codes, scales, fields[0], fields[2], fields[4], item, plan,
                         gu, bank.width, bank.dims, up_sk, False)
                hidden = bank.swiglu(gu[0], gu[1])
                down_codes = torch.empty((1, count, bank.width // 2), dtype=torch.uint8, device=x.device)
                down_scales = torch.empty((1, count, bank.width // 16), dtype=torch.uint8, device=x.device)
                # Hidden rows are pair-indexed; destination row/slot/cell
                # remain in the original plan for down scatter.
                ext.quantize(hidden, plan, fields[5], down_codes, down_scales, True)
                ext.gemm(down_codes, down_scales, fields[1].unsqueeze(1), fields[3], fields[5], item, plan,
                         out, bank.dims, bank.width, down_sk, True)
