"""Bounded grouped native-A4 execution for resident fields and one-source leases."""

from functools import lru_cache
from pathlib import Path

import torch

PAIRS = 128


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda import precision
    from tensorfold.cuda.build import MIN_CAPABILITY, load

    here = Path(__file__).parent
    return load(name="tensorfold_nvfp4_experts_a4_v5", sources=[str(here / "experts_a4.cpp"),
                str(here / "experts_a4.cu")], need=MIN_CAPABILITY,
                arch_specific=precision.own_math(torch.cuda.get_device_capability())["nvfp4"],
                extra_include_paths=[str(here)], extra_cuda_cflags=["-O3", "--ftz=false", "--prec-div=true"], verbose=False)


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


def run(bank, x, picks, out, *, prefill, maximum_id=None):
    """Fill routed slots with the prepared bank's exact native A4 math."""
    from tensorfold.cuda.kernels import qmm

    bank._check(x, picks, out, prefill=prefill)
    out.zero_()
    capacity = bank.capacity
    ext = _ext()
    direct = getattr(bank, "direct_input", False)
    up_sk = 1 if prefill else qmm.split_k(bank.width, bank.dims)
    down_sk = 1 if prefill else qmm.split_k(bank.dims, bank.width)
    for lo in range(0, x.shape[0], PAIRS):
        groups = {}
        snapshot = picks[lo:lo + PAIRS].cpu()
        if maximum_id is not None and bool(((snapshot < -1) | (snapshot > maximum_id)).any()):
            raise ValueError("native A4 local picks contain an out-of-range expert")
        for row, values in enumerate(snapshot.tolist()):
            for slot, expert in enumerate(values):
                if 0 <= expert < bank.count:
                    groups.setdefault(expert, []).append((lo + row, slot))
        for pairs, ids in _windows(groups, capacity):
            with bank.lease(ids) as (fields, mapping):
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
                ext.quantize(x, plan, fields[4], codes, scales, False, direct)
                ext.gemm(codes, scales, fields[0], fields[2], fields[4], item, plan,
                         gu, bank.width, bank.dims, up_sk, False, direct)
                hidden = bank.swiglu(gu[0], gu[1])
                down_codes = torch.empty((1, count, bank.width // 2), dtype=torch.uint8, device=x.device)
                down_scales = torch.empty((1, count, bank.width // 16), dtype=torch.uint8, device=x.device)
                # Hidden rows are pair-indexed; destination row/slot/cell
                # remain in the original plan for down scatter.
                ext.quantize(hidden, plan, fields[5], down_codes, down_scales, True, direct)
                native_down = fields[1].unsqueeze(1) if direct else bank.native_down
                ext.gemm(down_codes, down_scales, native_down, fields[3], fields[5], item, plan,
                         out, bank.dims, bank.width, down_sk, True, direct)
