"""Shared score/ID cases; CPU and CUDA greedy oracles remain independent."""

import numpy as np
import torch


def inputs(width, case):
    values = torch.from_numpy(np.random.default_rng(width).normal(size=(1, width)).astype(np.float32))
    if case == "ties":
        values.zero_()
    elif case == "zeros":
        values.view(torch.int32).view(-1).copy_(torch.from_numpy(
            np.resize(np.asarray([0, 0x80000000], dtype=np.uint32).view(np.int32), width)))
    elif case == "subnormal":
        values.view(torch.int32).view(-1).copy_(torch.from_numpy(np.resize(
            np.asarray([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff], dtype=np.uint32).view(np.int32), width)))
    elif case == "adjacent":
        values.fill_(1)
        values[0, -1] = torch.nextafter(torch.tensor(1.), torch.tensor(2.))
    elif case == "minus_inf":
        values.fill_(-float("inf"))
    elif case == "plus_inf":
        values[0, width//2] = float("inf")
    elif case == "nan":
        values.view(torch.int32)[0, 0] = 0x7fc00001
        values.view(torch.int32)[0, -1] = 0x7fc00002
    ids = torch.arange(400, 400+width, dtype=torch.int64).flip(0)
    return values, ids, torch.tensor([-3.125], dtype=torch.float32), torch.tensor([4, -1], dtype=torch.int32)

