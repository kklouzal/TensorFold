"""One rank of tests/cuda/test_sm70_small_gather.py: ``python small_gather_worker.py RANK WORLD PORT SEED REPLAYS``.

Runs with one visible GPU and TF_SMALL_GATHER_BYTES set by the parent; prints one
JSON report line. SEED < 0 keeps the fresh generation sequence; otherwise every
rank seeds the same last generation (near wrap) before its first gather.
"""

import json
import sys

import torch

from tensorfold.cuda.comm import NCCL

DTYPES = ("float16", "float32", "int64")


def fill(out, rank, salt):
    """Deterministic per-(rank, salt) bytes on device (any float bits, NaN/Inf included)."""
    raw = out.reshape(-1).view(torch.uint8)
    index = torch.arange(raw.numel(), device=out.device, dtype=torch.int64)
    if not torch.is_tensor(salt):
        salt = torch.tensor(salt, device=out.device, dtype=torch.int64)
    value = (index * 2654435761 + salt * 40503 + rank * 7919 + 12345) % 1000003
    raw.copy_((value ^ (value >> 7)).to(torch.uint8))


def main():
    rank, world, port, seed, replays = (int(value) for value in sys.argv[1:6])
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    comm = NCCL()
    comm.open(rank, world, "127.0.0.1", port)
    small = comm.small_gather
    report = {"rank": rank, "enabled": small is not None, "capacity": comm.small_gather_bytes, "errors": []}
    if small is None:
        comm.close()
        print(json.dumps(report), flush=True)
        return
    if seed >= 0:
        small.seed_generation(seed)
    capacity = comm.small_gather_bytes

    def gather(send, recv, custom):
        saved = comm.small_gather
        comm.small_gather = saved if custom else None
        try:
            comm.all_gather(send, recv)
        finally:
            comm.small_gather = saved

    def same(a, b):
        return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))

    # Eager: bitwise equal to NCCL and the oracle, including in place and offset storage.
    sizes = (8, 12, 16, 100, 4096, 51200, capacity, capacity + 8)
    cases = 0
    for name in DTYPES:
        dtype = getattr(torch, name)
        size = torch.empty((), dtype=dtype).element_size()
        for nbytes in sizes:
            if nbytes % size:
                continue
            count = nbytes // size
            salt = DTYPES.index(name) * 1000 + sizes.index(nbytes)
            send = torch.empty(count, dtype=dtype, device=device)
            fill(send, rank, salt)
            want = torch.empty((world, count), dtype=dtype, device=device)
            for peer in range(world):
                fill(want[peer], peer, salt)
            custom = torch.zeros(world * count, dtype=dtype, device=device)
            nccl = torch.zeros_like(custom)
            gather(send, custom, True)
            gather(send, nccl, False)
            inplace = torch.zeros_like(custom)
            inplace[rank * count:(rank + 1) * count].copy_(send)
            gather(inplace[rank * count:(rank + 1) * count], inplace, True)
            big = torch.zeros(world * count + 1, dtype=dtype, device=device)
            shifted = torch.zeros(count + 1, dtype=dtype, device=device)
            shifted[1:].copy_(send)
            gather(shifted[1:], big[1:], True)
            for label, got in (("custom", custom), ("nccl", nccl), ("inplace", inplace), ("offset", big[1:])):
                if not same(got, want):
                    report["errors"].append(f"eager {label} {name} {nbytes}B")
            cases += 1
    torch.cuda.synchronize()
    report["eager_cases"] = cases

    # One captured graph of mixed calls (one above capacity stays NCCL), replayed with varying data.
    plan = [("float32", 5 * 2560), ("int64", 1), ("float16", 1001), ("float32", capacity // 4),
            ("float32", capacity // 4 + 2)]
    counter = torch.zeros((), dtype=torch.int64, device=device)
    sends = [torch.empty(count, dtype=getattr(torch, name), device=device) for name, count in plan]
    recvs = [torch.empty(world * count, dtype=getattr(torch, name), device=device) for name, count in plan]

    def body():
        counter.add_(1)
        for send, recv in zip(sends, recvs):
            fill(send, rank, counter)
            comm.all_gather(send, recv)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        body()
    counter.zero_()
    bad = torch.zeros((), dtype=torch.int64, device=device)
    wants = [torch.empty((world, count), dtype=getattr(torch, name), device=device) for name, count in plan]
    for replay in range(1, replays + 1):
        graph.replay()
        salt = torch.tensor(replay, device=device)
        for recv, want in zip(recvs, wants):
            for peer in range(world):
                fill(want[peer], peer, salt)
            bad += (recv.view(torch.uint8) != want.reshape(-1).view(torch.uint8)).any().to(torch.int64)
    torch.cuda.synchronize()
    if int(bad):
        report["errors"].append(f"graph: {int(bad)} mismatching gathers over {replays} replays")
    report["graph_replays"] = replays
    del graph
    comm.close()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
