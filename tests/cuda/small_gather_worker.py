"""One rank of tests/cuda/test_sm70_small_gather.py:
``python small_gather_worker.py RANK WORLD PORT SEED REPLAYS WIDE``.

Runs with one visible GPU and TF_SMALL_GATHER_BYTES set by the parent; prints one
JSON report line. SEED < 0 keeps the fresh generation sequence; otherwise every
rank seeds the same last generation (near wrap) before its first gather. WIDE=1
opens the bulk mailbox for FP32 sums too (``fp32_rows``) and checks all_reduce_rows_f32.
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
    rank, world, port, seed, replays, wide = (int(value) for value in sys.argv[1:7])
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    comm = NCCL()
    comm.open(rank, world, "127.0.0.1", port, row_reduce=True, fp32_rows=bool(wide))
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

    # Row all-reduce: bitwise the gathered partials summed in rank order (FP32), rounded once to FP16;
    # eager at every admitted row count, then captured between gathers and replayed.
    def oracle(part):
        gathered = torch.empty(world * part.numel(), device=device)
        gather(part, gathered, False)
        gathered = gathered.view(world, *part.shape)
        total = gathered[0].clone()
        for peer in range(1, world):
            total = total + gathered[peer]
        return total.half()

    reduced = 0
    for rows in (1, 3, 4, 5, 8, 16, 22, 23, 40):
        generator = torch.Generator(device=device)
        generator.manual_seed(rows * 101 + rank)
        part = torch.randn(rows, 2560, device=device, generator=generator) * (10.0 ** (rows % 5 - 2))
        part[:, ::7] = -0.0
        got = torch.empty(rows, 2560, dtype=torch.float16, device=device)
        if comm.all_reduce_rows(part, got):
            reduced += 1
            if not same(got, oracle(part)):
                report["errors"].append(f"all-reduce {rows} rows")
    report["reduced_shapes"] = reduced
    # Bulk row all-reduce (TF_SMALL_ALLREDUCE_BYTES): the same oracle, called directly at shapes either path admits.
    bulk_shapes = 0
    stream = torch.cuda.current_stream().cuda_stream
    for rows in (5, 16, 40, 46, 64, 120, 160):
        if not small.bulk_chunk_rows(rows, 2560):
            continue
        generator = torch.Generator(device=device)
        generator.manual_seed(rows * 103 + rank)
        part = torch.randn(rows, 2560, device=device, generator=generator) * (10.0 ** (rows % 5 - 2))
        part[:, ::5] = -0.0
        got = torch.empty(rows, 2560, dtype=torch.float16, device=device)
        small.allreduce_bulk(part.data_ptr(), got.data_ptr(), rows, 2560, stream)
        bulk_shapes += 1
        if not same(got, oracle(part)):
            report["errors"].append(f"bulk all-reduce {rows} rows")
    report["bulk_shapes"] = bulk_shapes

    # FP32 bulk row all-reduce: bitwise the gathered partials summed in rank order (acc = x0; acc += x_r, FP32:
    # the Gemma 4 tails' order), with signed zeros, NaN, infinities and subnormals among the partials.
    def oracle32(part):
        gathered = torch.empty(world * part.numel(), device=device)
        gather(part, gathered, False)
        gathered = gathered.view(world, *part.shape)
        total = gathered[0].clone()
        for peer in range(1, world):
            total += gathered[peer]
        return total

    def special(part, rows, salt):
        generator = torch.Generator(device=device)
        generator.manual_seed(salt * 1009 + rank)
        part.copy_(torch.randn(part.shape, device=device, generator=generator) * (10.0 ** (rows % 5 - 2)))
        part[:, 0::11] = -0.0 if rank % 2 else 0.0         # mixed-sign zero columns: the sum's sign follows the order
        part[:, 1::11] = -0.0                              # all-negative zeros stay -0.0
        part[:, 2::11] = 1e-41 * (rank - 3)                # subnormals (no flush to zero on either side)
        if rank == world - 1:
            part[:, 3::22] = float("nan")
            part[:, 4::22] = float("inf")
        if rank == 1:
            part[:, 4::44] = float("-inf")                 # inf + -inf = NaN in some columns
        return part

    f32_shapes = 0
    for dims in (2816, 5632):
        for rows in (1, 2, 5, 16, 40, 72, 73):
            part = special(torch.empty(rows, dims, device=device), rows, dims + rows)
            got = torch.full((rows, dims), 7.0, device=device)
            if not comm.all_reduce_rows_f32(part, got):
                if wide and small.bulk_chunk_rows(rows, dims, True):
                    report["errors"].append(f"bulk f32 {rows}x{dims} refused")
                continue
            f32_shapes += 1
            if not same(got, oracle32(part)):
                report["errors"].append(f"bulk f32 all-reduce {rows}x{dims}")
    report["f32_shapes"] = f32_shapes
    part = torch.randn(5, 2560, device=device)
    got = torch.empty(5, 2560, dtype=torch.float16, device=device)
    want, mixed = oracle(part), torch.empty(world * 1001, dtype=torch.float16, device=device)
    filler = torch.empty(1001, dtype=torch.float16, device=device)
    fill(filler, rank, 7)

    wide_part = torch.randn(64, 2560, device=device)
    wide_got = torch.empty(64, 2560, dtype=torch.float16, device=device)
    wide_want = oracle(wide_part)

    f32_part = special(torch.empty(40, 5632, device=device), 40, 1)
    f32_got = torch.empty_like(f32_part)
    f32_want = oracle32(f32_part)

    def reduce_body():
        for _ in range(3):
            comm.all_reduce_rows(part, got)
            comm.all_gather(filler, mixed)
            comm.all_reduce_rows(wide_part, wide_got)  # bulk where TF_SMALL_ALLREDUCE_BYTES admits 64 rows
            if wide:
                comm.all_reduce_rows_f32(f32_part, f32_got)

    with torch.cuda.stream(side):
        reduce_body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        reduce_body()
    for _ in range(50):
        got.zero_()
        wide_got.zero_()
        f32_got.zero_()
        graph.replay()
        bad += (got.view(torch.int16) != want.view(torch.int16)).any().to(torch.int64)
        bad += (wide_got.view(torch.int16) != wide_want.view(torch.int16)).any().to(torch.int64)
        if wide:
            bad += (f32_got.view(torch.int32) != f32_want.view(torch.int32)).any().to(torch.int64)
    torch.cuda.synchronize()
    if int(bad):
        report["errors"].append(f"all-reduce graph: {int(bad)} mismatching replays")
    del graph
    comm.close()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
