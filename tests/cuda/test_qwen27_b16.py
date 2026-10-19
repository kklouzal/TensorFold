"""The bf16 Plain linear (the GDN gates of NVFP4 and EXL3 checkpoints): a row's bits never depend on the row count."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda.b16 import matmul, matmul_pair, prompt, prompt_pair, rows16


@pytest.mark.parametrize("n,k", [(48, 5120), (96, 6144), (130, 256)])
def test_rows_do_not_depend_on_the_row_count(n, k):
    g = torch.Generator(device="cuda").manual_seed(n)
    w = (torch.randn((n, k), generator=g, device="cuda") * 0.02).bfloat16()
    x = (torch.randn((300, k), generator=g, device="cuda") * 0.5).bfloat16()
    alone = torch.cat([matmul(x[r:r + 1], w) for r in range(300)])
    for m in (1, 2, 3, 4, 5, 15, 16, 17, 33, 64, 128, 300):
        assert torch.equal(matmul(x[:m], w), alone[:m]), m
    want = (x.float() @ w.float().t()).bfloat16()
    assert (alone.float() - want.float()).abs().max().item() < 2e-2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prepared_row_reuse_keeps_shipping_singleton_bits(dtype):
    from tensorfold.families.qwen3_5.cuda.b16 import _ext

    generator = torch.Generator(device="cuda").manual_seed(17032)
    x = torch.empty(33, 2048, dtype=dtype, device="cuda").uniform_(-0.5, 0.5, generator=generator)
    weight = torch.empty(130, 2048, dtype=dtype, device="cuda").uniform_(-0.02, 0.02, generator=generator)
    singleton = torch.cat([matmul(row[None], weight) for row in x])
    for rows in (1, 2, 3, 4, 5, 8, 12, 16, 17, 33):
        for reuse in (2, 4, 8, 16):
            got = _ext().b16_linear_rows(x[:rows], weight, reuse)
            assert torch.equal(got.view(torch.int16), singleton[:rows].view(torch.int16))
    order = torch.tensor([2, 0, 1], device="cuda")
    for reuse in (2, 4, 8, 16):
        got = _ext().b16_linear_rows(x[order], weight, reuse)
        assert torch.equal(got.view(torch.int16), singleton[order].view(torch.int16))
    for invalid in (0, 1, 3, 32):
        with pytest.raises(RuntimeError, match="reused rows"):
            _ext().b16_linear_rows(x, weight, invalid)


# Every row-block branch of b16.cu's policy: R = 1, 2, 4 and 8, then 8 or 16 by weight size, and 16 rows x 16 warps
# from 512 rows; sizes straddle each branch boundary. (10240, 2048) and the Qwen3.6 head are at or past 20 Mi
# elements (R = 16), (4160, 2048) and (2048, 4096) below it (R = 8).
POLICY_ROWS = (1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 31, 63, 64, 65, 200, 511, 512, 513)


@pytest.mark.parametrize("n,k", [(10240, 2048), (4160, 2048), (2048, 4096), (248320, 2048)])
def test_row_block_policy_keeps_singleton_bits_at_real_shapes(n, k):
    g = torch.Generator(device="cuda").manual_seed(n ^ k)
    w = torch.empty((n, k), dtype=torch.bfloat16, device="cuda").uniform_(-.02, .02, generator=g)
    rows = max(POLICY_ROWS) if n * k < 2**28 else 17
    x = torch.empty((rows, k), dtype=torch.bfloat16, device="cuda").uniform_(-.5, .5, generator=g)
    alone = torch.cat([matmul(x[r:r + 1], w) for r in range(rows)])
    for m in POLICY_ROWS:
        if m <= rows:
            assert torch.equal(matmul(x[:m], w).view(torch.int16), alone[:m].view(torch.int16)), m
    bias = torch.empty((n,), dtype=torch.bfloat16, device="cuda").uniform_(-.1, .1, generator=g)
    biased = torch.cat([matmul(x[r:r + 1], w, bias) for r in range(17)])
    assert torch.equal(matmul(x[:17], w, bias).view(torch.int16), biased.view(torch.int16))


def test_paired_gates_follow_the_policy_with_each_weights_bits():
    g = torch.Generator(device="cuda").manual_seed(9)
    w0, w1 = (torch.empty((32, 2048), dtype=torch.bfloat16, device="cuda").uniform_(-.02, .02, generator=g)
              for _ in range(2))
    x = torch.empty((513, 2048), dtype=torch.bfloat16, device="cuda").uniform_(-.5, .5, generator=g)
    a0 = torch.cat([matmul(x[r:r + 1], w0) for r in range(513)])
    a1 = torch.cat([matmul(x[r:r + 1], w1) for r in range(513)])
    for m in POLICY_ROWS:
        y0, y1 = matmul_pair(x[:m], w0, w1)
        assert (torch.equal(y0.view(torch.int16), a0[:m].view(torch.int16))
                and torch.equal(y1.view(torch.int16), a1[:m].view(torch.int16))), m


@pytest.mark.parametrize("n,k", [(32, 2048), (130, 256), (2048, 4096), (8192, 2048), (248320, 2048)])
def test_sixteen_row_decode_tile_keeps_prompt_bits(n, k):
    """rows16 issues prompt's mma chain from a 16-row tile (128-, 64- or 32-column blocks): the same bits."""

    g = torch.Generator(device="cuda").manual_seed(n + k)
    w = torch.empty((n, k), dtype=torch.bfloat16, device="cuda").uniform_(-.05, .05, generator=g)
    x = torch.empty((16, k), dtype=torch.bfloat16, device="cuda").uniform_(-1, 1, generator=g)
    want = prompt(x, w)
    for m in range(1, 17):
        assert torch.equal(rows16(x[:m], w).view(torch.int16), want[:m].view(torch.int16)), m


def test_sixteen_row_decode_tile_refuses_more_rows():
    with pytest.raises(RuntimeError, match="M <= 16"):
        rows16(torch.zeros((17, 64), dtype=torch.bfloat16, device="cuda"),
               torch.zeros((8, 64), dtype=torch.bfloat16, device="cuda"))


@pytest.mark.parametrize("n,k", [(48, 5120), (96, 6144), (130, 256)])
def test_prompt_rows_are_chunk_invariant(n, k):
    """The prompt kernel: one K chain a row, so any chunking and every tile height give a row the same bits."""

    g = torch.Generator(device="cuda").manual_seed(3 * n)
    w = (torch.randn((n, k), generator=g, device="cuda") * 0.02).bfloat16()
    x = (torch.randn((700, k), generator=g, device="cuda") * 0.5).bfloat16()
    full = prompt(x, w)
    parts, a = [], 0
    for size in (1, 37, 128, 200, 334):
        parts.append(prompt(x[a:a + size].contiguous(), w))
        a += size
    assert torch.equal(torch.cat(parts), full)
    for bm in (32, 64, 128):
        assert torch.equal(prompt(x, w, bm), full), bm
    want = (x.float() @ w.float().t()).bfloat16()
    assert (full.float() - want.float()).abs().max().item() < 2e-2


@pytest.mark.parametrize("n0,n1,k", [(48, 48, 5120), (64, 40, 256)])
def test_paired_launches_keep_each_weights_bits(n0, n1, k):
    """The GDN gates b and a in one launch: each output equals its own launch's, decode rows and prompt rows."""

    g = torch.Generator(device="cuda").manual_seed(n0 + n1)
    w0 = (torch.randn((n0, k), generator=g, device="cuda") * 0.02).bfloat16()
    w1 = (torch.randn((n1, k), generator=g, device="cuda") * 0.02).bfloat16()
    x = (torch.randn((300, k), generator=g, device="cuda") * 0.5).bfloat16()
    for m in (1, 5, 64, 300):
        y0, y1 = matmul_pair(x[:m], w0, w1)
        assert torch.equal(y0, matmul(x[:m], w0)) and torch.equal(y1, matmul(x[:m], w1)), m
    if k % 64 == 0:
        y0, y1 = prompt_pair(x, w0, w1)
        assert torch.equal(y0, prompt(x, w0)) and torch.equal(y1, prompt(x, w1))


def test_native_prompt_alignment_and_large_empty_storage_metadata_refuse_before_allocation():
    from tensorfold.families.qwen3_5.cuda.b16 import _ext

    native = _ext()
    x = torch.zeros((1, 64), dtype=torch.bfloat16, device='cuda')
    w = torch.zeros((3, 64), dtype=x.dtype, device=x.device)
    unaligned = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)[1:].view_as(x)
    malformed = [(native.b16_prompt, (unaligned, w, 32)),
                 (native.b16_prompt_pair, (unaligned, w, w, 32)),
                 (native.b16_linear, (torch.empty((2**31, 0), dtype=x.dtype, device=x.device),
                                      torch.empty((1, 0), dtype=x.dtype, device=x.device), x[:0])),
                 (native.b16_prompt, (torch.empty((1, 0), dtype=x.dtype, device=x.device),
                                      torch.empty((2**31, 0), dtype=x.dtype, device=x.device), 128))]
    for function, arguments in malformed:
        allocated = torch.cuda.memory_allocated()
        with pytest.raises(RuntimeError):
            function(*arguments)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated


def test_native_prompt_capture_after_every_tile_is_warm():
    x = torch.randn((3, 128), dtype=torch.bfloat16, device='cuda')
    w = torch.randn((7, 128), dtype=x.dtype, device=x.device)
    for bm in (0, 32, 64, 128):
        wanted = prompt(x, w, bm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = prompt(x, w, bm)
        graph.replay()
        assert torch.equal(actual, wanted)


def test_native_prompt_specializations_switch_between_compatible_devices():
    from tensorfold.families.qwen3_5.cuda.b16 import _ext

    if torch.cuda.device_count() < 2:
        pytest.skip('second compatible CUDA device unavailable')
    if torch.cuda.get_device_capability(0) != torch.cuda.get_device_capability(1):
        pytest.skip('same compiled CUDA capability required on both devices')
    native = _ext()
    for device in (0, 1, 0, 1):
        x = torch.ones((3, 128), dtype=torch.bfloat16, device=f'cuda:{device}')
        w = torch.ones((7, 128), dtype=x.dtype, device=x.device)
        for bm in (0, 32, 64, 128):
            torch.cuda.synchronize(device)
            result = native.b16_prompt(x, w, bm)
            pair = native.b16_prompt_pair(x, w, w, bm)
            assert torch.equal(result, torch.full_like(result, 128))
            assert all(torch.equal(value, result) for value in pair)
