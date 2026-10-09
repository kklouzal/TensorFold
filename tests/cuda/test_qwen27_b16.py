"""The bf16 Plain linear (the GDN gates of NVFP4 and EXL3 checkpoints): a row's bits never depend on the row count."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda.b16 import matmul, matmul_pair, prompt, prompt_pair  # noqa: E402


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
