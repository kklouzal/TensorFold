"""Native DFlash proposal/cache invariance on one CUDA GPU, separate from target quality gates."""
import json
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

from tensorfold.cuda.kernels import qmm
from tensorfold.cuda.requant import requantize
from tensorfold.families.qwen3_5.cuda.dflash2 import quantize4
from tensorfold.families.qwen3_5.cuda.qmm_fast import tile
from tensorfold.families.qwen3_5.cuda.weights import Plain
from tensorfold.families.qwen3_5_moe.cuda.dflash import DFlash
from tensorfold.families.qwen3_5_moe.dflash_metadata import DraftConfig, _affine4_bytes, draft_weights
from tests.test_qwen36_dflash import config

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


@pytest.mark.parametrize("rows,columns", [(64, 64), (128, 64), (192, 64), (128, 128)])
def test_native_affine_single_group_layout_keeps_exact_dyadic_outputs(rows, columns):
    words = torch.zeros((rows, columns // 8), dtype=torch.int32, device="cuda")
    scales = torch.ones((rows, columns // 64), dtype=torch.bfloat16, device="cuda")
    biases = torch.full_like(scales, 1 / 64)
    packed = qmm.pack(words, scales, biases, 64)
    actual = qmm.matmul(torch.ones((2, columns), dtype=torch.bfloat16, device="cuda"), packed)
    assert torch.equal(actual, torch.full((2, rows), columns / 64, dtype=torch.bfloat16, device="cuda"))


@pytest.mark.parametrize("n,k", [(64, 64), (192, 128), (4160, 64), (2048, 10240), (6144, 2048)])
def test_native_affine_reference_build_fits_its_admitted_phase_bound(n, k):
    weight = torch.linspace(-.1, .1, n * k, device="cuda").reshape(n, k).bfloat16()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    linear = tile(quantize4(weight))
    torch.cuda.synchronize()
    resident, staging = _affine4_bytes(n, k)
    assert torch.cuda.memory_allocated() - before == resident
    # The admission phase includes its BF16 source, which predates this peak counter's interval.
    assert torch.cuda.max_memory_allocated() - before + weight.numel() * weight.element_size() <= resident + staging
    assert linear.n == n and linear.k == k


@pytest.mark.parametrize("bits", [16, 8, 4])
def test_native_proposals_and_contexts_preserve_stream_partition_and_rewind(tmp_path, bits):
    raw = config()
    raw["rope_scaling"] = {"rope_type": "yarn", "factor": 8, "original_max_position_embeddings": 16}
    cfg = DraftConfig.from_dict(raw)
    generator = torch.Generator().manual_seed(391)
    arrays = {name: ((torch.randn(shape, generator=generator) * .05).bfloat16() if len(shape) == 2
                     else torch.ones(shape, dtype=torch.bfloat16)) for name, shape in cfg.shapes().items()}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    save_file(arrays, tmp_path / "model.safetensors")

    class Target:
        config = NS(hidden=cfg.hidden, layers=cfg.target_layers, vocab=cfg.vocab)
        norm = torch.ones(cfg.hidden, dtype=torch.bfloat16, device="cuda")
        embed = Plain(torch.randn((cfg.vocab, cfg.hidden), generator=generator).bfloat16().cuda())
        head = Plain(torch.randn((cfg.vocab, cfg.hidden), generator=generator).bfloat16().cuda())

        def bind_draft_taps(self, layers):
            if tuple(layers) != cfg.tap_layers:
                raise ValueError("test target tap binding differs")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    draft = DFlash(tmp_path, Target(), context_limit=64, bits=bits)
    torch.cuda.synchronize()
    owned = sum(linear.nbytes() for linear in draft.linears.values())
    owned += sum(t.numel() * t.element_size() for t in [*draft.weights.values(), draft.inv_freq])
    owned += draft.target_head.nbytes() if bits == 8 else 0  # affine4 borrows the original head
    estimate = draft_weights(tmp_path, bits=bits)
    if bits == 4:
        assert draft.target_head is Target.head
        assert torch.cuda.memory_allocated() - before == estimate.resident
        assert torch.cuda.max_memory_allocated() - before <= estimate.resident + estimate.staging
    else:
        assert owned == estimate.resident
    if bits in (4, 8):                     # a target head stored quantized is borrowed as it is: no copy, none counted
        stored = Target()
        stored.head = requantize(Target.head.weight, "fp8")
        borrowed = DFlash(tmp_path, stored, context_limit=64, bits=bits)
        assert borrowed.target_head is stored.head
        if bits == 8:
            assert owned - draft.target_head.nbytes() == draft_weights(tmp_path, bits=bits, private_head=False).resident
        else:
            assert estimate.resident == draft_weights(tmp_path, bits=bits, private_head=False).resident
        del borrowed
    empty = draft.snapshot()
    taps = torch.randn((7, len(cfg.tap_layers) * cfg.hidden), generator=generator).bfloat16().cuda()
    draft.add_taps(taps[:3])
    short = draft.snapshot()
    short_copy = [t.clone() for t in short[0]]
    draft.add_taps(taps[3:])
    long = draft.snapshot()
    batch = draft.launch_blocks([short, long], [2, 4], 3)
    batch_tokens = [draft.finish_tree(block, snap[3], 3) for block, snap in zip(batch, [short, long])]
    for snap, pending, expected in zip([short, long], [2, 4], batch_tokens):
        draft.restore(snap)
        actual = draft.finish_tree(draft.launch_block(pending, 3), snap[3], 3)
        assert actual == expected
    assert all(torch.equal(a, b) for a, b in zip(short[0], short_copy))
    together = draft.add_taps_streams([empty, empty], [taps[:3], taps])
    for actual, expected in zip(together, [short, long]):
        assert actual[2:] == expected[2:]
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                   for left, right in zip(actual[:2], expected[:2]) for a, b in zip(left, right))
    draft.restore(short)
    draft.add_taps(taps[3:])
    assert all(torch.equal(a, b) for a, b in zip(draft.kc, long[0]))
    torch.cuda.synchronize()
