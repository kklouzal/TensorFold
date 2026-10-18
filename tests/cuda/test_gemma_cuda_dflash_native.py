"""Native Gemma draft attention boundaries and immutable stream/rewind contracts."""

import json
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import torch.nn.functional as F
from safetensors.torch import save_file

from tensorfold.families.gemma4.cuda.dflash import DFlash
from tensorfold.families.gemma4.dflash_metadata import DraftConfig, draft_weights
from tensorfold.families.qwen3_5.cuda.draft_attention import block_attention
from tensorfold.families.qwen3_5.cuda.weights import Plain
from test_gemma_cuda_dflash_metadata import config

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


@pytest.mark.parametrize("sliding,causal", [(True, True), (False, False), (True, False), (False, True)])
def test_native_attention_matches_exact_uniform_oracle_at_2048_boundary(sliding, causal):
    # Zero q/k make every visible logit zero. The online softmax therefore adds
    # exact unit probabilities; integer-valued v followed by one division gives
    # an independent exact BF16 oracle without a chosen numerical tolerance.
    heads, kv, rows, dim, context = 4, 2, 4, 32, 2047
    q = torch.zeros((heads, rows, dim), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((kv, rows, dim), dtype=torch.bfloat16, device="cuda")
    v = torch.zeros_like(k)
    kc = torch.zeros((kv, context, dim), dtype=torch.bfloat16, device="cuda")
    vc = torch.zeros_like(kc)
    vc[:, 0] = 64
    vc[:, -1] = 32
    v[:, 0] = 8
    v[:, -1] = 16
    window = 2047 if sliding else 8192
    actual = block_attention(q, k, v, [kc], [vc], rows, window, dim ** -.5, causal)
    values = torch.cat((vc.cpu(), v.cpu()), 1).double()
    output = []
    for row in range(rows):
        query_position = context + row
        key_positions = torch.arange(context + rows)
        visible = torch.ones_like(key_positions, dtype=torch.bool)
        if causal:
            visible &= key_positions <= query_position
        if sliding:
            visible &= query_position - key_positions < 2048
            if not causal:
                visible &= key_positions - query_position < 2048
        averaged = (values[:, visible].sum(1) / int(visible.sum())).bfloat16()
        output.append(averaged.repeat_interleave(heads // kv, 0).reshape(-1))
    expected = torch.stack(output)
    assert torch.equal(actual.cpu(), expected)


def test_native_draft_bf16_residency_stream_partition_and_accepted_rewind(tmp_path):
    raw = {**config(), "sliding_window": 2048}
    cfg = DraftConfig.from_dict(raw)
    generator = torch.Generator().manual_seed(391)
    arrays = {name: ((torch.randn(shape, generator=generator) * .05).bfloat16() if len(shape) == 2
                     else torch.ones(shape, dtype=torch.bfloat16)) for name, shape in cfg.shapes().items()}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    save_file(arrays, tmp_path / "model.safetensors")

    class Target:
        config = NS(hidden=cfg.hidden, layers=cfg.target_layers, vocab=cfg.vocab)
        device = torch.device("cuda", torch.cuda.current_device())
        tap_layers = cfg.tap_layers
        embed = torch.randn((cfg.vocab, cfg.hidden), generator=generator).bfloat16().cuda()
        head = Plain(torch.randn((cfg.vocab, cfg.hidden), generator=generator).bfloat16().cuda())

        def raw_embed_tokens(self, ids):
            return F.embedding(ids.long(), self.embed)

        def embed_tokens(self, ids):
            pytest.fail("the official Torch companion must not receive Gemma-normalized embedding rows")

        def project_logits(self, hidden):
            return (self.head(hidden).float() / 30).tanh() * 30

    draft = DFlash(tmp_path, Target(), context_limit=8192)
    resident = sum(linear.nbytes() for linear in draft.linears.values())
    resident += sum(value.numel() * value.element_size() for value in [*draft.weights.values(), draft.inv_freq])
    assert resident == draft_weights(tmp_path).resident
    empty = draft.snapshot()
    taps = torch.randn((2053, 128), generator=generator).bfloat16().cuda()
    draft.add_taps(taps[:2047])
    short = draft.snapshot()
    copy = [[value.clone() for value in cache] for cache in short[:2]]
    draft.add_taps(taps[2047:])
    long = draft.snapshot()
    assert short[0][0].shape[1] == long[0][0].shape[1] == 2047
    assert short[0][1].shape[1] == 2047 and long[0][1].shape[1] == 2053
    launched = draft.launch_blocks([short, long], [2, 4], 3)
    batch = [draft.finish_tree(block, snap[3], 3) for block, snap in zip(launched, [short, long])]
    for snap, anchor, expected in zip([short, long], [2, 4], batch):
        draft.restore(snap)
        assert draft.finish_tree(draft.launch_block(anchor, 3), snap[3], 3) == expected
    assert all(torch.equal(a, b) for left, right in zip(short[:2], copy) for a, b in zip(left, right))
    together = draft.add_taps_streams([empty, empty], [taps[:2047], taps])
    for actual, expected in zip(together, [short, long]):
        assert actual[2:] == expected[2:]
        assert all(torch.equal(a, b) for left, right in zip(actual[:2], expected[:2]) for a, b in zip(left, right))
    # Restore the verified prefix, then add just the target-accepted row prefix.
    # Previously proposed/rejected rows were never written into the draft cache.
    draft.restore(short)
    draft.add_taps(taps[2047:2049])
    expected = draft.add_taps_streams([empty], [taps[:2049]])[0]
    assert all(torch.equal(a, b) for left, right in zip(draft.snapshot()[:2], expected[:2])
               for a, b in zip(left, right))
    torch.cuda.synchronize()
