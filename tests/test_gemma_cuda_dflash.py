"""Gemma DFlash companion on CUDA against independent Torch equations; persistent context lifecycle.

Proposals never change output (the target verifies every token), so the
companion contract is: the official equations within BF16/FP32 reduction-order
tolerance, deterministic proposals, and committed context that only target
-accepted taps change.
"""

import dataclasses
import json
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

import torch.nn.functional as F  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from tensorfold.families.gemma4.cuda import dflash  # noqa: E402
from tensorfold.families.gemma4.dflash_metadata import DraftConfig  # noqa: E402
from test_gemma_cuda_dflash_metadata import config  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the companion runs CUDA kernels")
DEV = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def norm(x, weight, eps):
    return (x.float() * (x.float().pow(2).mean(-1, keepdim=True) + eps).rsqrt()).to(x.dtype) * weight


def rotate(x, positions, cfg):
    # Direct Transformers5.6 FP32 frequency construction; independent of the
    # adapter's stored inverse-frequency tensor and rotary kernel.
    inverse = 1.0 / (cfg.rope.theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
    phase = (inverse[None, :, None] @ positions[None, None, :].float()).transpose(1, 2)
    phase = torch.cat((phase, phase), -1)
    left, right = x.chunk(2, -1)
    return x * phase.cos().to(x.dtype) + torch.cat((-right, left), -1) * phase.sin().to(x.dtype)


def raw(**changes):
    # Head width 64 and hidden 128: the smallest geometry of the companion's CUDA tiles.
    return {**config(), "hidden_size": 128, "intermediate_size": 128, "head_dim": 64, **changes}


def make_fixture(directory, *, window=8, causal=None, kv_heads=2):
    values = raw(sliding_window=window, num_key_value_heads=kv_heads)
    if causal is not None:
        values["is_causal"] = causal
    cfg = DraftConfig.from_dict(values)
    generator = torch.Generator().manual_seed(1493)
    arrays = {name: ((torch.randn(shape, generator=generator) * .05).bfloat16() if len(shape) == 2
                     else (torch.rand(shape, generator=generator) * .2 + .9).bfloat16())
              for name, shape in cfg.shapes().items()}
    (directory / "config.json").write_text(json.dumps(values))
    save_file(arrays, directory / "model.safetensors")

    class Target:
        config = NS(hidden=cfg.hidden, layers=cfg.target_layers, vocab=cfg.vocab)
        device = DEV
        tap_layers = cfg.tap_layers
        embed = torch.randn((cfg.vocab, cfg.hidden), generator=generator).bfloat16()
        head = torch.randn((cfg.vocab, cfg.hidden), generator=generator).bfloat16()

        def raw_embed_tokens(self, ids):
            return F.embedding(ids.long(), self.embed.to(ids.device))

        def embed_tokens(self, ids):
            pytest.fail("the official Torch companion must not receive Gemma-normalized embedding rows")

        def project_logits(self, hidden):
            # FP32 head: the comparison sees hidden-state differences, not BF16 logit rounding.
            return F.linear(hidden.float(), self.head.to(hidden.device).float()).div(30).tanh().mul(30)

    return cfg, arrays, Target()


def reference_block(taps, anchor, target, cfg, arrays, *, length=None):
    """Complete raw-context equations (CPU, BF16 storage, FP32 softmax) without adapter caches or kernels."""
    length = cfg.block if length is None else length
    context = norm(F.linear(taps, arrays["fc.weight"]), arrays["hidden_norm.weight"], cfg.eps)
    ids = torch.tensor([anchor] + [cfg.mask_id] * (length - 1))
    x = F.embedding(ids, target.embed)
    positions = torch.arange(len(taps) + length)
    query_positions, key_positions = positions[-length:, None], positions[None]
    for layer, kind in enumerate(cfg.layer_types):
        base = f"layers.{layer}."
        hidden = norm(x, arrays[base + "input_layernorm.weight"], cfg.eps)
        prefix = base + "self_attn."
        q = F.linear(hidden, arrays[prefix + "q_proj.weight"]).view(length, cfg.heads, cfg.head_dim)
        q = rotate(norm(q, arrays[prefix + "q_norm.weight"], cfg.eps).transpose(0, 1), positions[-length:], cfg)
        source = torch.cat((context, hidden))
        k = F.linear(source, arrays[prefix + "k_proj.weight"]).view(-1, cfg.kv_heads, cfg.head_dim)
        k = rotate(norm(k, arrays[prefix + "k_norm.weight"], cfg.eps).transpose(0, 1), positions, cfg)
        v = F.linear(source, arrays[prefix + "v_proj.weight"]).view(-1, cfg.kv_heads, cfg.head_dim).transpose(0, 1)
        key, value = [a.repeat_interleave(cfg.heads // cfg.kv_heads, 0) for a in (k, v)]
        visible = torch.ones((length, len(positions)), dtype=torch.bool)
        if cfg.causal_layers[layer]:
            visible &= key_positions <= query_positions
        if kind == "sliding_attention":
            visible &= query_positions - key_positions < cfg.sliding_window
            if not cfg.causal_layers[layer]:
                visible &= key_positions - query_positions < cfg.sliding_window
        scores = q.float() @ key.float().transpose(-1, -2) * cfg.head_dim ** -.5
        probabilities = scores.masked_fill(~visible, -torch.inf).softmax(-1)
        attended = (probabilities @ value.float()).bfloat16().transpose(0, 1).reshape(length, -1)
        x = x + F.linear(attended, arrays[prefix + "o_proj.weight"])
        hidden = norm(x, arrays[base + "post_attention_layernorm.weight"], cfg.eps)
        activation = F.silu(F.linear(hidden, arrays[base + "mlp.gate_proj.weight"]))
        activation = activation * F.linear(hidden, arrays[base + "mlp.up_proj.weight"])
        x = x + F.linear(activation, arrays[base + "mlp.down_proj.weight"])
    hidden = norm(x[1:], arrays["norm.weight"], cfg.eps)
    return (F.linear(hidden.float(), target.head.float()) / 30).tanh() * 30


# Attention rounds probabilities to BF16 before P@V (as the target kernels do) while the
# equations keep FP32 probabilities; the drift through two layers measured max 0.14, mean 0.027.
TOLERANCE = 0.25
MEAN_TOLERANCE = 0.04


def assert_matches(proposal, expected):
    """Logits within reduction-order tolerance of the equations; ids equal wherever the margin is clear."""
    error = (proposal.logits.cpu() - expected).abs()
    assert error.max() <= TOLERANCE and error.mean() <= MEAN_TOLERANCE
    top = expected.topk(2, -1).values
    clear = (top[:, 0] - top[:, 1]) > 2 * TOLERANCE
    assert clear.any()
    assert torch.equal(proposal.logits.cpu().argmax(-1)[clear], expected.argmax(-1)[clear])


def taps_for(cfg, rows, seed):
    return torch.randn((rows, len(cfg.tap_layers) * cfg.hidden), generator=torch.Generator().manual_seed(seed)).bfloat16()


@cuda
@pytest.mark.parametrize("window,rows", [(8, 7), (8, 8), (8, 9), (8, 40), (128, 127), (128, 129), (128, 300)])
def test_proposals_match_independent_equations_across_swa_boundary(tmp_path, window, rows):
    cfg, arrays, target = make_fixture(tmp_path, window=window)
    draft = dflash.DFlash(tmp_path, target, context_limit=8192, rows=128)
    state = draft.make_state()
    taps = taps_for(cfg, rows, 101)
    for start in range(0, rows, 128):
        draft.append(state, taps[start:start + 128].to(DEV))
    proposal = draft.propose(state, 2, 3)
    assert_matches(proposal, reference_block(taps, 2, target, cfg, arrays))
    if rows >= 2 * window:
        # The tolerance detects a companion that ignored its sliding window.
        wrong = reference_block(taps, 2, target, dataclasses.replace(cfg, sliding_window=1 << 20), arrays)
        assert (proposal.logits.cpu() - wrong).abs().max() > TOLERANCE
    # Proposing never commits: the block's slots are beyond the committed length.
    again = draft.propose(state, 2, 3)
    assert state.length == rows and torch.equal(again.logits, proposal.logits)
    assert len(proposal.tokens) == 3


@cuda
@pytest.mark.parametrize("causal", [True, False])
def test_explicit_causal_policy_matches_authoritative_override(tmp_path, causal):
    cfg, arrays, target = make_fixture(tmp_path, causal=causal)
    draft = dflash.DFlash(tmp_path, target, context_limit=8192)
    state = draft.make_state()
    taps = taps_for(cfg, 13, 19)
    draft.append(state, taps.to(DEV))
    assert_matches(draft.propose(state, 3, 3), reference_block(taps, 3, target, cfg, arrays))


@cuda
def test_only_accepted_rows_commit_and_snapshot_restores_ring_history(tmp_path):
    cfg, arrays, target = make_fixture(tmp_path)
    draft = dflash.DFlash(tmp_path, target, context_limit=8192, rows=16)
    state = draft.make_state()
    assert state.rings[0] == 64                     # sliding window 8 + block 4 and 16 append rows, rounded
    taps = taps_for(cfg, 130, 93)

    def append(lane, begin, end, step=14):
        for at in range(begin, end, step):
            draft.append(lane, taps[at:min(at + step, end)].to(DEV))

    append(state, 0, 60)
    accepted = state.snapshot()
    # A longer branch wraps the 64-slot ring past the accepted window (positions 53..59 -> 117..123).
    append(state, 60, 130)
    assert state.length == 130
    state.restore(accepted)
    # A verifier accepted two rows; rejected rows never reached append.
    append(state, 60, 62)
    restored = draft.propose(state, 2, 3)
    assert_matches(restored, reference_block(taps[:62], 2, target, cfg, arrays))
    serial = draft.make_state()
    append(serial, 0, 62, step=16)
    assert torch.equal(draft.propose(serial, 2, 3).logits, restored.logits)
    fresh = draft.make_state()
    fresh.restore(accepted)
    append(fresh, 60, 62)
    assert torch.equal(draft.propose(fresh, 2, 3).logits, restored.logits)


@pytest.mark.parametrize("close_failure", [False, True])
def test_reader_failure_preserves_primary_and_cleanup_transport(tmp_path, monkeypatch, close_failure):
    _cfg, _arrays, target = make_fixture(tmp_path)
    primary, cleanup = KeyboardInterrupt("read failed"), OSError("reader close failed")
    events = []

    class Reader:
        def __init__(self, paths):
            events.append("open")

        def get(self, name):
            raise primary

        def close(self):
            events.append("close")
            if close_failure:
                raise cleanup

    monkeypatch.setattr(dflash, "SafeTensors", Reader)
    with pytest.raises(KeyboardInterrupt) as caught:
        dflash.DFlash(tmp_path, target, context_limit=8192)
    assert caught.value is primary and events == ["open", "close"]
    if close_failure:
        pending, found = [primary], set()
        while pending:
            error = pending.pop()
            if id(error) in found:
                continue
            found.add(id(error))
            pending.extend(e for e in (error.__cause__, error.__context__) if e is not None)
            if isinstance(error, BaseExceptionGroup):
                pending.extend(error.exceptions)
        assert id(cleanup) in found


def test_successful_reads_do_not_hide_reader_retirement_failure(tmp_path, monkeypatch):
    _cfg, arrays, target = make_fixture(tmp_path)
    failure = OSError("reader retirement failed after successful reads")

    class Reader:
        def __init__(self, paths):
            pass

        def get(self, name):
            return arrays[name]

        def close(self):
            raise failure

    monkeypatch.setattr(dflash, "SafeTensors", Reader)
    with pytest.raises(OSError) as caught:
        dflash.DFlash(tmp_path, target, context_limit=8192)
    assert caught.value is failure


def test_precision_and_attention_geometry_fail_before_payload_allocation(tmp_path, monkeypatch):
    _cfg, _arrays, target = make_fixture(tmp_path)
    for bits in (0, 4, 8, True):
        with pytest.raises(ValueError, match="BF16 storage"):
            dflash.DFlash(tmp_path, target, context_limit=8192, bits=bits)
    cfg = DraftConfig.from_dict(raw(num_attention_heads=64, sliding_window=2048))
    monkeypatch.setattr(dflash, "checked_headers", lambda path: (cfg, {}))

    def unexpected(*args):
        pytest.fail("invalid attention geometry reached payload allocation")

    monkeypatch.setattr(dflash, "SafeTensors", unexpected)
    with pytest.raises(ValueError, match="query group"):
        dflash.DFlash(tmp_path, target, context_limit=8192)
    # 2**24 positions x 2 KV heads x 64 columns reaches the int32 offset range of the full layer's cache.
    wide = DraftConfig.from_dict(raw(max_position_embeddings=1 << 24))
    monkeypatch.setattr(dflash, "checked_headers", lambda path: (wide, {}))
    with pytest.raises(ValueError, match="int32"):
        dflash.DFlash(tmp_path, target, context_limit=1 << 24)


def test_target_mismatch_and_checkpoint_python_are_checked_without_execution(tmp_path):
    _cfg, _arrays, target = make_fixture(tmp_path)
    target.tap_layers = (1, 2)
    with pytest.raises(ValueError, match="trained layer IDs"):
        dflash.DFlash(tmp_path, target, context_limit=8192)
    target.tap_layers = (0, 2)
    target.config.layers = 5
    with pytest.raises(ValueError, match="match the target"):
        dflash.DFlash(tmp_path, target, context_limit=8192)
    target.config.layers = 4
    values = raw()
    values["auto_map"] = {"AutoModel": "dflash.DFlashDraftModel"}
    (tmp_path / "config.json").write_text(json.dumps(values))
    (tmp_path / "dflash.py").write_text("raise RuntimeError('checkpoint Python must remain inert')")
    draft = dflash.DFlash(tmp_path, target, context_limit=8192)
    assert draft.bits == 16 and all(v.dtype == torch.bfloat16 for v in draft.weights.values())


@cuda
def test_invalid_taps_anchors_and_limits_preserve_context(tmp_path):
    cfg, _arrays, target = make_fixture(tmp_path)
    draft = dflash.DFlash(tmp_path, target, context_limit=20, rows=16)
    with pytest.raises(ValueError, match="fit the declared sliding window"):
        dflash.DFlash(tmp_path, target, context_limit=20, block=9)
    state = draft.make_state()
    assert draft.propose(state, 2, 3) is None                # nothing committed yet
    taps = taps_for(cfg, 17, 7).to(DEV)
    draft.append(state, taps[:5])
    width = taps.shape[1]
    for invalid in (taps[:5].float(), taps[:5, :width // 2], taps, taps[:5].cpu(), [1]):
        with pytest.raises(ValueError, match="accepted taps"):
            draft.append(state, invalid)
        assert state.length == 5
    for anchor in (-1, 32, True, 2.0):
        with pytest.raises(ValueError, match="anchors"):
            draft.propose(state, anchor, 3)
    for nodes in (-1, 128, True):
        with pytest.raises(ValueError, match="proposal count"):
            draft.propose(state, 2, nodes)
    assert draft.propose(state, 2, 0) is None
    draft.append(state, taps[:13])
    assert state.length == 18 and len(draft.propose(state, 2, 3).tokens) == 1   # two block rows fit the window
    draft.append(state, taps[:1])
    assert draft.propose(state, 2, 3) is None
    with pytest.raises(ValueError, match="accepted taps"):
        draft.append(state, taps[:2])
