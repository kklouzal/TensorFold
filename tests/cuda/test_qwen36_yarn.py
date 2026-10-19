"""Qwen3.6 static YaRN on CUDA, affine MTP engine flow: every rotary site (prompt chunks and the MTP head's absorption,
verify trees and head drafts, CUDA graph captures, concurrent streams) rotates with the policy's amplitude, the
amplitude reaches the target's keys and the head's cache, and drafted decoding still equals serial decoding."""

import dataclasses

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_qwen36_moe import MIXED, PROMPTS, SAMPLED, SAMPLINGS, _drain, _model, _serial

from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen3_5.cuda import glue
from tensorfold.families.qwen3_5_moe.cuda import decode
from tensorfold.families.qwen3_5_moe.cuda.graphs import Graphs
from tensorfold.families.qwen3_5_moe.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.rope import RopeParameters

# --yarn-factor 4 over test_qwen36_moe's geometry (head_dim 128, partial rotary 0.25: its 32 rotary dims)
POLICY = RopeParameters.from_config({"text_config": {
    "head_dim": 128, "max_position_embeddings": 262144,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25}}}, 4.0)


def _yarn_model(amplitude: float = POLICY.attention_factor):
    """test_qwen36_moe's model under YaRN's frequencies and ``amplitude``, the two fields Config.read sets."""

    w, head = _model()
    assert "attention_origin" not in w.__dict__ and w.config.rope_dims == POLICY.rotary_dim
    w.config = dataclasses.replace(w.config, yarn=POLICY, rope_scale=float(amplitude))
    w.inv_freq = POLICY.inverse_frequencies(torch).cuda()
    return w, head


def test_every_rotary_site_takes_the_amplitude_and_drafts_keep_serial_tokens(monkeypatch):
    w, head = _yarn_model()
    seen, rotate = [], glue.attn_prep

    def recorded(*args, **kwargs):
        seen.append(kwargs.get("amplitude"))
        return rotate(*args, **kwargs)

    monkeypatch.setattr(glue, "attn_prep", recorded)
    runner = Graphs(w, head, 128)
    for prompt, sampling in zip(PROMPTS, SAMPLINGS):
        want = _serial(w, prompt, sampling, 24)
        for graphs in (None, runner):
            st, mc, first, carry = decode.prefill(w, head, prompt, sampling)
            got = decode.mtp_decode(w, head, st, mc, carry, first, 24, sampling, depth=3, confidence=0.0,
                                    runner=graphs)
            assert got.tokens == want, (prompt, graphs is not None)
    assert runner.target and runner.mtp                  # both graph kinds captured under the policy
    dec, outs = MultiDecoder(w, head, depth=3, confidence=0.3), []
    for prompt, sampling in zip(MIXED, SAMPLED):         # multi_tree_forward and the head's forward_streams
        got: list[int] = []
        dec.admit(Stream(prompt, 16, sampling, emit=lambda new, got=got: got.extend(new)))
        outs.append(got)
    _drain(dec)
    assert outs == [_serial(w, p, smp, 16) for p, smp in zip(MIXED, SAMPLED)]
    assert seen and set(seen) == {POLICY.attention_factor}


def test_the_amplitude_reaches_the_target_keys_and_the_head_cache():
    """Same weights and frequencies, amplitude 1 versus YaRN's: the attention layer (after a DeltaNet layer the
    amplitude cannot reach) keeps its unrotated key dims bit for bit and scales its rotated ones; the head's cached
    keys, which read the target's attention output, change."""

    runs = []
    for amplitude in (1.0, POLICY.attention_factor):
        w, head = _yarn_model(amplitude)
        st, mc, _, _ = decode.prefill(w, head, PROMPTS[1], None)
        runs.append((st.kv[1][0][:st.pos].float(), mc.k[:mc.pos]))
    (k1, h1), (k2, h2) = runs
    rot = POLICY.rotary_dim
    assert torch.equal(k1[..., rot:], k2[..., rot:])
    # each side is one bf16 rounding (unit 2**-8) of the same fp32 rotation, scaled or not: 2**-7 * |k2| / (1 - 2**-8)
    # relative, plus the fp32 error of xn*cos - xp*sin (a few 2**-24 of terms below 2**1.5 * the largest key)
    tolerance = 2 ** -7 * 1.01 * k2[..., :rot].abs() + 2 ** -18 * k1.abs().max()
    assert ((k2[..., :rot] - POLICY.attention_factor * k1[..., :rot]).abs() <= tolerance).all()
    assert not torch.equal(k1[..., :rot], k2[..., :rot]) and not torch.equal(h1, h2)
