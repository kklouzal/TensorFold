"""Actual-CUDA target identity, independent attention and cache lifecycle checks."""

from __future__ import annotations

import pytest
import torch

from tensorfold.families.gemma4.cuda.experts import ExpertLinear, Experts
from tensorfold.families.gemma4.cuda.forward import Gemma4Target, apply_rope, attend, attention_reference, rms_norm
from tensorfold.families.gemma4.cuda.linear import Linear
from tensorfold.families.gemma4.cuda.state import capacity_bytes
from tensorfold.families.gemma4.cuda.weights import Config, Gemma4Weights, Layer


def config():
    return Config(hidden=64, layers=2, vocab=128, heads=2, kv_heads=1, global_kv_heads=1,
                  head_dim=64, global_head_dim=128, intermediate=128, moe_width=64, experts=4, top_k=2,
                  window=8, max_position=256, eps=1e-6, softcap=30.,
                  layer_types=("sliding_attention", "full_attention"), rope_theta=10000.,
                  global_rope_theta=1000000., global_rotary_fraction=.25, vision_bidirectional=True)


def packed(shape, bits, generator):
    words = torch.randint(0, 2**32, (*shape[:-1], shape[-1] * bits // 32),
                          dtype=torch.int64, generator=generator).to(dtype=torch.uint32, device="cuda")
    scales = (torch.randn((*shape[:-1], shape[-1] // 64), generator=generator) * .0007).to("cuda", torch.bfloat16)
    bias = (torch.randn(scales.shape, generator=generator) * .003).to("cuda", torch.bfloat16)
    return (ExpertLinear if len(shape) == 3 else Linear)(words, scales, bias, bits=bits, group=64)


def target(seed=319):
    c = config()
    generator = torch.Generator().manual_seed(seed)
    def norm(width=64):
        return (torch.randn((width,), generator=generator) * .1 + 1).to("cuda", torch.bfloat16)
    layers = []
    for i in range(c.layers):
        hk, d = c.attention_geometry(i)
        linear = lambda n, k, bits=4: packed((n, k), bits, generator)
        banks = [packed((c.experts, *shape), bits, generator) for shape, bits in (
            ((c.moe_width, c.hidden), 4), ((c.moe_width, c.hidden), 6), ((c.hidden, c.moe_width), 6))]
        layers.append(Layer(q=linear(c.heads * d, c.hidden), k=linear(hk * d, c.hidden),
            v=linear(hk * d, c.hidden, 6) if c.sliding(i) else None, o=linear(c.hidden, c.heads * d),
            gate=linear(c.intermediate, c.hidden), up=linear(c.intermediate, c.hidden),
            down=linear(c.hidden, c.intermediate, 6), router=linear(c.experts, c.hidden), experts=Experts(*banks),
            input_norm=norm(), attention_norm=norm(), mlp_norm=norm(), expert_norm=norm(),
            dense_output_norm=norm(), expert_output_norm=norm(), output_norm=norm(), q_norm=norm(d), k_norm=norm(d),
            router_scale=norm(), expert_scale=norm(c.experts), scalar=torch.tensor([.99],device="cuda",dtype=torch.bfloat16)))
    weights = Gemma4Weights(c, packed((c.vocab, c.hidden), 6, generator), norm(), layers, torch.device("cuda:0"), {})
    return Gemma4Target(weights)


@pytest.mark.parametrize("width", [64, 256, 512, 2816])
def test_norm_uses_direct_scale_and_bf16_boundary(width):
    torch.manual_seed(82)
    x = torch.randn((5, width), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((width,), device="cuda", dtype=torch.bfloat16)
    actual = rms_norm(x, w, 1e-6)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * w
    # FP32 reduction trees differ between the independent Torch oracle and
    # Triton. Two BF16 storage boundaries bound the comparison to two ULPs.
    torch.testing.assert_close(actual, expected, atol=0, rtol=2**-6)
    serial = torch.cat([rms_norm(row[None], w, 1e-6) for row in x])
    assert torch.equal(actual, serial)


@pytest.mark.parametrize("d,hk,h,window,position", [(64,1,2,8,11), (256,8,16,1024,65), (512,2,16,0,130)])
def test_attention_matches_independent_reference_and_row_partition(d,hk,h,window,position):
    torch.manual_seed(71)
    rows = 5
    first = max(0, position-window+1) if window else 0
    q = torch.randn((rows,h,d),device="cuda",dtype=torch.bfloat16) * .1
    k = torch.randn((position+rows-first,hk,d),device="cuda",dtype=torch.bfloat16) * .1
    v = torch.randn_like(k)
    no = torch.full((rows,),-1,device="cuda",dtype=torch.int32)
    actual = attend(q,k,v,position,first,window,no,no)
    expected = attention_reference(q,k,v,position,first,window)
    # The production tensor-core probability operand is BF16. Independently
    # bounded quantization: each probability changes <=2^-8 relative before
    # the weighted value sum; this absolute envelope covers this input's L1.
    tolerance = 2**-8 * v.float().abs().amax().item() + 2**-7 * expected.float().abs().amax().item()
    torch.testing.assert_close(actual,expected,rtol=0,atol=tolerance)
    serial = torch.cat([attend(q[i:i+1],k[:position+i+1-first],v[:position+i+1-first],
                                     position+i,first,window,no[:1],no[:1]) for i in range(rows)])
    assert torch.equal(actual,serial)


def test_proportional_rope_leaves_nonrotary_pairs_unchanged():
    model = target()
    inverse = model._inverse[1]
    assert torch.count_nonzero(inverse).item() == 16
    x = torch.randn((4,2,128),device="cuda",dtype=torch.bfloat16)
    actual = apply_rope(x,inverse,31)
    assert torch.equal(actual[:,:,16:64],x[:,:,16:64])
    assert torch.equal(actual[:,:,80:],x[:,:,80:])
    assert not torch.equal(actual[:,:,:16],x[:,:,:16])


def test_layer_major_rows_match_serial_logits_taps_and_rollback():
    model = target()
    a = model.make_state(128,max_rows=16)
    model.forward(list(range(17)),a)
    snapshot = a.snapshot()
    b = a.fork()
    tokens = torch.tensor([29,31,7,90,54],device="cuda",dtype=torch.int64)
    batch = model.forward(tokens,a)
    serial = model.forward_serial(tokens,b)
    assert torch.equal(batch.hidden,serial.hidden)
    assert torch.equal(batch.logits,serial.logits)
    for layer in batch.taps:
        assert torch.equal(batch.taps[layer],serial.taps[layer])
    a.rollback(19)
    b.rollback(19)
    left = model.forward([93],a)
    right = model.forward([93],b)
    assert torch.equal(left.logits,right.logits)
    a.restore(snapshot)
    replay = model.forward(tokens,a)
    assert torch.equal(batch.logits,replay.logits)
    assert a.position == 22
    assert a.nbytes <= capacity_bytes(model.config,128,16)


def test_forks_do_not_share_mutable_storage_and_rollback_is_bounded():
    model = target()
    a = model.make_state(128,max_rows=4)
    model.forward(list(range(32)),a,output_logits=False)
    b = a.fork()
    assert all(x.keys.data_ptr()!=y.keys.data_ptr() for x,y in zip(a.kv,b.kv))
    model.forward([87,88,89,90],a,output_logits=False)
    assert b.position == 32
    with pytest.raises(ValueError,match="retained"):
        a.rollback(20)
    assert a.position == 36


def test_visual_swa_reads_future_in_block_but_full_attention_remains_causal():
    torch.manual_seed(31)
    q = torch.randn((4,2,64),device="cuda",dtype=torch.bfloat16) * .1
    k = torch.randn((4,1,64),device="cuda",dtype=torch.bfloat16) * .1
    v = torch.randn_like(k)
    starts = torch.zeros((4,),device="cuda",dtype=torch.int32)
    ends = torch.full((4,),4,device="cuda",dtype=torch.int32)
    visual = attend(q,k,v,0,0,8,starts,ends)
    expected = attention_reference(q,k,v,0,0,8,[(0,4)])
    torch.testing.assert_close(visual,expected,atol=.02,rtol=.01)
    causal = attend(q,k,v,0,0,0,starts,ends)
    assert torch.equal(causal[0,0],v[0,0])
    assert not torch.equal(visual[0],causal[0])


def test_bad_visual_split_does_not_modify_state():
    model = target()
    state = model.make_state(64)
    with pytest.raises(ValueError,match="complete"):
        model.forward([1,2],state,vision_blocks=((0,4),))
    assert state.position == 0 and not state.poisoned


def test_embedding_is_scaled_once_and_tied_head_is_softcapped_once():
    model = target()
    ids = torch.tensor([12,57],device="cuda",dtype=torch.int64)
    raw = model.weights.embed.embedding(ids)
    assert torch.equal(model.raw_embed_tokens(ids),raw)
    assert torch.equal(model.embed_tokens(ids),raw * 8)
    h = torch.randn((2,64),device="cuda",dtype=torch.bfloat16)
    expected = torch.tanh(model.weights.embed(h)/30)*30
    assert torch.equal(model.project_logits(h),expected)


def test_interrupted_snapshot_publication_poisoned_until_complete_restore(monkeypatch):
    from tensorfold.families.gemma4.cuda.state import _Cache

    model = target()
    state = model.make_state(64)
    model.forward([1,2,3],state,output_logits=False)
    saved = state.snapshot()
    failed_cache = state.kv[1]
    original = _Cache.__setattr__
    def interrupted(self,name,value):
        if self is failed_cache and name == "keys":
            raise KeyboardInterrupt("controlled restore publication interruption")
        original(self,name,value)
    with monkeypatch.context() as scope:
        scope.setattr(_Cache,"__setattr__",interrupted)
        with pytest.raises(KeyboardInterrupt):
            state.restore(saved)
    assert state.poisoned
    with pytest.raises(RuntimeError,match="poisoned"):
        model.forward([4],state)
    state.restore(saved)
    assert not state.poisoned and state.position == 3
    model.forward([4],state)
