"""YaRN at native/extended position edges and through sparse, image, graph and MTP CUDA execution.

These use synthetic tensors or the existing two-layer random fixture, never model
checkpoint weights. The largest boundary fixture allocates under 0.7 GiB; integrated
fixtures use 128-token caches. Full-model long-context quality is a separate gate.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402
from test_flashnext_vision import _image  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attention, glue  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.rope import RopeParameters  # noqa: E402


def _policy():
    return RopeParameters.from_config({"head_dim": 256, "max_position_embeddings": 262144,
        "rope_parameters": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144,
                            "partial_rotary_factor": 0.25, "rope_theta": 1e7,
                            "mrope_interleaved": True, "mrope_section": [11, 11, 10]}})


def _rotate_reference(x, scale, positions, inv, policy):
    """Independent PyTorch RMSNorm/rotate-half, with the production bf16 storage points."""

    width, half = x.shape[-1], inv.numel()
    xf = x.float()
    xn = (xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) / width + 1e-6) * scale).bfloat16().float()
    # Match the official module's recomposition by sliced assignment; the kernel
    # independently selects axes with index modulo three and section bounds.
    angles = positions[..., 0, None].float() * inv
    for axis in (1, 2):
        section = slice(axis, policy.mrope_section[axis] * 3, 3)
        angles[..., section] = positions[..., axis, None].float() * inv[section]
    while angles.ndim < xn.ndim:
        angles = angles.unsqueeze(-2)
    cos, sin = angles.cos() * policy.attention_factor, angles.sin() * policy.attention_factor
    result = xn.clone()
    result[..., :half] = xn[..., :half] * cos - xn[..., half:2 * half] * sin
    result[..., half:2 * half] = xn[..., half:2 * half] * cos + xn[..., :half] * sin
    return result.bfloat16()


@pytest.mark.parametrize("start,rows", [(0, 4), (262143, 1), (262144, 4), (524284, 4)])
@pytest.mark.parametrize("image_axes", [False, True])
def test_yarn_queries_keys_and_pooled_indexer_match_independent_rotation(start, rows, image_axes):
    policy = _policy()
    inv = policy.inverse_frequencies(torch).cuda()
    generator = torch.Generator(device="cuda").manual_seed(451)
    def random(shape):
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    heads, kv, dim, ih, index_dim = 2, 1, 256, 1, 128
    width = 2 * heads * dim + 2 * kv * dim + (ih + 1) * index_dim
    count, first_block = start + rows, start // 4 * 4
    projection = random((rows, width))
    qscale, kscale, iscale, pscale = [1 + random((d,)).float() * 0.05 for d in (dim, dim, index_dim, index_dim)]
    pos = torch.tensor([start], dtype=torch.int32, device="cuda")
    q = torch.empty((rows, heads, dim), device="cuda", dtype=torch.bfloat16)
    iq = torch.empty((rows, ih, index_dim), device="cuda", dtype=torch.bfloat16)
    kc = torch.empty((count, kv, dim), device="cuda", dtype=torch.bfloat16)
    vc = torch.empty_like(kc)
    index = torch.empty((count, index_dim), device="cuda", dtype=torch.bfloat16)
    index[first_block:] = random((count - first_block, index_dim))
    pooled = torch.empty(((count + 3) // 4, index_dim), device="cuda", dtype=torch.bfloat16)
    positions = torch.empty((count, 3), device="cuda", dtype=torch.int32)
    text = torch.arange(first_block, count, device="cuda", dtype=torch.int32)
    local_positions = torch.stack([text, text // 2, text // 3] if image_axes else [text] * 3, dim=-1)
    positions[first_block:] = local_positions
    kwargs = {"rope": positions, "length": count,
              "delta": torch.zeros((1,), dtype=torch.int32, device="cuda")} if image_axes else {}
    glue.attn_prep(projection, pos, qscale, kscale, iscale, inv, q, kc, vc, iq, index, 1e-6,
                   q_heads=heads, kv_heads=kv, head_dim=dim, index_heads=ih, index_dim=index_dim,
                   rope_scale=policy.attention_factor, sections=policy.mrope_section, **kwargs)
    attention.qsa_pool(index, pooled, pos, pscale, inv, 1e-6, SimpleNamespace(ratio=4), rows,
                       rope_scale=policy.attention_factor, sections=policy.mrope_section, **kwargs)
    qraw = projection[:, :2 * heads * dim].reshape(rows, heads, 2 * dim)[..., :dim]
    kraw = projection[:, 2 * heads * dim:2 * heads * dim + kv * dim].reshape(rows, kv, dim)
    iraw = projection[:, 2 * heads * dim + 2 * kv * dim:][:, :ih * index_dim].reshape(rows, ih, index_dim)
    axes = positions[start:count]
    for actual, source, scale in [(q, qraw, qscale), (kc[start:], kraw, kscale), (iq, iraw, iscale)]:
        expected = _rotate_reference(source, scale, axes, inv, policy)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.02, rtol=0.01)
    for block in range(start // 4, count // 4):
        values = index[block * 4:block * 4 + 4].float()
        summed = values[0]
        for row in values[1:]:
            summed = summed + row
        averaged = (summed / 4).bfloat16()
        expected = _rotate_reference(averaged, pscale, positions[block * 4], inv, policy)
        torch.testing.assert_close(pooled[block].float(), expected.float(), atol=0.02, rtol=0.01)
    # The same generated data and frequencies with amplitude one must retain
    # the unrotated tail bits: amplitude scaling belongs to rotary dimensions.
    q_tail, iq_tail = q[..., 64:].clone(), iq[..., 64:].clone()
    glue.attn_prep(projection, pos, qscale, kscale, iscale, inv, q, kc, vc, iq, index, 1e-6,
                   q_heads=heads, kv_heads=kv, head_dim=dim, index_heads=ih, index_dim=index_dim,
                   rope_scale=1.0, sections=policy.mrope_section, **kwargs)
    assert torch.equal(q[..., 64:], q_tail) and torch.equal(iq[..., 64:], iq_tail)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
def test_yarn_mtp_graphs_and_concurrent_image_attention_match_serial(kv_dtype):
    w = _model()
    policy = _policy()
    w.cfg = replace(w.cfg, rope=policy, index_budget=8)
    w.inv_freq = policy.inverse_frequencies(torch).cuda()
    prompt = [5, 17, 17, 17, 17] + list(range(20, 56))
    image = _image(w, prompt, 41)
    sampling = Sampling(seed=59, top_k=20, top_p=0.95)
    engine = Engine(w, capacity=128, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
    references = []
    for visual in (None, image):
        first = prefill(engine, prompt, sampling, vision=visual)
        references.append(serial_decode(engine, first, 12, sampling).tokens)
        first = prefill(engine, prompt, sampling, vision=visual)
        assert mtp_decode(engine, first, 12, sampling, depth=3, confidence=0.0).tokens == references[-1]
    # Production image serving uses concurrent eager forwards. Exercise graphs
    # on the supported text route with the eager result as an independent oracle.
    graphs = Engine(w, capacity=128, max_rows=8, prefill_rows=16, graphs=True, kv_dtype=kv_dtype)
    first = prefill(graphs, prompt, sampling)
    assert serial_decode(graphs, first, 12, sampling).tokens == references[0]
    first = prefill(graphs, prompt, sampling)
    assert mtp_decode(graphs, first, 12, sampling, depth=3, confidence=0.0).tokens == references[0]
    tower = SimpleNamespace(encode=lambda prepared, ids: prepared)
    dec = MultiDecoder(w, slots=2, capacity=128, depth=3, confidence=0.0, kv_dtype=kv_dtype,
                       vision=tower, prefill_rows=16)
    streams = [Stream(prompt, 12, sampling, stop_eos=False, vision=visual) for visual in (None, image)]
    for stream in streams:
        dec.admit(stream)
    for _ in range(64):
        if not dec.live():
            break
        dec.finish(dec.round())
        assert all(stream.error is None for stream in streams)
    assert not dec.live()
    assert [stream.out for stream in streams] == references
