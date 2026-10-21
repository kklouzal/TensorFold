"""Exact Gemma CT BF16 softcap dispatch, exhaustive values and graph ownership."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.gemma4.cuda import glue  # noqa: E402
from tensorfold.families.gemma4.cuda.forward import Gemma4Target  # noqa: E402
from tensorfold.families.gemma4.cuda.weights import Config  # noqa: E402


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="softcap runs CUDA kernels")


def _config(cap):
    return Config(hidden=128, layers=2, vocab=128, heads=16, kv_heads=8, global_kv_heads=2,
                  head_dim=256, global_head_dim=512, intermediate=128, moe_width=64, experts=4, top_k=2,
                  window=8, max_position=4096, eps=1e-6, softcap=cap,
                  layer_types=("sliding_attention", "full_attention"), rope_theta=10000.,
                  global_rope_theta=1000000., global_rotary_fraction=.25, vision_bidirectional=True)


@pytest.mark.parametrize("policy,cap,fused", [("ct_nvfp4", 30.0, True), ("ct_nvfp4", 7.75, False),
                                             ("mlx", 30.0, False)])
def test_preparation_binds_only_ct_cap30_and_caps_borrowed_head_once(monkeypatch, policy, cap, fused):
    # A head boundary adapter proves dispatch/ownership without emulating its CUDA dot product.
    logits = torch.tensor([[0.0, -0.0, -100.0, 1.234375, 30.0, 100.0]], dtype=torch.bfloat16)
    expected = torch.tanh(logits / cap) * cap
    calls, head_calls = [], []

    def selected(value):
        calls.append(value)
        return expected.clone()

    class Head:
        # Preparation consumes the already-loaded weight descriptor. This
        # adapter does not claim a CUDA allocation or reproduce head math.
        weight = SimpleNamespace(dtype=torch.bfloat16, shape=(262144, 2816), device=torch.device("cuda", 0),
                                 is_contiguous=lambda: True)

        def __call__(self, hidden):
            head_calls.append(hidden)
            return logits

        def embedding(self, ids):
            return self.weight[ids]

    monkeypatch.setattr(glue, "softcap30", selected)
    capabilities = []
    monkeypatch.setattr(torch.cuda, "get_device_capability",
                        lambda device: capabilities.append(device) or (12, 1))
    target = Gemma4Target(SimpleNamespace(config=_config(cap), numerics=policy, embed=Head(),
                                         norm=torch.ones(128, dtype=torch.bfloat16)))

    def changed_binding(value):
        pytest.fail("stable softcap policy was looked up again after preparation")

    monkeypatch.setattr(glue, "softcap30", changed_binding)
    hidden = torch.zeros((1, 128), dtype=torch.bfloat16)
    # BF16 DFlash borrows this bound method; there is no second cap in the borrower.
    borrower = SimpleNamespace(project_logits=target.project_logits)
    actual = borrower.project_logits(hidden)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert len(head_calls) == 1 and head_calls[0] is hidden
    assert len(calls) == int(fused)
    assert len(capabilities) == int(fused)
    if fused:
        assert calls[0] is logits


@pytest.mark.parametrize("dtype,shape,device,contiguous,capability", [
    (torch.float32, (262144, 2816), "cuda:0", True, (12, 1)),
    (torch.bfloat16, (128, 128), "cuda:0", True, (12, 1)),
    (torch.bfloat16, (262144, 2816), "cpu", True, (12, 1)),
    (torch.bfloat16, (262144, 2816), "cuda:0", False, (12, 1)),
    (torch.bfloat16, (262144, 2816), "cuda:0", True, (12, 0)),
])
def test_unqualified_head_regions_retain_original_operation(monkeypatch, dtype, shape, device, contiguous, capability):
    from tensorfold.families.gemma4.cuda.head import original_bf16_region

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    descriptor = SimpleNamespace(dtype=dtype, shape=shape, device=torch.device(device),
                                 is_contiguous=lambda: contiguous)
    assert not original_bf16_region(descriptor)


def _all_bf16(device):
    # Every raw representation exactly once, including signed zeros, all
    # subnormals, infinities, and both signs/payloads of NaNs.
    return torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(device)


def _same_bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    a, b = actual.view(torch.int16), expected.view(torch.int16)
    mismatches = (a != b).flatten().nonzero().flatten()
    assert mismatches.numel() == 0, (mismatches[:16].tolist(), a.flatten()[mismatches[:16]].tolist(),
                                   b.flatten()[mismatches[:16]].tolist())


@cuda
def test_softcap_matches_actual_torch_for_all_65536_bf16_bits_and_keeps_input():
    x = _all_bf16("cuda").reshape(256, 256)
    before = x.clone()
    expected = torch.tanh(x / 30.0) * 30.0
    actual = glue.softcap30(x)
    _same_bits(actual, expected)
    _same_bits(x, before)
    assert actual.data_ptr() != x.data_ptr()
    # The exhaustive oracle rejects the tempting FP32-only fusion that drops
    # the two required BF16 intermediate boundaries.
    one_round = (torch.tanh(x.float() / 30.0) * 30.0).to(torch.bfloat16)
    assert torch.any((actual.view(torch.int16) != one_round.view(torch.int16)) & torch.isfinite(x))


@cuda
@pytest.mark.parametrize("rows,columns", [(1, 1), (1, 1023), (1, 1025), (1, 262144), (4, 262144), (5, 262144)])
def test_softcap_masked_tail_and_actual_vocabulary_row_partitions(rows, columns):
    all_values = _all_bf16("cuda")
    x = all_values.repeat(-(-(rows * columns) // all_values.numel()))[:rows * columns].reshape(rows, columns)
    expected = torch.tanh(x / 30.0) * 30.0
    whole = glue.softcap30(x)
    _same_bits(whole, expected)
    parts = torch.cat([glue.softcap30(row[None]) for row in x])
    _same_bits(parts, whole)


@cuda
def test_softcap_graph_replays_new_input_and_owns_its_output():
    source = _all_bf16("cuda").reshape(256, 256)
    glue.softcap30(source)  # compile before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = glue.softcap30(source)
    address = actual.data_ptr()
    for shift in (1, 73, 8191):
        updated = torch.roll(_all_bf16("cuda"), shift).reshape_as(source)
        source.copy_(updated)
        graph.replay()
        _same_bits(actual, torch.tanh(updated / 30.0) * 30.0)
        assert actual.data_ptr() == address and address != source.data_ptr()
