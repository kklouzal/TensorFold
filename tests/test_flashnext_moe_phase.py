"""Prompt expert arithmetic is selected by phase, including short cached suffixes."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch


@pytest.mark.parametrize("kernel", ["gguf", "sm70_dense"])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("rows", [1, 18, 25])
def test_custom_expert_prefill_dispatch_uses_phase_not_row_count_or_output_policy(monkeypatch, kernel, prefill, optional, rows):
    from tensorfold.families.qwen4_exp.cuda import forward as module

    calls = []
    routing_policy = []

    class Experts:
        def forward(self, x, picks, out=None):
            calls.append("decode")
            out.fill_(2)
            return out

    def prompt(x, picks, out=None):
        calls.append("prefill")
        out.fill_(7)
        return out

    experts = Experts()
    experts.kernel = kernel
    if optional:
        experts.forward_prefill = prompt
    buf = SimpleNamespace(logits=torch.zeros(rows, 3), pick=torch.zeros(rows, 2, dtype=torch.int32),
                          y=torch.empty(rows, 2, 4), wts=torch.ones(rows, 2), plan=SimpleNamespace(prefill=True))
    b = SimpleNamespace(prefill=prefill, mixed=torch.ones(rows, 4, dtype=torch.float16), moe=buf)
    w = SimpleNamespace(comm=None, cfg=SimpleNamespace(top_k=1, experts=2))
    layer = SimpleNamespace(moe=SimpleNamespace(experts=experts, router=object()))
    def router(*args, independent_rows=False):
        routing_policy.append(independent_rows)

    monkeypatch.setattr(module.moe_mod, "router", router)
    monkeypatch.setattr(module.moe_mod, "select_rows", lambda *args: None)
    mode, output, weights = module.moe_block(layer, w, b, rows)
    expected = "prefill" if prefill and optional else "decode"
    assert calls == [expected]
    assert routing_policy == [not prefill]
    assert mode == 2 and output.data_ptr() == buf.y.data_ptr() and weights.data_ptr() == buf.wts.data_ptr()
    torch.testing.assert_close(output, torch.full_like(output, 7 if expected == "prefill" else 2), atol=0, rtol=0)
