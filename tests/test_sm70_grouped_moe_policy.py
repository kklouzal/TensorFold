"""Grouped SIMT selection retains the established startup and direct-call domains."""
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from tensorfold.cuda.sm70 import gguf_ops  # noqa: E402
from tensorfold.cuda import experts  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts  # noqa: E402

policy = family.policy
source = family.source
PROFILE = "source-grouped-moe-trial"


def test_profile_warms_before_publish_and_does_not_grant_other_operators(source, monkeypatch):
    monkeypatch.setenv(source.module.PROFILE_ENV, PROFILE)
    monkeypatch.setattr(gguf_ops, "extension", lambda: source.calls.append("gguf-warm") or
                        NS(prepare_grouped=lambda: source.calls.append("kernel-residency")))
    assert source.module.configure(source.text, tp=8, torch=torch) == PROFILE
    assert source.calls == ["audit", "extension", "gguf-warm", "kernel-residency", "audit"]
    assert family.selectors(source.policy) == (
        True, False, False, False, False, False, "packed_grouped_exact", "native32", None)
    w = family.weights(source)
    source.module.prepare_weights(w)
    assert w.meta["sm70_family_trial"]["workspace"] == "decoder-plan-act-t16"
    assert w.meta["sm70_family_trial"]["experts"] == "packed_grouped_exact"
    assert not any(w.meta.get(name) for name in (
        "capture_ready", "device_greedy_mtp_ready", "device_greedy_native_ready", "attention_qualification"))


@pytest.mark.parametrize("failure", ["extension", "kernel-residency"])
def test_preparation_failure_preserves_previous_selection(source, monkeypatch, failure):
    before = family.selectors(source.policy)
    monkeypatch.setenv(source.module.PROFILE_ENV, PROFILE)
    error = RuntimeError("grouped extension unavailable")

    def fail():
        raise error

    monkeypatch.setattr(gguf_ops, "extension", fail if failure == "extension" else
                        lambda: NS(prepare_grouped=fail))
    with pytest.raises(RuntimeError) as caught:
        source.module.configure(source.text, tp=8, torch=torch)
    assert caught.value is error and family.selectors(source.policy) == before


@pytest.mark.parametrize("old", ["reference", "optimized", "source-family-trial", "source-family-prefix-trial", "source-packed-trial"])
def test_prior_profile_settings_and_state_epochs_stay_distinct(source, monkeypatch, old):
    monkeypatch.setenv(source.module.PROFILE_ENV, old)
    monkeypatch.setattr(gguf_ops, "extension", lambda: None)
    source.module.configure(source.text, tp=8, torch=torch)
    assert source.policy.weights.GGUFExperts.implementation != "packed_grouped_exact"
    assert prefix_policy.identity(old) != prefix_policy.identity(PROFILE)
    assert prefix_policy.state_epoch(old) != prefix_policy.state_epoch(PROFILE)


@pytest.mark.parametrize("rows,route", [(0, "exact"), (1, "exact"), (2, "exact"), (3, "exact"),
                                      (4, "grouped"), (18, "grouped"), (32, "grouped"), (33, "reference")])
def test_actual_row_dispatch_preserves_single_active_stock_mtp_and_large_calls(monkeypatch, rows, route):
    bank = GGUFExperts(None, None, None, (), count=512, width=80, dims=2560)
    bank.implementation = "packed_grouped_exact"
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    x = torch.empty((rows, 2560), dtype=torch.float16)
    picks = torch.zeros((rows, 11), dtype=torch.int32)
    owner, output, calls = object(), torch.empty((rows, 11, 2560)), []
    monkeypatch.setattr(bank, "forward_grouped", lambda a, b, c, d: calls.append(("grouped", c)) or d)
    monkeypatch.setattr(bank, "forward_exact", lambda a, b, c: calls.append(("exact", None)) or c)
    monkeypatch.setattr(bank, "forward_reference", lambda a, b, c: calls.append(("reference", None)) or c)
    assert bank.forward(x, picks, output, workspace=owner) is output
    assert calls == [(route, owner if route == "grouped" else None)]


@pytest.mark.parametrize("case", ["direct", "int64", "width", "count", "input-layout", "output-layout", "cpu"])
def test_required_direct_and_generic_routes_do_not_borrow_a_grouped_workspace(monkeypatch, case):
    bank = GGUFExperts(None, None, None, (), count=512, width=80, dims=2560)
    bank.implementation = "packed_grouped_exact"
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: case != "cpu"))
    x = torch.empty((4, 2560), dtype=torch.float16)
    picks = torch.zeros((4, 11), dtype=torch.int32)
    out = torch.empty((4, 11, 2560))
    if case == "int64":
        picks = picks.long()
    elif case == "width":
        bank.width = 160
    elif case == "count":
        bank.count = 3
    elif case == "input-layout":
        x = torch.empty((4, 5120), dtype=torch.float16)[:, ::2]
    elif case == "output-layout":
        out = torch.empty((4, 11, 5120))[:, :, ::2]
    sentinel = object()
    monkeypatch.setattr(bank, "forward_exact", lambda *a: sentinel)
    monkeypatch.setattr(bank, "forward_grouped", lambda *a: pytest.fail("unsupported grouped route"))
    assert bank.forward(x, picks, out, workspace=None if case == "direct" else object()) is sentinel


@pytest.mark.parametrize("prefill", [False, True])
def test_real_moe_caller_borrows_only_its_owned_decode_arena_and_keeps_slot_output(monkeypatch, prefill):
    from tensorfold.families.qwen4_exp.cuda import forward

    calls = []
    buf = NS(logits=torch.zeros((4, 513)), pick=torch.zeros((4, 11), dtype=torch.int32),
             y=torch.zeros((4, 11, 2560)), wts=torch.zeros((4, 11)))
    bank = NS(implementation="packed_grouped_exact", kernel="gguf")
    bank.forward = lambda x, picks, **kw: calls.append(("decode", kw))
    bank.forward_prefill = lambda x, picks, **kw: calls.append(("prefill", kw))
    layer = NS(moe=NS(experts=bank, router=object()))
    b = NS(prefill=prefill, moe=buf, mixed=torch.empty((4, 2560)))
    w = NS(comm=None, cfg=NS(top_k=10, experts=512))
    monkeypatch.setattr(forward.moe_mod, "router", lambda *a, **kw: None)
    monkeypatch.setattr(forward.moe_mod, "select_rows", lambda *a: None)
    result = forward.moe_block(layer, w, b, 4)
    assert result[0] == 2 and torch.equal(result[1], buf.y) and torch.equal(result[2], buf.wts)
    assert len(calls) == 1 and calls[0][0] == ("prefill" if prefill else "decode")
    assert calls[0][1].keys() == ({"out"} if prefill else {"out", "workspace"})
    if not prefill:
        assert calls[0][1]["workspace"] is buf


@pytest.mark.parametrize("topk,expected", [(10, 62232), (32, 188468)])
def test_existing_per_buffer_plan_and_activation_hold_the_entire_declared_envelope(topk, expected):
    plan = experts.Plan(32, topk + 1, 513, "cpu")
    act = torch.empty((32, topk + 1, 80), dtype=torch.float16)
    arrays = [plan.members, plan.items, plan.counts, plan.rank, plan.hist, act]
    assert sum(x.numel() * x.element_size() for x in arrays) == expected
    assert plan.items.shape[0] >= min(32 * (topk + 1), 513) + 32 * (topk + 1) // 16
    assert len({x.data_ptr() for x in arrays}) == len(arrays)


def test_real_startup_warm_uses_existing_role_arenas_and_all_banks_before_admission(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    calls = []
    def bank(name):
        def run(x, picks, **kwargs):
            assert torch.count_nonzero(x) == 0
            assert torch.equal(picks[0], torch.tensor([*range(10), 512], dtype=torch.int32))
            assert all(torch.equal(row, picks[0]) for row in picks)
            calls.append((name, kwargs["workspace"], kwargs["out"].data_ptr()))
        return NS(implementation="packed_grouped_exact", forward=run)

    def buffer():
        return NS(rows=32, mixed=torch.empty((32, 2560)),
                  moe=NS(pick=torch.empty((32, 11), dtype=torch.int32), y=torch.empty((32, 11, 2560))))
    obj = object.__new__(MultiDecoder)
    obj.buf, obj.mbuf = buffer(), buffer()
    obj.w = NS(meta={"sm70_profile": PROFILE}, device="cpu", cfg=NS(top_k=10, experts=512),
               layers=[NS(moe=NS(experts=bank("layer0"))), NS(moe=NS(experts=bank("layer1")))],
               mtp=NS(layer=NS(moe=NS(experts=bank("mtp")))))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("joined", device)))
    obj._warm_grouped_experts()
    assert calls == [("layer0", obj.buf.moe, obj.buf.moe.y.data_ptr()),
                     ("layer1", obj.buf.moe, obj.buf.moe.y.data_ptr()),
                     ("mtp", obj.mbuf.moe, obj.mbuf.moe.y.data_ptr()), ("joined", "cpu")]
    obj.w.meta["sm70_profile"] = "source-packed-trial"
    obj._warm_grouped_experts()
    assert len(calls) == 4
