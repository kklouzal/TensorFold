"""Private native integration/storage contracts, not CUDA arithmetic evidence."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.owned_experts import Plan, POLICY  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State  # noqa: E402


def policy(*, mtp=True, rows=32, prefill=64):
    return Plan(ExpertPartition(8, 0, 512, 640, 4), 256, 11, rows, prefill, mtp)


def family(monkeypatch, *, mtp=True):
    from test_dspark_taps_forward import cpu_weights
    w = cpu_weights(monkeypatch)
    c = w.cfg
    c.experts, c.top_k, c.moe_width, c.shared_width = 512, 10, 160, 80
    p = ExpertPartition(8, 0, 512, 640, 4)
    w.meta.update(world=8, rank=0, full=NS(kv_heads=2, moe_width=640), expert_partition=p.identity())
    for layer in w.layers:
        layer.moe.experts = GGUFExperts(NS(n=p.count * 160), NS(n=p.count * 160), NS(n=p.count * c.hidden),
            (NS(n=80), NS(n=80), NS(k=80)), count=512, width=160, dims=c.hidden, partition=p)
    if mtp:
        # Real bank/type and Buffers, a component fixture rather than canonical MTP import.
        import copy
        layer = copy.copy(w.layers[-1])
        layer.moe = copy.copy(layer.moe)
        layer.moe.experts = GGUFExperts(NS(n=p.count * 160), NS(n=p.count * 160), NS(n=p.count * c.hidden),
            (NS(n=80), NS(n=80), NS(k=80)), count=512, width=160, dims=c.hidden, partition=p)
        w.mtp = NS(layer=layer)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: NS(multi_processor_count=80))
    return w


@pytest.mark.parametrize("mtp", [False, True])
def test_actual_three_buffer_storage_uses_existing_activation_and_slot_dtype(monkeypatch, mtp):
    w = family(monkeypatch, mtp=mtp)
    baseline = Buffers(w, 3, 16, moe_prefill=True)
    assert baseline.owned_experts is None
    plan = policy(mtp=mtp)
    plan.bind(w)
    buf = Buffers(w, 3, 16, moe_prefill=True)
    pbuf = Buffers(w, 64, 16, prefill=True)
    mbuf = Buffers(w, 3, 16, owner_role="mtp") if mtp else None
    dec = NS(buf=buf, pbuf=pbuf, mbuf=mbuf)
    receipt = plan.reconcile(dec)
    assert receipt["additional_device_bytes"] == 0
    assert receipt["local_arena_bytes"] == 8 * 11 * ((1 + int(mtp)) * 3 + 64)
    assert receipt["materialized_local_arena_bytes"] <= receipt["contained_routing_reserve_bytes"]
    for b in [buf, pbuf] + ([mbuf] if mtp else []):
        a = b.owned_experts
        assert a.activation is b.moe.act and a.output is b.moe.y
        assert a.local.dtype == torch.int64 and a.local.shape == (b.rows, 11)
        assert a.activation.view(-1, 160).data_ptr() == b.moe.act.data_ptr()
    assert buf.moe.y.dtype == pbuf.moe.y.dtype == torch.float16
    if mtp:
        assert mbuf.moe.y.dtype == torch.float32
    # Slots and serial twins share Buffers, adding no new native tensor owners.
    from tensorfold.families.qwen4_exp.cuda.decode import Engine
    engine = NS(w=w, capacity=16, rows=3, prefill_rows=64, buf=buf, mbuf=mbuf, pbuf=pbuf,
                kv_pair=None, _sample_plan=None, _draft_plan=None, _draft_map=None,
                logical_context=16, st=State(w, 16, 3, "fp16", limit=16))
    engine.kv_pair = engine.st.kv_pair
    twin = Engine.twin(engine)
    assert twin.buf is buf and twin.pbuf is pbuf and twin.mbuf is None
    assert plan.reconcile(dec) == receipt
    with pytest.raises(FrozenInstanceError):
        plan.rows = 64


@pytest.mark.parametrize("rows,prefill", [(5958, 1), (1, 5958), (True, 4), (1, 0)])
def test_pair_cap_and_exact_preparation_before_allocations(rows, prefill):
    with pytest.raises(ValueError):
        policy(rows=rows, prefill=prefill)
    assert policy(rows=5957, prefill=5957).resources(5957)["additional_device_bytes"] == 0


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("rows,prefill", [(1, 1), (32, 2048), (5957, 1), (1, 5957)])
def test_all_simultaneous_local_arena_residency_is_contained_without_new_charge(mtp, rows, prefill):
    p = policy(mtp=mtp, rows=rows, prefill=prefill)
    r = p.resources(rows)
    assert r["local_arena_bytes"] == 8 * 11 * ((1 + int(mtp)) * rows + prefill)
    assert r["contained_routing_reserve_bytes"] == (1 + int(mtp)) * 52 * 11 * max(rows, prefill)
    assert r["local_arena_bytes"] <= r["contained_routing_reserve_bytes"]


def test_binding_rejects_method_mismatch_before_mutating_any_bank(monkeypatch):
    w = family(monkeypatch, mtp=True)
    bad = policy(mtp=False)
    with pytest.raises(ValueError, match="target/method geometry"):
        bad.bind(w)
    assert "owned_expert_plan" not in w.meta
    assert all(layer.moe.experts.implementation == "reference" for layer in w.layers)
    plan = policy()
    w.mtp.layer.moe.experts.partition = ExpertPartition(8, 4, 512, 640, 4)
    with pytest.raises(ValueError, match="matching resident"):
        plan.bind(w)
    assert all(layer.moe.experts.implementation == "reference" for layer in w.layers)


def test_exclusive_native_cannot_reach_reference_compaction(monkeypatch):
    w = family(monkeypatch)
    plan = policy()
    plan.bind(w)
    for bank in [*(layer.moe.experts for layer in w.layers), w.mtp.layer.moe.experts]:
        assert bank.implementation == POLICY
        for call in (bank.forward, bank.forward_prefill, bank.forward_reference, bank.forward_partitioned_reference):
            with pytest.raises(ValueError, match="reference"):
                call(None, None)


def test_actual_moe_block_dispatches_target_method_and_prompt_through_owned_arenas(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import forward
    w = family(monkeypatch)
    plan = policy()
    plan.bind(w)
    buf = Buffers(w, 3, 16, moe_prefill=True)
    mbuf = Buffers(w, 3, 16, owner_role="mtp")
    pbuf = Buffers(w, 64, 16, prefill=True)
    calls = []
    monkeypatch.setattr(forward.moe_mod, "router", lambda *a, **kw: None)
    def select(logits, moe, top_k, experts):
        assert (top_k, experts) == (10, 512)
        moe.pick[:logits.shape[0]].copy_(torch.tensor([0, 1, 2, 3, 254, 255, 256, 257, 510, 511, 512]))
    monkeypatch.setattr(forward.moe_mod, "select_rows", select)
    def native(bank, x, picks, *, local, activation, out):
        calls.append((bank, local.data_ptr(), activation.data_ptr(), out.dtype, x.shape[0]))
        out.fill_(.5)
        return out
    monkeypatch.setattr(GGUFExperts, "forward_partitioned_native_candidate", native)
    for layer, b in [(w.layers[0], buf), (w.mtp.layer, mbuf), (w.layers[0], pbuf), (w.mtp.layer, pbuf)]:
        mode, out, _ = forward.moe_block(layer, w, b, 2)
        assert mode == 2 and out.data_ptr() == b.moe.y.data_ptr()
        assert calls[-1][1] == b.owned_experts.local.data_ptr()
        assert calls[-1][2] == b.moe.act.data_ptr()
    assert [call[3] for call in calls] == [torch.float16, torch.float32, torch.float16, torch.float16]
    assert calls[2][1] == calls[3][1]  # pbuf's MTP absorption serially reuses prompt scratch.
    buf.owned_experts = None
    with pytest.raises(ValueError, match="admitted arena"):
        forward.moe_block(w.layers[0], w, buf, 2)


def test_default_state_identity_unchanged_and_native_prefix_restore_are_isolated(monkeypatch):
    w = family(monkeypatch)
    reference = State(w, 16, 3, "fp16", limit=16)
    default = reference.kv_identity
    plan = policy()
    plan.bind(w)
    native = State(w, 16, 3, "fp16", limit=16)
    assert native.kv_identity == default + (("owned-expert-native", plan.identity),)
    native.restore(native.snapshot())
    reference.restore(reference.snapshot())
    for a, b in [(native, reference), (reference, native)]:
        with pytest.raises(ValueError):
            a.restore(b.snapshot())
        with pytest.raises(ValueError):
            a.copy_prefix(b, 0, 0)


@pytest.mark.parametrize("value", [1, None, "true"])
def test_private_optin_is_literal_boolean_before_initialization(value, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    monkeypatch.setattr(FlashNextEngine, "_initialize", lambda *a, **kw: pytest.fail("initialized invalid private policy"))
    with pytest.raises(ValueError, match="literal boolean"):
        FlashNextEngine("unused", _owned_expert_native=value)


def startup_policy_probe():
    """Execute the actual contiguous common-mode/private-check statements only.

    As in maintained host settings tests, isolate source without CUDA startup;
    this is a scheduling/refusal test, never a native arithmetic substitute.
    """
    import ast
    import copy
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_initialize")
    start = next(i for i, node in enumerate(method.body) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "sm70 and gather is not None")
    probe = ast.parse("def probe(self, gather, torch): pass").body[0]
    probe.body = copy.deepcopy(method.body[start:start + 4])
    namespace = dict(sm70=True, indexer_history="gpu", adaptive_sessions=True, session_ceiling=None,
                     execution_rows=None, source_is_gguf=True, tp=8, expert_tensor_parallel_size=4,
                     dspark_model_dir=None, dflash2_model_dir=None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["probe"]


@pytest.mark.parametrize("enabled", [False, True])
def test_mixed_private_flags_with_admin_root_all_join_and_refuse_before_private_checks(enabled):
    probe = startup_policy_probe()
    obj = NS(_owned_expert_requested=enabled, _owned_expert_admin_requested=True,
             expert_partition=ExpertPartition(8, 0, 512, 640, 4))
    calls = []
    def gather(values):
        calls.append(values)
        peer = values.copy()
        peer[-2], peer[-1] = int(not enabled), int(not enabled)
        return [values, peer]
    # No CUDA runtime object: reaching a capability check before agreement fails.
    with pytest.raises(RuntimeError, match="different history/session"):
        probe(obj, gather, None)
    assert len(calls) == 1


def test_uniform_private_admin_refusal_follows_mode_agreement_before_capability_or_payload():
    probe = startup_policy_probe()
    obj = NS(_owned_expert_requested=True, _owned_expert_admin_requested=True,
             expert_partition=ExpertPartition(8, 0, 512, 640, 4))
    calls = []
    def gather(values):
        calls.append(values)
        return [values] * 8
    with pytest.raises(ValueError, match="administrator profile transitions"):
        probe(obj, gather, None)
    assert len(calls) == 1


@pytest.mark.parametrize("other", [None, 16])
def test_rank_settings_reject_native_reference_or_prepared_envelope_mismatch(other):
    from test_flash_next_rank_settings_host import engine, same_settings, torch as host_torch
    a, b = engine(None), engine(None)
    a._owned_expert_plan = policy(rows=32)
    if other is not None:
        b._owned_expert_plan = policy(rows=other)
    same_settings(a, host_torch, None)
    same_settings(b, host_torch, None)
    peers = [a.comm.sent, b.comm.sent]
    for item in (a, b):
        item.comm.peers = peers
        with pytest.raises(RuntimeError, match="different settings"):
            same_settings(item, host_torch, None)


@pytest.mark.parametrize("capability,arch", [((8, 0), ["sm_80"]), ((7, 0), ["sm_80"])])
def test_native_hardware_refuses_after_agreement_before_any_payload(capability, arch):
    probe = startup_policy_probe()
    obj = NS(_owned_expert_requested=True, _owned_expert_admin_requested=False,
             expert_partition=ExpertPartition(8, 0, 512, 640, 4))
    calls = []
    def gather(values):
        calls.append(values)
        return [values] * 8
    runtime = NS(cuda=NS(get_device_capability=lambda: capability, get_arch_list=lambda: arch))
    with pytest.raises(ValueError, match="actual SM70"):
        probe(obj, gather, runtime)
    assert len(calls) == 1


def test_actual_arena_rejects_wrong_activation_extent_and_materialization_alias(monkeypatch):
    w = family(monkeypatch)
    plan = policy()
    plan.bind(w)
    buf = Buffers(w, 3, 16, moe_prefill=True)
    mbuf = Buffers(w, 3, 16, owner_role="mtp")
    pbuf = Buffers(w, 64, 16, prefill=True)
    buf.moe.act = torch.empty((3, 11, 80), dtype=torch.float16)
    with pytest.raises(ValueError, match="reuse the actual"):
        plan.arena(buf)
    # Both simultaneous decoded roles need separate local owners.
    mbuf.owned_experts = buf.owned_experts
    with pytest.raises(MemoryError, match="materialized owned"):
        plan.reconcile(NS(buf=buf, mbuf=mbuf, pbuf=pbuf))


def test_private_policy_is_not_an_administrator_json_option():
    import inspect
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    assert "_owned_expert_native" not in inspect.signature(FlashNextEngine._initialize).parameters
