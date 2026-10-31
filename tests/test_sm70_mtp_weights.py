"""Stock MTP weight mapping, bounded expert execution and precision contracts."""

from dataclasses import fields, is_dataclass, replace
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.qwen4_exp.cuda.sm70_mtp_weights import Dense, DenseExperts, attach, byte_plan  # noqa: E402


def stock(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(35)
    tensors = {}

    def add(name, shape):
        tensors[name] = (torch.randn(shape) * 0.1).to(torch.bfloat16)

    def hc(name, inject):
        add(name + ".input_mix_weight_down.weight", (3, 16))
        add(name + ".input_mix_weight_up.weight", (16, 3))
        add(name + ".hc_norm.weight", (16,))
        if inject:
            add(name + ".block_inject_weight.weight", (2, 16))

    hc("mtp.hyper_connection_mixer", False)
    hc("mtp.layers.0.attn_hyper_connection", True)
    hc("mtp.layers.0.mlp_hyper_connection", True)
    base = "mtp.layers.0."
    for name, shape in {"mlp.experts.gate_up_proj": (2, 16, 8), "mlp.experts.down_proj": (2, 8, 8),
                        "mlp.gate.weight": (2, 8), "mlp.shared_expert_gate.weight": (1, 8),
                        "mlp.shared_expert.gate_proj.weight": (8, 8),
                        "mlp.shared_expert.up_proj.weight": (8, 8),
                        "mlp.shared_expert.down_proj.weight": (8, 8),
                        "self_attn.q_proj.weight": (64, 8), "self_attn.k_proj.weight": (8, 8),
                        "self_attn.v_proj.weight": (8, 8), "self_attn.o_proj.weight": (8, 32),
                        "self_attn.q_norm.weight": (4,), "self_attn.k_norm.weight": (4,),
                        "self_attn.indexer.index_qk_proj.weight": (12, 8),
                        "self_attn.indexer.q_layernorm.weight": (4,),
                        "self_attn.indexer.k_layernorm.weight": (4,)}.items():
        add(base + name, shape)
    add("mtp.fc_embedding.weight", (8, 8))
    add("mtp.fc_hidden.weight", (8, 8))
    add("mtp.pre_fc_norm_embedding.weight", (8,))
    add("mtp.pre_fc_norm_hidden.weight", (16,))
    config = {"model_type": "qwen4_exp", "text_config": {
        "hidden_size": 8, "head_dim": 4, "num_attention_heads": 8, "num_key_value_heads": 2,
        "num_experts": 2, "hc_count": 2, "hc_lowrank": 3, "moe_intermediate_size": 8,
        "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    target = SimpleNamespace(cfg=SimpleNamespace(hidden=8, layers=2, heads=2, kv_heads=1, head_dim=4,
                             experts=2, streams=2, low=3, moe_width=2, index_heads=2, index_dim=4),
                             device=torch.device("cpu"), meta={})
    return target, tensors


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if hasattr(value, "device_tensors"):
        return sum(tensor_bytes(t) for t in value.device_tensors())
    if is_dataclass(value):
        return sum(tensor_bytes(getattr(value, f.name)) for f in fields(value))
    return 0


def test_stock_mtp_uses_explicit_centered_norms_and_replicated_kv(tmp_path):
    target, original = stock(tmp_path)
    plan = byte_plan(target, tmp_path, tp=(3, 4))
    attach(target, tmp_path, tp=(3, 4))
    assert tensor_bytes(target.mtp) == plan["resident_bytes"]
    assert target.mtp.layer.index == 2 and not target.mtp.layer.linear
    torch.testing.assert_close(target.mtp.norm_e, original["mtp.pre_fc_norm_embedding.weight"].float() + 1,
                               atol=0, rtol=0)
    source = original["mtp.layers.0.self_attn.k_proj.weight"].half()[4:8]
    # Rank3 Q rows=16 followed by the replicated second global KV head.
    torch.testing.assert_close(target.mtp.layer.attn.proj.weight[16:20], source, atol=0, rtol=0)
    assert target.draft_head is None and target.draft_ids is None
    assert target.meta["mtp_storage"] == "stock-bf16-execution-fp16"


def test_stock_mtp_rejects_incompatible_target_before_loading(tmp_path):
    target, _ = stock(tmp_path)
    target.cfg.hidden = 16
    with pytest.raises(ValueError, match="hidden_size"):
        attach(target, tmp_path, tp=(3, 4))


@pytest.mark.parametrize("change", [None, "theta", "reference", "layout"])
def test_prepared_stock_mtp_validates_its_own_position_source(tmp_path, monkeypatch, change):
    from tensorfold.families.qwen4_exp.rope import RopeParameters

    target, _ = stock(tmp_path)
    path = tmp_path / "config.json"
    cfg = json.loads(path.read_text())
    cfg["text_config"].update(max_position_embeddings=262144, rope_parameters={
        "rope_theta": 10000000., "partial_rotary_factor": .5, "mrope_section": [1, 0, 0]})
    path.write_text(json.dumps(cfg))
    source = own = RopeParameters.from_config(cfg)
    target.cfg.rope_theta = 123456.  # selected target policy cannot override this source check
    if change == "theta":
        own = replace(own, theta=123456.)
    elif change == "reference":
        own = replace(own, native_context=524288)
    elif change == "layout":
        own = replace(own, mrope_section=(0, 1, 0))
    target.meta["position_profile"] = SimpleNamespace(target=SimpleNamespace(rope=source), mtp=SimpleNamespace(rope=own))
    monkeypatch.setattr(torch, "empty", lambda *_a, **_kw: pytest.fail("position check allocated payload"))
    if change is None:
        assert byte_plan(target, tmp_path, tp=(3, 4))["resident_bytes"] > 0
    else:
        with pytest.raises(ValueError, match="positional source"):
            byte_plan(target, tmp_path, tp=(3, 4))


def test_prepared_legacy_mtp_omitted_fields_use_source_native_reference_only(tmp_path):
    from tensorfold.families.qwen4_exp.rope import RopeParameters

    target, _ = stock(tmp_path)  # actual legacy companion omits context and positional parameters
    native = RopeParameters("default", 1000000., 2, 262144, 262144, mrope_section=(1, 0, 0))
    extended = replace(native, rope_type="yarn", factor=4., attention_factor=1.25, context_limit=1048576)
    target.meta["position_profile"] = SimpleNamespace(target=SimpleNamespace(rope=extended),
        mtp=SimpleNamespace(rope=native))
    assert byte_plan(target, tmp_path, tp=(3, 4))["resident_bytes"] > 0


def test_dense_storage_and_row_grouping_preserve_same_values():
    torch.manual_seed(63)
    weight = torch.randn((17, 80), dtype=torch.float16)
    x = torch.randn((5, 80), dtype=torch.float16)
    linear = Dense(weight)
    batched = linear.matmul(x, f32=True)
    singles = torch.cat([linear.matmul(row[None], f32=True) for row in x])
    torch.testing.assert_close(batched, singles, atol=0, rtol=0)
    torch.testing.assert_close(batched.double(), x.double() @ weight.double().T, atol=1e-4, rtol=2e-5)


def test_dense_expert_tiles_match_independent_per_slot_policy():
    torch.manual_seed(914)
    gate, up = [torch.randn((4, 80, 128), dtype=torch.float16) * 0.1 for _ in range(2)]
    down = torch.randn((4, 128, 80), dtype=torch.float16) * 0.1
    x = torch.randn((3, 128), dtype=torch.float16) * 0.1
    picks = torch.tensor([[0, 3, 1], [2, 0, 3], [1, 1, 2]], dtype=torch.int32)
    bank = DenseExperts(gate, up, down, tile_pairs=4)
    result = bank.forward(x, picks)
    expected = torch.empty_like(result)
    for row in range(3):
        for slot in range(3):
            expert = int(picks[row, slot])
            g = (gate[expert].double() @ x[row].double()).half()
            u = (up[expert].double() @ x[row].double()).half()
            act = torch.nn.functional.silu(g) * u
            expected[row, slot] = down[expert].double() @ act.double()
    torch.testing.assert_close(result, expected, atol=2e-5, rtol=0.002)
