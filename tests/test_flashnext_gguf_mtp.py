"""Shared MTP import contracts, independent decoded fusion, and TP packed shards."""

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.qwen4_exp.cuda import gguf_mtp_weights as mtp  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.sm70_mtp_weights import attach, byte_plan  # noqa: E402
from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint, _shard, decode_rows, token_id_identity  # noqa: E402


def _string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _value(value):
    if type(value) is bool:
        return 7, struct.pack("<?", value)
    if type(value) is float:
        return 6, struct.pack("<f", value)
    if type(value) is str:
        return 8, _string(value)
    if isinstance(value, tuple):
        kind = 8 if value and isinstance(value[0], str) else 4
        data = b"".join(_string(v) if kind == 8 else struct.pack("<I", v) for v in value)
        return 9, struct.pack("<IQ", kind, len(value)) + data
    return 4, struct.pack("<I", value)


def fixture(tmp_path, *, world=8, rank=3, mutate=None):
    full = SimpleNamespace(hidden=64, layers=4, heads=8, kv_heads=2, head_dim=8,
                           experts=3, top_k=2, moe_width=64, shared_width=64, streams=4, low=8,
                           index_heads=2, index_dim=8, index_budget=2048, index_ratio=4,
                           rotary_dim=8, rope_theta=10000000.0, vocab=64, eps=1e-6)
    local = SimpleNamespace(**vars(full))
    local.heads //= world
    local.kv_heads = max(1, full.kv_heads // world)
    local.moe_width //= world
    local.shared_width //= world
    target = SimpleNamespace(cfg=local, meta={"full": full}, device=torch.device("cpu"),
                             embed=object(), head=object(), mtp=None)
    metadata = {"general.architecture": "qwen4exp", "qwen4exp.nextn_shared_target_tensors": True,
                "qwen4exp.nextn_predict_layers": 1, "qwen4exp.block_count": 5,
                "qwen4exp.attention.compress_ratios": (0, 0, 0, 4, 0),
                "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
                "tokenizer.ggml.tokens": tuple(f"t{i}" for i in range(64))}
    mapping = {"embedding_length": "hidden", "attention.head_count": "heads",
               "attention.head_count_kv": "kv_heads", "attention.key_length": "head_dim",
               "attention.value_length": "head_dim", "expert_count": "experts",
               "expert_used_count": "top_k", "expert_feed_forward_length": "moe_width",
               "expert_shared_feed_forward_length": "shared_width", "hyper_connection.count": "streams",
               "hyper_connection.low_rank": "low", "attention.indexer.head_count": "index_heads",
               "attention.indexer.key_length": "index_dim", "attention.indexer.top_k": "index_budget",
               "rope.dimension_count": "rotary_dim", "rope.freq_base": "rope_theta"}
    metadata.update({"qwen4exp." + key: getattr(full, attr) for key, attr in mapping.items()})
    target.meta["gguf_token_id_identity"] = token_id_identity(metadata)
    rng = np.random.default_rng(12)
    entries, values = {}, {}

    def add(name, shape, kind=0):
        if kind == 8:
            blocks = np.empty((int(np.prod(shape)) // 32, 34), dtype=np.uint8)
            blocks[:, :2] = np.array([0.0625], dtype="<f2").view(np.uint8)
            codes = rng.integers(-16, 17, (blocks.shape[0], 32), dtype=np.int8)
            blocks[:, 2:] = codes.view(np.uint8)
            raw = blocks.tobytes()
            value = (codes.astype(np.float32) * 0.0625).reshape(shape)
        else:
            value = (rng.normal(size=shape) * 0.1).astype(np.float32)
            raw = value.astype("<f4").tobytes()
        entries["blk.4." + name] = (shape, kind, raw)
        values[name] = value

    for name, shape in {"attn_q.weight": (128, 64), "attn_k.weight": (16, 64),
                        "attn_v.weight": (16, 64), "attn_output.weight": (64, 64),
                        "indexer.q_proj.weight": (16, 64), "indexer.k_proj.weight": (8, 64),
                        "nextn.eh_proj.weight": (64, 128)}.items():
        add(name, shape, 8)
    for name in ("attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"):
        add(name, (8,))
    for prefix in ("hc_attn_", "hc_ffn_", "nextn.hc_head_"):
        add(prefix + "down.weight", (8, 256), 8)
        add(prefix + "up.weight", (256, 8))
        add(prefix + "norm.weight", (256,))
        if prefix != "nextn.hc_head_":
            add(prefix + "inject.weight", (4, 256), 8)
    for name in ("gate", "up", "down"):
        add(f"ffn_{name}_exps.weight", (3, 64, 64), 8)
        add(f"ffn_{name}_shexp.weight", (64, 64), 8)
    add("ffn_gate_inp.weight", (3, 64))
    add("ffn_gate_inp_shexp.weight", (64,))
    add("nextn.enorm.weight", (64,))
    add("nextn.hnorm.weight", (256,))
    if mutate:
        mutate(metadata, entries)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(entries), len(metadata))
    for key, value in metadata.items():
        kind, raw = _value(value)
        header += _string(key) + struct.pack("<I", kind) + raw
    payload = b""
    for name, (shape, kind, raw) in entries.items():
        header += _string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        header += struct.pack("<IQ", kind, len(payload))
        payload += raw + bytes(-len(raw) % 32)
    file = tmp_path / "companion.gguf"
    file.write_bytes(header + bytes(-len(header) % 32) + payload)
    return target, file, values


def _owned(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if is_dataclass(value):
        return sum(_owned(getattr(value, f.name), seen) for f in fields(value))
    if hasattr(value, "device_tensors"):
        return sum(_owned(t, seen) for t in value.device_tensors())
    return 0


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 1), (4, 2), (8, 3), (8, 7)])
def test_real_import_retains_exact_planned_bytes_and_shared_target(tmp_path, world, rank):
    target, file, source = fixture(tmp_path, world=world, rank=rank)
    embedding, head = target.embed, target.head
    plan = byte_plan(target, file, tp=(rank, world))
    attach(target, file, tp=(rank, world))
    assert _owned(target.mtp) == plan["resident_bytes"]
    assert target.embed is embedding and target.head is head
    assert target.mtp.layer.index == 4 and target.mtp.layer.ple is None
    assert target.meta["mtp_import"]["source_final_compression_ratio"] == 0
    assert target.meta["mtp_import"]["policy"] == mtp.POLICY
    # GGUF norms are already centred. Adding another one would change the model.
    torch.testing.assert_close(target.mtp.norm_h, torch.from_numpy(source["nextn.hnorm.weight"]), atol=0, rtol=0)
    fusion = source["nextn.eh_proj.weight"]
    torch.testing.assert_close(target.mtp.fc_e.weight, torch.from_numpy(fusion[:, :64]).half(), atol=0, rtol=0)
    torch.testing.assert_close(target.mtp.fc_h.weight, torch.from_numpy(fusion[:, 64:]).half(), atol=0, rtol=0)
    bank = target.mtp.layer.moe.experts
    # Decode retained packed shards independently, including TP8's eight-wide
    # column shards crossing partial Q8 blocks. No requantization is permitted.
    for name, packed in (("gate", bank.gate), ("up", bank.up), ("down", bank.down)):
        shape = (packed.n, packed.data.shape[1] // 34 * 32)
        descriptor = replace(GGUFCheckpoint.open(file).tensors[f"blk.4.ffn_{name}_exps.weight"], shape=shape)
        got = decode_rows(descriptor, packed.data.numpy())[:, packed.column_offset:packed.column_offset + packed.k]
        width = 64 // world
        lo = rank * width
        expected = source[f"ffn_{name}_exps.weight"]
        expected = expected[:, :, lo:lo + width] if name == "down" else expected[:, lo:lo + width, :]
        np.testing.assert_array_equal(got, expected.reshape(packed.n, packed.k))


@pytest.mark.parametrize("field,value", [("nextn_shared_target_tensors", False), ("nextn_predict_layers", 2),
                                         ("nextn_predict_layers", True), ("nextn_predict_layers", 1.0),
                                         ("block_count", 5.0), ("attention.head_count", 8.0),
                                         ("hyper_connection.count", 4.0),
                                         ("block_count", 6), ("attention.head_count", 16),
                                         ("attention.indexer.top_k", 1024),
                                         ("attention.compress_ratios", (0, 0, 0, 4, 2))])
def test_incompatible_companion_rejected_before_payload_reads(tmp_path, monkeypatch, field, value):
    target, file, _ = fixture(tmp_path, mutate=lambda meta, _entries: meta.update({"qwen4exp." + field: value}))
    monkeypatch.setattr(mtp, "Reader", lambda: pytest.fail("invalid metadata reached payload reader"))
    with pytest.raises(ValueError):
        attach(target, file, tp=(3, 8))
    assert target.mtp is None


def test_unknown_tensor_is_not_ignored(tmp_path):
    target, file, _ = fixture(tmp_path, mutate=lambda _meta, entries: entries.update({
        "unrecognized.weight": ((1,), 0, bytes(4))}))
    with pytest.raises(ValueError, match="unmapped"):
        byte_plan(target, file, tp=(3, 8))


def test_same_vocabulary_size_does_not_admit_different_token_ids(tmp_path):
    def change(metadata, _entries):
        ids = metadata["tokenizer.ggml.tokens"]
        metadata["tokenizer.ggml.tokens"] = (ids[1], ids[0], *ids[2:])

    target, file, _ = fixture(tmp_path, mutate=change)
    with pytest.raises(ValueError, match="token-ID mapping"):
        byte_plan(target, file, tp=(3, 8))


def test_shared_materializer_preserves_target_leaf_order_hc_aliases_and_router_staging(tmp_path, monkeypatch):
    import weakref
    from tensorfold.cuda.direct_read import Reader
    from tensorfold.families.qwen4_exp.cuda import gguf_weights

    target, file, source = fixture(tmp_path)
    prototype, _, _ = mtp.model_plan(target, file, tp=(3, 8))
    calls, temporary = [], []
    original_array = gguf_weights._array
    router_parts = prototype.layer.moe.router.parts

    def dense(plan, reader, device):
        calls.append(("dense", plan.source.name))
        return mtp._cpu_dense(plan, reader, device)

    def array(plan, reader, device):
        calls.append(("array", plan.source.name))
        result = original_array(plan, reader, device)
        if any(plan is part for part in router_parts):
            temporary.append(weakref.ref(result))
        return result

    monkeypatch.setattr(gguf_weights, "_dense", dense)
    monkeypatch.setattr(gguf_weights, "_array", array)
    reader = Reader()
    try:
        materialize = gguf_weights.materializer(reader, "cpu")
        hc = materialize(prototype.layer.attn_hc)
        assert calls == [("dense", "blk.4.hc_attn_down.weight"),
                         ("dense", "blk.4.hc_attn_inject.weight"),
                         ("dense", "blk.4.hc_attn_up.weight"),
                         ("array", "blk.4.hc_attn_norm.weight")]
        assert hc.down is hc.prefill_down and hc.up is hc.prefill_up
        joined = hc.down.fused.weight
        assert all(part.weight.untyped_storage().data_ptr() == joined.untyped_storage().data_ptr()
                   for part in hc.down.parts)
        expected = np.concatenate([source["hc_attn_down.weight"], source["hc_attn_inject.weight"]]).astype(np.float16)
        np.testing.assert_array_equal(joined.numpy(), expected)
        calls.clear()
        assert materialize(prototype.layer.attn_hc.down) is hc.down
        assert not calls
        router = materialize(prototype.layer.moe.router)
        assert calls == [("array", "blk.4.ffn_gate_inp.weight"), ("array", "blk.4.ffn_gate_inp_shexp.weight")]
        assert len(temporary) == 2 and all(reference() is None for reference in temporary)
        expected = np.concatenate([source["ffn_gate_inp.weight"], source["ffn_gate_inp_shexp.weight"][None]])
        np.testing.assert_array_equal(router.numpy().view(np.uint32), expected.astype(np.float16).astype(np.float32).view(np.uint32))
        calls.clear()
        assert materialize(prototype.layer.moe.router) is router
        assert not calls
    finally:
        reader.close()


def test_shared_materializer_invokes_expert_and_table_owners_once(tmp_path):
    from tensorfold.cuda.direct_read import Reader
    from tensorfold.families.qwen4_exp.cuda import gguf_weights

    target, file, _ = fixture(tmp_path)
    prototype, _, _ = mtp.model_plan(target, file, tp=(3, 8))
    reader, prepared, tables = Reader(), [], []
    table_plan = gguf_weights.TablePlan(object())
    table_owner = object()
    def load_table(plan):
        tables.append(plan)
        return table_owner
    try:
        materialize = gguf_weights.materializer(reader, "cpu", dense_loader=mtp._cpu_dense,
                                                prepare_experts=prepared.append, load_table=load_table)
        bank = materialize(prototype.layer.moe.experts)
        assert materialize(prototype.layer.moe.experts) is bank and prepared == [bank]
        assert materialize(table_plan) is materialize(table_plan) is table_owner
        assert tables == [table_plan]
    finally:
        reader.close()


def test_actual_artifact_plan_when_explicitly_supplied():
    import os

    selected = os.environ.get("TENSORFOLD_TEST_MTP_GGUF")
    target_metadata = os.environ.get("TENSORFOLD_TEST_TARGET_METADATA_GGUF")
    if not selected or not target_metadata:
        pytest.skip("optional pinned companion and actual target metadata not supplied")
    full = SimpleNamespace(hidden=2560, layers=48, heads=24, kv_heads=2, head_dim=256,
                           experts=512, top_k=10, moe_width=640, shared_width=640, streams=4, low=320,
                           index_heads=4, index_dim=128, index_budget=2048, index_ratio=4,
                           rotary_dim=64, rope_theta=10000000.0, vocab=248320, eps=1e-6)
    local = SimpleNamespace(**vars(full))
    local.heads, local.kv_heads, local.moe_width, local.shared_width = 3, 1, 80, 80
    target = SimpleNamespace(cfg=local, meta={"full": full}, device=torch.device("cpu"))
    metadata, tensors = _shard(Path(target_metadata))
    assert not tensors  # The selected first target shard supplies metadata only.
    target.meta["gguf_token_id_identity"] = token_id_identity(metadata)
    plan = byte_plan(target, Path(selected), tp=(0, 8))
    assert plan["import"]["source_tensor_count"] == 32
    assert plan["import"]["source_payload_bytes"] == 2775621632
    assert plan["import"]["source_final_compression_ratio"] == 0
    target.embed, target.head = object(), object()
    embedding, head = target.embed, target.head
    attach(target, Path(selected), tp=(0, 8))
    assert _owned(target.mtp) == plan["resident_bytes"]
    assert target.embed is embedding and target.head is head
    assert torch.isfinite(target.mtp.fc_h.weight).all()
    assert torch.isfinite(target.mtp.fc_e.weight).all()
    print(json.dumps(plan, sort_keys=True))
