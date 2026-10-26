"""Shared MTP import contracts, independent decoded fusion, and TP packed shards."""

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import hashlib
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.qwen4_exp.cuda import gguf_mtp_weights as mtp  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.sm70_mtp_weights import attach, byte_plan  # noqa: E402
from tensorfold.families.qwen4_exp.gguf import (  # noqa: E402
    GGUFCheckpoint, _shard, decode_rows, metadata_config, token_id_identity,
)


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


def fixture(tmp_path, *, world=8, rank=3, mutate=None, vocabulary_copies=False):
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

    def add(name, shape, kind=0, *, prefix="blk.4."):
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
        entries[prefix + name] = (shape, kind, raw)
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
    if vocabulary_copies:
        del metadata["qwen4exp.nextn_shared_target_tensors"]
        metadata["qwen4exp.attention.compress_ratios"] = (0, 0, 0, 4, 4)
        add("token_embd.weight", (64, 64), 8, prefix="")
        add("output.weight", (64, 64), 8, prefix="")
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


def test_full_companion_never_reads_or_materializes_vocabulary_copies(tmp_path, monkeypatch):
    from tensorfold.cuda.direct_read import Reader

    target, file, _ = fixture(tmp_path, vocabulary_copies=True)
    checkpoint = GGUFCheckpoint.open(file)
    skipped = [checkpoint.tensors[name] for name in ("token_embd.weight", "output.weight")]
    read_spans = []

    class CoreReader(Reader):
        def read(self, path, offset, length, *args, **kwargs):
            assert all(offset + length <= t.offset or offset >= t.offset + t.nbytes for t in skipped)
            read_spans.append((offset, length))
            return super().read(path, offset, length, *args, **kwargs)

    monkeypatch.setattr(mtp, "Reader", CoreReader)
    target.frontend = object()
    embedding, head, frontend = target.embed, target.head, target.frontend
    plan = byte_plan(target, file, tp=(3, 8))
    attach(target, file, tp=(3, 8))
    assert read_spans
    assert target.embed is embedding and target.head is head and target.frontend is frontend
    assert _owned(target.mtp) == plan["resident_bytes"]
    receipt = plan["import"]
    assert receipt["source_tensor_count"] == 34 and receipt["imported_tensor_count"] == 32
    assert receipt["source_final_compression_ratio"] == 4
    assert receipt["source_payload_bytes"] - receipt["imported_source_payload_bytes"] == sum(t.nbytes for t in skipped)
    assert set(receipt["skipped_source_tensors"]) == {t.name for t in skipped}
    assert target.draft_head is None and target.draft_ids is None
    shared = tmp_path / "shared"
    shared.mkdir()
    control, control_file, _ = fixture(shared)
    assert byte_plan(control, control_file, tp=(3, 8))["resident_bytes"] == plan["resident_bytes"]


@pytest.mark.parametrize("change", ["missing-head", "wrong-shape", "shared-flag", "numeric-flag", "dense-ratio"])
def test_invalid_full_companion_is_rejected_before_payload_io(tmp_path, monkeypatch, change):
    def mutate(metadata, entries):
        if change == "missing-head":
            del entries["output.weight"]
        elif change == "wrong-shape":
            entries["output.weight"] = ((32, 64), 8, bytes(32 * 68))
        elif change in ("shared-flag", "numeric-flag"):
            metadata["qwen4exp.nextn_shared_target_tensors"] = True if change == "shared-flag" else 0
        else:
            metadata["qwen4exp.attention.compress_ratios"] = (0, 0, 0, 4, 0)

    target, file, _ = fixture(tmp_path, vocabulary_copies=True, mutate=mutate)
    monkeypatch.setattr(mtp, "Reader", lambda: pytest.fail("invalid full companion reached payload IO"))
    with pytest.raises(ValueError, match="vocabulary copies|QSA compression"):
        attach(target, file, tp=(3, 8))
    assert target.mtp is None


@pytest.mark.parametrize("field,original,changed", [
    ("token_type", (1, 3), (3, 1)), ("merges", ("a b", "b a"), ("b a", "a b")),
    ("bos_token_id", 0, 1), ("eos_token_id", 1, 0), ("padding_token_id", 0, 1),
])
def test_shared_token_identity_includes_ordered_semantics(field, original, changed):
    metadata = {"tokenizer.ggml.tokens": ("a", "b"), "tokenizer.ggml." + field: original}
    before = token_id_identity(metadata)
    assert token_id_identity({**metadata, "tokenizer.ggml." + field: changed}) != before
    assert token_id_identity({**metadata, "tokenizer.chat_template": "another target template",
                              "tokenizer.ggml.normalizer.nfc": True,
                              "tokenizer.ggml.add_eos_token": False}) == before


@pytest.mark.parametrize("field,value", [("token_type", (1,)), ("token_type", (True, 1)),
                                          ("merges", ("a b", 2)), ("eos_token_id", 2),
                                          ("bos_token_id", True)])
def test_shared_token_identity_rejects_invalid_role_metadata(field, value):
    with pytest.raises(ValueError, match="token"):
        token_id_identity({"tokenizer.ggml.tokens": ("a", "b"), "tokenizer.ggml." + field: value})


@pytest.mark.parametrize("directory", [False, True])
def test_mtp_only_source_preparation_uses_its_own_role_without_trunk_or_ple(tmp_path, monkeypatch, directory):
    from tensorfold.families.qwen4_exp.position_profile import (
        _source, prepare_position_vectors, resolve_position_profile,
    )

    def positions(metadata, _entries):
        metadata["qwen4exp.context_length"] = 262144
        metadata["qwen4exp.rope.dimension_sections"] = (2, 1, 1, 0)

    target, file, _ = fixture(tmp_path, vocabulary_copies=True, mutate=positions)
    raw = metadata_config(file, role="mtp")
    text = raw["text_config"]
    assert text["num_hidden_layers"] == 1 and text["gguf_source_block_count"] == 5
    assert text["layer_types"] == ["full_attention"]
    assert not any("ple" in key for key in text)
    assert np.float32(text["rms_norm_eps"]) == np.float32(1e-6)
    source = tmp_path if directory else file
    resolved, inputs, digest = _source(source, mtp=True)
    assert resolved == raw and inputs == (file.resolve(),)
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert digest == hashlib.sha256(encoded).hexdigest()
    target_directory = tmp_path / "target"
    target_directory.mkdir()
    target_config = {"model_type": "qwen4_exp", "text_config": {
        "hidden_size": 64, "head_dim": 8, "num_attention_heads": 8, "max_position_embeddings": 262144,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.,
                            "partial_rotary_factor": 1., "mrope_section": [2, 1, 1],
                            "mrope_interleaved": True}}}
    (target_directory / "config.json").write_text(json.dumps(target_config))
    profile = resolve_position_profile(target_directory, requested_context=262144,
                                       companion_configs={"mtp": source})
    assert profile.mtp.source_config_sha256 == digest
    assert profile.mtp.rope.mrope_section == (2, 1, 1)
    assert profile.mtp.rope.theta == 10000000. and profile.mtp.rope.rotary_dim == 8
    prepared = prepare_position_vectors(profile, mtp_active=True)
    expected = (10000000. ** (-torch.arange(0, 8, 2, dtype=torch.float64) / 8)).float()
    torch.testing.assert_close(prepared.mtp_cpu, expected, atol=0, rtol=0)
    target.meta["position_profile"] = profile
    monkeypatch.setattr(mtp, "Reader", lambda: pytest.fail("role preparation entered payload IO"))
    assert byte_plan(target, file, tp=(3, 8))["import"]["imported_tensor_count"] == 32


@pytest.mark.parametrize("change", ["extra-trunk", "missing-core", "bad-sections", "bad-epsilon"])
def test_mtp_role_metadata_rejects_incompatible_physical_source(tmp_path, change):
    def mutate(metadata, entries):
        metadata["qwen4exp.context_length"] = 262144
        metadata["qwen4exp.rope.dimension_sections"] = (2, 1, 1, 0)
        if change == "extra-trunk":
            entries["blk.0.nextn.enorm.weight"] = ((64,), 0, bytes(256))
        elif change == "missing-core":
            del entries["blk.4.nextn.hnorm.weight"]
        elif change == "bad-sections":
            del metadata["qwen4exp.rope.dimension_sections"]
        else:
            metadata["qwen4exp.attention.layer_norm_rms_epsilon"] = float("nan")

    _target, file, _ = fixture(tmp_path, vocabulary_copies=True, mutate=mutate)
    with pytest.raises(ValueError, match="prediction core|mRoPE|normalization"):
        metadata_config(file, role="mtp")


def test_configured_floating_mtp_source_keeps_its_config_with_unrelated_gguf(tmp_path):
    from tensorfold.families.qwen4_exp.position_profile import _source

    raw = {"model_type": "qwen4_exp", "text_config": {"head_dim": 8, "hidden_size": 64,
           "max_position_embeddings": 262144, "rope_parameters": {"rope_theta": 10000000.,
           "partial_rotary_factor": 1., "mrope_section": [2, 1, 1]}}}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw))
    (tmp_path / "unrelated.gguf").write_bytes(b"not a checkpoint")
    resolved, inputs, _digest = _source(tmp_path, mtp=True)
    assert resolved == raw and inputs == (config.resolve(),)


@pytest.mark.parametrize("change", [None, "theta", "reference", "sections"])
def test_bound_mtp_validates_own_source_role_before_payload(tmp_path, monkeypatch, change):
    from tensorfold.families.qwen4_exp.rope import RopeParameters

    def declared(metadata, _entries):
        metadata["qwen4exp.context_length"] = 262144
        metadata["qwen4exp.rope.dimension_sections"] = (2, 1, 1, 0)

    target, file, _source = fixture(tmp_path, mutate=declared)
    own = RopeParameters("default", 10000000., 8, 262144, 262144, mrope_section=(2, 1, 1))
    # The target's selected transform may differ. The companion remains bound
    # to the independently resolved native MTP source, including its reference.
    target.cfg.rope_theta = 123456.
    if change == "theta":
        own = replace(own, theta=123456.)
    elif change == "reference":
        own = replace(own, native_context=524288)
    elif change == "sections":
        own = replace(own, mrope_section=(1, 2, 1))
    target.meta["position_profile"] = SimpleNamespace(mtp=SimpleNamespace(rope=own))
    monkeypatch.setattr(mtp, "Reader", lambda: pytest.fail("position validation entered payload IO"))
    if change is None:
        assert byte_plan(target, file, tp=(3, 8))["resident_bytes"] > 0
    else:
        with pytest.raises(ValueError, match="positional|freq_base"):
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
    metadata, _tensors = _shard(Path(target_metadata))
    target.meta["gguf_token_id_identity"] = token_id_identity(metadata)
    plan = byte_plan(target, Path(selected), tp=(0, 8))
    assert plan["import"]["source_tensor_count"] == 34
    assert plan["import"]["source_payload_bytes"] == 4126482432
    assert plan["import"]["imported_tensor_count"] == 32
    assert plan["import"]["imported_source_payload_bytes"] == 2775621632
    assert plan["import"]["source_final_compression_ratio"] == 4
    target.embed, target.head = object(), object()
    embedding, head = target.embed, target.head
    attach(target, Path(selected), tp=(0, 8))
    assert _owned(target.mtp) == plan["resident_bytes"]
    assert target.embed is embedding and target.head is head
    assert torch.isfinite(target.mtp.fc_h.weight).all()
    assert torch.isfinite(target.mtp.fc_e.weight).all()
    print(json.dumps(plan, sort_keys=True))
