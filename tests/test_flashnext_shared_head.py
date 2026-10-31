"""Shared-target MTP uses the real head and only filters proposal logits."""

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gguf_fixture import blocks, scalar_reference, tiny_model, write
from tensorfold.cuda.direct_read import Reader
from tensorfold.cuda.sampling import _tp_candidates, _tp_choose
from tensorfold.engine.exact_sampling import Sampling, choose
from tensorfold.families.qwen4_exp.cuda import gguf_admission, gguf_mtp_weights, gguf_weights
from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint

pytestmark = pytest.mark.torch


def pair(tmp_path, *, invalid=False):
    """Real compatible directories; original target matrix bytes stay intact."""
    original = tiny_model(tmp_path / "original.gguf")
    checkpoint = GGUFCheckpoint.open(original)
    raw = original.read_bytes()
    metadata = dict(checkpoint.metadata)
    metadata.update({"tokenizer.ggml.tokens": tuple(f"t{i}" for i in range(32)),
                     "qwen4exp.attention.indexer.top_k": 2048, "qwen4exp.attention.value_length": 32})
    tensors = [(t.name, t.shape, t.ggml_type, np.frombuffer(raw[t.offset:t.offset + t.nbytes], np.uint8))
               for t in checkpoint.tensors.values()]
    target = write(tmp_path / "target.gguf", tensors, metadata)
    core = []
    for name, shape, kind, data in tensors:
        if name.startswith("blk.1."):
            if kind not in (0, 1, 8, 30):
                kind, data = 8, blocks(8, int(np.prod(shape[:-1])), shape[-1])
            core.append(("blk.2." + name.removeprefix("blk.1."), shape, kind, data))
        elif name.startswith("output_hc_"):
            core.append(("blk.2.nextn.hc_head_" + name.removeprefix("output_hc_"), shape, kind, data))
        elif name in ("token_embd.weight", "output.weight"):
            core.append((name, shape, kind, data))
    for name, shape in (("enorm", (256,)), ("hnorm", (1024,))):
        core.append((f"blk.2.nextn.{name}.weight", shape, 0, np.ones(shape, "<f4").view(np.uint8)))
    core.append(("blk.2.nextn.eh_proj.weight", (256, 512), 8, blocks(8, 256, 512)))
    metadata = {k: v for k, v in metadata.items() if not k.startswith("qwen4exp.ple.")}
    metadata.update({"qwen4exp.block_count": 3, "qwen4exp.nextn_predict_layers": 1,
                     "qwen4exp.attention.compress_ratios": (0, 4, 4)})
    if invalid:
        metadata["tokenizer.ggml.tokens"] = ("changed", *metadata["tokenizer.ggml.tokens"][1:])
    companion = write(tmp_path / "mtp.gguf", core, metadata)
    assert len(core) == 34
    return target, companion


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("selection", [None, (1, 5, 6, 29, 31)])
def test_shared_plan_owns_no_extra_vocabulary_matrix_and_counts_only_filter(tmp_path, monkeypatch, rank, selection):
    target, companion = pair(tmp_path)
    ids = None if selection is None else np.array(selection, dtype=np.int64)
    baseline, baseline_plan = gguf_weights.model_plan(target, (rank, 8))
    selected, selected_plan = gguf_weights.model_plan(target, (rank, 8), mtp=True,
                                                     mtp_model_dir=companion, draft_ids=ids)
    assert selected.head == baseline.head
    assert selected.draft_head is None and selected.draft_ids is None
    assert len(selected_plan.matrices) == len(baseline_plan.matrices)
    budget = gguf_weights.weight_byte_plan(target, (rank, 8), mtp=True,
                                           mtp_model_dir=companion, draft_ids=ids)
    ordinary = gguf_weights.weight_byte_plan(target, (rank, 8))
    expected_bytes = 0 if selection is None else selected.head.n
    assert budget["draft_head_bytes"] == budget["draft_id_bytes"] == 0
    assert budget["draft_filter_bytes"] == expected_bytes
    assert budget["resident_bytes"] - ordinary["resident_bytes"] == expected_bytes
    if selection is not None:
        expected = ~np.isin(np.arange(rank * 4, (rank + 1) * 4), ids)
        np.testing.assert_array_equal(selected.draft_filter, expected)
    reader = Reader()
    try:
        head = gguf_mtp_weights._cpu_dense(selected.head, reader, "cpu")
        instance = SimpleNamespace(head=head, draft_head=None, draft_ids=None, draft_filter=None,
                                   meta={"mtp_import": {}})
        pointer, before = head.weight.data_ptr(), head.weight.clone()
        monkeypatch.setattr(gguf_weights, "_draft_dense", lambda *_a: pytest.fail("shared MTP copied target head weights"))
        gguf_weights._bind_mtp_vocabulary(instance, selected, reader, "cpu")
        assert instance.head is head and instance.head.weight.data_ptr() == pointer
        assert torch.equal(instance.head.weight, before)
        assert instance.draft_head is None and instance.draft_ids is None
        assert instance.meta["mtp_import"]["vocabulary_selection"] == selected.meta["mtp_vocabulary_selection"]
        if selection is None:
            assert instance.draft_filter is None
        else:
            assert instance.draft_filter.dtype == torch.bool and instance.draft_filter.numel() == 4
            digest = hashlib.sha256(np.asarray(selection, "<i8")).hexdigest()
            assert selected.meta["mtp_vocabulary_selection"]["token_ids_sha256"] == digest
    finally:
        reader.close()


def test_actual_companion_header_is_validated_before_loader_payload_io(tmp_path, monkeypatch):
    target, incompatible = pair(tmp_path, invalid=True)
    monkeypatch.setattr(gguf_weights, "Reader", lambda: pytest.fail("invalid MTP pairing reached target payload reader"))
    with pytest.raises(ValueError, match="token-ID"):
        gguf_weights.load_gguf(target, "cpu", mtp=True, mtp_model_dir=incompatible)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("count", [1, 3, 8])
def test_real_engine_preload_accepts_sparse_shared_support_and_preserves_legacy_minimum(
    tmp_path, monkeypatch, shared, count,
):
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp.cuda import engine, speculation

    target, companion = pair(tmp_path)
    if not shared:
        companion = tmp_path / "floating-mtp"
        companion.mkdir()
        (companion / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": {
            "hidden_size": 256, "head_dim": 32, "num_attention_heads": 8, "max_position_embeddings": 256,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.,
                                "partial_rotary_factor": .25, "mrope_section": [1, 1, 2]}}}))
    # Frontend rendering has separate real-source coverage. Exercise this real
    # constructor's metadata/position/selector path up to its next preload stage.
    monkeypatch.setattr(chat_template, "prepare_chat_template", lambda *_a, **_k: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_a, **_k: pytest.fail("selector preflight touched CUDA"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: pytest.fail("selector preflight queried CUDA"))
    reached, retired = [], []

    class NextPreload(Exception):
        pass

    def next_preload(*args, **kwargs):
        reached.append((args, kwargs))
        raise NextPreload

    def retire(owner):
        assert owner.comm is None and owner.w is None and owner.scheduler is None and owner.multi is None
        retired.append(True)

    monkeypatch.setattr(speculation, "MethodGeometry", next_preload)
    monkeypatch.setattr(engine.FlashNextEngine, "_rollback_startup", retire)
    expected, message = ((NextPreload, None) if shared or count >= 8 else
                         (ValueError, "at least one ID per tensor-parallel rank"))
    with pytest.raises(expected, match=message):
        engine.FlashNextEngine(target, depth=2, tp=8, rank=0, master="fixture", graphs=False,
                                kv_dtype="int8", adaptive_sessions=True, mtp_model_dir=companion,
                                draft_vocab=count, max_waiting_input_bytes=4096)
    assert len(reached) == int(shared or count >= 8) and retired == [True]


def test_shared_admission_includes_filter_and_actual_core_without_duplicate_head(tmp_path):
    target, companion = pair(tmp_path)
    ids = np.array([1], dtype=np.int64)  # seven physical ranks legitimately have zero proposal mass
    rank = 7
    plan = gguf_weights.weight_byte_plan(target, (rank, 8), mtp=True,
                                         mtp_model_dir=companion, draft_ids=ids)
    prototype, _ = gguf_weights.model_plan(target, (rank, 8))
    core = gguf_mtp_weights.byte_plan(prototype, companion, tp=(rank, 8))
    admitted = gguf_admission.capacity_options(target, rank, 8, mtp=True,
                                               mtp_model_dir=companion, draft_ids=ids)
    weights, host = admitted["weight_estimator"](target)
    assert weights.resident == plan["resident_bytes"] + core["resident_bytes"]
    assert host == max(plan["host_staging_bytes"], core["host_staging_bytes"])
    assert plan["draft_filter_bytes"] == 4 and plan["draft_head_bytes"] == 0


def test_shared_subset_packets_preserve_ids_probability_and_empty_local_mass(tmp_path):
    target, companion = pair(tmp_path)
    ids = np.array([1, 5, 6, 29, 31], dtype=np.int64)
    x = torch.linspace(-.125, .25, 3 * 256).reshape(3, 256).half()
    shard_scores, packets = [], []

    class PacketReady(Exception):
        pass

    def save(packet):
        packets.append(packet.clone())
        raise PacketReady

    for rank in range(8):
        planned, _ = gguf_weights.model_plan(target, (rank, 8), mtp=True,
                                              mtp_model_dir=companion, draft_ids=ids)
        reader = Reader()
        try:
            head = gguf_mtp_weights._cpu_dense(planned.head, reader, "cpu")
            source = planned.head.source
            raw = np.fromfile(source.path, np.uint8, count=source.nbytes, offset=source.offset).reshape(32, -1)
            oracle = scalar_reference(raw, source.ggml_type, 256)[rank * 4:(rank + 1) * 4].astype(np.float16)
            np.testing.assert_array_equal(head.weight.numpy(), oracle)
            # CPU probability/transport oracle over the decoded source values.
            # GGUFDense's FP16->FP32 projection remains a CUDA-only kernel gate.
            logits = x.float() @ head.weight.float().T
            logits.masked_fill_(torch.from_numpy(planned.draft_filter.copy()), -float("inf"))
            shard_scores.append(logits)
            with pytest.raises(PacketReady):
                _tp_candidates(SimpleNamespace(gather=save, world=8), logits, 4, None, rank * 4,
                               with_norm=True, centered_norm=True)
        finally:
            reader.close()
    assert any(torch.isneginf(scores).all() for scores in shard_scores)
    values, tokens, norms = _tp_candidates(SimpleNamespace(gather=lambda _packet: torch.stack(packets), world=8),
                                           shard_scores[0], 4, None, 0, with_norm=True, centered_norm=True)
    assert not np.isnan(norms).any()
    full = torch.cat(shard_scores, dim=1).numpy()
    positions = [17, 18, 19]
    for policy in (None, Sampling(21, temperature=0), Sampling(21, top_k=20),
                   Sampling(3, temperature=.7, top_k=0, top_p=.9, min_p=.1)):
        chosen = _tp_choose(values, tokens, positions, policy)
        expected = [int(ids[np.argmax(row[ids])]) if policy is None or policy.temperature <= 0
                    else choose(row[ids], ids, position, policy) for row, position in zip(full, positions)]
        assert chosen == expected
        for row, selected in zip(full, chosen):
            maximum = float(row[ids].max())
            denominator = math.fsum(math.exp(float(row[token]) - maximum) for token in ids)
            compact_probability = math.exp(float(row[selected]) - maximum) / denominator
            full_probability = math.exp(float(row[selected]) - maximum) / math.fsum(
                math.exp(float(value) - maximum) for value in row)
            assert full_probability == compact_probability
