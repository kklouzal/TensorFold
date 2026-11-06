"""Private PP2 stage-owned imports; no transport, method service or device qualification."""
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.pipeline_stage import PipelineStage, for_weights


def _actual_storage_bytes(value):
    """Count complete reachable CPU tensor storage, with aliased views deduplicated."""
    import torch
    from dataclasses import fields, is_dataclass

    seen, storage_bytes = {}, {}

    def visit(item):
        if id(item) in seen:
            return
        seen[id(item)] = item  # Retain temporary tuples/owners so identities cannot recycle mid-walk.
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (item.device.type, item.device.index, storage.data_ptr())
            storage_bytes[key] = max(storage_bytes.get(key, 0), storage.nbytes())
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif callable(getattr(item, "device_tensors", None)):
            visit(item.device_tensors())
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            visit(vars(item))

    visit(value)
    total = sum(storage_bytes.values())
    seen.clear()
    return total


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from gguf_fixture import tiny_model

    return tiny_model(tmp_path_factory.mktemp("pp2-import") / "target.gguf",
                      layers=48, attention_interval=4, experts=4)


@pytest.mark.parametrize("index", [True, -1, 2, 0.0, "0"])
def test_stage_index_is_fixed_and_typed(index):
    with pytest.raises(ValueError):
        PipelineStage(index)


@pytest.mark.torch
@pytest.mark.parametrize("etp", [8, 4, 2])
@pytest.mark.parametrize("rank", range(8))
def test_two_actual_stage_plans_partition_the_complete_inventory(checkpoint, etp, rank):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan, weight_byte_plan

    full, whole = model_plan(checkpoint, (rank, 8), expert_tensor_parallel_size=etp)
    stages, plans = [], []
    for index in (0, 1):
        stage = PipelineStage(index)
        weights, plan = model_plan(checkpoint, (rank, 8), expert_tensor_parallel_size=etp,
                                   _pipeline_stage=stage)
        stages.append(weights); plans.append(plan)
        assert for_weights(weights) == stage
        assert weights.cfg.layers == weights.meta["full"].layers == 48
        assert weights.cfg.layer_types == full.cfg.layer_types
        assert [layer.index for layer in weights.layers] == list(stage.layer_ids)
        assert sum(not layer.linear for layer in weights.layers) == 6
        assert weights.meta["history_layout"] == "unique" and weights.meta["attention_degree"] == 4
        assert (weights.embed is not None) is stage.owns_input
        assert (weights.head is not None) is stage.owns_output
        assert (weights.mixer is not None) is stage.owns_output
        assert weights.cfg.moe_width == 640 // etp and weights.cfg.shared_width == 80
        ledger = weight_byte_plan(checkpoint, (rank, 8), expert_tensor_parallel_size=etp,
                                  _pipeline_stage=stage)
        resident = sum(matrix.resident_bytes for matrix in plan.matrices) + sum(array.resident_bytes for array in plan.arrays)
        assert ledger["resident_bytes"] == resident
        assert weights.meta["expert_partition_scope"] == ledger["expert_partition_scope"] == "stage-local"
        assert ledger["pipeline_stage"] == stage.identity()
        assert ledger["owned_global_layer_ids"] == list(stage.layer_ids)
        assert weights.meta["pipeline_tensor_ownership"]["owned"] == tuple(sorted(plan.used))
        assert set(weights.meta["pipeline_tensor_ownership"]["complement"]) == set(whole.used) - plan.used
    assert not plans[0].used & plans[1].used
    assert plans[0].used | plans[1].used == whole.used
    assert "token_embd.weight" in plans[0].used and "token_embd.weight" not in plans[1].used
    assert "output.weight" not in plans[0].used and "output.weight" in plans[1].used
    # Source row/block selections remain identical to the corresponding PP1 blocks.
    for weights in stages:
        for layer in weights.layers:
            expected = full.layers[layer.index].moe.experts
            for name in ("gate", "up", "down"):
                actual = getattr(layer.moe.experts, name)
                assert actual == getattr(expected, name)


@pytest.mark.torch
@pytest.mark.parametrize("stage_index", [0, 1])
def test_actual_loader_materializes_only_owned_coefficients_on_cpu(checkpoint, stage_index, monkeypatch):
    import torch
    from tensorfold.families.qwen4_exp.cuda import gguf_weights as module
    from tensorfold.families.qwen4_exp.cuda.gguf_mtp_weights import _cpu_dense
    from tensorfold.families.qwen4_exp.gguf import decode_rows
    from tensorfold.gguf_file import PackedTensor

    stage = PipelineStage(stage_index)
    prototype, planner = module.model_plan(checkpoint, (0, 8), expert_tensor_parallel_size=4, _pipeline_stage=stage)
    ledger = module.weight_byte_plan(checkpoint, (0, 8), expert_tensor_parallel_size=4, _pipeline_stage=stage)
    # Isolate only the dense decoder's device boundary, reusing the maintained CPU oracle.
    monkeypatch.setattr(module, "_dense", _cpu_dense)
    reads = []
    original = module.materializer

    def owned_materializer(*args, **kwargs):
        materialize = original(*args, **kwargs)

        def tracked(value):
            if isinstance(value, module.LayerW):
                reads.append(value.index)
            return materialize(value)
        return tracked

    monkeypatch.setattr(module, "materializer", owned_materializer)
    weights = module.load_gguf(checkpoint, "cpu", tp=(0, 8), expert_tensor_parallel_size=4, _pipeline_stage=stage)
    try:
        assert for_weights(weights) == stage
        assert reads == list(stage.layer_ids)
        # The existing loader charges its inverse-frequency vector as workspace.
        assert _actual_storage_bytes(weights) == ledger["resident_bytes"] + weights.inv_freq.numel() * weights.inv_freq.element_size()
        assert weights.device.type == "cpu" and not torch.cuda.is_initialized()
        bank = weights.layers[0].moe.experts
        planned = prototype.layers[0].moe.experts
        for name in ("gate", "up", "down"):
            packed = getattr(bank, name)
            source = getattr(planned, name)
            descriptor = PackedTensor("control", (packed.n, packed.data.shape[1] // module.FORMATS[packed.ggml_type][2]
                                                  * module.FORMATS[packed.ggml_type][1]),
                                      packed.ggml_type, checkpoint, 0, packed.data.numel())
            actual = decode_rows(descriptor, packed.data.numpy())[:, packed.column_offset:packed.column_offset + packed.k]
            raw = []
            with source.source.path.open("rb") as stream:
                for first, count in source.row_ranges:
                    for row in range(first, first + count):
                        stream.seek(source.source.offset + row * source.source.row_bytes)
                        raw.append(np.frombuffer(stream.read(source.source.row_bytes), np.uint8))
            expected = decode_rows(source.source, np.stack(raw))
            lo, count = source.columns[0]
            np.testing.assert_array_equal(actual, expected[:, lo:lo + count])
        assert not planner.tables  # This text fixture has no PLE; real lookup ownership is checked separately.
    finally:
        weights.meta["ple_tables"].close(lambda: None)


@pytest.mark.torch
@pytest.mark.parametrize("mutation", ["unknown_complement", "missing_complement", "later_ple"])
def test_complement_is_explicitly_validated_before_any_plan_allocation(checkpoint, monkeypatch, mutation):
    from tensorfold.families.qwen4_exp.cuda import gguf_weights as module
    from tensorfold.gguf_file import GGUFCheckpoint

    inventory = module.open_checkpoint(checkpoint)
    tensors = dict(inventory.tensors)
    if mutation == "unknown_complement":
        source = next(iter(tensors.values()))
        tensors["blk.25.unknown.weight"] = replace(source, name="blk.25.unknown.weight")
    elif mutation == "missing_complement":
        tensors.pop("blk.25.ffn_gate_exps.weight")
    else:
        original = module.Config.read
        monkeypatch.setattr(module.Config, "read", lambda *args, **kwargs: replace(original(*args, **kwargs), ple_layers=[25]))
    monkeypatch.setattr(module, "open_checkpoint", lambda _: GGUFCheckpoint(inventory.metadata, MappingProxyType(tensors), inventory.paths))
    monkeypatch.setattr(module, "Planner", lambda *args: pytest.fail("bad whole inventory must reject before planner/allocation"))
    with pytest.raises(ValueError, match="inventory|later-stage PLE"):
        module.model_plan(checkpoint, (0, 8), _pipeline_stage=PipelineStage(0))


@pytest.mark.torch
@pytest.mark.parametrize("method", [{"mtp": True}, {"mtp_model_dir": "companion"}, {"draft_vocab": "default"}])
def test_private_partial_import_rejects_methods_before_payload_io(checkpoint, method, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import gguf_weights as module

    monkeypatch.setattr(module, "Reader", lambda: pytest.fail("method service missing; no payload reader allowed"))
    with pytest.raises(ValueError, match="owner lookup|partial pipeline"):
        module.load_gguf(checkpoint, "cpu", tp=(0, 8), _pipeline_stage=PipelineStage(1), **method)


@pytest.mark.torch
def test_stage_descriptor_rejects_missing_endpoints_and_bad_identity(checkpoint):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan

    weights, _ = model_plan(checkpoint, (0, 8), _pipeline_stage=PipelineStage(0))
    weights.head = object()
    with pytest.raises(ValueError, match="endpoint"):
        for_weights(weights)
    weights.head = None
    weights.meta["pipeline_stage"]["index"] = True
    with pytest.raises(ValueError):
        for_weights(weights)


@pytest.mark.torch
def test_storage_oracle_counts_containers_and_deduplicates_views():
    import torch

    first, second = torch.empty(16), torch.empty(7)

    class Owner:
        def __init__(self, tensor):
            self.tensor = tensor

        def device_tensors(self):
            return (self.tensor,)

    graph = {"list": [first, first[4:], (first.reshape(4, 4),)],
             "owners": (Owner(first), Owner(second)), "empty": torch.empty(0)}
    assert _actual_storage_bytes(graph) == first.untyped_storage().nbytes() + second.untyped_storage().nbytes()


@pytest.mark.torch
def test_real_input_ple_table_and_global_ordinal_stay_exclusive_to_stage_zero(tmp_path, monkeypatch):
    import struct
    from gguf_fixture import tiny_model
    from tensorfold.families.qwen4_exp.cuda import gguf_weights as module
    from tensorfold.families.qwen4_exp.cuda.gguf_mtp_weights import _cpu_dense
    from tensorfold.families.qwen4_exp.gguf import decode_rows

    path = tiny_model(tmp_path / "with-ple.gguf", experts=4, layers=48, attention_interval=4, ple_layer=1)
    inventory = module.open_checkpoint(path)
    source = inventory.tensors["per_layer_token_embd.weight"]
    assert source.ggml_type == 20 and source.shape[-1] == 32 and source.row_bytes == 18
    raw = np.stack([np.frombuffer(struct.pack("<e", scale) + bytes([0x88] * 16), np.uint8)
                    for scale in (.03125, .0625)])
    with path.open("r+b") as stream:
        stream.seek(source.offset)
        stream.write(raw.tobytes())
    before = path.stat()
    input_weights, input_plan = module.model_plan(path, (0, 8), _pipeline_stage=PipelineStage(0))
    output_weights, output_plan = module.model_plan(path, (0, 8), _pipeline_stage=PipelineStage(1))
    assert input_weights.cfg.ple_layers == input_weights.meta["full"].ple_layers == [1]
    assert output_weights.cfg.ple_layers == [] and output_weights.meta["full"].ple_layers == [1]
    assert len(input_plan.tables) == 1 and input_plan.tables[0].source == source
    assert not output_plan.tables and all(layer.ple is None for layer in output_weights.layers)
    assert "per_layer_token_embd.weight" in input_plan.used
    assert "per_layer_token_embd.weight" in output_weights.meta["pipeline_tensor_ownership"]["complement"]
    ngram = input_weights.layers[1].ple.ngram
    original = input_weights.meta["full"].ngram(0)
    np.testing.assert_array_equal(ngram.multipliers, original.multipliers)
    np.testing.assert_array_equal(ngram.head_offsets, original.head_offsets)
    np.testing.assert_array_equal(ngram.head_sizes, original.head_sizes)
    first = module.weight_byte_plan(path, (0, 8), _pipeline_stage=PipelineStage(0))
    last = module.weight_byte_plan(path, (0, 8), _pipeline_stage=PipelineStage(1))
    assert first["host_resident_bytes"] == source.nbytes and last["host_resident_bytes"] == 0
    monkeypatch.setattr(module, "_dense", _cpu_dense)
    loaded = module.load_gguf(path, "cpu", tp=(0, 8), _pipeline_stage=PipelineStage(0), ple_on_ssd=True)
    try:
        values = loaded.layers[1].ple.table.gather(np.array([0, 1], np.int64))
        np.testing.assert_array_equal(values, decode_rows(source, raw).astype(np.float16))
        assert loaded.layers[1].ple.table.rows == source.shape[0]
    finally:
        loaded.meta["ple_tables"].close(lambda: None)
    with path.open("rb") as stream:
        stream.seek(source.offset)
        assert stream.read(raw.nbytes) == raw.tobytes()
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
