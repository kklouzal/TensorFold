"""Mixed-format loader plans preserve exact storage, topology and family identities."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gguf_fixture import blocks, scalar_reference, tiny_model
from tensorfold.gguf_file import PackedTensor, resolve_gguf
from tensorfold.families.qwen4_exp.gguf import decode_rows, metadata_config


def _incumbent_dense_m32(linear, x, out):
    """Frozen pre-candidate padding/product/copy oracle, independent of dispatch."""
    import torch
    for start in range(0, x.shape[0], 32):
        count = min(32, x.shape[0] - start)
        padded = torch.zeros((32, linear.k), dtype=x.dtype, device=x.device)
        padded[:count].copy_(x[start:start + count])
        result = torch.mm(padded, linear.weight.T)
        out[start:start + count].copy_(result[:count])
    return out


def _aligned_cpu_dense(shape):
    import torch
    count = int(np.prod(shape))
    owner = torch.zeros(count + 256, dtype=torch.float16)
    offset = ((-owner.data_ptr()) % 512) // owner.element_size()
    return owner[offset:offset + count].view(shape)


@pytest.mark.torch
@pytest.mark.parametrize("rows", [1, 31, 32, 33, 64, 65, 256])
def test_native32_fulltile_guard_dispatch_and_tail_use_frozen_oracle(rows, monkeypatch):
    """CPU spies validate dispatch only, NOT physical GPU numeric equivalence."""
    import torch
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
    x, out = _aligned_cpu_dense((rows, 64)), _aligned_cpu_dense((rows, 64))
    x.copy_(torch.arange(rows * 64).view(rows, 64).remainder(31).half() / 32)
    linear = GGUFDense(torch.eye(64, dtype=torch.float16))
    linear.implementation = "native32"
    calls = []
    monkeypatch.setattr(linear, "can_matmul_prepared", lambda dtype: True)
    def prepared(source, target):
        calls.append((source.shape, target.shape))
        return _incumbent_dense_m32(linear, source, target)
    monkeypatch.setattr(linear, "matmul_prepared", prepared)
    expected = _incumbent_dense_m32(linear, x, torch.empty_like(out))
    assert linear.matmul_reference(x, out=out) is out
    assert len(calls) == rows // 32
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.torch
@pytest.mark.parametrize("case", ["reference", "input-stride", "output-stride", "input-align",
                                  "output-align", "input-negative", "output-negative", "weight-grad", "weight-parameter",
                                  "input-output-overlap", "output-weight-overlap", "later-input-overlap"])
def test_fulltile_guard_preserves_incumbent_for_unsupported_layouts(case, monkeypatch):
    import torch
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
    x, out = _aligned_cpu_dense((64, 64)), _aligned_cpu_dense((64, 64))
    linear = GGUFDense(torch.eye(64, dtype=torch.float16))
    linear.implementation = "reference" if case == "reference" else "native32"
    if case == "input-stride":
        x = _aligned_cpu_dense((64, 128))[:, ::2]
    elif case == "output-stride":
        out = _aligned_cpu_dense((64, 128))[:, ::2]
    elif case == "input-align":
        x = _aligned_cpu_dense((64 * 64 + 1,))[1:].view(64, 64)
    elif case == "output-align":
        out = _aligned_cpu_dense((64 * 64 + 1,))[1:].view(64, 64)
    elif case == "input-negative":
        x = torch._neg_view(x)
    elif case == "output-negative":
        out = torch._neg_view(out)
    elif case == "weight-grad":
        linear.weight.requires_grad_()
    elif case == "weight-parameter":
        linear.weight = torch.nn.Parameter(linear.weight, requires_grad=False)
    elif case == "input-output-overlap":
        out = x
    elif case == "output-weight-overlap":
        out = linear.weight
    elif case == "later-input-overlap":
        owner = _aligned_cpu_dense((96, 64))
        x, out = owner[:64], owner[32:]
    monkeypatch.setattr(linear, "can_matmul_prepared", lambda dtype: True)
    def forbidden(*args):
        pytest.fail("unsupported dense layout selected the prepared candidate")
    monkeypatch.setattr(linear, "matmul_prepared", forbidden)
    # Some legacy alias/autograd combinations can reject during copy; the
    # important boundary is that the new prepared path cannot consume them.
    try:
        linear.matmul_reference(x, out=out)
    except RuntimeError as error:
        assert case in ("weight-grad", "output-weight-overlap") and "grad" in str(error)


@pytest.mark.parametrize("kind,width", [(0, 33), (1, 33), (30, 33), (8, 96), (7, 96),
                                      (12, 512), (13, 512), (20, 96)])
def test_mixed_cpu_decoders_match_independent_scalar_oracle(kind, width):
    data = blocks(kind, 3, width)
    descriptor = PackedTensor("test", (3, width), kind, Path("unused"), 0, data.nbytes)
    np.testing.assert_array_equal(decode_rows(descriptor, data), scalar_reference(data, kind, width))


@pytest.mark.torch
@pytest.mark.parametrize("rank", range(8))
def test_tp8_plans_use_legal_block_views_and_replicated_kv(tmp_path, rank):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan, weight_byte_plan

    path = tiny_model(tmp_path / "tiny.gguf")
    weights, plan = model_plan(path, (rank, 8))
    assert len(plan.used) == len(plan.checkpoint.tensors)
    assert (weights.cfg.heads, weights.cfg.kv_heads, weights.cfg.nk, weights.cfg.nv) == (1, 1, 1, 2)
    assert weights.cfg.execution_dtype == "float16" and weights.cfg.quant == "gguf"
    attention = weights.layers[1].attn
    expected_kv = rank // 4
    assert attention.proj.parts[1].row_ranges == ((expected_kv * 32, 32),)
    experts = weights.layers[0].moe.experts
    assert experts.down.k == 80 and experts.down.row_bytes == 72
    assert experts.down.column_offset == (rank * 80) % 32
    gdn = weights.layers[0].gdn
    expected_values = [h % 2 * 8 + h // 2 for h in range(rank * 2, rank * 2 + 2)]
    assert gdn.a_log is None and gdn.a_neg.transform == "negative_exp"
    assert list(gdn.a_neg.indices) == expected_values
    assert gdn.out.columns == tuple((h * 32, 32) for h in expected_values)
    assert weights.layers[0].attn_hc.down is weights.layers[0].attn_hc.prefill_down
    estimate = weight_byte_plan(path, (rank, 8))
    assert estimate["resident_bytes"] == (sum(m.resident_bytes for m in plan.matrices)
                                          + sum(a.resident_bytes for a in plan.arrays))


def test_config_metadata_uses_reply_eos_separately_from_ple(tmp_path):
    path = tiny_model(tmp_path / "tiny.gguf")
    config = metadata_config(path)
    assert config["eos_token_id"] == 14
    assert config["text_config"]["gguf_ple_eos_token_id"] == 13
    assert config["text_config"]["vocab_size"] == 32


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_model_plan_reuses_inventory_without_changing_json_rope_eos_authority(tmp_path, monkeypatch, prepared, override):
    import json
    from dataclasses import replace
    from tensorfold import gguf_file
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan
    from tensorfold.families.qwen4_exp.cuda.weight_types import Config

    source = tiny_model(tmp_path / "source.gguf")
    selected = source
    if prepared:
        selected = tmp_path / "prepared"
        selected.mkdir()
        raw = metadata_config(source)
        raw["gguf_file"] = str(source)
        raw["text_config"]["rms_norm_eps"] = 0.125
        raw["text_config"]["rope_parameters"]["rope_theta"] = 20000.0
        raw["text_config"]["gguf_ple_eos_token_id"] = 9
        (selected / "config.json").write_text(json.dumps(raw))
    native = Config.read(selected)
    rope = replace(native.rope, theta=30000.0) if override else None
    expected = Config.read(selected, rope=rope, eos_token_ids=[17])
    reads, original = [], gguf_file._shard
    def counted(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(gguf_file, "_shard", counted)
    weights, _planner = model_plan(selected, (3, 8), rope=rope, eos_token_ids=[17],
                                   expert_tensor_parallel_size=4)
    assert reads == [source] * (1 if prepared else 2)
    full = weights.meta["full"]
    assert full == replace(expected, quant="gguf", execution_dtype="float16", ple_eos=13)
    assert full.eos == (17,) and full.ple_eos == 13
    if prepared:
        assert full.eps == 0.125 and native.rope.theta == 20000.0
    if override:
        assert full.rope is rope


def test_split_model_plan_reuses_metadata_inventory_and_preserves_selected_second_entry(tmp_path, monkeypatch):
    from tensorfold import gguf_file
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan
    from gguf_fixture import write

    source = tiny_model(tmp_path / "source.gguf")
    inventory = gguf_file.GGUFCheckpoint.open(source)
    groups = [[], []]
    with source.open("rb") as stream:
        for index, tensor in enumerate(inventory.tensors.values()):
            stream.seek(tensor.offset)
            raw = np.frombuffer(stream.read(tensor.nbytes), dtype=np.uint8)
            groups[index % 2].append((tensor.name, tensor.shape, tensor.ggml_type, raw))
    shards = [tmp_path / f"split-{i:05d}-of-00002.gguf" for i in (1, 2)]
    for index, path in enumerate(shards):
        write(path, groups[index], {**inventory.metadata, "split.no": index, "split.count": 2,
                                    "split.tensors.count": len(inventory.tensors)})
    reads, original = [], gguf_file._shard
    def counted(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(gguf_file, "_shard", counted)
    weights, _ = model_plan(shards[1], (3, 8), expert_tensor_parallel_size=4)
    assert reads == shards * 2
    assert weights.meta["gguf_entrypoint"] == str(shards[1])


def row_runs_reference(indices):
    """Independent integer adjacency oracle, including duplicate/permuted runs."""
    from itertools import groupby

    return tuple((group[0][1], len(group)) for group in (
        list(items) for _key, items in groupby(enumerate(map(int, indices)), lambda pair: pair[1] - pair[0])))


@pytest.mark.parametrize("indices", [range(2, 9), range(3, 3), range(4, 2), range(-8, -2),
                                     range(-3, 4), range(8, 1, -1), range(1, 9, 2),
                                     range(2**70, 2**70 + 3), [3, 1, 1, 2], [2, 3, 8, 9, 8]])
def test_row_ranges_preserve_order_duplicates_and_non_unit_step_semantics(indices):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import _ranges

    expected = row_runs_reference(indices)
    assert _ranges(indices) == expected
    assert _ranges(list(indices)) == expected
    assert _ranges(iter(indices)) == expected


@pytest.mark.parametrize("rows", [range(-1, 2), range(7, 9)])
def test_compact_ranges_keep_the_existing_matrix_bounds(rows):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import Planner

    source = PackedTensor("w", (8, 64), 8, Path("unused"), 0, 8 * 68)
    for values in (rows, list(rows)):
        planner = Planner(SimpleNamespace(tensors={"w": source}))
        with pytest.raises(ValueError, match="invalid GGUF matrix row shard"):
            planner.matrix("w", rows=values, packed=True)


@pytest.fixture(scope="module")
def range_plan_checkpoint(tmp_path_factory):
    return tiny_model(tmp_path_factory.mktemp("range-plans") / "source.gguf", experts=8)


@pytest.mark.parametrize("etp", [2, 4, 8])
@pytest.mark.parametrize("rank", range(8))
def test_target_plan_descriptors_and_aliases_match_expanded_row_oracle(range_plan_checkpoint, monkeypatch, etp, rank):
    from tensorfold.families.qwen4_exp.cuda import gguf_weights as weights

    actual, actual_plan = weights.model_plan(range_plan_checkpoint, (rank, 8), expert_tensor_parallel_size=etp)
    actual_bytes = weights.weight_byte_plan(range_plan_checkpoint, (rank, 8), expert_tensor_parallel_size=etp)
    monkeypatch.setattr(weights, "_ranges", row_runs_reference)
    expected, expected_plan = weights.model_plan(range_plan_checkpoint, (rank, 8), expert_tensor_parallel_size=etp)
    expected_bytes = weights.weight_byte_plan(range_plan_checkpoint, (rank, 8), expert_tensor_parallel_size=etp)
    assert actual == expected and actual_bytes == expected_bytes
    assert actual_plan.matrices == expected_plan.matrices
    assert actual_plan.arrays == expected_plan.arrays and actual_plan.tables == expected_plan.tables
    assert [matrix.byte_segments for matrix in actual_plan.matrices] == [matrix.byte_segments for matrix in expected_plan.matrices]
    for layer in actual.layers:
        for hc in (layer.attn_hc, layer.mlp_hc):
            assert hc.down is hc.prefill_down and hc.up is hc.prefill_up


@pytest.mark.parametrize("tp", [(0, 1), (0, 2), (1, 2), (0, 4), (3, 4)])
def test_smaller_world_default_range_callers_match_integer_oracle(range_plan_checkpoint, monkeypatch, tp):
    from tensorfold.families.qwen4_exp.cuda import gguf_weights as weights

    actual, plan = weights.model_plan(range_plan_checkpoint, tp)
    monkeypatch.setattr(weights, "_ranges", row_runs_reference)
    expected, expected_plan = weights.model_plan(range_plan_checkpoint, tp)
    assert actual == expected and plan.matrices == expected_plan.matrices


@pytest.mark.parametrize("malformed", ["config.json", "hf_quant_config.json"])
def test_model_plan_never_masks_malformed_authoritative_json(tmp_path, malformed):
    import json
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan

    source = tiny_model(tmp_path / "source.gguf")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    raw = metadata_config(source)
    raw["gguf_file"] = str(source)
    (prepared / "config.json").write_text(json.dumps(raw))
    (prepared / malformed).write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match="duplicate"):
        model_plan(prepared)


def test_gguf_sidecar_retains_canonical_duplicate_key_refusal(tmp_path):
    (tmp_path / "config.json").write_text('{"gguf_file":"first.gguf","gguf_file":"second.gguf"}')
    with pytest.raises(ValueError, match="duplicate"):
        resolve_gguf(tmp_path)


def test_ordinary_checkpoint_config_keeps_the_existing_metadata_size_bound(tmp_path):
    import json

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "other", "padding": "x" * ((4 << 20) + 1)}))
    assert resolve_gguf(tmp_path) is None


@pytest.mark.torch
def test_unsupported_parallelism_is_refused_before_allocation(tmp_path):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan

    path = tiny_model(tmp_path / "tiny.gguf")
    with pytest.raises(ValueError, match="TP"):
        model_plan(path, (0, 3))


@pytest.mark.torch
@pytest.mark.parametrize("tables,expected_mib", [(0, 96), (1, 224), (2, 352)])
def test_host_staging_includes_each_overlapping_ple_prefetch(monkeypatch, tables, expected_mib):
    from tensorfold.families.qwen4_exp.cuda import gguf_weights

    prototype = SimpleNamespace(cfg=SimpleNamespace(streams=4, hidden=256, rotary_dim=64),
                                head=SimpleNamespace(n=32), draft_head=None, draft_ids=None, draft_filter=None,
                                meta={"draft_host_id_bytes": 0})
    planner = SimpleNamespace(matrices=[], arrays=[],
                              tables=[SimpleNamespace(source=SimpleNamespace(nbytes=4096)) for _ in range(tables)])
    monkeypatch.setattr(gguf_weights, "model_plan", lambda *a, **k: (prototype, planner))
    plan = gguf_weights.weight_byte_plan("unused")
    assert plan["host_staging_bytes"] == expected_mib << 20
    assert plan["host_resident_bytes"] == tables * 4096


@pytest.mark.torch
def test_packed_expert_provider_requires_explicit_selection_and_keeps_large_prompt_fallback(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts

    bank = GGUFExperts(None, None, None, (), count=2, width=80, dims=256)
    monkeypatch.setattr(bank, "forward_reference", lambda *a: "reference")
    monkeypatch.setattr(bank, "forward_packed", lambda *a: "packed")
    monkeypatch.setattr(bank, "forward_splitk4", lambda *a: "splitk4")
    monkeypatch.setattr(bank, "forward_blocks", lambda *a: "blocks")
    assert bank.forward(SimpleNamespace(shape=(1, 256)), None) == "reference"
    bank.implementation = "packed"
    assert bank.forward(SimpleNamespace(shape=(32, 256)), None) == "packed"
    assert bank.forward(SimpleNamespace(shape=(33, 256)), None) == "reference"
    bank.implementation = "packed_blocks"
    assert bank.forward(SimpleNamespace(shape=(32, 256)), None) == "blocks"
    assert bank.forward(SimpleNamespace(shape=(33, 256)), None) == "reference"
    bank.implementation = "packed_splitk4"
    assert bank.forward(SimpleNamespace(shape=(32, 256)), None) == "splitk4"
    assert bank.forward(SimpleNamespace(shape=(33, 256)), None) == "reference"
    monkeypatch.setattr(bank, "forward_exact", lambda *a: "exact")
    bank.implementation = "packed_exact"
    for rows in (1, 18, 19, 24, 32):
        assert bank.forward(SimpleNamespace(shape=(rows, 256)), None) == "exact"
    assert bank.forward(SimpleNamespace(shape=(33, 256)), None) == "reference"
    bank.implementation = "unqualified"
    with pytest.raises(ValueError, match="implementation"):
        bank.forward(SimpleNamespace(shape=(1, 256)), None)


@pytest.mark.torch
@pytest.mark.parametrize("implementation", ["reference", "packed", "packed_splitk4", "packed_blocks", "packed_exact"])
@pytest.mark.parametrize("rows", [1, 18, 19, 24, 32, 33, 2048])
def test_prompt_phase_keeps_one_reference_provider_for_short_cache_tails_and_large_chunks(
    monkeypatch, implementation, rows,
):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts

    bank = GGUFExperts(None, None, None, (), count=2, width=80, dims=256)
    bank.implementation = implementation
    calls = []
    output, picks = object(), object()
    x = SimpleNamespace(shape=(rows, 256))
    monkeypatch.setattr(bank, "forward_reference", lambda *args: calls.append(args) or output)
    assert bank.forward_prefill(x, picks, output) is output
    assert calls == [(x, picks, output)]


@pytest.mark.torch
def test_packed_ple_uses_existing_read_ahead_and_close_ownership(tmp_path, monkeypatch):
    pytest.importorskip("tensorfold._fd_owner", reason="needs the installed native FD owner")
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFPLETable
    from tensorfold.families.qwen4_exp.read_ahead import ReadAhead

    packed = blocks(20, 8, 32)
    path = tmp_path / "ple.bin"
    path.write_bytes(b"\0" * 64 + packed.tobytes())
    descriptor = PackedTensor("ple", (8, 32), 20, path, 64, packed.nbytes)
    table = ReadAhead(GGUFPLETable(descriptor))
    table.read_ahead([3, 4, 3])
    np.testing.assert_array_equal(table.gather([3, 4, 3]), scalar_reference(packed, 20, 32)[[3, 4, 3]].astype(np.float16))
    calls = []
    monkeypatch.setattr(table.table, "_lock_arrays", lambda arrays: calls.append(arrays) or False)
    assert table.lock() is False
    assert calls == [table.table.values]
    table.close()
    with pytest.raises(ValueError, match="closed"):
        table.gather([0])
    with pytest.raises(ValueError, match="closed"):
        table.lock()
