"""Mixed-format loader plans preserve exact storage, topology and family identities."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gguf_fixture import blocks, scalar_reference, tiny_model
from tensorfold.families.qwen4_exp.gguf import PackedTensor, decode_rows, metadata_config, resolve_gguf


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
    assert list(gdn.a_log.indices) == expected_values
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
                                head=SimpleNamespace(n=32), draft_head=None, draft_ids=None,
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
