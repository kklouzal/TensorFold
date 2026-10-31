"""GGUF MTP vocabulary selection preserves global IDs and the target head."""

from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from gguf_fixture import scalar_reference, tiny_model

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.cuda.direct_read import Reader  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import gguf_weights  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weight_types import draft_token_ids  # noqa: E402


@pytest.fixture
def checkpoint(tmp_path):
    return tiny_model(tmp_path / "tiny.gguf")


@pytest.mark.parametrize("selection", ["implicit", "default", None, 0, 9, "selected-ids.txt"])
def test_family_factory_preserves_engine_default_and_explicit_draft_vocabulary(checkpoint, monkeypatch, selection):
    """Exercise the serving factory boundary that direct engine tests bypass."""
    from tensorfold.families import qwen4_exp

    module = ModuleType("tensorfold.families.qwen4_exp.cuda.engine")
    calls, result = [], object()

    def construct(directory, **options):
        calls.append((directory, options))
        return result

    module.FlashNextEngine = construct
    monkeypatch.setitem(sys.modules, module.__name__, module)
    options = {} if selection == "implicit" else {"draft_vocab": selection}
    assert qwen4_exp.cuda_engine(checkpoint, drafter="/stock-mtp", mtp_drafts=3,
                                kv_dtype="int8", **options) is result
    assert len(calls) == 1 and calls[0][0] == checkpoint
    received = calls[0][1]
    assert received["mtp_model_dir"] == Path("/stock-mtp") and received["depth"] == 3
    if selection == "implicit":
        assert "draft_vocab" not in received  # the engine's existing default remains authoritative
    else:
        assert received["draft_vocab"] == selection


@pytest.mark.parametrize("rank", range(8))
def test_sparse_draft_plan_splits_global_ids_and_charges_only_new_head_and_mapping(checkpoint, rank):
    ids = np.array([0, 2, 4, 6, 9, 17, 18, 22, 29, 31], dtype=np.int64)
    original, original_plan = gguf_weights.model_plan(checkpoint, (rank, 8))
    selected, plan = gguf_weights.model_plan(checkpoint, (rank, 8), mtp=True, draft_ids=ids)
    local = np.array_split(ids, 8)[rank]
    np.testing.assert_array_equal(selected.draft_ids, local)
    assert selected.draft_head.row_ranges == gguf_weights._ranges(local)
    assert selected.head == original.head and selected.cfg == original.cfg
    assert selected.draft_head.source is selected.head.source
    assert len(plan.matrices) == len(original_plan.matrices) + 1
    baseline = gguf_weights.weight_byte_plan(checkpoint, (rank, 8))
    budget = gguf_weights.weight_byte_plan(checkpoint, (rank, 8), mtp=True, draft_ids=ids)
    assert budget["draft_head_bytes"] == local.size * 256 * 2
    assert budget["draft_id_bytes"] == local.nbytes
    assert budget["resident_bytes"] - baseline["resident_bytes"] == local.size * (256 * 2 + 8)
    assert budget["host_staging_bytes"] - baseline["host_staging_bytes"] == ids.nbytes + local.nbytes
    if rank == 3:
        assert local[0] == 17 and original.head.row_ranges == ((12, 4),)


@pytest.mark.parametrize("selection", [None, 0, 9, 100, "default", "file"])
def test_draft_vocab_uses_the_existing_default_file_integer_and_full_contract(checkpoint, tmp_path, selection):
    if selection == "file":
        path = tmp_path / "ids.txt"
        path.write_text("31 0 17 7 7 1000 # repeated and upper IDs are ignored\n")
        selection = str(path)
    expected = draft_token_ids(selection, 32)
    selected, _ = gguf_weights.model_plan(checkpoint, mtp=True, draft_vocab=selection)
    if expected is None:
        assert selected.draft_head is None and selected.draft_ids is None
    else:
        np.testing.assert_array_equal(selected.draft_ids, expected)


@pytest.mark.parametrize("selection", [-1, True, 1.5, "", "empty", "upper", "negative"])
def test_invalid_or_empty_draft_selections_fail_before_payload_reads(checkpoint, tmp_path, selection):
    if isinstance(selection, str) and selection:
        path = tmp_path / "ids.txt"
        path.write_text({"empty": "# no IDs\n", "upper": "32 9999", "negative": "-1"}[selection])
        selection = str(path)
    with pytest.raises((TypeError, ValueError), match="draft vocab"):
        gguf_weights.model_plan(checkpoint, mtp=True, draft_vocab=selection)


def test_every_rank_requires_a_draft_row_but_no_mtp_ignores_unused_selection(checkpoint, tmp_path):
    with pytest.raises(ValueError, match="one ID per"):
        gguf_weights.model_plan(checkpoint, (0, 8), mtp=True, draft_vocab=7)
    original, _ = gguf_weights.model_plan(checkpoint, (0, 8))
    unused, _ = gguf_weights.model_plan(checkpoint, (0, 8), draft_vocab=str(tmp_path / "missing"))
    assert unused.head == original.head and unused.draft_ids is None and unused.draft_head is None
    assert (gguf_weights.weight_byte_plan(checkpoint, (0, 8), draft_vocab=-1)
            == gguf_weights.weight_byte_plan(checkpoint, (0, 8)))


@pytest.mark.parametrize("ids", [np.array([], dtype=np.int64), np.array([1], dtype=np.int32),
                                np.array([2, 1], dtype=np.int64), np.array([1, 1], dtype=np.int64),
                                np.array([-1], dtype=np.int64), np.array([32], dtype=np.int64)])
def test_preparsed_handshake_ids_cannot_bypass_global_id_validation(checkpoint, ids):
    with pytest.raises(ValueError, match="sorted unique global int64"):
        gguf_weights.model_plan(checkpoint, mtp=True, draft_ids=ids)


def test_global_draft_rows_gather_in_bounded_spans_and_match_independent_cpu_format(checkpoint, monkeypatch):
    ids = np.array([0, 2, 3, 7, 13, 19, 30, 31], dtype=np.int64)
    weights, _ = gguf_weights.model_plan(checkpoint, mtp=True, draft_ids=ids)
    plan = weights.draft_head
    source = plan.source
    before = Path(checkpoint).read_bytes()[source.offset:source.offset + source.nbytes]
    raw = np.frombuffer(before, np.uint8).reshape(source.rows, source.row_bytes)
    oracle = scalar_reference(raw, source.ggml_type, source.shape[-1])[ids].astype(np.float16)
    monkeypatch.setattr(gguf_weights, "STAGING_BYTES", 3 * source.row_bytes)
    monkeypatch.setattr(gguf_weights, "DECODE_BYTES", 2 * 256 * 2)
    reader = Reader()
    reader.direct = False
    reads = []
    original = reader.read

    def read(path, offset, count, *args, **kwargs):
        reads.append((offset, count))
        return original(path, offset, count, *args, **kwargs)

    monkeypatch.setattr(reader, "read", read)
    try:
        result = gguf_weights._draft_dense(plan, reader, "cpu")
    finally:
        reader.close()
    np.testing.assert_array_equal(result.weight.numpy().view(np.uint16), oracle.view(np.uint16))
    assert len(reads) > 1 and max(count for _, count in reads) <= 3 * source.row_bytes
    assert all(source.offset <= at and at + count <= source.offset + source.nbytes for at, count in reads)
    assert Path(checkpoint).read_bytes()[source.offset:source.offset + source.nbytes] == before


def test_common_loader_passes_selection_and_handshaked_ids_to_gguf(checkpoint, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import weights

    ids = np.array([0, 7, 17, 31], dtype=np.int64)
    result, calls = object(), []
    monkeypatch.setattr(gguf_weights, "load_gguf", lambda *args, **kwargs: calls.append((args, kwargs)) or result)
    assert weights.load(checkpoint, mtp=True, draft_vocab="default", _draft_ids=ids) is result
    assert calls[0][1]["draft_vocab"] == "default" and calls[0][1]["_draft_ids"] is ids


def test_read_only_handshaked_ids_remain_authoritative_without_reopening_the_selection(checkpoint, tmp_path):
    ids = np.array([0, 2, 4, 6, 9, 17, 18, 22, 29, 31], dtype=np.int64)
    ids.setflags(write=False)
    before = ids.tobytes()
    plan, _ = gguf_weights.model_plan(checkpoint, (3, 8), mtp=True,
                                     draft_vocab=str(tmp_path / "selection-no-longer-present"), draft_ids=ids)
    np.testing.assert_array_equal(plan.draft_ids, [17])
    assert not plan.draft_ids.flags.writeable and ids.tobytes() == before


def test_admission_charges_pruned_head_ids_and_stock_mtp_before_loading(checkpoint, tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import gguf_admission, sm70_mtp_weights

    ids = np.array([0, 2, 4, 6, 9, 17, 18, 22, 29, 31], dtype=np.int64)
    plan = gguf_weights.weight_byte_plan(checkpoint, (3, 8), mtp=True, draft_ids=ids)
    monkeypatch.setattr(sm70_mtp_weights, "byte_plan", lambda *args, **kwargs: {
        "resident_bytes": 1234, "staging_bytes": 1024, "host_staging_bytes": 2048, "workspace_bytes": 4096,
    })
    # This oracle covers the independent floating companion's legacy sliced
    # head. The target's own directory is not an MTP GGUF companion.
    options = gguf_admission.capacity_options(checkpoint, 3, 8, mtp=True,
                                              mtp_model_dir=tmp_path / "stock-mtp", draft_ids=ids)
    weights, host_staging = options["weight_estimator"](checkpoint)
    assert weights.resident == plan["resident_bytes"] + 1234
    assert weights.staging == plan["staging_bytes"]
    assert host_staging == plan["host_staging_bytes"]
    assert options["device_workspace_bytes"] == plan["workspace_bytes"] + 4096
