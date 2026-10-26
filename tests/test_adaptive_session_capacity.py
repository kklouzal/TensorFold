"""Full-window admission keeps actual rank ownership and execution obligations."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda import capacity
from tensorfold.cuda.capacity import Geometry, Weights, fit_sessions
from tensorfold.cuda.geometry import indexed_stream_geometry
from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement


@pytest.mark.parametrize("maximum", [1, 3, 18, 24, 33, 40, 127])
def test_full_context_solver_keeps_exact_endpoint_above_execution_buckets(maximum):
    observed = []

    def geometry(count):
        observed.append(count)
        return Geometry(lambda rows: 47 + count * rows * 7, 5)

    weights, window = Weights(53, 11), 100
    budget = weights.resident + 47 + maximum * 105 * 7
    assert fit_sessions(window, budget, weights, geometry) == maximum
    assert max(observed) <= 2 * maximum + 1  # bounded doubling; never probe billions of synthetic slots.
    assert fit_sessions(window, budget, weights, geometry, ceiling=maximum + 1) == maximum
    assert fit_sessions(window, budget, weights, geometry, ceiling=1) == 1


def test_loading_overlap_and_host_domain_can_limit_capacity_without_changing_context():
    def geometry(n):
        return Geometry(lambda rows: n * rows, 0)
    assert fit_sessions(10, 50, Weights(20, 31), geometry) == 0
    assert fit_sessions(10, 50, Weights(20, 30), geometry) == 3
    assert fit_sessions(10, 500, Weights(20, 30), geometry, additional_fits=lambda n: n <= 7) == 7


@pytest.mark.parametrize("ceiling", [False, 0, -1, 1.5, "auto"])
def test_invalid_ceiling_is_not_coerced(ceiling):
    with pytest.raises(ValueError, match="ceiling"):
        fit_sessions(10, 100, Weights(0, 0), lambda n: Geometry(lambda r: n * r, 0), ceiling=ceiling)


@pytest.fixture
def admission(monkeypatch):
    from tensorfold.cuda import build

    monkeypatch.setattr(build, "refuse_old_gpu", lambda *a: None)
    monkeypatch.setattr(capacity, "floor", lambda p: (7, 0))
    monkeypatch.setattr(capacity, "config", lambda p: {"max_position_embeddings": 100})
    monkeypatch.setattr(capacity, "unified", lambda t: False)
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda: None)
    monkeypatch.setattr(capacity, "host_stream_bytes", lambda: None)
    monkeypatch.setattr(capacity, "page_room", lambda t: None)
    monkeypatch.setattr(capacity, "available_bytes", lambda t: 4500)
    return dict(model_dir=".", requested=None, explicit=False, torch=SimpleNamespace(),
                geometry=Geometry(lambda r: r, 0), transform=lambda n, i: (0, 0),
                weight_estimator=lambda p: (Weights(100, 50), 50),
                session_geometry=lambda text, n: Geometry(lambda r: 200 + n * r, 0))


def test_all_ranks_choose_worst_rank_capacity_without_shortening_native_context(admission):
    packets = []

    def gather(packet):
        packets.append(packet)
        if len(packet) == 8:
            peer = list(packet)
            peer[6] = 18
            return [packet, peer]
        return [packet, [1, 2100, 100, 50, 2000, 2100, 1]]

    result = capacity.admit(**admission, world=2, gather=gather)
    assert result["context_window"] == 100
    plan = result["session_capacity"]
    assert plan["resident_capacity"] == plan["full_context_commitments"] == 18
    assert plan["binding_ranks"] == [1]
    assert plan["evidence"] == "header-plan-unqualified"
    assert plan["ranks"][0]["session_execution_bytes"] == 2000
    assert plan["ranks"][1]["next_session_execution_bytes"] == 2100
    assert len(packets) == 2


def test_administrator_ceiling_and_host_workspace_are_explicit(admission, monkeypatch):
    monkeypatch.setattr(capacity, "host_stream_bytes", lambda: 1000)
    result = capacity.admit(**admission, session_ceiling=40, session_host_workspace=lambda n: n * 100)
    assert result["session_capacity"]["resident_capacity"] == 10
    assert result["session_capacity"]["limiting_resource"] == "host-memory"
    assert result["host_workspace_bytes_estimate"] == 1000
    result = capacity.admit(**admission, session_ceiling=3)
    assert result["session_capacity"]["resident_capacity"] == 3
    assert result["session_capacity"]["limiting_resource"] == "administrator-ceiling"


def test_unfit_complete_session_fails_instead_of_publishing_shorter_context(admission, monkeypatch):
    monkeypatch.setattr(capacity, "available_bytes", lambda t: 399)
    with pytest.raises(ValueError, match="one complete 100-token session"):
        capacity.admit(**admission)


def test_reclaimable_mapped_pages_do_not_silently_shorten_a_fixed_context(admission, monkeypatch):
    monkeypatch.setattr(capacity, "page_room", lambda t: 0)
    admission["weight_estimator"] = lambda p: (Weights(100, 50, 100000), 50)
    result = capacity.admit(**admission)
    assert result["context_window"] == result["session_capacity"]["context_tokens"] == 100
    assert result["mapped_pages_reclaimable"] is True
    assert result["mapped_tables_resident"] is None


TEXT = dict(hidden_size=2560, num_attention_heads=24, num_key_value_heads=8, head_dim=256,
            num_hidden_layers=48, layer_types=["linear_attention"] * 36 + ["full_attention"] * 12,
            linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128,
            linear_value_head_dim=128, linear_conv_kernel_dim=4, vocab_size=248320, hc_count=4,
            num_experts_per_tok=10, moe_intermediate_size=640, indexer_head_dim=128,
            indexer_compress_ratio=4)


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("sessions", [1, 3, 18, 40])
@pytest.mark.parametrize("rows", [127, 128, 129, 513])
def test_unique_history_byte_sum_matches_single_owned_payload_and_metadata(mtp, sessions, rows):
    placement = HistoryPlacement(8, 2, 12)
    geometries = [indexed_stream_geometry(TEXT, sessions, 4, 8, mtp=mtp, world=8, prefill_rows=256,
                 kv_pair=((8, 32, 2),) * 2, full_windows=sessions, execution_rows=32,
                 history_placement=placement, rank=rank) for rank in range(8)]
    ledgers = [g.breakdown_at(rows) for g in geometries]
    caches = 12 + int(mtp)
    physical_rows = (rows + 127) // 128 * 128
    payload = sessions * caches * physical_rows * 2 * 544
    pooled = sessions * caches * ((rows + 3) // 4) * 128 * 2
    # Four raw rows+THW coordinates, protected append version, plus the MTP
    # base/four-row proposal envelope. No completed raw-history allocation.
    bounded = sessions * (caches * 8 + int(mtp) * 8) * (128 * 2 + 3 * 8)
    indexer = pooled + bounded
    metadata = sessions * caches * 8 * ((rows + 127) // 128) * 8
    assert sum(ledger["history"] for ledger in ledgers) == payload + indexer + metadata
    assert all(sum(ledger.values()) == g.bytes_at(rows) for ledger, g in zip(ledgers, geometries))
    assert all(ledger["serving_workspace"] == 0 for ledger in ledgers)
    assert all(ledger["replacement_overlap"] == 0 for ledger in ledgers)


def test_bounded_verification_rows_do_not_release_resident_state_or_histories():
    args = dict(t=TEXT, streams=40, each=4, keep=8, mtp=True, world=8, prefill_rows=256, full_windows=40)
    all_rows = indexed_stream_geometry(**args)
    split = indexed_stream_geometry(**args, execution_rows=32)
    old, new = all_rows.breakdown_at(1028), split.breakdown_at(1028)
    for key in ("session_state", "retained_recovery", "slot_metadata", "history", "replacement_overlap"):
        assert old[key] == new[key]
    assert new["execution_arenas"] < old["execution_arenas"]
    assert new["attention_scratch"] < old["attention_scratch"]


@pytest.mark.torch
def test_actual_sm70_constructor_supplies_owned_full_window_session_geometry(monkeypatch, tmp_path):
    import torch
    import tensorfold.families as families
    import tensorfold.families.qwen4_exp as family
    from tensorfold.cuda import chat_template
    from tensorfold.cuda.sm70 import attention_qualification, execution_policy, family_ops
    from tensorfold.families.qwen4_exp import gguf
    from tensorfold.families.qwen4_exp.cuda import engine, exl3_pack, gguf_admission, padded_linears
    from tensorfold.families.qwen4_exp.cuda.history_placement import MIGRATION_OVERLAP_BYTES

    text = {**TEXT, "num_key_value_heads": 2, "shared_expert_intermediate_size": 640,
            "max_position_embeddings": 262144}
    monkeypatch.setattr(gguf, "resolve_gguf", lambda p: tmp_path / "target.gguf")
    monkeypatch.setattr(chat_template, "prepare_chat_template", lambda *a, **kw: None)
    monkeypatch.setattr(family, "rope_parameters", lambda *a: SimpleNamespace(rope_type="default", context_limit=262144))
    monkeypatch.setattr(families, "read_config", lambda p: {"quantization_config": {"quant_method": "gguf"}})
    monkeypatch.setattr(exl3_pack, "is_exl3", lambda p: False)
    monkeypatch.setattr(capacity, "config", lambda p: text)
    monkeypatch.setattr(execution_policy, "configure", lambda *a, **kw: "reference")
    monkeypatch.setattr(attention_qualification, "runtime_identity", lambda d: {})
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 0))
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", False)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    monkeypatch.setattr(padded_linears, "_ENABLED", False)
    monkeypatch.setattr(gguf_admission, "capacity_options", lambda *a, **kw: {})
    import tensorfold.cuda.comm as comm
    monkeypatch.setattr(comm, "open_comm", lambda *a: SimpleNamespace(barrier=lambda: None, all_gather=None))
    monkeypatch.setattr(capacity, "gather_ints", lambda t, g, v, world=2: [list(v) for _ in range(world)])

    class Inspected(Exception):
        pass

    def inspect(*args, **kwargs):
        assert kwargs["session_ceiling"] == 40
        make = kwargs["session_geometry"]
        first, forty = make(text, 1), make(text, 40)
        a, b = first.breakdown_at(262145), forty.breakdown_at(262145)
        assert first.reserve == forty.reserve == 1
        assert a["migration_overlap_limit"] == b["migration_overlap_limit"] == MIGRATION_OVERLAP_BYTES
        assert b["history"] > a["history"]
        assert a["sm70_native_workspace"] > 0  # Owner buffers remain charged with native kernels disabled.
        assert sum(b.values()) == forty.bytes_at(262145)
        from tensorfold.cuda.geometry import multi_static_domains

        assert kwargs["session_host_workspace"](40) == 8 * (multi_static_domains(text, 40, 1, False)[1] + 8 * 40 * 8)
        raise Inspected

    monkeypatch.setattr(capacity, "admit", inspect)
    from tensorfold.cuda.sm70 import compact_indexer
    monkeypatch.setattr(compact_indexer, "prepare", lambda device: None)
    owner = engine.FlashNextEngine.__new__(engine.FlashNextEngine)
    owner._startup_store = owner._owner_preparation = owner._dflash_pairing = None
    owner._profile_build = None  # __init__ normally establishes this lifecycle before _initialize.
    owner._startup_reads = []
    with pytest.raises(Inspected):
        owner._initialize(tmp_path, depth=0, tp=8, rank=0, master="test", graphs=False,
                          kv_dtype="int8", adaptive_sessions=True, session_ceiling=40)


@pytest.mark.torch
@pytest.mark.parametrize("mtp", [False, True])
def test_host_mirrors_are_charged_to_host_not_device_using_real_table_constructors(mtp, monkeypatch):
    import torch
    from tensorfold.cuda.geometry import multi_static_domains
    from tensorfold.families.qwen4_exp.cuda import attn_multi, gdn_multi

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    owner = SimpleNamespace(device=torch.device("cpu"),
                            layers=[SimpleNamespace(linear=kind == "linear_attention") for kind in TEXT["layer_types"]])
    tables = [gdn_multi.RoundTables(owner, SimpleNamespace(lin=36), 40, 4),
              attn_multi.StepTables(owner, 40, 4, False)]
    if mtp:
        tables.append(attn_multi.StepTables(owner, 40, 4, True))
    device = sum(table.i32.nbytes + table.i64.nbytes for table in tables)
    host = sum(table.h32.nbytes + table.h64.nbytes for table in tables) + 40 * 2 * 4
    assert multi_static_domains(TEXT, 40, 4, mtp) == (device, host)
    old = indexed_stream_geometry(TEXT, 40, 4, 8, mtp=mtp, world=8, full_windows=40)
    adaptive = indexed_stream_geometry(TEXT, 40, 4, 8, mtp=mtp, world=8, full_windows=40,
                                       device_metadata_only=True)
    assert old.needed(1000) - adaptive.needed(1000) == host
