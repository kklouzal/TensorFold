"""Source admission units and actual selected-tokenizer presentation boundaries."""

from types import SimpleNamespace
import sys

import pytest

from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.families.qwen4_exp.cuda.retention_resources import (
    continuation_payload_bytes, finalize_retention_resources, retained_constraint_reason,
    HostBaseline, native_sync_control_payload, native_sync_data_payload, require_native_payload_runtime,
    retained_recovery_extra, retention_device_resources,
    retention_header_resources, tokenizer_presentation_bound,
)


def tokenizer():
    native = pytest.importorskip("tokenizers")
    if native.__version__ != "0.23.3":
        pytest.skip("the selected tier-resource frontend is tokenizers 0.23.3")
    result = native.Tokenizer(native.models.BPE({"a": 0, "b": 1, "\u0100": 2}, []))
    result.decoder = native.decoders.ByteLevel()
    result.add_tokens(["\U0010ffff" * 5])
    return result


def test_global_policy_reservation_is_not_multiplied_by_tp():
    policy = RetentionPolicy("ram", 5, 60, 1 << 30, 4096, 65536, 1 << 20)
    value = retention_header_resources(policy, 8)
    native = native_sync_data_payload(world=8, chunk_bytes=4096)
    control = native_sync_control_payload(world=8, chunk_bytes=4096, ram=True)
    assert value.global_host_bytes == ((1 << 30) + 65536 + native["global_native_data_payload_bytes"]
        + control["global_native_control_value_bytes"] + control["global_native_rpc_value_bytes"])
    assert native["pinned_ordinary_bytes_already_in_progress"] == 65536
    assert value.per_rank_gpu_overlap_bytes == 1 << 20
    assert value.record()["world"] == 8
    ordinary = retention_header_resources(None, 8)
    assert ordinary.global_host_native_data_bytes == 0
    assert ordinary.global_host_bytes == (ordinary.global_host_native_control_bytes + ordinary.global_host_native_rpc_bytes)
    assert ordinary.global_host_native_rpc_bytes > 0
    assert retention_header_resources(RetentionPolicy("gpu", 5, 60), 8).per_rank_gpu_overlap_bytes == 0


def test_native_control_and_rpc_are_global_and_do_not_re_add_manifest_backing():
    without = native_sync_control_payload(world=8, chunk_bytes=0, ram=False, slots=2)
    ram = native_sync_control_payload(world=8, chunk_bytes=4096, ram=True, slots=2)
    assert without["worker_frame_bytes"] == without["daemon_worker_value_bytes"] == 0
    assert ram["worker_frame_bytes"] > 0 and ram["daemon_worker_value_bytes"] > 0
    assert ram["global_native_rpc_value_bytes"] == without["global_native_rpc_value_bytes"]
    peers = 7
    frame = without["rpc_command_bytes"]
    assert without["global_native_rpc_value_bytes"] == (max((2 * peers + 1) * frame, peers * frame)
        + peers * 3 * frame + peers * 2 * frame)
    assert ram["global_native_control_value_bytes"] >= ram["shared_daemon_handler_delta_bytes"]
    assert "already reserved manifest backing" in ram["excluded"]


def test_native_data_extra_bounds_selected_vector_growth_without_repaying_python_or_pinned():
    from tensorfold.cuda.store_fields import MAX_KEY_BYTES

    def allocated_peak(payload):
        # Independent selected-STL recurrence from the pinned source evidence:
        # old and new allocations coexist during a growing range insertion.
        size, capacity, peak = 1, 32, 32
        for count in (8, MAX_KEY_BYTES, 8, payload):
            if size + count > capacity:
                grown = size + max(size, count)
                peak = max(peak, capacity + grown)
                capacity = grown
            size += count
            peak = max(peak, capacity)
            if size >= 1440:
                size = 0
        return peak
    for world in (1, 2, 8):
        for chunk in (1, 37, 1439, 1440, 8191, 8192, 8193, 1 << 20, 8 << 20):
            bound = native_sync_data_payload(world=world, chunk_bytes=chunk)
            for open_waves in range(world + 1):
                for publishers in range(open_waves + 1):
                    ordinary_native = (world + publishers + open_waves + 1) * chunk
                    assert ordinary_native + publishers * allocated_peak(chunk) <= bound["global_native_data_payload_bytes"]
            assert bound["pinned_ordinary_bytes_already_in_progress"] == 2 * world * chunk
            assert bound["native_chunk_payload_bytes"] == (4 * world + 1) * chunk


def test_native_payload_bound_verifies_actual_selected_abi_and_rejects_other_torch(monkeypatch):
    torch = pytest.importorskip("torch")
    result = require_native_payload_runtime()
    assert result["torch_git"] == "5ab14acbced0c33e6b6e8fdbc5b541b57487f7e1"
    assert result["libstdcxx_sha256"] == "1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11"
    assert result["python_implementation"] == "cpython" and result["python_version"] == [3, 12, 3]
    monkeypatch.setattr(torch.version, "git_version", "other-native-build")
    with pytest.raises(ValueError, match="selected Torch"):
        require_native_payload_runtime()


def test_native_payload_bound_rejects_different_actual_runtime_content(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from tensorfold.families.qwen4_exp.cuda import execution_identity

    wrong = tmp_path / "libstdc++.so.6.fake"
    wrong.write_bytes(b"not the selected ELF")
    monkeypatch.setattr(execution_identity, "_small", lambda *_args: f"0-1 r--p 0 00:01 1 {wrong}\n".encode())
    with pytest.raises(ValueError, match=r"C\+\+ runtime content"):
        require_native_payload_runtime()


def test_native_payload_bound_requires_finite_existing_hard_fd_limit(monkeypatch):
    pytest.importorskip("torch")
    import resource

    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (1024, resource.RLIM_INFINITY))
    with pytest.raises(ValueError, match="finite existing hard descriptor"):
        require_native_payload_runtime()


@pytest.mark.parametrize("variant", ["implementation", "version"])
def test_new_unique_native_payload_rejects_unproved_python_layout_before_native_reads(monkeypatch, variant):
    from tensorfold.families.qwen4_exp.cuda import execution_identity

    def forbidden(*_args):
        pytest.fail("unsupported Python must refuse before process/native identity reads")
    monkeypatch.setattr(execution_identity, "_small", forbidden)
    if variant == "version":
        monkeypatch.setattr(sys, "version_info", (3, 12, 4))
    else:
        values = vars(sys.implementation).copy()
        values["name"] = "pypy"
        monkeypatch.setattr(sys, "implementation", SimpleNamespace(**values))
    with pytest.raises(ValueError, match="selected CPython3.12.3"):
        require_native_payload_runtime()


def test_recovery_allowance_reuses_existing_authority_without_baseline_double_charge():
    # One global prefix budget, two slots and four mandatory per-slot points.
    assert retained_recovery_extra((8 + 2 * 4) * 91, slots=2, keep=8, points=4) == 8 * 91
    assert retained_recovery_extra((8 + 1 * 4) * 91, slots=1, keep=8, points=4) == 0
    assert retained_recovery_extra(0, slots=2, keep=0, points=0) == 0
    with pytest.raises(ValueError, match="geometry"):
        retained_recovery_extra(18, slots=2, keep=8, points=4)


def test_runtime_device_limits_and_header_delta_share_actual_boundary_credit():
    from tensorfold.families.qwen4_exp.cuda.media_mtp import boundary_resources

    slots, keep, points = 3, 2, 4
    policy = RetentionPolicy("gpu", slots, 10)
    versions = keep + slots * points
    plain = retention_device_resources({"retained_recovery": versions * 1000},
        policy=policy, slots=slots, keep=keep, points=points, mtp=True, hidden_width=10)
    assert plain["header_extra_device_bytes"] == 3 * 10 * 2
    assert plain["limits"] == {"recovery_pool_bytes": 3 * 2 * 1000,
                               "pending_hc_bytes": 3 * 20, "snapshot_bytes": 1000}
    boundary = boundary_resources(hidden_width=10, slots=slots, keep=keep, points=points)
    media = retention_device_resources({"retained_recovery": versions * 1000,
        "media_mtp_boundary": boundary["additional_device_bytes"]},
        policy=policy, slots=slots, keep=keep, points=points, mtp=True, hidden_width=10, media_inputs_enabled=True)
    assert media["text_live_hc_header_extra_bytes"] == 0
    assert media["header_extra_device_bytes"] == (3 * 2 - 2) * 24
    assert media["limits"] == {"recovery_pool_bytes": 3 * 2 * (1000 + 24),
                               "pending_hc_bytes": 3 * (20 + 24), "snapshot_bytes": 1024}
    with pytest.raises(ValueError, match="boundary header credit"):
        retention_device_resources({"retained_recovery": versions * 1000}, policy=policy, slots=slots, keep=keep,
            points=points, mtp=True, hidden_width=10, media_inputs_enabled=True)
    df = retention_device_resources({"retained_recovery": versions * 1000,
        "dflash_recovery_snapshots": versions * 2000}, policy=policy, slots=slots, keep=keep,
        points=points, mtp=False, hidden_width=0)
    assert df["limits"] == {"recovery_pool_bytes": 3 * 2 * 3000,
                            "pending_hc_bytes": 0, "snapshot_bytes": 3000}
    assert df["header_extra_device_bytes"] == 0
    optional = retention_device_resources({"retained_recovery": versions * 1000},
        policy=None, slots=slots, keep=keep, points=points, mtp=True, hidden_width=10)
    assert optional["limits"]["recovery_pool_bytes"] == keep * 1000
    assert optional["header_extra_device_bytes"] == slots * 20


def test_actual_added_tokens_and_lossy_bytelevel_replacement_are_bounded():
    tok = tokenizer()
    bound = tokenizer_presentation_bound(tok)
    assert bound.piece_utf8_bytes == 20  # actual added token, not an assumed 256.
    assert bound.decoded_utf8_bytes_per_token == 60
    for ids in ([0], [2], [2, 2], [0, 3, 2], [3] * 17):
        assert len(tok.decode(ids, skip_special_tokens=False).encode()) <= len(ids) * 60
    assert tokenizer_presentation_bound(tok) == bound


def test_vocab_compatible_replace_decoder_does_not_inherit_bytelevel_authority():
    tok = tokenizer()
    original = tokenizer_presentation_bound(tok)
    native = pytest.importorskip("tokenizers")
    tok.decoder = native.decoders.Sequence([native.decoders.ByteLevel(), native.decoders.Replace("a", "b" * 1000)])
    assert tok.get_vocab_size() == original.vocab_size
    expanded = tokenizer_presentation_bound(tok)
    assert expanded.tokenizer_identity != original.tokenizer_identity
    assert expanded.decoded_utf8_bytes_per_token > original.decoded_utf8_bytes_per_token
    for ids in ([0], [0] * 23, [0, 3, 2]):
        assert len(tok.decode(ids, skip_special_tokens=False).encode()) <= (
            len(ids) * expanded.decoded_utf8_bytes_per_token + expanded.decoded_utf8_fixed_bytes)


@pytest.mark.parametrize("kind", ["none", "wordpiece", "empty-bpe", "empty-ctc", "metaspace", "strip",
                                   "fallback", "empty-regex", "fuse-replace"])
def test_builtin_decoder_affine_bound_covers_multibyte_empty_and_cross_token_cases(kind):
    tok = tokenizer()
    import tokenizers as native

    tok.add_tokens(["<0xFF>", "<0xC3>", "<0x28>", "\u2581a", "##b"])
    factories = {
        "none": lambda: None,
        "wordpiece": lambda: native.decoders.WordPiece(prefix="##", cleanup=True),
        "empty-bpe": lambda: native.decoders.BPEDecoder(suffix=""),
        "empty-ctc": lambda: native.decoders.CTC(pad_token="", word_delimiter_token="", cleanup=True),
        "metaspace": lambda: native.decoders.Metaspace(replacement="\u2581", prepend_scheme="never"),
        "strip": lambda: native.decoders.Strip(content="a", left=0, right=0),
        "fallback": lambda: native.decoders.ByteFallback(),
        "empty-regex": lambda: native.decoders.Sequence([native.decoders.ByteLevel(),
            native.decoders.Replace(native.Regex(""), "\u00e9\U0010ffff")]),
        "fuse-replace": lambda: native.decoders.Sequence([native.decoders.Fuse(),
            native.decoders.Replace(native.Regex("ab"), "\u00e9" * 17), native.decoders.ByteLevel()]),
    }
    tok.decoder = factories[kind]()
    bound = tokenizer_presentation_bound(tok)
    for ids in ([], [0], [0, 1] * 31, [4, 5, 6], [3, 7, 8, 2]):
        actual = len(tok.decode(ids, skip_special_tokens=False).encode())
        assert actual <= len(ids) * bound.decoded_utf8_bytes_per_token + bound.decoded_utf8_fixed_bytes
    assert bound.working_utf8_bytes_per_token >= bound.decoded_utf8_bytes_per_token


def test_whole_context_continuation_includes_real_python_int_lists_copy_index_and_output():
    small = continuation_payload_bytes(context=64, vocab_size=248064, presentation=768,
                                       max_body_bytes=32 << 20, copy_enabled=True)
    large = continuation_payload_bytes(context=128, vocab_size=248064, presentation=768,
                                       max_body_bytes=32 << 20, copy_enabled=True)
    assert small["continuation_lists"] > 7 * 64 * (sys.getsizeof(248063) + 8)
    assert small["mailbox"] > 64 * 4
    assert small["copy_index"] > 64 * 16
    assert small["presentation_strings"] > 64 * 768
    assert small["input_prompt"] == 0  # actual InputClaim independently owns this obligation.
    assert large["project_payload_bytes"] > small["project_payload_bytes"]


def test_native_constraint_gate_preserves_unconfigured_and_nonretained_gpu_paths():
    assert retained_constraint_reason(SimpleNamespace(policy=None), logical_retained=True) is None
    gpu = SimpleNamespace(policy=RetentionPolicy("gpu", 1, 10))
    assert retained_constraint_reason(gpu, logical_retained=False) is None
    assert "native matcher" in retained_constraint_reason(gpu, logical_retained=True)
    ram = SimpleNamespace(policy=RetentionPolicy("ram", 1, 10, 1 << 30, 4096, 8192, 0))
    assert "native matcher" in retained_constraint_reason(ram, logical_retained=False)
    with pytest.raises(ValueError, match="trusted logical"):
        retained_constraint_reason(gpu, logical_retained=1)


def test_finalizer_checks_global_remaining_room_and_freezes_actual_frontend(monkeypatch):
    from tensorfold.cuda import capacity

    calls = []
    def admit(total, *, materialized_owned_bytes):
        calls.append((total, materialized_owned_bytes))
        return {"remaining": total - materialized_owned_bytes, "release_qualified": False}
    monkeypatch.setattr(capacity, "admit_host_commitment", admit)
    catalog = SimpleNamespace(policy=RetentionPolicy("ram", 2, 10, 1 << 30, 4096, 65536, 0),
        world=8, states=[SimpleNamespace(limit=128)], _manifest_bytes=1024,
        _buffer=SimpleNamespace(nbytes=4096, is_pinned=lambda: True), _ram=None, check_health=lambda: None,
        evidence="source-candidate", has_retained=False, prefix_limit=2, minimum_future_bytes=1 << 20,
        host_resources=HostBaseline(1000, 500), _cohort=SimpleNamespace(multi=SimpleNamespace(copy=False)))
    record = finalize_retention_resources(catalog, tokenizer=tokenizer(), max_pending_requests=3,
        max_waiting_input_bytes=100000, max_http_connections=4)
    assert calls == [(record["global_host_commitment_bytes"], 500 + 8 * 4096)]
    assert record["header"]["global_host_backing_bytes"] == 1 << 30
    assert record["global_waiting_input_bytes"] == 100000
    assert record["authority"]["release_qualified"] is False
    assert catalog.host_resource_envelope is record
    with pytest.raises(ValueError, match="immutable"):
        finalize_retention_resources(catalog, tokenizer=tokenizer(), max_pending_requests=4,
            max_waiting_input_bytes=100000, max_http_connections=4)
    assert len(calls) == 1


def test_host_baseline_cannot_credit_virtual_or_overdrawn_bytes():
    assert HostBaseline(100, 0).materialized_host_bytes == 0
    with pytest.raises(ValueError, match="exceeds"):
        HostBaseline(100, 101)


def test_resource_arithmetic_refuses_unrepresentable_commitments_before_allocation():
    with pytest.raises(ValueError, match="resource sum"):
        retained_recovery_extra(sys.maxsize, slots=sys.maxsize, keep=sys.maxsize, points=0)
    with pytest.raises(ValueError, match="host-sized"):
        retention_header_resources(None, True)
