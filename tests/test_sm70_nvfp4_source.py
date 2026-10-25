"""Owned source boundary, TP layout and CPU construction controls, not GPU qualification."""

import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.cuda import precision
from tensorfold.cuda.sm70.nvfp4_layout import (EXPERT_DEVICE, EXPERT_FIELDS, EXPERT_WORDS, LINEAR_REFERENCE,
                                             LINEAR_WMMA, expert_workspace_bytes, render_header)
from tensorfold.families.qwen4_exp.cuda import sm70_nvfp4 as runtime
from tensorfold.families.qwen4_exp.cuda.sm70_nvfp4_weights import byte_plan, load, model_plan, _validate_nvfp4
from tensorfold.families.qwen4_exp.cuda.sm70_weight_manifest import prepare, load as load_manifest, required_source_files
from sm70_nvfp4_fixture import write


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory, tensors = write(tmp_path_factory.mktemp("sm70-nvfp4-source"))
    document = prepare(directory, repository="controlled/source", revision="fixed-control", norm_offset=1)
    manifest = directory / "family.json"
    manifest.write_text(json.dumps(document))
    return directory, manifest, tensors


def test_owned_generated_interface_is_fresh():
    from tensorfold.cuda.sm70 import nvfp4_layout

    assert nvfp4_layout.__file__
    assert __import__("pathlib").Path(nvfp4_layout.__file__).with_suffix(".cuh").read_text() == render_header()


def test_manifest_freezes_actual_inputs_and_refuses_changed_cached_identity(source):
    directory, manifest, tensors = source
    admitted = load_manifest(manifest, directory)
    assert set(admitted.infos) == set(tensors)
    assert manifest.resolve() in required_source_files(manifest, directory)
    assert directory / "model.safetensors" in required_source_files(manifest, directory)
    admitted.check_cached(manifest, directory)
    selected = directory / "tokenizer.json"
    before = selected.stat()
    try:
        os.utime(selected, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
        with pytest.raises(ValueError, match="changed"):
            admitted.check_versions(directory)
    finally:
        os.utime(selected, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_cached_manifest_refuses_same_size_source_write_with_restored_mtime(tmp_path):
    directory, _ = write(tmp_path / "source")
    body = prepare(directory, repository="controlled/source", revision="fixed-control", norm_offset=1)
    manifest = directory / "family.json"
    manifest.write_text(json.dumps(body))
    admitted = load_manifest(manifest, directory)
    selected = directory / "model.safetensors"
    before = selected.stat()
    with selected.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        original = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes((original[0] ^ 1,)))
    os.utime(selected, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = selected.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    assert after.st_ctime_ns != before.st_ctime_ns
    with pytest.raises(ValueError, match="prepared SM70 source changed"):
        admitted.check_cached(manifest, directory)


@pytest.mark.parametrize("field,value", [("transforms", ["hadamard"]), ("qualification", "qualified"),
                                        ("schema_version", True), ("norm_offset", 1.0)])
def test_unknown_or_unverified_recipe_refuses_before_source_payload_hash(source, tmp_path, monkeypatch, field, value):
    directory, manifest, _ = source
    body = json.loads(manifest.read_text())
    body[field] = value
    changed = tmp_path / "invalid.json"
    changed.write_text(json.dumps(body))
    monkeypatch.setattr("tensorfold.families.qwen4_exp.cuda.sm70_weight_manifest._digest",
                        lambda *args: pytest.fail("invalid declaration hashed source payload"))
    with pytest.raises(ValueError):
        load_manifest(changed, directory)


def test_tp8_shards_preserve_original_group16_bits_and_shared_factor_bank(source):
    directory, manifest, tensors = source
    prototype, planner = model_plan(directory, manifest, (7, 8))
    assert prototype.cfg.kv_heads == 1 and prototype.cfg.heads == 1
    assert prototype.meta["full"].kv_heads == 2
    with precision.using(precision.FULL, asked=True):
        weights = load(directory, manifest, "cpu", tp=(7, 8), _plan=(prototype, planner))
    gate, _, down = weights.layers[0].moe.experts.parts[0]
    prefix = "model.layers.0.mlp.experts.0."
    assert torch.equal(gate.codes, torch.from_numpy(tensors[prefix + "gate_proj.weight"][0][112:128].copy()))
    assert torch.equal(down.codes, torch.from_numpy(tensors[prefix + "down_proj.weight"][0][:, 56:64].copy()))
    factors = [part.global_scale for layer in weights.layers for group in layer.moe.experts.parts
               for part in group if isinstance(part, runtime.PackedLinear)]
    assert len(factors) == 12 and len({t.untyped_storage().data_ptr() for t in factors}) == 1
    assert factors[0].untyped_storage().nbytes() == 12 * 4
    assert torch.equal(weights.layers[0].gdn.a_log, torch.from_numpy(tensors["model.layers.0.linear_attn.A_log"][0][42:48]))
    assert torch.equal(weights.layers[0].gdn.norm, torch.from_numpy(tensors["model.layers.0.linear_attn.norm.weight"][0]).half())
    expected_gamma = torch.from_numpy(tensors["model.layers.1.self_attn.q_norm.weight"][0]) + 1
    assert torch.equal(weights.layers[1].attn.q_scale, expected_gamma)
    source_k = torch.from_numpy(tensors["model.layers.1.self_attn.k_proj.weight"][0][256:512].copy())
    assert torch.equal(weights.layers[1].attn.proj.parts[1].weight, source_k)


@pytest.mark.parametrize("family", [LINEAR_REFERENCE, LINEAR_WMMA])
def test_workspace_payload_ledger_matches_construction_without_a_history_mirror(family):
    ws = runtime.Workspace("cpu", width=80, dims=2560, selection_rows=1024, slots=11, experts=513, kernel_family=family)
    owned = sum(t.numel() * t.element_size() for t in ws.device_tensors())
    assert owned == runtime.workspace_bytes(width=80, dims=2560, kernel_family=family)
    host = ws.host_selection.numel() * 4 + ws.host_ids.numel() * 8
    host += sum(t.nbytes for t in (ws.members, ws.counts, ws.starts, ws.cursor))
    assert host == runtime.host_workspace_bytes(1024, 11, 513)
    assert all(t.numel() < 262144 * 256 for t in ws.device_tensors())


@pytest.mark.parametrize("family", [LINEAR_REFERENCE, LINEAR_WMMA])
def test_pre_capacity_warm_covers_both_linear_output_entrypoints(family, monkeypatch):
    ws = runtime.Workspace("cpu", kernel_family=family)
    calls = []
    def matmul(part, x, *, out=None, f32=False):
        calls.append(("packed" if isinstance(part, runtime.PackedLinear) else "float", f32))
        assert x.shape == (1, 16) and x.dtype == torch.float16 and x.device == ws.device
        return torch.empty((len(x), part.n), dtype=torch.float32 if f32 else torch.float16)
    monkeypatch.setattr(runtime.PackedLinear, "matmul", matmul)
    monkeypatch.setattr(runtime.FloatLinear, "matmul", matmul)
    runtime.warm(ws)
    assert calls == [("packed", False), ("float", False), ("packed", True), ("float", True)]
    assert ws.kernel_family == family


def test_device_group_plan_storage_and_descriptors_match_imported_tensor_owners(source, tmp_path):
    directory, manifest, _ = source
    body = json.loads(manifest.read_text())
    body["numerical"]["expert_kernel_family"] = EXPERT_DEVICE
    candidate = tmp_path / "device.json"
    candidate.write_text(json.dumps(body))
    prototype, planner = model_plan(directory, candidate, (7, 8))
    predicted = byte_plan(prototype, planner)
    with precision.using(precision.FULL, asked=True):
        weights = load(directory, candidate, "cpu", tp=(7, 8), _plan=(prototype, planner))
    bank = weights.layers[0].moe.experts
    assert bank.descriptor.shape == (3, 3, EXPERT_WORDS)
    record = dict(zip(EXPERT_FIELDS, bank.descriptor[0, 0].tolist()))
    gate = bank.parts[0][0]
    assert record["DATA"] == gate.codes.data_ptr()
    assert record["GLOBAL"] == gate.global_scale.data_ptr()
    assert record["DATA_BYTES"] == gate.codes.numel()
    assert record["SCALE_ELEMENTS"] == gate.scales.numel()
    assert predicted["expert_descriptor_bytes"] == 2 * 3 * 3 * EXPERT_WORDS * 8
    ws = bank.workspace
    assert sum(t.numel() * t.element_size() for t in ws.expert_tensors) == expert_workspace_bytes(1024, 3, 3, 16)
    assert runtime.host_workspace_bytes(1024, 3, 3, expert_family=EXPERT_DEVICE) == 0


@pytest.mark.parametrize("family", [runtime.EXPERT_REFERENCE, EXPERT_DEVICE])
@pytest.mark.parametrize("source_kind", ["codes", "scales", "global", "float"])
def test_expert_output_refuses_retained_source_alias_before_execution(family, source_kind, monkeypatch):
    ws = runtime.Workspace("cpu", width=16, dims=16, selection_rows=1, slots=1, experts=1,
                           expert_family=family)
    # Each typed operand is a narrow view into a larger retained allocation.
    # The output can overlap an otherwise unused portion of that allocation;
    # the immutable whole-storage contract still forbids the write.
    codes = torch.zeros(512, dtype=torch.uint8)
    scales = torch.zeros(512, dtype=torch.uint8)
    factors = torch.ones(16, dtype=torch.float32)
    plain = torch.ones((16, 16), dtype=torch.float16)
    packed = runtime.PackedLinear(codes[:128].view(16, 8), scales[:16].view(16, 1), factors[:1], ws)
    floating = runtime.FloatLinear(plain)
    bank = runtime.ExpertBank(((packed, floating, packed),), ws, width=16, dims=16)
    retained = {"codes": codes, "scales": scales, "global": factors, "float": plain}[source_kind]
    output = retained.view(torch.float16).reshape(-1)[-16:].view(1, 1, 16)
    before = retained.clone()
    monkeypatch.setattr(runtime.nvfp4_ops, "experts", lambda *a, **k: pytest.fail("aliased output reached CUDA"))
    monkeypatch.setattr(packed, "matmul", lambda *a, **k: pytest.fail("aliased output reached reference"))
    with pytest.raises(ValueError, match="immutable retained source storage"):
        bank.forward(torch.zeros((1, 16), dtype=torch.float16), torch.zeros((1, 1), dtype=torch.int32), out=output)
    assert torch.equal(retained, before)


@pytest.mark.parametrize("packed", [False, True])
def test_expert_construction_refuses_cross_device_parts_before_descriptor_publication(packed):
    ws = runtime.Workspace("cpu", width=16, dims=16, selection_rows=1, slots=1, experts=1,
                           expert_family=EXPERT_DEVICE)
    if packed:
        foreign = runtime.Workspace("meta", kernel_family=LINEAR_WMMA)
        part = runtime.PackedLinear(torch.empty((16, 8), dtype=torch.uint8, device="meta"),
                                    torch.empty((16, 1), dtype=torch.uint8, device="meta"),
                                    torch.empty(1, dtype=torch.float32, device="meta"), foreign)
    else:
        part = runtime.FloatLinear(torch.empty((16, 16), dtype=torch.float16, device="meta"))
    with pytest.raises(ValueError, match="source device"):
        runtime.ExpertBank(((part, part, part),), ws, width=16, dims=16)


def test_expert_input_device_and_borrowed_output_validate_before_execution(monkeypatch):
    ws = runtime.Workspace("cpu", width=16, dims=16, selection_rows=1, slots=1, experts=1)
    part = runtime.FloatLinear(torch.ones((16, 16), dtype=torch.float16))
    bank = runtime.ExpertBank(((part, part, part),), ws, width=16, dims=16)
    with pytest.raises(ValueError, match="input/selection"):
        bank.forward(torch.empty((1, 16), dtype=torch.float16, device="meta"),
                     torch.empty((1, 1), dtype=torch.int32, device="meta"))
    monkeypatch.setattr(part, "matmul", lambda *a, **k: pytest.fail("invalid output reached reference"))
    x, picks = torch.zeros((1, 16), dtype=torch.float16), torch.zeros((1, 1), dtype=torch.int32)
    for output in (x.view(1, 1, 16), ws.expert_input[:1].view(1, 1, 16)):
        with pytest.raises(ValueError, match="borrowed execution storage"):
            bank.forward(x, picks, out=output)
    with pytest.raises(ValueError, match="output differs"):
        bank.forward(x, picks, out=torch.empty((1, 1, 32), dtype=torch.float16)[:, :, ::2])


@pytest.mark.parametrize("family", [runtime.EXPERT_REFERENCE, EXPERT_DEVICE])
def test_expert_mutable_workspace_cannot_supply_later_input_rows(family, monkeypatch):
    ws = runtime.Workspace("cpu", width=16, dims=16, selection_rows=1, slots=1, experts=1,
                           expert_family=family)
    part = runtime.FloatLinear(torch.ones((16, 16), dtype=torch.float16))
    bank = runtime.ExpertBank(((part, part, part),), ws, width=16, dims=16)
    monkeypatch.setattr(runtime.nvfp4_ops, "experts", lambda *a, **k: pytest.fail("alias reached CUDA"))
    monkeypatch.setattr(part, "matmul", lambda *a, **k: pytest.fail("alias reached reference"))
    activation = ws.activation_device if family == EXPERT_DEVICE else ws.activation
    picks = torch.zeros((1, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="input/selection overlaps"):
        bank.forward(activation[:1], picks)
    selected = ws.counts_device[:1].view(1, 1) if family == EXPERT_DEVICE else ws.ids.view(torch.int32)[:1].view(1, 1)
    with pytest.raises(ValueError, match="input/selection overlaps"):
        bank.forward(torch.zeros((1, 16), dtype=torch.float16), selected)


def test_projection_and_expert_source_workspace_alias_refuse_before_mutation():
    ws = runtime.Workspace("cpu", width=16, dims=16, selection_rows=1, slots=1, experts=1)
    factor = torch.ones(1, dtype=torch.float32)
    scale = torch.ones((16, 1), dtype=torch.uint8)
    # The low-even source must not borrow bytes the tile decoder overwrites.
    with pytest.raises(ValueError, match="immutable source overlaps"):
        runtime.PackedLinear(ws.decoded.view(torch.uint8).flatten()[:128].view(16, 8), scale, factor, ws)
    packed = runtime.PackedLinear(torch.zeros((16, 8), dtype=torch.uint8), scale, factor, ws)
    ws.input.fill_(.75)
    before = ws.input.clone()
    with pytest.raises(ValueError, match="input overlaps borrowed mutable tile storage"):
        packed.matmul(ws.input[:1, :16])
    assert torch.equal(ws.input, before)
    part = runtime.FloatLinear(ws.activation[:16])
    with pytest.raises(ValueError, match="immutable source overlaps"):
        runtime.ExpertBank(((part, part, part),), ws, width=16, dims=16)
    other = runtime.Workspace("cpu")
    separate = runtime.PackedLinear(torch.zeros((16, 8), dtype=torch.uint8), scale, factor, other)
    with pytest.raises(ValueError, match="single immutable shared workspace"):
        runtime.ExpertBank(((separate, separate, separate),), ws, width=16, dims=16)


def test_embedding_output_alias_cannot_modify_retained_source_rows():
    table = torch.arange(64, dtype=torch.float16).view(4, 16)
    embedding = runtime.FloatEmbedding(table)
    before = table.clone()
    with pytest.raises(ValueError, match="immutable retained source storage"):
        embedding.lookup(torch.tensor([3, 2]), out=table[:2])
    assert torch.equal(table, before)
    result = embedding.lookup(torch.tensor([3, 2]), copies=2)
    assert torch.equal(result, torch.cat((before[[3, 2]], before[[3, 2]]), dim=-1))
    assert embedding.lookup(torch.empty(0, dtype=torch.int64)).shape == (0, 16)
    empty = table[:0]
    assert embedding.lookup(torch.empty(0, dtype=torch.int64), out=empty) is empty


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_rank_retention_matches_independent_family_role_arithmetic(source, world):
    directory, manifest, _ = source
    prototype, planner = model_plan(directory, manifest, (world - 1, world))
    # Literal fixture geometry, independently summed by semantic tensor role.
    d, wide, low, streams, vocab = 256, 1024, 32, 4, 256
    hc = 4 * (4 * low * wide + 2 * streams * wide + 4 * wide) + 4 * low * wide + 4 * wide
    embed_head = 2 * vocab * d + 2 * (vocab // world) * d
    routed = 2 * 3 * (128 // world) * d * 9 // 16 + 2 * 3 * 4
    shared = 3 * (128 // world) * d * 2
    moe = 2 * (3 * d * 4 + routed + shared)
    recurrent = ((10240 + 6144 + 96) // world) * d * 2
    recurrent += (10240 // world) * 4 * 2 + 2 * (48 // world) * 4 + 128 * 2
    recurrent += d * (6144 // world) * 2
    local_kv = 2 // world if world <= 2 else 1
    attention = (4096 // world) * d * 2 + 2 * local_kv * 256 * d * 2 + 640 * d * 2
    attention += d * (2048 // world) * 2 + (2 * 256 + 2 * 128) * 4
    assert byte_plan(prototype, planner)["resident_bytes"] == hc + embed_head + moe + recurrent + attention


def test_adopted_preparation_is_charged_once_while_future_plain_embedding_and_rope_scratch_remains(source):
    from tensorfold.families.qwen4_exp.cuda.sm70_nvfp4_weights import capacity_options

    directory, manifest, _ = source
    prototype, planner = model_plan(directory, manifest, (7, 8))
    ordinary, _ = capacity_options(prototype, planner, prefill_rows=1024, execution_rows=32)
    adopted, plan = capacity_options(prototype, planner, prefill_rows=1024, execution_rows=32, workspace_preallocated=True)
    persistent = runtime.workspace_bytes(width=16, dims=256)
    assert ordinary["device_workspace_bytes"] - adopted["device_workspace_bytes"] == persistent
    assert adopted["device_workspace_bytes"] == 1024 * 256 * 2 + plan["plain_linear_temporary_bytes"] + 32 * 4
    assert adopted["host_workspace_bytes"] == 0  # No PLE in this literal source; CPU selection planes already owned.


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partial_reader_failure_retires_acquired_io_or_marks_actual_owner_terminal(source, monkeypatch, cleanup_fails):
    from tensorfold.families.qwen4_exp.cuda import sm70_nvfp4_weights as importer
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired

    directory, manifest, _ = source
    prepared = model_plan(directory, manifest, (7, 8))
    primary, cleanup = ValueError("controlled reader construction"), OSError("controlled reader retirement")
    retired = []
    def close():
        retired.append(True)
        if cleanup_fails:
            raise cleanup
    class Reader:
        def __init__(self, *args):
            self.io = SimpleNamespace(close=close)
            raise primary
    monkeypatch.setattr(importer, "_Reader", Reader)
    with precision.using(precision.FULL, asked=True), pytest.raises(BaseException) as caught:
        importer.load(directory, manifest, "cpu", tp=(7, 8), _plan=prepared)
    assert retired == [True]
    assert _unretired(caught.value) is cleanup_fails
    if cleanup_fails:
        assert primary.__dict__["_tensorfold_retained_owners"][0].io.close is close
    else:
        assert caught.value is primary


def test_mtp_positions_require_explicit_matching_native_policy_without_implicit_extension(source, tmp_path):
    from tensorfold.families.qwen4_exp.cuda.sm70_nvfp4_weights import check_mtp_positions

    directory, manifest, _ = source
    target, _ = model_plan(directory, manifest, (7, 8))
    companion = tmp_path / "companion"
    companion.mkdir()
    config = json.loads((directory / "config.json").read_text())
    (companion / "config.json").write_text(json.dumps(config))
    check_mtp_positions(target, companion, context=1024)
    with pytest.raises(ValueError, match="position pairing"):
        check_mtp_positions(target, companion, context=target.cfg.rope.context_limit + 1)
    config["text_config"]["rope_parameters"]["rope_theta"] = 20000000
    (companion / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="position pairing"):
        check_mtp_positions(target, companion, context=1024)


def test_group_range_proof_matches_independent_full_decode_on_adverse_encodings():
    rng = np.random.default_rng(714)
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
    for factor in (.125, 100, 10000, 3e38):
        for _ in range(12):
            words = rng.integers(0, 256, size=(4, 32), dtype=np.uint8)
            scales = rng.integers(0, 256, size=(4, 4), dtype=np.uint8)
            exp, mantissa = (scales >> 3) & 15, scales & 7
            fp8 = np.where(exp == 0, mantissa.astype(np.float32) / 512,
                           np.ldexp(1 + mantissa.astype(np.float32) / 8, exp.astype(np.int32) - 7)).astype(np.float32)
            fp8 = np.where(scales & 128, -fp8, fp8)
            fp8[(scales & 127) == 127] = np.nan
            codes = np.empty((4, 64), dtype=np.uint8)
            codes[:, ::2], codes[:, 1::2] = words & 15, words >> 4
            with np.errstate(over="ignore", invalid="ignore"):
                decoded = (levels[codes] * np.repeat(fp8, 16, axis=1)).astype(np.float32) * np.float32(factor)
                rounded = decoded.astype(np.float16)
            valid = np.isfinite(decoded).all() and np.isfinite(rounded).all()
            if valid:
                _validate_nvfp4(torch.from_numpy(words), torch.from_numpy(scales), np.float32(factor), "adverse")
            else:
                with pytest.raises(ValueError):
                    _validate_nvfp4(torch.from_numpy(words), torch.from_numpy(scales), np.float32(factor), "adverse")


def test_nvfp4_group_range_proof_handles_actual_zero_codes_and_half_overflow_midpoint():
    zero = torch.zeros((1, 8), dtype=torch.uint8)
    one = torch.full((1, 8), 0x22, dtype=torch.uint8)  # E2M1 magnitude1
    scales = torch.full((1, 1), 0x38, dtype=torch.uint8)
    _validate_nvfp4(zero, scales, np.float32(3e38), "zero")
    _validate_nvfp4(one, scales, np.float32(65519), "finite-rounding")
    with pytest.raises(ValueError, match="finite FP16"):
        _validate_nvfp4(one, scales, np.float32(65520), "overflow-midpoint")
    with pytest.raises(ValueError):
        _validate_nvfp4(zero, torch.full((1, 1), 0x7f, dtype=torch.uint8), 1.0, "NaN-scale")


@pytest.mark.parametrize("kind,expected", [("bf16", .75), ("fp8", .5), ("nvfp4", None)])
def test_ple_owned_rows_reconstruct_directly_to_fp16_for_mapped_and_positional_readers(tmp_path, kind, expected):
    directory, _ = write(tmp_path / kind, ple=kind)
    body = prepare(directory, repository="controlled/ple", revision="fixed", norm_offset=1)
    manifest = directory / "family.json"
    manifest.write_text(json.dumps(body))
    for positional in (False, True):
        with precision.using(precision.FULL, asked=True):
            weights = load(directory, manifest, "cpu", tp=(7, 8), ple_on_ssd=positional)
        try:
            table = weights.layers[1].ple.table
            values = table.gather(np.array([1, 0, 1], dtype=np.int64))
            assert values.dtype == np.float16 and values.shape == (3, 64)
            if expected is not None:
                assert np.array_equal(values, np.full_like(values, expected))
            else:
                assert np.array_equal(values[:, 0::2], np.full((3, 32), .5, dtype=np.float16))
                assert np.array_equal(values[:, 1::2], np.full((3, 32), .75, dtype=np.float16))
        finally:
            weights.meta["ple_tables"].close(lambda: None)


def test_declared_wmma_candidate_is_profile_identity_and_not_automatic_promotion(source, tmp_path):
    directory, manifest, _ = source
    body = copy.deepcopy(json.loads(manifest.read_text()))
    body["numerical"]["linear_kernel_family"] = LINEAR_WMMA
    candidate = tmp_path / "wmma.json"
    candidate.write_text(json.dumps(body))
    reference, plans = model_plan(directory, manifest, (7, 8))
    selected, selected_plans = model_plan(directory, candidate, (7, 8))
    assert reference.meta["family_manifest"] != selected.meta["family_manifest"]
    assert byte_plan(reference, plans)["workspace_bytes"] - byte_plan(selected, selected_plans)["workspace_bytes"] == 65536
    assert selected.meta["source_activation_departure"] == "unqualified"
