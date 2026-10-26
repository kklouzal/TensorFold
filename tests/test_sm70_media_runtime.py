"""CPU startup/resource/retirement contracts; these grant no device qualification."""
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.media_runtime import MediaRuntime, media_resources, media_unit_limit  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.media_producer import producer_resources  # noqa: E402


def profile(encoder_rank):
    # Explicit synthetic test values, never serving defaults.
    return SimpleNamespace(execution=SimpleNamespace(encoder_rank=encoder_rank, input_rows=16,
        encoder_patches=64, projected_cache_rows=32, vision_workspace_bytes=123456),
        cpu=SimpleNamespace(max_staging_bytes=512, max_replay_bytes=256, max_videos=1),
        layout=SimpleNamespace(hidden=1152, heads=16),
        mmproj_plan=SimpleNamespace(resident_bytes=897862112,
            weights={"temporal": SimpleNamespace(temporal_stack=True, resident_bytes=3538944),
                     "ordinary": SimpleNamespace(temporal_stack=False, resident_bytes=4096)}))


@pytest.mark.parametrize("encoder_rank", [0, 7])
def test_resource_placement_charges_one_encoder_and_exact_producer_payload(monkeypatch, encoder_rank):
    monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("CPU resource planning allocated a tensor"))
    selected = profile(encoder_rank)
    result = media_resources(selected, hidden=2560, world=8)
    producer = producer_resources(rows=16, patches=64, cache_rows=32, hidden=2560, world=8,
                                  encoder_rank=encoder_rank)
    assert result["vision_frequency_bytes"] == 144
    assert result["vision_weight_bytes"] == 897862112
    assert sum(result["static_with_vision_bytes_by_rank"]) == (
        sum(producer["static_device_bytes_by_rank"]) + 897862112 + 144)
    for rank in range(8):
        assert result["static_with_vision_bytes_by_rank"][rank] == (
            producer["static_device_bytes_by_rank"][rank] + (897862256 if rank == encoder_rank else 0))
        assert result["media_transient_by_rank"][rank] == (
            producer["temporary_device_bytes_by_rank"][rank] + (123456 if rank == encoder_rank else 0))
    assert result["shared_host_payload_bound"] == (
        512 + 256 + 64 * 1536 * 6 + 8 * producer["cpu_coordinate_arrays_bytes"])
    assert "RSS qualification pending" in result["host_scope"]


def runtime(directory):
    owner = MediaRuntime.__new__(MediaRuntime)
    import os

    owner.pid, owner.closed = os.getpid(), False
    owner.work = owner.producer = owner.encoder = owner.reader = None
    owner.requests = None
    owner.executor = owner.store = owner.frontend = owner.planner = owner.contract = owner.tokenizer = None
    owner.directory, owner.directory_failure = directory, None
    owner._warm_output = owner._warm_pixels = None
    return owner


def test_runtime_health_checks_authoritative_cpu_work_failure(tmp_path):
    owner = runtime(tmp_path)
    owner.producer = object()
    primary = RuntimeError("CPU work retirement is uncertain")

    def failed():
        raise primary

    owner.work = SimpleNamespace(check_health=failed)
    checked = []
    owner.executor = SimpleNamespace(check_health=lambda: checked.append(True))
    with pytest.raises(RuntimeError) as caught:
        owner.check_media_health()
    assert caught.value is primary and checked == []


@pytest.mark.parametrize("prepared_metadata", [False, True])
def test_actual_engine_resolves_visual_assets_from_selected_raw_checkpoint(tmp_path, monkeypatch, prepared_metadata):
    import json
    from gguf_fixture import tiny_model
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp.gguf import metadata_config
    from tensorfold.families.qwen4_exp.cuda import engine, media_profile

    source = tmp_path / "raw"
    source.mkdir()
    target = tiny_model(source / "target.gguf")
    metadata = metadata_config(target)
    metadata.update(gguf_file=str(target), quantization_config={"quant_method": "gguf"})
    selected = tmp_path / "prepared-cache" if prepared_metadata else source
    selected.mkdir(exist_ok=True)
    (selected / "config.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(chat_template, "prepare_chat_template", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: pytest.fail("media source admission reached CUDA"))
    observed = []

    class Inspected(Exception):
        pass

    def inspect(path, *, model_dir, target_hidden, world, context_rows):
        observed.append((path, model_dir, target_hidden, world, context_rows))
        raise Inspected

    monkeypatch.setattr(media_profile, "read_media_profile", inspect)
    owner = engine.FlashNextEngine.__new__(engine.FlashNextEngine)
    owner._profile_build, owner._startup_reads = None, []
    manifest = tmp_path / "media.json"
    with pytest.raises(Inspected):
        owner._initialize(selected, depth=0, tp=8, rank=0, master="test", graphs=False, vision=True,
                          kv_dtype="int8", adaptive_sessions=True, media_profile_manifest=manifest)
    assert observed == [(manifest, source, 256, 8, 256)]


@pytest.mark.parametrize("stale_override", [False, True])
def test_bind_issues_native_role_identity_and_complete_model_domain(tmp_path, monkeypatch, stale_override):
    import hashlib
    import json
    from tensorfold.families.qwen4_exp import gguf
    from tensorfold.families.qwen4_exp.cuda import media_runtime
    from tensorfold.token_metadata import tokenizer_identity
    from tensorfold import token_metadata
    from test_media_replay import envelope
    from test_qwen4_exp_media_plan import tokenizer as tokenizer_fixture

    native = tokenizer_fixture.__wrapped__()
    lexical = native.get_vocab_size(with_added_tokens=True)
    text = native.to_str()
    (tmp_path / "tokenizer.json").write_text(text)
    roles = {"version": 1, "vocab_size": lexical + 3, "control": list(range(2, 6)),
             "unused": list(range(lexical, lexical + 3))}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tensorfold_token_roles": roles}))
    source_sha = hashlib.sha256(text.encode()).hexdigest()
    weights = SimpleNamespace(cfg=SimpleNamespace(vocab=lexical + 3),
                              meta={"coordinate_profile_identity": "1" * 64})
    owner = runtime(None)
    owner.identity = None
    owner.engine = SimpleNamespace(rank=0, w=weights, model_dir=tmp_path, context_window=512,
        chat_template=SimpleNamespace(provenance=lambda **_: {"sha256": "3" * 64}))
    owner.profile = SimpleNamespace(cpu=envelope(max_expanded_rows=256), vision_sha256="4" * 64, processor_id="processor",
        execution=SimpleNamespace(encoder_patches=64, max_rendered_bytes=4096, max_expanded_bytes=8192))
    checked = []
    owner.authority = SimpleNamespace(verify_unchanged=lambda: checked.append(True),
        files=(SimpleNamespace(logical_name="metadata/tokenizer.json", sha256=source_sha),))
    monkeypatch.setattr(token_metadata, "tokenizer_override", lambda: SimpleNamespace(sha256="a" * 64)
                        if stale_override else None)
    monkeypatch.setattr(gguf, "resolve_gguf", lambda _: tmp_path / "target.gguf")
    monkeypatch.setattr(gguf.GGUFCheckpoint, "open", lambda _: SimpleNamespace(metadata={"qwen4exp.ple.image_token_id": 2}))
    published = {}

    class Planned(Exception):
        pass

    def phase(engine, label, operation, **_):
        if label == "contract":
            raise Planned
        value = published[label] = operation()
        if label == "planner-source":
            return [value, *[{**value, "tokenizer": None} for _ in range(7)]]
        assert label == "planner"
        return [value, *([None] * 7)]

    monkeypatch.setattr(media_runtime, "media_phase", phase)
    if stale_override:
        with pytest.raises(ValueError, match="captured source authority"):
            owner.bind(weights)
        assert owner.tokenizer is None and not published
        return
    with pytest.raises(Planned):
        owner.bind(weights)
    contract = published["planner"]
    assert contract["vocab_size"] == lexical + 3
    assert contract["identities"][1] == tokenizer_identity(owner.tokenizer)
    assert contract["identities"][1] != source_sha
    assert owner.tokenizer.unused_token_ids == frozenset(range(lexical, lexical + 3))
    assert len(checked) == 3


@pytest.mark.parametrize("resource", ["workspace", "shared_host"])
def test_combined_resource_products_refuse_int64_overflow_before_allocation(resource):
    selected = profile(7)
    if resource == "workspace":
        selected.execution.vision_workspace_bytes = 2**63 - 1
    else:
        selected.cpu.max_staging_bytes = 2**63 - 1
    with pytest.raises(ValueError, match="signed-int64"):
        media_resources(selected, hidden=2560, world=8)


def test_startup_publishes_actual_partial_owner_before_source_or_encoder_work():
    selected = profile(7)
    seen = []
    primary = MemoryError("journal could not finish publication")

    def register(owner):
        seen.append(owner)
        raise primary

    engine = SimpleNamespace(rank=7, tp=8, w=SimpleNamespace(cfg=SimpleNamespace(hidden=2560)))
    with pytest.raises(MemoryError) as caught:
        MediaRuntime.create(engine, selected, authority=None, software_binding=None,
                            allow_urls=False, register=register)
    assert caught.value is primary and len(seen) == 1
    assert seen[0].encoder is None and seen[0].producer is None and seen[0].reader is None
    assert seen[0].resources["media_transient_bytes"] == seen[0].resources["media_transient_by_rank"][7]


def test_source_phase_captures_complete_content_once_and_adopts_before_any_device(tmp_path, monkeypatch):
    from tensorfold.engine.model_identity import ModelIdentity
    from tensorfold.families.qwen4_exp.cuda import execution_identity, media_runtime
    from tensorfold.cuda.sm70 import media_software_binding

    target, manifest, vision = [tmp_path / name for name in ("target.bin", "profile.json", "vision.bin")]
    for index, path in enumerate((target, manifest, vision)):
        path.write_bytes(bytes([index]) * (index + 5))
    phases, adopted, retired = [], [], []

    def phase(_engine, _channel, stage, operation, **kwargs):
        phases.append(stage)
        return [operation()]

    def verify(authority):
        assert type(authority) is ModelIdentity
        authority.verify_unchanged()
        assert {item.path for item in authority.files} == {str(target), str(manifest), str(vision)}
        adopted.append(authority)

    monkeypatch.setattr(execution_identity, "artifact_files", lambda _engine: ({"target/model": target}, None))
    monkeypatch.setattr(execution_identity, "_phase", phase)
    monkeypatch.setattr(execution_identity, "_retire_store", lambda *a, **k: retired.append(True))
    software = SimpleNamespace(record=lambda: {"identity_sha256": "a" * 64})
    monkeypatch.setattr(media_software_binding, "issue_media_binding", lambda: software)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: pytest.fail("CPU source phase touched CUDA"))
    selected = SimpleNamespace(identity="profile", manifest_path=manifest, vision_path=vision, verify_source=verify)
    engine = SimpleNamespace(rank=0, tp=1, _startup_store=object())
    authority, binding = media_runtime.prepare_media_source(engine, selected)
    assert adopted == [authority] and binding is software and retired == [True]
    assert authority.bytes_read == sum(path.stat().st_size for path in (target, manifest, vision))
    assert phases == ["preflight", "inventory", "agreement", "content", "verification", "issuance", "adoption"]
    # DFlash pairing may already own the exact immutable shared closure. Reuse
    # that issued authority rather than a second target/mmproj payload pass.
    engine._dflash_pairing = SimpleNamespace(authority=authority)
    reused, _ = media_runtime.prepare_media_source(engine, selected)
    assert reused.bytes_read == 0 and reused.hashes_reused == 3
    assert reused.files == authority.files and reused.model_prefix == authority.model_prefix


def test_runtime_retires_workers_then_device_owners_before_removing_directory(tmp_path):
    path = tmp_path / "replay"
    path.mkdir()
    owner, retired = runtime(path), []
    for name in ("work", "producer", "encoder", "reader"):
        setattr(owner, name, SimpleNamespace(close=lambda name=name: retired.append(name)))
    owner.close()
    assert retired == ["work", "producer", "encoder", "reader"]
    assert owner.closed and not path.exists()
    owner.close()
    assert len(retired) == 4


def test_failed_device_fence_preserves_actual_owner_and_backing_directory(tmp_path):
    path = tmp_path / "replay"
    path.mkdir()
    owner = runtime(path)
    primary = RuntimeError("encoder fence failed")

    def fail():
        raise primary

    encoder = SimpleNamespace(close=fail)
    owner.encoder = encoder
    with pytest.raises(RuntimeError) as caught:
        owner.close()
    assert caught.value is primary
    assert owner.encoder is encoder and owner.directory == path and not owner.closed
    encoder.close = lambda: None
    owner.close()
    assert owner.closed and not path.exists()


def test_ambiguous_directory_removal_spends_path_authority_once(monkeypatch, tmp_path):
    import tensorfold.families.qwen4_exp.cuda.media_runtime as module

    owner = runtime(tmp_path / "replay")
    primary, attempts = OSError("ambiguous remove"), []

    def remove(path):
        attempts.append(path)
        raise primary

    monkeypatch.setattr(module.os, "rmdir", remove)
    for _ in range(2):
        with pytest.raises(OSError) as caught:
            owner.close()
        assert caught.value is primary
    assert attempts == [tmp_path / "replay"]
    assert owner.directory is None and owner.directory_failure is primary and not owner.closed


def test_foreign_process_runtime_does_not_enter_any_owner_cleanup(monkeypatch, tmp_path):
    import tensorfold.families.qwen4_exp.cuda.media_runtime as module

    owner = runtime(tmp_path)
    owner.work = SimpleNamespace(close=lambda: pytest.fail("foreign process touched worker"))
    monkeypatch.setattr(module.os, "getpid", lambda: owner.pid + 1)
    with pytest.raises(RuntimeError, match="another process"):
        owner.close()


def test_replay_directory_is_journaled_before_path_conversion_can_fail(tmp_path, monkeypatch):
    import tensorfold.families.qwen4_exp.cuda.media_runtime as module

    owner = runtime(None)
    create = module.tempfile.mkdtemp
    created = []
    primary = MemoryError("Path allocation failed")

    def mkdtemp(**kwargs):
        path = create(dir=tmp_path, **kwargs)
        created.append(path)
        return path

    def convert(_path):
        raise primary

    monkeypatch.setattr(module.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(module, "Path", convert)
    with pytest.raises(MemoryError) as caught:
        owner._create_replay_directory()
    assert caught.value is primary and owner.directory == created[0]
    assert Path(created[0]).is_dir()
    owner.close()
    assert not Path(created[0]).exists()


@pytest.mark.parametrize("images,videos,items,frames,rows,expected", [
    (0, 1, 1, 4, 256, 2), (0, 1, 1, 5, 256, 3),
    (4, 3, 2, 64, 256, 64), (4, 0, 3, 64, 256, 3), (4, 3, 16, 64, 8, 8),
])
def test_unit_limit_uses_temporal_groups_and_declared_total_rows(images, videos, items, frames, rows, expected):
    from test_media_replay import envelope

    policy = envelope(max_images=images, max_videos=videos, max_replay_items=items,
                      max_selected_frames=frames, max_expanded_rows=rows)
    assert media_unit_limit(policy) == expected


def test_one_encoded_video_can_admit_its_two_temporal_attention_units():
    from test_media_replay import envelope
    from test_qwen4_exp_media_plan import NAMES, planner, probe, tokenizer

    policy = envelope(max_images=0, max_videos=1, max_replay_items=1, max_selected_frames=4,
                      max_expanded_rows=256)
    selected = planner(tokenizer.__wrapped__(), max_segments=media_unit_limit(policy))
    _, video, start, end = NAMES
    plan = selected(f"{start}{video}{end}", (probe("video", 0),), max_prompt_tokens=512, max_visual_tokens=None)
    assert len(plan.units) == 2 and [unit.run.temporal_group for unit in plan.units] == [0, 1]


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    from dataclasses import replace
    from tensorfold.families.qwen4_exp.cuda.media_requests import MediaRequests
    from tensorfold.vision.media_frontend import MediaExecutor, PreparedReplayMedia
    from tensorfold.vision.replay import ReplayStore
    from test_media_replay import envelope, image_source, video_source
    from test_qwen4_exp_media_plan import NAMES, planner, probe, tokenizer

    policy = envelope(max_expanded_rows=256)
    store = ReplayStore(tmp_path, policy)
    lease = store.stage([image_source(), video_source()], allow_urls=False, cancelled=lambda: False)
    executor = MediaExecutor(store)
    sources = tuple(replace(probe(kind, index), ref=lease.refs[index], envelope_id=policy.identity)
                    for index, kind in enumerate(("image", "video")))
    selected = planner(tokenizer.__wrapped__())
    image, video, start, end = NAMES
    plan = selected(f"hello {start}{image}{end} hello {start}{video}{end}", sources,
                    max_prompt_tokens=512, max_visual_tokens=None)
    prepared = PreparedReplayMedia(plan, sources, lease, executor)
    owner = runtime(None)
    owner.producer, owner.executor, owner.store = object(), executor, store
    owner.contract, owner.identity = selected.contract, selected.contract.identities[0]
    owner.profile = SimpleNamespace(cpu=policy)
    owner.requests = MediaRequests(owner)

    def forbidden(*args, **kwargs):
        pytest.fail("continuation decoded media or acquired another replay borrow")

    monkeypatch.setattr(executor, "probe_replay", forbidden)
    monkeypatch.setattr(executor, "pixel_unit", forbidden)
    monkeypatch.setattr(lease, "borrow", forbidden)
    try:
        yield owner, prepared
    finally:
        prepared.close()


@pytest.mark.parametrize("container", [tuple, list])
def test_continuation_reuses_one_physical_replay_without_decode_or_quota_borrow(continuation, container):
    owner, prepared = continuation
    before = tuple(owner.store.root.rglob("*"))
    ids = container((*prepared.token_ids, 1, 1))
    view = owner.continued(prepared, ids)
    again = owner.continued(view, (*view.token_ids, 1))
    assert view.token_ids == tuple(ids) and again.token_ids == (*view.token_ids, 1)
    assert view.plan.units == again.plan.units == prepared.plan.units
    assert view.plan.media_identity == again.plan.media_identity == prepared.plan.media_identity
    assert view.probes is again.probes is prepared.probes
    assert view.replay is again.replay is prepared.replay and view.executor is prepared.executor
    assert prepared.replay._borrows == 0 and owner.store._leases == 1
    assert tuple(owner.store.root.rglob("*")) == before
    assert owner.requests.close_if_unclaimed(prepared)
    assert view.replay._closed and again.replay._closed and owner.store._leases == 0


@pytest.mark.parametrize("defect", ["changed-prefix", "shorter", "unchanged", "context", "visual", "boolean", "iterator"])
def test_continuation_rejects_changed_prefix_geometry_and_invalid_appends(continuation, defect):
    owner, prepared = continuation
    ids = [*prepared.token_ids, 1]
    if defect == "changed-prefix":
        ids[0] = 0
    elif defect == "shorter":
        ids = ids[:1]
    elif defect == "unchanged":
        ids = list(prepared.token_ids)
    elif defect == "context":
        ids += [1] * owner.contract.max_rows
    elif defect == "visual":
        ids[-1] = owner.contract.roles[0]
    elif defect == "boolean":
        ids[-1] = True
    elif defect == "iterator":
        ids = iter(ids)
    with pytest.raises(ValueError):
        owner.continued(prepared, ids)
    assert not prepared.replay._closed and prepared.replay._borrows == 0


def test_continuation_refuses_retired_or_other_profile_replay(continuation):
    owner, prepared = continuation
    owner.identity = "f" * 64
    with pytest.raises(ValueError, match="live runtime"):
        owner.continued(prepared, (*prepared.token_ids, 1))
    owner.identity = prepared.plan.profile_identity
    prepared.close()
    with pytest.raises(ValueError, match="retired replay"):
        owner.continued(prepared, (*prepared.token_ids, 1))


def test_continuation_verifies_visual_unit_spans_and_order_unchanged(continuation):
    from dataclasses import replace

    owner, prepared = continuation
    first = prepared.plan.units[0]
    prepared.plan = replace(prepared.plan, units=(replace(first, run=replace(first.run, start=first.run.start + 1)),
                                                 *prepared.plan.units[1:]))
    with pytest.raises(ValueError, match="visual units"):
        owner.continued(prepared, (*prepared.token_ids, 1))


@pytest.mark.parametrize("changes", [
    {"vision": False}, {"graphs": True}, {"tp": 1}, {"adaptive_sessions": False},
    {"indexer_history": "ram"}, {"dspark_model_dir": Path("draft")}, {"vram_experts": 1.},
])
def test_profile_unsupported_startup_refuses_before_model_or_cuda(monkeypatch, tmp_path, changes):
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp import gguf
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setattr(gguf, "resolve_gguf", lambda _path: tmp_path / "target.gguf")
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported media profile reached payload/device acquisition")
    monkeypatch.setattr(chat_template, "prepare_chat_template", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    options = dict(vision=True, graphs=False, tp=8, adaptive_sessions=True, kv_dtype="int8",
                   master="127.0.0.1", media_profile_manifest=tmp_path / "media.json")
    options.update(changes)
    with pytest.raises(ValueError, match="media profile requires"):
        FlashNextEngine(tmp_path, **options)
