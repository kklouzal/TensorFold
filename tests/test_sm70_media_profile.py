"""CPU admission/source controls; sparse GGUF extents contain no model payload."""
from dataclasses import fields, FrozenInstanceError
import hashlib
import json
import math
import struct

import pytest

torch = pytest.importorskip("torch")

from tensorfold.engine.model_identity import capture_model_identity  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.media_profile import (  # noqa: E402
    MediaExecution, SCHEMA, read_media_profile,
)
from tensorfold.families.qwen4_exp.cuda.mmproj_weights import VisionLayout  # noqa: E402
from tensorfold.vision.media_worker import PROCESSOR_ID  # noqa: E402
from tensorfold.vision.replay import MediaEnvelope  # noqa: E402


LAYOUT = dict(depth=27, hidden=1152, heads=16, intermediate=4304, output=2560,
              patch=16, temporal=2, merge=2, channels=3, positions=2304, theta=10000.)


def _sparse_mmproj(path, corruption=None, *, content_hash=False):
    """Independent original-converter names; only header bytes occupy storage."""
    metadata = {"general.architecture": "clip", "general.type": "mmproj", "clip.has_vision_encoder": True,
                "clip.projector_type": "qwen3vl_merger", "clip.use_gelu": True,
                "clip.vision.projection_dim": 2560, "clip.vision.patch_size": 16,
                "clip.vision.embedding_length": 1152, "clip.vision.feed_forward_length": 4304,
                "clip.vision.block_count": 27, "clip.vision.attention.head_count": 16,
                "clip.vision.spatial_merge_size": 2, "clip.vision.is_deepstack_layers": [False] * 27,
                "clip.vision.image_mean": [.5] * 3, "clip.vision.image_std": [.5] * 3,
                "clip.vision.attention.layer_norm_epsilon": 1e-6}
    shapes = []
    for index in range(27):
        for name, shape in (("attn_qkv", (3456, 1152)), ("attn_out", (1152, 1152)),
                            ("ffn_up", (4304, 1152)), ("ffn_down", (1152, 4304)),
                            ("ln1", (1152,)), ("ln2", (1152,))):
            shapes.extend(((f"v.blk.{index}.{name}.weight", shape),
                           (f"v.blk.{index}.{name}.bias", (shape[0],))))
    shapes.extend((("v.patch_embd.weight", (1152, 3, 16, 16)),
                   ("v.patch_embd.weight.1", (1152, 3, 16, 16)), ("v.patch_embd.bias", (1152,)),
                   ("v.position_embd.weight", (2304, 1152))))
    for name, shape in (("v.post_ln", (1152,)), ("mm.0", (4608, 4608)), ("mm.2", (2560, 4608))):
        shapes.extend(((name + ".weight", shape), (name + ".bias", (shape[0],))))
    if corruption == "missing":
        shapes.pop()
    elif corruption == "extra":
        shapes.append(("unknown", (1,)))
    elif corruption == "duplicate":
        shapes.append(shapes[-1])
    elif corruption == "shape":
        name, shape = shapes[0]
        shapes[0] = (name, (1, *shape))
    elif corruption == "metadata":
        metadata["clip.vision.patch_size"] = 8

    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    def value(item):
        if type(item) is bool:
            return struct.pack("<I?", 7, item)
        if type(item) is str:
            return struct.pack("<I", 8) + string(item)
        if type(item) is float:
            return struct.pack("<If", 6, item)
        if type(item) is list:
            kind, fmt = (7, "?") if type(item[0]) is bool else (6, "f")
            return struct.pack("<IIQ", 9, kind, len(item)) + struct.pack("<" + fmt * len(item), *item)
        return struct.pack("<Iq", 11, item)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(shapes), len(metadata))
    for name, item in metadata.items():
        header += string(name) + value(item)
    payload_size = 0
    for name, shape in shapes:
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        header += struct.pack("<IQ", 30, payload_size)
        payload_size += math.prod(shape) * 2
        payload_size += -payload_size % 32
    header += b"\0" * (-len(header) % 32)
    with path.open("wb") as stream:
        stream.write(header)
        stream.truncate(len(header) + payload_size)
    digest = "0" * 64
    if content_hash:
        hasher = hashlib.sha256(header)
        zeros = bytes(1 << 20)
        remaining = payload_size
        while remaining:
            count = min(remaining, len(zeros))
            hasher.update(memoryview(zeros)[:count])
            remaining -= count
        digest = hasher.hexdigest()
    return path, digest


def _cpu():
    # Explicit test policy only. These values are not production defaults.
    return dict(max_images=4, max_videos=3, max_upload_bytes=1024**2, max_total_upload_bytes=4 * 1024**2,
                max_dimension=4096, max_pixels_per_frame=8 * 1024**2, max_total_decoded_pixels=64 * 1024**2,
                max_decoded_bytes=256 * 1024**2, max_selected_frames=64, max_video_seconds=30.,
                max_expanded_rows=1024, max_workers=1, max_worker_threads=1,
                max_worker_address_bytes=32 * 1024**3, max_worker_cpu_seconds=60, max_worker_wall_seconds=90.,
                max_staging_wall_seconds=90., max_single_allocation_bytes=128 * 1024**2,
                max_probe_bytes=1024**2, max_probe_seconds=10., max_scan_frames=128, max_scan_packets=256,
                max_staging_bytes=256 * 1024**2, max_replay_bytes=8 * 1024**2, max_replay_items=16,
                max_replay_leases=4, max_replay_borrows=16, max_manifest_bytes=65_536, max_result_bytes=65_536,
                max_redirects=0, max_url_chars=4096, retirement_grace_seconds=.2, assumed_fps=None)


@pytest.fixture(scope="module")
def vision_source(tmp_path_factory):
    return _sparse_mmproj(tmp_path_factory.mktemp("media-profile") / "vision.gguf")


@pytest.fixture
def profile_record(tmp_path, vision_source):
    path, digest = vision_source
    body = dict(schema=SCHEMA, version=1,
                vision=dict(path=path.name, size_bytes=path.stat().st_size, sha256=digest, layout=dict(LAYOUT)),
                processor=dict(id=PROCESSOR_ID), cpu=_cpu(),
                execution=dict(encoder_rank=7, input_rows=512, encoder_patches=4096, projected_cache_rows=0,
                               max_rendered_bytes=65536, max_expanded_bytes=65536, vision_workspace_bytes=1024**3))
    return body, tmp_path / "profile.json", path.parent


def _read(profile_record):
    body, path, root = profile_record
    path.write_text(json.dumps(body))
    return read_media_profile(path, model_dir=root, target_hidden=2560, world=8, context_rows=2048)


def test_complete_profile_has_immutable_explicit_policies_and_no_tensor_allocation(profile_record, monkeypatch):
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("tensor allocation during admission"))
    profile = _read(profile_record)
    assert profile.layout == VisionLayout(**LAYOUT)
    assert len(profile.mmproj_plan.weights) == 333
    assert profile.mmproj_plan.resident_bytes > 0
    assert profile.cpu == MediaEnvelope.from_record(_cpu())
    assert profile.execution.encoder_rank == 7 and profile.execution.projected_cache_rows == 0
    assert profile.required_input_files() == frozenset((profile.manifest_path, profile.vision_path))
    assert profile.expected_inputs()[profile.manifest_path] == (
        profile.manifest_path.stat().st_size, hashlib.sha256(profile.manifest_path.read_bytes()).hexdigest())
    with pytest.raises(FrozenInstanceError):
        profile.execution.input_rows = 17
    with pytest.raises(TypeError):
        profile.expected_inputs()[profile.manifest_path] = (0, "")


_REQUIRED = [(section, field) for section, names in (
    (None, ("schema", "version", "vision", "processor", "execution", "cpu")),
    ("vision", ("path", "size_bytes", "sha256", "layout")),
    ("processor", ("id",)), ("execution", tuple(f.name for f in fields(MediaExecution))),
    ("cpu", tuple(f.name for f in fields(MediaEnvelope))),
    ("layout", tuple(f.name for f in fields(VisionLayout)))) for field in names]


def _section(body, section):
    return body if section is None else body["vision"]["layout"] if section == "layout" else body[section]


@pytest.mark.parametrize("section,field", _REQUIRED)
def test_every_field_is_mandatory(profile_record, section, field):
    del _section(profile_record[0], section)[field]
    with pytest.raises(ValueError, match="requires exactly"):
        _read(profile_record)


@pytest.mark.parametrize("section", [None, "vision", "processor", "execution", "cpu", "layout"])
def test_unknown_fields_are_rejected(profile_record, section):
    _section(profile_record[0], section)["extra"] = True
    with pytest.raises(ValueError, match="requires exactly"):
        _read(profile_record)


@pytest.mark.parametrize("section,name,value", [
    (None, "version", True), (None, "version", 2), (None, "schema", "other"),
    ("vision", "sha256", "A" * 64), ("vision", "size_bytes", True), ("vision", "size_bytes", 1),
    ("processor", "id", "other"), ("cpu", "max_workers", True), ("cpu", "max_worker_threads", 1.5),
    ("cpu", "max_worker_address_bytes", 2**63), ("cpu", "max_expanded_rows", 2049),
    ("cpu", "max_replay_borrows", 1),
    ("cpu", "assumed_fps", [True, 1]), ("cpu", "assumed_fps", [1]), ("cpu", "assumed_fps", "24"),
    ("execution", "encoder_rank", -1), ("execution", "encoder_rank", 8),
    ("execution", "encoder_rank", True), ("execution", "input_rows", 0),
    ("execution", "input_rows", 2049), ("execution", "encoder_patches", 2),
    ("execution", "encoder_patches", 2**31), ("execution", "projected_cache_rows", -1),
    ("execution", "projected_cache_rows", 2049), ("execution", "vision_workspace_bytes", 0),
    ("execution", "max_rendered_bytes", 2**63), ("execution", "max_expanded_bytes", True),
    ("layout", "depth", 26), ("layout", "hidden", 1024), ("layout", "heads", 8),
    ("layout", "intermediate", 4000), ("layout", "output", 2561), ("layout", "patch", 8),
    ("layout", "temporal", 1), ("layout", "merge", 1), ("layout", "channels", 1),
    ("layout", "positions", 1024), ("layout", "theta", 10000000.), ("layout", "theta", True),
])
def test_invalid_types_ranges_and_unimplemented_profiles_fail(profile_record, section, name, value):
    _section(profile_record[0], section)[name] = value
    with pytest.raises(ValueError):
        _read(profile_record)


@pytest.mark.parametrize("name", ["../vision.gguf", "/vision.gguf", "./vision.gguf", "a//b.gguf",
                                   "a/../vision.gguf", "a\\vision.gguf", "vision\0.gguf", "vision\n.gguf"])
def test_checkpoint_paths_are_normalized_and_authorized(profile_record, name):
    profile_record[0]["vision"]["path"] = name
    with pytest.raises(ValueError):
        _read(profile_record)


def test_symlink_escape_is_rejected(profile_record, tmp_path):
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"GGUF")
    root = tmp_path / "model"
    root.mkdir()
    (root / "vision.gguf").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        _read((profile_record[0], profile_record[1], root))


@pytest.mark.parametrize("corruption", ["missing", "extra", "duplicate", "shape", "metadata"])
def test_complete_real_gguf_header_is_checked_before_allocation(profile_record, tmp_path, corruption, monkeypatch):
    source, _ = _sparse_mmproj(tmp_path / "vision.gguf", corruption)
    profile_record[0]["vision"]["size_bytes"] = source.stat().st_size
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("allocation before complete map"))
    with pytest.raises(ValueError):
        _read((profile_record[0], profile_record[1], tmp_path))


@pytest.mark.parametrize("suffix", ['"version":1,"version":1', '"version":NaN', '"version":Infinity',
                                     '"version":1e999', '"version":"\\ud800"'])
def test_strict_json_rejects_duplicate_nonfinite_and_invalid_unicode(profile_record, suffix):
    body, path, root = profile_record
    raw = json.dumps(body).replace('"version": 1', suffix)
    path.write_text(raw)
    with pytest.raises((ValueError, UnicodeError)):
        read_media_profile(path, model_dir=root, target_hidden=2560, world=8, context_rows=2048)


def test_manifest_fingerprint_uses_one_captured_read(profile_record, monkeypatch):
    import tensorfold.families.qwen4_exp.cuda.media_profile as module
    from contextlib import contextmanager
    original = module._regular_stream
    calls = []

    @contextmanager
    def counted(path):
        calls.append(path)
        with original(path) as stream:
            yield stream

    monkeypatch.setattr(module, "_regular_stream", counted)
    profile = _read(profile_record)
    assert calls == [profile.manifest_path]
    initial = profile.identity
    profile_record[0]["cpu"]["max_workers"] = 2
    assert _read(profile_record).identity != initial
    profile_record[0]["cpu"]["max_workers"] = 1
    profile_record[0]["execution"]["projected_cache_rows"] = 1
    assert _read(profile_record).identity != initial


def test_issued_content_authority_checks_exact_expected_bytes_and_staleness(profile_record, tmp_path):
    source, digest = _sparse_mmproj(tmp_path / "vision.gguf", content_hash=True)
    profile_record[0]["vision"].update(size_bytes=source.stat().st_size, sha256=digest)
    configured = (profile_record[0], profile_record[1], tmp_path)
    profile = _read(configured)
    inputs = {"manifest": profile.manifest_path, "vision": profile.vision_path}
    authority = capture_model_identity(inputs, runtime_identity={"media": profile.identity},
                                       authorize=lambda _name, path: path.resolve(),
                                       max_file_bytes=source.stat().st_size,
                                       max_total_bytes=sum(path.stat().st_size for path in inputs.values()))
    assert profile.verify_source(authority) is profile
    with pytest.raises(ValueError, match="issued"):
        profile.verify_source({"files": authority.files})
    from dataclasses import replace
    with pytest.raises(ValueError, match="source identity"):
        replace(profile, vision_sha256="f" * 64).verify_source(authority)
    with pytest.raises(ValueError, match="parsed execution input"):
        replace(profile, _source_stats=((profile.manifest_path, (0,) * 5),
                                        (profile.vision_path, (0,) * 5))).verify_source(authority)
    raw = profile.manifest_path.read_bytes()
    changed = raw.replace(b'"max_workers": 1', b'"max_workers": 2')
    assert changed != raw and len(changed) == len(raw)
    profile.manifest_path.write_bytes(changed)
    with pytest.raises(ValueError, match="changed"):
        profile.verify_source(authority)


def test_source_omission_is_rejected_without_hashing_model_payload(profile_record):
    profile = _read(profile_record)
    authority = capture_model_identity({"manifest": profile.manifest_path}, runtime_identity={},
                                       authorize=lambda _name, path: path.resolve(), max_file_bytes=65536,
                                       max_total_bytes=65536)
    with pytest.raises(ValueError, match="omits"):
        profile.verify_source(authority)


def test_empty_oversized_or_nested_json_is_bounded(profile_record, monkeypatch):
    import tensorfold.families.qwen4_exp.cuda.media_profile as module
    _, path, root = profile_record
    for raw in (b"", b"[]", b'{"a":' * 2000 + b"0" + b"}" * 2000):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            read_media_profile(path, model_dir=root, target_hidden=2560, world=8, context_rows=2048)
    monkeypatch.setattr(module, "MAX_HEADER_BYTES", 32)
    with pytest.raises(ValueError, match="byte bound"):
        _read(profile_record)
