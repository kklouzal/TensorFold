"""Explicit CPU admission for the single-encoder SM70 media path.

Parsing grants neither source authority nor resource/device qualification. The
startup owner must include required_input_files() in its immutable content
cohort, call verify_source() before retaining tensors, and separately adopt the
combined software authority and measured/bounded execution resources.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from tensorfold.cuda.tensor_file import (
    MAX_HEADER_BYTES, _metadata_object, _regular_stream, checkpoint_path,
)
from tensorfold.vision.media_worker import PROCESSOR_ID
from tensorfold.vision.replay import MediaEnvelope

from .mmproj_weights import VisionLayout, VisionPlan, plan_mmproj

SCHEMA = "tensorfold.sm70-media-profile"
VERSION = 1
_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1


def _object(value, names, label):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"media profile {label} requires exactly: {', '.join(sorted(names))}")
    return value


def _integer(value, name, *, minimum=1, maximum=_INT64_MAX):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"media profile {name} must be an integer in [{minimum}, {maximum}]")
    return value


def _stat_identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _capture(path):
    # read_metadata_json uses these same owned IO/strict JSON primitives. Keep
    # its captured bytes here so the content identity needs no second read.
    with _regular_stream(path) as stream:
        before = os.fstat(stream.fileno())
        if not 0 < before.st_size <= MAX_HEADER_BYTES:
            raise ValueError("media profile manifest exceeds the checkpoint metadata byte bound")
        raw = stream.read(before.st_size + 1)
        if (len(raw) != before.st_size
                or _stat_identity(os.fstat(stream.fileno())) != _stat_identity(before)):
            raise ValueError("media profile manifest changed during its read")
    return _metadata_object(raw, path), raw, _stat_identity(before)


@dataclass(frozen=True, slots=True)
class MediaExecution:
    encoder_rank: int
    input_rows: int
    encoder_patches: int
    projected_cache_rows: int
    max_rendered_bytes: int
    max_expanded_bytes: int
    vision_workspace_bytes: int


@dataclass(frozen=True, slots=True)
class MediaProfile:
    identity: str
    manifest_path: Path
    vision_path: Path
    layout: VisionLayout
    mmproj_plan: VisionPlan
    cpu: MediaEnvelope
    execution: MediaExecution
    processor_id: str
    manifest_size_bytes: int
    manifest_sha256: str
    vision_size_bytes: int
    vision_sha256: str
    _source_stats: tuple

    def required_input_files(self):
        """Actual files for the startup owner's complete model input closure."""
        return frozenset((self.manifest_path, self.vision_path))

    def expected_inputs(self):
        """Path -> (exact byte length, expected SHA256), never hash authority."""
        return MappingProxyType({self.manifest_path: (self.manifest_size_bytes, self.manifest_sha256),
                                 self.vision_path: (self.vision_size_bytes, self.vision_sha256)})

    def verify_source(self, authority):
        """Adopt only a live process-issued ModelIdentity, without rehashing.

        The owning loader preserves immutability from profile/header capture
        through cohort verification, tensor loading and serving. Stat checks
        detect changes; they do not replace that ownership requirement.
        """
        from tensorfold.engine.model_identity import ModelIdentity

        if type(authority) is not ModelIdentity:
            raise ValueError("media profile requires an issued ModelIdentity content authority")
        authority.verify_unchanged()
        expected = self.expected_inputs()
        stats = dict(self._source_stats)
        found = set()
        for content in authority.files:
            path = Path(content.path)
            if path in expected:
                if ((content.size, content.sha256) != expected[path]
                        or content.stat_identity != stats[path]):
                    raise ValueError("media profile source identity differs from the parsed execution input")
                found.add(path)
        if found != set(expected):
            raise ValueError("media profile content authority omits a required execution input")
        return self


def read_media_profile(manifest_path, *, model_dir, target_hidden, world, context_rows):
    """Resolve explicit admission and a complete CPU-only mmproj header plan.

    vision_workspace_bytes is an administrator hard bound, not a measured
    requirement. The runtime must enforce/warm/measure it before capacity is
    published. No request-processing limit or resource estimate is inferred.
    """
    _integer(target_hidden, "target_hidden", maximum=_INT32_MAX)
    _integer(world, "world", maximum=8)
    _integer(context_rows, "context_rows", maximum=_INT32_MAX)
    manifest_path = Path(manifest_path).resolve(strict=True)
    body, raw, manifest_stat = _capture(manifest_path)
    _object(body, {"schema", "version", "vision", "processor", "execution", "cpu"}, "record")
    if body["schema"] != SCHEMA or type(body["version"]) is not int or body["version"] != VERSION:
        raise ValueError("unsupported SM70 media profile schema/version")
    vision = _object(body["vision"], {"path", "size_bytes", "sha256", "layout"}, "vision")
    name = vision["path"]
    if (type(name) is not str or not name or "\\" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or name != PurePosixPath(name).as_posix()):
        raise ValueError("media profile vision path must be a normalized relative checkpoint path")
    vision_path = checkpoint_path(model_dir, name)
    if vision_path == manifest_path:
        raise ValueError("media profile manifest and vision source must be distinct")
    size = _integer(vision["size_bytes"], "vision.size_bytes")
    digest = vision["sha256"]
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("media profile vision.sha256 must be a lowercase SHA256")
    declared = _object(vision["layout"], {item.name for item in fields(VisionLayout)}, "vision.layout")
    for item in fields(VisionLayout):
        if item.name != "theta":
            _integer(declared[item.name], "vision.layout." + item.name, maximum=_INT32_MAX)
    layout = VisionLayout(**declared)
    if ((layout.depth, layout.hidden, layout.heads, layout.intermediate, layout.patch, layout.temporal,
         layout.merge, layout.channels, layout.positions) != (27, 1152, 16, 4304, 16, 2, 2, 3, 2304)
            or layout.output != target_hidden or target_hidden != 2560 or layout.theta != 10000.):
        raise ValueError("media profile requires the initial independent Qwen4Exp H16/D72 vision geometry")
    processor = _object(body["processor"], {"id"}, "processor")
    if processor["id"] != PROCESSOR_ID:
        raise ValueError("media profile requires the pinned CPU processor policy")
    execution = _object(body["execution"], {item.name for item in fields(MediaExecution)}, "execution")
    _integer(execution["encoder_rank"], "execution.encoder_rank", minimum=0, maximum=world - 1)
    _integer(execution["input_rows"], "execution.input_rows", maximum=context_rows)
    patches = _integer(execution["encoder_patches"], "execution.encoder_patches", maximum=_INT32_MAX)
    if patches % layout.merge**2:
        raise ValueError("media profile encoder patch bound requires complete merge-aligned units")
    _integer(execution["projected_cache_rows"], "execution.projected_cache_rows", minimum=0, maximum=context_rows)
    for name in ("max_rendered_bytes", "max_expanded_bytes", "vision_workspace_bytes"):
        _integer(execution[name], "execution." + name)
    cpu = _object(body["cpu"], {item.name for item in fields(MediaEnvelope)}, "cpu")
    # The existing envelope owns decoder/timing policy. Also keep integer
    # resources in the selected host's checked signed-int64 accounting domain.
    for name, value in cpu.items():
        if type(value) is int:
            _integer(value, "cpu." + name, minimum=0)
    if cpu["assumed_fps"] is not None:
        fps = cpu["assumed_fps"]
        if type(fps) is not list or len(fps) != 2:
            raise ValueError("media profile assumed_fps must be an explicit rational pair or null")
        for value in fps:
            _integer(value, "cpu.assumed_fps")
    envelope = MediaEnvelope.from_record(cpu)
    if envelope.max_replay_borrows < 2:
        raise ValueError("media profile requires two replay borrows for one live encoder job")
    if envelope.max_expanded_rows > context_rows:
        raise ValueError("media profile CPU expanded rows exceed the selected context")
    before = _stat_identity(vision_path.stat())
    if before[2] != size:
        raise ValueError("media profile vision byte length differs from its actual source")
    plan = plan_mmproj(vision_path, layout)
    if _stat_identity(vision_path.stat()) != before:
        raise ValueError("media profile vision source changed during header planning")
    if plan.resident_bytes > _INT64_MAX:
        raise ValueError("media profile retained vision weights exceed signed-int64 accounting")
    manifest_digest = hashlib.sha256(raw).hexdigest()
    identity = hashlib.sha256(json.dumps({"manifest_sha256": manifest_digest, "profile": body},
                                        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return MediaProfile(SCHEMA + ":" + identity, manifest_path, vision_path, layout, plan, envelope,
                        MediaExecution(**execution), PROCESSOR_ID, len(raw), manifest_digest, size, digest,
                        ((manifest_path, manifest_stat), (vision_path, before)))
