"""Selected-tokenizer media expansion with the pinned processor's exact syntax.

The profile owner lends an immutable, untruncated native tokenizer and complete
model/software identities. Probes already own bounded immutable encoded replay;
this layer retains geometry and canonical IDs, never normalized/projected rows.
"""

from dataclasses import dataclass, fields
import hashlib
import json
import math
import re
from types import MethodType, SimpleNamespace

from .media_positions import MediaPositionPlan, PositionRun, plan_media_positions
from .position_profile import MAX_POSITION


@dataclass(frozen=True)
class MediaUnit:
    run: PositionRun
    probe_index: int


@dataclass(frozen=True)
class QwenMediaPlan:
    positions: MediaPositionPlan
    units: tuple[MediaUnit, ...]
    probe_identities: tuple[str, ...]
    media_identity: str
    profile_identity: str

    @property
    def token_ids(self):
        return self.positions.token_ids


@dataclass(frozen=True)
class MediaPlanContract:
    """Trusted immutable geometry/identity; carries no tokenizer or replay owner."""
    identities: tuple[str, str, str, str]
    processor_id: str
    roles: tuple[int, int, int, int]
    vocab_size: int
    max_rows: int
    max_segments: int
    max_visual_rows: int
    max_total_visual_rows: int

    def __post_init__(self):
        if (type(self.identities) is not tuple or len(self.identities) != 4
                or any(type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                       for value in self.identities)
                or type(self.processor_id) is not str or not 0 < len(self.processor_id) <= 128
                or any(type(value) is not int or value <= 0 for value in
                       (self.vocab_size, self.max_rows, self.max_segments, self.max_visual_rows, self.max_total_visual_rows))
                or self.vocab_size > MAX_POSITION or self.max_rows >= MAX_POSITION
                or self.max_visual_rows > self.max_total_visual_rows or self.max_total_visual_rows > self.max_rows
                or type(self.roles) is not tuple or len(self.roles) != 4
                or any(type(value) is not int or not 0 <= value < self.vocab_size for value in self.roles)
                or len(set(self.roles)) != 4):
            raise ValueError("media contract requires admitted immutable identities, token roles and geometry")

    def record(self):
        body = {field.name: getattr(self, field.name) for field in fields(self)}
        body.update(identities=list(self.identities), roles=list(self.roles))
        return {"schema": "tensorfold.media-plan-contract", "version": 1, **body}

    @classmethod
    def from_record(cls, record):
        names = {field.name for field in fields(cls)}
        if (type(record) is not dict or len(record) != len(names) + 2
                or any(type(key) is not str for key in record) or record.keys() != names | {"schema", "version"}
                or type(record["schema"]) is not str or record["schema"] != "tensorfold.media-plan-contract"
                or type(record["version"]) is not int or record["version"] != 1
                or type(record["identities"]) is not list or len(record["identities"]) != 4
                or type(record["roles"]) is not list or len(record["roles"]) != 4):
            raise ValueError("unknown or malformed media planning contract record")
        values = {name: record[name] for name in names}
        values.update(identities=tuple(values["identities"]), roles=tuple(values["roles"]))
        return cls(**values)

    def check_probes(self, probes, *, visual_limit=None):
        from tensorfold.vision.replay import MediaProbe

        if (type(probes) is not tuple or not 0 < len(probes) <= self.max_segments
                or any(type(probe) is not MediaProbe or probe.processor_id != self.processor_id
                       or probe.ref.ordinal != index for index, probe in enumerate(probes))):
            raise ValueError("media probes differ from their admitted processor/order contract")
        if visual_limit is not None and (type(visual_limit) is not int or visual_limit <= 0):
            raise ValueError("media visual allowance must be an explicit positive count")
        limit = self.max_total_visual_rows if visual_limit is None else min(self.max_total_visual_rows, visual_limit)
        if (sum(probe.visual_rows for probe in probes) > limit
                or sum(probe.unit_count for probe in probes) > self.max_segments
                or any(probe.grid_thw[1] * probe.grid_thw[2] // 4 > self.max_visual_rows for probe in probes)):
            raise ValueError("processed media exceeds its declared expanded-row or complete-unit allowance")
        times = []
        for probe in probes:
            means = None
            if probe.ref.kind == "video":
                try:
                    means = tuple(float(value) for value in probe.temporal_timestamps())
                except OverflowError as error:
                    raise ValueError("media timestamps have no finite processor representation") from error
                if any(not math.isfinite(value) or not math.isfinite(value + value) for value in means):
                    raise ValueError("media timestamps have no finite processor representation")
            times.append(means)
        return tuple(times)

    def from_expanded(self, token_ids, probes):
        """One geometry/unit-map/identity algorithm for leader and every follower."""
        self.check_probes(probes)
        return self._finish(token_ids, probes)

    def _finish(self, token_ids, probes):
        # Private shared finish: both public entry points already validated the
        # immutable contract/probe values; no public bypass flag is accepted.
        images = [probe.grid_thw for probe in probes if probe.ref.kind == "image"]
        videos = [probe.grid_thw for probe in probes if probe.ref.kind == "video"]
        ordered_images = [index for index, probe in enumerate(probes) if probe.ref.kind == "image"]
        ordered_videos = [index for index, probe in enumerate(probes) if probe.ref.kind == "video"]
        positions = plan_media_positions(token_ids, image_grids=images, video_grids=videos,
            image_token_id=self.roles[0], video_token_id=self.roles[1], vision_start_token_id=self.roles[2],
            vision_end_token_id=self.roles[3], vocab_size=self.vocab_size, merge=2,
            max_rows=self.max_rows, max_segments=self.max_segments, max_visual_rows=self.max_visual_rows)
        units = tuple(MediaUnit(run, (ordered_images if run.kind == "image" else ordered_videos)[run.item])
                      for run in positions.visual_runs)
        expected = ((index, group) for index, probe in enumerate(probes) for group in range(probe.unit_count))
        if any(unit.probe_index != index or unit.run.temporal_group != group
               for unit, (index, group) in zip(units, expected, strict=True)):
            raise ValueError("canonical media order differs from the immutable probe sequence")
        probe_identities = tuple(probe.identity for probe in probes)
        identity = {"schema": "tensorfold.qwen-media-plan-v1", "profile_tokenizer_template_vision": self.identities,
                    "processor": self.processor_id, "ordered_media": probe_identities,
                    "timestamp_adapter": "exact-pair-mean-float-pinned-qwen3vl-format-v1"}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return QwenMediaPlan(positions, units, probe_identities, digest, self.identities[0])


class QwenMediaPlanner:
    def __init__(self, tokenizer, *, profile_identity, tokenizer_identity, template_identity, vision_identity,
                 processor_id, expected_image_id, max_rows, max_segments, max_visual_rows,
                 max_total_visual_rows, max_rendered_bytes, max_expanded_bytes):
        from tokenizers import Tokenizer

        identities = profile_identity, tokenizer_identity, template_identity, vision_identity
        if (type(tokenizer) is not Tokenizer or tokenizer.padding is not None or tokenizer.truncation is not None
                or any(type(value) is not int or value <= 0 for value in (max_rendered_bytes, max_expanded_bytes))):
            raise ValueError("media planning requires the selected untruncated tokenizer and explicit work bounds")
        names = ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")
        roles = tuple(tokenizer.token_to_id(name) for name in names)
        if (any(type(value) is not int or tokenizer.id_to_token(value) != name
                for value, name in zip(roles, names, strict=True)) or len(set(roles)) != 4
                or type(expected_image_id) is not int or roles[0] != expected_image_id):
            raise ValueError("selected media token IDs disagree with the model's declared image/PLE role")
        self.contract = MediaPlanContract(identities, processor_id, roles, tokenizer.get_vocab_size(with_added_tokens=True),
                                          max_rows, max_segments, max_visual_rows, max_total_visual_rows)
        self.tokenizer, self.names = tokenizer, names
        self.max_rendered_bytes, self.max_expanded_bytes = max_rendered_bytes, max_expanded_bytes
        self.video_marker = names[2] + names[1] + names[3]
        self.markers = re.compile("(" + "|".join(re.escape(name) for name in
                                               (self.video_marker, names[0], names[1])) + ")")

    def __call__(self, rendered, probes, *, max_prompt_tokens, max_visual_tokens):
        contract = self.contract
        if (type(rendered) is not str or len(rendered) > self.max_rendered_bytes
                or len(rendered.encode("utf-8")) > self.max_rendered_bytes
                or type(max_prompt_tokens) is not int or max_prompt_tokens <= 0
                or max_visual_tokens is not None and (type(max_visual_tokens) is not int or max_visual_tokens <= 0)):
            raise ValueError("media planning input differs from its admitted source/order/work contract")
        row_limit = min(contract.max_rows, max_prompt_tokens)
        times = contract.check_probes(probes, visual_limit=max_visual_tokens)
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        import numpy as np

        holder = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2),
                                 video_processor=SimpleNamespace(merge_size=2, temporal_patch_size=2),
                                 image_token=self.names[0], video_token=self.names[1],
                                 vision_start_token=self.names[2], vision_end_token=self.names[3])
        holder._calculate_timestamps = MethodType(Qwen3VLProcessor._calculate_timestamps, holder)
        expanded, byte_count, used = [], 0, 0
        for piece in self.markers.split(rendered):
            if piece in (self.names[0], self.names[1], self.video_marker):
                kind = "image" if piece == self.names[0] else "video"
                if used == len(probes) or probes[used].ref.kind != kind:
                    raise ValueError("rendered media markers disagree with immutable source order")
                probe = probes[used]
                # Positive T and even H/W give product=4*visual_rows. The
                # admitted total is below 2**31 rows, so every dimension and
                # the complete product fit the library's int64 metadata.
                grid = np.asarray([probe.grid_thw], dtype=np.int64)
                if kind == "image":
                    required = probe.visual_rows * len(self.names[0].encode("utf-8"))
                    if byte_count + required > self.max_expanded_bytes:
                        raise ValueError("expanded media text exceeds its admitted byte bound")
                    piece = Qwen3VLProcessor.replace_image_token(holder, {"image_grid_thw": grid}, 0)
                else:
                    means = times[used]
                    # The explicit actual-PTS adapter computes exact pair means
                    # first. Repeating each mean with fps=1 preserves that float
                    # through the unchanged processor's mean/format routine.
                    metadata = SimpleNamespace(fps=1., frames_indices=[v for value in means for v in (value, value)])
                    required = sum(len(f"<{value:.1f} seconds>".encode("utf-8")) for value in means)
                    required += probe.visual_rows * len(self.names[1].encode("utf-8"))
                    required += len(means) * len((self.names[2] + self.names[3]).encode("utf-8"))
                    if byte_count + required > self.max_expanded_bytes:
                        raise ValueError("expanded media text exceeds its admitted byte bound")
                    piece = Qwen3VLProcessor.replace_video_token(
                        holder, {"video_grid_thw": grid, "video_metadata": [metadata]}, 0)
                used += 1
            byte_count += len(piece.encode("utf-8"))
            if byte_count > self.max_expanded_bytes:
                raise ValueError("expanded media text exceeds its admitted byte bound")
            expanded.append(piece)
        if used != len(probes):
            raise ValueError("immutable media source has no rendered marker")
        ids = tuple(self.tokenizer.encode("".join(expanded), add_special_tokens=False).ids)
        if len(ids) > row_limit:
            raise ValueError("expanded media input exceeds the selected context window")
        return contract._finish(ids, probes)
