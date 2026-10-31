"""Bounded TP media metadata, with no follower tokenizer or replay ownership.

The leader retains its real encoded replay lease through every rank's last use.
Paths in these records are observational identity metadata: followers never open
or retire them. Canonical prompt IDs already travel in the enclosing request.
The caller compares the CPU epoch to its request owner; it is not a slot generation.
"""
from dataclasses import dataclass
from fractions import Fraction
import json

from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
from tensorfold.vision.replay import MediaEnvelope, MediaProbe

from ..media_plan import MediaPlanContract, QwenMediaPlan

SCHEMA = "tensorfold.tp-media"
VERSION = 1
_FIELDS = frozenset({"schema", "version", "epoch", "probes", "media_identity"})
_PROBE = frozenset({"ref", "envelope_id", "processor_id", "width", "height", "source_frames",
                    "selected_indices", "selected_pts", "fps", "timestamp_policy", "resized_hw", "grid_thw",
                    "decoded_pixels", "decoded_bytes"})
_REF = frozenset({"path", "sha256", "byte_length", "ordinal", "kind", "detail", "version"})


@dataclass(frozen=True)
class RemotePreparedMedia:
    """Only immutable metadata; deliberately has no executor/replay/close API."""
    plan: QwenMediaPlan
    probes: tuple[MediaProbe, ...]
    epoch: int

    @property
    def token_ids(self):
        return self.plan.token_ids


def _limit(value):
    if type(value) is not int or not 0 < value <= MAX_MESSAGE_BYTES:
        raise ValueError("TP media requires an explicit byte bound within the rank transport envelope")
    return value


def _epoch(value):
    if type(value) is not int or not 0 <= value < 1 << 63:
        raise ValueError("TP media CPU epoch must be a nonnegative signed-int64 value")


class _Budget:
    """Cumulative minimum JSON bytes bound validation work before expansion."""
    def __init__(self, maximum):
        self.remaining = maximum

    def take(self, count):
        if count > self.remaining:
            raise ValueError("TP media metadata exceeds its admitted byte envelope")
        self.remaining -= count


def _object(value, fields):
    if (type(value) is not dict or len(value) != len(fields)
            or any(type(key) is not str for key in value) or value.keys() != fields):
        raise ValueError("TP media has unknown, missing or nested schema fields")


def _string(value, maximum, budget):
    if type(value) is not str or len(value) > maximum:
        raise ValueError("TP media string exceeds its typed transport bound")
    budget.take(len(value) + 2)  # Quotes plus at least one encoded byte per codepoint.


def _integer(value, maximum, budget):
    # Do not narrow exact PTS/time-base rationals to int64. Valid decoder
    # products may be wider. A lower bound on decimal digits limits cumulative
    # work before the real JSON encoder enforces exact bytes/conversion rules.
    if type(value) is not int or value.bit_length() > maximum * 4:
        raise ValueError("TP media integer exceeds its typed transport bound")
    budget.take(max(1, 1 + 3 * (value.bit_length() - 1) // 10))  # 0.3 < log10(2).


def _integers(value, maximum, budget, *, count=None):
    if (type(value) is not list or len(value) > min(maximum, budget.remaining)
            or count is not None and len(value) != count):
        raise ValueError("TP media list exceeds its typed count/shape bound")
    for item in value:
        _integer(item, maximum, budget)


def _probe(record, maximum, selected_maximum, budget):
    _object(record, _PROBE)
    _object(record["ref"], _REF)
    for key in ("path", "sha256", "kind", "detail"):
        _string(record["ref"][key], maximum, budget)
    for key in ("byte_length", "ordinal", "version"):
        _integer(record["ref"][key], maximum, budget)
    for key in ("envelope_id", "processor_id", "timestamp_policy"):
        _string(record[key], maximum, budget)
    for key in ("width", "height", "source_frames", "decoded_pixels", "decoded_bytes"):
        _integer(record[key], maximum, budget)
    _integers(record["resized_hw"], maximum, budget, count=2)
    _integers(record["grid_thw"], maximum, budget, count=3)
    selected = record["selected_indices"]
    if type(selected) is not list or len(selected) > min(selected_maximum, budget.remaining):
        raise ValueError("TP media selected frames exceed its admitted temporal-unit bound")
    _integers(selected, maximum, budget)
    pts = record["selected_pts"]
    if type(pts) is not list or len(pts) != len(selected):
        raise ValueError("TP media timestamp list differs from selected frame count")
    for pair in pts:
        _integers(pair, maximum, budget, count=2)
    if record["fps"] is not None:
        _integers(record["fps"], maximum, budget, count=2)


def _size(record, maximum):
    # Validated fixed-depth schema only. iterencode avoids a second complete
    # encoded message. Escaping scratch is bounded by the explicit byte cap;
    # no arbitrary parser recursion, type coercion or integer expansion occurs.
    total = 0
    for part in json.JSONEncoder(separators=(",", ":"), allow_nan=False).iterencode(record):
        total += len(part)  # Default ensure_ascii=True: characters are bytes.
        if total > maximum:
            raise ValueError("TP media metadata exceeds its admitted byte envelope")


def _record(probe):
    value = probe.record()
    for name in ("selected_indices", "resized_hw", "grid_thw"):
        value[name] = list(value[name])
    value["selected_pts"] = [list(pair) for pair in value["selected_pts"]]
    if value["fps"] is not None:
        value["fps"] = list(value["fps"])
    return value


def _envelope(value):
    if type(value) is not MediaEnvelope:
        raise ValueError("TP media requires the resident profile's trusted CPU envelope")


def _policy(probes, envelope):
    # Reuse the worker's pure dimension/allocation rules. Importing this module
    # starts no worker, loads no decoder, opens no file and allocates no pixels.
    from tensorfold.vision.media_worker import _dimensions, _unit_allocation

    identity = envelope.identity
    images = sum(probe.ref.kind == "image" for probe in probes)
    videos = len(probes) - images
    if (images > envelope.max_images or videos > envelope.max_videos or len(probes) > envelope.max_replay_items
            or sum(probe.ref.byte_length for probe in probes) > min(envelope.max_total_upload_bytes, envelope.max_replay_bytes)
            or sum(probe.decoded_pixels for probe in probes) > envelope.max_total_decoded_pixels
            or sum(probe.decoded_bytes for probe in probes) > envelope.max_decoded_bytes
            or sum(probe.visual_rows for probe in probes) > envelope.max_expanded_rows):
        raise ValueError("TP media totals exceed the admitted CPU envelope")
    for probe in probes:
        if (probe.envelope_id != identity or probe.ref.byte_length > envelope.max_upload_bytes
                or probe.source_frames > envelope.max_scan_frames
                or len(probe.selected_indices) > envelope.max_selected_frames
                or probe.decoded_pixels != probe.width * probe.height * probe.source_frames
                or probe.ref.kind == "image" and probe.decoded_bytes != probe.decoded_pixels * 4):
            raise ValueError("TP media probe differs from its admitted CPU envelope")
        _dimensions(probe.width, probe.height, envelope)
        _unit_allocation(probe, envelope)
        if probe.ref.kind == "video":
            rate = Fraction(*probe.fps)
            times = tuple(Fraction(*pair) for pair in probe.selected_pts)
            if (any(left >= right for left, right in zip(times, times[1:]))
                    or max(Fraction(probe.source_frames, 1) / rate, times[-1] - times[0]) > envelope.max_video_seconds
                    or probe.timestamp_policy == "explicit-assumed-fps-v1" and (
                        envelope.assumed_fps is None or rate != Fraction(*envelope.assumed_fps)
                        or any(time != Fraction(index, 1) / rate for time, index in zip(times, probe.selected_indices)))):
                raise ValueError("TP media timestamps differ from the admitted CPU policy")


def pack_media(prepared, *, epoch, max_bytes, envelope):
    from tensorfold.vision.media_frontend import PreparedReplayMedia

    maximum = _limit(max_bytes)
    _envelope(envelope)
    _epoch(epoch)
    if (type(prepared) is not PreparedReplayMedia or type(prepared.plan) is not QwenMediaPlan
            or type(prepared.probes) is not tuple or not prepared.probes
            or any(type(probe) is not MediaProbe for probe in prepared.probes)):
        raise ValueError("TP media packing requires the actual leader's prepared media owner")
    if len(prepared.probes) > min(maximum, envelope.max_replay_items, envelope.max_images + envelope.max_videos):
        raise ValueError("TP media probe count exceeds its transport bound")
    _policy(prepared.probes, envelope)
    # Bound every per-probe copy before dataclass serialization. A temporal
    # group owns at most two selected frames, including odd final padding.
    copy_items = 0
    for probe in prepared.probes:
        copy_items += len(probe.selected_indices) + 2 * len(probe.selected_pts)
        if copy_items > maximum:
            raise ValueError("TP media selected frames exceed its transport bound")
    record = {"schema": SCHEMA, "version": VERSION, "epoch": epoch,
              "probes": [_record(probe) for probe in prepared.probes], "media_identity": prepared.plan.media_identity}
    budget = _Budget(maximum)
    for item in record["probes"]:
        _probe(item, maximum, min(maximum, envelope.max_selected_frames), budget)
        _size({"probe": item}, min(maximum, envelope.max_result_bytes))
    identity = record["media_identity"]
    if type(identity) is not str or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("TP media requires an exact canonical media identity")
    _string(identity, 64, budget)
    _size(record, maximum)
    if prepared.plan.probe_identities != tuple(probe.identity for probe in prepared.probes):
        raise ValueError("leader media probe metadata changed after canonical planning")
    return record


def unpack_media(record, *, token_ids, contract, max_bytes, envelope):
    maximum = _limit(max_bytes)
    _envelope(envelope)
    if type(contract) is not MediaPlanContract:
        raise ValueError("TP media requires the trusted resident model's immutable contract")
    _object(record, _FIELDS)
    if (type(record["schema"]) is not str or record["schema"] != SCHEMA
            or type(record["version"]) is not int or record["version"] != VERSION):
        raise ValueError("unknown TP media schema/version")
    _epoch(record["epoch"])
    identity = record["media_identity"]
    if type(identity) is not str or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("TP media requires an exact canonical media identity")
    records = record["probes"]
    if type(records) is not list or not 0 < len(records) <= min(contract.max_segments, maximum, envelope.max_replay_items,
                                                    envelope.max_images + envelope.max_videos):
        raise ValueError("TP media probe count exceeds the admitted contract")
    budget = _Budget(maximum)
    _string(identity, 64, budget)
    for item in records:
        _probe(item, maximum, min(2 * contract.max_segments, maximum, envelope.max_selected_frames), budget)
        _size({"probe": item}, min(maximum, envelope.max_result_bytes))
    _size(record, maximum)
    probes = tuple(MediaProbe.from_record(item) for item in records)
    _policy(probes, envelope)
    plan = contract.from_expanded(token_ids, probes)
    if plan.media_identity != identity:
        raise ValueError("TP media differs from the trusted profile or canonical probe identity")
    return RemotePreparedMedia(plan, probes, record["epoch"])
