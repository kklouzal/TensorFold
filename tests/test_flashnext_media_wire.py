"""Bounded real media metadata reconstruction; no replay files or GPU execution."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json
from types import SimpleNamespace

import pytest

from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
from tensorfold.families.qwen4_exp.cuda.media_wire import pack_media, unpack_media
from tensorfold.families.qwen4_exp.media_plan import MediaPlanContract
from tensorfold.vision.media_frontend import PreparedReplayMedia
from tensorfold.vision.replay import MediaProbe
from test_qwen4_exp_media_plan import NAMES, planner, probe
from test_qwen4_exp_media_plan import tokenizer as tokenizer_source
from test_media_replay import envelope as make_envelope


@pytest.fixture
def tokenizer():
    return tokenizer_source.__wrapped__()


@pytest.fixture
def prepared(tokenizer):
    image, video, start, end = NAMES
    owner = planner(tokenizer)
    envelope = make_envelope()
    probes = tuple(replace(value, envelope_id=envelope.identity) for value in
                   (probe("image", 0), probe("video", 1), probe("image", 2, grid=(1, 4, 4))))
    plan = owner(f"hello {start}{image}{end} hello {start}{video}{end} hello {start}{image}{end}",
                 probes, max_prompt_tokens=512, max_visual_tokens=None)
    # The wire must never inspect or copy these actual owner's subsystems.
    media = PreparedReplayMedia(plan, probes, SimpleNamespace(), SimpleNamespace())
    return media, owner.contract, envelope


def test_exact_json_roundtrip_has_no_tokenizer_pixels_or_replay_owner(prepared, monkeypatch):
    import sys

    media, contract, envelope = prepared
    record = pack_media(media, epoch=14, max_bytes=65536, envelope=envelope)
    assert set(record) == {"schema", "version", "epoch", "probes", "media_identity"}
    body = json.loads(json.dumps(record))
    with monkeypatch.context() as blocked:
        blocked.setitem(sys.modules, "tokenizers", None)
        blocked.setitem(sys.modules, "transformers", None)
        blocked.setattr("builtins.open", lambda *_a, **_k: pytest.fail("follower opened a replay file"))
        remote = unpack_media(body, token_ids=media.token_ids, contract=contract, max_bytes=65536, envelope=envelope)
    assert remote.plan == media.plan and remote.probes == media.probes and remote.epoch == 14
    assert remote.token_ids == media.token_ids
    assert all(not hasattr(remote, name) for name in ("replay", "executor", "close", "tokenizer", "pixels"))
    with pytest.raises(FrozenInstanceError):
        remote.epoch = 0


def test_exact_serialized_byte_boundary_is_checked_on_both_sides(prepared):
    media, contract, envelope = prepared
    record = pack_media(media, epoch=1, max_bytes=65536, envelope=envelope)
    maximum = len(json.dumps(record, separators=(",", ":")).encode())
    assert pack_media(media, epoch=1, max_bytes=maximum, envelope=envelope) == record
    unpack_media(record, token_ids=media.token_ids, contract=contract, max_bytes=maximum, envelope=envelope)
    with pytest.raises(ValueError, match="byte envelope"):
        pack_media(media, epoch=1, max_bytes=maximum - 1, envelope=envelope)
    with pytest.raises(ValueError, match="byte envelope"):
        unpack_media(record, token_ids=media.token_ids, contract=contract, max_bytes=maximum - 1, envelope=envelope)


@pytest.mark.parametrize("maximum", [None, True, 0, -1, MAX_MESSAGE_BYTES + 1])
def test_no_implicit_or_oversized_transport_default(prepared, maximum):
    media, contract, envelope = prepared
    with pytest.raises(ValueError, match="explicit byte bound"):
        pack_media(media, epoch=1, max_bytes=maximum, envelope=envelope)
    with pytest.raises(ValueError, match="explicit byte bound"):
        unpack_media({}, token_ids=media.token_ids, contract=contract, max_bytes=maximum, envelope=envelope)


@pytest.mark.parametrize("epoch", [None, True, -1, 1 << 63])
def test_epoch_is_not_coerced_or_confused_with_slot_generation(prepared, epoch):
    media, contract, envelope = prepared
    with pytest.raises(ValueError, match="CPU epoch"):
        pack_media(media, epoch=epoch, max_bytes=65536, envelope=envelope)
    record = pack_media(media, epoch=1, max_bytes=65536, envelope=envelope)
    record["epoch"] = epoch
    with pytest.raises(ValueError, match="CPU epoch"):
        unpack_media(record, token_ids=media.token_ids, contract=contract, max_bytes=65536, envelope=envelope)


@pytest.mark.parametrize("defect", ["extra", "missing", "version-bool", "schema", "profile-claim", "probe-extra",
    "ref-extra", "nested-string", "float", "tuple-array", "grid-shape", "pts-shape", "selected-count",
    "nested-array", "recursive", "large-string", "large-integer"])
def test_invalid_schema_fails_before_probe_or_plan_expansion(prepared, monkeypatch, defect):
    media, contract, envelope = prepared
    record = deepcopy(pack_media(media, epoch=1, max_bytes=65536, envelope=envelope))
    item = record["probes"][1]
    if defect == "extra":
        record["unknown"] = 1
    elif defect == "missing":
        del record["epoch"]
    elif defect == "version-bool":
        record["version"] = True
    elif defect == "schema":
        record["schema"] = []
    elif defect == "profile-claim":
        record["profile_identity"] = "f" * 64
    elif defect == "probe-extra":
        item["pixel_values"] = []
    elif defect == "ref-extra":
        item["ref"]["owned"] = True
    elif defect == "nested-string":
        item["ref"]["path"] = {"nested": "path"}
    elif defect == "float":
        item["source_frames"] = 4.0
    elif defect == "tuple-array":
        item["selected_indices"] = tuple(item["selected_indices"])
    elif defect == "grid-shape":
        item["grid_thw"] = [2, 4, 4, 4]
    elif defect == "pts-shape":
        item["selected_pts"][0] = [0]
    elif defect == "selected-count":
        item["selected_indices"] = list(range(17))
    elif defect == "nested-array":
        item["selected_pts"][0] = [[0], 1]
    elif defect == "recursive":
        item["selected_pts"][0] = item["selected_pts"]
    elif defect == "large-string":
        item["ref"]["path"] = "/" + "x" * 65536
    else:
        item["selected_pts"][0] = [1 << (65536 * 4 + 1), 1]

    def reject(*_args, **_kwargs):
        pytest.fail("unvalidated wire reached probe/plan expansion")

    monkeypatch.setattr(MediaProbe, "from_record", reject)
    monkeypatch.setattr(MediaPlanContract, "from_expanded", reject)
    with pytest.raises(ValueError):
        unpack_media(record, token_ids=media.token_ids, contract=contract, max_bytes=65536, envelope=envelope)


def test_claimed_digest_and_stale_trusted_profile_are_both_refused(prepared):
    media, contract, envelope = prepared
    record = pack_media(media, epoch=1, max_bytes=65536, envelope=envelope)
    bad = {**record, "media_identity": "0" * 64}
    with pytest.raises(ValueError, match="canonical probe identity"):
        unpack_media(bad, token_ids=media.token_ids, contract=contract, max_bytes=65536, envelope=envelope)
    changed = replace(contract, identities=("5" * 64, *contract.identities[1:]))
    with pytest.raises(ValueError, match="trusted profile"):
        unpack_media(record, token_ids=media.token_ids, contract=changed, max_bytes=65536, envelope=envelope)


def test_paths_are_observational_and_never_supply_ownership(prepared):
    media, contract, envelope = prepared
    record = pack_media(media, epoch=8, max_bytes=65536, envelope=envelope)
    for item in record["probes"]:
        item["ref"]["path"] = "/this/path/does/not/exist"
    remote = unpack_media(record, token_ids=media.token_ids, contract=contract, max_bytes=65536, envelope=envelope)
    assert remote.plan == media.plan and all(probe.ref.path.endswith("not/exist") for probe in remote.probes)


def test_supported_exact_pts_rationals_are_not_silently_narrowed_to_int64(tokenizer):
    _, video, start, end = NAMES
    owner = planner(tokenizer)
    envelope = make_envelope()
    probes = (replace(probe("video", 0, pts=tuple((10**30 + i * 10**25, 10**25) for i in range(4))),
                      envelope_id=envelope.identity),)
    plan = owner(f"{start}{video}{end}", probes, max_prompt_tokens=512, max_visual_tokens=None)
    media = PreparedReplayMedia(plan, probes, None, None)
    record = pack_media(media, epoch=1, max_bytes=65536, envelope=envelope)
    remote = unpack_media(record, token_ids=plan.token_ids, contract=owner.contract, max_bytes=65536, envelope=envelope)
    assert remote.probes[0].selected_pts == probes[0].selected_pts and remote.plan == plan


def test_stale_mutated_leader_probe_set_cannot_be_published(prepared):
    media, _, envelope = prepared
    media.probes = (replace(probe("image", 0, sha="f" * 64), envelope_id=envelope.identity), *media.probes[1:])
    with pytest.raises(ValueError, match="changed after canonical planning"):
        pack_media(media, epoch=1, max_bytes=65536, envelope=envelope)


@pytest.mark.parametrize("policy", ["identity", "selected", "images", "videos", "upload", "total-upload",
    "dimensions", "pixels", "decoded-bytes", "expanded", "allocation", "scan", "duration", "probe-bytes"])
def test_actual_cpu_envelope_is_required_on_both_wire_sides(prepared, policy):
    media, contract, envelope = prepared
    record = pack_media(media, epoch=1, max_bytes=65536, envelope=envelope)
    changes = {"selected": {"max_selected_frames": 2}, "images": {"max_images": 1},
        "videos": {"max_videos": 0}, "upload": {"max_upload_bytes": 100},
        "total-upload": {"max_total_upload_bytes": 300}, "dimensions": {"max_dimension": 63},
        "pixels": {"max_total_decoded_pixels": 100}, "decoded-bytes": {"max_decoded_bytes": 100},
        "expanded": {"max_expanded_rows": 10}, "allocation": {"max_single_allocation_bytes": 100},
        "scan": {"max_scan_frames": 2}, "duration": {"max_video_seconds": .05},
        "probe-bytes": {"max_result_bytes": 100}}
    if policy == "identity":
        bad = replace(envelope, max_workers=envelope.max_workers + 1)
    else:
        bad = replace(envelope, **changes[policy])
        # Keep identity consistent with the chosen policy, so this control
        # checks real limits rather than relying on only a digest mismatch.
        probes = tuple(replace(value, envelope_id=bad.identity) for value in media.probes)
        media = PreparedReplayMedia(contract.from_expanded(media.token_ids, probes), probes, None, None)
        record["probes"] = [dict(item, envelope_id=bad.identity) for item in record["probes"]]
        record["media_identity"] = media.plan.media_identity
    with pytest.raises(ValueError):
        pack_media(media, epoch=1, max_bytes=65536, envelope=bad)
    with pytest.raises(ValueError):
        unpack_media(record, token_ids=media.token_ids, contract=contract, max_bytes=65536, envelope=bad)


def test_untrusted_envelope_or_underreported_raw_pixels_cannot_be_accepted(prepared):
    media, contract, envelope = prepared
    record = pack_media(media, epoch=1, max_bytes=65536, envelope=envelope)
    with pytest.raises(ValueError, match="trusted CPU envelope"):
        unpack_media(record, token_ids=media.token_ids, contract=contract, max_bytes=65536, envelope=asdict(envelope))
    record["probes"][1]["decoded_pixels"] -= 1
    with pytest.raises(ValueError, match="admitted CPU envelope"):
        unpack_media(record, token_ids=media.token_ids, contract=contract, max_bytes=65536, envelope=envelope)



def test_image_byte_accounting_cannot_be_underreported_even_with_matching_identity(prepared):
    media, contract, envelope = prepared
    probes = (replace(media.probes[0], decoded_bytes=1), *media.probes[1:])
    plan = contract.from_expanded(media.token_ids, probes)
    changed = PreparedReplayMedia(plan, probes, None, None)
    with pytest.raises(ValueError, match="admitted CPU envelope"):
        pack_media(changed, epoch=1, max_bytes=65536, envelope=envelope)
    # Forge both the metadata and its hash; policy validation must be a real
    # value check, rather than treating a claimed digest as authenticity.
    record = pack_media(media, epoch=1, max_bytes=65536, envelope=envelope)
    record["probes"][0]["decoded_bytes"] = 1
    record["media_identity"] = plan.media_identity
    with pytest.raises(ValueError, match="admitted CPU envelope"):
        unpack_media(record, token_ids=media.token_ids, contract=contract, max_bytes=65536, envelope=envelope)
