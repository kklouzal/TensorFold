# ruff: noqa: E402 - optional torch capability must be checked before CUDA-family imports.
"""Actual bounded method tensors survive catalog-style byte copies without RNG draws."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda.dflash2_config import DFlash2Config
from tensorfold.families.qwen4_exp.cuda.dflash2_runtime import DraftWindow
from tensorfold.families.qwen4_exp.cuda.media_mtp import MTPBoundary
from tensorfold.families.qwen4_exp.cuda.method_retention import (
    commit_method_restore, method_inventory, method_restore_resources, method_restore_targets, validate_method_restore,
)
from tensorfold.families.qwen4_exp.cuda.position_coordinates import copy_coordinates, issue_coordinates
from tensorfold.families.qwen4_exp.cuda.retained_parts import part_specs
from test_flashnext_dflash2 import small_config
from test_qwen4_exp_position_coordinates import policy
from test_flashnext_mtp_inputs import family as _registered_mtp_family  # noqa: F401 - pytest fixture registration

PROFILE, MEDIA = "a" * 64, "b" * 64


def state(*, position=19, generation=3, df=True, mtp=False):
    value = SimpleNamespace(pos=position, limit=128, mtp_len=position if mtp else 0, mtp_drafted=0,
        slot_generation=generation, compact_proposal_rows=4, mtp_hidden_width=16 if mtp else 0,
        media_mtp_enabled=mtp, media_input_span=None, mtp_boundary=None,
        media_inputs_enabled=mtp,
        execution_dtype=torch.float16, pos_dev=torch.tensor(position, dtype=torch.int32),
        media_profile_identity=PROFILE if mtp else None, media_identity=MEDIA if mtp else None,
        coordinate_profile_identity=PROFILE if mtp else None,
        media_mtp_position=policy("mtp", limit=128), rope_delta=0,
        retention_method_recovery_versions=2, dflash2=None)
    value.set_rope_delta = lambda delta: setattr(value, "rope_delta", delta)
    if df:
        value.dflash2 = DraftWindow(DFlash2Config.from_dict(small_config()), tp=(0, 1), context=128,
                                   device="cpu", generation=generation, profile_identity="same-runtime-profile")
        rows = torch.arange(2 * 2 * position * 8 * 4).reshape(2, 2, position, 8, 4).half()
        value.dflash2.append(rows, 0)
    return value


def stream(value):
    result = Stream(prompt=[1] * value.pos, count=64, sampling=Sampling(19, .7, 4, .9))
    result.out = [7]
    result.context = [*result.prompt, 7]
    result.drafts = [8, 9]
    return result


def transfer(source, destination, *, source_stream=None, destination_stream=None, pending_hc=None,
             continuation="same-request", recoveries=()):
    inventory = method_inventory(source, stream=source_stream, pending_hc=pending_hc, recoveries=recoveries)
    specs = part_specs(inventory.parts)
    # The actual catalog owns/seals these CPU bytes; this fixture only emulates
    # its copy boundary while exercising all real method views and validators.
    owned = tuple(part.tensor.detach().cpu().clone() for part in inventory.parts)
    targets = method_restore_targets(destination, inventory.metadata, specs,
                                     stream=destination_stream, continuation=continuation)
    for part, payload in zip(targets, owned, strict=True):
        part.tensor.copy_(payload)
    handoff = commit_method_restore(destination, inventory.metadata, targets,
                                    stream=destination_stream, continuation=continuation)
    return inventory, targets, handoff


@pytest.mark.parametrize("position", [0, 5, 8, 19])
def test_actual_wrapped_dflash_ring_excludes_uninitialized_rows_and_restores_new_generation(position):
    original = state(position=position)
    destination = state(position=0, generation=8)
    original.dflash2.kv[:, :, min(position, 8):].fill_(float("nan"))
    inventory, targets, _ = transfer(original, destination)
    available = min(position, original.dflash2.capacity)
    assert sum(part.spec().nbytes for part in inventory.parts) == 2 * 2 * available * 8 * 4 * 2
    assert all(part.tensor.is_contiguous() for part in inventory.parts)
    assert inventory.future_bytes == original.dflash2.nbytes * 3
    assert destination.dflash2.end == position and destination.dflash2.available == available
    assert destination.dflash2.generation == 8
    assert torch.equal(destination.dflash2.snapshot()["kv"], original.dflash2.snapshot()["kv"])
    assert part_specs(targets) == part_specs(inventory.parts)


def test_actual_deferred_media_hc_reissues_cpu_coordinate_rows_in_current_generation():
    original = state(position=6, df=False, mtp=True)
    original.mtp_len = 5
    issued = issue_coordinates(torch.tensor([[5, 3, 4]], dtype=torch.int64),
        position=original.media_mtp_position, profile_identity=PROFILE, generation=3, media_identity=MEDIA,
        row_start=5, max_rows=1, device="cpu")
    issued = copy_coordinates(issued, source_position=original.media_mtp_position, position=original.media_mtp_position,
                              row_start=5, row_count=1, dtype=torch.int64)
    hidden = torch.arange(16).reshape(1, 16).half()
    original.mtp_boundary = MTPBoundary(5, hidden, issued, original.media_mtp_position, 3)
    destination = state(position=0, generation=8, df=False, mtp=True)
    inventory, _targets, handoff = transfer(original, destination)
    restored = destination.mtp_boundary
    assert torch.equal(restored.hidden, hidden) and restored.hidden is not hidden
    assert torch.equal(restored.coordinates, torch.tensor([[5, 3, 4]]))
    restored.coordinate_record.proof.validate(restored.coordinates, destination.media_mtp_position,
        profile_identity=PROFILE, generation=8, media_identity=MEDIA, row_start=5, row_count=1)
    assert restored.coordinate_record.proof is not issued.proof
    assert restored.position is destination.media_mtp_position
    assert handoff["pending_hc"] is restored.hidden
    assert inventory.host == {}
    assert inventory.future_bytes == (16 * 2 + 24) * 3
    assert method_restore_resources(destination, inventory.metadata) == {"host_bytes": 24, "device_bytes": 12}


def test_legacy_text_hc_and_already_drawn_proposals_remain_exact_without_sampling(monkeypatch):
    original = state(position=6, df=False, mtp=True)
    original.media_profile_identity = original.media_identity = None
    original.mtp_len, original.mtp_drafted = 7, 2
    hidden = torch.arange(16).reshape(1, 16).half()
    request = stream(original)
    destination = state(position=0, generation=8, df=False, mtp=True)
    destination.media_profile_identity = destination.media_identity = None
    from tensorfold.engine import exact_sampling
    monkeypatch.setattr(exact_sampling, "uniform", lambda *_a: pytest.fail("retention consumed a draw"))
    before = (list(request.out), list(request.drafts), request.sampling)
    inventory, _targets, handoff = transfer(original, destination, source_stream=request,
                                           destination_stream=request, pending_hc=hidden)
    assert handoff["mtp_len"] == 7 and handoff["mtp_drafted"] == 2
    assert destination.mtp_len == destination.mtp_drafted == 0  # core owns frontier/device-counter publication
    assert torch.equal(handoff["pending_hc"], hidden)
    assert handoff["draws"]["pending_anchor"] == 7
    assert handoff["draws"]["drawn_proposals"] == [8, 9]
    assert (request.out, request.drafts, request.sampling) == before
    assert inventory.metadata["mtp_len"] != inventory.metadata["mtp_len"] - inventory.metadata["mtp_drafted"]


def test_new_turn_keeps_fresh_request_draws_and_rejects_changed_profile_before_allocation(monkeypatch):
    original = state(position=6, df=False)
    old, fresh = stream(original), stream(original)
    fresh.sampling = Sampling(91, 0)
    fresh.out, fresh.drafts, fresh.context = [], [], []
    destination = state(position=0, generation=8, df=False)
    inventory, _targets, handoff = transfer(original, destination, source_stream=old,
        destination_stream=fresh, continuation="new-turn")
    assert fresh.sampling.seed == 91 and fresh.out == fresh.drafts == [] and handoff["draws"] is None
    specs = part_specs(inventory.parts)
    bad = {**inventory.metadata, "target": 129}
    monkeypatch.setattr(torch, "empty", lambda *_a, **_kw: pytest.fail("metadata refusal allocated tensors"))
    with pytest.raises(ValueError, match="target frontier"):
        method_restore_targets(destination, bad, specs, stream=fresh, continuation="new-turn")


@pytest.mark.parametrize("failure", ["feature", "span", "missing-hc", "recovery-budget"])
def test_unresolved_owners_or_unreserved_future_are_explicit_refusals(failure):
    original = state(position=6, mtp=True, df=failure != "missing-hc")
    if failure == "feature":
        original.dflash2.pending = object()
    elif failure == "span":
        original.media_input_span = object()
    elif failure == "missing-hc":
        original.mtp_len = 5
    else:
        del original.retention_method_recovery_versions
    with pytest.raises(ValueError):
        method_inventory(original)


def test_missing_or_retyped_tensor_parts_refuse_before_destinations_are_allocated(monkeypatch):
    original, destination = state(), state(position=0, generation=8)
    inventory = method_inventory(original)
    specs = part_specs(inventory.parts)
    monkeypatch.setattr(torch, "empty", lambda *_a, **_kw: pytest.fail("bad descriptors allocated tensors"))
    for broken in (specs[:-1], (replace(specs[0], dtype="float32"), *specs[1:])):
        with pytest.raises(ValueError, match="tensor descriptors"):
            validate_method_restore(destination, inventory.metadata, broken)


def test_actual_state_mtp_handoff_leaves_core_scalar_and_device_frontiers_unpublished(request):
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import POINTS
    from tensorfold.families.qwen4_exp.cuda.state import State

    weights = request.getfixturevalue("_registered_mtp_family")
    original, destination = [State(weights, 32, 4, "fp16") for _ in range(2)]
    for item in (original, destination):
        item.retention_method_recovery_versions = POINTS
    original.set_pos(3)
    original.set_mtp_len(2)
    hidden = torch.arange(original.mtp_hidden_width).reshape(1, -1).half()
    _inventory, _targets, handoff = transfer(original, destination, pending_hc=hidden)
    assert torch.equal(handoff["pending_hc"], hidden)
    assert handoff["mtp_len"] == 2 and handoff["mtp_drafted"] == 0
    assert destination.pos == destination.mtp_len == int(destination.mtp_pos[0]) == 0


def test_new_dflash_turn_honors_fresh_method_choice_and_refuses_missing_conditioning():
    original = state(position=6)
    old = stream(original)
    fresh = stream(original)
    fresh.draft = False
    destination = state(position=0, generation=8)
    transfer(original, destination, source_stream=old, destination_stream=fresh, continuation="new-turn")
    assert destination.dflash2.enabled is False and fresh.draft is False
    # An earlier serial request may leave an explicitly disabled, cold method.
    cold = state(position=0)
    cold.pos = 6
    cold.dflash2.enabled = False
    old = stream(cold)
    old.draft = False
    inventory = method_inventory(cold, stream=old)
    fresh.draft = True
    with pytest.raises(ValueError, match="lacks valid conditioning"):
        validate_method_restore(state(position=0, generation=9), inventory.metadata, part_specs(inventory.parts),
                                stream=fresh, continuation="new-turn")


def test_restored_cpu_coordinate_values_are_validated_before_method_publication():
    original = state(position=6, mtp=True)
    original.mtp_len = 5
    issued = issue_coordinates(torch.tensor([[5, 3, 4]], dtype=torch.int64),
        position=original.media_mtp_position, profile_identity=PROFILE, generation=3, media_identity=MEDIA,
        row_start=5, max_rows=1, device="cpu")
    issued = copy_coordinates(issued, source_position=original.media_mtp_position, position=original.media_mtp_position,
                              row_start=5, row_count=1, dtype=torch.int64)
    original.mtp_boundary = MTPBoundary(5, torch.zeros((1, 16)).half(), issued, original.media_mtp_position, 3)
    inventory = method_inventory(original)
    destination = state(position=0, generation=8, mtp=True)
    targets = method_restore_targets(destination, inventory.metadata, part_specs(inventory.parts))
    for source, target in zip(inventory.parts, targets, strict=True):
        target.tensor.copy_(source.tensor)
    coordinates = next(part.tensor for part in targets if part.name == "method.mtp.coordinates")
    coordinates[0, 0] = 1 << 32
    with pytest.raises(ValueError, match="coordinate|representation"):
        commit_method_restore(destination, inventory.metadata, targets)
    assert destination.mtp_boundary is None and destination.dflash2.end == destination.pos == 0


def test_actual_protected_dflash_window_remains_distinct_from_overwritten_live_ring():
    original = state(position=11)
    protected = {"pos": 11, "mtp_len": 0, "dflash2": original.dflash2.snapshot()}
    saved = protected["dflash2"]["kv"].clone()
    fresh = torch.full((2, 2, 8, 8, 4), -123., dtype=torch.float16)
    original.dflash2.append(fresh, 11)
    original.pos = 19
    destination = state(position=0, generation=8)
    inventory, _targets, handoff = transfer(original, destination, recoveries=(protected,))
    part = next(part for part in inventory.parts if part.name == "method.recovery.0.dflash.kv")
    assert part.tensor is protected["dflash2"]["kv"]
    restored = handoff["recoveries"][0]
    assert restored["pos"] == 11 and restored["mtp_len"] == 0
    assert torch.equal(restored["dflash2"]["kv"], saved)
    assert restored["dflash2"]["generation"] == 8
    assert torch.equal(destination.dflash2.snapshot()["kv"], fresh)
    assert not torch.equal(restored["dflash2"]["kv"], destination.dflash2.snapshot()["kv"])
    assert sum(part.spec().nbytes for part in inventory.parts) <= inventory.future_bytes


def test_protected_media_hc_coordinates_are_exported_and_reissued_without_old_proof_retention():
    from tensorfold.families.qwen4_exp.cuda.media_mtp import snapshot_state, validate_snapshot

    original = state(position=6, df=False, mtp=True)
    original.mtp_len = 5
    issued = issue_coordinates(torch.tensor([[5, 3, 4]], dtype=torch.int64),
        position=original.media_mtp_position, profile_identity=PROFILE, generation=3, media_identity=MEDIA,
        row_start=5, max_rows=1, device="cpu")
    issued = copy_coordinates(issued, source_position=original.media_mtp_position, position=original.media_mtp_position,
                              row_start=5, row_count=1, dtype=torch.int64)
    original.mtp_boundary = MTPBoundary(5, torch.arange(16).reshape(1, 16).half(), issued, original.media_mtp_position, 3)
    protected = {"pos": 6, "mtp_len": 5, **snapshot_state(original)}
    old_proof = protected["media_state"]["mtp_boundary"]["coordinates"].proof
    original.pos = original.mtp_len = 7
    original.mtp_boundary = None
    destination = state(position=0, generation=8, df=False, mtp=True)
    inventory, _targets, handoff = transfer(original, destination, recoveries=(protected,))
    restored = handoff["recoveries"][0]
    validate_snapshot(destination, restored["media_state"], 6, 5)
    boundary = restored["media_state"]["mtp_boundary"]
    assert boundary["coordinates"].proof is not old_proof
    assert boundary["coordinates"].proof.generation == restored["media_state"]["generation"] == 8
    assert torch.equal(boundary["coordinates"].tensor, torch.tensor([[5, 3, 4]]))
    assert torch.equal(boundary["hidden"], protected["media_state"]["mtp_boundary"]["hidden"])
    assert method_restore_resources(destination, inventory.metadata) == {"host_bytes": 24, "device_bytes": 12}


def test_missing_protected_method_payload_or_excess_versions_cannot_be_ignored():
    original = state(position=6)
    with pytest.raises(ValueError, match="omits or changes"):
        method_inventory(original, recoveries=({"pos": 6, "mtp_len": 0},))
    protected = {"pos": 6, "mtp_len": 0, "dflash2": original.dflash2.snapshot()}
    with pytest.raises(ValueError, match="recovery commitment"):
        method_inventory(original, recoveries=(protected,) * 3)
