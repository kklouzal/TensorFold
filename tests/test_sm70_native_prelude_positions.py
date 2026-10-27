"""Native prelude accepts real Engine position binding and refuses extended roles."""
from types import SimpleNamespace as NS
from dataclasses import replace

import pytest
torch = pytest.importorskip("torch")

from tests.test_qwen4_exp_position_profile import config, put  # noqa: E402
from tensorfold.families.qwen4_exp.position_profile import (  # noqa: E402
    resolve_position_profile, prepare_position_vectors, bind_position_profile, PositionOperands,
)
from tensorfold.families.qwen4_exp.cuda.history_owner import (  # noqa: E402
    _native_prelude_binding, _native_prelude_position,
)


def test_actual_resolve_prepare_bind_native_roles_are_admissible(tmp_path):
    path = put(tmp_path / "native", config())
    profile = resolve_position_profile(path.parent, requested_context=262144)
    prepared = prepare_position_vectors(profile, mtp_active=True)
    weights = NS(cfg=NS(rope=profile.target.rope), positions=None, inv_freq=prepared.target_cpu,
                 mtp=object(), device="cpu")
    binding = bind_position_profile(weights, profile, mtp_active=True, prepared=prepared)
    assert weights.positions is binding and _native_prelude_binding(binding)
    assert _native_prelude_position(binding.target) and _native_prelude_position(binding.mtp)
    assert binding.target.inv_freq is weights.inv_freq
    assert binding.target.domain.rope.native_context == 262144
    from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation

    weights.meta = {"coordinate_profile_identity": profile.identity}
    weights.comm = None
    context = install_input_preparation(weights)
    assert weights.meta["input_preparation_context"] is context
    assert context.prepare("fixture-native-roles", lambda: binding.target) is binding.target
    context.close()


@pytest.mark.parametrize("change", ["yarn", "extended", "foreign-sections", "amplitude", "unissued"])
def test_native_binding_refuses_non_native_or_inconsistent_role_without_mutation(tmp_path, change):
    path = put(tmp_path / "native", config())
    profile = resolve_position_profile(path.parent, requested_context=262144)
    prepared = prepare_position_vectors(profile, mtp_active=False)
    weights = NS(cfg=NS(rope=profile.target.rope), positions=None, inv_freq=prepared.target_cpu,
                 mtp=None, device="cpu")
    binding = bind_position_profile(weights, profile, mtp_active=False, prepared=prepared)
    role = binding.target
    if change == "yarn":
        role = replace(role, domain=replace(role.domain, rope=replace(role.domain.rope, rope_type="yarn")))
    elif change == "extended":
        role = replace(role, domain=replace(role.domain, max_rows=524288))
    elif change == "foreign-sections":
        role = replace(role, sections=(10, 11, 11))
    elif change == "amplitude":
        role = replace(role, amplitude=2.)
    else:
        role = PositionOperands(role.inv_freq, role.amplitude, role.sections, None, None, None)
    assert not _native_prelude_position(role)
    assert not _native_prelude_binding(replace(binding, target=role))
    assert weights.positions is binding

def test_only_actual_text_coordinate_producer_marks_scalar_recipe(monkeypatch):
    from tests.test_compact_position_frames import prepared
    from tensorfold.families.qwen4_exp.cuda import compact_indexer as indexer

    buffers, _position, states, windows, segments = prepared(context_identity="1" * 64)
    for state in states:
        state.media_input_span = None
        state.image_positions = None
        state.rope_delta = 0
        state.rope_delta_dev = torch.zeros(1, dtype=torch.int32)
    indexer.stage_target_indexer_frame(buffers, segments, tuple(windows))
    assert indexer._scalar_text_frame(buffers.target_indexer_frame) is False
    # A fresh actual text producer creates canonical rows under the same issued
    # role/context, and is the only path that grants the private scalar recipe.
    indexer.stage_text_indexer_frame(buffers, segments)
    assert indexer._scalar_text_frame(buffers.target_indexer_frame) is True
    buffers.input_preparation_context.close()

def test_text_producer_marker_reuses_exact_paid_frame_field_layout():
    from tensorfold.families.qwen4_exp.cuda.compact_indexer import IndexerFrame, _ScalarTextIndexerFrame
    from dataclasses import fields

    args = (1, 3, (), False, None, None, None)
    ordinary, text = IndexerFrame(*args), IndexerFrame(*args)
    object.__setattr__(text, "__class__", _ScalarTextIndexerFrame)
    assert tuple(x.name for x in fields(ordinary)) == tuple(x.name for x in fields(text))
    assert ordinary.__sizeof__() == text.__sizeof__()
    assert ordinary.__dict__.__sizeof__() == text.__dict__.__sizeof__()
    assert ordinary.__dict__.keys() == text.__dict__.keys()

def test_shared_parent_source_path_keeps_all_eight_context_channels_identical(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import hashlib

    path = put(tmp_path / "shared-native", config())
    def resolve(_rank):
        profile = resolve_position_profile(path.parent, requested_context=257)
        return profile.identity, hashlib.sha256((profile.identity + "/fixture-model/2").encode()).hexdigest()
    with ThreadPoolExecutor(max_workers=8) as workers:
        values = list(workers.map(resolve, range(8)))
    assert len(set(values)) == 1
    other = put(tmp_path / "different-native", config())
    assert resolve_position_profile(other.parent, requested_context=257).identity != values[0][0]
