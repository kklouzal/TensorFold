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
