"""The cached FAMILY candidate borrows only its own validated INT8 greedy text state."""

import hashlib
import json
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from . import test_sm70_family_trial_policy as family  # noqa: E402
from . import test_sm70_rotorquant_state as state_controls  # noqa: E402
from . import test_cuda_geometry as geometry_controls  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import State  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import get_pair  # noqa: E402

policy = family.policy
source = family.source
allocations = geometry_controls.allocations
PROFILE = prefix_policy.FAMILY_PREFIX_PROFILE


def weights():
    return NS(meta={"execution_provider": "sm70", "sm70_profile": PROFILE})


def test_cached_trial_reuses_all_source_guards_and_only_family_flags(source, monkeypatch):
    monkeypatch.setenv(source.module.PROFILE_ENV, PROFILE)
    assert source.module.configure(source.text, tp=8, torch=torch) == PROFILE
    assert family.selectors(source.policy) == (True, False, False, False, False, False, "reference", "reference", None)
    loaded = family.weights(source)
    source.module.prepare_weights(loaded)
    assert loaded.meta["sm70_profile"] == PROFILE
    assert loaded.meta["sm70_family_trial"]["prefix_reuse"] == "int8-greedy-text-trial"
    assert loaded.meta["sm70_family_trial"]["qualification"] == "unqualified-candidate"
    assert not {"attention_qualification", "capture_ready", "device_greedy_mtp_ready",
                "device_greedy_native_ready"} & loaded.meta.keys()


@pytest.mark.parametrize("change", ["geometry", "runtime", "software"])
def test_cached_trial_refuses_unverified_foundation_before_selectors(source, monkeypatch, change):
    monkeypatch.setenv(source.module.PROFILE_ENV, PROFILE)
    if change == "geometry":
        source.text["head_dim"] = 128
    elif change == "runtime":
        monkeypatch.setattr(torch.version, "git_version", "wrong")
    else:
        source.record["recorded_components"]["cuda"] = "12.6"
    before = family.selectors(source.policy)
    with pytest.raises(ValueError):
        source.module.configure(source.text, tp=8, torch=torch)
    assert family.selectors(source.policy) == before


@pytest.mark.parametrize("pair,allowed", [
    (None, False), (get_pair("int8"), True), (get_pair("fp16"), False),
    (get_pair("int8", "fp16", "int8"), False), (get_pair("int8", "int8", "fp16"), False),
    (get_pair("rotorquant-signed128-4"), False), (get_pair("rotorquant-signed64-norm4"), False),
    (get_pair("rotorquant-iso4-sm70-v1"), False),
])
@pytest.mark.parametrize("sampling,greedy", [(None, True), (Sampling(17, 0), True),
                                           (Sampling(17, -1), True), (Sampling(17, .7), False)])
def test_only_symmetric_int8_and_valid_greedy_requests_borrow(pair, allowed, sampling, greedy):
    expected = allowed and greedy
    assert prefix_policy.reuse_allowed(weights(), sampling, draft=True, vision=None,
                                        position=24, kv_pair=pair) is expected
    assert prefix_policy.serial_reuse_allowed(weights(), sampling, draft=True,
                                               position=24, kv_pair=pair) is expected


@pytest.mark.parametrize("sampling", [Sampling("bad", 0), Sampling(True, 0), Sampling(17, float("nan")),
    Sampling(17, float("inf")), Sampling(17, 0, 0, 1, -.1), Sampling(17, 0, 0, 1.1)])
def test_invalid_sampling_stays_cold_without_replacing_producing_validation(sampling):
    assert not prefix_policy.reuse_allowed(weights(), sampling, draft=True, vision=None,
                                           position=24, kv_pair=get_pair("int8"))
    from tensorfold.cuda.sampling import validate_policy
    with pytest.raises((ValueError, TypeError, OverflowError)):
        validate_policy([24], sampling)


@pytest.mark.parametrize("position", [-1, True, 2**64])
def test_invalid_position_cannot_gain_a_prefix_lease(position):
    assert not prefix_policy.reuse_allowed(weights(), None, draft=True, vision=None,
                                           position=position, kv_pair=get_pair("int8"))


def test_media_and_nondrafted_requests_stay_cold_and_other_providers_unchanged():
    assert not prefix_policy.reuse_allowed(weights(), None, draft=True, vision=object(),
                                           position=24, kv_pair=get_pair("int8"))
    assert not prefix_policy.serial_reuse_allowed(weights(), None, draft=True, vision=object(),
                                                  position=24, kv_pair=get_pair("int8"))
    assert not prefix_policy.serial_reuse_allowed(weights(), None, draft=False,
                                                  position=24, kv_pair=get_pair("int8"))
    other = NS(meta={"execution_provider": "cuda", "sm70_profile": PROFILE})
    assert prefix_policy.reuse_allowed(other, None, draft=True, vision=None, position=24)


def test_cache_epoch_is_exact_distinct_and_only_added_to_cached_trial():
    epoch = prefix_policy.state_epoch(PROFILE)
    assert epoch == ("sm70-prefix-epoch-v1", hashlib.sha256(prefix_policy.identity(PROFILE).encode()).hexdigest())
    assert len(epoch) == 2 and len(epoch[1]) == 64 and set(epoch[1]) <= set("0123456789abcdef")
    for old in (None, "reference", "optimized", "source-family-trial"):
        assert prefix_policy.state_epoch(old) is None
        assert prefix_policy.identity(old) != prefix_policy.identity(PROFILE)


@pytest.mark.usefixtures("allocations")
def test_cached_state_restore_copy_and_reference_rejection_precede_mutation(monkeypatch):
    w = state_controls.model(8)
    w.meta["sm70_profile"] = PROFILE
    cached = State(w, 37, 4, kv_pair=get_pair("int8"), limit=128)
    destination = State(w, 37, 4, kv_pair=get_pair("int8"), limit=128)
    w.meta["sm70_profile"] = "reference"
    reference = State(w, 37, 4, kv_pair=get_pair("int8"), limit=128)
    assert prefix_policy.FAMILY_PREFIX_STATE in cached.kv_identity
    assert reference.kv_identity == (get_pair("int8").identity, state_controls.check_pair(get_pair("int8")))
    assert cached.cache_bytes() == reference.cache_bytes()
    for tensor in state_controls.tensors(cached):
        tensor.fill_(17)
    cached.set_pos(9)
    cached.set_mtp_len(8)
    snapshot = cached.snapshot()
    destination.copy_prefix(cached, 9, 8)
    destination.restore(snapshot)
    assert destination.pos == 9 and destination.mtp_len == 8
    for left, right in zip(cached.kc, destination.kc):
        assert torch.equal(left.k[:9], right.k[:9]) and left.k.data_ptr() != right.k.data_ptr()
    old = reference.rec.clone()
    with pytest.raises(ValueError, match="matching ordered"):
        reference.restore(snapshot)
    assert torch.equal(reference.rec, old)
    monkeypatch.setattr(reference, "kv_check", lambda: pytest.fail("cross-profile source was borrowed"))
    with pytest.raises(ValueError, match="source/profile"):
        destination.copy_prefix(reference, 0, 0)


@pytest.mark.usefixtures("allocations")
def test_added_epoch_uses_existing_bounded_metadata_without_tensor_or_payload_growth():
    from tensorfold.engine.model_identity import canonical_runtime_json
    from tensorfold.families.qwen4_exp.cuda.retained_parts import manifest_bound
    from tensorfold.families.qwen4_exp.cuda.state_retention import state_inventory

    w = state_controls.model(8, mtp=False)
    plain = State(w, 37, 4, kv_pair=get_pair("int8"), limit=128)
    w.meta["sm70_profile"] = PROFILE
    cached = State(w, 37, 4, kv_pair=get_pair("int8"), limit=128)
    before = state_inventory(plain, recovery_versions=4)
    after = state_inventory(cached, recovery_versions=4)
    assert len(before.parts) == len(after.parts) and before.future_bytes == after.future_bytes
    old_metadata = json.dumps(before.metadata, separators=(",", ":"), sort_keys=True)
    new_metadata = json.dumps(after.metadata, separators=(",", ":"), sort_keys=True)
    identity_delta = len(json.dumps(cached.kv_identity, separators=(",", ":"))) - len(
        json.dumps(plain.kv_identity, separators=(",", ":")))
    assert len(new_metadata) - len(old_metadata) == identity_delta
    canonical_runtime_json({"document": new_metadata}, max_bytes=16 << 20)
    assert manifest_bound(len(before.parts)) == manifest_bound(len(after.parts))


def test_worker_rejects_a_cold_family_peer_before_payload_loading(monkeypatch):
    from .test_flashnext_stochastic_prefix import method
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    precision = NS(fp8=lambda: False, same_on_ranks=lambda a, b: None)
    agree = method(FlashNextEngine, "_same_settings", {"json": json, "prompt_precision": precision})
    proxy = NS(int64=torch.int64, tensor=lambda data, **kw: torch.tensor(data, dtype=kw["dtype"]),
               empty=lambda shape, **kw: torch.empty(shape, dtype=kw["dtype"]), equal=torch.equal)
    packets, stale = [], []

    def gather(value, destination):
        packets.append(value.clone())
        destination.copy_(value.repeat(8))
        if stale:
            destination.view(8, -1)[1, 9] = stale[0]

    engine = NS(execution_provider="sm70", sm70_profile="source-family-trial", depth=2, confidence=.7,
        max_len=262144, streams=26, graphs_enabled=False, prefill_rows=256, tp=8,
        rope=NS(metadata=lambda: {"rope_type": "default"}), kv_pair=NS(handshake=0), comm=NS(all_gather=gather))
    agree(engine, proxy, None)
    previous = int(packets[-1][9])
    engine.sm70_profile = PROFILE
    agree(engine, proxy, None)
    assert int(packets[-1][9]) != previous
    stale.append(previous)
    with pytest.raises(RuntimeError, match="different settings"):
        agree(engine, proxy, None)


@pytest.mark.parametrize("media", [False, True])
def test_actual_serial_caller_forwards_media_before_borrowing(media):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    engine = object.__new__(FlashNextEngine)
    engine.w, engine.kv_pair, engine.tp, engine.served = weights(), get_pair("int8"), 8, 0
    engine.vision = object()
    engine._begin_request = lambda _: None
    engine._limit = lambda _, count: count
    retired_calls, borrowed, shared = [], [], []
    engine._retire_request = lambda request, primary, *, retired=False: retired_calls.append(primary)

    hit = ([11] * 23, object())

    def resume(prompt):
        borrowed.append(prompt)
        return hit

    def share(prompt, count, sampling, draft, cached, constraint, stop_eos, token_limit):
        shared.append(cached)
        return prompt, count, sampling, draft, cached, constraint, stop_eos, None, token_limit

    engine._resume, engine._share = resume, share
    engine._decode = lambda *args, **kwargs: {"cached": len(args[4][0]) if args[4] else 0}
    result = engine.generate([11] * 24, 32, Sampling(17, 0), lambda _: None, draft=True,
                             vision=object() if media else None)
    assert bool(borrowed) is not media
    assert shared == [0 if media else 23] and result == {"cached": 0 if media else 23}
    assert retired_calls == [None]
