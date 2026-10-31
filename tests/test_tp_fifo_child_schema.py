"""FIFO does not retain the legacy continuation's opaque statistics tree."""

from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder


def peer(policy):
    value = object.__new__(TensorParallelDecoder)
    value.admission_policy = policy
    value.decoder = NS(vision=None, logical_context=128, w=NS(cfg=NS(vocab=1024)))
    value.engine = NS()
    value.requests, value.wires = {}, {}
    return value


def fifo_stream():
    value = Stream([1, 2], 3, owed=[4, 5])
    value.admission_ticket = 7
    value.principal = "alice"
    value.admission_status = NS(ticket=7, principal="alice")
    value.retention_intent = None
    value.output_capacity = 3
    return value


def test_fifo_roundtrip_keeps_none_carry_and_existing_context_replay_roles():
    sender, receiver = peer("fifo"), peer("fifo")
    original = fifo_stream()
    body = sender._pack(original)
    restored = receiver._unpack(body)
    assert body["carry"] is None and restored.carry is None
    assert restored.prompt == [1, 2] and restored.owed == [4, 5]
    assert restored.prompt is not body["prompt"] and restored.owed is not body["owed"]
    assert restored.admission_ticket == 7 and restored.principal == "alice"
    assert restored.output_capacity == 3


@pytest.mark.parametrize("carry", [{}, {"prefill_s": 1.0}, [], 0, False, ""])
def test_fifo_sender_rejects_every_non_none_carry_before_grammar_preparation(monkeypatch, carry):
    from tensorfold.engine import grammar
    sender, stream = peer("fifo"), fifo_stream()
    stream.carry = carry
    monkeypatch.setattr(grammar, "pack", lambda _constraint: pytest.fail("prepared invalid FIFO child"))
    with pytest.raises(ValueError, match="cannot carry legacy continuation statistics"):
        sender._pack(stream)
    assert not sender.requests and not sender.wires


@pytest.mark.parametrize("carry", [{}, {"prefill_s": 1.0}, [], 0, False, ""])
def test_fifo_receiver_rejects_non_none_carry_before_grammar_or_stream_publication(monkeypatch, carry):
    from tensorfold.engine import grammar
    receiver = peer("fifo")
    body = peer("fifo")._pack(fifo_stream())
    body.update(carry=carry, grammar={"untrusted": "not compiled"})
    monkeypatch.setattr(grammar, "compiler", lambda *_args: pytest.fail("compiled invalid FIFO child"))
    with pytest.raises(ValueError, match="received FIFO request cannot carry"):
        receiver._unpack(body)
    assert not receiver.requests and not receiver.wires


@pytest.mark.parametrize("carry", [None, {}, {"prefill_s": 1.0, "cached": 2, "stages": [1, 2]}])
def test_legacy_continuation_statistics_remain_supported(carry):
    sender, receiver = peer("legacy"), peer("legacy")
    original = Stream([1], 3, carry=carry, owed=[2])
    body = sender._pack(original)
    restored = receiver._unpack(body)
    assert restored.carry is carry and restored.owed == [2]


def test_fifo_schema_capability_requires_actual_geometry_and_preserves_legacy():
    value = peer("fifo")
    assert value.fifo_retained_child_schema() == {
        "schema": "tensorfold.tp-fifo-retained-children", "version": 1, "context": 128, "vocab": 1024}
    value.decoder.logical_context = None
    with pytest.raises(ValueError, match="actual logical context"):
        value.fifo_retained_child_schema()
    assert peer("legacy").fifo_retained_child_schema() is None


@pytest.mark.parametrize("role", ["prompt", "owed"])
@pytest.mark.parametrize("invalid", [{"opaque": []}, [[1]], [True], [-1], [1024], [0] * 129, (1, 2)])
@pytest.mark.parametrize("receiving", [False, True], ids=["source", "received"])
def test_fifo_token_children_reject_opaque_invalid_or_overwindow_values_before_grammar(
        monkeypatch, role, invalid, receiving):
    from tensorfold.engine import grammar
    owner, stream = peer("fifo"), fifo_stream()
    if receiving:
        body = owner._pack(stream)
        body[role] = invalid
        body["grammar"] = {"untrusted": "not compiled"}
        monkeypatch.setattr(grammar, "compiler", lambda *_a: pytest.fail("compiled invalid token children"))
        call, argument = owner._unpack, body
    else:
        setattr(stream, role, invalid)
        monkeypatch.setattr(grammar, "pack", lambda *_a: pytest.fail("packed invalid token children"))
        call, argument = owner._pack, stream
    with pytest.raises(ValueError, match=f"FIFO {role}"):
        call(argument)
    assert not owner.requests and not owner.wires


@pytest.mark.parametrize("field,value", [
    ("profile_epoch", {"opaque": []}), ("profile_epoch", True), ("profile_epoch", -1),
    ("committed_output_tokens", []), ("committed_output_tokens", False), ("committed_output_tokens", -1),
    ("sampling", [[1], 1.0, 0, 1.0, 0.0]), ("sampling", [1, {}, 0, 1.0, 0.0]),
    ("sampling", [1, 1.0, 0, [], 0.0]), ("sampling", [1, 1.0, 0, 1.0, {}]),
    ("sampling", [1, 1.0, 0, 1.0]),
])
def test_fifo_received_scalar_roles_cannot_retain_opaque_json_trees(monkeypatch, field, value):
    from tensorfold.engine import grammar
    owner = peer("fifo")
    body = owner._pack(fifo_stream())
    body.update({field: value, "grammar": {"untrusted": "not compiled"}})
    monkeypatch.setattr(grammar, "compiler", lambda *_a: pytest.fail("compiled opaque scalar children"))
    with pytest.raises(ValueError):
        owner._unpack(body)


def test_fifo_children_keep_exact_window_and_unrestricted_supported_scalar_widths():
    owner = peer("fifo")
    stream = fifo_stream()
    stream.prompt, stream.owed = [0] * 128, [1023] * 128
    body = owner._pack(stream)
    big = 10**4000
    body.update(profile_epoch=big, committed_output_tokens=big,
                sampling=[-big, 0.5, big, 1.0, 0.0])
    restored = owner._unpack(body)
    assert len(restored.prompt) == len(restored.owed) == 128
    assert restored.profile_epoch == restored.committed_output_tokens == big
    assert restored.sampling.seed == -big and restored.sampling.top_k == big


@pytest.mark.parametrize("policy", ["legacy", "fifo"])
@pytest.mark.parametrize("feature", ["vision", "probabilities", "penalties"])
def test_existing_unsupported_feature_refusal_precedes_sampling_field_reads(policy, feature):
    owner, stream = peer(policy), fifo_stream()
    stream.sampling = object()  # Accessing its missing sampling fields would fail.
    setattr(stream, feature, object())
    with pytest.raises(ValueError, match="do not support vision, logprobs or penalties"):
        owner._pack(stream)
