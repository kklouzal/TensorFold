"""Explicit synthetic CPU fixtures only; NOT real model/provider calibration."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorfold.offline_quantization.assistant_frames import (
    AssistantCalibrationActor,
    CalibrationSource,
    CORE22,
    FrameSample,
    INITIAL20,
    PhaseQuota,
    config_digest,
    modelopt_forward_step,
)


class ScaledFixtureEmbedding(nn.Embedding):
    def forward(self, ids):
        return super().forward(ids) * 2


class SyntheticTarget:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(vocab=16, hidden=4, kv_heads=1, head_dim=2, global_head_dim=4, window=2)
        self.raw_config = {"model_type": "explicit_synthetic_fixture"}
        self.calibration_source = CalibrationSource(
            "synthetic_fixture", "1" * 40, config_digest(self.raw_config), "2" * 64, "3" * 64, "4" * 64, "5" * 64
        )
        self.embedding = ScaledFixtureEmbedding(16, 4, 0).to(dtype=torch.bfloat16)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(64).reshape(16, 4) / 100)
        self.model = SimpleNamespace(get_input_embeddings=lambda: self.embedding)
        self.hidden = (torch.arange(4, dtype=torch.bfloat16).reshape(1, 1, 4) + 1) / 10
        self.local = tuple(torch.arange(4, dtype=torch.bfloat16).reshape(1, 1, 2, 2) + i for i in (1, 2))
        self.full = tuple(torch.arange(12, dtype=torch.bfloat16).reshape(1, 1, 3, 4) + i for i in (1, 2))
        self.mask = torch.ones(1, 3, dtype=torch.long)

    def assistant_state(self, state):
        if state.owner is not self or state.poisoned:
            raise RuntimeError("unhealthy fixture state")
        return {
            "position": state.position,
            "last_hidden_state": self.hidden,
            "attention_mask": self.mask,
            "shared_kv_states": {"sliding_attention": self.local, "full_attention": self.full},
        }


class SyntheticAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_projection = nn.Linear(8, 4, bias=False)
        self.post_projection = nn.Linear(4, 4, bias=False)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(4):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.mlp = nn.Module()
            for name in ("q_proj", "o_proj"):
                setattr(layer.self_attn, name, nn.Linear(4, 4, bias=False))
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(layer.mlp, name, nn.Linear(4, 4, bias=False))
            self.model.layers.append(layer)
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.to(dtype=torch.bfloat16)
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(0.01)
        self.seen = []
        self.skip = False

    def forward(self, inputs_embeds, position_ids, shared_kv_states, attention_mask, use_cache, return_dict):
        assert use_cache is False and return_dict is True
        self.seen.append(
            {
                "inputs": inputs_embeds.detach().clone(),
                "position": position_ids.item(),
                "kv_ptr": shared_kv_states["full_attention"][0].data_ptr(),
                "mask_ptr": attention_mask.data_ptr(),
            }
        )
        value = self.pre_projection(inputs_embeds)
        for index, layer in enumerate(self.model.layers):
            value = value + layer.self_attn.o_proj(layer.self_attn.q_proj(value))
            gate = layer.mlp.gate_proj(value)
            up = layer.mlp.up_proj(value)
            if not (self.skip and index == 3):
                value = value + layer.mlp.down_proj(gate * up)
        projected = self.post_projection(value)
        return SimpleNamespace(last_hidden_state=projected, logits=self.lm_head(value))


@pytest.fixture
def setup():
    target = SyntheticTarget()
    state = SimpleNamespace(owner=target, position=3, limit=64, poisoned=False)
    calls = []

    @contextmanager
    def exclusive():
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    def create(**overrides):
        options = dict(
            exclusive=exclusive,
            retire=lambda: calls.append("retire"),
            accepted_tokens=lambda: [2, 3, 4],
            quotas={"calibration": PhaseQuota(2, 3), "holdout": PhaseQuota(1, 1)},
            assistant_revision="6" * 40,
            assistant_config_sha256="7" * 64,
            recipe_sha256="8" * 64,
            calibration_data_sha256="9" * 64,
            heldout_data_sha256="a" * 64,
            max_borrowed_bytes=4096,
        )
        options.update(overrides)
        return AssistantCalibrationActor(target, state, **options)

    return target, state, calls, create


def sample(phase="calibration", value="b", trajectory="assistant_rollout"):
    return FrameSample(phase, value * 64, 29, trajectory)


def test_public_scaled_inputs_real_borrows_frozen_recurrence_coverage_and_expiry(setup):
    target, state, calls, create = setup
    actor = create()
    model = SyntheticAssistant()
    with actor.session():
        with actor.frame(5, sample()) as frame:
            first = modelopt_forward_step(model, frame)
            frame.advance_recurrent_token(7, origin="assistant_draw")
            modelopt_forward_step(model, frame)
            assert frame.position == state.position == 3 and frame.step == 1
            assert frame._hidden is first.last_hidden_state
            assert all(x["position"] == 3 and x["kv_ptr"] == target.full[0].data_ptr() for x in model.seen)
            assert torch.equal(model.seen[0]["inputs"][..., :4], target.embedding(torch.tensor([[5]])))
            assert torch.equal(model.seen[0]["inputs"][..., 4:], target.hidden)
            assert not torch.equal(model.seen[0]["inputs"][..., :4], target.embedding.weight[5].reshape(1, 1, 4))
        assert frame._held is None and frame._hidden is None
        with pytest.raises(RuntimeError):
            modelopt_forward_step(model, frame)
    receipt = actor.receipt()
    assert receipt["evidence"] == "synthetic-fixture-only"
    assert receipt["provider_calibration_status"].startswith("not-invoked")
    assert not receipt["quantized_model_quality_or_capacity_verified"]
    assert len(CORE22) == 22 and len(INITIAL20) == 20
    assert not receipt["pre_post_selected"]
    assert all(n == 2 for n in receipt["counts"]["calibration"]["observations"].values())
    assert [r["recurrent_step"] for r in receipt["records"]] == [0, 1]
    receipt["counts"]["calibration"]["frames"] = 999
    assert actor.receipt()["counts"]["calibration"]["frames"] == 1
    assert calls == ["enter", "retire", "exit"]


def test_quota_and_train_holdout_identity_guards(setup):
    _, _, _, create = setup
    actor = create()
    model = SyntheticAssistant()
    with actor.session():
        with actor.frame(5, sample()) as frame:
            modelopt_forward_step(model, frame)
        with pytest.raises(ValueError, match="cross"):
            with actor.frame(5, sample("holdout")):
                pass
        with actor.frame(5, sample(value="c")):
            pass
        with pytest.raises(RuntimeError, match="frame quota"):
            with actor.frame(5, sample(value="d")):
                pass


def test_source_restore_or_pending_mutation_invalidates_frame_before_next_forward(setup):
    target, _, _, create = setup
    actor = create()
    model = SyntheticAssistant()
    with pytest.raises(RuntimeError, match="mutated or restored"):
        with actor.session():
            with actor.frame(5, sample()) as frame:
                target.hidden = target.hidden.clone()
                modelopt_forward_step(model, frame)
    assert not model.seen and actor.receipt()["failed"]


def test_model_coverage_gap_removes_hooks_and_fails_receipt(setup):
    _, _, _, create = setup
    actor = create()
    model = SyntheticAssistant()
    model.skip = True
    with pytest.raises(RuntimeError, match="observed exactly once"):
        with actor.session():
            with actor.frame(5, sample()) as frame:
                modelopt_forward_step(model, frame)
    assert all(not m._forward_pre_hooks for m in model.modules())
    assert actor.receipt()["failed"] and not actor.receipt()["records"]


def test_retirement_failure_preserves_primary_and_retains_borrowed_owners(setup):
    _, state, calls, create = setup

    def fail():
        calls.append("retire-failed")
        raise OSError("retirement failed")

    actor = create(retire=fail)
    primary = ValueError("primary consumer failure")
    with pytest.raises(ValueError) as caught:
        with actor.session():
            with actor.frame(5, sample()):
                raise primary
    assert caught.value is primary and isinstance(primary.__cause__, OSError)
    assert calls.count("retire-failed") == 1 and state.poisoned
    assert actor._active.retirement_failed and actor._active._held is not None
    with pytest.raises(RuntimeError):
        with actor.session():
            pass


def test_gradient_fixture_does_not_train_target_and_requires_noninference_source(setup):
    target, _, _, create = setup
    actor = create(quotas={"qad": PhaseQuota(1, 1, gradients=True)})
    model = SyntheticAssistant()
    with actor.session():
        with actor.frame(5, sample("qad")) as frame:
            out = modelopt_forward_step(model, frame)
            out.logits.float().sum().backward()  # Complete while borrowed KV lease is active.
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert target.embedding.weight.grad is None
    with torch.inference_mode():
        target.hidden = target.hidden.clone()
    actor = create(quotas={"sensitivity": PhaseQuota(1, 1, gradients=True)})
    with pytest.raises(ValueError, match="no_grad"):
        with actor.session():
            with actor.frame(5, sample("sensitivity")):
                pass


def test_false_quantized_source_cannot_label_bf16_fixture_calibration(setup):
    target, _, _, create = setup
    target.calibration_source = replace(
        target.calibration_source, kind="quantized_verifier", quant_manifest_sha256="c" * 64, payloads_sha256="d" * 64
    )
    target.calibration_payloads_sha256 = "d" * 64
    target.quantization_manifest_sha256 = "c" * 64
    with pytest.raises(ValueError, match="quantized runtime metadata"):
        create()


def test_budget_transcript_selection_and_phase_metadata_fail_early(setup):
    _, _, _, create = setup
    with pytest.raises(ValueError):
        create(selected_modules=["masked_embedding.centroids"])
    with pytest.raises(ValueError):
        create(quotas={"holdout": PhaseQuota(1, 1, True)})
    actor = create(max_borrowed_bytes=1)
    with pytest.raises(ValueError, match="byte budget"):
        with actor.session():
            with actor.frame(5, sample()):
                pass
    actor = create(accepted_tokens=lambda: [2, 3])
    with pytest.raises(ValueError, match="transcript"):
        with actor.session():
            with actor.frame(5, sample()):
                pass


def test_forward_quota_and_recurrence_origin_are_independent_budgets(setup):
    _, _, _, create = setup
    actor = create(quotas={"calibration": PhaseQuota(1, 1)}, max_recurrent_steps=1)
    model = SyntheticAssistant()
    with actor.session():
        with actor.frame(5, sample()) as frame:
            with pytest.raises(ValueError, match="prior assistant output"):
                frame.advance_recurrent_token(6, origin="assistant_draw")
            modelopt_forward_step(model, frame)
            with pytest.raises(RuntimeError, match="forward-step quota"):
                modelopt_forward_step(model, frame)
            with pytest.raises(RuntimeError, match="recurrent-step quota"):
                frame.advance_recurrent_token(6, origin="assistant_draw")
    assert len(model.seen) == 1
    assert actor.receipt()["counts"]["calibration"]["forward_steps"] == 1


def test_partial_hook_registration_failure_removes_prior_hooks(setup, monkeypatch):
    _, _, _, create = setup
    actor = create()
    model = SyntheticAssistant()

    def fail(*args, **kwargs):
        raise RuntimeError("hook registration failed")

    last = dict(model.named_modules())[INITIAL20[-1]]
    monkeypatch.setattr(last, "register_forward_pre_hook", fail)
    with pytest.raises(RuntimeError, match="hook registration failed"):
        with actor.session():
            with actor.frame(5, sample()) as frame:
                modelopt_forward_step(model, frame)
    assert not model.seen
    assert all(not module._forward_pre_hooks for module in model.modules())
    assert actor.receipt()["failed"]


def test_foreign_thread_cannot_use_active_frame(setup):
    import threading

    _, _, _, create = setup
    actor = create()
    model = SyntheticAssistant()
    errors = []
    with actor.session():
        with actor.frame(5, sample()) as frame:

            def foreign():
                try:
                    modelopt_forward_step(model, frame)
                except RuntimeError as error:
                    errors.append(error)

            worker = threading.Thread(target=foreign)
            worker.start()
            worker.join()
            assert len(errors) == 1
            assert not model.seen
            modelopt_forward_step(model, frame)
    assert not actor.receipt()["failed"]


def test_stream_break_retires_lease_without_prefetch_or_large_frame_archive(setup):
    _, _, calls, create = setup
    actor = create()
    model = SyntheticAssistant()
    prepared = []

    def descriptors():
        for token in (5, 6):
            prepared.append(token)
            yield token, sample(value="b" if token == 5 else "c")

    with actor.session():
        with actor.stream(descriptors()) as frames:
            for frame in frames:
                modelopt_forward_step(model, frame)
                break
        assert prepared == [5]
        assert frame._held is None and actor._active is None
        with actor.frame(6, sample(value="c")) as next_frame:
            modelopt_forward_step(model, next_frame)
    assert not actor.receipt()["failed"]
    assert calls.count("retire") == 2


def test_stream_close_cannot_hide_retirement_error_as_generator_exit(setup):
    _, state, _, create = setup

    def fail():
        raise OSError("stream retirement failed")

    actor = create(retire=fail)
    with pytest.raises(OSError, match="stream retirement failed"):
        with actor.session():
            with actor.stream([(5, sample())]) as frames:
                for _frame in frames:
                    break
    assert state.poisoned and actor._active.retirement_failed


def test_exclusive_context_receives_actual_primary_exception_triple(setup):
    _, _, _, create = setup
    primary = ValueError("consumer transaction failed")
    observed = []

    class Exclusive:
        def __enter__(self):
            pass

        def __exit__(self, kind, error, trace):
            observed.append((kind, error, trace))
            return True  # Cannot suppress the required calibration failure.

    actor = create(exclusive=Exclusive)
    with pytest.raises(ValueError) as caught:
        with actor.session():
            raise primary
    assert caught.value is primary
    assert observed[0][0] is ValueError and observed[0][1] is primary
    assert observed[0][2] is not None


def test_expired_frame_does_not_retain_actor_target_or_large_state():
    import gc
    import weakref

    target = SyntheticTarget()
    state = SimpleNamespace(owner=target, position=3, limit=64, poisoned=False)

    @contextmanager
    def exclusive():
        yield

    actor = AssistantCalibrationActor(
        target,
        state,
        exclusive=exclusive,
        retire=lambda: None,
        accepted_tokens=lambda: [2, 3, 4],
        quotas={"calibration": PhaseQuota(1, 1)},
        assistant_revision="6" * 40,
        assistant_config_sha256="7" * 64,
        recipe_sha256="8" * 64,
        calibration_data_sha256="9" * 64,
        heldout_data_sha256="a" * 64,
        max_borrowed_bytes=4096,
    )
    target_ref, actor_ref = weakref.ref(target), weakref.ref(actor)
    with actor.session():
        with actor.frame(5, sample()) as expired:
            pass
    assert expired.actor is None
    del actor, state, target
    gc.collect()
    assert actor_ref() is None and target_ref() is None
    with pytest.raises(RuntimeError, match="expired"):
        modelopt_forward_step(SyntheticAssistant(), expired)


def test_failed_exclusive_acquisition_is_terminal_without_invented_exit(setup):
    _, _, _, create = setup
    primary = RuntimeError("partial actor acquisition")
    exits = []

    class Exclusive:
        def __enter__(self):
            raise primary

        def __exit__(self, *args):
            exits.append(args)

    actor = create(exclusive=Exclusive)
    with pytest.raises(RuntimeError) as caught:
        with actor.session():
            pass
    assert caught.value is primary and not exits
    assert actor.receipt()["failed"]
    with pytest.raises(RuntimeError, match="failed"):
        with actor.session():
            pass
