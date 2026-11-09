"""Private full-model selector wiring; no CUDA arithmetic or model emulation."""
import json
import os
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli
from tools import launch_cuda_tp as launcher
from tools import v100_owned_qualification as owned
from tests import test_v100_owned_qualification as qualification

channel = qualification.channel


def selection(rows):
    from tensorfold.cuda.sm70.gguf_ops import _GroupedGateUp
    return _GroupedGateUp(rows)


@pytest.mark.parametrize("rows", [None, 2, 4])
def test_parent_places_selection_before_separator_on_every_rank(monkeypatch, tmp_path, rows):
    seen = []
    monkeypatch.setattr(launcher, "supervise", lambda commands, envs, **kw: seen.extend(commands) or 0)
    option = [] if rows is None else ["--grouped-gate-up-rows", str(rows)]
    assert owned.main([*option, "--gpus", "0,1,2,3,4,5,6,7", "--out", str(tmp_path / "fresh"),
                       "--", "model", "--no-drafts"]) == 0
    assert len(seen) == 8
    for command in seen:
        assert "--normalization-parity" not in command
        if rows is None:
            assert "--grouped-gate-up-rows" not in command
        else:
            at = command.index("--grouped-gate-up-rows")
            assert at < command.index("--") and command[at + 1] == str(rows)


@pytest.mark.parametrize("rows", ["1", "3", "8", "2.0"])
def test_private_parser_rejects_other_modes_before_output_or_spawn(monkeypatch, tmp_path, rows):
    monkeypatch.setattr(launcher, "supervise", lambda *a, **k: pytest.fail("invalid selector spawned ranks"))
    target = tmp_path / "fresh"
    with pytest.raises(SystemExit):
        owned.main(["--gpus", "0,1,2,3,4,5,6,7", "--out", str(target),
                    "--grouped-gate-up-rows", rows, "--", "model"])
    assert not target.exists()


def test_public_serving_parser_has_no_grouped_gate_up_flag():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "model", "--grouped-gate-up-rows", "2"])


@pytest.mark.parametrize("value", [True, 2, "2", object()])
def test_cli_capability_rejects_untyped_gate_up(value):
    with pytest.raises(TypeError, match="typed preparation"):
        cli._OwnedExpertQualification(lambda _: None, grouped_gate_up=value)


@pytest.mark.parametrize("rows", [2, 4])
def test_real_worker_publishes_bound_requested_actual_marker(channel, monkeypatch, tmp_path, rows):
    from tensorfold.families.qwen4_exp.cuda import owned_experts
    from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition

    monkeypatch.setattr(owned_experts, "_GROUPED_DOWN", "grouped-v1")
    ready = []
    monkeypatch.setattr(owned, "bound_receipt", lambda e: {"plan_identity": list(e._owned_expert_plan.identity)})
    def serve(args):
        capability = args._owned_expert_qualification
        assert capability.reference_tiles is False
        selected = capability.grouped_gate_up
        assert selected.rows == rows
        engine = NS(rank=3, _abort_comm=False, _owned_expert_gate_up_requested=selected,
                    _owned_expert_plan=owned_experts.Plan(ExpertPartition(8, 3, 512, 640, 4),
                        2560, 11, 32, 256, True, _gate_up=selected), comm=NS(ready=ready.append))
        capability.before_listener(engine)
        return 0
    monkeypatch.setattr(cli, "cmd_serve", serve)
    assert owned.worker(["--grouped-gate-up-rows", str(rows), *qualification.worker_args(tmp_path)]) == 0
    receipt = json.loads((tmp_path / "rank-3.json").read_text())
    assert receipt["grouped_gate_up"] == {"requested_rows": rows, "actual_rows": rows}
    assert receipt["plan_identity"][-2:] == ["grouped-gate-up", rows]
    assert ready == ["owned_expert_qualification"]


@pytest.mark.parametrize("old", ["cli", "engine"])
def test_selected_worker_refuses_old_application_before_constructor(channel, monkeypatch, tmp_path, old):
    if old == "cli":
        monkeypatch.setattr(cli, "_OwnedExpertQualification", NS(__dataclass_fields__={}))
    else:
        from tensorfold.families.qwen4_exp.cuda import engine
        class OldEngine:
            def __init__(self, unrelated):
                pytest.fail("old Engine constructed")
        monkeypatch.setattr(engine, "FlashNextEngine", OldEngine)
    monkeypatch.setattr(cli, "cmd_serve", lambda _: pytest.fail("old application reached constructor"))
    with pytest.raises(ValueError, match="installed application lacks"):
        owned.worker(["--grouped-gate-up-rows", "2", *qualification.worker_args(tmp_path)])
    assert os.read(channel[0], 8) == b"F"
    assert not (tmp_path / "rank-3.json").exists()


@pytest.mark.parametrize("requested", [None, 4, 2.0])
def test_unexpected_or_mismatched_plan_poisons_before_receipt_and_ready(channel, monkeypatch, tmp_path, requested):
    from tensorfold.families.qwen4_exp.cuda import owned_experts
    from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition

    monkeypatch.setattr(owned_experts, "_GROUPED_DOWN", "grouped-v1")
    selected = selection(2)
    engine = NS(rank=0, _abort_comm=False, _owned_expert_gate_up_requested=selected,
                _owned_expert_plan=owned_experts.Plan(ExpertPartition(8, 0, 512, 640, 4),
                    2560, 11, 32, 256, True, _gate_up=selected),
                comm=NS(ready=lambda _: pytest.fail("mismatch reached ready")))
    monkeypatch.setattr(owned, "bound_receipt", lambda _: {})
    with pytest.raises(ValueError, match="grouped gate/up"):
        owned.publish_receipt(engine, tmp_path, grouped_gate_up=requested)
    assert engine._abort_comm and os.read(channel[0], 8) == b"F"
    assert not (tmp_path / "rank-0.json").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_engine_keeps_selector_private_from_initializer_and_admin_options(monkeypatch, tmp_path, enabled):
    from tensorfold.families.qwen4_exp.cuda import engine, serving_profiles, serving_runtime

    seen = []
    monkeypatch.setattr(serving_profiles, "profile_root", lambda p: p)
    monkeypatch.setattr(serving_runtime, "ServingProfiles", lambda e, root, options: seen.append(options))
    monkeypatch.setattr(engine.FlashNextEngine, "_initialize", lambda *a, **kw: seen.append(kw))
    selected = selection(2) if enabled else None
    obj = engine.FlashNextEngine(tmp_path, _owned_expert_native=enabled,
        _owned_expert_gate_up=selected, serving_profile_root=tmp_path, adaptive_sessions=True)
    assert obj._owned_expert_gate_up_requested is selected
    assert len(seen) == 2
    for options in seen:
        assert "_owned_expert_gate_up" not in options and "_GroupedGateUp" not in options


@pytest.mark.parametrize("owned_mode,value", [(False, 2), (True, 2), (True, True)])
def test_engine_refuses_invalid_selection_before_initialization(monkeypatch, owned_mode, value):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    monkeypatch.setattr(FlashNextEngine, "_initialize", lambda *a, **kw: pytest.fail("invalid selector initialized"))
    with pytest.raises(ValueError, match="typed owned"):
        FlashNextEngine("unused", _owned_expert_native=owned_mode, _owned_expert_gate_up=value)
