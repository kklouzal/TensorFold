"""Real metadata/manifest CLI preflight; server/device launch is the boundary."""
import json

import pytest

from tensorfold import cli
from tensorfold.families.qwen4_exp.position_profile import resolve_position_profile
from gguf_fixture import tiny_model


def position_manifest(directory, *, head_dim, factor=2):
    base = resolve_position_profile(directory, requested_context=None)
    policy = base.target.rope
    rows = policy.native_context * factor
    body = {"schema_version": 1, "family": "qwen4_exp", "epoch": 1, "requested_context": rows,
            "qualification": "unqualified-candidate", "target": {
                "source_config_sha256": base.target.source_config_sha256,
                "rope": {"rope_type": "yarn", "rope_theta": policy.theta,
                         "partial_rotary_factor": policy.rotary_dim / head_dim,
                         "mrope_section": list(policy.mrope_section), "mrope_interleaved": True,
                         "original_max_position_embeddings": policy.native_context, "factor": factor},
                "coordinate_limits": [rows] * 3}, "mtp": None, "dflash2": None}
    path = directory / "positions.json"
    path.write_text(json.dumps(body))
    return path, rows


def gguf_source(directory):
    from tensorfold.families.qwen4_exp.gguf import metadata_config

    directory.mkdir()
    path = tiny_model(directory / "target.gguf")
    raw = metadata_config(path)
    raw["gguf_file"] = path.name
    (directory / "config.json").write_text(json.dumps(raw))
    return directory


def command(directory, manifest):
    return ["serve", str(directory), "--backend", "cuda", "--no-drafts", "--no-update-check", "--tp", "8",
            "--master", "127.0.0.1", "--max-pending-requests", "1", "--max-http-connections", "1",
            "--position-profile-manifest", str(manifest)]


@pytest.mark.parametrize("context", [None, 0, 512])
def test_cli_resolves_manifest_before_native_guard_and_keeps_fixed_rows(tmp_path, monkeypatch, context):
    source = gguf_source(tmp_path / "source")
    manifest, rows = position_manifest(source, head_dim=32)
    observed = []
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, family, model, context: observed.append(
        (args.position_profile_manifest, model, context)) or 0)
    args = command(source, manifest)
    if context is not None:
        args += ["--context", str(context)]
    assert cli.cmd_serve(cli.build_parser().parse_args(args)) == 0
    assert observed == [(manifest, source, rows)]


def test_cli_refuses_mismatched_or_replica_profile_before_server_launch(tmp_path, monkeypatch):
    source = gguf_source(tmp_path / "source")
    manifest, _ = position_manifest(source, head_dim=32)
    monkeypatch.setattr(cli, "_serve_cuda", lambda *a, **k: pytest.fail("invalid position profile launched server"))
    with pytest.raises(ValueError, match="requested rows"):
        cli.cmd_serve(cli.build_parser().parse_args(command(source, manifest) + ["--context", "513"]))
    args = command(source, manifest)
    args[args.index("--tp") + 1] = "1"
    with pytest.raises(ValueError, match="unique TP8"):
        cli.cmd_serve(cli.build_parser().parse_args(args))


def test_nv_cached_plan_uses_resolved_position_policy_not_native_default(tmp_path, monkeypatch):
    from sm70_nvfp4_fixture import write
    from tensorfold.families.qwen4_exp.cuda.sm70_weight_manifest import prepare

    source, _ = write(tmp_path / "source")
    family = source / "family.json"
    family.write_text(json.dumps(prepare(source, repository="controlled/source", revision="fixed", norm_offset=1)))
    manifest, rows = position_manifest(source, head_dim=256)
    observed = []
    def launch(args, selected, directory, context):
        prototype, _planner = args._prepared_family_plan
        observed.append((prototype.cfg.rope.rope_type, prototype.cfg.rope.factor, context))
        return 0
    monkeypatch.setattr(cli, "_serve_cuda", launch)
    args = command(source, manifest) + ["--family-manifest", str(family), "--precision", "full"]
    assert cli.cmd_serve(cli.build_parser().parse_args(args)) == 0
    assert observed == [("yarn", 2, rows)]
