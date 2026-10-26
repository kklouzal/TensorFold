"""Explicit media flags reach only the new resident SM70 startup path."""
import pytest

from tensorfold import cli
from test_position_cli import gguf_source


def command(source, manifest):
    return ["serve", str(source), "--backend", "cuda", "--no-drafts", "--no-update-check",
            "--tp", "8", "--master", "127.0.0.1", "--max-pending-requests", "1",
            "--max-http-connections", "1", "--vision", "--no-graphs", "--media-profile-manifest", str(manifest)]


def test_cli_accepts_explicit_profile_without_entering_legacy_vision_configuration(tmp_path, monkeypatch):
    source = gguf_source(tmp_path / "source")
    manifest = tmp_path / "media.json"
    seen = []
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, *_a, **_k: seen.append(
        (args.media_profile_manifest, args.no_graphs, args.vision)) or 0)
    assert cli.cmd_serve(cli.build_parser().parse_args(command(source, manifest))) == 0
    assert seen == [(manifest, True, True)]


@pytest.mark.parametrize("removed", ["--vision", "--no-graphs"])
def test_cli_requires_explicit_enable_and_eager_contract(tmp_path, monkeypatch, removed):
    source = gguf_source(tmp_path / "source")
    args = command(source, tmp_path / "media.json")
    args.remove(removed)
    monkeypatch.setattr(cli, "_serve_cuda", lambda *_a, **_k: pytest.fail("invalid profile launched server"))
    with pytest.raises(ValueError, match="media-profile-manifest requires"):
        cli.cmd_serve(cli.build_parser().parse_args(args))


@pytest.mark.parametrize("extra", [["--vision-image-tokens", "32"], ["--vision-max-images", "2"], ["--vision-offload"]])
def test_profile_refuses_legacy_limits_and_offload(tmp_path, monkeypatch, extra):
    source = gguf_source(tmp_path / "source")
    monkeypatch.setattr(cli, "_serve_cuda", lambda *_a, **_k: pytest.fail("legacy override launched server"))
    with pytest.raises(ValueError, match="media profile owns all media limits"):
        cli.cmd_serve(cli.build_parser().parse_args(command(source, tmp_path / "media.json") + extra))


def test_family_preserves_new_options_and_automatic_full_window_capacity(tmp_path, monkeypatch):
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine

    source = gguf_source(tmp_path / "source")
    seen = []
    monkeypatch.setattr(engine, "FlashNextEngine", lambda directory, **options: seen.append((directory, options)))
    qwen4_exp.cuda_engine(source, no_drafts=True, tp=8, master="127.0.0.1", kv_dtype="int8",
                         vision=True, graphs=False, media_profile_manifest=tmp_path / "media.json")
    assert len(seen) == 1
    directory, options = seen[0]
    assert directory == source
    assert options["media_profile_manifest"] == tmp_path / "media.json"
    assert options["graphs"] is False and options["vision"] is True
    assert options["adaptive_sessions"] is True and options["session_ceiling"] is None


@pytest.mark.parametrize("value", ["0", "-1"])
def test_principal_limit_rejects_nonpositive_count_before_startup(tmp_path, monkeypatch, value):
    source = gguf_source(tmp_path / "source")
    monkeypatch.setattr(cli, "_serve_cuda", lambda *_a, **_k: pytest.fail("invalid quota launched server"))
    args = command(source, tmp_path / "media.json") + ["--max-pending-requests-per-principal", value]
    with pytest.raises(ValueError, match="max-pending-requests-per-principal must be a positive integer"):
        cli.cmd_serve(cli.build_parser().parse_args(args))


def test_principal_limit_requires_explicit_total_limit():
    from tensorfold.serve_options import check_numbers

    args = cli.build_parser().parse_args(["serve", "model", "--max-pending-requests-per-principal", "2"])
    with pytest.raises(ValueError, match="requires --max-pending-requests"):
        check_numbers(args)
