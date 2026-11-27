import json

import pytest

from tensorfold import cli, families
from tensorfold.families import deepseek_v4


@pytest.fixture
def config(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "gguf_file": "backbone.gguf"}))
    (model / "backbone.gguf").touch()
    draft = tmp_path / "dspark"
    draft.mkdir()
    return model, draft


def command(model, draft):
    return ["serve", str(model), "--backend", "cuda", "--tp", "8", "--master", "127.0.0.1",
            "--kv-dtype", deepseek_v4.ROTOR_KV_DTYPE, "--kv-full-windows", "auto", "--context", "1048576",
            "--draft-method", "dspark", "--drafter", str(draft)]


def test_full_window_d3_uses_the_local_draft_bundle(config, monkeypatch):
    from tensorfold.families.deepseek_v4.cuda import engine

    model, draft = config
    args = cli.build_parser().parse_args([*command(model, draft), "--mtp-drafts", "3"])
    cli._check_serve_options(args, families.detect(model), "cuda", model)
    monkeypatch.setattr(engine, "DS41Engine", lambda path, **options: (path, options))
    path, options = deepseek_v4.cuda_engine(model, drafter=args.drafter, draft_method=args.draft_method,
                                            mtp_drafts=args.mtp_drafts, tp=8, context=args.context,
                                            kv_full_windows=args.kv_full_windows, kv_dtype=args.kv_dtype)
    assert path == model and options["context"] == 1048576
    assert options["draft_root"] == draft and options["draft_depth"] == 3


@pytest.mark.parametrize(("opts", "message"), [
    (["--mtp-drafts", "5"], "supports depth 3"),
    (["--no-drafts"], "cannot use --no-drafts"),
    (["--drafter", "auto"], "local FT bundle"),
    (["--drafter", "nonexistent-local-fixture"], "local FT bundle"),
    (["--kv-dtype", "bf16"], "requires TP8/ETP4 RotorQuant"),
    (["--expert-tensor-parallel-size", "2"], "requires TP8/ETP4 RotorQuant"),
    (["--no-graphs"], "CUDA graphs"),
    (["--yarn-factor", "2"], "native YaRN"),
])
def test_d3_refuses_unsupported_profiles(config, opts, message):
    model, draft = config
    args = cli.build_parser().parse_args([*command(model, draft), *opts])
    with pytest.raises(ValueError, match=message):
        cli._check_serve_options(args, families.detect(model), "cuda", model)


def test_ordinary_deepseek_does_not_construct_a_draft_owner(config, monkeypatch):
    from tensorfold.families.deepseek_v4.cuda import engine

    model, draft = config
    monkeypatch.setattr(engine, "DS41Engine", lambda path, **options: options)
    options = deepseek_v4.cuda_engine(model, drafter=str(draft), tp=8)
    assert "draft_root" not in options and "draft_depth" not in options
