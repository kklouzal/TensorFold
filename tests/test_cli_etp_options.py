"""Prepared expert-degree options cross CLI and family boundaries without model loads."""

from types import SimpleNamespace

import pytest

from tensorfold import cli, families, hub, serve_options


def parsed(model, degree=None):
    argv = ["serve", str(model), "--backend", "cuda", "--tp", "8", "--master", "127.0.0.1",
            "--no-drafts", "--max-pending-requests", "16", "--max-http-connections", "16",
            "--max-waiting-input-bytes", "4096"]
    if degree is not None:
        argv += ["--expert-tensor-parallel-size", str(degree)]
    return cli.build_parser().parse_args(argv)


@pytest.mark.parametrize("degree", [None, 2, 4, 8])
def test_parser_preserves_default_and_accepts_only_prepared_degrees(degree):
    assert parsed("checkpoint", degree).expert_tensor_parallel_size == degree


@pytest.mark.parametrize("degree", [0, 1, 3, 16])
def test_parser_rejects_excluded_degrees(degree):
    with pytest.raises(SystemExit):
        parsed("checkpoint", degree)


@pytest.mark.parametrize("degree", [True, False, 2.0, "4", 1, 16])
def test_programmatic_cli_namespace_rejects_invalid_degrees_before_source_io(degree, monkeypatch):
    from tensorfold import gguf_file

    args = parsed("checkpoint", 2)
    args.expert_tensor_parallel_size = degree
    family = SimpleNamespace(model_type="qwen4_exp")
    monkeypatch.setattr(gguf_file, "resolve_gguf", lambda *_: pytest.fail("invalid degree reached source I/O"))
    with pytest.raises(ValueError, match="must be 2, 4 or 8"):
        serve_options.check(args, family, "cuda")


@pytest.mark.parametrize("changes,backend,model_type", [
    ({"tp": 4}, "cuda", "qwen4_exp"),
    ({}, "mlx", "qwen4_exp"),
    ({}, "cuda", "gemma4"),
    ({"expert_ranks": 2}, "cuda", "qwen4_exp"),
    ({"expert_shards": "256,256"}, "cuda", "qwen4_exp"),
    ({"vram_experts": "auto"}, "cuda", "qwen4_exp"),
    ({"indexer_history": "ram"}, "cuda", "qwen4_exp"),
])
def test_etp_refuses_incompatible_service_ownership_before_source_io(changes, backend, model_type, monkeypatch):
    from tensorfold import gguf_file

    args = parsed("checkpoint", 4)
    vars(args).update(changes)
    monkeypatch.setattr(gguf_file, "resolve_gguf", lambda *_: pytest.fail("invalid service reached source I/O"))
    with pytest.raises(ValueError, match="requires"):
        serve_options.check(args, SimpleNamespace(model_type=model_type), backend)


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_real_gguf_metadata_passes_service_preflight(tmp_path, degree):
    from gguf_fixture import tiny_model

    path = tiny_model(tmp_path / "target.gguf", experts=8)
    args = parsed(path, degree)
    family = families.detect(path)
    serve_options.check(args, family, "cuda", config_dir=path)


@pytest.mark.parametrize("degree", [None, 2, 4, 8])
def test_cli_passes_explicit_degree_to_family_factory_only(degree, tmp_path, monkeypatch):
    from tensorfold.cuda import precision, prompt_precision

    class ReachedFactory(Exception):
        pass

    seen = {}

    def factory(_directory, **kwargs):
        seen.update(kwargs)
        raise ReachedFactory

    family = SimpleNamespace(title="Fixture", model_type="qwen4_exp",
                             package=SimpleNamespace(cuda_engine=factory))
    monkeypatch.setattr(cli, "_sampling_defaults", lambda *args: None)
    monkeypatch.setattr(cli, "_reply_tokens", lambda *args: 1)
    monkeypatch.setattr(precision, "set_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_precision, "set_fp8", lambda *args: None)
    with pytest.raises(ReachedFactory):
        cli._serve_cuda(parsed(tmp_path, degree), family, tmp_path)
    if degree is None:
        assert "expert_tensor_parallel_size" not in seen
    else:
        assert seen["expert_tensor_parallel_size"] == degree


@pytest.mark.parametrize("degree", [None, 2, 4, 8])
def test_family_forwards_explicit_degree_to_engine_only(tmp_path, degree, monkeypatch):
    from gguf_fixture import tiny_model
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine

    path = tiny_model(tmp_path / "target.gguf", experts=8)
    seen = {}

    class ReachedEngine(Exception):
        pass

    def factory(_directory, **kwargs):
        seen.update(kwargs)
        raise ReachedEngine

    monkeypatch.setattr(engine, "FlashNextEngine", factory)
    kwargs = {} if degree is None else {"expert_tensor_parallel_size": degree}
    with pytest.raises(ReachedEngine):
        qwen4_exp.cuda_engine(path, tp=8, no_drafts=True, max_waiting_input_bytes=4096, **kwargs)
    if degree is None:
        assert "expert_tensor_parallel_size" not in seen
    else:
        assert seen["expert_tensor_parallel_size"] == degree


@pytest.mark.parametrize("changes", [
    {"expert_tensor_parallel_size": True},
    {"expert_tensor_parallel_size": 2.0},
    {"expert_tensor_parallel_size": 1},
    {"tp": 4},
    {"rank": True},
    {"rank": 8},
    {"expert_ranks": 2},
    {"expert_shards": "256,256"},
    {"vram_experts": "auto"},
    {"indexer_history": "ram"},
])
def test_direct_family_boundary_rejects_invalid_partition_before_engine(tmp_path, changes, monkeypatch):
    from gguf_fixture import tiny_model
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine

    path = tiny_model(tmp_path / "target.gguf", experts=8)
    monkeypatch.setattr(engine, "FlashNextEngine", lambda *args, **kwargs: pytest.fail("invalid profile reached engine"))
    kwargs = {"tp": 8, "no_drafts": True, "expert_tensor_parallel_size": 4}
    kwargs.update(changes)
    with pytest.raises(ValueError, match="expert_tensor_parallel_size"):
        qwen4_exp.cuda_engine(path, **kwargs)


@pytest.mark.parametrize("degree", [True, False, 1, 3, 16, 2.0, "4"])
def test_prepared_manifest_refuses_invalid_degree_before_model_resolution(tmp_path, degree, monkeypatch):
    import json
    from test_serving_profile_manifest import profile_files
    from test_serving_profile_positions import UNIQUE_DEPLOYMENT
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.serving_profiles import prepare

    root, path, raw = profile_files(tmp_path)
    raw["options"]["expert_tensor_parallel_size"] = degree
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(hub, "resolve", lambda *args, **kwargs: pytest.fail("invalid degree reached model resolution"))
    with pytest.raises(ValueError, match="expert_tensor_parallel_size must be"):
        prepare(root, path.name, initializer=FlashNextEngine._initialize, fixed_options=UNIQUE_DEPLOYMENT)


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_prepared_manifest_validates_actual_partition_before_loading(tmp_path, degree):
    import hashlib
    import json
    from gguf_fixture import write
    from test_serving_profile_manifest import profile_files
    from test_serving_profile_positions import UNIQUE_DEPLOYMENT
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.serving_profiles import prepare
    from tensorfold.families.qwen4_exp.gguf import metadata_config
    from tensorfold.gguf_file import GGUFCheckpoint

    root, path, raw = profile_files(tmp_path)
    source = root / "target/model.gguf"
    fields = dict(GGUFCheckpoint.open(source).metadata)
    fields["qwen4exp.expert_count"] = 8
    fields["qwen4exp.expert_feed_forward_length"] = 640
    fields["qwen4exp.expert_shared_feed_forward_length"] = 640
    write(source, [("token_embd.weight", (1,), 0, [0, 0, 0, 0])], fields)
    config = metadata_config(source)
    config["gguf_file"] = source.name
    (source.parent / "config.json").write_text(json.dumps(config))
    raw["options"]["expert_tensor_parallel_size"] = degree
    for artifact in raw["artifacts"]:
        file = root / artifact["path"]
        artifact.update(size=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    path.write_text(json.dumps(raw))
    profile = prepare(root, path.name, initializer=FlashNextEngine._initialize, fixed_options=UNIQUE_DEPLOYMENT)
    assert profile.options["expert_tensor_parallel_size"] == degree


def test_prepared_manifest_rejects_incompatible_expert_coverage(tmp_path):
    import json
    from test_serving_profile_manifest import profile_files
    from test_serving_profile_positions import UNIQUE_DEPLOYMENT
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.serving_profiles import prepare

    root, path, raw = profile_files(tmp_path)  # Two experts cannot cover four EP groups.
    raw["options"]["expert_tensor_parallel_size"] = 2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="expert count and intermediate width"):
        prepare(root, path.name, initializer=FlashNextEngine._initialize, fixed_options=UNIQUE_DEPLOYMENT)


def test_expert_degree_is_not_a_live_runtime_execution_setting():
    from test_adaptive_runtime import app_for
    from tensorfold.server import runtime
    from tensorfold.server.errors import RequestError

    app = app_for()
    before = runtime.snapshot(app)
    with pytest.raises(RequestError, match="unsupported runtime setting"):
        runtime.request(app, {"expert_tensor_parallel_size": 4})
    assert runtime.snapshot(app) == before
