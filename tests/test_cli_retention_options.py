"""Admission and retained-service options are resolved before model acquisition."""

import json
import threading
from types import SimpleNamespace

import pytest

from tensorfold import cli
from tensorfold.server import authentication
from tensorfold.serve_options import check_numbers, prepare_service_options


@pytest.mark.parametrize("flag", ["--max-pending-requests-per-principal", "--max-waiting-input-bytes-per-principal"])
def test_principal_request_quota_requires_authentication_before_model_metadata(monkeypatch, flag):
    monkeypatch.delenv("TENSORFOLD_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_config_dir", lambda *_a, **_k: pytest.fail("unauthenticated quota read model metadata"))
    args = cli.build_parser().parse_args(["serve", "unresolved-model", "--no-update-check",
                                          "--max-pending-requests", "4",
                                          flag, "2"])
    with pytest.raises(ValueError, match="requires configured API keys before model loading"):
        cli.cmd_serve(args)


@pytest.mark.parametrize("source", ["argument", "environment", "file"])
@pytest.mark.parametrize("flag", ["--max-pending-requests-per-principal", "--max-waiting-input-bytes-per-principal"])
def test_principal_quota_reuses_one_consumed_authentication_store(tmp_path, monkeypatch, source, flag):
    monkeypatch.delenv("TENSORFOLD_API_KEY", raising=False)
    flags = []
    if source == "argument":
        flags = ["--api-key", "fixture-key"]
    elif source == "environment":
        monkeypatch.setenv("TENSORFOLD_API_KEY", "fixture-key")
    else:
        path = tmp_path / "keys.txt"
        path.write_text("fixture-key\n")
        path.chmod(0o600)
        flags = ["--api-key-file", str(path)]
    stores, configure = [], authentication.configure

    def configured(args):
        value = configure(args)
        stores.append(value)
        return value

    def metadata(*_a, **_k):
        raise RuntimeError("configured metadata boundary")

    monkeypatch.setattr(authentication, "configure", configured)
    monkeypatch.setattr(cli, "_config_dir", metadata)
    args = cli.build_parser().parse_args(["serve", "unresolved-model", "--no-update-check",
                                          "--max-pending-requests", "4",
                                          flag, "2", *flags])
    with pytest.raises(RuntimeError, match="configured metadata boundary"):
        cli.cmd_serve(args)
    assert len(stores) == 1 and stores[0].enabled and args.auth is stores[0]
    assert args.api_key == [] and "TENSORFOLD_API_KEY" not in cli.os.environ


def adaptive_command(directory, *extra):
    return ["serve", str(directory), "--backend", "cuda", "--no-drafts", "--no-update-check", "--tp", "8",
            "--master", "127.0.0.1", "--max-pending-requests", "2", "--max-http-connections", "2",
            "--max-waiting-input-bytes", "4096", *extra]


def test_absent_retention_and_api_storage_remain_request_scoped():
    args = cli.build_parser().parse_args(["serve", "model"])
    check_numbers(args)
    prepare_service_options(args)
    assert args.retention_policy is None and args._api_record_limits is None
    assert args.max_waiting_input_bytes is args.queue_timeout_s is args.output_queue_tokens is None
    assert args.api_record_store is args.api_record_limits is None


@pytest.mark.parametrize("flag,value", [("--max-waiting-input-bytes", "0"),
                                        ("--max-waiting-input-bytes", "-1"),
                                        ("--max-waiting-input-bytes-per-principal", "0"),
                                        ("--max-waiting-input-bytes-per-principal", "-1"),
                                        ("--output-queue-tokens", "0"), ("--output-queue-tokens", "-1"),
                                        ("--queue-timeout-s", "0"), ("--queue-timeout-s", "-1"),
                                        ("--queue-timeout-s", "inf"), ("--queue-timeout-s", "nan"),
                                        ("--queue-timeout-s", str(threading.TIMEOUT_MAX * 2))])
def test_fifo_invalid_bounds_refuse_before_auth_or_model_access(monkeypatch, flag, value):
    monkeypatch.setattr(authentication, "configure", lambda *_a: pytest.fail("invalid limits acquired authentication"))
    args = cli.build_parser().parse_args(["serve", "model", flag, value])
    with pytest.raises(ValueError, match="positive"):
        cli.cmd_serve(args)


def test_programmatic_queue_timeout_huge_integer_is_a_configuration_error():
    args = cli.build_parser().parse_args(["serve", "model"])
    args.queue_timeout_s = 10 ** 400
    with pytest.raises(ValueError, match="queue-timeout-s must be positive"):
        check_numbers(args)


@pytest.mark.parametrize("extra", [["--retained-sessions", "2"], ["--retention-tier", "gpu"],
                                   ["--retention-tier", "ram", "--retained-sessions", "2",
                                    "--retention-lifetime-s", "60"]])
def test_partial_retention_policy_has_no_implicit_limits(extra):
    args = cli.build_parser().parse_args(["serve", "model", *extra])
    with pytest.raises(ValueError, match="retention"):
        prepare_service_options(args)


def test_actual_unique_profile_requires_explicit_waiting_input_budget(tmp_path, monkeypatch):
    from test_position_cli import gguf_source

    directory = gguf_source(tmp_path / "model")
    command = adaptive_command(directory)
    at = command.index("--max-waiting-input-bytes")
    del command[at:at + 2]
    monkeypatch.setattr(cli, "_serve_cuda", lambda *_a, **_k: pytest.fail("unbounded FIFO launched a model"))
    with pytest.raises(ValueError, match="requires explicit positive --max-waiting-input-bytes"):
        cli.cmd_serve(cli.build_parser().parse_args(command))


def test_principal_waiting_byte_limit_is_not_clamped_to_global_budget(tmp_path, monkeypatch):
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine
    from test_position_cli import gguf_source

    directory = gguf_source(tmp_path / "model")
    seen = []
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, *_a, **_k: seen.append(args) or 0)
    command = adaptive_command(directory, "--max-waiting-input-bytes-per-principal", "8192",
                               "--api-key", "fixture-key")
    assert cli.cmd_serve(cli.build_parser().parse_args(command)) == 0
    assert seen[0].max_waiting_input_bytes == 4096
    assert seen[0].max_waiting_input_bytes_per_principal == 8192
    captured = []
    monkeypatch.setattr(engine, "FlashNextEngine", lambda *_a, **kwargs: captured.append(kwargs))
    qwen4_exp.cuda_engine(directory, no_drafts=True, tp=8, master="fixture",
                         max_waiting_input_bytes=4096, max_waiting_input_bytes_per_principal=8192)
    assert captured[0]["max_waiting_input_bytes_per_principal"] == 8192


@pytest.mark.parametrize("tier", ["gpu", "ram"])
def test_policy_and_fifo_values_reach_cli_and_family_unchanged(tmp_path, monkeypatch, tier):
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine
    from test_position_cli import gguf_source

    directory = gguf_source(tmp_path / "model")
    flags = ["--queue-timeout-s", ".25", "--output-queue-tokens", "1",
             "--retention-tier", tier, "--retained-sessions", "2",
             "--retention-lifetime-s", "60"]
    if tier == "ram":
        flags.extend(["--retention-ram-bytes", "1073741824", "--retention-chunk-bytes", "4096",
                      "--retention-progress-bytes", "65536", "--retention-overlap-bytes", "0"])
    seen = []
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, *_a, **_k: seen.append(args) or 0)
    assert cli.cmd_serve(cli.build_parser().parse_args(adaptive_command(directory, *flags))) == 0
    args = seen[0]
    policy = args.retention_policy
    assert policy.tier == tier and policy.max_sessions == 2 and policy.lifetime_s == 60
    assert args.max_waiting_input_bytes == 4096 and args.queue_timeout_s == .25 and args.output_queue_tokens == 1
    factory = []
    monkeypatch.setattr(engine, "FlashNextEngine", lambda *a, **k: factory.append(k))
    qwen4_exp.cuda_engine(directory, no_drafts=True, tp=8, master="fixture",
                         max_waiting_input_bytes=args.max_waiting_input_bytes,
                         output_queue_tokens=args.output_queue_tokens,
                         queue_timeout_s=args.queue_timeout_s, retention_policy=policy)
    assert factory[0]["retention_policy"] is policy
    assert factory[0]["max_waiting_input_bytes"] == 4096 and factory[0]["queue_timeout_s"] == .25
    assert factory[0]["output_queue_tokens"] == 1


def test_ram_progress_budget_is_global_for_the_actual_world(tmp_path, monkeypatch):
    from test_position_cli import gguf_source

    directory = gguf_source(tmp_path / "model")
    flags = ["--retention-tier", "ram", "--retained-sessions", "2", "--retention-lifetime-s", "60",
             "--retention-ram-bytes", "1073741824", "--retention-chunk-bytes", "4096",
             "--retention-progress-bytes", "8192", "--retention-overlap-bytes", "0"]
    monkeypatch.setattr(cli, "_serve_cuda", lambda *_a, **_k: pytest.fail("insufficient TP8 progress launched a model"))
    with pytest.raises(ValueError, match="progress must cover.*per rank"):
        cli.cmd_serve(cli.build_parser().parse_args(adaptive_command(directory, *flags)))


@pytest.mark.parametrize("flag", ["--max-waiting-input-bytes", "--output-queue-tokens"])
def test_fifo_and_retention_are_rejected_for_a_legacy_profile(tmp_path, flag):
    from tensorfold.families import qwen4_exp

    args = cli.build_parser().parse_args(["serve", str(tmp_path), flag, "4096"])
    check_numbers(args)
    family = SimpleNamespace(title=qwen4_exp.TITLE, model_type="qwen4_exp", package=qwen4_exp)
    with pytest.raises(ValueError, match="require actual adaptive Flash Next unique TP8"):
        cli._check_serve_options(args, family, "cuda", tmp_path)


def test_enabled_dspark_cli_and_factory_preserve_legacy_admission(tmp_path, monkeypatch):
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine
    from test_position_cli import gguf_source

    directory = gguf_source(tmp_path / "model")
    draft = tmp_path / "dspark"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({"architectures": ["DSparkDraftModel"]}))
    command = adaptive_command(directory, "--drafter", str(draft), "--mtp-drafts", "2")
    command.remove("--no-drafts")
    at = command.index("--max-waiting-input-bytes")
    del command[at:at + 2]
    seen = []
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, *_a, **_k: seen.append(args) or 0)
    assert cli.cmd_serve(cli.build_parser().parse_args(command)) == 0
    assert seen[0].max_waiting_input_bytes is None and seen[0].retention_policy is None
    captured = []
    monkeypatch.setattr(engine, "FlashNextEngine", lambda *_a, **kwargs: captured.append(kwargs))
    qwen4_exp.cuda_engine(directory, drafter=str(draft), mtp_drafts=2, tp=8, master="fixture")
    assert captured[0]["dspark_model_dir"] == draft and captured[0]["adaptive_sessions"] is True
    assert "max_waiting_input_bytes" not in captured[0]
    with pytest.raises(ValueError, match="legacy DSpark without implemented positions"):
        cli.cmd_serve(cli.build_parser().parse_args(command + ["--max-waiting-input-bytes", "4096"]))
    with pytest.raises(ValueError, match="legacy DSpark without implemented positions"):
        qwen4_exp.cuda_engine(directory, drafter=str(draft), mtp_drafts=2, tp=8, master="fixture",
                             max_waiting_input_bytes=4096)
    with pytest.raises(ValueError, match="legacy DSpark without implemented positions"):
        cli.cmd_serve(cli.build_parser().parse_args(command + ["--output-queue-tokens", "1"]))
    with pytest.raises(ValueError, match="legacy DSpark without implemented positions"):
        qwen4_exp.cuda_engine(directory, drafter=str(draft), mtp_drafts=2, tp=8, master="fixture",
                             output_queue_tokens=1)


@pytest.mark.parametrize("limits", [{}, {"max_waiting_input_bytes_per_principal": 8192}, {"output_queue_tokens": 1}])
def test_actual_dspark_preload_keeps_no_position_context_and_legacy_admission(tmp_path, monkeypatch, limits):
    import torch
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp.cuda import dspark_admission, engine
    from test_flashnext_shared_head import pair

    target, _companion = pair(tmp_path)
    draft = tmp_path / "dspark"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({"architectures": ["DSparkDraftModel"]}))
    monkeypatch.setattr(chat_template, "prepare_chat_template", lambda *_a, **_k: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_a, **_k: pytest.fail("DSpark metadata preflight touched CUDA"))
    seen = []

    class InspectDraft(Exception):
        pass

    def inspected(directory, target_config, *, tp):
        assert directory == draft and target_config["hidden_size"] == 256 and tp == (0, 8)
        raise InspectDraft

    def retired(owner):
        assert owner._history_placement is not None and owner.position_profile is None
        assert owner.w is None and owner.comm is None and owner.scheduler is None
        seen.append(getattr(owner, "_admission_options", None))

    monkeypatch.setattr(dspark_admission, "inspect", inspected)
    monkeypatch.setattr(engine.FlashNextEngine, "_rollback_startup", retired)
    expected = pytest.raises(InspectDraft) if not limits else pytest.raises(ValueError, match="FIFO input/deadline")
    with expected:
        engine.FlashNextEngine(target, depth=2, tp=8, rank=0, master="fixture", graphs=False,
                                adaptive_sessions=True, kv_dtype="int8", dspark_model_dir=draft,
                                **limits)
    assert seen == ([{"admission_policy": "legacy", "max_waiting_input_bytes": None,
                     "max_waiting_input_bytes_per_principal": None, "output_queue_tokens": None, "queue_timeout_s": None}]
                    if not limits else [None])


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8192"])
@pytest.mark.parametrize("name", ["max_waiting_input_bytes_per_principal", "output_queue_tokens"])
def test_real_engine_rejects_invalid_queue_budget_before_source_metadata(tmp_path, monkeypatch, value, name):
    from tensorfold.families.qwen4_exp import gguf
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setattr(gguf, "resolve_gguf", lambda *_a, **_k: pytest.fail("invalid byte cap read source metadata"))
    with pytest.raises(ValueError, match=name + " must be a positive integer"):
        FlashNextEngine(tmp_path / "unread-model", **{name: value})


def test_mtp_gguf_drafter_resolution_never_prepares_a_second_target_frontend(tmp_path, monkeypatch):
    from tensorfold import hub
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp import gguf_assets
    from test_flashnext_shared_head import pair

    _target, companion = pair(tmp_path)
    family = SimpleNamespace(model_type="qwen4_exp", package=qwen4_exp)
    monkeypatch.setattr(hub, "resolve", lambda *_a, **_k: pytest.fail("MTP-only GGUF entered target/weight resolution"))
    monkeypatch.setattr(gguf_assets, "prepare", lambda *_a, **_k: pytest.fail("MTP-only GGUF prepared another frontend"))
    assert cli._drafter(family, str(companion), "cuda") == str(companion.resolve())
    assert cli._drafter_metadata_source(str(companion)) == companion.resolve()


def test_explicit_mtp_gguf_position_preflight_uses_its_actual_role(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp import gguf_assets
    from tensorfold.families.qwen4_exp.gguf import metadata_config
    from test_flashnext_shared_head import pair
    from test_position_cli import position_manifest

    directory = tmp_path / "model"
    directory.mkdir()
    target, companion = pair(directory)
    config = metadata_config(target)
    config["gguf_file"] = target.name
    (directory / "config.json").write_text(json.dumps(config))
    manifest, rows = position_manifest(directory, head_dim=32, factor=1)
    command = adaptive_command(directory, "--drafter", str(companion), "--mtp-drafts", "1",
                               "--position-profile-manifest", str(manifest))
    command.remove("--no-drafts")
    monkeypatch.setattr(gguf_assets, "prepare", lambda *_a, **_k: pytest.fail("companion entered target frontend preparation"))
    seen = []
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, _family, _directory, context:
                        seen.append((args.position_profile_manifest, context)) or 0)
    assert cli.cmd_serve(cli.build_parser().parse_args(command)) == 0
    assert seen == [(manifest, rows)]


@pytest.mark.parametrize("tier", ["gpu", "ram"])
@pytest.mark.parametrize("principal_bytes", [None, 8192])
@pytest.mark.parametrize("output_tokens", [None, 1])
def test_real_engine_preload_binds_fifo_and_typed_policy_before_cuda(tmp_path, monkeypatch, tier, principal_bytes, output_tokens):
    import torch
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp.cuda import engine, speculation
    from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
    from test_flashnext_shared_head import pair

    target, _companion = pair(tmp_path)
    policy = (RetentionPolicy("gpu", 2, 60.) if tier == "gpu" else
              RetentionPolicy("ram", 2, 60., 1073741824, 4096, 65536, 0))
    monkeypatch.setattr(chat_template, "prepare_chat_template", lambda *_a, **_k: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_a, **_k: pytest.fail("preload configuration touched CUDA"))
    seen = []

    class Preloaded(Exception):
        pass

    def next_stage(*_a, **_k):
        raise Preloaded

    def retired(owner):
        assert owner.w is None and owner.comm is None and owner.scheduler is None
        seen.append((owner._admission_options, owner.retention_policy))

    monkeypatch.setattr(speculation, "MethodGeometry", next_stage)
    monkeypatch.setattr(engine.FlashNextEngine, "_rollback_startup", retired)
    with pytest.raises(Preloaded):
        engine.FlashNextEngine(target, depth=0, tp=8, rank=0, master="fixture", graphs=False,
                                adaptive_sessions=True, kv_dtype="int8", max_waiting_input_bytes=4096,
                                max_waiting_input_bytes_per_principal=principal_bytes,
                                output_queue_tokens=output_tokens,
                                queue_timeout_s=.25, retention_policy=policy)
    assert seen == [({"admission_policy": "fifo", "max_waiting_input_bytes": 4096,
                     "max_waiting_input_bytes_per_principal": principal_bytes,
                     "output_queue_tokens": output_tokens, "queue_timeout_s": .25}, policy)]
    from tensorfold.cuda.scheduler import Scheduler
    from test_cuda_fifo_admission import Decoder

    scheduler = Scheduler(Decoder(), max_pending_requests=2, **seen[0][0])
    try:
        assert scheduler.queue_record()["max_waiting_input_bytes_per_principal"] == principal_bytes
        assert scheduler.waiting.max_principal_bytes == principal_bytes
        assert scheduler.output_queue_tokens == output_tokens
    finally:
        scheduler.close()


def limits_record():
    return {"schema": "tensorfold.api-record-limits", "version": 1, "max_responses": 4,
            "max_conversations": 4, "max_items": 32, "max_total_bytes": 1048576,
            "max_record_bytes": 65536, "max_input_items": 4, "max_output_items": 8,
            "max_history_items": 16, "max_history_bytes": 262144, "max_active_operations": 2,
            "max_principal_responses": 2, "max_principal_conversations": 2, "max_principal_items": 16,
            "max_principal_bytes": 524288, "max_principal_operations": 1, "max_materialized_bytes": 1048576,
            "max_database_bytes": 8388608, "max_operation_bytes": 262144, "response_retention_seconds": None}


@pytest.mark.parametrize("flag", ["--api-record-store", "--api-record-limits"])
def test_api_records_require_both_explicit_path_and_limits(flag):
    args = cli.build_parser().parse_args(["serve", "model", flag, "file"])
    with pytest.raises(ValueError, match="must be configured together"):
        prepare_service_options(args)


def test_api_record_limits_are_frozen_before_authentication_or_model_work(tmp_path, monkeypatch):
    path = tmp_path / "limits.json"
    path.write_text(json.dumps(limits_record()))
    args = cli.build_parser().parse_args(["serve", "model", "--api-record-store", str(tmp_path / "api.db"),
                                          "--api-record-limits", str(path), "--max-http-connections", "2"])
    check_numbers(args)
    prepare_service_options(args)
    prepared = args._api_record_limits
    path.write_text("{}")
    app, owners = SimpleNamespace(), {}
    cli._configure_api_records(args, app, owners)
    assert app.api_record_store.limits is prepared and owners["api_records"] is app.api_record_store
    cli._close_serving(owners, None)
    assert app.api_record_store._closed


def test_api_records_require_explicit_transport_bound_before_auth_or_model_work(tmp_path, monkeypatch):
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps(limits_record()))
    monkeypatch.setattr(authentication, "configure", lambda *_a: pytest.fail("unbounded API records acquired authentication"))
    args = cli.build_parser().parse_args(["serve", "model", "--api-record-store", str(tmp_path / "api.db"),
                                          "--api-record-limits", str(limits)])
    with pytest.raises(ValueError, match="durable API records require explicit positive --max-http-connections"):
        cli.cmd_serve(args)
    assert not (tmp_path / "api.db").exists()


@pytest.mark.parametrize("drain_failure", [False, True])
def test_logical_store_closes_after_its_request_and_model_owners(drain_failure):
    events = []
    failure = RuntimeError("handlers remain active")

    def server_close():
        events.append("server")
        if drain_failure:
            raise failure

    owners = {"server": SimpleNamespace(server_close=server_close, handlers_drained=not drain_failure),
              "app": SimpleNamespace(close=lambda: events.append("app")),
              "engine": SimpleNamespace(close=lambda: events.append("engine")),
              "api_records": SimpleNamespace(close=lambda: events.append("records"))}
    if drain_failure:
        with pytest.raises(RuntimeError) as caught:
            cli._close_serving(owners, None)
        assert caught.value is failure and events == ["server"]
    else:
        cli._close_serving(owners, None)
        assert events == ["server", "app", "engine", "records"]


@pytest.mark.parametrize("failure_phase,large_body", [(None, False), ("app", False), ("binding", False),
                                                     ("handler", False), (None, True)])
def test_cuda_cli_forwards_policy_and_owns_store_before_handler_publication(tmp_path, monkeypatch, request, failure_phase, large_body):
    from tensorfold.cuda import server
    from tensorfold.families import qwen4_exp
    from tensorfold.server.api_records import ApiRecordStore
    from tensorfold.server.request_body import LIMIT
    from tests.cli_server_fixture import cuda_cli_app, cuda_cli_engine, install_cli_server
    from test_position_cli import gguf_source

    directory = gguf_source(tmp_path / "model")
    limits = tmp_path / "limits.json"
    record = limits_record()
    if large_body:
        record.update(max_operation_bytes=LIMIT + 1, max_total_bytes=LIMIT + 1, max_principal_bytes=LIMIT + 1)
    limits.write_text(json.dumps(record))
    args = cli.build_parser().parse_args(adaptive_command(directory,
        "--queue-timeout-s", ".5", "--max-waiting-input-bytes-per-principal", "8192",
        "--output-queue-tokens", "1",
        "--retention-tier", "gpu", "--retained-sessions", "2",
        "--retention-lifetime-s", "60", "--api-record-store", str(tmp_path / "api.db"),
        "--api-record-limits", str(limits), "--api-key", "fixture-key", "--kv-dtype", "int8"))
    check_numbers(args)
    prepare_service_options(args)
    args.auth = authentication.configure(args)
    family = SimpleNamespace(title=qwen4_exp.TITLE, model_type="qwen4_exp", package=qwen4_exp)
    cli._check_serve_options(args, family, "cuda", directory)
    options, stores, bindings, events = [], [], [], []
    primary = RuntimeError("record binding or handler publication failed")

    def bind(store):
        assert not store._closed
        bindings.append(store)
        if failure_phase == "binding":
            raise primary

    model = cuda_cli_engine(max_len=256, close=lambda: events.append("engine"))
    model.bind_api_record_store = bind
    monkeypatch.setattr(qwen4_exp, "cuda_engine", lambda *_a, **kwargs: options.append(kwargs) or model)

    def make_app(*_a, **kwargs):
        assert kwargs["max_http_connections"] == 2
        assert kwargs["max_http_body_bytes"] == max(LIMIT, args._api_record_limits.max_operation_bytes)
        assert kwargs["logical_api_limits"] is args._api_record_limits
        app = cuda_cli_app(effective_context_window=256, close=lambda: events.append("app"))
        kwargs["startup_owner"](app)
        if failure_phase == "app":
            raise primary
        return app

    if failure_phase == "app":
        monkeypatch.setattr(qwen4_exp, "CUDA_APP", make_app, raising=False)
    else:
        monkeypatch.setattr(server, "App", make_app)
    monkeypatch.setattr(cli.stacks, "arm", lambda: None)

    def handler(app):
        assert app.auth is args.auth and not app.api_record_store._closed
        stores.append(app.api_record_store)
        if failure_phase == "handler":
            raise primary
        return object()

    close = ApiRecordStore.close

    def closed(store):
        events.append("records")
        return close(store)

    monkeypatch.setattr(server, "make_handler", handler)
    monkeypatch.setattr(ApiRecordStore, "close", closed)
    install_cli_server(monkeypatch, request, server)
    if failure_phase is not None:
        with pytest.raises(RuntimeError) as caught:
            cli._serve_cuda(args, family, directory, 256)
        assert caught.value is primary
    else:
        assert cli._serve_cuda(args, family, directory, 256) == 0
    assert options[0]["retention_policy"] is args.retention_policy
    assert options[0]["max_waiting_input_bytes"] == 4096 and options[0]["queue_timeout_s"] == .5
    assert options[0]["max_waiting_input_bytes_per_principal"] == 8192
    assert options[0]["output_queue_tokens"] == 1
    assert events == (["app", "engine"] if failure_phase == "app" else ["app", "engine", "records"])
    assert len(bindings) == int(failure_phase != "app") and all(store._closed for store in bindings)
    assert len(stores) == int(failure_phase not in ("app", "binding"))
    if stores:
        assert stores[0] is bindings[0]


def test_managed_profile_keeps_policy_typed_and_binds_its_record_to_identity(tmp_path):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
    from tensorfold.families.qwen4_exp.cuda.serving_profiles import prepare
    from test_serving_profile_manifest import profile_files

    root, path, raw = profile_files(tmp_path)
    policy = RetentionPolicy("gpu", 2, 60.0)
    fixed = {"tp": 8, "rank": 0, "adaptive_sessions": True, "indexer_history": "gpu",
             "max_waiting_input_bytes": 4096, "queue_timeout_s": .5, "retention_policy": policy}
    profile = prepare(root, path.name, initializer=FlashNextEngine._initialize, fixed_options=fixed)
    assert profile.options["retention_policy"] is policy
    assert profile.options["max_waiting_input_bytes"] == 4096
    equivalent = prepare(root, path.name, initializer=FlashNextEngine._initialize,
                         fixed_options={**fixed, "retention_policy": RetentionPolicy.from_record(policy.record())})
    assert equivalent.identity == profile.identity
    explicit_unset = prepare(root, path.name, initializer=FlashNextEngine._initialize,
                             fixed_options={**fixed, "max_waiting_input_bytes_per_principal": None,
                                            "output_queue_tokens": None})
    assert explicit_unset.identity == profile.identity
    assert explicit_unset.options["max_waiting_input_bytes_per_principal"] is None
    assert explicit_unset.options["output_queue_tokens"] is None
    capped = prepare(root, path.name, initializer=FlashNextEngine._initialize,
                     fixed_options={**fixed, "max_waiting_input_bytes_per_principal": 8192})
    assert capped.identity != profile.identity
    assert capped.options["max_waiting_input_bytes_per_principal"] == 8192
    output_capped = prepare(root, path.name, initializer=FlashNextEngine._initialize,
                            fixed_options={**fixed, "output_queue_tokens": 1})
    assert output_capped.identity != profile.identity
    assert output_capped.options["output_queue_tokens"] == 1
    changed = prepare(root, path.name, initializer=FlashNextEngine._initialize,
                      fixed_options={**fixed, "retention_policy": RetentionPolicy("gpu", 3, 60.0)})
    assert changed.identity != profile.identity
    raw["options"]["retention_policy"] = policy.record()
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="process-fixed"):
        prepare(root, path.name, initializer=FlashNextEngine._initialize, fixed_options=fixed)
    del raw["options"]["retention_policy"]
    raw["options"]["output_queue_tokens"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="process-fixed"):
        prepare(root, path.name, initializer=FlashNextEngine._initialize, fixed_options=fixed)


@pytest.mark.parametrize("tier", ["gpu", "ram"])
def test_profile_preparation_cli_carries_actual_fifo_and_typed_retention_settings(tmp_path, monkeypatch, capsys, tier):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.retention_policy import parse_retention_policy
    from tensorfold.families.qwen4_exp.cuda.serving_profiles import prepare
    from test_serving_profile_manifest import profile_files
    from tools import prepare_serving_profile

    root, _path, raw = profile_files(tmp_path)
    options = root / "options.json"
    options.write_text(json.dumps(raw["options"]))
    output = root / "prepared.json"
    command = ["prepare_serving_profile.py", "--model", str(root / "target"), "--id", "typed-policy",
               "--options", str(options), "--out", str(output), "--tp", "8", "--adaptive-sessions",
               "--max-waiting-input-bytes", "4096", "--max-waiting-input-bytes-per-principal", "8192",
               "--output-queue-tokens", "1",
               "--queue-timeout-s", ".5", "--retention-tier", tier,
               "--retained-sessions", "2", "--retention-lifetime-s", "60"]
    values = {"retention_tier": tier, "retained_sessions": 2, "retention_lifetime_s": 60.0}
    if tier == "ram":
        command.extend(["--retention-ram-bytes", "1073741824", "--retention-chunk-bytes", "4096",
                        "--retention-progress-bytes", "65536", "--retention-overlap-bytes", "0"])
        values.update(retention_ram_bytes=1073741824, retention_chunk_bytes=4096,
                      retention_progress_bytes=65536, retention_overlap_bytes=0)
    monkeypatch.setattr("sys.argv", command)
    prepare_serving_profile.main()
    published = json.loads(capsys.readouterr().out)
    policy = parse_retention_policy(**values)
    profile = prepare(root, output.name, initializer=FlashNextEngine._initialize,
        fixed_options={"tp": 8, "rank": 0, "adaptive_sessions": True, "indexer_history": "gpu",
                       "max_waiting_input_bytes": 4096, "max_waiting_input_bytes_per_principal": 8192,
                       "output_queue_tokens": 1,
                       "queue_timeout_s": .5, "retention_policy": policy})
    assert published["identity"] == profile.identity and profile.options["retention_policy"] is policy
    assert profile.options["max_waiting_input_bytes_per_principal"] == 8192
    assert profile.options["output_queue_tokens"] == 1
    serialized = json.loads(output.read_text())["options"]
    assert not {"max_waiting_input_bytes", "max_waiting_input_bytes_per_principal", "output_queue_tokens",
                "queue_timeout_s", "retention_policy"}.intersection(serialized)
