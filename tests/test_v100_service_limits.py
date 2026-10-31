"""Explicit endpoint limits fail before model loading or listener publication."""

from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace as NS

import pytest

from tensorfold.cli import build_parser
from tensorfold.serve_options import check
from tensorfold.server.http import Server
from tensorfold.server.request_limits import require_finite_endpoint_limits


def _options(tmp_path, *flags):
    checkpoint = tmp_path / "target.gguf"
    checkpoint.write_bytes(b"GGUF")  # Resolver-only local file; no tensor/header parser is invoked.
    return build_parser().parse_args(["serve", str(checkpoint), "--backend", "cuda", "--no-drafts", *flags])


def _family(model_type="qwen4_exp"):
    return NS(model_type=model_type, title="controlled family",
              package=NS(CUDA_KV_DTYPES=("bf16", "fp16", "int8")))


@pytest.mark.parametrize("flags,missing", [([], "--max-pending-requests"),
    (["--max-pending-requests", "3"], "--max-http-connections"),
    (["--max-http-connections", "4"], "--max-pending-requests")])
def test_adaptive_gguf_endpoint_requires_both_explicit_caps(tmp_path, flags, missing):
    options = _options(tmp_path, *flags)
    with pytest.raises(ValueError, match=missing):
        check(options, _family(), "cuda", tmp_path)


def test_finite_explicit_policy_preserves_values_without_inferred_defaults(tmp_path):
    options = _options(tmp_path, "--max-pending-requests", "3", "--max-http-connections", "4")
    check(options, _family(), "cuda", tmp_path)
    assert options.max_pending_requests == 3 and options.max_http_connections == 4


def test_cli_refuses_missing_policy_before_payload_resolution_or_model_loading(tmp_path, monkeypatch):
    from tensorfold import cli, families, hub

    options = _options(tmp_path, "--no-update-check")
    family = _family()
    family.package.cuda_engine = lambda *args, **kwargs: pytest.fail("missing policy loaded model weights")
    monkeypatch.setattr(cli, "_config_dir", lambda model: tmp_path / "target.gguf")
    monkeypatch.setattr(families, "detect", lambda path: family)
    monkeypatch.setattr(hub, "resolve", lambda *args, **kwargs: pytest.fail("missing policy resolved payload files"))
    with pytest.raises(ValueError, match="explicit positive"):
        cli.cmd_serve(options)


def test_other_provider_and_legacy_ram_mode_keep_optional_policy(tmp_path):
    options = _options(tmp_path)
    check(options, _family("other"), "cuda", tmp_path)
    options.indexer_history = "ram"
    check(options, _family(), "cuda", tmp_path)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_finite_policy_never_coerces_invalid_count_types(value):
    with pytest.raises(ValueError, match="positive integer"):
        require_finite_endpoint_limits(max_pending_requests=value, max_http_connections=3)


def test_programmatic_marked_handler_cannot_publish_an_unbounded_endpoint(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        requires_finite_limits = True
        pending_request_limit = 3
    monkeypatch.setattr("socket.socket", lambda *args, **kwargs: pytest.fail("unbounded endpoint created a socket"))
    with pytest.raises(ValueError, match="--max-http-connections"):
        Server(("127.0.0.1", 0), Handler)


def test_finite_programmatic_endpoint_and_legacy_default_remain_supported():
    class Handler(BaseHTTPRequestHandler):
        requires_finite_limits = True
        pending_request_limit = 3
    server = Server(("127.0.0.1", 0), Handler, max_connections=4)
    server.server_close()
    legacy = Server(("127.0.0.1", 0), BaseHTTPRequestHandler)
    legacy.server_close()
