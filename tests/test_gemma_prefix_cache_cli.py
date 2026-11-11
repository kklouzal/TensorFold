"""Shared-prefix byte configuration routing; no weights, CUDA or model execution."""
import sys
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli, families
from tensorfold.families import gemma4
from test_gemma_prepared_global_prompt_cli import raw, family
from tests.cli_server_fixture import cuda_cli_app, cuda_cli_engine, install_cli_server


def test_parser_byte_option_default_zero_and_help(capsys):
    parser = cli.build_parser()
    assert parser.parse_args(["serve", "owner/model"]).prefix_cache_bytes is None
    assert parser.parse_args(["serve", "owner/model", "--prefix-cache-bytes", "0"]).prefix_cache_bytes == 0
    assert parser.parse_args(["serve", "owner/model", "--prefix-cache-bytes", "123456"]).prefix_cache_bytes == 123456
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(["serve", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert "--prefix-cache-bytes" in text and "0 disables" in text and "no entry ceiling" in text


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "12", sys.maxsize + 1])
def test_bad_byte_counts_refuse_before_metadata(monkeypatch, tmp_path, value):
    monkeypatch.setattr(families, "read_config", lambda *_: pytest.fail("invalid bytes reached metadata"))
    args = cli.build_parser().parse_args(["serve", str(tmp_path)])
    args.prefix_cache_bytes = value
    with pytest.raises(ValueError, match="nonnegative native byte count"):
        cli._check_serve_options(args, family(), "cuda", tmp_path)
    with pytest.raises(ValueError, match="nonnegative native byte count"):
        gemma4.cuda_engine(tmp_path, prefix_cache_bytes=value)


@pytest.mark.parametrize("backend,capable", [("mlx", True), ("cuda", False)])
def test_unsupported_backend_or_family_never_ignores_explicit_zero(monkeypatch, tmp_path, backend, capable):
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--prefix-cache-bytes", "0"])
    found = NS(title="other", model_type="other", package=NS(CUDA_PREFIX_CACHE_BYTES=capable))
    monkeypatch.setattr(families, "read_config", lambda *_: pytest.fail("unsupported option read metadata"))
    with pytest.raises(ValueError, match="Gemma 4 MoE on CUDA"):
        cli._check_serve_options(args, found, backend, tmp_path)
    if backend == "mlx":
        with pytest.raises(ValueError, match="Gemma 4 MoE on CUDA"):
            gemma4.load(tmp_path, prefix_cache_bytes=0)


@pytest.mark.parametrize("metadata", [raw(text_config={"enable_moe_block": False}),
                                      raw(text_config={"enable_moe_block": 1}), raw(model_type="other")])
def test_non_moe_metadata_is_refused_before_engine(monkeypatch, tmp_path, metadata):
    monkeypatch.setattr(families, "read_config", lambda *_: metadata)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--prefix-cache-bytes", "1024"])
    with pytest.raises(ValueError, match="Gemma 4 MoE on CUDA"):
        cli._check_serve_options(args, family(), "cuda", tmp_path)
    with pytest.raises(ValueError, match="Gemma 4 MoE on CUDA"):
        gemma4.cuda_engine(tmp_path, prefix_cache_bytes=1024)


@pytest.mark.parametrize("selected", [None, 0, 123456])
def test_family_forwards_exact_bytes_without_changing_serving_arguments(monkeypatch, tmp_path, selected):
    from tensorfold.cuda import precision
    from tensorfold.families.gemma4.cuda import engine
    monkeypatch.setattr(families, "read_config", lambda *_: raw())
    monkeypatch.setattr(precision, "mode", lambda: precision.CHECKPOINT)
    seen = []
    monkeypatch.setattr(engine, "Gemma4Engine", lambda *args, **kwargs: seen.append(kwargs) or "loaded")
    options = {} if selected is None else {"prefix_cache_bytes": selected}
    assert gemma4.cuda_engine(tmp_path, drafter="owned/draft", drafter_bits=8, draft_rows=5,
        context=262144, context_explicit=True, vision=True, chat_template_file="owned-template.jinja",
        **options) == "loaded"
    assert ("prefix_cache_bytes" in seen[0]) is (selected is not None)
    if selected is not None:
        assert seen[0]["prefix_cache_bytes"] == selected
    assert seen[0]["depth"] == 4 and seen[0]["drafter_bits"] == 8
    assert seen[0]["context"] == 262144 and seen[0]["vision"] is True
    assert seen[0]["chat_template_file"] == "owned-template.jinja"
    assert "prefix_cache_entries" not in seen[0]


@pytest.mark.parametrize("selected", [None, 0, 123456])
def test_cuda_service_forwards_explicit_byte_control(monkeypatch, tmp_path, request, selected):
    import tensorfold.cuda.server as server
    from tensorfold.cuda import prompt_precision
    seen = []
    def engine(*args, **kwargs):
        seen.append(kwargs)
        return cuda_cli_engine(max_len=8192, w=NS(fast_prefill=False))
    found = NS(title="fixture", model_type="fixture",
               package=NS(cuda_engine=engine, CUDA_PREFIX_CACHE_BYTES=True))
    monkeypatch.setattr(server, "App", lambda *args, **kwargs: cuda_cli_app(effective_context_window=8192))
    install_cli_server(monkeypatch, request, server)
    options = [] if selected is None else ["--prefix-cache-bytes", str(selected)]
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts", *options])
    try:
        assert cli._serve_cuda(args, found, tmp_path, 8192) == 0
    finally:
        prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)
    assert len(seen) == 1 and ("prefix_cache_bytes" in seen[0]) is (selected is not None)
    if selected is not None:
        assert seen[0]["prefix_cache_bytes"] == selected
