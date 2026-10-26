"""Admission and retained-service options are resolved before model acquisition."""

import pytest

from tensorfold import cli
from tensorfold.server import authentication


def test_principal_request_quota_requires_authentication_before_model_metadata(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_config_dir", lambda *_a, **_k: pytest.fail("unauthenticated quota read model metadata"))
    args = cli.build_parser().parse_args(["serve", "unresolved-model", "--no-update-check",
                                          "--max-pending-requests", "4",
                                          "--max-pending-requests-per-principal", "2"])
    with pytest.raises(ValueError, match="requires configured API keys before model loading"):
        cli.cmd_serve(args)


@pytest.mark.parametrize("source", ["argument", "environment", "file"])
def test_principal_quota_reuses_one_consumed_authentication_store(tmp_path, monkeypatch, source):
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
                                          "--max-pending-requests-per-principal", "2", *flags])
    with pytest.raises(RuntimeError, match="configured metadata boundary"):
        cli.cmd_serve(args)
    assert len(stores) == 1 and stores[0].enabled and args.auth is stores[0]
    assert args.api_key == [] and "TENSORFOLD_API_KEY" not in cli.os.environ
