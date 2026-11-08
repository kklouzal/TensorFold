"""Actual V100 Compose argv adaptation; no GPU or model qualification."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test_discrete_gpu_compose import compose as _compose

compose = _compose


DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "v100"


def render(compose, tmp_path, *, waiting=None, method="mtp", etp=None, drafts=None, prefill=None, variables=None):
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("TF_", "TENSORFOLD_")) and name not in
           ("MODEL_ROOT", "GGUF_FILE", "MTP_DIRECTORY", "DRAFTER_DIRECTORY")}
    env.update(MODEL_ROOT=str(tmp_path / "model files"), GGUF_FILE="/models/first shard; literal $(not-a-command).gguf",
               TENSORFOLD_V100_IMAGE="tensorfold-v100:controlled-compose-fixture",
               TF_MAX_PENDING_REQUESTS="3", TF_MAX_HTTP_CONNECTIONS="4", TF_DRAFT_METHOD=method)
    if drafts is not None:
        env["TF_DRAFTS"] = drafts
    if prefill is not None:
        env["TF_PREFILL_ROWS"] = prefill
    if waiting is not None:
        env["TF_MAX_WAITING_INPUT_BYTES"] = waiting
    if etp is not None:
        env["TF_EXPERT_TENSOR_PARALLEL_SIZE"] = etp
    for name, value in (variables or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    empty = tmp_path / "empty.env"
    empty.write_text("")
    result = subprocess.run([*compose, "--env-file", str(empty), "-f", str(DEPLOY / "compose.yaml"),
                             "config", "--format", "json"], env=env, capture_output=True, text=True,
                            timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    # Compose config re-escapes every dollar for round-trip serialization
    # (Docker Compose cmd/compose/config.go, runConfig). Decode that display
    # layer once before executing the actual runtime command/entrypoint.
    return json.loads(result.stdout.replace("$$", "$"))["services"]["tensorfold"]


def actual_argv(service, *, succeeds=True):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("V100 Compose entrypoint requires the declared Linux Bash host capability")
    entry = service["entrypoint"]
    assert entry[:2] == ["/bin/bash", "-c"] and len(entry) == 4
    # Capture only the exec boundary. The actual rendered Bash array/quoting
    # code runs unchanged, without starting a model or an alternate interpreter.
    script = ("source() { [[ $1 == /run/tensorfold-v100/personal_profile.sh ]] || return 2; "
              "builtin source \"$TF_TEST_PROFILE_HELPER\"; }\n"
              "exec() { printf '%s\\0' \"$@\"; }\n" + entry[2])
    env = {name: value for name, value in os.environ.items()
           if name not in ("TF_MAX_WAITING_INPUT_BYTES", "TF_EXPERT_TENSOR_PARALLEL_SIZE", "BASH_ENV")}
    env.update(service["environment"])
    env["TF_TEST_PROFILE_HELPER"] = str(DEPLOY / "personal_profile.sh")
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", script, entry[3], *service["command"]],
                            env=env, capture_output=True, timeout=10, check=False)
    if not succeeds:
        assert result.returncode != 0
        return result
    assert result.returncode == 0, result.stderr
    return [value.decode() for value in result.stdout.split(b"\0")[:-1]]


@pytest.mark.parametrize("codec,cap,waiting", [
    ("int8", 24, 226493088), ("rotorquant-iso4-sm70-v1", 32, 301990784)])
def test_personal_codecs_reach_actual_cli_with_finite_admin_ceilings(compose, tmp_path, codec, cap, waiting):
    from tensorfold.cli import build_parser
    service = render(compose, tmp_path, variables={
        "TF_KV_DTYPE": codec, "TF_MAX_PENDING_REQUESTS": None, "TF_MAX_HTTP_CONNECTIONS": None})
    argv = actual_argv(service)
    parsed = build_parser().parse_args(["serve", *argv[argv.index("--") + 1:]])
    assert str(parsed.parallel) == str(cap) and str(parsed.kv_full_windows) == str(cap)
    assert parsed.context == 262144 and parsed.mtp_drafts == 4
    assert parsed.max_pending_requests == cap and parsed.max_http_connections == cap + 1
    assert parsed.max_waiting_input_bytes == waiting
    mounted = [v for v in service["volumes"] if v["target"] == "/run/tensorfold-v100/personal_profile.sh"]
    assert len(mounted) == 1 and mounted[0]["read_only"] is True


def test_personal_profile_keeps_explicit_lower_parallel_and_queued_pending(compose, tmp_path):
    argv = actual_argv(render(compose, tmp_path, variables={"TF_KV_DTYPE": "int8",
        "TF_PARALLEL": "12", "TF_KV_FULL_WINDOWS": "8", "TF_MAX_PENDING_REQUESTS": "16",
        "TF_MAX_HTTP_CONNECTIONS": "19"}))
    assert argv[argv.index("--parallel") + 1] == "12"
    assert argv[argv.index("--kv-full-windows") + 1] == "8"
    assert argv[argv.index("--max-pending-requests") + 1] == "16"
    assert argv[argv.index("--max-http-connections") + 1] == "19"
    assert argv[argv.index("--max-waiting-input-bytes") + 1] == str(16 * 9437212)


def test_explicit_above_personal_cap_never_reaches_python_exec(compose, tmp_path):
    service = render(compose, tmp_path, variables={"TF_KV_DTYPE": "int8", "TF_PARALLEL": "25"})
    rejected = actual_argv(service, succeeds=False)
    assert rejected.returncode == 2 and b"personal V100 ceiling 24" in rejected.stderr
    assert rejected.stdout == b""


def test_unknown_codec_still_requires_explicit_request_limits_before_exec(compose, tmp_path):
    service = render(compose, tmp_path, variables={"TF_MAX_PENDING_REQUESTS": None})
    rejected = actual_argv(service, succeeds=False)
    assert b"Set TF_MAX_PENDING_REQUESTS" in rejected.stderr and rejected.stdout == b""


def test_personal_ram_history_keeps_legacy_argv_without_fifo_credit(compose, tmp_path):
    service = render(compose, tmp_path, variables={"TF_KV_DTYPE": "int8", "TF_CONTEXT": "262144",
        "TF_PARALLEL": "2", "TF_KV_FULL_WINDOWS": "2", "TF_INDEXER_HISTORY": "ram"})
    assert service["environment"]["TF_INDEXER_HISTORY"] == "ram"
    assert actual_argv(service) == ["/opt/venv/bin/python", *service["command"]]
    assert "--max-waiting-input-bytes" not in actual_argv(service)


@pytest.mark.parametrize("waiting", [None, ""])
def test_legacy_dspark_omits_fifo_option_and_preserves_exact_base_argv(compose, tmp_path, waiting):
    service = render(compose, tmp_path, waiting=waiting, method="dspark")
    argv = actual_argv(service)
    assert argv == ["/opt/venv/bin/python", *service["command"]]
    assert "--max-waiting-input-bytes" not in argv
    assert service["command"][service["command"].index("--draft-method") + 1] == "dspark"


def test_fifo_uses_only_the_explicit_waiting_budget(compose, tmp_path):
    service = render(compose, tmp_path, waiting="8192")
    expected = ["/opt/venv/bin/python", *service["command"], "--max-waiting-input-bytes", "8192"]
    assert actual_argv(service) == expected
    assert service["command"][service["command"].index("--") + 1] == "/models/first shard; literal $(not-a-command).gguf"
    # Compiler and frontend state stay under the same persistent writable
    # named volume rather than the image user's read-only HOME.
    caches = (service["environment"]["TRITON_CACHE_DIR"], service["environment"]["TF_GGUF_ASSETS_DIR"])
    owners = [mount for mount in service["volumes"] if all(Path(cache).is_relative_to(mount["target"]) for cache in caches)]
    assert len(owners) == 1
    assert owners[0]["target"] == "/cache" and owners[0]["type"] == "volume"
    assert owners[0]["source"] == "tensorfold-v100-cache" and not owners[0].get("read_only", False)
    assert caches[0] != caches[1] and all(Path(cache).is_absolute() for cache in caches)


def test_environment_metacharacters_remain_one_literal_argument(compose, tmp_path):
    witness = tmp_path / "must not be created"
    value = f"8192; touch '{witness}' $(touch '{witness}')"
    service = render(compose, tmp_path, waiting=value)
    assert actual_argv(service) == ["/opt/venv/bin/python", *service["command"], "--max-waiting-input-bytes", value]
    assert not witness.exists()


@pytest.mark.parametrize("etp", [None, ""])
def test_unspecified_expert_degree_preserves_legacy_argv(compose, tmp_path, etp):
    service = render(compose, tmp_path, method="dspark", etp=etp)
    assert actual_argv(service) == ["/opt/venv/bin/python", *service["command"]]
    assert "--expert-tensor-parallel-size" not in actual_argv(service)


@pytest.mark.parametrize("etp", ["8", "4", "2"])
def test_explicit_expert_degree_reaches_only_the_server_argv(compose, tmp_path, etp):
    service = render(compose, tmp_path, waiting="8192", etp=etp)
    argv = actual_argv(service)
    assert argv == ["/opt/venv/bin/python", *service["command"],
                    "--max-waiting-input-bytes", "8192", "--expert-tensor-parallel-size", etp]
    assert argv.count("--expert-tensor-parallel-size") == 1
    # The launcher consumes only its leading GPU arguments; the degree belongs
    # after its separator, where the existing CLI validates the model profile.
    assert argv.index("--expert-tensor-parallel-size") > argv.index("--")


def test_expert_degree_metacharacters_are_literal_and_rejected_by_cli(compose, tmp_path, capsys):
    from tensorfold.cli import build_parser

    witness = tmp_path / "must not be created"
    value = f"4; touch '{witness}' $(touch '{witness}')"
    service = render(compose, tmp_path, etp=value)
    argv = actual_argv(service)
    assert argv[-2:] == ["--expert-tensor-parallel-size", value]
    with pytest.raises(SystemExit) as rejected:
        build_parser().parse_args(["serve", *argv[argv.index("--") + 1:]])
    assert rejected.value.code == 2
    assert "argument --expert-tensor-parallel-size: invalid int value" in capsys.readouterr().err
    assert not witness.exists()


@pytest.mark.parametrize("drafts,expected", [(None, 4), ("", 4), ("0", 0), ("2", 2), ("6", 6)])
def test_mtp_default_four_and_explicit_overrides_reach_actual_argv(compose, tmp_path, drafts, expected):
    from tensorfold.cli import build_parser

    argv = actual_argv(render(compose, tmp_path, drafts=drafts))
    parsed = build_parser().parse_args(["serve", *argv[argv.index("--") + 1:]])
    assert parsed.mtp_drafts == expected
    assert argv[argv.index("--mtp-drafts") + 1] == str(expected)


@pytest.mark.parametrize("prefill,expected", [(None, None), ("", None), ("256", 256), ("2048", 2048)])
def test_prefill_rows_environment_reaches_the_existing_backend_parser(compose, tmp_path, monkeypatch, prefill, expected):
    from tensorfold.cuda.geometry import indexed_prefill_rows

    service = render(compose, tmp_path, prefill=prefill)
    forwarded = service["environment"]["TENSORFOLD_PREFILL_ROWS"]
    assert forwarded == ("" if prefill is None else prefill)
    assert actual_argv(service) == ["/opt/venv/bin/python", *service["command"]]
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", forwarded)
    assert indexed_prefill_rows() == expected


def test_prefill_rows_metacharacters_stay_literal_and_backend_refuses(compose, tmp_path, monkeypatch):
    from tensorfold.cuda.geometry import indexed_prefill_rows

    witness = tmp_path / "must not be created"
    value = f"256; touch '{witness}' $(touch '{witness}')"
    service = render(compose, tmp_path, prefill=value)
    assert service["environment"]["TENSORFOLD_PREFILL_ROWS"] == value
    assert actual_argv(service) == ["/opt/venv/bin/python", *service["command"]]
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", service["environment"]["TENSORFOLD_PREFILL_ROWS"])
    with pytest.raises(ValueError, match="TENSORFOLD_PREFILL_ROWS"):
        indexed_prefill_rows()
    assert not witness.exists()
