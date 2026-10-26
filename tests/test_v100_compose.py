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


def render(compose, tmp_path, *, waiting=None, method="mtp"):
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("TF_", "TENSORFOLD_")) and name not in
           ("MODEL_ROOT", "GGUF_FILE", "MTP_DIRECTORY", "DRAFTER_DIRECTORY")}
    env.update(MODEL_ROOT=str(tmp_path / "model files"), GGUF_FILE="/models/first shard; literal $(not-a-command).gguf",
               TENSORFOLD_V100_IMAGE="tensorfold-v100:controlled-compose-fixture",
               TF_MAX_PENDING_REQUESTS="3", TF_MAX_HTTP_CONNECTIONS="4", TF_DRAFT_METHOD=method)
    if waiting is not None:
        env["TF_MAX_WAITING_INPUT_BYTES"] = waiting
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


def actual_argv(service):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("V100 Compose entrypoint requires the declared Linux Bash host capability")
    entry = service["entrypoint"]
    assert entry[:2] == ["/bin/bash", "-c"] and len(entry) == 4
    # Capture only the exec boundary. The actual rendered Bash array/quoting
    # code runs unchanged, without starting a model or an alternate interpreter.
    script = "exec() { printf '%s\\0' \"$@\"; }\n" + entry[2]
    env = {name: value for name, value in os.environ.items()
           if name not in ("TF_MAX_WAITING_INPUT_BYTES", "BASH_ENV")}
    env.update(service["environment"])
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", script, entry[3], *service["command"]],
                            env=env, capture_output=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    return [value.decode() for value in result.stdout.split(b"\0")[:-1]]


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


def test_environment_metacharacters_remain_one_literal_argument(compose, tmp_path):
    witness = tmp_path / "must not be created"
    value = f"8192; touch '{witness}' $(touch '{witness}')"
    service = render(compose, tmp_path, waiting=value)
    assert actual_argv(service) == ["/opt/venv/bin/python", *service["command"], "--max-waiting-input-bytes", value]
    assert not witness.exists()
