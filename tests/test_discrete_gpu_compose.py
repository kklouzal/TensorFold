"""Render actual Compose merges; these checks need the declared host Compose tool."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "discrete-gpu"
UUID = "GPU-00000000-0000-0000-0000-000000000001"
SECOND = "GPU-00000000-0000-0000-0000-000000000002"


@pytest.fixture(scope="module")
def compose():
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("requires host Docker Compose >=2.30")
    version = subprocess.run([docker, "compose", "version", "--short"], capture_output=True,
                             text=True, timeout=30, check=False)
    match = re.search(r"(\d+)\.(\d+)", version.stdout)
    if version.returncode or match is None or tuple(map(int, match.groups())) < (2, 30):
        pytest.skip("requires host Docker Compose >=2.30")
    return [docker, "compose"]


@pytest.fixture
def rendered(compose, tmp_path):
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("TF_") and key != "TENSORFOLD_IMAGE"}
    environment.update(TENSORFOLD_IMAGE="tensorfold-fork:local", TF_MODEL_DIR=str(tmp_path / "model cache"),
                       TF_CACHE_DIR=str(tmp_path / "cache"), TF_TMP_DIR=str(tmp_path / "tmp"),
                       TF_WORK_DIR=str(tmp_path / "work"))

    def render(*overrides, variables=None, succeeds=True):
        command = [*compose, "--env-file", str(empty_env), "-f", str(DEPLOY / "compose.yaml")]
        for override in overrides:
            command.extend(["-f", str(override if isinstance(override, Path) else DEPLOY / override)])
        result = subprocess.run([*command, "config", "--format", "json"],
                                env=environment | (variables or {}), capture_output=True, text=True,
                                timeout=30, check=False)
        if not succeeds:
            assert result.returncode != 0
            return result.stderr
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)["services"]["tensorfold"]

    return render


def test_base_profile_preserves_all_gpu_exl3_defaults_and_no_extra_swap(rendered):
    service = rendered()
    assert len(service["gpus"]) == 1 and service["gpus"][0]["count"] == -1
    assert service["mem_limit"] == service["memswap_limit"]
    assert int(service["mem_limit"]) == 56_000_000_000
    assert service["restart"] == "unless-stopped"
    assert service["command"][:2] == ["serve", "/model"]
    assert service["read_only"] is True and service["user"] == "1000:1000"
    assert all(volume["bind"]["create_host_path"] is False for volume in service["volumes"])


@pytest.mark.parametrize("policy", ["no", "on-failure:3"])
def test_checkpoint_budget_and_docker_retry_policy_are_configurable(rendered, policy):
    service = rendered(variables={"TF_RAM_BYTES": "96000000000", "TF_RESTART_POLICY": policy})
    assert service["mem_limit"] == service["memswap_limit"]
    assert int(service["mem_limit"]) == 96_000_000_000
    assert service["restart"] == policy


def test_uuid_override_replaces_all_gpu_request(rendered):
    service = rendered("compose.gpus.yaml", variables={"TF_GPU_UUID": UUID})
    assert len(service["gpus"]) == 1
    assert service["gpus"][0]["driver"] == "nvidia"
    assert service["gpus"][0]["device_ids"] == [UUID]
    assert service["gpus"][0].get("count", 0) != -1
    assert not service.get("devices")


def test_cdi_override_removes_legacy_request_and_inherited_all_injection(rendered):
    service = rendered("compose.cdi.yaml", variables={"TF_CDI_DEVICE": "nvidia.com/gpu=" + UUID})
    assert not service.get("gpus")
    assert [device["source"] for device in service["devices"]] == ["nvidia.com/gpu=" + UUID]
    assert service["environment"]["NVIDIA_VISIBLE_DEVICES"] == "void"


@pytest.mark.parametrize("filename,variable", [("compose.gpus.yaml", "TF_GPU_UUID"),
                                               ("compose.cdi.yaml", "TF_CDI_DEVICE")])
def test_explicit_device_override_requires_operator_selection(rendered, filename, variable):
    assert variable in rendered(filename, succeeds=False)


@pytest.mark.parametrize("kind", ["gpus", "cdi"])
def test_explicit_multi_gpu_lists_replace_single_selection_without_duplicates(rendered, tmp_path, kind):
    override = tmp_path / "multiple.yaml"
    if kind == "gpus":
        override.write_text("services:\n  tensorfold:\n    gpus: !override\n      - driver: nvidia\n"
                            f"        device_ids: [{UUID}, {SECOND}]\n")
        service = rendered("compose.gpus.yaml", override, variables={"TF_GPU_UUID": UUID})
        assert len(service["gpus"]) == 1 and service["gpus"][0]["device_ids"] == [UUID, SECOND]
    else:
        override.write_text("services:\n  tensorfold:\n    devices: !override\n"
                            f"      - nvidia.com/gpu={UUID}\n      - nvidia.com/gpu={SECOND}\n")
        service = rendered("compose.cdi.yaml", override, variables={"TF_CDI_DEVICE": "nvidia.com/gpu=" + UUID})
        assert not service.get("gpus")
        assert [device["source"] for device in service["devices"]] == [
            "nvidia.com/gpu=" + UUID, "nvidia.com/gpu=" + SECOND]
