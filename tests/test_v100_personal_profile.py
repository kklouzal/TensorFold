"""Personal deployment policy, exercised as Bash; no model/hardware qualification."""
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy/v100/personal_profile.sh"
FIELDS = ("TF_PARALLEL", "TF_KV_FULL_WINDOWS", "TF_MAX_PENDING_REQUESTS", "TF_MAX_HTTP_CONNECTIONS",
          "TF_MAX_WAITING_INPUT_BYTES", "TF_CONTEXT", "TF_DRAFTS", "TF_V100_PERSONAL_CAP")


def resolve(**values):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TF_")}
    environment.update(values)
    output = " ".join('"${' + name + '-}"' for name in FIELDS)
    command = 'source "$1"; tf_v100_personal_profile || exit $?; printf "%s\\0" ' + output
    result = subprocess.run(["bash", "--noprofile", "--norc", "-c", command, "profile-test", str(SCRIPT)],
                            env=environment, capture_output=True, timeout=10)
    fields = dict(zip(FIELDS, (v.decode() for v in result.stdout.split(b"\0")[:-1])))
    return result, fields


@pytest.mark.parametrize("codec,cap", [("int8", 24), ("rotorquant-iso4-sm70-v1", 32)])
def test_auto_defaults_are_bounded_and_full_context(codec, cap):
    result, fields = resolve(TF_KV_DTYPE=codec, TF_PARALLEL="auto", TF_KV_FULL_WINDOWS="auto")
    assert result.returncode == 0, result.stderr
    assert fields == dict(zip(FIELDS, map(str, (cap, cap, cap, cap + 1, cap * 9437212, 262144, 4, cap))))


def test_lower_counts_and_explicit_queued_pending_are_preserved():
    result, fields = resolve(TF_KV_DTYPE="int8", TF_PARALLEL="00012", TF_KV_FULL_WINDOWS="8",
                            TF_MAX_PENDING_REQUESTS="16", TF_MAX_HTTP_CONNECTIONS="99",
                            TF_MAX_WAITING_INPUT_BYTES="17")
    assert result.returncode == 0, result.stderr
    assert [fields[key] for key in FIELDS[:5]] == ["12", "8", "16", "99", "17"]


def test_lower_effective_default_couples_pending_monitor_and_input_credit():
    result, fields = resolve(TF_KV_DTYPE="int8", TF_PARALLEL="12", TF_KV_FULL_WINDOWS="8")
    assert result.returncode == 0, result.stderr
    assert [fields[key] for key in FIELDS[2:5]] == ["8", "9", str(8 * 9437212)]


@pytest.mark.parametrize("key", ["TF_PARALLEL", "TF_KV_FULL_WINDOWS", "TF_MAX_PENDING_REQUESTS"])
@pytest.mark.parametrize("value", ["25", "999999999999999999999999999999999999"])
def test_explicit_above_q8_cap_refuses_without_overflow_or_exec(key, value):
    result, _ = resolve(TF_KV_DTYPE="int8", **{key: value})
    assert result.returncode == 2 and b"personal V100 ceiling 24" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "auto", "3; exit 77", "$(exit 77)"])
def test_invalid_explicit_pending_refuses_as_data(value):
    result, _ = resolve(TF_KV_DTYPE="int8", TF_MAX_PENDING_REQUESTS=value)
    assert result.returncode == 2 and b"positive decimal integer" in result.stderr


@pytest.mark.parametrize("changes", [{"TF_CONTEXT": "8192"}, {"TF_DRAFTS": "6"},
                                    {"TF_DRAFT_METHOD": "dspark"}, {"TF_INDEXER_HISTORY": "ram"}])
def test_other_geometry_or_method_gets_no_invented_input_credit(changes):
    result, fields = resolve(TF_KV_DTYPE="int8", **changes)
    assert result.returncode == 0, result.stderr
    assert fields["TF_MAX_WAITING_INPUT_BYTES"] == ""


@pytest.mark.parametrize("codec", ["fp16", "rotorquant-signed128-4", "unknown-codec"])
def test_other_codecs_are_not_reinterpreted(codec):
    values = dict(TF_KV_DTYPE=codec, TF_PARALLEL="auto", TF_KV_FULL_WINDOWS="77",
                  TF_MAX_PENDING_REQUESTS="40", TF_MAX_HTTP_CONNECTIONS="50", TF_CONTEXT="8192")
    result, fields = resolve(**values)
    assert result.returncode == 0, result.stderr
    assert all(fields[key] == values[key] for key in values if key in fields)
    assert fields["TF_V100_PERSONAL_CAP"] == ""


@pytest.mark.parametrize("name,cap", [("personal-q8.env.example", 24), ("personal-q4.env.example", 32)])
def test_persistent_examples_resolve_to_the_declared_profiles(name, cap):
    values = dict(line.split("=", 1) for line in (SCRIPT.parent / name).read_text().splitlines()
                  if line and not line.startswith("#"))
    result, fields = resolve(**values)
    assert result.returncode == 0, result.stderr
    assert fields["TF_V100_PERSONAL_CAP"] == str(cap)
    assert fields["TF_MAX_WAITING_INPUT_BYTES"] == str(cap * 9437212)
