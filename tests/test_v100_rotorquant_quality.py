"""Frozen-authority, quality-gate and supervisor controls without any GPU/model."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v100_rotorquant_quality", ROOT / "tools/v100_rotorquant_quality.py")
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)
IMAGE = "sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def ordinary_profile(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "optimized")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("frozen-quality")
    directory = root / "baseline-vectors"
    directory.mkdir()
    raw = {
        "complete": True,
        "tokens": 714,
        "dtype": "<f4",
        "operator": quality.REFERENCE_OPERATOR,
        "image": IMAGE,
        "cases": [],
        "vectors": [],
    }
    lengths = [32] * 9 + [37] + [35] * 10 + [39]
    for index, count in enumerate(lengths):
        suite = "baseline325" if index < 10 else "extended389"
        raw["cases"].append(
            {
                "index": index,
                "name": f"case-{index}",
                "suite": suite,
                "input_ids": [0, 2, 3],
                "teacher_ids": [1] * count,
            }
        )
        for step in range(count):
            path = directory / f"{index:03d}-{step:04d}.npy"
            np.save(path, np.asarray([0, 1, -1, 0.5], dtype="<f4"), allow_pickle=False)
            raw["vectors"].append(
                {
                    "case_index": index,
                    "step": step,
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    path = root / "baseline-vectors-manifest.json"
    path.write_text(json.dumps(raw))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, raw


def test_fixed_authority_checks_all714_payloads_and_both_suites(corpus):
    path, digest, _ = corpus
    authority = quality.read_authority(path, digest)
    assert authority.vocab == 4 and len(authority.cases) == 21 and len(authority.vectors) == 714
    assert {name: sum(len(c.teacher_ids) for c in authority.cases if c.suite == name) for name in quality.COUNTS} == {
        "baseline325": 325,
        "extended389": 389,
    }
    with pytest.raises(TypeError):
        authority.vectors[0, 0] = None


@pytest.mark.parametrize(
    "change",
    [
        "incomplete",
        "case-count",
        "duplicate-case",
        "suite",
        "tokens",
        "bool-id",
        "context",
        "duplicate-vector",
        "filename",
        "sha",
    ],
)
def test_invalid_authority_structure_is_refused_before_payload_use(corpus, tmp_path, change):
    _, _, template = corpus
    raw = json.loads(json.dumps(template))
    if change == "incomplete":
        raw["complete"] = False
    elif change == "case-count":
        raw["cases"].pop()
    elif change == "duplicate-case":
        raw["cases"][1]["index"] = 0
    elif change == "suite":
        raw["cases"][0]["suite"] = []
    elif change == "tokens":
        raw["tokens"] = 714.0
    elif change == "bool-id":
        raw["cases"][0]["input_ids"][0] = True
    elif change == "context":
        raw["cases"][0]["input_ids"] = [0] * 8192
    elif change == "duplicate-vector":
        raw["vectors"][1] = raw["vectors"][0].copy()
    elif change == "filename":
        raw["vectors"][0]["file"] = "../000-0000.npy"
    elif change == "sha":
        raw["vectors"][0]["sha256"] = "not-a-checksum"
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        quality.read_authority(path, hashlib.sha256(path.read_bytes()).hexdigest(), verify_vectors=False)


@pytest.mark.parametrize(
    "array",
    [
        np.array([1.0], dtype="<f8"),
        np.array([1.0], dtype=">f4"),
        np.array([[1.0]], dtype="<f4"),
        np.array([np.nan], dtype="<f4"),
        np.array([{"object": 1}], dtype=object),
    ],
)
def test_vector_requires_finite_full_float32_and_never_unpickles(tmp_path, array):
    path = tmp_path / "000-0000.npy"
    np.save(path, array, allow_pickle=True)
    vector = quality.Vector(path, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        quality.load_vector(vector)


def test_vector_rechecks_checksum_shape_symlink_and_trailing_payload(tmp_path):
    path = tmp_path / "000-0000.npy"
    np.save(path, np.zeros(4, dtype="<f4"), allow_pickle=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="checksum"):
        quality.load_vector(quality.Vector(path, "0" * 64))
    with pytest.raises(ValueError, match="vocabulary"):
        quality.load_vector(quality.Vector(path, digest), vocab=5)
    link = tmp_path / "link.npy"
    link.symlink_to(path)
    with pytest.raises(OSError):
        quality.load_vector(quality.Vector(link, digest))
    path.write_bytes(path.read_bytes() + b"trailing")
    with pytest.raises(ValueError, match="vocabulary"):
        quality.load_vector(quality.Vector(path, hashlib.sha256(path.read_bytes()).hexdigest()))


def test_logit_metrics_match_independent_probability_oracle_and_minimum_id_ties():
    base = np.log(np.array([0.5, 0.25, 0.25])).astype("<f4")
    candidate = np.log(np.array([0.25, 0.5, 0.25])).astype("<f4")
    row = quality.compare_logits(base, candidate, 1)
    assert row["kl"] == pytest.approx(0.25 * math.log(2), abs=1e-7)
    assert row["nll_delta"] == pytest.approx(-math.log(2), abs=1e-7)
    tied = quality.compare_logits(np.zeros(4, dtype="<f4"), np.zeros(4, dtype="<f4"), 1)
    assert tied["base_top1"] == tied["candidate_top1"] == 0 and tied["kl"] == 0
    with pytest.raises(ValueError):
        quality.compare_logits(base, np.array([0, 1, np.inf], dtype="<f4"), 1)


def rows_for(authority):
    return [
        {
            "case_index": c.index,
            "step": step,
            "suite": c.suite,
            "teacher_id": token,
            "kl": 0.0,
            "nll_delta": 0.0,
            "base_top1": 1,
            "candidate_top1": 1,
        }
        for c in authority.cases
        for step, token in enumerate(c.teacher_ids)
    ]


def test_suite_gate_cannot_be_diluted_by_the_other_suite(corpus):
    authority = quality.read_authority(*corpus[:2], verify_vectors=False)
    rows = rows_for(authority)
    assert quality.summarize(authority, rows)["passed"]
    for row in rows:
        if row["suite"] == "baseline325":
            row["kl"] = 0.0002
    result = quality.summarize(authority, rows)
    assert result["mean_kl"] < 1e-4 and not result["passed"]
    assert not result["suites"]["baseline325"]["passed"] and result["suites"]["extended389"]["passed"]
    assert result["gate"] == {"mean_kl_max": 1e-4, "ppl_ratio_max": 1.001, "top1_agreement_min": 0.995}
    with pytest.raises(ValueError, match="all714"):
        quality.summarize(authority, rows[:-1])
    with pytest.raises(ValueError, match="identities"):
        quality.summarize(authority, [rows[0], *rows[:-1]])


def test_teacher_protocol_uses_frozen_tokens_not_candidate_argmax():
    case = quality.Case(0, "tokens", "baseline325", (0, 2), (3, 0, 2))
    consumed = []

    def advance(token):
        consumed.append(token)
        return np.array([0, 1, 0, 0], dtype="<f4")

    frames = list(quality.teacher_frames(case, np.array([0, 1, 0, 0], dtype="<f4"), advance))
    assert consumed == [3, 0] and [token for _, token, _ in frames] == [3, 0, 2]


@pytest.mark.parametrize("kcodec,vcodec", [(4, 0), (0, 4), (5, 0), (0, 5), (4, 5), (5, 4), (5, 5)])
def test_fused_native_proxy_preserves_call_and_qualifies_without_standalone_encoder(kcodec, vcodec):
    called, result = [], object()

    class Native:
        ordinary = object()

        def prep_q4(self, *args, **kwargs):
            called.append((args, kwargs))
            return result

    native = Native()
    observer = quality.NativeObserver()
    observer.phase, observer.case = "teacher", 7
    proxy = observer.family_proxy(native)
    assert proxy.ordinary is native.ordinary
    assert proxy.prep_q4("unchanged", kcodec, vcodec) is result
    assert called == [(("unchanged", kcodec, vcodec), {})]
    observer.returned("tensorfold.cuda.sm70.attention_native.attention", {"kcodec": kcodec, "vcodec": vcodec})
    coverage = quality.native_coverage(observer.counts, q4_required=True)
    assert coverage["passed"] and coverage["fused_prep_q4_calls"] == 1 and coverage["standalone_encode_calls"] == 0
    warm = {key.replace("teacher:", "warm:"): value for key, value in observer.counts.items()}
    assert not quality.native_coverage(warm, q4_required=True)["passed"]


def test_required_native_coverage_names_match_exact_sm70_candidates():
    from tensorfold.families.qwen4_exp.cuda.sm70_kv import ROTOR_Q4, ROTOR_Q4_NORM, COMPACT_Q4

    # This frozen 714-step quality tool observes the historical Signed128 and
    # Signed64-norm kernels. Compact Planar4 has separate codec/native gates;
    # adding its registered name cannot transfer this tool's native coverage.
    assert quality.Q4_FORMATS == frozenset((ROTOR_Q4, ROTOR_Q4_NORM))
    assert COMPACT_Q4 not in quality.Q4_FORMATS
    for name in (ROTOR_Q4, ROTOR_Q4_NORM):
        assert name in quality.Q4_FORMATS
        assert not quality.native_coverage({}, q4_required=name in quality.Q4_FORMATS)["passed"]


@pytest.mark.parametrize("codec", [0, 1, 2, 3, 6, 7])
def test_other_codec_flags_cannot_satisfy_q4_native_coverage(codec):
    observer = quality.NativeObserver()
    observer.phase = "teacher"
    observer.returned("tensorfold.cuda.sm70.family_native.prep_q4", {"kcodec": codec, "vcodec": 0})
    observer.returned("tensorfold.cuda.sm70.attention_native.attention", {"kcodec": codec, "vcodec": 0})
    assert not quality.native_coverage(observer.counts, q4_required=True)["passed"]


def cli(corpus, output):
    return [
        "/unused/model.gguf",
        "--baseline-manifest",
        str(corpus[0]),
        "--expected-manifest-sha",
        corpus[1],
        "--output-dir",
        str(output),
        "--image-identity",
        IMAGE,
    ]


@pytest.mark.parametrize("exit_code,receipts,expected", [(0, True, 0), (2, True, 2), (1, True, 1), (0, False, 1)])
def test_supervisor_preserves_failed_gates_and_simultaneous_rank_failures(
    monkeypatch, corpus, tmp_path, exit_code, receipts, expected
):
    output = tmp_path / "run"
    authority = quality.read_authority(*corpus[:2])
    monkeypatch.setattr(quality, "read_authority", lambda *args, **kwargs: authority)
    commands = []

    class Process:
        pid = 123456789
        returncode = exit_code

        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))
            rank = int(command[command.index("--rank") + 1])
            if receipts:
                quality.write_json(
                    output / f"rank-{rank}-complete.json",
                    {"complete": True, "steps": 714, "cases": 21, "prediction_frames": 714},
                )
                if rank == 0:
                    quality.write_json(output / "summary.json", {"complete": True, "passed": exit_code == 0})

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(quality.subprocess, "Popen", Process)
    assert quality.main(cli(corpus, output)) == expected
    assert len(commands) == 8 and all(item[1]["start_new_session"] for item in commands)
    assert len({item[1]["env"]["CUDA_VISIBLE_DEVICES"] for item in commands}) == 8
    supervisor = json.loads((output / "supervisor.json").read_text())
    assert supervisor["passed"] is (expected == 0)


def test_validate_only_never_starts_a_rank(monkeypatch, corpus, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("authority validation started a model rank")

    monkeypatch.setattr(quality.subprocess, "Popen", forbidden)
    output = tmp_path / "validation"
    assert quality.main([*cli(corpus, output), "--validate-only"]) == 0
    receipt = json.loads((output / "authority.json").read_text())
    assert receipt["all_vector_payloads_verified"] and receipt["model_run"] is False


def retiring_owners(events, observer_error=None, engine_error=None):
    class Observer:
        def close(self):
            events.append("observer")
            if observer_error is not None:
                raise observer_error

    class Engine:
        def close(self):
            events.append("engine")
            if engine_error is not None:
                raise engine_error

    return Observer(), Engine()


def contains_error(root, expected):
    pending, seen = [root], set()
    while pending:
        error = pending.pop()
        if error is expected:
            return True
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        pending.extend((error.__cause__, error.__context__))
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
    return False


def test_successful_worker_retirement_keeps_callback_order(tmp_path):
    events = []
    observer, engine = retiring_owners(events)
    assert quality.retire_worker(observer, engine, tmp_path, 3, None, []) is None
    assert events == ["observer", "engine"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("primary", [RuntimeError("teacher failure"), KeyboardInterrupt("cancelled")])
def test_worker_retirement_keeps_primary_and_both_cleanup_failures(tmp_path, primary):
    events = []
    observer_failure, engine_failure = RuntimeError("restore failed"), ValueError("close failed")
    native_cause = OSError("original native cause")
    primary.__cause__ = native_cause
    observer, engine = retiring_owners(events, observer_failure, engine_failure)
    with pytest.raises(type(primary)) as caught:
        quality.retire_worker(observer, engine, tmp_path, 3, primary, [])
    assert caught.value is primary
    assert events == ["observer", "engine"]
    for failure in (native_cause, observer_failure, engine_failure):
        assert contains_error(caught.value, failure)
    artifact = json.loads((tmp_path / "rank-3-close-failure.json").read_text())
    assert artifact["passed"] is False and artifact["error"] == "close failed"


def test_worker_close_only_failure_is_reported_without_replacing_it(tmp_path):
    events = []
    failure = OSError("close failed")
    observer, engine = retiring_owners(events, engine_error=failure)
    with pytest.raises(OSError) as caught:
        quality.retire_worker(observer, engine, tmp_path, 0, None, [])
    assert caught.value is failure and events == ["observer", "engine"]
    assert json.loads((tmp_path / "rank-0-close-failure.json").read_text())["complete"] is False


def test_worker_retirement_without_constructed_engine_preserves_primary(tmp_path):
    events = []
    observer, _ = retiring_owners(events)
    failure = RuntimeError("construction failed")
    with pytest.raises(RuntimeError) as caught:
        quality.retire_worker(observer, None, tmp_path, 0, failure, [])
    assert caught.value is failure and events == ["observer"]


def test_close_failure_artifact_error_retains_all_failures(monkeypatch, tmp_path):
    events = []
    primary, closing, reporting = RuntimeError("teacher"), ValueError("close"), OSError("disk full")
    observer, engine = retiring_owners(events, engine_error=closing)

    def fail_write(*args):
        raise reporting

    monkeypatch.setattr(quality, "write_json", fail_write)
    with pytest.raises(RuntimeError) as caught:
        quality.retire_worker(observer, engine, tmp_path, 0, primary, [])
    assert caught.value is primary and events == ["observer", "engine"]
    assert contains_error(primary, closing) and contains_error(primary, reporting)


def test_diagnostic_retention_exhaustion_happens_after_both_owners_retire(tmp_path):
    events = []
    primary, cleanup, exhausted = RuntimeError("teacher"), OSError("restore"), MemoryError("retention")
    native_cause, native_context = ValueError("native cause"), OSError("native context")
    primary.__cause__, primary.__context__ = native_cause, native_context
    observer, engine = retiring_owners(events, observer_error=cleanup)

    class Exhausted(list):
        def extend(self, errors):
            raise exhausted

    with pytest.raises(RuntimeError) as caught:
        quality.retire_worker(observer, engine, tmp_path, 0, primary, Exhausted())
    assert caught.value is primary and primary.__cause__ is exhausted
    assert events == ["observer", "engine"]
    trace = primary.__traceback__
    owned = []
    while trace is not None:
        if trace.tb_frame.f_code is quality.retire_worker.__code__:
            owned.extend(trace.tb_frame.f_locals.values())
        trace = trace.tb_next
    assert any(value is native_cause for value in owned)
    assert any(value is native_context for value in owned)
