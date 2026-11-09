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


def owned_plan_data(mode='capture', reference=None):
    return {'schema': 'tensorfold.owned-teacher-plan-v1', 'mode': mode, 'vocab': 16, 'context': 64,
            'input_ids': [1, 2, 3], 'max_frames': 2, 'reference': reference, 'read_timeout_s': 60,
            'identity': {'run_id': 'run1', 'case_id': 'case1', 'model_sha256': 'a' * 64,
                         'frontend_sha256': 'b' * 64, 'profile_sha256': 'c' * 64}}


def load_owned_plan(tmp_path, monkeypatch, data):
    from types import SimpleNamespace as NS
    from tensorfold.families.qwen4_exp.cuda import retention_resources
    # Pure logic tests deliberately substitute the selected-runtime proof;
    # they do not qualify this ARM test container's Python/native projection.
    monkeypatch.setattr(retention_resources, '_json_payload_policy',
                        lambda: NS(frame_work=1024, identity='CPU-spy-policy-not-runtime-proof'))
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    path = tmp_path / ('plan-' + data['mode'] + '.json')
    path.write_text(json.dumps(data))
    return quality.owned_teacher_plan(path)


def teacher_fixture(tmp_path, monkeypatch, prepared, *, chosen=3, close_error=None):
    from types import ModuleType, SimpleNamespace as NS
    import threading
    events, notified = [], []
    kv_identity = ('pair', 'working-policy', ('logical-context-v1', 64),
                   ('position-profile-v1', 'position-v1', 0), ('coordinate-profile-v1', 'e' * 64))
    monkeypatch.setattr(quality, '_owned_runtime_sources',
                        lambda engine: {key: 'a' * 64 for key in ('engine', 'multi', 'decode', 'sampling', 'teacher')})
    class Tensor:
        is_cuda = True
        requires_grad = False
        device = 'cuda:0'
        def __init__(self, data):
            self.data = np.asarray(data, dtype='<f4')
        @property
        def shape(self):
            return self.data.shape
        @property
        def dtype(self):
            return 'float32'
        def is_contiguous(self):
            return self.data.flags.c_contiguous
        def numel(self):
            return self.data.size
        def element_size(self):
            return self.data.itemsize
        def data_ptr(self):
            return self.data.ctypes.data
        def __getitem__(self, key):
            return Tensor(self.data[key])
        def reshape(self, *shape):
            return Tensor(self.data.reshape(*shape))
        def copy_(self, source):
            self.data[:] = source.data
            events.append('CPUcopy')
            return self
        def numpy(self):
            return self.data
    runtime = NS(float32='float32', empty=lambda shape, **kw: Tensor(np.empty(shape, dtype='<f4')),
                 cuda=NS(current_stream=lambda device: NS(synchronize=lambda: events.append('fence'))))
    decode, multi = NS(), NS()
    def sample(*args, **kwargs):
        events.append('original-sample')
        return [chosen]
    def gathered(w, packets, count, positions, sampling, **kwargs):
        events.append('original-candidates')
        return decode.tp_sample_rows(w, kwargs['logits'], positions, sampling)
    decode.tp_sample_rows = multi.tp_sample_rows = sample
    decode.choose_gathered = multi.choose_gathered = gathered
    package = ModuleType('tensorfold.families.qwen4_exp.cuda')
    package.decode, package.multi = decode, multi
    package.retention_resources = sys.modules['tensorfold.families.qwen4_exp.cuda.retention_resources']
    monkeypatch.setitem(sys.modules, 'tensorfold.families.qwen4_exp.cuda', package)
    monkeypatch.setitem(sys.modules, 'torch', runtime)
    def gather(source, destination):
        events.append('all-gather')
        destination.data[:] = np.tile(source.data, 8)
    comm = NS(all_gather=gather)
    w = NS(comm=comm, mtp=None, meta={'vocab_offset': 0})
    decoder = NS(buf=NS(logits=Tensor(np.zeros((9, 2))), rows=9,
                       owner_kernel_family='torch-reference', owner_graph_family='uncaptured'),
                 pbuf=NS(owner_kernel_family='torch-reference', owner_graph_family='uncaptured'),
                 graphs=None, streams={}, filling=[], _ends=lambda stream: [],
                 capacity=64, prefill_rows=8, states=[NS(kv_identity=kv_identity), NS(kv_identity=kv_identity)],
                 _sample_plan=NS(widths=[2] * 8, mapping=None, offset=0))
    def admit(stream):
        stream.cached = 0
        stream.st = NS(pos=3, slot_generation=1, history_slot=0, kv_identity=kv_identity)
        decoder.filling.append(stream)
        events.append('admit')
    step = 0
    def round_():
        nonlocal step
        stream = decoder.filling[0] if decoder.filling else list(decoder.streams.values())[0]
        decoder.filling.clear()
        decoder.streams[0] = stream
        if step == 0:
            values = Tensor([[1, 2]])  # Actual prefill source can be outside target buf.logits.
            tokens = decode.tp_sample_rows(w, values, [3], stream.sampling, plan=decoder._sample_plan)
        else:
            decoder.buf.logits.data[0] = [2, 1]
            tokens = multi.choose_gathered(w, object(), 1, [3 + step], stream.sampling,
                                          logits=decoder.buf.logits[:1], plan=decoder._sample_plan)
        stream.out.extend(tokens)
        stream.st.pos = 3 + step
        step += 1
        return [stream] if step == 2 else []
    def finish(done):
        decoder.streams.clear()
        events.append('finish')
    decoder.admit, decoder.round, decoder.finish = admit, round_, finish
    def close():
        events.append('close')
        if close_error is not None:
            raise close_error
        engine._closed = True
    engine = NS(multi=decoder, rank=0, tp=8, comm=comm, w=w, context_window=64, sm70_profile='reference', close=close,
                kv_pair=NS(identity='pair'), position_profile=NS(identity='position-v1', epoch=0),
                _lifecycle=threading.Condition(), _closed=False)
    directory = tmp_path / ('cohort-' + prepared['data']['mode'])
    directory.mkdir()
    observer = quality.OwnedTeacherFrames(engine, prepared, directory / 'teacher-rank0',
                                          fail=lambda e, error: notified.append(error),
                                          profile={'rank': 0, 'world': 8, 'target_banks': 48,
                                                   'mtp_banks': 0, 'capability': [7, 0],
                                                   'coordinate_profile_identity': 'e' * 64, 'CPU_spy': True})
    observer.setup()
    observer.install()
    stream = NS(prompt=[1, 2, 3], count=2, draft=False, vision=None, constraint=None, penalties=None,
                probabilities=None, carry=None, owed=[], background=False,
                sampling=NS(temperature=0), out=[], error=None, cancel=None)
    return engine, observer, stream, events, notified, decode, multi


def spy_teacher_rank_records(observer, prepared):
    summary = observer.directory / 'summary.json'
    for rank in range(1, 8):
        directory = observer.directory.parent / f'teacher-rank{rank}'
        directory.mkdir()
        copied = json.loads(summary.read_text())
        copied['rank'] = copied['actual_owned_profile']['rank'] = rank
        # Explicitly legitimate rank-local observations need not match rank0.
        copied['actual_owned_profile']['device'] = f'cuda:{rank}'
        copied['actual_owned_profile']['resources'] = {'observed_free': rank * 100}
        for frame in copied['frames']:
            frame['vector'] = None
        (directory / 'summary.json').write_text(json.dumps(copied))
    for rank in range(8):
        profile = json.loads((observer.directory.parent / f'teacher-rank{rank}/summary.json').read_text())['actual_owned_profile']
        (observer.directory.parent / f'rank-{rank}.json').write_text(json.dumps({
            'schema': 'tensorfold.private-owned-qualification-v1', 'rank': rank, 'world': 8, 'mtp_banks': 0,
            'teacher_profile': profile, 'tool_sha256': 'c' * 64, 'runtime_cli_sha256': 'd' * 64,
            'history_host_obligation': {'global_bytes': prepared['resources']['global_host_obligation_bytes'],
                                        'input_sha256': prepared['input_sha256']}}))


def test_owned_teacher_capture_freezes_raw_full_vocab_and_replay_prefix(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, stream, events, notified, decode, multi = teacher_fixture(tmp_path, monkeypatch, prepared)
    before = decode.choose_gathered, decode.tp_sample_rows, multi.choose_gathered, multi.tp_sample_rows
    engine.multi.admit(stream)
    assert engine.multi.round() == []
    done = engine.multi.round()
    engine.multi.finish(done)
    assert observer.tokens == [3, 3] and len(observer.frames) == 2
    assert events.count('all-gather') == 2  # Nested gathered fallback must not recapture.
    assert events.count('original-sample') == 2
    assert events.index('fence') < events.index('original-sample')
    assert before == (decode.choose_gathered, decode.tp_sample_rows, multi.choose_gathered, multi.tp_sample_rows)
    engine.close()
    assert observer.closed and observer.cpu is None and observer.gathered is None and not notified
    summary = observer.directory / 'summary.json'
    recorded = json.loads(summary.read_text())
    assert recorded['closed'] and not recorded['failed'] and not recorded['qualification'] and not recorded['P3_valid']
    assert recorded['method'] == 'none' and recorded['teacher_ids'] == [3, 3]
    assert np.load(observer.directory / 'frame-000000.npy').shape == (16,)
    spy_teacher_rank_records(observer, prepared)
    quality.seal_owned_teacher(observer.directory.parent, prepared, 0)
    authority = observer.directory.parent / 'teacher-authority.json'
    ref = {'path': str(authority), 'sha256': hashlib.sha256(authority.read_bytes()).hexdigest()}
    replay = load_owned_plan(tmp_path, monkeypatch, owned_plan_data('replay', ref))
    second, replayed, request, calls, failures, *_ = teacher_fixture(tmp_path, monkeypatch, replay, chosen=9)
    second.multi.admit(request)
    second.multi.round()
    second.multi.finish(second.multi.round())
    assert request.out == [3, 3]  # The original candidate choice9 never becomes the next teacher input.
    assert [r['selected_original'] for r in replayed.frames] == [9, 9]
    assert calls.count('original-sample') == 2 and calls.count('all-gather') == 2 and not failures
    second.close()


@pytest.mark.parametrize('change', [{'input_ids': [True]}, {'max_frames': 0}, {'max_frames': 62},
                                    {'mode': 'mtp'}, {'vocab': 0}, {'reference': {}}])
def test_owned_teacher_plan_refuses_before_runtime_effects(tmp_path, monkeypatch, change):
    with pytest.raises(ValueError):
        load_owned_plan(tmp_path, monkeypatch, {**owned_plan_data(), **change})


def test_owned_teacher_foreign_or_draft_admission_fails_before_original(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, stream, events, notified, *_ = teacher_fixture(tmp_path, monkeypatch, prepared)
    stream.draft = True
    with pytest.raises(ValueError, match='one exact cold') as error:
        engine.multi.admit(stream)
    assert events == [] and notified == [error.value]
    assert observer.stream is None


def test_owned_teacher_close_preserves_primary_and_retains_uncertain_borrow(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    primary = OSError('real Engine close failed')
    engine, observer, *_ = teacher_fixture(tmp_path, monkeypatch, prepared, close_error=primary)
    with pytest.raises(OSError) as caught:
        engine.close()
    assert caught.value is primary and primary.__cause__ is not None
    assert observer.gathered is not None and observer.cpu is not None and not observer.closed
    with pytest.raises(RuntimeError, match='already attempted'):
        engine.close()


def test_owned_teacher_finiteness_and_cancel_failure_notify_before_sampler(tmp_path, monkeypatch):
    import threading
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, stream, events, notified, *_ = teacher_fixture(tmp_path, monkeypatch, prepared)
    engine.multi.admit(stream)
    stream.cancel = threading.Event()
    stream.cancel.set()
    with pytest.raises(ValueError, match='frozen prefix') as caught:
        engine.multi.round()
    assert 'all-gather' not in events and 'original-sample' not in events
    assert notified == [caught.value]
    assert observer.frames == []


def test_owned_teacher_json_publication_is_exclusive_bounded_and_primary_safe(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    path = tmp_path / 'seal.json'
    quality._owned_teacher_json(path, {'valid': True}, 100)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        quality._owned_teacher_json(path, {'valid': False}, 100)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='prepaid'):
        quality._owned_teacher_json(tmp_path / 'too-large.json', {'payload': 'x' * 100}, 10)
    assert (tmp_path / 'too-large.json').stat().st_size <= 10
    primary = RuntimeError('deadline')
    class Control:
        def check(self):
            raise primary
    with pytest.raises(RuntimeError) as caught:
        quality._owned_teacher_json(tmp_path / 'cancelled.json', {'valid': True}, 100, Control())
    assert caught.value is primary


def test_owned_teacher_hash_streams_exact_extent_and_observes_cooperative_deadline(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    path = tmp_path / 'raw.bin'
    payload = b'0123456789abcdef' * 600000  # Greater than the structured8MiB cap.
    path.write_bytes(payload)
    assert quality._owned_teacher_hash(path, len(payload), len(payload)) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match='extent'):
        quality._owned_teacher_hash(path, len(payload) - 1, len(payload))
    primary = TimeoutError('declared read deadline')
    class Control:
        calls = 0
        def check(self):
            self.calls += 1
            if self.calls == 4:
                raise primary
    control = Control()
    with pytest.raises(TimeoutError) as caught:
        quality._owned_teacher_hash(path, len(payload), len(payload), control)
    assert caught.value is primary and control.calls == 4


def test_owned_teacher_local_rank_close_is_not_a_cohort_authority(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, stream, *_ = teacher_fixture(tmp_path, monkeypatch, prepared)
    engine.multi.admit(stream)
    engine.multi.round()
    engine.multi.finish(engine.multi.round())
    engine.close()
    with pytest.raises(ValueError, match='cohort retirement'):
        quality.seal_owned_teacher(observer.directory.parent, prepared, 7)
    with pytest.raises(FileNotFoundError):
        quality.seal_owned_teacher(observer.directory.parent, prepared, 0)
    assert not (observer.directory.parent / 'teacher-authority.json').exists()
    summary = observer.directory / 'summary.json'
    data = owned_plan_data('replay', {'path': str(summary), 'sha256': hashlib.sha256(summary.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match='incomplete'):
        load_owned_plan(tmp_path, monkeypatch, data)


def test_owned_teacher_replay_metadata_failure_precedes_any_payload_traversal(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    reference = {'schema': 'tensorfold.owned-teacher-authority-v1', 'mode': 'capture', 'method': 'none',
                 'closed': True, 'finished': True, 'failed': False, 'cohort_complete': True, 'launch_exit': 0,
                 'vocab': 16, 'input_ids': [1, 2, 3], 'teacher_ids': [3, 3],
                 'declared_identity': prepared['data']['identity'], 'rank_receipts': [{}] * 8,
                 'startup_receipts': [{}] * 8, 'frames': [
                     {'step': 0, 'teacher_id': 3, 'position': 3, 'vector': {'file': 'teacher-rank0/frame-000000.npy',
                      'bytes': 100, 'sha256': 'a' * 64, 'dtype': '<f4', 'shape': [16]}},
                     {'step': 1, 'teacher_id': 3, 'position': 4, 'vector': {'file': 'teacher-rank0/frame-000001.npy',
                      'bytes': 2**63, 'sha256': 'b' * 64, 'dtype': '<f4', 'shape': [16]}}]}
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(reference))
    reads = []
    monkeypatch.setattr(quality, '_owned_teacher_hash', lambda *args: reads.append(args))
    with pytest.raises(ValueError, match='identity'):
        load_owned_plan(tmp_path, monkeypatch, owned_plan_data('replay', {
            'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    assert reads == []


def test_owned_teacher_parent_plan_validation_is_torch_and_numpy_free(tmp_path, monkeypatch):
    import os
    import subprocess
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    path = tmp_path / 'parent-plan.json'
    path.write_text(json.dumps(owned_plan_data()))
    program = '''
import importlib.abc, sys
from types import SimpleNamespace
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','numpy'):
   raise AssertionError('accelerator/array runtime imported by parent DATA validation')
sys.meta_path.insert(0, Block())
from tensorfold.families.qwen4_exp.cuda import retention_resources
retention_resources._json_payload_policy=lambda:SimpleNamespace(frame_work=1024,identity='CPU-spy-not-selected-proof')
from tools import v100_rotorquant_quality as quality
prepared=quality.owned_teacher_plan(sys.argv[1])
assert prepared['data']['mode']=='capture'
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
    completed = subprocess.run([sys.executable, '-B', '-c', program, str(path)],
                               env={**os.environ, 'PYTHONPATH': str(ROOT / 'src') + os.pathsep + str(ROOT)},
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_owned_teacher_unsupported_runtime_policy_refuses_before_input_read(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import retention_resources
    primary = ValueError('selected runtime proof unavailable')
    def refuse():
        raise primary
    monkeypatch.setattr(retention_resources, '_json_payload_policy', refuse)
    with pytest.raises(ValueError) as caught:
        quality.owned_teacher_plan(tmp_path / 'does-not-exist.json')
    assert caught.value is primary


@pytest.mark.parametrize('field', ['KV', 'context', 'position', 'execution', 'expert-profile'])
def test_owned_teacher_seal_rejects_single_rank_actual_profile_drift(tmp_path, monkeypatch, field):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, stream, *_ = teacher_fixture(tmp_path, monkeypatch, prepared)
    engine.multi.admit(stream)
    engine.multi.round()
    engine.multi.finish(engine.multi.round())
    engine.close()
    spy_teacher_rank_records(observer, prepared)
    path = observer.directory.parent / 'teacher-rank3/summary.json'
    record = json.loads(path.read_text())
    profile = record['actual_owned_profile']
    if field == 'KV':
        profile['teacher_semantics']['state_kv_identity'][0] = 'wrong-KV-pair'
    elif field == 'context':
        profile['teacher_semantics']['context'] = 63
    elif field == 'position':
        identity = profile['teacher_semantics']['state_kv_identity']
        next(item for item in identity if isinstance(item, list) and item[0] == 'position-profile-v1')[1] = 'wrong-position'
    elif field == 'execution':
        profile['teacher_semantics']['prepared_execution']['graph_family'] = 'wrong-execution-family'
        record['owner_family'] = profile['teacher_semantics']['prepared_execution']
    else:
        profile['plan_identity'] = ['different-expert-profile']
    path.write_text(json.dumps(record))
    # Matching two local documents does NOT excuse a different semantic cohort.
    if field in ('context', 'position', 'execution'):
        startup_path = observer.directory.parent / 'rank-3.json'
        startup = json.loads(startup_path.read_text())
        startup['teacher_profile'] = profile
        startup_path.write_text(json.dumps(startup))
    with pytest.raises(ValueError):
        quality.seal_owned_teacher(observer.directory.parent, prepared, 0)
    assert not (observer.directory.parent / 'teacher-authority.json').exists()


def test_owned_teacher_all_actual_states_bind_same_semantics_before_capture(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, *_ = teacher_fixture(tmp_path, monkeypatch, prepared)
    engine.multi.states[1].kv_identity = ('another-State-profile',)
    with pytest.raises(ValueError, match='State numerical profiles'):
        quality.OwnedTeacherFrames(engine, prepared, tmp_path / 'never-created',
                                  fail=lambda *args: None, profile=observer.profile)
    assert not (tmp_path / 'never-created').exists()


def test_owned_teacher_failure_path_uses_preloaded_helper_and_preserves_notification(tmp_path, monkeypatch):
    import builtins
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, _stream, _events, notified, *_ = teacher_fixture(tmp_path, monkeypatch, prepared)
    primary, secondary = RuntimeError('authentic sampler failure'), OSError('notification failure')
    def notify_then_fail(selected, error):
        notified.append(error)
        raise secondary
    observer.fail = notify_then_fail
    original_import = builtins.__import__
    def late_import(name, *args, **kwargs):
        if name == 'tensorfold.cleanup':
            raise MemoryError('late failure-path helper import is forbidden')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', late_import)
    with pytest.raises(RuntimeError) as caught:
        observer.abort(primary)
    assert caught.value is primary and primary.__cause__ is secondary
    assert observer.failed is primary and notified == [primary]


def test_owned_teacher_none_return_without_closed_lifecycle_is_not_retirement(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, *_ = teacher_fixture(tmp_path, monkeypatch, prepared)
    observer.old_close = lambda: None  # Deliberately violates the fully constructed Engine.close contract.
    with pytest.raises(RuntimeError, match='positive lifecycle'):
        engine.close()
    assert not observer.closed and observer.gathered is not None and observer.cpu is not None
    summary = json.loads((observer.directory / 'summary.json').read_text())
    assert not summary['closed'] and summary['failed']


def test_owned_teacher_status_storage_exhaustion_precedes_hook_effects(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, *_ = teacher_fixture(tmp_path, monkeypatch, prepared)
    original = engine.close, engine.multi.admit, engine.multi.round, engine.multi.finish
    primary = MemoryError('preallocated status storage')
    class Exhausted(quality.OwnedTeacherFrames):
        def __setattr__(self, name, value):
            if name == 'close_status':
                raise primary
            super().__setattr__(name, value)
    with pytest.raises(MemoryError) as caught:
        Exhausted(engine, prepared, tmp_path / 'never-admitted', fail=lambda *args: None, profile=observer.profile)
    assert caught.value is primary
    assert original == (engine.close, engine.multi.admit, engine.multi.round, engine.multi.finish)
    assert not (tmp_path / 'never-admitted').exists()


def test_owned_teacher_close_primary_keeps_summary_failure_in_fixed_status(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    primary, publication = OSError('authentic Engine close'), OSError('summary failed')
    engine, observer, *_ = teacher_fixture(tmp_path, monkeypatch, prepared, close_error=primary)
    def failed_summary(*args, **kwargs):
        raise publication
    monkeypatch.setattr(quality, '_owned_teacher_json', failed_summary)
    with pytest.raises(OSError) as caught:
        engine.close()
    assert caught.value is primary and observer.close_status[5] is publication
    assert observer.engine is engine and observer.gathered is not None
    assert observer.close_attempted and not observer.closed


def test_owned_teacher_restore_failures_survive_transport_exhaustion(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, *_ = teacher_fixture(tmp_path, monkeypatch, prepared)
    restore, allocation = OSError('admit restore'), MemoryError('neutral transport preparation')
    decoder, attempted = observer.decoder, []
    class RestoreProxy:
        def __getattr__(self, name):
            return getattr(decoder, name)
        def __setattr__(self, name, value):
            attempted.append(name)
            if name == 'admit':
                raise restore
            setattr(decoder, name, value)
    observer.decoder = RestoreProxy()
    def exhausted(*args):
        raise allocation
    observer.raise_failures = exhausted
    with pytest.raises(OSError) as caught:
        engine.close()
    assert caught.value is restore and restore.__cause__ is allocation
    assert attempted == ['admit', 'round', 'finish']
    assert observer.close_status[1] is restore
    assert isinstance(observer.close_status[4], RuntimeError)  # Validation still attempted.
    assert (observer.directory / 'summary.json').exists()  # Publication still attempted.
    assert observer.closed and observer.close_attempted


def test_owned_teacher_alias_restore_keeps_sampler_primary_under_aggregation_exhaustion(tmp_path, monkeypatch):
    prepared = load_owned_plan(tmp_path, monkeypatch, owned_plan_data())
    engine, observer, stream, events, notified, decode, _multi = teacher_fixture(tmp_path, monkeypatch, prepared)
    sampler, restore, allocation = RuntimeError('sampler frame'), OSError('alias restore'), MemoryError('transport')
    module, name, original, wrapped = observer.alias_records[0]
    class ModuleProxy:
        def __getattr__(self, key):
            return getattr(module, key)
        def __setattr__(self, key, value):
            if key == name and value is original:
                raise restore
            setattr(module, key, value)
    observer.alias_records = ((ModuleProxy(), name, original, wrapped), *observer.alias_records[1:])
    def fail_vector(*args):
        raise sampler
    def fail_transport(*args):
        raise allocation
    observer.write_vector, observer.raise_failures = fail_vector, fail_transport
    engine.multi.admit(stream)
    with pytest.raises(RuntimeError) as caught:
        engine.multi.round()
    assert caught.value is sampler and sampler.__cause__ is allocation
    assert observer.alias_status[0] is restore and notified == [sampler]
    assert 'original-sample' not in events
    assert observer.thread is None and observer.engine is engine
    # Retire only the test's artificial fault; production must stop, not retry.
    setattr(decode, name, original)
    with pytest.raises(RuntimeError) as second:
        engine.multi.round()
    assert second.value is sampler
