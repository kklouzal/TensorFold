"""Actual TP status/ownership methods on CPU; no model/GPU qualification."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import dp4a_target as target
from tensorfold.families.qwen4_exp.cuda.forward import _kv_check
from tensorfold.families.qwen4_exp.cuda.kv_status import KVStatusCohort, workspace_bytes
from test_flashnext_kv_status_cohort import states_for


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def numeric():
    owner = object.__new__(target.Owner)
    owner.ready = True
    owner.closed = owner.retiring = False
    owner.failure = None
    owner.pending = True
    owner.status_words = (torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32))
    owner.main = NS(status=owner.status_words[0])
    owner.capture = NS(status=owner.status_words[1])
    owner.cohort = None
    return owner


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("role", [0, 1])
def test_every_rank_fails_before_authority_with_one_existing_gather(rank, role):
    owner = numeric()
    states = states_for([[0, -2], [0, -2]], 8)
    for st in states:
        st._expert_numeric_owner = owner
    packet = torch.zeros((8, 6), dtype=torch.int32)
    packet[:, 1] = packet[:, 3] = -2
    packet[7, 4 + role] = 16
    calls = []

    def gather(send, receive):
        calls.append(send.clone())
        assert send.shape == (6,)
        receive.copy_(packet.reshape(-1))

    owner.comm = NS(rank=rank, world=8, all_gather=gather)
    owner.cohort = KVStatusCohort(24, owner.comm, "cpu", numeric=owner)
    with pytest.raises(RuntimeError, match="before authority"):
        states[0].kv_check()
    with pytest.raises(RuntimeError, match=f"rank=7, role={role}, bits=16") as caught:
        # Named warm/capture callers also omit workspace/comm. Discovery must
        # forward the original owner rather than perform per-State readbacks.
        _kv_check([(st, i, i + 1) for i, st in enumerate(states)])
    assert len(calls) == 1 and owner.failure is caught.value
    with pytest.raises(RuntimeError, match="retirement"):
        owner.assert_exposure()
    assert all(not st._kv_pending for st in states)


def test_success_header_preserves_feedback_and_uses_no_new_wait(monkeypatch):
    owner = numeric()
    states = states_for([[0, -2]], 8)
    states[0]._expert_numeric_owner = owner
    calls = []

    def gather(send, receive):
        calls.append(send.clone())
        receive.copy_(send.repeat(8))

    owner.comm = NS(rank=0, world=8, all_gather=gather)
    owner.cohort = KVStatusCohort(24, owner.comm, "cpu", candidate_rows=32, numeric=owner)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: pytest.fail("newstatuswait"))
    feedback = torch.tensor([3, 2, 11, 12, 13], dtype=torch.int32)
    got = owner.cohort.read_feedback(states, feedback, comm=owner.comm)
    assert got.tolist() == feedback.tolist() and len(calls) == 1 and not owner.pending
    states[0].kv_check()
    owner.assert_exposure()
    assert workspace_bytes(24, 8, numeric_words=2)[0] - workspace_bytes(24, 8)[0] == 72
    assert workspace_bytes(24, 8, numeric_words=2)[1] - workspace_bytes(24, 8)[1] == 64


def test_transport_failure_poison_is_retained_and_never_retried():
    owner = numeric()
    error = OSError("gather failed")

    def gather(*_):
        raise error

    owner.comm = NS(rank=0, world=8, all_gather=gather)
    owner.cohort = KVStatusCohort(24, owner.comm, "cpu", numeric=owner)
    states = states_for([[0, -2]], 8)
    with pytest.raises(OSError) as caught:
        owner.cohort.check(states, comm=owner.comm)
    assert caught.value is owner.failure is owner.cohort._readback_error is error
    with pytest.raises(RuntimeError, match="retirement"):
        owner.assert_exposure()
    with pytest.raises(RuntimeError, match="retirement"):
        owner.cohort.check(states, comm=owner.comm)


def test_private_identity_and_exact_funding_terms():
    from tensorfold.families.qwen4_exp.cuda import prefix_policy
    from tensorfold.cuda.sm70.router_blas import PROFILES
    assert target.PROFILE in PROFILES
    assert prefix_policy.identity(target.PROFILE) != prefix_policy.identity(prefix_policy.ROUTER_BLAS_PROFILE)
    assert prefix_policy.state_epoch(target.PROFILE) != prefix_policy.state_epoch(prefix_policy.ROUTER_BLAS_PROFILE)
    values = target.funding()
    assert values["roles_device_bytes"] == 825344
    assert values["status_device_delta_bytes"] == 512
    assert values["host_commitment_bytes"] == target.HOST_BYTES + 64
    for slots in [1, 16, 25]:
        with pytest.raises(ValueError, match="native24"):
            target.funding(slots)


def test_capture_borrow_requires_explicit_accepted_begin(monkeypatch):
    owner = numeric()
    owner.borrowed = owner.capturing = False
    stream = NS(cuda_stream=17)
    owner.weights = NS(device="cpu")
    owner.stream = NS(cuda_stream=18)
    parent = NS(side=stream)
    owner.capture_parent = parent
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: stream)
    role = object.__new__(target.Role)
    role.owner, role.name = owner, "capture"
    with pytest.raises(RuntimeError, match="borrower"):
        role.check()
    with owner.borrow(parent):
        role.check()  # Explicit side-stream warmup, not capture.
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="not begun"):
            role.check()
        owner.capturing = True
        role.check()  # Only after this owner's accepted capture_begin.
        with pytest.raises(RuntimeError, match="nonnested"):
            with owner.borrow(parent):
                pytest.fail("nested")
    assert not owner.borrowed and not owner.capturing


def test_failed_close_keeps_bindings_and_poison(monkeypatch):
    owner = numeric()
    owner.weights = NS(device="cpu")
    owner.main, owner.capture = object(), object()
    original = owner.main
    error = OSError("retirement fence")

    def fence(*_):
        raise error

    monkeypatch.setattr(torch.cuda, "synchronize", fence)
    with pytest.raises(OSError) as caught:
        owner.close()
    assert caught.value is owner.failure is error and owner.main is original and owner.retiring
    with pytest.raises(RuntimeError, match="retirement"):
        owner.assert_exposure()


def constructed_owner(monkeypatch, *, overbudget=False):
    from test_dp4a_lifetime import setup
    from tensorfold.cuda import capacity
    from tensorfold.families.qwen4_exp.cuda import capture_lifetime
    original, log, stream, backend = setup(monkeypatch)
    bank = original.bank
    bank.implementation = "packed_blocks2w"
    bank.forward = lambda *_args, **_kwargs: log.append("fallback")
    from tensorfold.cuda.sm70 import gguf_ops
    monkeypatch.setattr(gguf_ops, "experts", lambda *_args, **_kwargs: log.append("selected-shared-only"))
    backend.place_shared = lambda *_: log.append("place-shared")
    backend.project_routed = backend.project
    w = NS(device="cpu", layers=[NS(moe=NS(experts=bank)) for _ in range(48)],
           comm=NS(world=8, rank=0), meta={"sm70_profile": target.PROFILE}, cfg=NS(execution_dtype="float16", hidden=2560, moe_width=80, experts=512, top_k=10))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    receipts = []
    monkeypatch.setattr(capacity, "admit_host_commitment", lambda value: receipts.append(value))
    observations = iter([{"VmRSS": 0, "VmHWM": 0}, {"VmRSS": target.HOST_BYTES + 1 if overbudget else 0, "VmHWM": 0}])
    monkeypatch.setattr(capture_lifetime, "host_memory", lambda: next(observations))
    backend.contain = lambda *_: log.append("contain")
    backend.metadata_header_bytes = lambda: 512
    monkeypatch.setattr(target, "extension", lambda: backend)
    owner = object.__new__(target.Owner)
    if overbudget:
        monkeypatch.setattr(target, "MAX_METADATA_OBJECTS", 1)
        with pytest.raises(MemoryError, match="metadata object count"):
            owner.__init__(w)
        return owner
    owner.__init__(w)
    assert receipts == [8 * (target.HOST_BYTES + 64)]
    return owner, bank, log, stream


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_real_role_constructor_funds_two_serial_workspaces_and_exact_fallback(monkeypatch, rows):
    owner, bank, log, _stream = constructed_owner(monkeypatch)
    assert owner.main.qx.data_ptr() != owner.capture.qx.data_ptr()
    def rounded(n):
        return (n + 511) // 512 * 512
    for role in [owner.main, owner.capture]:
        assert sum(rounded(t.nbytes) for t in role.buffers) == 412672
    owner.validate_frame()
    x = torch.zeros((rows, 2560), dtype=torch.float16)
    picks = torch.zeros((rows, 11), dtype=torch.int32)
    out = torch.zeros((rows, 11, 2560), dtype=torch.float16)
    owner.main.bind_output(out)
    owner.main.input(x)
    owner.main.run(bank, x, picks, out)
    assert owner.pending and owner.main.calls == 1 and log[-1] == "contain"
    bank.gate.ggml_type = 21
    owner.main.run(bank, x, picks, out)
    assert owner.main.exact_calls == 1 and log[-2:] == ["fallback", "contain"]
    with pytest.raises(RuntimeError, match="layout"):
        owner.validate_frame()


def test_overbudget_partial_owner_stays_unready_and_retains_roles(monkeypatch):
    owner = constructed_owner(monkeypatch, overbudget=True)
    assert not owner.ready and owner.main is not None and owner.capture is not None
    with pytest.raises(RuntimeError, match="retirement"):
        owner.check_alive()


def test_closed_tool_dp4a_intent_requires_current_composition(tmp_path, monkeypatch):
    from tools import v100_backend_probe as probe
    from test_gdn_state_batch_probe import options
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", target.PROFILE)
    monkeypatch.setenv("TENSORFOLD_SM70_GDN_STATE_BATCH", "1")
    args = options(tmp_path) + ["--router-blas-trial", "--packed-shared-fusion-trial", "--dp4a-target-trial"]
    parsed = probe.arguments(args)
    assert parsed.dp4a_target_trial and parsed.router_blas_trial and parsed.session_ceiling == 24
    assert (probe.CONTEXT, probe.WORLD, probe.DEPTH, probe.CONFIDENCE) == (262144, 8, 2, .7)
    with pytest.raises(SystemExit):
        probe.arguments(args[:-1])


def test_actual_worker_factory_checker_requires_role_and_cohort_identity():
    from tools.v100_backend_probe import check_dp4a_target_engine
    owner = numeric()
    owner.weights = NS(meta={})
    owner.host_observed_bytes = 0
    owner.eligible_mask = (True,) * 48
    owner.metadata_receipt = {}
    owner.main = NS(qx=torch.empty(8), status=owner.status_words[0])
    owner.capture = NS(qx=torch.empty(8), status=owner.status_words[1])
    owner.pending = False
    owner.cohort = NS(numeric=owner, numeric_words=2)
    engine = NS(sm70_profile=target.PROFILE, execution_rows=32, w=owner.weights,
                capacity_plan={"dp4a_target": {"funding": target.funding()}},
                multi=NS(kv_cohort=owner.cohort, buf=NS(dp4a_target=owner.main),
                         mbuf=NS(dp4a_target=None), pbuf=NS(dp4a_target=None)))
    engine.w.meta["dp4a_target_owner"] = owner
    assert check_dp4a_target_engine(engine)["policy"] == target.POLICY
    engine.multi.mbuf.dp4a_target = owner.main
    with pytest.raises(ValueError, match="separation"):
        check_dp4a_target_engine(engine)


@pytest.mark.parametrize("direct,graph,passed", [(0,0,False), (48,0,True), (0,30,True), (1,48,True)])
def test_counter_evidence_excludes_cold_capture_and_requires_actual_target_submission(direct, graph, passed):
    from tools.v100_backend_probe import dp4a_target_evidence
    keys = ("dp4a.main.python_submissions", "dp4a.graph_submitted_hot_layer_calls")
    before = {"counters": {keys[0]: 10, keys[1]: 20, "dp4a.capture.python_submissions": 0}}
    after = {"counters": {keys[0]: 10 + direct, keys[1]: 20 + graph, "dp4a.capture.python_submissions": 500}}
    assert dp4a_target_evidence(before, after)["passed"] is passed
    assert not dp4a_target_evidence(None, after)["passed"]


def test_capture_does_not_bypass_state_or_prefix_authority(monkeypatch):
    owner = numeric()
    state = states_for([[0, -2]], 8)[0]
    state._expert_numeric_owner = owner
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for operation in [state.kv_check, state.kv_snapshot_metadata, lambda: state.set_pos(18)]:
        with pytest.raises(RuntimeError, match="before authority"):
            operation()


def test_real_finish_contains_terminal_streams_mixed_and_logits_before_candidates(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import forward
    log = []
    buffers = NS(prefill=False, h=torch.zeros((1, 10240), dtype=torch.float16),
                 streams=torch.empty((1, 10240), dtype=torch.float16),
                 mixed=torch.zeros((1, 2560), dtype=torch.float16), xs_mixed=None,
                 logits=torch.zeros((1, 5)), dp4a_target=NS(terminal=lambda value: log.append(tuple(value.shape))))
    weights = NS(cfg=NS(eps=1e-6, streams=4, low=320), head=None, comm=object(), meta={})
    monkeypatch.setattr(forward, "_writeback", lambda *_: log.append("writeback"))
    monkeypatch.setattr(forward, "_readout", lambda *_: log.append("readout"))
    monkeypatch.setattr(forward, "_mm", lambda *_: log.append("head") or buffers.logits)
    monkeypatch.setattr(forward, "candidates", lambda *_a, **_k: log.append("candidates"))
    assert forward.finish(weights, None, buffers, 1, None) is buffers.logits
    assert log == ["writeback", (1, 10240), "readout", (1, 2560), "head", (1, 5), "candidates"]


def test_status_word_or_communicator_rebinding_refuses_before_collective():
    owner = numeric()
    owner.comm = NS(rank=0, world=8, all_gather=lambda *_: pytest.fail("foreigncollective"))
    owner.cohort = KVStatusCohort(24, owner.comm, "cpu", numeric=owner)
    states = states_for([[0, -2]], 8)
    with pytest.raises(ValueError, match="original admitted"):
        owner.cohort.check(states, comm=NS(rank=0, world=8))
    owner.status_words = (owner.status_words[0].clone(), owner.status_words[1])
    with pytest.raises(RuntimeError, match="status bindings"):
        owner.check_alive()
