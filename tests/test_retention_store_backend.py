# ruff: noqa: E402 - selected CUDA orchestration requires optional Torch.
"""Prepared transport selection, actual CPU Store borrowing and rank identity.

Native NCCL calls use independent boundary-provider doubles; the eight-rank
catalog control uses real CPU State/Store/worker ownership, not GPU execution.
"""
import ast
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
import weakref

import pytest
pytest.importorskip("torch")
import torch
from torch.distributed import TCPStore

from tensorfold.cuda import comm
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from test_flashnext_tp_control import CPUWords, Store, engine as protocol_engine
from test_nccl_buffer_boundary import source as native_provider


def policy(tier):
    return None if tier is None else RetentionPolicy(tier, 1, 30, **({
        "ram_bytes": 2 << 30, "chunk_bytes": 128 << 10,
        "progress_bytes": 2 * 8 * (128 << 10), "overlap_bytes": 0} if tier == "ram" else {}))


def prepared(implemented=True, tier="ram", *, existing=None, managed=False):
    owner = object.__new__(FlashNextEngine)
    owner._store_use_libuv = None
    owner._profile_build = object() if managed else None
    owner.comm = existing
    owner.retention_policy = policy(tier)
    owner._prepare_store_backend(implemented, rank=0, world=1)
    return owner


def cpu_owned_cohort():
    """Exercise the real issuance/agreement logic with a CPU collective."""
    origin = comm.create_owned_tcp_store(0, 1, "127.0.0.1", 0, timeout_s=2)
    owner = comm.NCCL()
    owner.rank, owner.world, owner.device = 0, 1, torch.device("cpu")
    owner.store, owner.store_origin = origin.store, origin
    origin.adopt(owner._store_origin_token)
    owner.all_gather = lambda mine, received: received.copy_(mine)
    owner.verify_store_cohort()
    owner.device = None  # No device/native communicator was acquired by this CPU boundary control.
    return owner


@pytest.mark.parametrize("implemented,tier,expected", [
    (False, None, None), (True, None, False), (True, "gpu", False), (True, "ram", False)])
def test_selection_covers_all_actual_unique_profiles_and_preserves_other_regions(implemented, tier, expected):
    assert prepared(implemented, tier)._store_use_libuv is expected


def test_managed_backend_is_process_fixed_and_never_replaces_borrowed_store(monkeypatch):
    transport = cpu_owned_cohort()
    server = transport.store
    owner = prepared(existing=transport, managed=True)
    monkeypatch.setattr("torch.distributed.TCPStore", lambda *_a, **_k: pytest.fail("borrow opened another listener"))
    owner._prepare_store_backend(True, rank=0, world=1)
    assert owner.comm.store is server
    with pytest.raises(ValueError, match="process-fixed"):
        owner._prepare_store_backend(False, rank=0, world=1)
    assert owner._store_use_libuv is False and owner.comm.store is server
    transport.close()


@pytest.mark.parametrize("backend", [True, False])
def test_actual_borrowed_mismatch_refuses_before_nccl_or_listener(backend, monkeypatch):
    server = TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=2), wait_for_workers=False, use_libuv=backend)
    monkeypatch.setattr(comm, "_library", lambda: pytest.fail("borrow mismatch reached NCCL"))
    monkeypatch.setattr("torch.distributed.TCPStore", lambda *_a, **_k: pytest.fail("borrow replaced its listener"))
    owner = comm.NCCL()
    with pytest.raises(ValueError, match="owned daemon/cohort provenance"):
        owner.open(0, 1, "unused", 0, store=server, use_libuv=False)
    assert server.num_keys() == 0


@pytest.mark.parametrize("selected", [None, True, False])
def test_fallback_factory_preserves_default_and_exact_timeout(selected, monkeypatch):
    _cls, lib, _streams, _distributed = native_provider()
    calls = []
    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return NS(libuvBackend=kwargs.get("use_libuv", True), set=lambda *_a: None)
    monkeypatch.setattr(comm, "_library", lambda: lib)
    actual_owned = comm.create_owned_tcp_store
    if selected is False:
        def owned(rank, world, master, port, *, timeout_s):
            calls.append(((master, port, world, rank == 0), {"timeout": timedelta(seconds=timeout_s), "owned": True}))
            return actual_owned(0, 1, "127.0.0.1", 0, timeout_s=2)
        monkeypatch.setattr(comm, "create_owned_tcp_store", owned)
    else:
        monkeypatch.setattr("torch.distributed.TCPStore", factory)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.synchronize", lambda *_a: None)
    owner = comm.NCCL()
    try:
        owner.open(0, 1, "host", 29551, use_libuv=selected)
        expected = {"timeout": timedelta(seconds=600)}
        if selected is not None:
            expected["use_libuv"] = selected
        if selected is False:
            expected = {"timeout": timedelta(seconds=600), "owned": True}
        assert calls == [(("host", 29551, 1, True), expected)]
    finally:
        owner.close()


def early_factory():
    """Run the canonical metadata-before-native Store statements in isolation."""
    import tensorfold.families.qwen4_exp.cuda.engine as source
    tree = ast.parse(Path(source.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashNextEngine")
    initialize = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_initialize")
    branch = next(n for n in ast.walk(initialize) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "dflash_plan is not None or media_requested")
    prefix = []
    for statement in branch.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "selected_context"
                                                    for t in statement.targets):
            break
        prefix.append(statement)
    function = ast.parse("def acquire(self, tp, rank, master, port):\n    pass\n").body[0]
    function.body = prefix
    namespace = {"__package__": source.__package__, "timedelta": timedelta}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), source.__file__, "exec"), namespace)
    return namespace["acquire"]


@pytest.mark.parametrize("implemented,tier", [(False, None), (True, None), (True, "gpu"), (True, "ram")])
def test_early_factory_preserves_default_and_one_hour_timeout(implemented, tier, monkeypatch):
    owner = prepared(implemented, tier)
    selected = owner._store_use_libuv
    calls = []
    actual_owned = comm.create_owned_tcp_store
    if selected is False:
        def owned(rank, world, master, port, *, timeout_s):
            calls.append(((master, port, world, rank == 0), {"timeout": timedelta(seconds=timeout_s), "owned": True}))
            return actual_owned(0, 1, "127.0.0.1", 0, timeout_s=2)
        monkeypatch.setattr(comm, "create_owned_tcp_store", owned)
    else:
        monkeypatch.setattr("torch.distributed.TCPStore", lambda *a, **k:
            calls.append((a, k)) or NS(libuvBackend=k.get("use_libuv", True)))
    early_factory()(owner, 8, 0, "host", 29551)
    expected = {"timeout": timedelta(seconds=3600)}
    if selected is not None:
        expected["use_libuv"] = selected
    if selected is False:
        expected = {"timeout": timedelta(seconds=3600), "owned": True}
    assert calls == [(("host", 29551, 8, True), expected)]
    if owner._startup_store_origin is not None:
        owner._startup_store_origin.retire()
        owner._startup_store = None


def test_early_managed_borrow_requires_the_exact_verified_transport(monkeypatch):
    transport = cpu_owned_cohort()
    matching, wrong = transport.store, NS(libuvBackend=False)
    owner = prepared()
    owner._profile_build, owner.comm = object(), transport
    monkeypatch.setattr("torch.distributed.TCPStore", lambda *_a, **_k: pytest.fail("borrow opened a listener"))
    early_factory()(owner, 1, 0, "host", 29551)
    assert owner._startup_store is matching
    transport.store = wrong
    with pytest.raises(ValueError, match="owned TCPStore/cohort origin"):
        early_factory()(owner, 1, 0, "host", 29551)
    assert owner._startup_store is wrong  # Failure journal retains the actual borrowed owner.
    transport.store = matching
    transport.close()


def test_open_comm_forwards_only_an_explicit_backend_selection(monkeypatch):
    seen = []
    class Owner:
        store = None
        def open(self, *_a, **kwargs):
            seen.append(kwargs)
        def verify_store_cohort(self):
            seen.append("cohort")
    monkeypatch.setattr(comm, "NCCL", Owner)
    comm.open_comm(0, 1, "host", 29551)
    comm.open_comm(0, 1, "host", 29551, use_libuv=False)
    assert seen == [{}, {"use_libuv": False}, "cohort"]


def test_prepared_backend_is_part_of_eight_rank_settings_consensus():
    leader = protocol_engine(0, 8, Store())
    leader.depth, leader.confidence, leader.max_len = 0, 0.5, 4096
    leader.streams, leader.graphs_enabled, leader.prefill_rows = 1, False, 256
    leader.kv_pair, leader.rope = NS(handshake=123), NS(metadata=lambda: {})
    leader._admission_options, leader.retention_policy = {"admission_policy": "fifo"}, policy("ram")
    words = []
    def capture(mine, received):
        words.append(mine.clone())
        received.view(8, -1).copy_(mine.expand(8, -1))
    leader.comm.all_gather = capture
    leader._store_use_libuv = False
    leader._same_settings(CPUWords, None)
    baseline = words[-1]
    leader._store_use_libuv = True
    def disagreement(mine, received):
        received.view(8, -1).copy_(baseline.expand(8, -1))
        received.view(8, -1)[7].copy_(mine)
    leader.comm.all_gather = disagreement
    with pytest.raises(RuntimeError, match="rank 7"):
        leader._same_settings(CPUWords, None)


def test_actual_eight_rank_legacy_store_catalog_transfer_without_gpu_wake_claim(monkeypatch):
    import test_retained_catalog_cohort as cohort
    def selected_store(*args, **kwargs):
        return TCPStore(*args, **kwargs, use_libuv=False)
    monkeypatch.setattr(cohort, "TCPStore", selected_store)
    # Preserve the existing independent full transfer/lag/relocation/key oracle.
    cohort.test_real_eight_rank_cpu_transfer_retires_keys_and_relocates_packed_histories()


def test_false_client_and_clone_on_true_daemon_never_gain_borrowed_authority(monkeypatch):
    server = TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=2), wait_for_workers=False, use_libuv=True)
    client = TCPStore("127.0.0.1", server.port, None, False, timedelta(seconds=2), use_libuv=False)
    assert client.libuvBackend is False and client.clone().libuvBackend is False
    monkeypatch.setattr(comm, "_library", lambda: pytest.fail("unproved daemon reached NCCL"))
    for borrowed in (client, client.clone()):
        with pytest.raises(ValueError, match="owned daemon/cohort provenance"):
            comm.NCCL().open(1, 2, "unused", server.port, store=borrowed, use_libuv=False)


@pytest.mark.parametrize("value", ["1", "Y", "yes", "t", "TrUe"])
def test_active_native_agent_fallback_refuses_before_store_or_nccl(monkeypatch, value):
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", value)
    monkeypatch.setenv("MASTER_PORT", " +29551suffix")
    monkeypatch.setattr(comm, "_library", lambda: pytest.fail("agent reuse reached NCCL"))
    monkeypatch.setattr("torch.distributed.TCPStore", lambda *_a, **_k: pytest.fail("agent reuse mutated Store"))
    with pytest.raises(ValueError, match="native agent-store reuse"):
        comm.NCCL().open(0, 1, "host", 29551, use_libuv=False)


@pytest.mark.parametrize("agent,port", [("false", "unused"), ("invalid", "29551"), ("true", "invalid"),
                                      ("true", "9999999999999"), ("true", "29552")])
def test_unused_or_native_invalid_fallback_configuration_does_not_restrict_fresh_store(agent, port, monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", agent)
    monkeypatch.setenv("MASTER_PORT", port)
    assert comm._agent_store_reuse(29551) is False


def test_retired_origin_releases_exact_shared_store_and_rejects_reborrow():
    transport = cpu_owned_cohort()
    origin = transport.store_origin
    transport.close()
    assert origin.retired and origin.store is None and not origin.verified
    with pytest.raises(ValueError, match="owned TCPStore/cohort origin"):
        comm.validate_store_origin(origin, None, 0, 1, verified=True)


def test_borrowed_adopt_after_effect_failure_retires_journal_and_traceback_aliases(monkeypatch):
    origin = comm.create_owned_tcp_store(0, 1, "127.0.0.1", 0, timeout_s=2)
    raw = origin.store
    original, failure = type(origin).adopt, MemoryError("after origin transfer")
    def adopt(owner, token):
        original(owner, token)
        raise failure
    monkeypatch.setattr(type(origin), "adopt", adopt)
    monkeypatch.setattr(comm, "_library", lambda: pytest.fail("failed transfer reached NCCL"))
    owner = comm.NCCL()
    with pytest.raises(MemoryError) as caught:
        owner.open(0, 1, "unused", 0, store=raw, use_libuv=False, store_origin=origin)
    assert caught.value is failure and owner.retired
    assert origin.retired and origin.store is None and owner.store is None
    trace = failure.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code.co_name in ("open", "_initialize"):
            assert trace.tb_frame.f_locals.get("store") is None
            assert trace.tb_frame.f_locals.get("store_origin") is None
        trace = trace.tb_next


def test_optional_store_verification_policy_agrees_before_any_gpu_collective(monkeypatch):
    keys = {"tf_comm_backend/1": b"nccl"}
    closed = []
    owner = NS(store=Store(), open=lambda *_a, **_k: None,
               verify_store_cohort=lambda: pytest.fail("mismatched peers reached a GPU collective"),
               close=lambda **kwargs: closed.append(kwargs["abort"]))
    owner.store.values = keys
    monkeypatch.setattr(comm, "NCCL", lambda: owner)
    with pytest.raises(RuntimeError, match="different communication or adaptive unique profile Store policies"):
        comm.open_comm(0, 2, "host", 29551, use_libuv=False)
    assert closed == [True]


def test_selected_engine_journal_precedes_native_open_and_after_effect_registration_failure(monkeypatch):
    published, opened, closed = [], [], []
    class Owner:
        store = None
        def open(self, *_a, **_k):
            assert published[-1] is self
            opened.append(self)
        def verify_store_cohort(self):
            pass
        def close(self, **_kwargs):
            closed.append(self)
    monkeypatch.setattr(comm, "NCCL", Owner)
    value = comm.open_comm(0, 1, "host", 29551, use_libuv=False, register=published.append)
    assert published == opened == [value]
    primary = MemoryError("after partial communicator publication")
    def fail(owner):
        published.append(owner)
        raise primary
    with pytest.raises(MemoryError) as caught:
        comm.open_comm(0, 1, "host", 29551, use_libuv=False, register=fail)
    assert caught.value is primary and closed == [published[-1]] and len(opened) == 1


@pytest.mark.parametrize("after", [False, True])
def test_owned_factory_preserves_primary_when_origin_retirement_fails(monkeypatch, after):
    primary, cleanup = ValueError("post-construction validation"), MemoryError("retirement failure")
    original, retired = comm._TCPStoreOrigin.retire, []
    def fail_retire(owner, token=None):
        retired.append(owner)
        if after:
            original(owner, token)
        raise cleanup
    monkeypatch.setattr(comm, "validate_store_backend", lambda *_a: (_ for _ in ()).throw(primary))
    monkeypatch.setattr(comm._TCPStoreOrigin, "retire", fail_retire)
    try:
        with pytest.raises(ValueError) as caught:
            comm.create_owned_tcp_store(0, 1, "127.0.0.1", 0, timeout_s=2)
        assert caught.value is primary and retired
        assert retired[0] in primary._tensorfold_retained_owners
    finally:
        for owner in retired:
            original(owner)


@pytest.mark.parametrize("master", ["127.0.0.1", "localhost"])
def test_owned_fd_endpoint_proof_preserves_numeric_and_hostname_master(master):
    origin = comm.create_owned_tcp_store(0, 1, master, 0, timeout_s=2)
    try:
        assert origin.server_owned and origin.listener.phase == "native"
        assert origin.listener.descriptor is None and origin.listener.socket.fileno() == -1
        assert origin.store.num_keys() == 0  # Temporary endpoint nonce and client are fully retired.
    finally:
        origin.retire()


def test_configured_client_must_reach_exact_owned_fd_endpoint(monkeypatch):
    from tensorfold.cuda import local_store
    foreign = TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=2), wait_for_workers=False, use_libuv=True)
    references = []
    class Routed:
        def __init__(self, owned):
            self.owned = owned
            self.remote = TCPStore("127.0.0.1", foreign.port, None, False, timedelta(seconds=2), use_libuv=False)
        @property
        def port(self):
            return self.owned.port
        @property
        def libuvBackend(self):
            return False
        def check(self, keys):
            return self.remote.check(keys)
        def get(self, key):
            return self.remote.get(key)
    def factory(_master, port, world, is_master, **options):
        assert is_master and options["master_listen_fd"] >= 0
        owned = TCPStore("127.0.0.1", port, world, True, **options)
        result = Routed(owned)
        references.append(weakref.ref(result))
        return result
    monkeypatch.setattr(local_store, "_store_type", lambda: factory)
    with pytest.raises(ValueError, match="does not reach the owned TCPStore listener"):
        comm.create_owned_tcp_store(0, 1, "127.0.0.1", 0, timeout_s=2)
    assert references and references[0]() is None
