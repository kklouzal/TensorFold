"""Real TP8 NCCL/native partial captures; full-model qualification is separate."""

from contextlib import contextmanager
import faulthandler
import os
import socket
import sys
import time
import traceback
from types import SimpleNamespace as NS

import pytest
import torch


def _diagnostic_phase(rank, head4, phase, rows=None):
    """Bounded test-only stderr records; diagnostic I/O cannot replace failure."""
    try:
        print(f"OWNER_GRAPH_PHASE pid={os.getpid()} rank={rank} head4={int(head4)} "
              f"phase={phase} rows={rows}", file=sys.stderr, flush=True)
    except BaseException:
        # O0/C6: this optional diagnostic has an explicit no-escape contract,
        # even for asynchronous BaseException. Worker-body cancellation and the
        # original parent/cohort cleanup remain authoritative and unchanged.
        pass


def _diagnostic_exception(rank, head4, error):
    """No-escape primary breadcrumb: at most 16 frames/16384 trace characters.

    No local values are captured. The original exception object is neither
    changed nor retained by the diagnostic after it returns. Cleanup and its
    existing failure policy still run exactly as before this optional report.
    """
    try:
        _diagnostic_phase(rank, head4, "worker.primary.exception." + type(error).__name__[:64])
        trace = traceback.TracebackException(type(error), error, error.__traceback__,
                                             limit=16, capture_locals=False)
        chunks, remaining, truncated = [], 16384, False
        for chunk in trace.format(chain=True):
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated = True
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining == 0:
                truncated = True
                break
        text = "".join(chunks)
        if truncated:
            text += "\n[primary traceback truncated at 16384 characters]\n"
        print(f"OWNER_GRAPH_PRIMARY pid={os.getpid()} rank={rank} head4={int(head4)}\n{text}",
              file=sys.stderr, flush=True)
    except BaseException:
        _diagnostic_phase(rank, head4, "worker.primary.breadcrumb.failed")


@contextmanager
def _worker_diagnostics(rank, head4):
    """This fresh spawned process owns one dump, at120s within the600s test.

    sys.stderr remains open/owned by the worker; no file descriptor is closed or
    replaced. Cancel only after the original worker's final cleanup unwinds.
    One nonrepeating schedule bounds dumps even if parent startup itself stalls.
    Setup/cancel/report plumbing is no-escape, including BaseException, so it
    cannot replace an authentic worker/cleanup result. Exceptions from yield's
    body are never caught here; original parent cancellation remains in force.
    """
    armed = False
    try:
        faulthandler.dump_traceback_later(120, repeat=False, file=sys.stderr, exit=False)
        armed = True
        _diagnostic_phase(rank, head4, "dump.armed.120s")
    except BaseException as error:
        _diagnostic_phase(rank, head4, "dump.start.failed."+type(error).__name__)
    try:
        yield
    finally:
        if armed:
            try:
                faulthandler.cancel_dump_traceback_later()
                _diagnostic_phase(rank, head4, "dump.cancelled.after_cleanup")
            except BaseException as error:
                _diagnostic_phase(rank, head4, "dump.cancel.failed."+type(error).__name__)



def _populate_compact_indexer(state, staging, raw, coordinates, weight, frequencies,
                              eps, sections, amplitude, committed):
    """Produce a real prefill span, pool it, then commit its accepted prefix.

    This setup frame is independent of the unchanged four-row native verify
    workspace. Span/staging ownership remains live until State reset/retirement;
    no frontier, generation, position or span field is assigned by the fixture.
    """
    # Continue the caller's original KV frame: never clear earlier sticky errors.
    expected = (state.slot_generation, state.history_generation)
    staging.begin(mtp=False)
    with staging.reading(mtp=False, borrowers=(raw, coordinates)):
        for ordinal, owner in enumerate(state.compact_indexer.entries):
            if owner is not None:
                entry = state.compact_indexer.append(ordinal, state.pos, raw, coordinates,
                    staging, 0, len(raw), mtp=False, expected=expected)
                entry.pool(state.pooled[ordinal], weight, frequencies, eps, sections, amplitude)
        state.set_pos(committed)


def _retire_worker_resources(rank, head4, graphs, workspace, reference, head_comm, comm, *, failed):
    """Retire dependency owners before native communicator release, once only.

    Captured NCCL graph references must die before NCCL destroy/abort. A failed
    graph close retains every dependent owner. Workspace failures retain both
    communicators; a failed head release retains its borrowed parent Store.
    Returned False requires terminal process containment, not Python teardown.
    """
    errors = []
    if graphs is not None:
        try:
            _diagnostic_phase(rank, head4, 'graphs.close.before')
            graphs.close()
            if graphs.closed is not True:
                raise RuntimeError('graph owner did not prove retirement')
            _diagnostic_phase(rank, head4, 'graphs.close.after')
        except BaseException as error:
            errors.append(error)
            return errors, False
    for label, owner in (('owner', workspace), ('reference', reference)):
        if owner is not None:
            try:
                _diagnostic_phase(rank, head4, label + '.close.before')
                owner.close()
                if owner.closed is not True:
                    raise RuntimeError('workspace owner did not prove retirement')
                _diagnostic_phase(rank, head4, label + '.close.after')
            except BaseException as error:
                errors.append(error)
    if errors:
        return errors, False
    if head_comm is not None:
        try:
            _diagnostic_phase(rank, head4, 'head_comm.close.before')
            head_comm.close(abort=failed)
            if head_comm.retired is not True:
                raise RuntimeError('head communicator did not prove retirement')
            _diagnostic_phase(rank, head4, 'head_comm.close.after')
        except BaseException as error:
            errors.append(error)
            return errors, False
    try:
        _diagnostic_phase(rank, head4, 'comm.close.before')
        comm.close(abort=failed)
        if comm.retired is not True:
            raise RuntimeError('communicator did not prove retirement')
        _diagnostic_phase(rank, head4, 'comm.close.after')
    except BaseException as error:
        errors.append(error)
        return errors, False
    return errors, True


def _worker_outcome(rank, head4, primary, errors, retired):
    if retired:
        from tensorfold.cleanup import raise_failures
        raise_failures(primary, errors)
        return
    try:
        from tensorfold.cleanup import raise_failures
        raise_failures(primary, errors)
    except BaseException as error:
        if primary is not None and error is not primary:
            _diagnostic_exception(rank, head4, primary)
        _diagnostic_exception(rank, head4, error)
    finally:
        _diagnostic_phase(rank, head4, 'terminal.unretired.owners')
        os._exit(1)  # Caller frames retain owners; no destructor or pointer retry.


def _finish_worker(rank, head4, primary, graphs, workspace, reference, staging, head_comm, comm, *, failed):
    # Arguments and the suspended worker frame retain every owner, including
    # staging, without allocating a separate retention tuple during exhaustion.
    try:
        errors, retired = _retire_worker_resources(rank, head4, graphs, workspace,
            reference, head_comm, comm, failed=failed)
    except BaseException as error:
        # Includes preparation/list/tuple allocation failure inside retirement.
        try:
            if primary is not None:
                _diagnostic_exception(rank, head4, primary)
            _diagnostic_exception(rank, head4, error)
        finally:
            _diagnostic_phase(rank, head4, 'terminal.cleanup.preparation.failed')
            os._exit(1)
    else:
        _worker_outcome(rank, head4, primary, errors, retired)


def _worker(rank, port, head4):
    with _worker_diagnostics(rank, head4):
        _diagnostic_phase(rank, head4, "imports.before")
        from tensorfold.cuda.comm import NCCL
        from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch
        from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers, _encode
        from tensorfold.families.qwen4_exp.cuda.owner_graphs import OwnerAttentionGraphs
        from tensorfold.families.qwen4_exp.cuda.owner_execution import FULL_COLLECTIVE, HEAD_COLLECTIVE
        from tensorfold.families.qwen4_exp.cuda.state import State
        from tensorfold.families.qwen4_exp.kv_formats import get_pair

        _diagnostic_phase(rank, head4, "imports.after")
        _diagnostic_phase(rank, head4, "threads.before")
        torch.set_num_threads(1)
        _diagnostic_phase(rank, head4, "threads.after")
        _diagnostic_phase(rank, head4, "device.before")
        torch.cuda.set_device(rank)
        _diagnostic_phase(rank, head4, "device.after")
        _diagnostic_phase(rank, head4, "comm.construct.before")
        comm, head_comm, failed, graphs = NCCL(), None, False, None
        workspace = reference = staging = None
        primary = None
        _diagnostic_phase(rank, head4, "comm.construct.after")
        try:
            _diagnostic_phase(rank, head4, "comm.open.before")
            comm.open(rank, 8, "127.0.0.1", port, timeout_s=90, key="tf_owner_graph_component")
            _diagnostic_phase(rank, head4, "comm.open.after")
            if head4:
                _diagnostic_phase(rank, head4, "head_comm.open.before")
                head_comm = NCCL()
                head_comm.open(rank % 4, 4, "", 0, timeout_s=90, store=comm.store,
                               key="tf_owner_graph_head/" + str(rank // 4))
                _diagnostic_phase(rank, head4, "head_comm.open.after")
            device = torch.device("cuda", rank)
            cfg = NS(hidden=512, streams=4, conv_kernel=4, conv_dim=1024, nk=2, nv=4, dk=128, dv=128,
                     kv_heads=1, head_dim=256, index_dim=128, index_ratio=4, ple_kernel=4, ngram_size=3,
                     ple_layers=[], eos=(0,), heads=3, index_heads=4, index_budget=8, eps=1e-6,
                     mrope_section=(1, 1, 1), rope_attention_factor=1.0, execution_dtype="float16")
            weights = NS(cfg=cfg, device=device, layers=[NS(index=i, linear=False) for i in range(12)],
                         mtp=None, comm=comm, meta={"world": 8, "rank": rank, "full": NS(kv_heads=2),
                                                   "execution_provider": "sm70", "history_layout": "unique",
                                                   "attention_degree": 1})
            _diagnostic_phase(rank, head4, "workspace.construct.before")
            scratch = AttnScratch(4, 3, 256, 257, device, budget=8, ratio=4, dtype=torch.float16)
            workspace = object.__new__(OwnerBuffers)
            workspace.__init__(weights, 4, scratch)
            _diagnostic_phase(rank, head4, "workspace.construct.after")
            pair = get_pair("int8")
            _diagnostic_phase(rank, head4, "owner.prepare_native.before")
            workspace.prepare_native(pair=pair, page_tokens=128)
            _diagnostic_phase(rank, head4, "owner.prepare_native.after")
            _diagnostic_phase(rank, head4, "owner.transport.before")
            workspace.configure_native_transport(HEAD_COLLECTIVE if head4 else FULL_COLLECTIVE, head_comm or comm)
            _diagnostic_phase(rank, head4, "owner.transport.after")
            _diagnostic_phase(rank, head4, "graphs.construct.before")
            graphs = object.__new__(OwnerAttentionGraphs)
            graphs.__init__(workspace, rows=4, budget_bytes=64 << 20, comm=comm)
            _diagnostic_phase(rank, head4, "graphs.construct.after")
            # The actual startup capture has no target weights or State cache yet.
            _diagnostic_phase(rank, head4, "graphs.prepare.before")
            graphs.prepare(quiescent=lambda: workspace.native_states is None)
            _diagnostic_phase(rank, head4, "graphs.prepare.after")
            # The canonical controlled driver declares/stages this inert tool.
            # Observe this worker only, after native capture preparation executed kernels.
            native_evidence = os.environ.get('TENSORFOLD_NATIVE_CONTRACT_CHILD_EVIDENCE')
            if native_evidence:
                from v100_native_contract_driver import record_native_elfs
                record_native_elfs(['tf_sm70_paged_attention_'], rank=rank,
                    tag='owner-head4-' + str(int(head4)), evidence_dir=native_evidence)
            _diagnostic_phase(rank, head4, "state.adopt.before")
            state = State(weights, 257, 4, "int8")
            workspace.bind_native_states((state,))
            graphs.validate_adoption()
            _diagnostic_phase(rank, head4, "state.adopt.after")
            if head4:
                _diagnostic_phase(rank, head4, "reference.prepare.before")
                reference = object.__new__(OwnerBuffers)
                reference.__init__(weights, 4, scratch)
                reference.prepare_native((state,), pair=pair, page_tokens=128)
                _diagnostic_phase(rank, head4, "reference.prepare.after")
            _diagnostic_phase(rank, head4, "state.populate.before")
            state.kv_begin()
            key = torch.zeros((133, 1, 256), dtype=torch.float16, device=device)
            value = torch.full_like(key, 1 if rank < 4 else 2)
            codes, scales = _encode(key, pair.key, state.kv_status, 0, error=2)
            values, value_scales = _encode(value, pair.value, state.kv_status, 0, error=4)
            for cache in state.kc:
                cache.append(0, codes, values, scales, value_scales)
            from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging, coordinate_rows

            staging = object.__new__(Staging)
            staging.__init__(133, len(state.kc), device, mtp=False)
            raw_indexer = torch.arange(133 * cfg.index_dim, dtype=torch.float32, device=device).reshape(133, cfg.index_dim).half()
            coordinates = coordinate_rows(state, torch.arange(133, dtype=torch.int64, device=device))
            _populate_compact_indexer(state, staging, raw_indexer, coordinates,
                torch.ones(cfg.index_dim, dtype=torch.float32, device=device),
                torch.zeros(1, dtype=torch.float32, device=device), cfg.eps, cfg.mrope_section,
                cfg.rope_attention_factor, 129)
            layer = NS(index=0)
            _diagnostic_phase(rank, head4, "state.populate.after")

            def inputs(rows):
                _diagnostic_phase(rank, head4, "inputs.before", rows)
                bucket = graphs.bucket(rows)
                workspace.clear_native_inputs(bucket)
                workspace.stage_native_descriptors([(state, 0, rows)], layer, 0, rows, bucket_rows=bucket)
                _diagnostic_phase(rank, head4, "inputs.after", rows)

            for rows in (1, 3, 4, 1):
                inputs(rows)
                _diagnostic_phase(rank, head4, "eager.before", rows)
                eager = graphs.run(rows, graph=False).clone()
                _diagnostic_phase(rank, head4, "eager.after", rows)
                workspace.mark_native_submitted()  # Include the clone, the last borrowed-output reader.
                inputs(rows)
                _diagnostic_phase(rank, head4, "replay.before", rows)
                captured = graphs.run(rows, graph=True).clone()
                _diagnostic_phase(rank, head4, "replay.after", rows)
                workspace.mark_native_submitted()
                assert torch.equal(captured, eager)
                _diagnostic_phase(rank, head4, "eager_replay.compare.after", rows)
                if reference is not None:
                    bucket = reference.native_bucket(rows)
                    reference.clear_native_inputs(bucket)
                    reference.stage_native_descriptors([(state, 0, rows)], layer, 0, rows, bucket_rows=bucket)
                    reference.wait_native_inputs()
                    _diagnostic_phase(rank, head4, "reference.run.before", rows)
                    full = reference.native_run(bucket)[:rows].clone()
                    _diagnostic_phase(rank, head4, "reference.run.after", rows)
                    reference.mark_native_submitted()
                    assert torch.equal(captured, full)
                assert not int(state.kv_status[0])
                _diagnostic_phase(rank, head4, "status.after", rows)
            # Reset/reuse advances a stable device generation. A previously staged
            # graph must reject its old row descriptor before reading live history.
            inputs(3)
            _diagnostic_phase(rank, head4, "stale.reset.before")
            state.reset(weights)
            _diagnostic_phase(rank, head4, "stale.reset.after")
            _diagnostic_phase(rank, head4, "stale.replay.before")
            stale = graphs.run(3, graph=True).clone()
            _diagnostic_phase(rank, head4, "stale.replay.after")
            workspace.mark_native_submitted()
            assert not bool(stale.any()) and int(state.kv_status[0]) & 32
            _diagnostic_phase(rank, head4, "ready.before")
            comm.ready("owner-native-graphs-complete", every=1, timeout=60)
            _diagnostic_phase(rank, head4, "ready.after")
        except BaseException as error:
            failed, primary = True, error
            _diagnostic_exception(rank, head4, error)
        finally:
            # Successful graph retirement is required even when the body failed
            # after capture. Uncertain owners remain live for process containment.
            _finish_worker(rank, head4, primary, graphs, workspace, reference,
                           staging, head_comm, comm, failed=failed)



@pytest.mark.parametrize("head4", [False, True])
def test_v100_tp8_owner_partial_graphs_replay_live_and_masked_rows(head4):
    if torch.cuda.device_count() < 8 or any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(8)):
        pytest.skip("requires eight real SM70 GPUs and the supported source-built V100 foundation")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = torch.multiprocessing.spawn(_worker, args=(port, head4), nprocs=8, join=False)
    try:
        deadline = time.monotonic() + 600
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("native owner graph participant did not retire")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=10)
