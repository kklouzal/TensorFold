"""Real TP8 NCCL/native partial captures; full-model qualification is separate."""

import faulthandler
import json
import socket
import sys
import time
import traceback
from types import SimpleNamespace as NS

import pytest
import torch


def _phase(rank, head4, phase, *, rows=None):
    print(json.dumps({"event": "owner-graph-fixture", "rank": rank, "head4": head4,
                      "phase": phase, "rows": rows, "monotonic_ns": time.monotonic_ns()}), flush=True)


def _worker(rank, port, head4):
    from test_sm70_unique_history import _prepare_zero_indexer_rows
    from tensorfold.cuda.comm import NCCL
    from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch
    from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers, _encode
    from tensorfold.families.qwen4_exp.cuda.owner_graphs import OwnerAttentionGraphs
    from tensorfold.families.qwen4_exp.cuda.owner_execution import FULL_COLLECTIVE, HEAD_COLLECTIVE
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    from tensorfold.cleanup import raise_failures

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    comm, head_comm, failed, graphs = NCCL(), None, False, None
    primary, failures, watchdog = None, [None] * 6, False
    try:
        watchdog = True  # A partially armed timer must still reach cancellation.
        faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
        _phase(rank, head4, "full-communicator-open")
        comm.open(rank, 8, "127.0.0.1", port, timeout_s=90, key="tf_owner_graph_component")
        if head4:
            head_comm = NCCL()
            _phase(rank, head4, "head-communicator-open")
            head_comm.open(rank % 4, 4, "", 0, timeout_s=90, store=comm.store,
                           key="tf_owner_graph_head/" + str(rank // 4))
        _phase(rank, head4, "workspace-prepare")
        device = torch.device("cuda", rank)
        cfg = NS(hidden=512, streams=4, conv_kernel=4, conv_dim=1024, nk=2, nv=4, dk=128, dv=128,
                 kv_heads=1, head_dim=256, index_dim=128, index_ratio=4, ple_kernel=4, ngram_size=3,
                 ple_layers=[], eos=(0,), heads=3, index_heads=4, index_budget=8, eps=1e-6,
                 mrope_section=(1, 1, 1), rope_attention_factor=1.0, execution_dtype="float16")
        weights = NS(cfg=cfg, device=device, layers=[NS(index=i, linear=False) for i in range(12)],
                     mtp=None, comm=comm, inv_freq=torch.tensor([1.0], device=device),
                     meta={"world": 8, "rank": rank, "full": NS(kv_heads=2),
                                               "execution_provider": "sm70", "history_layout": "unique",
                                               "attention_degree": 1})
        scratch = AttnScratch(4, 3, 256, 257, device, budget=8, ratio=4, dtype=torch.float16)
        workspace = OwnerBuffers(weights, 4, scratch)
        pair = get_pair("int8")
        workspace.prepare_native(pair=pair, page_tokens=128)
        workspace.configure_native_transport(HEAD_COLLECTIVE if head4 else FULL_COLLECTIVE, head_comm or comm)
        graphs = OwnerAttentionGraphs(workspace, rows=4, budget_bytes=64 << 20, comm=comm)
        # The actual startup capture has no target weights or State cache yet.
        _phase(rank, head4, "graphs-prepare")
        graphs.prepare(quiescent=lambda: workspace.native_states is None)
        _phase(rank, head4, "state-bind")
        state = State(weights, 257, 4, "int8")
        workspace.bind_native_states((state,))
        graphs.validate_adoption()
        reference = None
        if head4:
            reference = OwnerBuffers(weights, 4, scratch)
            reference.prepare_native((state,), pair=pair, page_tokens=128)
        state.kv_begin()
        _phase(rank, head4, "cache-produce")
        key = torch.zeros((133, 1, 256), dtype=torch.float16, device=device)
        value = torch.full_like(key, 1 if rank < 4 else 2)
        codes, scales = _encode(key, pair.key, state.kv_status, 0, error=2)
        values, value_scales = _encode(value, pair.value, state.kv_status, 0, error=4)
        for cache in state.kc:
            cache.append(0, codes, values, scales, value_scales)
        _prepare_zero_indexer_rows(state, weights, 0, 129, range(len(state.kc)))
        state.set_pos(129)
        layer = NS(index=0)

        def inputs(rows):
            bucket = graphs.bucket(rows)
            workspace.clear_native_inputs(bucket)
            workspace.stage_native_descriptors([(state, 0, rows)], layer, 0, rows, bucket_rows=bucket)

        for rows in (1, 3, 4, 1):
            _phase(rank, head4, "eager", rows=rows)
            inputs(rows)
            eager = graphs.run(rows, graph=False).clone()
            workspace.mark_native_submitted()  # Include the clone, the last borrowed-output reader.
            inputs(rows)
            _phase(rank, head4, "captured", rows=rows)
            captured = graphs.run(rows, graph=True).clone()
            workspace.mark_native_submitted()
            _phase(rank, head4, "eager-captured-equality", rows=rows)
            assert torch.equal(captured, eager)
            if reference is not None:
                _phase(rank, head4, "head-full-reference", rows=rows)
                bucket = reference.native_bucket(rows)
                reference.clear_native_inputs(bucket)
                reference.stage_native_descriptors([(state, 0, rows)], layer, 0, rows, bucket_rows=bucket)
                reference.wait_native_inputs()
                full = reference.native_run(bucket)[:rows].clone()
                reference.mark_native_submitted()
                assert torch.equal(captured, full)
            _phase(rank, head4, "status-check", rows=rows)
            assert not int(state.kv_status[0])
        # Reset/reuse advances a stable device generation. A previously staged
        # graph must reject its old row descriptor before reading live history.
        _phase(rank, head4, "stale-generation")
        inputs(3)
        state.reset(weights)
        stale = graphs.run(3, graph=True).clone()
        workspace.mark_native_submitted()
        assert not bool(stale.any()) and int(state.kv_status[0]) & 32
        _phase(rank, head4, "cohort-ready")
        comm.ready("owner-native-graphs-complete", every=1, timeout=60)
        _phase(rank, head4, "graphs-close")
        graphs.close()
        _phase(rank, head4, "workspace-retire")
        workspace.retire_native()
        if reference is not None:
            _phase(rank, head4, "reference-close")
            reference.close()
    except BaseException as error:
        primary = error
        failed = True
        try:
            _phase(rank, head4, "primary-before-retirement")
            traceback.print_exception(error, file=sys.stderr)
        except BaseException as diagnostic:
            failures[0] = diagnostic
    finally:
        # Failed graph owners remain reachable until this terminal cohort exits;
        # abort is the communicator's existing failed-operation retirement path.
        if head_comm is not None:
            try:
                _phase(rank, head4, "head-retirement-begin")
            except BaseException as diagnostic:
                failures[1] = diagnostic
            try:
                head_comm.close(abort=failed)
            except BaseException as error:
                failures[2] = error
        try:
            _phase(rank, head4, "full-retirement-begin")
        except BaseException as diagnostic:
            failures[3] = diagnostic
        try:
            comm.close(abort=failed)
        except BaseException as error:
            failures[4] = error
        if watchdog:
            try:
                faulthandler.cancel_dump_traceback_later()
            except BaseException as error:
                failures[5] = error
    try:
        errors = [error for error in failures if error is not None]
    except BaseException as allocation:
        if primary is not None:
            raise primary from allocation
        for error in failures:
            if error is not None:
                raise error from allocation
        raise
    raise_failures(primary, errors)


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
