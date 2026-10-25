"""Real TP8 NCCL/native partial captures; full-model qualification is separate."""

import socket
import time
from types import SimpleNamespace as NS

import pytest
import torch


def _worker(rank, port):
    from tensorfold.cuda.comm import NCCL
    from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch
    from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers, _encode
    from tensorfold.families.qwen4_exp.cuda.owner_graphs import OwnerAttentionGraphs
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    comm, failed, graphs = NCCL(), False, None
    try:
        comm.open(rank, 8, "127.0.0.1", port, timeout_s=90, key="tf_owner_graph_component")
        device = torch.device("cuda", rank)
        cfg = NS(hidden=512, streams=4, conv_kernel=4, conv_dim=1024, nk=2, nv=4, dk=128, dv=128,
                 kv_heads=1, head_dim=256, index_dim=128, index_ratio=4, ple_kernel=4, ngram_size=3,
                 ple_layers=[], eos=(0,), heads=3, index_heads=4, index_budget=8, eps=1e-6,
                 mrope_section=(1, 1, 1), rope_attention_factor=1.0, execution_dtype="float16")
        weights = NS(cfg=cfg, device=device, layers=[NS(index=i, linear=False) for i in range(12)],
                     mtp=None, comm=comm, meta={"world": 8, "rank": rank, "full": NS(kv_heads=2),
                                               "execution_provider": "sm70", "history_layout": "unique",
                                               "attention_degree": 1})
        scratch = AttnScratch(4, 3, 256, 257, device, budget=8, ratio=4, dtype=torch.float16)
        workspace = OwnerBuffers(weights, 4, scratch)
        pair = get_pair("int8")
        workspace.prepare_native(pair=pair, page_tokens=128)
        graphs = OwnerAttentionGraphs(workspace, rows=4, budget_bytes=64 << 20, comm=comm)
        # The actual startup capture has no target weights or State cache yet.
        graphs.prepare(quiescent=lambda: workspace.native_states is None)
        state = State(weights, 257, 4, "int8")
        workspace.bind_native_states((state,))
        state.kv_begin()
        key = torch.zeros((133, 1, 256), dtype=torch.float16, device=device)
        value = torch.full_like(key, 1 if rank < 4 else 2)
        codes, scales = _encode(key, pair.key, state.kv_status, 0, error=2)
        values, value_scales = _encode(value, pair.value, state.kv_status, 0, error=4)
        for cache in state.kc:
            cache.append(0, codes, values, scales, value_scales)
        state.set_pos(129)
        layer = NS(index=0)

        def inputs(rows):
            bucket = graphs.bucket(rows)
            workspace.clear_native_inputs(bucket)
            workspace.stage_native_descriptors([(state, 0, rows)], layer, 0, rows, bucket_rows=bucket)

        for rows in (1, 3, 4, 1):
            inputs(rows)
            eager = graphs.run(rows, graph=False).clone()
            workspace.mark_native_submitted()  # Include the clone, the last borrowed-output reader.
            inputs(rows)
            captured = graphs.run(rows, graph=True).clone()
            workspace.mark_native_submitted()
            assert torch.equal(captured, eager)
            assert not int(state.kv_status[0])
        # Reset/reuse advances a stable device generation. A previously staged
        # graph must reject its old row descriptor before reading live history.
        inputs(3)
        state.reset(weights)
        stale = graphs.run(3, graph=True).clone()
        workspace.mark_native_submitted()
        assert not bool(stale.any()) and int(state.kv_status[0]) & 32
        comm.ready("owner-native-graphs-complete", every=1, timeout=60)
        graphs.close()
        workspace.retire_native()
    except BaseException:
        failed = True
        raise
    finally:
        # Failed graph owners remain reachable until this terminal cohort exits;
        # abort is the communicator's existing failed-operation retirement path.
        comm.close(abort=failed)


def test_v100_tp8_owner_partial_graphs_replay_live_and_masked_rows():
    if torch.cuda.device_count() < 8 or any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(8)):
        pytest.skip("requires eight real SM70 GPUs and the supported source-built V100 foundation")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = torch.multiprocessing.spawn(_worker, args=(port,), nprocs=8, join=False)
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
