"""Native TP4 device token handoff preserves the ordinary MTP protocol."""

import copy
import json
import multiprocessing
import queue
import socket
import time
import traceback

import pytest
import torch


def _qualify(rank, world, port, results):
    from test_flashnext_sm70_bound_graphs import _assert_state
    from test_flashnext_sm70_mtp import _weights_with_stock_mtp_protocol

    from tensorfold.cuda.comm import NCCL
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda import padded_linears
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
    from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs
    from tensorfold.families.qwen4_exp.cuda.state import KVNumericError

    comm, failed = NCCL(), False
    graphs = []
    try:
        torch.cuda.set_device(rank)
        torch.cuda.set_per_process_memory_fraction(0.05, rank)
        family_ops._NATIVE_FAMILY = family_ops._NATIVE_ATTENTION = True
        padded_linears._ENABLED = True
        comm.open(rank, world, "127.0.0.1", port, timeout_s=120, key="tf_device_greedy_uid")
        receipts = []
        for depth, tied in ((2, False), (7, False), (3, True)):
            w = _weights_with_stock_mtp_protocol()
            width = w.cfg.vocab // world
            w.head.weight = w.head.weight[rank*width:(rank+1)*width].contiguous()
            w.head.n, w.comm = width, comm
            if tied:
                # More than top32 equal winners on each shard exercises the
                # ordinary sampler's exact global-ID tie repair as authority.
                assert width > 32
                w.head.weight.zero_()
            w.meta.update(world=world, vocab_offset=rank*width, capture_ready=True, device_greedy_mtp_ready=True, device_greedy_native_ready=True)
            previous = copy.copy(w)
            previous.meta = dict(w.meta, device_greedy_native_ready=False)
            decoders = [MultiDecoder(weights, slots=1, capacity=64, depth=depth, confidence=0,
                                     prefill_rows=8, kv_dtype="int8", graphs=False, keep=0)
                        for weights in (previous, w)]
            for decoder in decoders:
                assert decoder.buf.padded_linears is not None and decoder.mbuf.padded_linears is not None
                graph = RoundGraphs(decoder, 1024*2**20)
                graphs.append(graph)
                graph.capture()
                decoder.graphs = graph
                assert all(buffer.padded_linears is not None for buffer in graph.bound.buffers.values())
            reference, actual = decoders
            handoff = actual.graphs.bound.greedy
            assert handoff is not None and handoff.native is not None
            ordinary_handoff = reference.graphs.bound.greedy
            assert ordinary_handoff is not None and ordinary_handoff.native is None
            streams = [Stream([4, 17, 3, 99, 33, 2, 7, 5, 6], 24, stop_eos=False) for _ in range(2)]
            for decoder, stream in zip(decoders, streams):
                decoder.admit(stream)
            for _iteration in range(26):
                reference.round()
                actual.round()
                a, b = streams
                assert a.out == b.out and a.drafts == b.drafts
                assert a.error is b.error is None and a.done == b.done
                assert (a.accepted, a.drafted) == (b.accepted, b.drafted)
                torch.testing.assert_close(handoff.storage, ordinary_handoff.storage, rtol=0, atol=0)
                assert (handoff.completed, handoff.steps, handoff.readbacks) == (ordinary_handoff.completed, ordinary_handoff.steps, ordinary_handoff.readbacks)
                _assert_state(a.st, b.st)
                assert (a.st.mtp_len, a.st.mtp_drafted) == (b.st.mtp_len, b.st.mtp_drafted)
                for name in ("k", "v", "ks", "vs"):
                    torch.testing.assert_close(getattr(a.st.mtp_kc, name)[:a.st.mtp_len],
                                               getattr(b.st.mtp_kc, name)[:b.st.mtp_len], rtol=0, atol=0)
                torch.testing.assert_close(a.st.mtp_ikc[:a.st.mtp_len], b.st.mtp_ikc[:b.st.mtp_len], rtol=0, atol=0)
                torch.testing.assert_close(a.st.mtp_pooled[:a.st.mtp_len//a.st.ratio],
                                           b.st.mtp_pooled[:b.st.mtp_len//b.st.ratio], rtol=0, atol=0)
                if b.done:
                    break
            assert streams[1].done and len(streams[1].out) == 24
            assert handoff.completed > 0 and handoff.steps > handoff.readbacks == handoff.completed
            if tied:
                assert streams[1].accepted > 0 and set(streams[1].out) == {0}
            receipts.append({"depth": depth, "tied": tied, "outputs": 24, "accepted": streams[1].accepted,
                             "chains": handoff.completed, "steps": handoff.steps, "readbacks": handoff.readbacks})
            if depth == 7:
                # A remote error from the middle step must survive later healthy
                # status resets, and no provisional position can be published.
                state = streams[1].st
                before = (state.mtp_len, state.mtp_drafted, list(streams[1].drafts), list(streams[1].out))
                for step in range(3):
                    state.kv_status[0].fill_(4 if rank == world-1 and step == 1 else 0)
                    state.kv_status[1].fill_(-1)
                    handoff.gather(step, actual.mbuf.logits[:1], state)
                with pytest.raises(KVNumericError):
                    handoff.finish(state, 3, state.mtp_len, 1)
                assert state._kv_error is not None
                assert (state.mtp_len, state.mtp_drafted, streams[1].drafts, streams[1].out) == before
            for decoder in decoders:
                decoder.graphs.close()
                graphs.remove(decoder.graphs)
                decoder.graphs = None
        peak = torch.cuda.max_memory_reserved(rank)
        assert peak < 2 * 1024**3
        print(json.dumps({"rank": rank, "scope": "privatefused-vsC14epiloguecorrectnessonly", "peak_reserved": peak,
                          "cases": receipts}), flush=True)
        comm.ready("device-greedy-qualification-complete", every=1, timeout=60)
        results.put((rank, None))
    except BaseException:
        failed = True
        results.put((rank, traceback.format_exc()))
        raise
    finally:
        for graph in graphs:
            graph.close()
        comm.close(abort=failed)


def test_four_rank_device_greedy_mtp_matches_ordinary_captured_protocol():
    world = 4
    if torch.cuda.device_count() < world:
        pytest.skip("requires four CUDA devices")
    from tensorfold.cuda.sm70 import attention_native, family_native, gdn_native

    family_native.extension()
    attention_native.extension()
    gdn_native.extension()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    children = [context.Process(target=_qualify, args=(rank, world, port, results)) for rank in range(world)]
    observed = []
    try:
        for child in children:
            child.start()
        end = time.monotonic()+240
        while len(observed) < world and time.monotonic() < end:
            try:
                observed.append(results.get(timeout=1))
            except queue.Empty:
                if all(not child.is_alive() for child in children):
                    break
        assert sorted(observed) == [(rank, None) for rank in range(world)], observed
        for child in children:
            child.join(timeout=10)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.pid is not None:
                if child.is_alive():
                    child.kill()
                child.join(timeout=10)
        results.close()
        results.join_thread()
