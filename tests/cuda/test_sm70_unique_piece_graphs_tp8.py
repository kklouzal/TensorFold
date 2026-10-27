"""Real TP8 unique-history frame/commit oracles for exact N1 graph pieces."""

import socket
import sys
import time
from types import SimpleNamespace as NS

import pytest
import torch


def _equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _committed(actual, expected):
    assert (actual.pos, actual.mtp_len, actual.cur) == (expected.pos, expected.mtp_len, expected.cur)
    for li, parity in enumerate(actual.cur):
        _equal(actual.rec[parity, li], expected.rec[expected.cur[li], li])
    _equal(actual.conv, expected.conv)
    _equal(actual.ple_tail, expected.ple_tail)
    for left, right in zip((*actual.kc, actual.mtp_kc), (*expected.kc, expected.mtp_kc)):
        end = actual.mtp_len if left is actual.mtp_kc else actual.pos
        logical = torch.arange(end, device=left.k.device)
        indices, owned = left.local_indices(logical)
        reference, reference_owned = right.local_indices(logical)
        assert torch.equal(owned, reference_owned)
        for name in ("k", "v", "ks", "vs"):
            _equal(getattr(left, name)[indices[owned]], getattr(right, name)[reference[owned]])
    for left, right in zip(actual.compact_indexer.entries, expected.compact_indexer.entries):
        assert (left is None) is (right is None)
        if left is not None:
            assert left.n == right.n
            _equal(left.raw, right.raw)
            _equal(left.coordinates, right.coordinates)
    for left, right in zip(actual.pooled, expected.pooled):
        _equal(left[:actual.pos // actual.ratio], right[:expected.pos // expected.ratio])
    _equal(actual.mtp_pooled[:actual.mtp_len // actual.ratio],
           expected.mtp_pooled[:expected.mtp_len // expected.ratio])


def _worker(rank, port):
    import faulthandler
    import traceback
    from tensorfold.cleanup import finish, rollback
    from test_flashnext_sm70_mtp import _weights_with_stock_mtp_protocol
    from tensorfold.cuda.comm import NCCL
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.decode import prefill
    from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, stage
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder, _slot
    from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs, UNIQUE_CAPTURE_CONTRACT
    from tensorfold.families.qwen4_exp.cuda.state import CAND, Buffers

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    family_ops._NATIVE_FAMILY, family_ops._NATIVE_ATTENTION = True, False
    comm, failed, graphs = NCCL(), False, None
    faulthandler.dump_traceback_later(60, file=sys.stderr)
    diagnostic = None

    def diagnostic_failure():
        if diagnostic is not None:
            raise diagnostic

    retire_operations = (lambda: comm.close(abort=failed), faulthandler.cancel_dump_traceback_later,
                         diagnostic_failure)

    def retire():
        # A blocked native close keeps the watchdog armed. Independent cleanup
        # failures and diagnostic failures retain their owning primary.
        finish(retire_operations)

    def phase(label):
        print(f"[unique-piece rank={rank} t={time.monotonic():.3f}] {label}", file=sys.stderr, flush=True)

    try:
        phase("communicator/open")
        comm.open(rank, 8, "127.0.0.1", port, timeout_s=90, key="tf_unique_n1_piece_graphs")
        phase("communicator/ready")
        weights = _weights_with_stock_mtp_protocol(head_dim=256)
        width = weights.cfg.vocab // 8
        weights.head.weight = weights.head.weight[rank * width:(rank + 1) * width].contiguous()
        weights.head.n, weights.comm = width, comm
        weights.meta.update(world=8, rank=rank, vocab_offset=rank * width, full=NS(kv_heads=2),
            history_layout="unique", indexer_history="gpu", attention_degree=1,
            capture_ready=True, unique_capture_contract=UNIQUE_CAPTURE_CONTRACT)
        decoders = [MultiDecoder(weights, slots=1, capacity=260, logical_context=257, depth=2,
            confidence=0.7, prefill_rows=8, kv_dtype="int8", graphs=False, keep=1, full_windows=1)
            for _ in range(2)]
        reference, decoder = decoders
        phase("decoders/ready")
        graphs = RoundGraphs(decoder, 1 << 30)
        original_capture, original_pieces = graphs.bound._capture, graphs._pieces

        def capture(kind, key, lanes, buffers, segments, run):
            label, calls = f"{kind}/rows={key[1]}", [0]
            phase("capture/begin/" + label)

            def body():
                calls[0] += 1
                phase(f"body/begin/{label}/call={calls[0]}")
                result = run()
                phase(f"body/end/{label}/call={calls[0]}")
                return result

            result = original_capture(kind, key, lanes, buffers, segments, body)
            phase("capture/end/" + label)
            return result

        def pieces(run, buffers, **kwargs):
            phase("pieces/begin")
            result = original_pieces(run, buffers, **kwargs)
            phase("pieces/end")
            return result

        graphs.bound._capture, graphs._pieces = capture, pieces
        phase("catalogue/begin")
        graphs.capture()
        phase("catalogue/ready")
        decoder.graphs = graphs
        assert {key[1] for key in graphs.verify} == {1, 2, 3}
        assert {key[1] for key in graphs.mtp} == {1, 2, 3}
        state = decoder.free[0]
        prompt = ([4, 17, 3, 99, 33, 2, 7, 5, 6] * 15)[:129]
        prefill(_slot(weights, state, decoder.buf, decoder.mbuf, decoder.pbuf, 260, 8), prompt, None)
        oracle = Buffers(weights, decoder.buf.rows, 260, moe_prefill=True)
        for rows, kept in ((1, 1), (2, 1), (3, 2)):
            phase(f"manual/begin/rows={rows}/kept={kept}")
            before = state.clone()
            tokens = [31 + row for row in range(rows)]
            expected = compute(weights, stage(weights, oracle, [(before, tokens)]), oracle).clone()
            actual = graphs.verify_forward(stage(weights, decoder.buf, [(state, tokens)], late=True))
            _equal(actual, expected)
            _equal(decoder.buf.streams[:rows], oracle.streams[:rows])
            words = 8 * rows * (2 * CAND + 1)
            _equal(decoder.buf.cand_all[:words], oracle.cand_all[:words])
            commit(weights, state, decoder.buf, rows, kept)
            commit(weights, before, oracle, rows, kept)
            _committed(state, before)
            phase(f"manual/end/rows={rows}")
        # Generation reset keeps storage addresses while retiring the old raw
        # spans. Existing graph keys must then serve real fresh/prefix requests.
        state.reset(weights)
        for repetition in range(2):
            phase(f"request/begin/repetition={repetition}")
            expected = Stream([4, 17, 3, 99, 33, 2, 7, 5, 6], 12, stop_eos=False)
            actual = Stream(list(expected.prompt), 12, stop_eos=False)
            phase(f"admit/eager/begin/repetition={repetition}")
            reference.admit(expected)
            phase(f"admit/eager/end/repetition={repetition}")
            phase(f"admit/graph/begin/repetition={repetition}")
            decoder.admit(actual)
            phase(f"admit/graph/end/repetition={repetition}")
            for round_index in range(128):
                phase(f"round/eager/repetition={repetition}/round={round_index}")
                reference.round()
                phase(f"round/graph/repetition={repetition}/round={round_index}")
                decoder.round()
                phase(f"round/compare/repetition={repetition}/round={round_index}")
                assert expected.error is actual.error is None
                assert (actual.out, actual.drafts, actual.done) == (expected.out, expected.drafts, expected.done)
                _committed(actual.st, expected.st)
                if actual.done:
                    break
            else:
                raise AssertionError("finite unique graph request did not complete")
            assert len(actual.out) == 12
            assert actual.cached == expected.cached
            if repetition:
                assert actual.cached > 0
            phase(f"request/end/repetition={repetition}")
        assert graphs.bound.hits > 0
        comm.ready("unique-n1-piece-oracles-complete", every=1, timeout=60)
        graphs.close()
        phase("graphs/closed")
        decoder.graphs = None
    except BaseException as primary:
        failed = True
        try:
            phase("primary/" + type(primary).__name__[:64])
            traceback.print_exception(primary, limit=32, file=sys.stderr)
        except BaseException as error:
            diagnostic = error
        rollback(comm, primary, retire)
    else:
        retire()


def test_v100_tp8_unique_n1_exact_pieces_match_eager_frames_mtp_partial_commit_reset_and_prefix():
    if torch.cuda.device_count() < 8 or any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(8)):
        pytest.skip("requires eight real SM70 GPUs and the supported source foundation")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = torch.multiprocessing.spawn(_worker, args=(port,), nprocs=8, join=False)
    try:
        deadline = time.monotonic() + 240
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("unique exact-piece graph participant did not retire")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=10)
