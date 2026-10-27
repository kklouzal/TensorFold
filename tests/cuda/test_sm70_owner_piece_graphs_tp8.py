"""Actual TP8 native-eager vs combined exact-piece frame/state oracles.

Torch-owner numerical differences are reported separately without inventing an
acceptance tolerance. They never weaken the exact same-native graph oracle.
"""
import copy
import hashlib
import json
import socket
import sys
import time
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")


def _worker(rank, port, target_greedy=False):
    import faulthandler
    import traceback
    from tensorfold.cleanup import finish, rollback
    from tensorfold.cuda.comm import NCCL
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.decode import prefill
    from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, stage
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder, _slot
    from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs, UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT
    from tensorfold.families.qwen4_exp.cuda.owner_execution import KERNEL_FAMILY, FULL_COLLECTIVE, UNCAPTURED
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation
    from tensorfold.families.qwen4_exp.cuda.prefix_policy import OWNER_GRAPH_PROFILE, OWNER_TARGET_GREEDY_PROFILE, OWNER_TARGET_GUARD_PROFILE
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    from test_sm70_unique_piece_graphs_tp8 import _equal, _committed

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    from test_flashnext_sm70_mtp import _weights_with_stock_mtp_protocol

    family_ops._NATIVE_FAMILY, family_ops._NATIVE_ATTENTION = True, False
    comm, failed, graphs, diagnostic = NCCL(), False, None, None
    preparations, target_owners = [], []
    faulthandler.dump_traceback_later(60, file=sys.stderr)

    def phase(label):
        print(f"[owner-piece rank={rank} t={time.monotonic():.3f}] {label}", file=sys.stderr, flush=True)

    def close_graphs():
        if graphs is not None:
            graphs.close()

    def close_preparations():
        finish(preparation.close for preparation in reversed(preparations))

    def diagnostic_failure():
        if diagnostic is not None:
            raise diagnostic

    def retire():
        # The graph borrower retires before either canonical native workspace.
        # A failed fence preserves owners and keeps the watchdog armed.
        close_graphs()
        if target_owners:
            torch.cuda.synchronize()
            finish(lambda owner=owner: owner.retire_after_fence(fenced=True) for owner in target_owners)
        close_preparations()
        finish((lambda: comm.close(abort=failed), faulthandler.cancel_dump_traceback_later, diagnostic_failure))

    try:
        phase("communicator/open")
        comm.open(rank, 8, "127.0.0.1", port, timeout_s=90, key="tf_native_owner_exact_pieces")
        base = _weights_with_stock_mtp_protocol(head_dim=256)
        width = base.cfg.vocab // 8
        base.head.weight = base.head.weight[rank * width:(rank + 1) * width].contiguous()
        base.head.n, base.comm = width, comm
        base.meta.update(world=8, rank=rank, vocab_offset=rank * width, full=NS(kv_heads=2),
            history_layout="unique", indexer_history="gpu", attention_degree=1,
            capture_ready=True, unique_capture_contract=UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT,
            sm70_profile=OWNER_GRAPH_PROFILE)
        text = {"hidden_size": base.cfg.hidden, "num_attention_heads": 24, "num_key_value_heads": 2,
                "head_dim": 256, "indexer_budget": base.cfg.index_budget,
                "indexer_compress_ratio": base.cfg.index_ratio}
        spec = dict(kernel_family=KERNEL_FAMILY, graph_family=UNCAPTURED,
                    collective_family=FULL_COLLECTIVE, graph_budget_bytes=0, trial=True)

        def decoder(native, *, compact_target=False):
            weights = copy.copy(base)
            weights.meta = dict(base.meta)
            if compact_target:
                from tensorfold.families.qwen4_exp.cuda.target_greedy import POLICY

                weights.meta.update(sm70_profile=OWNER_TARGET_GUARD_PROFILE if compact_target == "guard" else OWNER_TARGET_GREEDY_PROFILE,
                    sm70_target_greedy_trial={"policy": POLICY}, target_greedy_owner=None)
                if compact_target == "guard":
                    weights.meta["sm70_native_guard_trial"] = True
            preparation = None
            if native:
                preparation = OwnerPreparation(spec, role_rows={"decode": 3, "mtp": 3, "prefill": 8},
                    manifest_sha256=hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest())
                preparations.append(preparation)
                engine = NS(comm=comm, rank=rank, tp=8, multi=None, _calls=0)
                preparation.prepare(engine, text, get_pair("int8"), page_tokens=128, device=base.device)
                weights.meta["owner_workspaces"] = preparation.workspaces
            result = MultiDecoder(weights, slots=1, capacity=260, logical_context=257, depth=2,
                confidence=.7, prefill_rows=8, kv_dtype="int8", graphs=False, keep=1, full_windows=1)
            if compact_target:
                target_owners.append(weights.meta["target_greedy_owner"])
            return weights, result, preparation

        torch_weights, torch_decoder, _ = decoder(False)
        eager_weights, eager, eager_preparation = decoder(True)
        combined_weights, combined, combined_preparation = decoder(True, compact_target=target_greedy)
        eager_preparation.adopt(eager_weights, eager)
        phase("capture/reference-before-adoption")
        graphs = RoundGraphs(combined, 1 << 30)
        graphs.capture()
        combined.graphs = graphs
        assert {key[1] for key in graphs.verify} == {1, 2, 3}
        assert {key[1] for key in graphs.mtp} == {1, 2, 3}
        addresses = {}
        for kind, source in (("verify", combined.buf), ("mtp", combined.mbuf)):
            private = graphs.bound.buffers[kind]
            assert private.pa.data_ptr() == source.pa.data_ptr()
            assert private.attn_o.data_ptr() == source.attn_o.data_ptr()
            addresses[kind] = (private.pa.data_ptr(), private.attn_o.data_ptr())
            assert not any(hasattr(lane.view, "graph_rows") for lane in graphs.bound.lanes.values())
        combined_preparation.adopt(combined_weights, combined)
        for kind, source in (("verify", combined.buf), ("mtp", combined.mbuf)):
            private = graphs.bound.buffers[kind]
            assert (private.pa.data_ptr(), private.attn_o.data_ptr()) == addresses[kind]
            assert source.owner_kernel_family == KERNEL_FAMILY and source.owner_graph_family == UNCAPTURED
        phase("adopt/native-on-same-pointers")

        prompt = ([4, 17, 3, 99, 33, 2, 7, 5, 6] * 15)[:129]
        for weights, dec in ((torch_weights, torch_decoder), (eager_weights, eager), (combined_weights, combined)):
            prefill(_slot(weights, dec.free[0], dec.buf, dec.mbuf, dec.pbuf, 260, 8), prompt, None)
        for rows, kept in ((1, 1), (2, 1), (3, 2)):
            phase(f"verify/rows={rows}/kept={kept}")
            tokens = [31 + row for row in range(rows)]
            target = eager.free[0]
            actual_state = combined.free[0]
            reference_state = torch_decoder.free[0]
            expected = compute(eager_weights, stage(eager_weights, eager.buf, [(target, tokens)]), eager.buf).clone()
            actual = graphs.verify_forward(stage(combined_weights, combined.buf, [(actual_state, tokens)], late=True))
            _equal(actual, expected)
            _equal(combined.buf.streams[:rows], eager.buf.streams[:rows])
            reference = compute(torch_weights,
                stage(torch_weights, torch_decoder.buf, [(reference_state, tokens)]), torch_decoder.buf).clone()
            assert torch.isfinite(reference).all() and torch.isfinite(expected).all()
            difference = (expected.float() - reference.float()).abs()
            print(json.dumps({"native_vs_torch_component": {"rank": rank, "rows": rows,
                "max_abs_logits": float(difference.max()), "mean_abs_logits": float(difference.mean()),
                "same_bits": bool(torch.equal(expected, reference)),
                "qualification": "measurement-only; no empirical tolerance or full-model quality claim"}}), flush=True)
            for weights, dec in ((torch_weights, torch_decoder), (eager_weights, eager), (combined_weights, combined)):
                commit(weights, dec.free[0], dec.buf, rows, kept)
            _committed(actual_state, target)

        for weights, dec in ((eager_weights, eager), (combined_weights, combined)):
            dec.free[0].reset(weights)
        for repetition in range(2):
            expected = Stream([4, 17, 3, 99, 33, 2, 7, 5, 6], 12, stop_eos=False)
            actual = Stream(list(expected.prompt), 12, stop_eos=False)
            eager.admit(expected)
            combined.admit(actual)
            for iteration in range(128):
                expected_done, actual_done = eager.round(), combined.round()
                assert expected.error is actual.error is None
                assert (actual.out, actual.drafts, actual.done) == (expected.out, expected.drafts, expected.done)
                _committed(actual.st, expected.st)
                eager.finish(expected_done)
                combined.finish(actual_done)
                if actual.done:
                    assert len(expected_done) == len(actual_done) == 1
                    break
            else:
                raise AssertionError("finite combined native-owner graph request did not complete")
            assert len(actual.out) == 12 and actual.cached == expected.cached
            if repetition:
                assert actual.cached > 0
            phase(f"request/completed/repetition={repetition}")
        assert graphs.bound.hits > 0
        if target_greedy:
            assert combined_weights.meta["target_greedy_owner"].completed > 0
        comm.ready("owner-piece-oracles-complete", every=1, timeout=60)
    except BaseException as primary:
        failed = True
        try:
            phase("primary/" + type(primary).__name__)
            traceback.print_exception(primary, limit=32, file=sys.stderr)
        except BaseException as error:
            diagnostic = error
        rollback(comm, primary, retire)
    else:
        retire()


@pytest.mark.parametrize("target_greedy", [False, True, "guard"])
def test_v100_tp8_native_owner_exact_pieces_match_native_eager_frames_and_commits(target_greedy):
    if torch.cuda.device_count() < 8 or any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(8)):
        pytest.skip("requires eight real SM70 GPUs and the supported source foundation")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = torch.multiprocessing.spawn(_worker, args=(port, target_greedy), nprocs=8, join=False)
    try:
        deadline = time.monotonic() + 240
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("combined native-owner exact-piece participant did not retire")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=10)
