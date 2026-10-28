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


def _batch_fixture_requests():
    from tensorfold.cuda.streams import Stream

    # At most28 single-piece prompt passes at8 rows, with at most3 committed
    # decode outputs per pass.96 keeps the first request alive until the eighth
    # can join without bypassing normal admission, prefill or State ownership.
    return [Stream(([4, 17, 3, 99, 33, 2, 7, 5, 6] * 5)[:9 + 4 * index],
                   96, stop_eos=False) for index in range(8)]

def _worker(rank, port, target_greedy=False, profile_directory=None):
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
    from tensorfold.families.qwen4_exp.cuda.prefix_policy import OWNER_GRAPH_PROFILE, OWNER_TARGET_GREEDY_PROFILE, OWNER_TARGET_GUARD_PROFILE, OWNER_TARGET_PRELUDE_PROFILE, OWNER_TARGET_DENSE_PROFILE, OWNER_BATCH_PRELUDE_PROFILE
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    from test_sm70_unique_piece_graphs_tp8 import _equal, _committed

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    from test_flashnext_sm70_mtp import _weights_with_stock_mtp_protocol

    family_ops._NATIVE_FAMILY, family_ops._NATIVE_ATTENTION = True, False
    comm, failed, graphs, diagnostic = NCCL(), False, None, None
    preparations, target_owners, input_contexts = [], [], []
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
        finish(context.close for context in reversed(input_contexts))
        finish((lambda: comm.close(abort=failed), faulthandler.cancel_dump_traceback_later, diagnostic_failure))

    try:
        phase("communicator/open")
        comm.open(rank, 8, "127.0.0.1", port, timeout_s=90, key="tf_native_owner_exact_pieces")
        base = _weights_with_stock_mtp_protocol(head_dim=256)
        if target_greedy in ("bound-prelude", "bound-dense", "dense-batch-prelude"):
            from dataclasses import replace

            # A real model has repeated attention layers sharing the issued
            # coordinate frame. One layer cannot expose a version bump before
            # the next layer validates that same frame.
            base.layers.append(replace(base.layers[1], index=base.cfg.layers))
            base.cfg.layers += 1
            base.cfg.layer_types = [*base.cfg.layer_types, "attention"]
            base.mtp = replace(base.mtp, layer=replace(base.mtp.layer, index=base.cfg.layers))
        if target_greedy in ("bound-dense", "dense-batch-prelude"):
            from dataclasses import fields, is_dataclass, replace
            from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense

            converted = {}

            def actual_target_dense(value):
                # Only this new case replaces synthetic target Linear adapters
                # with actual owners on both sides. Memoization preserves the
                # head/draft-head and repeated-layer owner aliases; tensors and
                # embeddings retain their original objects and storage.
                if id(value) in converted:
                    return converted[id(value)]
                if (getattr(value, "kernel", None) == "gguf"
                        and isinstance(getattr(value, "weight", None), torch.Tensor)
                        and value.weight.ndim == 2):
                    result = GGUFDense(value.weight)
                elif is_dataclass(value):
                    children = {item.name: actual_target_dense(getattr(value, item.name))
                                for item in fields(value) if item.init}
                    result = (value if all(child is getattr(value, name) for name, child in children.items())
                              else replace(value, **children))
                elif isinstance(value, list):
                    children = [actual_target_dense(item) for item in value]
                    result = value if all(child is old for child, old in zip(children, value)) else children
                elif isinstance(value, tuple):
                    children = tuple(actual_target_dense(item) for item in value)
                    result = value if all(child is old for child, old in zip(children, value)) else children
                else:
                    return value
                converted[id(value)] = result
                return result

            original = base
            base = actual_target_dense(base)
            assert base.embed is original.embed
            assert base.mtp.fc_e is original.mtp.fc_e and base.mtp.fc_h is original.mtp.fc_h
            assert base.head.weight is original.head.weight
            assert base.layers[1].attn.proj is base.layers[2].attn.proj
            assert isinstance(base.head, GGUFDense)
            assert isinstance(base.layers[1].attn.proj, GGUFDense)
        width = base.cfg.vocab // 8
        base.head.weight = base.head.weight[rank * width:(rank + 1) * width].contiguous()
        base.head.n, base.comm = width, comm
        base.meta.update(world=8, rank=rank, vocab_offset=rank * width, full=NS(kv_heads=2),
            history_layout="unique", indexer_history="gpu", attention_degree=1,
            capture_ready=True, unique_capture_contract=UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT,
            sm70_profile=OWNER_GRAPH_PROFILE)
        if target_greedy in ("bound-prelude", "bound-dense", "dense-batch-prelude"):
            from tensorfold.families.qwen4_exp.position_profile import (
                resolve_position_profile, prepare_position_vectors, bind_position_profile,
            )

            # The parent owns one real shared source path through all worker
            # retirement. Resolved identity includes source paths, so separate
            # rank-local temporary directories are different authorities.
            assert profile_directory is not None
            profile = resolve_position_profile(profile_directory, requested_context=257)
            prepared = prepare_position_vectors(profile, mtp_active=True)
            base.cfg.rope = profile.target.rope
            base.inv_freq = prepared.target_cpu.to(base.device)
            bind_position_profile(base, profile, mtp_active=True, prepared=prepared)
            base.meta.update(position_profile_identity=profile.identity, position_profile_epoch=profile.epoch)
            phase("position-profile/" + profile.identity)
            assert base.positions is not None and base.positions.mtp is not None

        text = {"hidden_size": base.cfg.hidden, "num_attention_heads": 24, "num_key_value_heads": 2,
                "head_dim": 256, "indexer_budget": base.cfg.index_budget,
                "indexer_compress_ratio": base.cfg.index_ratio}
        spec = dict(kernel_family=KERNEL_FAMILY, graph_family=UNCAPTURED,
                    collective_family=FULL_COLLECTIVE, graph_budget_bytes=0, trial=True)

        slots = 8 if target_greedy == "dense-batch-prelude" else 1

        def decoder(native, *, compact_target=False):
            weights = copy.copy(base)
            weights.meta = dict(base.meta)
            if compact_target:
                from tensorfold.families.qwen4_exp.cuda.target_greedy import POLICY

                weights.meta.update(sm70_profile=OWNER_BATCH_PRELUDE_PROFILE if compact_target == "dense-batch-prelude" else OWNER_TARGET_DENSE_PROFILE if compact_target == "bound-dense" else OWNER_TARGET_PRELUDE_PROFILE if compact_target in ("prelude", "bound-prelude", "bound-dense", "dense-batch-prelude") else
                    OWNER_TARGET_GUARD_PROFILE if compact_target == "guard" else OWNER_TARGET_GREEDY_PROFILE,
                    sm70_target_greedy_trial={"policy": POLICY}, target_greedy_owner=None)
                if compact_target in ("guard", "prelude", "bound-prelude", "bound-dense", "dense-batch-prelude"):
                    weights.meta["sm70_native_guard_trial"] = True
                if compact_target in ("prelude", "bound-prelude", "bound-dense", "dense-batch-prelude"):
                    weights.meta["sm70_owner_prelude_zero"] = torch.zeros(1, dtype=torch.int32, device=weights.device)
                    weights.meta["sm70_owner_prelude_submitted_frames"] = 0
                    weights.meta["sm70_owner_prelude_submitted_mtp_frames"] = 0
                    weights.meta["sm70_owner_prelude_mtp_counter_overflowed"] = False
                    if compact_target == "dense-batch-prelude":
                        weights.meta.update(sm70_owner_batch_prelude_trial=True,
                            sm70_owner_prelude_submitted_batches=0, sm70_owner_prelude_submitted_mtp_batches=0,
                            sm70_owner_prelude_batch_counter_overflowed=False,
                            sm70_owner_prelude_max_batch_segments=0)
                    torch.cuda.current_stream(weights.device).synchronize()
            if weights.positions is not None:
                from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation

                # The Engine issues one shared coordinate activation before
                # this actual installer. Use a deterministic fixture identity,
                # distinct per model owner and equal across all eight ranks.
                weights.meta["coordinate_profile_identity"] = hashlib.sha256(
                    (weights.positions.profile.identity + f"/fixture-model/{len(input_contexts)}").encode()).hexdigest()
                context = install_input_preparation(weights)
                phase("input-profile/" + context.identity)
                input_contexts.append(context)
                assert context is weights.meta["input_preparation_context"]
            preparation = None
            if native:
                preparation = OwnerPreparation(spec, role_rows={"decode": slots * 3, "mtp": slots * 3, "prefill": 8},
                    manifest_sha256=hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest())
                preparations.append(preparation)
                engine = NS(comm=comm, rank=rank, tp=8, multi=None, _calls=0)
                preparation.prepare(engine, text, get_pair("int8"), page_tokens=128, device=base.device)
                weights.meta["owner_workspaces"] = preparation.workspaces
            result = MultiDecoder(weights, slots=slots, capacity=260, logical_context=257, depth=2,
                confidence=.7, prefill_rows=8, kv_dtype="int8", graphs=False, keep=1, full_windows=slots)
            if compact_target:
                target_owners.append(weights.meta["target_greedy_owner"])
            return weights, result, preparation

        torch_weights, torch_decoder, _ = decoder(False)
        eager_weights, eager, eager_preparation = decoder(
            True, compact_target="bound-dense" if target_greedy == "dense-batch-prelude" else False)
        combined_weights, combined, combined_preparation = decoder(True, compact_target=target_greedy)
        if target_greedy in ("bound-dense", "dense-batch-prelude"):
            assert torch_weights.head is eager_weights.head is combined_weights.head
            assert torch_weights.embed is eager_weights.embed is combined_weights.embed
            assert torch_weights.mtp is eager_weights.mtp is combined_weights.mtp
            assert combined_weights.head.n == width and combined_weights.head.weight.shape[0] == width
        eager_preparation.adopt(eager_weights, eager)
        from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
        from tensorfold.cuda.sm70 import gguf_ops

        native_calls = [0]
        coalesced_calls = [0]
        previous_candidate = [None]
        import weakref
        from tensorfold.families.qwen4_exp.cuda.kv_status import KVStatusCohort

        ordinary_candidates = KVStatusCohort.read_candidates

        def counted_candidates(owner, *args, **kwargs):
            if previous_candidate[0] is not None:
                assert previous_candidate[0]() is None  # One escaped payload, retired before the next.
            coalesced_calls[0] += 1
            # Explicit comparison-mode API: exercise split IO in the real-model
            # oracle while the production default remains the packed incumbent.
            kwargs.setdefault("layout", "split")
            packet = ordinary_candidates(owner, *args, **kwargs)
            previous_candidate[0] = weakref.ref(packet)
            return packet

        if target_greedy == "dense-batch-prelude":
            KVStatusCohort.read_candidates = counted_candidates
        ordinary_dense32 = gguf_ops.dense32

        def counted_dense32(*args, **kwargs):
            native_calls[0] += 1
            return ordinary_dense32(*args, **kwargs)

        if target_greedy in ("bound-dense", "dense-batch-prelude"):
            gguf_ops.dense32 = counted_dense32

        def select_dense(weights):
            # Fixture owners share immutable weights; switch only the dispatch
            # around serialized calls, leaving eager controls on reference math.
            GGUFDense.implementation = ("native32" if target_greedy == "dense-batch-prelude"
                                       or target_greedy == "bound-dense" and weights is combined_weights else "reference")

        select_dense(combined_weights)
        phase("capture/reference-before-adoption")
        graphs = RoundGraphs(combined, 1 << 30)
        graphs.capture()
        capture_native_calls = native_calls[0]
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
            select_dense(weights)
            prefill(_slot(weights, dec.free[0], dec.buf, dec.mbuf, dec.pbuf, 260, 8), prompt, None)
        for rows, kept in ((1, 1), (2, 1), (3, 2)):
            phase(f"verify/rows={rows}/kept={kept}")
            tokens = [31 + row for row in range(rows)]
            target = eager.free[0]
            actual_state = combined.free[0]
            reference_state = torch_decoder.free[0]
            select_dense(eager_weights)
            expected = compute(eager_weights, stage(eager_weights, eager.buf, [(target, tokens)]), eager.buf).clone()
            select_dense(combined_weights)
            actual = graphs.verify_forward(stage(combined_weights, combined.buf, [(actual_state, tokens)], late=True))
            _equal(actual, expected)
            _equal(combined.buf.streams[:rows], eager.buf.streams[:rows])
            select_dense(torch_weights)
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
                select_dense(eager_weights)
                expected_done = eager.round()
                select_dense(combined_weights)
                actual_done = combined.round()
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
        if target_greedy in ("bound-dense", "dense-batch-prelude"):
            assert capture_native_calls > 0 and native_calls[0] > capture_native_calls
            print(json.dumps({"rank": rank, "dense_native_submissions": native_calls[0],
                              "dense_native_capture_submissions": capture_native_calls,
                              "dense_native_after_capture_submissions": native_calls[0] - capture_native_calls}), flush=True)
        if target_greedy == "dense-batch-prelude":
            assert coalesced_calls[0] == 0
            assert combined_weights.meta["sm70_owner_prelude_submitted_batches"] == 0
            assert combined_weights.meta["sm70_owner_prelude_submitted_mtp_batches"] == 0
            references = _batch_fixture_requests()
            actuals = [Stream(list(stream.prompt), stream.count, stop_eos=False) for stream in references]
            for expected, actual in zip(references, actuals):
                eager.admit(expected)
                combined.admit(actual)
            for iteration in range(128):
                select_dense(eager_weights)
                expected_done = eager.round()
                select_dense(combined_weights)
                actual_done = combined.round()
                for expected, actual in zip(references, actuals):
                    assert expected.error is actual.error is None
                    assert (actual.out, actual.drafts, actual.done) == (expected.out, expected.drafts, expected.done)
                    if actual.st is not None and expected.st is not None:
                        _committed(actual.st, expected.st)
                eager.finish(expected_done)
                combined.finish(actual_done)
                if all(stream.done for stream in actuals):
                    break
            else:
                raise AssertionError("finite eight-session batched owner prelude did not complete")
            assert all(len(stream.out) == 96 for stream in actuals)
            assert combined_weights.meta["sm70_owner_prelude_submitted_batches"] > 0
            assert combined_weights.meta["sm70_owner_prelude_max_batch_segments"] == 8
            assert combined_weights.meta["sm70_owner_prelude_submitted_mtp_batches"] > 0
            assert not combined_weights.meta["sm70_owner_prelude_batch_counter_overflowed"]
            print(json.dumps({"rank": rank, "batch_counters": {name: value for name, value in combined_weights.meta.items()
                if name in ("sm70_owner_prelude_submitted_batches", "sm70_owner_prelude_submitted_mtp_batches",
                            "sm70_owner_prelude_max_batch_segments")}}), flush=True)
            assert coalesced_calls[0] > 0
            assert previous_candidate[0]() is None
            print(json.dumps({"rank": rank, "coalesced_candidate_readbacks": coalesced_calls[0]}), flush=True)
            phase("eight-session-native-batch-completed")
        assert graphs.bound.hits > 0
        if target_greedy in ("prelude", "bound-prelude", "bound-dense", "dense-batch-prelude"):
            assert combined_weights.meta["sm70_owner_prelude_submitted_frames"] > 0
            assert combined_weights.meta["sm70_owner_prelude_submitted_mtp_frames"] > 0
            assert combined_weights.meta["sm70_owner_prelude_zero"].tolist() == [0]
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



def _retire_fixture_context_outcomes(context, directory, primary=None):
    """Retain shared source if a killed child has not positively retired."""
    import shutil
    from tensorfold.cleanup import raise_failures

    errors = []
    if context is None:
        errors.append(RuntimeError(f"worker startup ownership is uncertain; source retained at {directory}"))
    else:
        for process in context.processes:
            try:
                if process.is_alive():
                    process.kill()
            except BaseException as error:
                errors.append(error)
        for process in context.processes:
            try:
                process.join(timeout=10)
            except BaseException as error:
                errors.append(error)
        try:
            retired = all(not process.is_alive() and process.exitcode is not None for process in context.processes)
        except BaseException as error:
            errors.append(error)
            retired = False
        if retired:
            try:
                shutil.rmtree(directory)
            except BaseException as error:
                errors.append(error)
        else:
            errors.append(RuntimeError(f"child retirement is uncertain; source retained at {directory}"))
    raise_failures(primary, errors)


def _retire_fixture_context(context, directory, primary=None):
    """Cold outcome preparation cannot replace the worker's named primary."""
    try:
        _retire_fixture_context_outcomes(context, directory, primary)
    except BaseException as secondary:
        if primary is None or secondary is primary:
            raise
        # No new outcome container/message is allocated on this last-resort
        # failure cut. The traceback retains context and source ownership.
        BaseException.__context__.__set__(primary, secondary)
        raise primary

@pytest.mark.parametrize("target_greedy", [False, True, "guard", "prelude", "bound-prelude", "bound-dense", "dense-batch-prelude"])
def test_v100_tp8_native_owner_exact_pieces_match_native_eager_frames_and_commits(target_greedy):
    if torch.cuda.device_count() < 8 or any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(8)):
        pytest.skip("requires eight real SM70 GPUs and the supported source foundation")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    import tempfile
    from pathlib import Path

    directory = tempfile.mkdtemp(prefix="tf-owner-prelude-shared-")
    context = None
    try:
        config = {"model_type": "qwen4_exp", "text_config": {
            "head_dim": 256, "hidden_size": 256, "num_attention_heads": 24,
            "max_position_embeddings": 262144,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                "partial_rotary_factor": .25, "mrope_interleaved": True,
                "mrope_section": [11, 11, 10]}}}
        (Path(directory) / "config.json").write_text(json.dumps(config))
        context = torch.multiprocessing.spawn(_worker, args=(port, target_greedy, directory), nprocs=8, join=False)
        deadline = time.monotonic() + 240
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("combined native-owner exact-piece participant did not retire")
    except BaseException as primary:
        _retire_fixture_context(context, directory, primary)
    else:
        _retire_fixture_context(context, directory)
