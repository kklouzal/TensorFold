"""Real SM70 family/NCCL requests through the existing concurrent Scheduler."""

import multiprocessing
import queue
import socket
import threading
import time
import traceback
from types import SimpleNamespace

import pytest
import torch


def _requests(rank, world, port, outcomes):
    from test_flashnext_sm70_family import _small_weights

    from tensorfold.cuda.comm import NCCL
    from tensorfold.cuda.scheduler import Scheduler
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder

    owner, scheduler = NCCL(), None
    failed = False
    try:
        torch.cuda.set_device(rank)
        owner.open(rank, world, "127.0.0.1", port, timeout_s=90, key="tf_sm70_scheduler_uid")
        w = _small_weights(include_ple=True)
        width = w.cfg.vocab // world
        w.head.weight = w.head.weight[rank * width:(rank + 1) * width].contiguous()
        w.head.n = width
        w.comm = owner
        w.meta.update(world=world, vocab_offset=rank * width)
        prompts = ([4, 17, 3, 99, 33], [31, 18, 9, 14])
        serial = Engine(w, capacity=64, max_rows=3, prefill_rows=8, kv_dtype="fp16")
        expected = []
        for prompt in prompts:
            first = prefill(serial, prompt, None, mtp=False)
            expected.append(serial_decode(serial, first, 5, None).tokens)
        del serial
        decoder = MultiDecoder(w, slots=2, capacity=64, depth=2, prefill_rows=8,
                               kv_dtype="fp16", graphs=False, keep=0)
        engine = SimpleNamespace(_abort_comm=False, _closing=False, _lifecycle=threading.Condition())
        control = TensorParallelDecoder(decoder, engine)
        owner.ready("scheduler-before-requests", every=1, timeout=60)
        if rank:
            while True:
                body = control.receive()
                if body is None:
                    break
                control.apply(body)
        else:
            scheduler = Scheduler(control, max_streams=2).start()
            observed, failures = {}, []

            def request(index):
                try:
                    tokens = []
                    scheduler.submit(prompts[index], 5, None, False, lambda new: tokens.extend(new),
                                     stop_eos=False)
                    observed[index] = tokens
                except BaseException as error:
                    failures.append(error)

            clients = [threading.Thread(target=request, args=(index,)) for index in range(len(prompts))]
            for client in clients:
                client.start()
            for client in clients:
                client.join(60)
                assert not client.is_alive(), "concurrent request failed to retire"
            scheduler.close()
            scheduler = None
            assert failures == [], failures
            assert observed == dict(enumerate(expected))
            assert all(0 <= token < w.cfg.vocab for words in observed.values() for token in words)
            control.shutdown()
        assert decoder.live() == 0 and control.requests == {}
        owner.ready("scheduler-component-complete", every=1, timeout=60)
    except BaseException:
        failed = True
        outcomes.put((rank, traceback.format_exc()))
        raise
    finally:
        # A failed model operation poisons the rank cohort; no further model
        # operations are published while this process retires its native owner.
        if scheduler is not None and not failed:
            scheduler.close()
        owner.close(abort=failed)
    outcomes.put((rank, None))


@pytest.mark.parametrize("world", [4, 8])
def test_sm70_tp_scheduler_concurrent_requests_equal_serial(world):
    """Tiny half GDN/QSA/PLE/MoE integration; full checkpoint qualification is separate."""
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} visible CUDA GPUs")
    if any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(world)):
        pytest.skip("SM70 family integration")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    processes = [context.Process(target=_requests, args=(rank, world, port, outcomes)) for rank in range(world)]
    results = []
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 150
        while len(results) < world and time.monotonic() < deadline:
            try:
                results.append(outcomes.get(timeout=min(1, max(0.01, deadline - time.monotonic()))))
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
        assert len(results) == world, f"native participants did not retire: {results}"
        assert sorted(results) == [(rank, None) for rank in range(world)], results
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join(timeout=10)
        outcomes.close()
        outcomes.join_thread()
