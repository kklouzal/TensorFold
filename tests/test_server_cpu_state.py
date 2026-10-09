"""Snapshot lifetime, explicit sampling seeds, and bounded headless telemetry."""
import copy
import gc
import sys
import threading
from types import ModuleType
import unittest
from unittest.mock import patch
import weakref

from tensorfold.server import responses
from tensorfold.server.live import Meter
from tensorfold.server.request_options import RequestOptions


class StoreTests(unittest.TestCase):
    def response(self, text="original"):
        return {"id": "r", "output": [{"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": text}]}]}

    def test_owned_and_returned_snapshots_do_not_alias(self):
        store = responses.Store()
        original = self.response()
        store.put(original, [])
        original["output"][0]["content"][0]["text"] = "caller mutation"
        returned = store.get("r")
        returned["output"][0]["content"][0]["text"] = "reader mutation"
        self.assertEqual(store.get("r"), self.response())
        self.assertIsNone(store.get("missing"))

    def test_delete_progresses_while_snapshot_copy_is_in_flight(self):
        store = responses.Store()
        store.put(self.response(), [])
        owned = store.entries["r"].response
        copying, release, deleted = threading.Event(), threading.Event(), threading.Event()
        read, failures = [], []
        original_copy = copy.deepcopy

        def paused(value):
            if value is owned:
                copying.set()
                if not release.wait(2):
                    raise TimeoutError("snapshot fixture copy was not released")
            return original_copy(value)

        def reader():
            try:
                read.append(store.get("r"))
            except BaseException as error:
                failures.append(error)

        def remover():
            if store.delete("r"):
                deleted.set()

        with patch.object(responses.copy, "deepcopy", paused):
            worker = threading.Thread(target=reader)
            worker.start()
            remover_thread = None
            try:
                self.assertTrue(copying.wait(1))
                remover_thread = threading.Thread(target=remover)
                remover_thread.start()
                self.assertTrue(deleted.wait(1), "delete held up by response copying")
            finally:
                release.set()
                worker.join(2)
                if remover_thread is not None:
                    remover_thread.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(failures)
        self.assertEqual(read, [self.response()])
        self.assertIsNone(store.get("r"))

    def test_retrieved_snapshot_survives_eviction_and_replacement(self):
        store = responses.Store(limit=1)
        store.put(self.response(), [])
        old = store.get("r")
        store.put({"id": "new", "output": []}, [])
        self.assertIsNone(store.get("r"))
        store.put(self.response("replacement"), [])
        self.assertEqual(old, self.response())
        self.assertEqual(store.get("r"), self.response("replacement"))

    def test_store_is_created_once_and_owner_is_not_retained(self):
        owner = type("Owner", (), {})()
        reference = weakref.ref(owner)
        with patch.object(responses, "Store", wraps=responses.Store) as factory:
            first = responses.store_for(owner)
            self.assertIs(responses.store_for(owner), first)
            self.assertEqual(factory.call_count, 1)
        del owner
        gc.collect()
        self.assertIsNone(reference())

    def test_replacement_budget_counts_only_current_owned_entry(self):
        store = responses.Store()
        store.put(self.response(), [])
        old = store.get("r")
        store.put({"id": "second", "output": []}, [])
        order = list(store.entries)
        store.put(self.response("longer replacement content"), [])
        self.assertEqual(store.bytes, sum(entry.size for entry in store.entries.values()))
        self.assertEqual(list(store.entries), order)
        self.assertEqual(old, self.response())

    def test_replacement_evicts_by_real_size_and_preserves_existing_order(self):
        store = responses.Store()
        store.put(self.response(), [])
        store.put({"id": "second", "output": []}, [])
        # Replacing the oldest entry with a larger response keeps its position,
        # so the actual byte budget evicts it first.
        store.max_bytes = store.bytes
        store.put(self.response("replacement " * 100), [])
        self.assertIsNone(store.get("r"))
        self.assertEqual(list(store.entries), ["second"])
        self.assertEqual(store.bytes, store.entries["second"].size)


class SamplingOptionsTests(unittest.TestCase):
    def resolve(self, fields, defaults=None):
        module = ModuleType("tensorfold.engine.exact_sampling")
        module.Sampling = lambda **values: values
        calls = []
        module.seed_for = lambda tokens: (calls.append(list(tokens)), 41)[1]
        owner = type("Options", (RequestOptions,), {})()
        owner.default_sampling = defaults or {}
        with patch.dict(sys.modules, {module.__name__: module}):
            result = owner._resolve_sampling(fields, .7, [1, 2, 3])
        return result, calls

    def test_explicit_zero_and_negative_seed_never_hash_prompt(self):
        for seed in (0, -7, "123"):
            result, calls = self.resolve({"temperature": .7, "seed": seed})
            self.assertEqual(result["seed"], int(seed))
            self.assertEqual(calls, [])

    def test_default_seed_is_also_explicit_and_none_falls_back(self):
        result, calls = self.resolve({"temperature": .7, "seed": None}, {"seed": 0})
        self.assertEqual(result["seed"], 0)
        self.assertEqual(calls, [])
        result, calls = self.resolve({"temperature": .7, "seed": None})
        self.assertEqual(result["seed"], 41)
        self.assertEqual(calls, [[1, 2, 3]])

    def test_greedy_resolution_does_not_hash(self):
        result, calls = self.resolve({"temperature": 0})
        self.assertIsNone(result)
        self.assertEqual(calls, [])


class MeterTests(unittest.TestCase):
    def test_rate_samples_time_and_events_in_one_locked_snapshot(self):
        reading, release, adding, added = (threading.Event() for _ in range(4))
        now, results, errors = [0.0], [], []

        def clock():
            self.assertTrue(meter._lock.locked())
            value = now[0]
            if threading.current_thread().name == "paused-meter-rate":
                reading.set()
                if not release.wait(2):
                    raise TimeoutError("paused clock fixture was not released")
            return value

        meter = Meter(window=2, clock=clock)
        meter.add(10)

        def read():
            try:
                results.append(meter.rate())
            except BaseException as error:
                errors.append(error)

        def add():
            adding.set()
            meter.add(4)
            added.set()

        reader = threading.Thread(target=read, name="paused-meter-rate")
        writer = threading.Thread(target=add)
        reader.start()
        try:
            self.assertTrue(reading.wait(1))
            now[0] = 3
            writer.start()
            self.assertTrue(adding.wait(1))
            self.assertFalse(added.wait(.02))
        finally:
            release.set()
            reader.join(2)
            if writer.ident is not None:
                writer.join(2)
        self.assertFalse(reader.is_alive())
        self.assertFalse(writer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [5])
        self.assertEqual(meter.rate(), 2)

    def test_prune_preserves_inclusive_exact_window_boundary(self):
        now = [0.0]
        meter = Meter(window=2, clock=lambda: now[0])
        meter.add(10)
        now[0] = 2
        meter.add(4)
        self.assertEqual(meter.rate(), 7)
        now[0] = 2.0001
        meter.add(2)
        self.assertEqual(meter.rate(), 3)
        self.assertEqual(len(meter._events), 2)
        now[0] = 10
        self.assertEqual(meter.rate(), 0)

    def test_headless_adds_release_old_events_without_rate_calls(self):
        now = [0.0]
        meter = Meter(window=2, clock=lambda: now[0])
        for second in range(10000):
            now[0] = second
            meter.add(3)
        self.assertEqual(len(meter._events), 3)
        self.assertEqual(meter.rate(), 4.5)

    def test_nonpositive_adds_do_not_read_the_clock(self):
        meter = Meter(clock=lambda: self.fail("clock read for no tokens"))
        meter.add(0)
        meter.add(-1)
        self.assertEqual(len(meter._events), 0)

    def test_concurrent_add_and_rate_preserve_every_token(self):
        meter = Meter(clock=lambda: 0)
        barrier = threading.Barrier(3)

        def writer():
            barrier.wait(timeout=1)
            for _ in range(1000):
                meter.add(2)
                meter.rate()

        workers = [threading.Thread(target=writer) for _ in range(2)]
        for worker in workers:
            worker.start()
        barrier.wait(timeout=1)
        for worker in workers:
            worker.join(3)
            self.assertFalse(worker.is_alive())
        self.assertEqual(meter.rate(), 2000)


if __name__ == "__main__":
    unittest.main()
