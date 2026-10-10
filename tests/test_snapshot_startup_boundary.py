"""Actual startup owner flow and probe callbacks without a numerical runtime."""

import ast
import __future__
import gc
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from tensorfold.engine.snapshot_codec import DTYPES
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.prompt_memory import PromptMemory
from tensorfold.server.request_limits import RequestLimit, optional_limit


class ProbeRuntime:
    def __init__(self):
        self.clears = 0

    def clear_cache(self):
        self.clears += 1

    def get_active_memory(self):
        return 0


class ProbeControls(unittest.TestCase):
    def run_probe(self, failure=False):
        runtime, caches, observed = ProbeRuntime(), [], []
        model = SimpleNamespace()
        primary = KeyboardInterrupt("observer interrupted")

        class Cache:
            pass

        class Engine:
            prefill_guard = object()

            def prefill_prefix(self, tokens, **kwargs):
                value = Cache()
                caches.append(weakref.ref(value))
                return [value]

        def observe(cache):
            observed.append(type(cache[0]))
            if failure:
                raise primary

        memory = PromptMemory(1 << 20, model, runtime=runtime, chunk_rows=4, overhead_bytes=0, profile_observer=observe)
        engine = Engine()
        guard = engine.prefill_guard
        if failure:
            with self.assertRaises(KeyboardInterrupt) as caught:
                memory.profile_probe(engine, [1, 2, 3])
            self.assertIs(caught.exception, primary)
        else:
            memory.profile_probe(engine, [1, 2, 3])
            gc.collect()
            self.assertEqual(len(observed), 3)
            self.assertTrue(all(ref() is None for ref in caches))
        self.assertIs(engine.prefill_guard, guard)
        self.assertIsNone(memory._probe_base)
        self.assertEqual(runtime.clears, len(observed) + 1)

    def test_existing_three_complete_probes_observe_without_retaining_cache_storage(self):
        self.run_probe()

    def test_observer_interruption_restores_guard_and_preserves_primary(self):
        self.run_probe(failure=True)


class StartupControls(unittest.TestCase):
    def test_actual_app_initialization_registers_before_loader_scheduler_and_spill_callers(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "src/tensorfold/server/app.py").read_text())
        app = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChatApp")
        initialize = next(n for n in app.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        events, references = [], []
        core, numpy = ModuleType("mlx.core"), ModuleType("numpy")
        for name, _, _ in DTYPES:
            setattr(core, name, name)

        class Array:
            def __init__(self, shape, dtype="float32"):
                self.shape, self.dtype = shape, dtype

        core.array = Array
        numpy.ndarray = type("Host", (), {})
        numpy.dtype = lambda name: name
        module = ModuleType("mlx_lm.models.cache")

        class KVCache:
            step = 256

            def __init__(self):
                self.keys = self.values = None
                self.offset = 0

        KVCache.__module__ = module.__name__
        KVCache.__qualname__ = "KVCache"
        module.KVCache = KVCache

        class Model:
            vision = None
            args = SimpleNamespace(vocab_size=256)
            layers = [object()]

            def make_cache(self):
                events.append("prototype")
                return [KVCache()]

        class Engine:
            prefill_step = 4
            prefill_plan = SimpleNamespace(step=4)

            def __init__(self, model, **kwargs):
                self.max_draft = kwargs["max_draft"]
                events.append("engine")

            def cache_nbytes(self, cache):
                return 0

            def copy_single_cache(self, cache):
                return cache

            def prefill_prefix(self, tokens, **kwargs):
                events.append("profile")
                cache = KVCache()
                cache.offset = len(tokens)
                cache.keys = cache.values = Array((1, 2, 256, 4))
                references.append(weakref.ref(cache.keys))
                return [cache]

        class Scheduler:
            def __init__(self, engine, **kwargs):
                events.append("scheduler")
                self.received = kwargs
                self.session_dir = kwargs["session_dir"]
                self.model_id = kwargs["model_id"]

            def start(self):
                events.append("start")

        ns = {
            "optional_limit": optional_limit,
            "RequestLimit": RequestLimit,
            "LaneEngine": Engine,
            "Scheduler": Scheduler,
            "CheckpointStore": CheckpointStore,
            "Path": Path,
            "time": time,
            "threading": __import__("threading"),
            "DEFAULT_LIMITS": object(),
            "ImageLimits": lambda **kwargs: object(),
            "served_model_ids": lambda name, aliases: [name],
            "think_markers": lambda tokenizer: (),
            "eos_ids_of": lambda tokenizer: frozenset(),
            "template_late_system": lambda tokenizer: False,
            "SuffixLookupProposer": lambda **kwargs: object(),
            "spill_conversation": lambda *args, **kwargs: events.append(("spill", kwargs)),
            "concurrency": lambda *args: self.fail("unrequested concurrency probe"),
        }
        exec(
            compile(
                ast.Module([initialize], []),
                "<actual ChatApp.__init__>",
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            ns,
        )

        class Owner:
            def _warm_known_blocks(self, *args):
                events.append("warm")

        Owner.__init__ = ns["__init__"]
        mlx = ModuleType("mlx")
        mlx.core = core
        replacements = {"mlx": mlx, "mlx.core": core, "numpy": numpy, "mlx_lm.models.cache": module}
        prefix_name = "tensorfold.engine.prefix_snapshots"
        spec = importlib.util.spec_from_file_location(prefix_name, root / "src/tensorfold/engine/prefix_snapshots.py")
        prefix = importlib.util.module_from_spec(spec)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, replacements):
            spec.loader.exec_module(prefix)
            with patch.dict(sys.modules, {prefix_name: prefix}):
                # Relative import reads the package attribute when already
                # bound, so replace that exact owner for this bounded fixture.
                import tensorfold.engine as package

                with patch.object(package, "prefix_snapshots", prefix, create=True):
                    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3])
                    owner = Owner(
                        Model(),
                        tokenizer,
                        served_name="fixture",
                        engine_factory=Engine,
                        context_window=64,
                        checkpoint_slots=1,
                        snapshot_dir=Path(directory),
                        model_id="content-v1:fixture",
                        memory_budget_bytes=None,
                        max_draft=4,
                        spill_bytes=1024,
                    )
                    registry = owner.snapshot_registry
                    self.assertIs(owner.scheduler.received["snapshot_registry"], registry)
                    self.assertIs(owner.scheduler.received["snapshot_codec"], owner.snapshot_codec)
                    self.assertEqual(registry.token_limit, 64)
                    self.assertEqual(len(registry.layers[0].token_invariants), 1)
                    owner.checkpoints.on_evict(SimpleNamespace())
        self.assertLess(events.index("prototype"), events.index("profile"))
        self.assertLess(events.index("profile"), events.index("scheduler"))
        self.assertLess(events.index("scheduler"), events.index("start"))
        self.assertTrue(all(ref() is None for ref in references))
        self.assertEqual(events[-1][0], "spill")
        self.assertIs(events[-1][1]["registry"], registry)


if __name__ == "__main__":
    unittest.main()
