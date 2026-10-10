"""CLI ownership/configuration boundaries without importing accelerator runtimes."""

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold import cli
from tensorfold.serve_options import check_numbers
from tensorfold.server.request_limits import RequestLimit, optional_limit
from tensorfold.server import live
from tensorfold.thread_work import ThreadWork


ROOT = Path(__file__).resolve().parents[1]


def app_close():
    path = ROOT / "src/tensorfold/server/app.py"
    tree = ast.parse(path.read_bytes())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChatApp")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "close")
    namespace = {"threading": threading}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["close"]


class OpaqueFailure(BaseException):
    def add_note(self, text):
        raise AssertionError("foreign note hook invoked")

    def __repr__(self):
        raise AssertionError("foreign exception formatting invoked")


class Ownership(unittest.TestCase):
    def test_constructor_registers_partial_owner_before_fallible_stages(self):
        path = ROOT / "src/tensorfold/server/app.py"
        tree = ast.parse(path.read_bytes())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChatApp")
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        selected = ast.ClassDef(name="PartialApp", bases=[], keywords=[], decorator_list=[], body=[init])
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), selected],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        for phase in ("engine", "scheduler", "start", "warmup"):
            order, owners = [], []
            engine = SimpleNamespace(
                drain=lambda: order.append("drain"),
                reset=lambda: order.append("reset"),
                copy_single_cache=lambda value: value,
                cache_nbytes=lambda _: 0,
            )
            scheduler = SimpleNamespace(
                start=lambda: order.append("start"),
                stop=lambda **_: order.append("stop"),
                _thread=SimpleNamespace(is_alive=lambda: False),
                session_dir=None,
                cancel=lambda _: order.append("cancel"),
            )

            def make_engine(*a, **k):
                self.assertTrue(owners)
                if phase == "engine":
                    raise OpaqueFailure()
                return engine

            def make_scheduler(*a, **k):
                if phase == "scheduler":
                    raise OpaqueFailure()
                if phase == "start":
                    scheduler.start = lambda: (_ for _ in ()).throw(OpaqueFailure())
                return scheduler

            ns = {
                "threading": threading,
                "LaneEngine": make_engine,
                "Scheduler": make_scheduler,
                "served_model_ids": lambda name, aliases: [name],
                "think_markers": lambda _: ("", ""),
                "eos_ids_of": lambda _: [],
                "template_late_system": lambda _: False,
                "DEFAULT_LIMITS": object(),
                "CheckpointStore": lambda *a, **k: SimpleNamespace(),
                "concurrency": lambda *a, **k: None,
                "RequestLimit": RequestLimit,
                "optional_limit": optional_limit,
                "Path": Path,
                "time": __import__("time"),
            }
            exec(compile(module, str(path), "exec"), ns)
            app_type = ns["PartialApp"]

            def warm(self, *unused):
                self.warmup_work = ThreadWork(lambda: None)
                self.warmup_thread = threading.Thread(target=self.warmup_work.run)
                self.warmup_cancellation = object()
                raise OpaqueFailure()

            app_type._warm_known_blocks = warm
            snapshots = ModuleType("tensorfold.engine.prefix_snapshots")
            snapshots.load_snapshots = lambda *a, **k: []
            from tensorfold.engine.snapshot_codec import DTYPES
            from tensorfold.engine.snapshot_registry import LayerSchema, Registry

            class Cache:
                def __init__(self):
                    self.offset = 0

            registry = Registry(
                (LayerSchema(Cache(), {}, {"offset": (0, 1)}),),
                token_limit=1,
                token_id_limit=2,
                tensor_byte_limit=1024,
                sizes={"U8": 1},
            )
            mlx, core, numpy = ModuleType("mlx"), ModuleType("mlx.core"), ModuleType("numpy")
            mlx.core = core
            core.array = type("OpaqueArray", (), {})
            for name, _, _ in DTYPES:
                setattr(core, name, name)
            numpy.ndarray = type("OpaqueHostArray", (), {})
            numpy.dtype = lambda name: name
            with patch.dict(sys.modules, {snapshots.__name__: snapshots, "mlx": mlx, "mlx.core": core, "numpy": numpy}):
                with self.assertRaises(OpaqueFailure):
                    app_type(
                        object(),
                        object(),
                        served_name="fixture",
                        engine_factory=make_engine,
                        checkpoint_slots=1 if phase == "warmup" else 0,
                        snapshot_dir=Path("unused") if phase == "warmup" else None,
                        snapshot_registry=registry if phase == "warmup" else None,
                        startup_owner=owners.append,
                    )
            self.assertEqual(len(owners), 1)
            partial = owners[0]
            self.assertFalse(partial._startup_complete)
            app_close()(partial)
            self.assertEqual(
                order,
                {
                    "engine": [],
                    "scheduler": ["drain", "reset"],
                    "start": ["stop"],
                    "warmup": ["start", "cancel", "stop"],
                }[phase],
            )
            if hasattr(partial, "scheduler"):
                self.assertIsNone(partial.scheduler.on_stop)

    def test_success_and_independent_cleanup_failure_order(self):
        for failed in (None, "line", "server"):
            order = []
            failure = OpaqueFailure()

            def operation(name):
                order.append(name)
                if name == failed:
                    raise failure

            owners = {
                name: SimpleNamespace(**{method: lambda n=name: operation(n)})
                for name, method in (("line", "stop"), ("server", "server_close"), ("app", "close"))
            }
            if failed is None:
                cli._close_serving(owners, None, lambda: order.append("unwire"))
            else:
                with self.assertRaises(OpaqueFailure) as raised:
                    cli._close_serving(owners, None, lambda: order.append("unwire"))
                self.assertIs(raised.exception, failure)
            self.assertEqual(order, ["line", "server"] if failed == "server" else ["line", "server", "app", "unwire"])

    def test_primary_failure_and_failed_drain_retain_wiring(self):
        primary, cleanup = OpaqueFailure(), OpaqueFailure()
        order = []

        def fail():
            order.append("app")
            raise cleanup

        owners = {"app": SimpleNamespace(close=fail), "model": object()}
        with self.assertRaises(OpaqueFailure) as raised:
            cli._close_serving(owners, primary, lambda: order.append("unwire"))
        self.assertIs(raised.exception, primary)
        self.assertIs(primary.__cause__, cleanup)
        self.assertEqual(order, ["app"])
        self.assertIsNotNone(owners["model"])

    def test_multiple_cleanup_errors_bypass_foreign_note_hook(self):
        primary = OpaqueFailure()

        def fail():
            raise OpaqueFailure()

        with self.assertRaises(OpaqueFailure) as raised:
            cli._close_serving({"line": SimpleNamespace(stop=fail), "app": SimpleNamespace(close=fail)}, primary)
        self.assertIs(raised.exception, primary)
        self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
        self.assertEqual(len(primary.__cause__.exceptions), 2)

    def test_malformed_notes_cannot_replace_primary_or_drop_cleanup_failures(self):
        primary = OpaqueFailure()
        primary.__notes__ = 123
        failures = [OpaqueFailure(), OpaqueFailure()]

        def fail(index):
            raise failures[index]

        with self.assertRaises(OpaqueFailure) as result:
            cli._close_serving(
                {"line": SimpleNamespace(stop=lambda: fail(0)), "app": SimpleNamespace(close=lambda: fail(1))}, primary
            )
        self.assertIs(result.exception, primary)
        self.assertEqual(primary.__cause__.exceptions, tuple(failures))

    def test_mlx_wrapper_startup_faults_and_two_stream_quiescence(self):
        core, mlx = ModuleType("mlx.core"), ModuleType("mlx")
        order = []
        core.gpu, core.cpu = "GPU", "CPU"
        core.default_stream = lambda device: device
        core.synchronize = lambda stream: order.append("sync-" + stream)
        mlx.core = core

        def start(*args):
            owners = args[-1]
            owners["model"] = object()
            owners["app"] = SimpleNamespace(close=lambda: order.append("app"))
            raise OpaqueFailure()

        with (
            patch.dict(sys.modules, {"mlx": mlx, "mlx.core": core}),
            patch.object(cli, "_serve_mlx_start", side_effect=start),
            patch("tensorfold.server.residency.unwire", side_effect=lambda _: order.append("unwire")),
        ):
            with self.assertRaises(OpaqueFailure):
                cli._serve_mlx(SimpleNamespace(), None, Path("unused"), 0, (), 1)
        self.assertEqual(order, ["app", "sync-GPU", "sync-CPU", "unwire"])

    def test_cuda_app_failure_refusal_follow_and_success_all_close_engine(self):
        args = cli.build_parser().parse_args(["serve", "unused", "--no-drafts"])
        server = ModuleType("tensorfold.cuda.server")
        for phase in ("app", "serve", "follow", "refusal", "success"):
            order = []
            args.tp, args.rank, args.master = (2, 1, "owner") if phase == "follow" else (1, 0, "")
            args.prefill_fp8 = phase == "refusal"
            engine = SimpleNamespace(
                w=SimpleNamespace(precision="checkpoint", fast_prefill=False), close=lambda: order.append("close")
            )

            def fail_follow():
                order.append("follow")
                raise OpaqueFailure()

            engine.follow = fail_follow

            def app(*unused, **kwargs):
                order.append("app")
                if phase == "app":
                    raise OpaqueFailure()
                return SimpleNamespace(effective_context_window=2048)

            def serve(*unused, **kwargs):
                order.append("serve")
                if phase == "serve":
                    raise OpaqueFailure()

            server.App, server.serve = app, serve
            def http_server(*unused, **kwargs):
                self.assertEqual(kwargs, {"max_connections": None})
                return SimpleNamespace(server_close=lambda: order.append("server-close"), handlers_drained=True)

            server.Server = http_server
            server.make_handler = lambda app: object()
            family = SimpleNamespace(
                title="fixture", model_type="fixture", package=SimpleNamespace(cuda_engine=lambda *a, **k: engine)
            )
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.dict(sys.modules, {server.__name__: server}),
                patch.object(cli.stacks, "arm"),
                redirect_stdout(io.StringIO()),
            ):
                if phase == "success":
                    self.assertEqual(cli._serve_cuda(args, family, Path(tmp), 2048), 0)
                else:
                    with self.assertRaises(ValueError if phase == "refusal" else OpaqueFailure):
                        cli._serve_cuda(args, family, Path(tmp), 2048)
            self.assertEqual(order[-1], "close")
            self.assertEqual(order.count("close"), 1)

    def test_partial_app_without_scheduler_drains_before_reset(self):
        order = []
        app = SimpleNamespace(
            _startup_complete=False,
            engine=SimpleNamespace(drain=lambda: order.append("drain"), reset=lambda: order.append("reset")),
        )
        app_close()(app)
        self.assertEqual(order, ["drain", "reset"])

    def test_failed_startup_never_saves_sessions_and_joins_on_stop_failure(self):
        order = []
        cancelled, entered = threading.Event(), threading.Event()

        def work():
            entered.set()
            self.assertTrue(cancelled.wait(2))

        scope = ThreadWork(work)
        thread = threading.Thread(target=scope.run)
        scope.launch(thread)
        self.assertTrue(entered.wait(1))
        primary = OpaqueFailure()

        def stop(timeout):
            order.append("stop")
            raise primary

        scheduler = SimpleNamespace(
            cancel=lambda _: cancelled.set(),
            stop=stop,
            _thread=SimpleNamespace(is_alive=lambda: False),
            on_stop=object(),
        )
        app = SimpleNamespace(
            _startup_complete=False,
            scheduler=scheduler,
            warmup_thread=thread,
            warmup_work=scope,
            warmup_cancellation=object(),
            save_sessions=lambda: order.append("saved"),
        )
        try:
            with self.assertRaises(OpaqueFailure) as raised:
                app_close()(app)
            self.assertIs(raised.exception, primary)
            self.assertIsNone(scheduler.on_stop)
            self.assertFalse(thread.is_alive())
            self.assertEqual(order, ["stop"])
        finally:
            cancelled.set()
            thread.join(2)

    def test_never_started_warmup_requires_no_join(self):
        work = ThreadWork(lambda: None)
        thread = threading.Thread(target=work.run)
        scheduler = SimpleNamespace(
            cancel=lambda _: None, stop=lambda **_: None, _thread=SimpleNamespace(is_alive=lambda: False)
        )
        app = SimpleNamespace(
            _startup_complete=False,
            scheduler=scheduler,
            warmup_thread=thread,
            warmup_work=work,
            warmup_cancellation=object(),
        )
        app_close()(app)
        self.assertIsNone(scheduler.on_stop)


class Numbers(unittest.TestCase):
    def test_valid_values_and_existing_off_range_sampling_preserved(self):
        args = cli.build_parser().parse_args(["serve", "unused"])
        for name, value in (
            ("temperature", -1.0),
            ("top_p", -1.0),
            ("top_p", 2.0),
            ("prompt_cache_gib", 0.0),
            ("decode_share", 0.0),
            ("ssd_experts", 0.25),
        ):
            setattr(args, name, value)
            before = dict(vars(args))
            check_numbers(args)
            self.assertEqual(vars(args), before)

    def test_all_float_controls_refuse_nonfinite_and_memory_ranges(self):
        names = (
            "temperature",
            "top_p",
            "min_p",
            "prompt_cache_gib",
            "spill_gib",
            "pass_cache_gib",
            "mlx_cache_gib",
            "ssd_experts",
            "decode_share",
            "mtp_confidence",
            "yarn_factor",
        )
        for name in names:
            for value in (float("nan"), float("inf"), -float("inf"), True, "1"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    check_numbers(SimpleNamespace(**{name: value}))
        for name in ("prompt_cache_gib", "spill_gib", "pass_cache_gib", "mlx_cache_gib", "ssd_experts", "decode_share"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_numbers(SimpleNamespace(**{name: -1}))
        for name in ("min_p", "mtp_confidence"):
            for value in (-0.1, 1.1):
                with self.assertRaises(ValueError):
                    check_numbers(SimpleNamespace(**{name: value}))

    def test_invalid_numbers_fail_before_download_or_background_update(self):
        args = cli.build_parser().parse_args(["serve", "unused", "--decode-share", "nan"])
        with patch.object(cli, "_config_dir", side_effect=AssertionError("config/IO reached")):
            with self.assertRaises(ValueError):
                cli.cmd_serve(args)

    def test_generation_config_validation_and_do_sample_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generation_config.json"
            for value in ([], {"temperature": float("nan")}, {"min_p": 2}, {"top_k": True}):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    cli._generation_config(Path(tmp))
            path.write_text(json.dumps({"do_sample": False, "temperature": float("nan"), "top_k": 20}))
            self.assertEqual(cli._generation_config(Path(tmp)), {"temperature": 0.0, "top_k": 20})


class LiveLifetime(unittest.TestCase):
    def test_tick_that_passed_wait_cannot_draw_after_stop(self):
        out = io.StringIO()
        line = live.LiveLine(lambda: "visible", out, every=0.001)
        entered, release = threading.Event(), threading.Event()
        draw = line.draw

        def delayed():
            entered.set()
            self.assertTrue(release.wait(2))
            draw()

        line.draw = delayed
        line._work.launch(line._thread)
        self.assertTrue(entered.wait(1))
        stopped = threading.Thread(target=line.stop)
        stopped.start()
        self.assertTrue(line._stop.wait(1))
        release.set()
        stopped.join(2)
        self.assertFalse(stopped.is_alive())
        self.assertFalse(line._thread.is_alive())
        self.assertEqual(out.getvalue(), "")
        line.stop()

    def test_start_failure_restores_both_proxies(self):
        line = live.LiveLine(lambda: "visible", io.StringIO())
        original = sys.stdout, sys.stderr
        with patch.object(line._thread, "start", side_effect=RuntimeError("controlled start failure")):
            with self.assertRaises(RuntimeError):
                line.install()
        self.assertEqual((sys.stdout, sys.stderr), original)
        self.assertIsNone(line._saved)

    def test_start_interruption_after_native_bootstrap_cancels_callback_before_restore(self):
        line = live.LiveLine(lambda: "visible", io.StringIO(), every=1)
        entered = []
        line._work = ThreadWork(lambda: (entered.append(True), line._tick()))
        line._thread = threading.Thread(target=line._work.run)
        start = line._thread.start
        primary = OpaqueFailure()

        def interrupted():
            start()
            raise primary

        original = sys.stdout, sys.stderr
        with patch.object(line._thread, "start", side_effect=interrupted):
            with self.assertRaises(OpaqueFailure) as raised:
                line.install()
        self.assertIs(raised.exception, primary)
        self.assertFalse(line._thread.is_alive())
        self.assertFalse(entered)
        self.assertEqual((sys.stdout, sys.stderr), original)
        self.assertIsNone(line._saved)

    def test_join_timeout_retains_output_owners_and_self_join_rejected(self):
        line = live.LiveLine(lambda: "visible", io.StringIO())
        line._saved = (object(), object())
        saved = line._saved
        entered, release = threading.Event(), threading.Event()
        line._work = ThreadWork(lambda: (entered.set(), release.wait(3)))
        line._thread = threading.Thread(target=line._work.run)
        line._work.launch(line._thread)
        self.assertTrue(entered.wait(1))
        drain = line._work.drain
        with patch.object(line._work, "drain", side_effect=lambda thread, **kwargs: drain(thread, timeout=0.01)):
            with self.assertRaises(TimeoutError):
                line.stop()
        self.assertIs(line._saved, saved)
        release.set()
        line._work.drain(line._thread, timeout=2)
        line._saved = None
        line._thread = threading.current_thread()
        with self.assertRaises(RuntimeError):
            line.stop()


if __name__ == "__main__":
    unittest.main()
