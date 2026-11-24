"""DeepSeek-V4.1 SM70 startup helpers on CPU: the phase timeline and the parallel extension prebuild (every module's
``extension()`` runs once, concurrently; a build failure re-raises at ``wait``)."""

import sys
import threading
import time
import types

import pytest

from tensorfold.families.deepseek_v4.cuda.startup import Prebuild, Timeline


def _fake(name, seconds, error=None, calls=None):
    module = types.ModuleType(name)

    def extension():
        calls.append((name, threading.get_ident()))
        time.sleep(seconds)
        if error:
            raise error
    module.extension = extension
    sys.modules[name] = module
    return name


def test_timeline_phases_and_notes():
    clock = Timeline(3)
    with clock.phase("engram") as note:
        note[0] = "fill 1.0"
    clock.add("graphs", time.perf_counter(), 2.0)
    line = clock.line()
    assert line.startswith("[tensorfold] rank 3 startup (")
    assert "engram (fill 1.0) @" in line and "graphs @" in line and "2.0s" in line


def test_prebuild_runs_every_extension_once_in_parallel():
    calls = []
    names = [_fake(f"_ds41_fake_ext{i}", 0.2, calls=calls) for i in range(4)]
    clock = Timeline(0)
    started = time.perf_counter()
    prebuild = Prebuild(clock, names)
    prebuild.wait(names[0])
    prebuild.wait()
    assert time.perf_counter() - started < 0.6          # four 0.2 s builds at once, not one after another
    assert sorted(n for n, _ in calls) == sorted(names) and len({t for _, t in calls}) == 4
    assert "extensions (slowest _ds41_fake_ext" in clock.line()
    prebuild.wait()                                     # idempotent: no second timeline row
    assert clock.line().count("extensions (") == 1


def test_prebuild_reraises_a_build_failure():
    calls = []
    good, bad = _fake("_ds41_fake_good", 0.0, calls=calls), _fake("_ds41_fake_bad", 0.0, RuntimeError("nvcc"), calls)
    prebuild = Prebuild(Timeline(0), [good, bad])
    prebuild.wait(good)
    with pytest.raises(RuntimeError, match="nvcc"):
        prebuild.wait()


def test_deepseek_q4_and_yarn_flags_are_family_bound(tmp_path, monkeypatch):
    from tensorfold import cli, families
    from tensorfold.families import deepseek_v4
    from tensorfold.families.deepseek_v4.cuda.weights import Geometry

    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41"}')
    family = families.detect(tmp_path)
    monkeypatch.setattr(Geometry, "from_config", classmethod(lambda cls, raw: cls()))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--tp", "8",
                                          "--master", "127.0.0.1", "--kv-dtype", "rotorquant-iso64-norm4-h64", "--yarn-factor", "2"])
    cli._check_serve_options(args, family, "cuda", tmp_path)
    rope = deepseek_v4.rope_parameters(tmp_path, args.yarn_factor)
    assert (args.kv_dtype, rope.factor, rope.context_limit) == ("rotorquant-iso64-norm4-h64", 2, 131_072)
    assert deepseek_v4.rope_parameters(tmp_path).factor == 16
    for invalid in (0.5, float("inf"), float("nan")):
        args.yarn_factor = invalid
        with pytest.raises(ValueError, match="YaRN factor"):
            cli._check_serve_options(args, family, "cuda", tmp_path)
    args.yarn_factor = None
    args.kv_dtype = "int4"
    with pytest.raises(ValueError, match="not --kv-dtype int4"):
        cli._check_serve_options(args, family, "cuda", tmp_path)


def test_deepseek_gguf_keeps_its_native_default_and_accepts_q4_yarn2(tmp_path, monkeypatch):
    import json

    from tensorfold import cli
    from tensorfold.families.deepseek_v4.cuda.weights import Geometry

    config = {"model_type": "deepseek_v41", "quantization_config": {"quant_method": "gguf"},
              "gguf_file": "backbone.gguf", "engram_file": "engram.gguf", "max_position_embeddings": 1_048_576}
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in ("gguf_file", "engram_file"):
        (tmp_path / config[name]).touch()
    monkeypatch.setattr(Geometry, "from_config", classmethod(lambda cls, raw: cls()))
    monkeypatch.setattr(cli.stacks, "start", lambda: None)
    supplied = []

    def serve(args, family, directory, context):
        assert family.model_type == "deepseek_v41" and directory == tmp_path
        supplied.append((args.kv_dtype, args.yarn_factor, context))
        return 0

    monkeypatch.setattr(cli, "_serve_cuda", serve)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--tp", "8", "--master", "127.0.0.1",
               "--no-update-check"]
    parser = cli.build_parser()
    assert cli.cmd_serve(parser.parse_args(command)) == 0
    assert cli.cmd_serve(parser.parse_args([*command, "--kv-dtype", "rotorquant-iso64-norm4-h64", "--yarn-factor", "2",
                                           "--context", "65536"])) == 0
    assert supplied == [("bf16", None, 1_048_576), ("rotorquant-iso64-norm4-h64", 2.0, 65_536)]


@pytest.mark.parametrize("fault", [False, True])
def test_rotorquant_follower_detects_device_fault_while_its_pipe_is_idle(monkeypatch, fault):
    torch = pytest.importorskip("torch")
    from tensorfold.cuda import cohort
    from tensorfold.families.deepseek_v4.cuda import engine

    events = []
    frames = iter((engine._HEAD.pack(engine.OP_LAYERS, 1, 0, 0, 0), TimeoutError(), None))

    def receive(timeout):
        events.append(("receive", timeout))
        frame = next(frames)
        if isinstance(frame, TimeoutError):
            raise frame
        return frame

    def query():
        events.append(("query",))
        if fault:
            raise RuntimeError("device trap")
        return True

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: types.SimpleNamespace(query=query))
    monkeypatch.setattr(cohort, "notify_failure", lambda: events.append(("notify",)))
    follower = object.__new__(engine.DS41Engine)
    follower.target = types.SimpleNamespace(rotor_cache=object())
    follower.pipes = types.SimpleNamespace(receive=receive)
    follower._run = lambda op, first, *_: events.append(("run", op, first))
    if fault:
        with pytest.raises(RuntimeError, match="device trap"):
            follower.follow()
    else:
        follower.follow()
    expected = [("receive", engine.PIPE_TIMEOUT), ("run", engine.OP_LAYERS, 1),
                ("receive", engine.ROTOR_ERROR_POLL_S), ("query",)]
    assert events == expected + ([("notify",)] if fault else [("receive", engine.PIPE_TIMEOUT)])


@pytest.mark.parametrize("rank", [0, 1])
def test_engram_unmaps_after_device_retirement(monkeypatch, rank):
    torch = pytest.importorskip("torch")
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine

    e = object.__new__(DS41Engine)
    e._closed, e.rank, e.graphs = False, rank, object()
    e.drafter = None
    actions = []

    def unmap():
        assert e.graphs is None
        actions.append("unmap")

    e.scheduler = types.SimpleNamespace(close=lambda: actions.append("drain")) if rank == 0 else None
    e.pipes = types.SimpleNamespace(send=lambda _: actions.append("stop peers"))
    e.engram = types.SimpleNamespace(close=unmap)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: actions.append("device fence"))
    e.close()
    e.close()
    assert actions == (["drain", "stop peers"] if rank == 0 else []) + ["device fence", "unmap"]


def test_engram_stays_mapped_if_device_retirement_fails(monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine

    e = object.__new__(DS41Engine)
    graph = object()
    e._closed, e.rank, e.graphs = False, 1, graph
    e.drafter = None
    e.engram = types.SimpleNamespace(close=lambda: pytest.fail("released unretired mapping"))

    def fail():
        raise RuntimeError("device progress failed")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="device progress failed"):
        e.close()
    assert e.graphs is graph


def test_target_reuses_h64_current_window_in_qualified_shapes(monkeypatch):
    torch = pytest.importorskip("torch")

    from tensorfold.cuda.sm70 import ds41_sparse_attention as fused
    from tensorfold.families.deepseek_v4.cuda.target import Target

    def selected(*args, decode_window, **kwargs):
        return decode_window

    monkeypatch.setattr(fused, "attend", selected)
    h64 = torch.empty(0, dtype=torch.uint8)
    fp16 = torch.empty(0, dtype=torch.float16)
    pages = object()
    for tokens, start, ring, cache, image, requests, expected in (
        (1023, 0, h64, pages, None, 1, False),
        (1024, 0, h64, pages, None, 1, True),
        (1024, 193, h64, pages, None, 1, False),
        (4096, 193, h64, pages, None, 1, True),
        (8192, 193, h64, pages, None, 1, True),
        (8193, 0, h64, pages, None, 1, False),
        (8192, 0, h64, None, None, 1, False),
        (8192, 0, fp16, pages, None, 1, False),
        (8192, 0, h64, pages, object(), 1, False),
        (8192, 0, h64, pages, None, 2, False),
    ):
        query = torch.empty(tokens, 1)
        step = types.SimpleNamespace(prefill=[(0, 0, tokens, start + tokens - 1)], image=image,
                               num_reqs=requests, positions=None, req_slot=None, spec=None)
        assert Target._attend(None, query, step, ring, ring, None, cache=cache) is expected


def test_single_request_verification_uses_the_c1_window_reader(monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorfold.cuda.sm70 import ds41_sparse_attention as fused
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as packed
    from tensorfold.families.deepseek_v4.cuda.target import Target

    monkeypatch.setattr(fused, "attend", lambda *args, **kwargs: "fused")
    monkeypatch.setattr(packed, "attend", lambda *args, **kwargs: "gather")
    window = torch.empty(0, dtype=torch.uint8)
    for rows, requests, verify, expected in (
        (1, 1, False, "gather"),
        (4, 1, True, "gather"),
        (6, 1, True, "gather"),
        (4, 1, False, "fused"),
        (8, 2, True, "fused"),
    ):
        step = types.SimpleNamespace(spec=object() if verify else None,
                                     num_reqs=requests, req_slot=None, positions=None)
        query = torch.empty(rows, 1)
        assert Target._attend(None, query, step, window, window, None) == expected
