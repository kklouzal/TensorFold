"""DeepSeek-V4.1 SM70 startup helpers on CPU: the phase timeline and the parallel extension prebuild (every module's
``extension()`` runs once, concurrently; a build failure re-raises at ``wait``)."""

import os
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


@pytest.mark.parametrize("degree", [None, 1, 2, 4])
def test_deepseek_serving_selects_expert_degree(tmp_path, degree):
    from tensorfold import cli, families
    from tensorfold.families.deepseek_v4.cuda.weights import checked_etp

    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41","gguf_file":"backbone.gguf"}')
    (tmp_path / "backbone.gguf").touch()
    command = ["serve", str(tmp_path), "--backend", "cuda", "--tp", "8", "--master", "127.0.0.1"]
    if degree is not None:
        command += ["--expert-tensor-parallel-size", str(degree)]
    parser = cli.build_parser()
    family = families.detect(tmp_path)
    args = parser.parse_args(command)
    cli._check_serve_options(args, family, "cuda", tmp_path)
    assert checked_etp(args.expert_tensor_parallel_size) == (4 if degree is None else degree)
    rejected = parser.parse_args([*command, "--expert-tensor-parallel-size", "8"])
    with pytest.raises(ValueError, match="degree 1, 2 or 4"):
        cli._check_serve_options(rejected, family, "cuda", tmp_path)


def test_deepseek_cli_full_windows_is_opt_in_auto(tmp_path):
    from tensorfold import cli, families

    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41","gguf_file":"backbone.gguf"}')
    (tmp_path / "backbone.gguf").touch()
    command = ["serve", str(tmp_path), "--backend", "cuda", "--tp", "8", "--master", "127.0.0.1",
               "--kv-dtype", "rotorquant-iso64-norm4-h64", "--parallel", "1"]
    parser = cli.build_parser()
    family = families.detect(tmp_path)
    requested = parser.parse_args([*command, "--kv-full-windows", "auto"])
    cli._check_serve_options(requested, family, "cuda", tmp_path)
    numeric = parser.parse_args([*command, "--kv-full-windows", "1"])
    with pytest.raises(ValueError, match="only --kv-full-windows auto"):
        cli._check_serve_options(numeric, family, "cuda", tmp_path)


def test_auto_full_windows_owns_startup_when_expert_balancing_is_disabled(tmp_path, monkeypatch):
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine
    from tensorfold.families.deepseek_v4.cuda.startup import ExpertStartup

    options = []
    monkeypatch.setattr(DS41Engine, "_initialize", lambda self, model, **kw: options.append(kw))
    DS41Engine(tmp_path, rank=0, tp=8, serve=True, no_expert_balance=True)
    assert options.pop()["startup"] is None
    DS41Engine(tmp_path, rank=0, tp=8, serve=True, no_expert_balance=True, kv_full_windows="auto")
    selected = options.pop()
    assert selected["full_windows_auto"] and isinstance(selected["startup"], ExpertStartup)
    assert not selected["expert_balance_live"] and selected["expert_balance_state"] is None


@pytest.mark.parametrize("degree,experts,width,offsets,blocks", [
    (1, 48, 2304, (0,), 9),
    (2, 96, 1152, (0, 128), 5),
    (4, 192, 576, (0, 64, 128, 192), 3),
])
def test_deepseek_expert_carriers_match_rank_ownership(degree, experts, width, offsets, blocks):
    from tensorfold.families.deepseek_v4.cuda.weights import Geometry, Loader

    geometry = Geometry()
    byte_width = 66
    down = types.SimpleNamespace(ggml_type=16, shape=(384, 5120, 2304), path="down", offset=64)
    gate = types.SimpleNamespace(ggml_type=16, shape=(384, 2304, 5120), path="gate", offset=128)
    gguf = types.SimpleNamespace(tensors={"down": down, "gate": gate})
    copied = []
    staging = types.SimpleNamespace(copy=lambda path, spans, _out: copied.append((path, spans)))
    for rank in range(8):
        loader = Loader(gguf, rank, "meta", staging, expert_tensor_parallel_size=degree)
        first = rank // degree * experts
        start = rank % degree * width
        tensor, quant, offset = loader.expert_bank("down", geometry, down=True)
        assert (tuple(tensor.shape), quant, offset) == ((experts, 5120, blocks * byte_width), 16, offsets[rank % degree])
        row_bytes = 9 * byte_width
        assert copied[-1] == ("down", [(64 + first * 5120 * row_bytes, experts * 5120 * row_bytes)])
        tensor, quant, offset = loader.expert_bank("gate", geometry, down=False)
        assert (tuple(tensor.shape), quant, offset) == ((experts, width, 20 * byte_width), 16, 0)
        row_bytes = 20 * byte_width
        if degree == 1:
            assert copied[-1] == ("gate", [(128 + first * 2304 * row_bytes, experts * 2304 * row_bytes)])
        else:
            assert copied[-1] == ("gate", [(128 + ((first + i) * 2304 + start) * row_bytes, width * row_bytes)
                                           for i in range(experts)])
    down.shape = (400, 5120, 2304)
    with pytest.raises(ValueError, match="expert tensor shape"):
        loader.expert_bank("down", geometry, down=True)


@pytest.mark.parametrize("degree", (1, 2, 4))
def test_deepseek_permutation_keeps_gate_up_and_down_experts_in_the_same_slots(monkeypatch, degree):
    from tensorfold.cuda.sm70 import ds41_decode_moe
    from tensorfold.families.deepseek_v4.cuda.weights import Geometry, Loader

    geometry = Geometry()
    hidden, intermediate, experts = geometry.hidden, geometry.moe_inter, geometry.experts
    kind, block_bytes, packed_bytes = 29, 56, 132
    gate_row, down_row = hidden // 256 * block_bytes, intermediate // 256 * block_bytes
    tensor = lambda name, shape, offset: types.SimpleNamespace(ggml_type=kind, shape=shape, path=name, offset=offset)
    gguf = types.SimpleNamespace(tensors={
        "gate": tensor("gate", (experts, intermediate, hidden), 64),
        "up": tensor("up", (experts, intermediate, hidden), 96),
        "down": tensor("down", (experts, hidden, intermediate), 128),
    })
    packed_groups = []
    ext = types.SimpleNamespace(compact_down_row_bytes=lambda codec: packed_bytes,
                                pack_down_fragments=lambda _src, _dst, codec, group: packed_groups.append((codec, group)))
    monkeypatch.setattr(ds41_decode_moe, "extension", lambda: ext)
    reads = []

    def copy(path, spans, _out):
        reads.append((path, spans))

    def copy_packed(path, offset, source_row_bytes, out, pack):
        assert source_row_bytes == down_row and out.shape[1:] == (hidden, packed_bytes)
        reads.append((path, [(offset, out.shape[0] * hidden * source_row_bytes)]))
        pack(None, None)

    staging = types.SimpleNamespace(copy=copy, copy_packed_rows=copy_packed)
    permutation = tuple((5 * i + 53) % experts for i in range(experts))
    width = intermediate // degree
    group_size = experts // (8 // degree)
    for rank in range(8):
        loader = Loader(gguf, rank, "meta", staging, expert_tensor_parallel_size=degree, compact_expert_down=degree == 4)
        first = rank // degree * group_size
        cut = rank % degree * width
        order = permutation[first:first + group_size]
        for name, base in (("gate", 64), ("up", 96)):
            read, _, _ = loader.expert_bank(name, geometry, down=False, expert_ids=order)
            assert tuple(read.shape) == (group_size, width, gate_row)
            assert reads.pop() == (name, [(base + (expert * intermediate + cut) * gate_row, width * gate_row)
                                          for expert in order])
        result, _, left = loader.expert_bank("down", geometry, down=True, expert_ids=order)
        blocks = (cut % 256 + width + 255) // 256
        assert left == cut % 256
        assert tuple(result.shape) == (group_size, hidden, packed_bytes if degree == 4 else blocks * block_bytes)
        spans = [(128 + expert * hidden * down_row, hidden * down_row) for expert in order]
        expected = [("down", [span]) for span in spans] if degree == 4 else [("down", spans)]
        assert reads == expected
        if degree == 4:
            assert packed_groups == [(kind, cut // 32)] * group_size
            packed_groups.clear()
        reads.clear()


def test_bound_deepseek_loader_reads_the_checkpoint_incarnation_it_parsed(tmp_path):
    from gguf_fixture import write
    from tensorfold.families.deepseek_v4.cuda.weights import ENGRAM_TABLE, IQ4_NL, Shards

    original, replacement = (tmp_path / name for name in ("original.gguf", "replacement.gguf"))
    write(original, [("w", (1,), 0, [1, 2, 3, 4])], {})
    write(replacement, [("w", (1,), 0, [5, 6, 7, 8])], {})
    engram = tmp_path / "engram.gguf"
    write(engram, [(ENGRAM_TABLE, (1, 256), IQ4_NL, [0] * 144)], {})
    entry = tmp_path / "model.gguf"
    entry.symlink_to(original)

    shards = Shards(entry, engram, bind_files=True)
    try:
        before = shards.placement_files()
        assert [path for path, *_ in before] == [str(original), str(engram)]
        entry.unlink()
        entry.symlink_to(replacement)
        assert shards.placement_files() == before
        tensor, data = shards.raw("w")
        assert bytes(data) == bytes((1, 2, 3, 4))
        fd = shards.duplicate_file(tensor.path)
        try:
            assert os.pread(fd, 4, tensor.offset) == bytes((1, 2, 3, 4))
        finally:
            os.close(fd)
        with original.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(ValueError, match="checkpoint changed during startup"):
            shards.verify_placement_files()
    finally:
        shards.close()


@pytest.mark.parametrize("kind,block_bytes,packed_bytes", [(16, 66, 150), (17, 74, 168), (29, 56, 132)])
def test_compact_etp4_loader_selects_whole_rows_and_rank_cut(monkeypatch, kind, block_bytes, packed_bytes):
    from tensorfold.cuda.sm70 import ds41_decode_moe
    from tensorfold.families.deepseek_v4.cuda.weights import Geometry, Loader

    calls = []
    ext = types.SimpleNamespace(compact_down_row_bytes=lambda codec: packed_bytes,
                                pack_down_fragments=lambda *args: calls.append(args))
    monkeypatch.setattr(ds41_decode_moe, "extension", lambda: ext)
    tensor = types.SimpleNamespace(ggml_type=kind, shape=(384, 5120, 2304), path="down", offset=64)
    gguf = types.SimpleNamespace(tensors={"down": tensor})

    def copy(path, offset, source_row_bytes, out, pack):
        calls.append((path, offset, source_row_bytes, tuple(out.shape)))
        pack("input", "output")
        return out

    staging = types.SimpleNamespace(copy_packed_rows=copy)
    for rank in range(8):
        loader = Loader(gguf, rank, "meta", staging, expert_tensor_parallel_size=4, compact_expert_down=True)
        result, codec, left = loader.expert_bank("down", Geometry(), down=True)
        row = 9 * block_bytes
        assert loader.compact_expert_down and codec == kind and left == rank % 4 * 64
        assert tuple(result.shape) == (192, 5120, packed_bytes)
        assert calls[-2:] == [
            ("down", 64 + rank // 4 * 192 * 5120 * row, row, tuple(result.shape)),
            ("input", "output", kind, rank % 4 * 18),
        ]
    for degree in (1, 2):
        assert not Loader(gguf, 0, "meta", staging, expert_tensor_parallel_size=degree,
                          compact_expert_down=True).compact_expert_down


def test_compact_staging_keeps_row_boundaries_across_buffer_reuse(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorfold.families.deepseek_v4.cuda.staging import Staging

    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, pin_memory=False, **kw: empty(*args, **kw))
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: None)

    class Event:
        recorded = False

        def record(self):
            self.recorded = True

        def synchronize(self):
            assert self.recorded

    monkeypatch.setattr(torch.cuda, "Event", Event)
    path = tmp_path / "rows"
    payload = torch.arange(54, dtype=torch.uint8).view(9, 6)
    path.write_bytes(b"pad" + bytes(payload.flatten().tolist()))
    packed = empty((9, 3), dtype=torch.uint8)
    raw = empty(13, dtype=torch.uint8)
    staging = Staging(chunk=23, buffers=2)
    try:
        def pack(source, destination):
            destination.view(-1, 3).copy_(source.view(-1, 6)[:, 1::2])

        assert staging.copy_packed_rows(path, 3, 6, packed, pack) is packed
        assert torch.equal(packed, payload[:, 1::2])
        assert staging.copy(path, [(4, 7), (26, 6)], raw) is raw
        assert torch.equal(raw, torch.cat((payload.flatten()[1:8], payload.flatten()[23:29])))
        oversized = empty((10, 3), dtype=torch.uint8)
        with pytest.raises(OSError, match="short read"):
            staging.copy_packed_rows(path, 3, 6, oversized, pack)
    finally:
        staging.close()


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
    native = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--tp", "8",
                                             "--master", "127.0.0.1", "--kv-dtype", "fp8-window+fp4-compressed"])
    assert native.kv_dtype == deepseek_v4.NATIVE_KV_DTYPE
    cli._check_serve_options(native, family, "cuda", tmp_path)
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
    e.expert_observer = None
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
                               num_reqs=requests, positions=None, req_slot=None)
        assert Target._attend(None, query, step, ring, ring, None, cache=cache) is expected
