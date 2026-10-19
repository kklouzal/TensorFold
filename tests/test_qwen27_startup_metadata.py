"""Full actual Qwen constructor with CPU weights and labeled distributed transport.

The tests cover preparation/dispatch through successful post-load startup. They
do not qualify CUDA allocation, native NCCL, model arithmetic or two real GPUs.
"""
import json
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
from tokenizers import Tokenizer, models

from tensorfold.cuda import capacity, geometry, markers, prompt_precision
from tensorfold.cuda.nvfp4 import format as nv_format
from tensorfold.families.qwen3_5.cuda import affine_memory, distributed, exl3_load, weights
from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
from tensorfold.vision import qwen_cuda


@pytest.fixture
def startup(tmp_path, monkeypatch):
    tokenizer = Tokenizer(models.WordLevel({"unknown": 0, "model_stop": 1, "hello": 2, "stop": 3},
                                          unk_token="unknown"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}", "eos_token": "stop"}))
    text = {"hidden_size": 64, "intermediate_size": 128}
    # the constructor reads its RoPE policy (default here) from the checkpoint's own config
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {**text, "head_dim": 64}}))
    plan = {"context_window": 32, "cache_slots": 33, "budget_bytes": 1024,
            "weight_bytes_estimate": 128, "cache_workspace_bytes_estimate": 256}
    events, options, gathers = [], [], []
    state = NS(format="affine", mismatch=False)
    tensor, empty = torch.tensor, torch.empty

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return tensor(*args, **kwargs)

    def cpu_empty(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return empty(*args, **kwargs)

    def gathered(destination, source):
        # Preserve actual PyTorch argument direction and complete rank-major
        # payloads. Only the labeled peer's EOS payload may be changed.
        assert destination.numel() == 2 * source.numel() and source.dtype == destination.dtype == torch.int64
        assert source.device.type == destination.device.type == "cpu"
        values = source.tolist()
        gathers.append(values)
        peer = source.clone()
        if "load" in events and len(gathers) == 3 and state.mismatch:
            peer[-1] += 1
        destination.view(-1).copy_(torch.cat((source, peer)))

    def load(*args, **kwargs):
        events.append("load")
        options.append(kwargs)
        return NS(config=NS(eos=tuple(kwargs.get("eos_token_ids", (1, 3)))), norm=torch.ones(1))

    def split(full, rank, **kwargs):
        events.append("split")
        return full

    def resume(directory, *, template):
        assert template.provenance()["source"] == "tokenizer_config.chat_template"
        events.append("resume")

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: events.append("device"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(prompt_precision, "fp8", lambda: False)
    monkeypatch.setattr(exl3_load, "quant_config", lambda directory: None)
    monkeypatch.setattr(nv_format, "is_quantized", lambda directory: state.format == "nvfp4")
    monkeypatch.setattr(capacity, "total_bytes", lambda runtime: 1024)
    monkeypatch.setattr(capacity, "config", lambda directory: text)
    monkeypatch.setattr(capacity, "unified", lambda runtime: False)
    monkeypatch.setattr(capacity, "admit", lambda *args, **kwargs: plan)
    monkeypatch.setattr(geometry, "prompt_rows", lambda *args: 16)
    monkeypatch.setattr(geometry, "prompt_row_bytes", lambda *args: 1)
    monkeypatch.setattr(geometry, "live_kv", lambda *args: 0)
    monkeypatch.setattr(affine_memory, "weight_transform", lambda *args, **kwargs: object())
    monkeypatch.setattr(qwen_cuda, "capacity_geometry", lambda geometry, *args, **kwargs: geometry)
    monkeypatch.setattr(qwen_cuda, "weight_transform", lambda transform, *args, **kwargs: transform)
    monkeypatch.setattr(weights, "load", load)
    monkeypatch.setattr(distributed, "split_weights", split)
    monkeypatch.setattr(dist, "init_process_group", lambda *args, **kwargs: events.append("process_group"))
    monkeypatch.setattr(dist, "all_gather_into_tensor", gathered)
    monkeypatch.setattr(markers, "resume_points", resume)
    return tmp_path, state, events, options, gathers


@pytest.mark.parametrize("format,tp", [("affine", 1), ("nvfp4", 1), ("affine", 2)])
def test_actual_constructor_completes_after_loading_and_uses_its_actual_transport(startup, format, tp):
    directory, state, events, options, gathers = startup
    state.format = format
    engine = Qwen27Engine(directory, None, tp=tp, master="host", eos_token_ids=[3, 1])
    try:
        assert engine.eos == engine.w.config.eos == (3, 1)
        assert engine.context_window == 32 and engine.w.prompt_rows == 16
        assert engine.scheduler is None and engine.draft is None
        assert options == [{"prefill_world": 2, "eos_token_ids": [3, 1]} if tp == 2 else
                           {"tiled": True, "eos_token_ids": [3, 1]}]
        assert events[-1] == "resume" and "load" in events
        if tp == 1:
            assert gathers == [] and "process_group" not in events
        else:
            assert len(gathers) == 3 and gathers[1:] == [[1, 2], [1, 3]]
            assert events.index("load") < events.index("split") < events.index("resume")
    finally:
        engine.close()


def test_actual_tp_constructor_refuses_effective_eos_mismatch_before_resume_setup(startup):
    directory, state, events, _options, gathers = startup
    state.mismatch = True
    with pytest.raises(RuntimeError, match="different effective EOS"):
        Qwen27Engine(directory, None, tp=2, master="host")
    assert "load" in events and "split" in events and "resume" not in events
    assert len(gathers) == 3 and gathers[1:] == [[1, 2], [1, 3]]


@pytest.fixture
def serving(startup, monkeypatch):
    from tensorfold.cuda import scheduler
    from tensorfold.families.qwen3_5.cuda import multi

    directory, _state, events, _options, _gathers = startup
    owner = object.__new__(Qwen27Engine)
    failures = NS(calibrate=None, start=None, close=None)
    decoders, actors, allocator_settings = [], [], []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    def configure_allocator(settings):
        allocator_settings.append(settings)
        events.append("allocator")

    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", configure_allocator)

    class Decoder:
        def __init__(self, w, draft, **kwargs):
            self.w, self.draft, self.options = w, draft, kwargs
            self.costs = []
            decoders.append(self)
            events.append("decoder")

        def calibrate(self, streams):
            assert owner.multi is self and owner.scheduler is None
            self.costs = [(1, .1), (streams * 12, .2)]
            events.append("calibrate")
            if failures.calibrate is not None:
                raise failures.calibrate

    class Actor:
        def __init__(self, decoder, *, max_streams):
            assert decoder is owner.multi and "calibrate" in events
            self.decoder, self.max_streams = decoder, max_streams
            actors.append(self)
            events.append("actor")

        def start(self):
            assert owner.scheduler is self
            events.append("start")
            if failures.start is not None:
                raise failures.start

        def close(self):
            assert owner.scheduler is self
            events.append("close")
            if failures.close is not None:
                raise failures.close

    monkeypatch.setattr(multi, "MultiDecoder", Decoder)
    monkeypatch.setattr(scheduler, "Scheduler", Actor)
    return NS(owner=owner, directory=directory, events=events,
              failures=failures, decoders=decoders, actors=actors, allocator_settings=allocator_settings)


@pytest.mark.parametrize("tp,rank,streams", [(1, 0, 1), (2, 0, 1), (2, 1, 1),
                                            (1, 0, 3), (2, 0, 3), (2, 1, 3)])
def test_actual_constructor_serving_setup_keeps_serial_and_rank_bookkeeping(serving, tp, rank, streams):
    fixture, owner = serving, serving.owner
    owner.__init__(fixture.directory, None, tp=tp, rank=rank, master="host", allow_copy=False,
                   streams=streams, keep="5", eos_token_ids=[3, 1])
    try:
        assert owner.eos == (3, 1) and owner.model_dir == fixture.directory
        assert owner.cache.keep == 4 and owner.cache.entries == []
        assert fixture.events.index("load") < fixture.events.index("empty_cache") < fixture.events.index("resume")
        assert fixture.allocator_settings == (["expandable_segments:True"] if streams > 1 and tp == 1 else [])
        if fixture.allocator_settings:
            assert fixture.events.index("device") < fixture.events.index("allocator") < fixture.events.index("load")
        assert owner.concurrent is (streams > 1)
        if streams == 1:
            assert not fixture.decoders and not fixture.actors
            assert owner.multi is owner.scheduler is None
            if tp == 1:
                assert owner.room.cache is owner.cache and owner.room.budget == 640
            else:
                assert owner.room is None
        else:
            assert fixture.decoders == [owner.multi]
            assert owner.multi.w is owner.w and owner.multi.draft is owner.draft
            assert owner.multi.options == {"allow_copy": False, "rank": rank, "world": tp, "context": 33, "keep": 5,
                                           "points": owner.points, "vision": owner.vision}
            assert owner.multi.model_dir == fixture.directory and owner.room is None
            assert fixture.events.index("resume") < fixture.events.index("decoder") < fixture.events.index("calibrate")
            if rank == 0:
                assert fixture.actors == [owner.scheduler] and owner.scheduler.max_streams == streams
                assert fixture.events.index("calibrate") < fixture.events.index("actor") < fixture.events.index("start")
            else:
                assert owner.scheduler is None and not fixture.actors
    finally:
        owner.close()
    assert owner.scheduler is None


def test_loaded_subclass_prepares_requests_without_running_base_loader(serving, monkeypatch):
    from tensorfold.cuda.chat_template import prepare_chat_template

    class LoadedVariant(Qwen27Engine):
        pass

    fixture = serving
    owner = object.__new__(LoadedVariant)
    owner.torch, owner.tp, owner.rank, owner.allow_copy = torch, 1, 0, False
    owner.capacity_plan = {"context_window": 32, "cache_slots": 33, "budget_bytes": 1024,
                           "weight_bytes_estimate": 128, "cache_workspace_bytes_estimate": 256}
    owner.context_window = 32
    owner.w = NS(config=NS(eos=(3, 1)), norm=torch.ones(1))
    owner.draft, owner.vision = object(), object()
    owner.chat_template = prepare_chat_template(fixture.directory)

    def no_load(*args, **kwargs):
        pytest.fail("shared serving preparation must not discover or load checkpoint formats")

    monkeypatch.setattr(weights, "load", no_load)
    monkeypatch.setattr(exl3_load, "quant_config", no_load)
    monkeypatch.setattr(nv_format, "is_quantized", no_load)
    owner._prepare_serving(fixture.directory, streams=1, keep=5)
    try:
        assert owner.eos == (3, 1) and owner.cache.keep == 4
        assert owner.room.cache is owner.cache and owner.room.budget == 640
        assert owner.model_dir == fixture.directory and owner.points is None
        assert owner.scheduler is owner.multi is None and not owner.concurrent
        assert fixture.events == ["empty_cache", "resume"]
    finally:
        owner.close()


@pytest.mark.parametrize("stage,close_failure", [("calibrate", False), ("start", False), ("start", True)])
def test_actual_serving_setup_keeps_startup_failure_and_actor_retirement(serving, stage, close_failure):
    fixture, owner = serving, serving.owner
    primary, cleanup = KeyboardInterrupt("startup interrupted"), OSError("actor close failed")
    setattr(fixture.failures, stage, primary)
    if close_failure:
        fixture.failures.close = cleanup
    with pytest.raises(KeyboardInterrupt) as caught:
        owner.__init__(fixture.directory, None, streams=3)
    assert caught.value is primary
    if stage == "calibrate":
        assert not fixture.actors and owner.scheduler is None
        assert fixture.events[-1] == "calibrate"
    else:
        assert fixture.actors == [owner.scheduler]
        assert fixture.events[-2:] == ["start", "close"]
        assert caught.value.__cause__ is (cleanup if close_failure else None)
    # The failed owner remains reachable; retire it through the same public close.
    fixture.failures.close = None
    owner.close()
    assert owner.scheduler is None


@pytest.mark.parametrize("selected_rows", [None, 8, 0, 17])
def test_serving_row_limit_reaches_real_decoder_before_cold_calibration(startup, monkeypatch, selected_rows):
    from tensorfold.cuda import scheduler
    from tensorfold.families.qwen3_5.cuda import multi

    directory, _state, _events, _options, _gathers = startup
    decoders, constructor_options, calibration, launches = [], [], [], []
    initialize = multi.MultiDecoder.__init__
    calibrate = multi.MultiDecoder.calibrate

    def initialize_decoder(decoder, *args, **kwargs):
        constructor_options.append(kwargs)
        initialize(decoder, *args, **kwargs)
        decoders.append(decoder)

    def cold_calibration(decoder, streams):
        calibration.append((decoder.max_rows, decoder.block, streams))
        calibrate(decoder, streams, reps=1)

    def forward(w, wins, *, full_logits, tp):
        assert w is decoders[0].w and full_logits and not tp
        launches.append(sum(len(tokens) for tokens, _parents, _state in wins))

    class Actor:
        def __init__(self, decoder, *, max_streams):
            assert decoder.costs and max_streams == 3

        def start(self):
            pass

        def close(self):
            pass

    class SelectedRows(Qwen27Engine):
        def _prepare_serving(self, model_dir, *, streams, keep):
            super()._prepare_serving(model_dir, streams=streams, keep=keep, max_rows=self.max_rows)

    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", lambda settings: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(multi.MultiDecoder, "__init__", initialize_decoder)
    monkeypatch.setattr(multi.MultiDecoder, "calibrate", cold_calibration)
    monkeypatch.setattr(multi, "State", lambda w: NS(w=w))
    monkeypatch.setattr(multi, "multi_tree_forward", forward)
    monkeypatch.setattr(scheduler, "Scheduler", Actor)
    engine_type = Qwen27Engine if selected_rows is None else SelectedRows
    options = {} if selected_rows is None else {"max_rows": selected_rows}
    if selected_rows in (0, 17):
        with pytest.raises(ValueError, match="a stream's window is 1 to 16 rows"):
            engine_type(directory, None, streams=3, **options)
        assert constructor_options[0]["max_rows"] == selected_rows
        assert not decoders and not calibration and not launches
        return

    engine = engine_type(directory, None, streams=3, **options)
    try:
        expected = 16 if selected_rows is None else selected_rows
        assert decoders == [engine.multi]
        assert engine.multi.max_rows == engine.multi.block == expected
        assert calibration == [(expected, expected, 3)]
        assert constructor_options[0].get("max_rows") == selected_rows
        # Calibration intentionally retains its independent 16-row-window grid;
        # the NVFP4 caller budgets that scratch separately from its proposal cap.
        assert sorted(set(launches)) == multi.calibration_rows(3, False)

        class ProposalReached(Exception):
            pass

        class Draft:
            def launch_blocks(self, snapshots, pending, guesses, block, histories=None):
                assert snapshots == ["snapshot"] and pending == [2]
                assert guesses == expected - 1 and block == expected
                raise ProposalReached

        engine.multi.draft, engine.multi.drafts = Draft(), True
        engine.multi.streams[0] = NS(snap="snapshot", draft=True, sampling=None, history=None)
        # Stop at the cold drafter provider: the real verifier must select its
        # admitted proposal width before any target arithmetic is attempted.
        with pytest.raises(ProposalReached):
            engine.multi._verify([(0, multi.TREE, 2, None)])
    finally:
        engine.close()


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("factor", [None, 4.0])
def test_yarn_factor_reaches_admission_loader_rank_agreement_and_log(startup, monkeypatch, capsys, tp, factor):
    directory, _state, _events, options, gathers = startup
    (directory / "config.json").write_text(json.dumps({"text_config": {
        "hidden_size": 64, "intermediate_size": 128, "head_dim": 64, "max_position_embeddings": 262144,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25}}}))
    admitted, admit = [], capacity.admit
    monkeypatch.setattr(capacity, "admit", lambda *args, **kwargs: admitted.append(kwargs) or admit(*args, **kwargs))
    engine = Qwen27Engine(directory, None, tp=tp, master="host", yarn_factor=factor)
    try:
        (kwargs,) = admitted
        (loaded,) = options
        log = capsys.readouterr().out
        if factor is None:
            assert "context_limit" not in kwargs and "original_context" not in kwargs and "rope" not in loaded
            assert "static YaRN" not in log
        else:
            assert (kwargs["context_limit"], kwargs["original_context"]) == (1048576, 262144)
            rope = loaded["rope"]
            assert (rope.rope_type, rope.factor, rope.native_context, rope.context_limit) == ("yarn", 4.0, 262144,
                                                                                            1048576)
            assert "static YaRN factor 4, rotary amplitude 1.1386294" in log
        if tp == 2:                     # the factor (millionths) sits before the prompt precision the ranks compare
            assert gathers[0][-2] == (0 if factor is None else 4_000_000)
    finally:
        engine.close()
