"""Actual NVFP4 startup wiring with header-only checkpoints and cold CPU providers.

The real constructor, geometry, pool planner and shared serving preparation run.
Payload loading, device operations and owned native allocations are stand-ins;
these controls do not qualify model arithmetic, CUDA allocation or native pins.
"""
# Optional CUDA dependencies must be detected before importing backend modules.
import json
import math
import struct
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tokenizers import Tokenizer, models

from tensorfold.cuda import allocation_bound, capacity, expert_pin, geometry, kv_arena, precision, prompt_precision
from tensorfold.cuda.nvfp4 import cache_a4
from tensorfold.families.qwen3_5.cuda import multi, nvfp4_load, nvfp4_memory
from tensorfold.families.qwen3_5_moe import dflash_metadata, ram_experts
from tensorfold.families.qwen3_5_moe.cuda import chain_graphs, dflash, weights
from tensorfold.families.qwen3_5_moe.cuda.nvfp4_engine import QwenNvfp4MoeEngine

GIB = 1 << 30
ALLOCATION_OVERHEAD = 2 << 20


def _checkpoint(directory, config, fields):
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config))
    entries, offset = {}, 0
    for name, dtype, shape in fields:
        size = math.prod(shape) * capacity.SIZES[dtype]
        entries[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(entries).encode()
    # Sparse payload extents let real header admission see enough experts to
    # distinguish a 1GiB pool from the single-cell bootstrap, without payload IO.
    with (directory / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        stream.truncate(8 + len(encoded) + offset)


@pytest.fixture
def startup(tmp_path, monkeypatch):
    target, draft_dir = tmp_path / "target", tmp_path / "draft"
    text = {"hidden_size": 4096, "moe_intermediate_size": 4096, "num_experts": 32,
            "num_experts_per_tok": 2, "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 32, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}
    fields = []
    for layer in range(2):
        base = f"model.language_model.layers.{layer}.mlp"
        for expert in (*[f"experts.{i}" for i in range(32)], "shared_expert"):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for field, dtype, shape in (("weight_packed", "U8", (4096, 2048)),
                                           ("weight_scale", "F8_E4M3", (4096, 256)),
                                           ("weight_global_scale", "F32", (1,)),
                                           ("input_global_scale", "F32", (1,))):
                    fields.append((f"{base}.{expert}.{projection}.{field}", dtype, shape))
        fields.extend(((base + ".gate.weight", "BF16", (32, 4096)),
                       (base + ".shared_expert_gate.weight", "BF16", (1, 4096))))
    fields.extend((("model.language_model.embed_tokens.weight", "BF16", (32, 4096)),
                   ("model.language_model.norm.weight", "BF16", (4096,)),
                   ("lm_head.weight", "BF16", (32, 4096))))
    _checkpoint(target, {"text_config": text}, fields)
    tokenizer = Tokenizer(models.WordLevel({"unknown": 0, "stop": 1}, unk_token="unknown"))
    tokenizer.save(str(target / "tokenizer.json"))
    (target / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}", "eos_token": "stop"}))
    draft_text = {"architectures": ["DFlashDraftModel"], "model_type": "qwen3", "hidden_size": 4096,
                  "intermediate_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
                  "head_dim": 128, "num_hidden_layers": 1, "num_target_layers": 2, "vocab_size": 32,
                  "block_size": 16, "rms_norm_eps": 1e-6, "max_position_embeddings": 128,
                  "rope_theta": 10000, "dflash_config": {"target_layer_ids": [0, 1], "mask_token_id": 31}}
    draft_config = dflash_metadata.DraftConfig.from_dict(draft_text)
    _checkpoint(draft_dir, draft_text, [(name, "BF16", shape) for name, shape in draft_config.shapes().items()])
    events = []
    state = NS(target=target, draft_dir=draft_dir, text=text, draft_text=draft_text, events=events,
               available=16 * GIB, raw_free=16 * GIB, admission_error=None, pin_close_error=None,
               pins=[], caches=[], admissions=[], loads=[], drafts=[], runtimes=[], draft_runtimes=[], calibrations=[],
               prepared=[])

    class Pins:
        def __init__(self, planned, **options):
            self.planned, self.options, self.closed = planned, options, False
            state.pins.append(self)
            events.append("pins")

        def close(self):
            events.append("pins-close")
            if state.pin_close_error is not None:
                raise state.pin_close_error
            self.closed = True

    class Cache:
        def __init__(self, gpu_bytes, entry_bytes, device, *, pin_policy, width, dims):
            assert gpu_bytes == entry_bytes and device == "cuda"
            assert width == text["moe_intermediate_size"] and dims == text["hidden_size"]
            self.gpu_bytes, self.entry_bytes, self.capacity = gpu_bytes, entry_bytes, 1
            self.pin_policy, self._pool, self.closed = pin_policy, object(), False
            self.allocated_bytes = gpu_bytes + ALLOCATION_OVERHEAD
            state.raw_free -= self.allocated_bytes
            self.requests = [gpu_bytes]
            state.caches.append(self)
            events.append("bootstrap")

        def configure_before_use(self, requested):
            events.append("configure")
            self.requests.append(requested)
            state.raw_free += self.allocated_bytes - (requested + ALLOCATION_OVERHEAD)
            self.allocated_bytes = requested + ALLOCATION_OVERHEAD
            self.gpu_bytes, self.capacity = requested, requested // self.entry_bytes

        def close(self):
            events.append("cache-close")
            self.pin_policy.close()
            self.closed = True

    class Allocation:
        def __init__(self):
            self.receipt = {"provider": "cold allocation bound"}

        def __call__(self, requested):
            return requested + ALLOCATION_OVERHEAD

        def validate_current(self):
            events.append("validate")

    def allocation(runtime, pool, *, before_snapshot):
        assert runtime is torch and pool is state.caches[0]._pool and before_snapshot is snapshot
        events.append("allocation-bound")
        return Allocation()

    def admit(directory, context, explicit, runtime, target_geometry, transform, **options):
        events.append("admit")
        state.admissions.append((directory, context, explicit, runtime, target_geometry, transform, options))
        if state.admission_error is not None:
            raise state.admission_error
        estimate, _host = options["weight_estimator"](directory)
        return {"context_window": 64, "cache_slots": 72, "budget_bytes": state.raw_free,
                "weight_bytes_estimate": estimate.resident,
                "cache_workspace_bytes_estimate": target_geometry.bytes_at(72)}

    def load(directory, **options):
        events.append("model-load")
        state.loads.append((directory, options))
        return NS(config=NS(eos=tuple(options["eos_token_ids"]), vocab=32), norm=torch.ones(1), head=NS(n=32),
                  layers=[])

    def draft(directory, target_weights, **options):
        events.append("draft-load")
        state.drafts.append((directory, target_weights, options))
        return NS(world=1)

    runtime_geometry, draft_geometry = ram_experts.runtime_geometry, dflash_metadata.draft_geometry

    def runtime(text, **options):
        result = runtime_geometry(text, **options)
        state.runtimes.append((options, result))
        return result

    def draft_runtime(raw, **options):
        result = draft_geometry(raw, **options)
        state.draft_runtimes.append((options, result))
        return result

    def calibrate(decoder, streams):
        events.append("calibrate")
        state.calibrations.append((decoder, decoder.max_rows, decoder.block, streams))
        decoder.costs = [(1, .1), (streams * 8, .2)]
        # The CPU decoder has no device gate; install a labeled live-budget
        # provider to exercise startup's CUDA reserve adjustment.
        decoder.memory_gate = NS(reserve=4 * GIB)

    class Scheduler:
        def __init__(self, decoder, *, max_streams):
            assert decoder.costs and max_streams == 3

        def start(self):
            events.append("scheduler-start")

        def close(self):
            events.append("scheduler-close")

    def prepare(runner, widths, *, taps):
        state.prepared.append((list(widths), taps))    # the fake weights hold no layers to capture
        return 0

    snapshot = object()
    monkeypatch.setattr(chain_graphs.ChainGraphs, "prepare", prepare)
    monkeypatch.setattr(precision, "mode", lambda: precision.CHECKPOINT)
    monkeypatch.setattr(prompt_precision, "fp8", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: events.append("device"))
    monkeypatch.setattr(kv_arena, "driver", lambda device: events.append("vmm"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("fence"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty-cache"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (state.raw_free, 16 * GIB))
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", lambda: snapshot)
    monkeypatch.setattr(capacity, "total_bytes", lambda runtime: 16 * GIB)
    monkeypatch.setattr(capacity, "available_bytes", lambda runtime: state.available)
    monkeypatch.setattr(capacity, "cuda_reserve_bytes", lambda total: 0)
    monkeypatch.setattr(capacity, "unified", lambda runtime: False)
    monkeypatch.setattr(capacity, "admit", admit)
    monkeypatch.setattr(geometry, "prompt_rows", lambda *args: 512)
    monkeypatch.setattr(expert_pin, "ExpertPinPolicy", Pins)
    monkeypatch.setattr(cache_a4, "A4ExpertCache", Cache)
    monkeypatch.setattr(allocation_bound, "pool_allocation_bound", allocation)
    monkeypatch.setattr(weights, "load", load)
    monkeypatch.setattr(dflash, "DFlash", draft)
    monkeypatch.setattr(ram_experts, "runtime_geometry", runtime)
    monkeypatch.setattr(dflash_metadata, "draft_geometry", draft_runtime)
    monkeypatch.setattr(multi.MultiDecoder, "calibrate", calibrate)
    monkeypatch.setattr("tensorfold.cuda.scheduler.Scheduler", Scheduler)
    return state


@pytest.mark.parametrize("budget,streams,automatic", [("auto", 3, False), (1.0, 3, False),
                                                      (None, 3, False), ("auto", 1, False), (None, 3, True)])
def test_actual_startup_wires_pool_admission_draft_and_shared_serving(startup, budget, streams, automatic):
    state = startup
    if automatic:
        state.available = 1
    engine = QwenNvfp4MoeEngine(state.target, state.draft_dir, streams=streams, max_rows=8,
                               vram_experts=budget, vram_experts_pin="auto", drafter_bits=4,
                               context=64, context_explicit=True, eos_token_ids=[3, 1], allow_copy=False)
    try:
        assert engine.max_rows == engine.tree_rows == 8 and engine.context_window == 64
        assert engine.eos == (3, 1) and engine.model_dir == state.target
        assert state.events.index("device") < state.events.index("vmm") < state.events.index("admit")
        assert state.events.index("admit") < state.events.index("model-load") < state.events.index("draft-load")
        # Residency is decided with the resident banks' device-plan scratch (and one stream's chain graphs);
        # leased host-pool banks are then admitted without them.
        offload = budget is not None or automatic
        options, _ = state.runtimes[0]
        base = {"streams": streams, "keep": 3 if streams > 1 else 4, "max_rows": 8, "prompt_rows": 512}
        assert options == {**base, "graphs": streams == 1, "planned": True}
        assert len(state.runtimes) == (2 if offload else 1)
        options, runtime = state.runtimes[-1]
        assert options == (base if offload else {**base, "graphs": streams == 1, "planned": True})
        draft_options, draft_runtime = state.draft_runtimes[0]
        assert draft_options == {"streams": streams, "kept": 4, "block": 8, "prompt_rows": 512}
        directory, context, explicit, provider, target_geometry, transform, admission = state.admissions[0]
        assert (directory, context, explicit, provider) == (state.target, 64, True, torch)
        assert target_geometry is runtime.geometry and admission["draft_geometry"] is draft_runtime
        assert admission["draft_weights"].func is dflash_metadata.draft_weights
        assert admission["draft_weights"].keywords == {"bits": 4, "private_head": True}   # a BF16 target head
        assert admission["draft_dir"] == state.draft_dir
        assert state.drafts == [(state.draft_dir, engine.w, {"context_limit": 64, "block": 8, "bits": 4,
                                                             "stop_score": dflash.STOP_SCORE if streams == 1 else None})]
        assert state.loads[0][1]["expert_factory"].keywords["cache"] is engine.expert_cache
        assert state.loads[0][1]["eos_token_ids"] == [3, 1]
        if streams > 1:
            assert state.calibrations == [(engine.multi, 8, 8, streams)]
            assert engine.multi.w is engine.w and engine.multi.draft is engine.draft
            assert engine.multi.context == 72 and engine.multi.allow_copy is False
            assert engine.multi.memory_gate.reserve == 0
            assert state.events.index("draft-load") < state.events.index("calibrate") < state.events.index("scheduler-start")
        else:
            assert engine.multi is engine.scheduler is None and not state.calibrations
        assert (engine.runner is not None) == (streams == 1 and not offload)
        assert state.prepared == ([(list(range(1, 9)), True)] if engine.runner is not None else [])
        if offload:
            plan = ram_experts.layout(state.target, "auto" if automatic else budget)
            cache, pins = state.caches[0], state.pins[0]
            assert state.events.index("bootstrap") < state.events.index("admit")
            assert cache.requests[0] == plan.entry_bytes and cache.capacity > 1
            assert cache.requests[-1] == engine.expert_pool_plan["requested_pool_bytes"]
            bootstrap_physical = plan.entry_bytes + ALLOCATION_OVERHEAD
            assert engine.expert_pool_plan["raw_free_bytes"] == 16 * GIB - bootstrap_physical
            assert engine.capacity_plan["budget_bytes"] == 16 * GIB - bootstrap_physical
            assert engine.expert_pool_plan["new_allocation_bound_bytes"] == cache.requests[-1] + ALLOCATION_OVERHEAD
            if budget == 1.0:
                assert cache.requests[-1] == (GIB // plan.entry_bytes) * plan.entry_bytes == plan.gpu_bytes
                # Only the future replacement's physical allocation is charged:
                # the bootstrap is already absent from observed free memory.
                assert admission["resident_extra"] == plan.gpu_bytes + ALLOCATION_OVERHEAD
            else:
                assert admission["resident_extra"] == 0
            host_extra = (plan.loading_bytes + plan.staging_bytes + plan.metadata_host_bytes
                          + plan.control_host_bytes + plan.source_filesystem_bytes)
            _, host_loading = nvfp4_memory.weight_estimate(state.target, nvfp4_load.weight_bytes)
            assert pins.planned == admission["host_resident"] == plan.host_bytes
            assert pins.options == {"mode": "auto", "future_bytes": host_extra + host_loading,
                                    "allocation_bytes": plan.authority_allocations}
            assert admission["host_extra_staging"] == host_extra
            assert engine.expert_pool_plan["future_draft_workspace_bytes"] == draft_runtime.bytes_at(72)
            for name, count in runtime.components(72).items():
                assert engine.expert_pool_plan[name] == count
            assert state.events.index("draft-load") < state.events.index("configure")
            if streams == 1:
                assert engine.room.cache is engine.cache and engine.room.budget == runtime.components(72)["future_kv_bytes"]
        else:
            assert engine.expert_cache is None and not state.caches and not state.pins
            assert transform is nvfp4_load.weight_bytes
            assert admission["host_resident"] == admission["host_extra_staging"] == admission["resident_extra"] == 0
    finally:
        engine.close()
    assert engine._closed
    if state.caches:
        assert state.caches[0].closed and state.pins[0].closed
        assert state.events.index("cache-close") < state.events.index("pins-close")


def test_missing_vmm_refuses_before_expert_ownership_admission_or_payload(startup, monkeypatch):
    failure = RuntimeError("CUDA device lacks virtual memory management")

    def refused(device):
        assert device == 0 and startup.events == ["device"]
        raise failure

    monkeypatch.setattr(kv_arena, "driver", refused)
    with pytest.raises(RuntimeError) as caught:
        QwenNvfp4MoeEngine(startup.target, startup.draft_dir, vram_experts="auto")
    assert caught.value is failure
    assert startup.events == ["device"] and not startup.caches and not startup.loads and not startup.pins


def test_draft_row_one_is_refused_by_actual_startup_before_device_or_payload(startup):
    with pytest.raises(ValueError, match="at least 2 with a secondary model"):
        QwenNvfp4MoeEngine(startup.target, startup.draft_dir, max_rows=1, vram_experts="auto")
    assert not startup.events and not startup.caches and not startup.loads and not startup.drafts


def test_one_row_without_secondary_model_keeps_the_supported_serial_path(startup):
    engine = QwenNvfp4MoeEngine(startup.target, max_rows=1, eos_token_ids=[1])
    try:
        assert engine.max_rows == 1 and engine.draft is None
        assert engine.multi is engine.scheduler is engine.expert_cache is None
        assert engine.runner is not None and engine.runner.capacity == engine.capacity_plan["cache_slots"]
        assert startup.prepared == [([1], False)]      # every chain width captured at startup, no taps (no draft)
        assert not startup.drafts and startup.events.index("admit") < startup.events.index("model-load")
    finally:
        engine.close()


@pytest.mark.parametrize("bits", [4, 8])
def test_stored_fp8_target_head_does_not_reserve_a_second_private_draft_head(startup, bits):
    state = startup
    stored = state.target.parent / "stored-fp8-head"
    fields = [(name, info["dtype"], info["shape"]) for name, info in capacity.headers(state.target).items()
              if name != "lm_head.weight"]
    fields += [("lm_head.weight", "F8_E4M3", (32, 4096)), ("lm_head.weight_scale", "F32", (1,))]
    _checkpoint(stored, {"text_config": state.text}, fields)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (stored / name).write_bytes((state.target / name).read_bytes())
    state.target = stored
    engine = QwenNvfp4MoeEngine(stored, state.draft_dir, max_rows=8, drafter_bits=bits,
                               context=64, context_explicit=True, eos_token_ids=[1])
    try:
        admission = state.admissions[0][-1]
        assert admission["draft_weights"].keywords == {"bits": bits, "private_head": False}
        assert admission["draft_weights"](state.draft_dir) == dflash_metadata.draft_weights(
            state.draft_dir, bits=bits, private_head=False)
    finally:
        engine.close()


def test_unified_cache_refusal_precedes_native_ownership_and_payload(startup, monkeypatch):
    monkeypatch.setattr(capacity, "unified", lambda runtime: True)
    with pytest.raises(ValueError, match="require discrete CUDA memory"):
        QwenNvfp4MoeEngine(startup.target, startup.draft_dir, max_rows=8, vram_experts="auto")
    assert startup.events == ["device", "vmm", "fence"]
    assert not startup.pins and not startup.caches and not startup.admissions and not startup.loads


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_admission_failure_retires_bootstrap_and_preserves_primary(startup, failure_type, cleanup_fails):
    state = startup
    primary, cleanup = failure_type("admission refused"), OSError("pin retirement refused")
    state.admission_error = primary
    state.pin_close_error = cleanup if cleanup_fails else None
    engine = object.__new__(QwenNvfp4MoeEngine)
    with pytest.raises(failure_type) as caught:
        engine.__init__(state.target, state.draft_dir, max_rows=8, vram_experts="auto", vram_experts_pin="on")
    assert caught.value is primary
    assert state.events == ["device", "vmm", "pins", "bootstrap", "allocation-bound", "admit", "fence", "cache-close",
                            "pins-close"]
    assert not state.loads and not state.drafts and engine.expert_cache is state.caches[0]
    if cleanup_fails:
        assert primary.__cause__ is cleanup and engine in primary._tensorfold_retained_owners
        assert not engine._closed and not state.caches[0].closed and not state.pins[0].closed
        state.pin_close_error = None
        engine.close()
    else:
        assert primary.__cause__ is None
    assert engine._closed and state.caches[0].closed and state.pins[0].closed


def _native_rope(state):
    """The target declares Qwen3.6's 262,144 positions of default RoPE (what --yarn-factor extends)."""

    raw = json.loads((state.target / "config.json").read_text())
    raw["text_config"].update(max_position_embeddings=262144, rope_parameters={
        "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25})
    (state.target / "config.json").write_text(json.dumps(raw))


@pytest.mark.parametrize("factor,declared,expected", [(None, False, None), (4.0, False, 4.0),
                                                    (None, True, 4.0), (8.0, True, 8.0)])
def test_yarn_factor_reaches_admission_loader_and_log(startup, capsys, factor, declared, expected):
    state = startup
    _native_rope(state)
    if declared:
        raw = json.loads((state.target / "config.json").read_text())
        text = raw["text_config"]
        text["max_position_embeddings"] = 1048576
        text["rope_parameters"].update(rope_type="yarn", factor=4.0, original_max_position_embeddings=262144)
        (state.target / "config.json").write_text(json.dumps(raw))
    engine = QwenNvfp4MoeEngine(state.target, None, context=64, context_explicit=True, eos_token_ids=[3, 1],
                               yarn_factor=factor)
    try:
        admission = state.admissions[0][-1]
        loaded = state.loads[0][1]
        log = capsys.readouterr().out
        if expected is None:
            assert "context_limit" not in admission and loaded["rope"] is None and "static YaRN" not in log
        else:
            assert (admission["context_limit"], admission["original_context"]) == (int(262144 * expected), 262144)
            rope = loaded["rope"]
            assert (rope.rope_type, rope.factor, rope.native_context, rope.context_limit) == (
                "yarn", expected, 262144, int(262144 * expected))
            assert f"static YaRN factor {expected:g}, rotary amplitude" in log
    finally:
        engine.close()


def test_a_yarn_window_past_the_drafter_rope_limit_is_refused_before_any_load(startup, monkeypatch):
    """DFlash v1 keeps its own RoPE limit (128 here): an admitted YaRN window past it is refused right after the
    admission estimate, before the target or the drafter loads."""

    state = startup
    _native_rope(state)
    admitted = capacity.admit
    monkeypatch.setattr(capacity, "admit", lambda *args, **options: {**admitted(*args, **options),
                                                                     "context_window": 129})
    with pytest.raises(ValueError, match=r"declared rotary context \(128\).*--drafter none"):
        QwenNvfp4MoeEngine(state.target, state.draft_dir, max_rows=8, eos_token_ids=[3, 1], yarn_factor=4.0)
    assert state.admissions[0][-1]["context_limit"] == 1048576
    assert not state.loads and not state.drafts


def test_a_yarn_window_at_the_drafter_rope_limit_keeps_the_requested_drafter(startup, monkeypatch):
    state = startup
    _native_rope(state)
    admitted = capacity.admit
    monkeypatch.setattr(capacity, "admit", lambda *args, **options: {**admitted(*args, **options),
                                                                     "context_window": 128})
    engine = QwenNvfp4MoeEngine(state.target, state.draft_dir, context=128, context_explicit=True,
                               max_rows=8, eos_token_ids=[3, 1], yarn_factor=4.0)
    try:
        assert engine.context_window == 128 and engine.draft is not None
        assert state.loads and state.drafts
        assert state.admissions[0][1:3] == (128, True)
    finally:
        engine.close()


@pytest.mark.parametrize("factor", [True, False, 0.0, 0.5, -1.0, float("nan"), float("inf")])
def test_bad_nvfp4_yarn_factor_is_refused_before_admission_or_payload(startup, monkeypatch, factor):
    _native_rope(startup)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: pytest.fail("CUDA prepared"))
    with pytest.raises(ValueError, match="yarn-factor"):
        QwenNvfp4MoeEngine(startup.target, None, context=64, yarn_factor=factor)
    assert not startup.admissions and not startup.loads and not startup.drafts
