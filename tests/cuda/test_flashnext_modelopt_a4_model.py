"""Complete native-A4 RAM cache parity, separate from the legacy W4A16 oracles.

Original-linear numerical oracles live in test_flashnext_modelopt_a4. This
gate compares different resident/cache ownership, routing and sequence-state
paths under the SAME declared checkpoint math, including real MTP proposals.
"""

import json
import struct
from types import SimpleNamespace

import pytest
import torch
from nvfp4_tiny import write

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native checkpoint-A4 whole-model execution requires SM12", allow_module_level=True)

from tensorfold.cleanup import finish, raise_failures
from tensorfold.cuda import moe as moe_mod
from tensorfold.cuda import precision
from tensorfold.cuda.nvfp4.modelopt_experts import CachedModelOptExpertsA4, ModelOptExpertsA4
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp import ram_experts
from tensorfold.families.qwen4_exp.cuda import nvfp4_moe
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.cuda.weights import load
from tensorfold.families.qwen4_exp.ssd_table import EncodedSSDTable


@pytest.fixture
def checkpoint_math():
    with precision.using(precision.CHECKPOINT, asked=True):
        yield


def _calibrated_model(path):
    """Author explicit nonunit synthetic calibrations; leave every weight byte unchanged.

    The old fixture's input_scale=1 underflows small SwiGLU block scales. Main
    gate/up use a complete common FP32 calibration, like the audited checkpoint;
    down scales alternate by expert to exercise cached scalar-cell remapping.
    All values are exact powers of two, and MTP's excluded BF16 payload is intact.
    """
    model = write(path, hidden=512, experts=8, shared_width=128, ple_bf16=True)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    assert len(shards) == 1
    changes = []
    with (model / shards.pop()).open("r+b") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        for name, field in header.items():
            if not name.startswith("model.layers.") or not name.endswith(".input_scale"):
                continue
            parts = name.split(".")
            assert parts[4] == "experts" and field["dtype"] == "F32" and field["shape"] == []
            lo, hi = field["data_offsets"]
            assert hi - lo == 4
            layer, expert, projection = int(parts[2]), int(parts[5]), parts[6]
            scale = 2.0 ** (-9 - expert % 2) if projection == "down_proj" else 2.0 ** (-5 + layer)
            stream.seek(8 + size + lo)
            written = stream.write(struct.pack("<f", scale))
            assert written == 4
            changes.append(name)
    assert len(changes) == 2 * 8 * 3
    return model


def _routed_slots_are_nonzero_and_equal(resident, cached):
    cfg = resident.cfg
    shape = SimpleNamespace(num_experts=cfg.experts, num_experts_per_tok=cfg.top_k,
                            hidden_size=cfg.hidden, moe_intermediate_size=cfg.moe_width)
    x = (torch.randn(17, cfg.hidden, generator=torch.Generator().manual_seed(601)) * .5).to("cuda", torch.bfloat16)
    for prefill_mode in (False, True):
        native = moe_mod.MoEBuffers(17, shape, "cuda", prefill=prefill_mode)
        offloaded = moe_mod.MoEBuffers(17, shape, "cuda", prefill=prefill_mode)
        native.y.fill_(float("nan"))
        offloaded.y.fill_(float("nan"))
        for weights, buf in ((resident, native), (cached, offloaded)):
            layer = weights.layers[0]
            nvfp4_moe.moe(x, x, layer.moe.router, layer.moe.experts, buf, shape)
        assert torch.isfinite(native.y).all()
        assert (torch.count_nonzero(native.y[:, :cfg.top_k], dim=-1) > 0).all()
        assert torch.equal(offloaded.pick, native.pick)
        assert torch.equal(offloaded.y, native.y)
    torch.cuda.synchronize()


def _serial(weights, prompt, sampling):
    engine = Engine(weights, capacity=128, max_rows=8, prefill_rows=5, graphs=False)
    first = prefill(engine, prompt, sampling)
    result = serial_decode(engine, first, 8, sampling, stop_eos=False)
    assert len(result.tokens) == 8 and result.drafted == 0
    torch.cuda.synchronize()
    return result.tokens


def _concurrent(weights, prompts):
    decoder = MultiDecoder(weights, slots=2, capacity=128, depth=2, confidence=0., prefill_rows=5,
                           share=.5, stop_eos=False)
    assert decoder.copy is False
    streams = [Stream(prompt, 8, Sampling(seed=7 + index, top_k=20, top_p=.95), draft=True, stop_eos=False)
               for index, prompt in enumerate(prompts)]
    for stream in streams:
        decoder.admit(stream)
    for _ in range(100):
        if not decoder.live():
            assert all(stream.error is None and len(stream.out) == 8 and stream.drafted > 0 for stream in streams)
            torch.cuda.synchronize()
            return [stream.out for stream in streams]
        decoder.finish(decoder.round())
    pytest.fail("native-A4 cached/resident concurrent requests did not finish")


@pytest.mark.usefixtures("checkpoint_math")
def test_complete_cached_a4_serial_mtp_and_concurrent_equal_resident(tmp_path, monkeypatch):
    """BF16 backbone/MTP/SSD-table case; other storage formats have component oracles."""
    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "0")
    model = _calibrated_model(tmp_path)
    layout = ram_experts.layout(model, "auto", native_a4=True, checkpoint=True)
    assert layout.native_a4_banks == ("model.layers.0.mlp", "model.layers.1.mlp")
    owners = [None, None]
    cleanup_slot = [None]
    primary = cleanup = None

    def close_cached_tables():
        if owners[1] is not None:
            owners[1].meta["ple_tables"].close(torch.cuda.synchronize)
    def close_cached_authority():
        if owners[1] is not None:
            owners[1].meta["expert_cache"].close()
    def close_resident_tables():
        if owners[0] is not None:
            owners[0].meta["ple_tables"].close(torch.cuda.synchronize)
    retirement = (close_cached_tables, close_cached_authority, close_resident_tables)
    try:
        owners[0] = a = load(model, draft_vocab=128, ple_on_ssd=True)
        owners[1] = b = load(model, draft_vocab=128, ple_on_ssd=True, vram_experts=layout.entry_bytes / 2**30)
        cache = b.meta["expert_cache"]
        assert cache.capacity == 1 and cache.entry_bytes == layout.entry_bytes
        assert [len(layer.source) for layer in cache._layers.values()] == [6, 6, 4]
        for weights, kind in ((a, ModelOptExpertsA4), (b, CachedModelOptExpertsA4)):
            assert all(isinstance(layer.moe.experts, nvfp4_moe.MoEA4)
                       and isinstance(layer.moe.experts.routed_experts, kind)
                       and layer.moe.experts.routed_experts.direct_input for layer in weights.layers)
            assert isinstance(weights.mtp.layer.moe.experts, nvfp4_moe.MoE4 if weights is a else nvfp4_moe.CachedMoE4)
            tables = [layer.ple.table for layer in weights.layers if layer.ple is not None]
            assert len(tables) == 1 and isinstance(tables[0], EncodedSSDTable) and tables[0].kind == "bf16"
        assert all(layer.moe.experts.routed_experts.shared_input for layer in a.layers)
        _routed_slots_are_nonzero_and_equal(a, b)
        prompt = [5, 17, 99, 250, 7, 64, 30, 11, 12, 13]
        for sampling in (None, Sampling(seed=7, top_k=20, top_p=.95)):
            expected = _serial(a, prompt, sampling)
            cached_serial = _serial(b, prompt, sampling)
            assert cached_serial == expected
            for weights in (a, b):
                engine = Engine(weights, capacity=128, max_rows=8, prefill_rows=5, graphs=True)
                if weights is b:
                    assert engine.graphs is None
                first = prefill(engine, prompt, sampling)
                result = mtp_decode(engine, first, 8, sampling, depth=2, confidence=0., stop_eos=False)
                assert result.tokens == expected and result.drafted > 0
        prompts = (prompt, prompt[::-1])
        references = [_serial(a, value, Sampling(seed=7 + index, top_k=20, top_p=.95))
                      for index, value in enumerate(prompts)]
        resident_concurrent = _concurrent(a, prompts)
        cached_concurrent = _concurrent(b, prompts)
        assert resident_concurrent == references
        assert cached_concurrent == references
        assert cache.evictions > 0
    except BaseException as error:  # noqa: BLE001 -- preserve operation failure through independent retirement
        primary = error
    finally:
        try:
            finish(retirement)
        except BaseException as error:  # noqa: BLE001 -- transport only after every retirement attempt
            cleanup = cleanup_slot[0] = error
    if cleanup is not None:
        raise_failures(primary, cleanup_slot)
    if primary is not None:
        raise primary
    assert owners[0].meta["ple_tables"].closed and owners[1].meta["ple_tables"].closed
    assert owners[1].meta["expert_cache"]._closed
