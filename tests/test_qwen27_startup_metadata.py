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
