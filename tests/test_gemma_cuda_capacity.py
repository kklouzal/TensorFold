"""Processor metadata and the DFlash companion's precision reach Gemma's real header-only serving geometry."""

import math
import sys
from types import ModuleType

import pytest
from gemma_cuda_target_metadata_test import inventory, tiny_config
from test_gemma_cuda_dflash_metadata import config as draft_config
from test_gemma_cuda_dflash_metadata import write_checkpoint

from tensorfold.cuda import capacity as common
from tensorfold.families.gemma4.cuda import capacity
from tensorfold.families.gemma4.cuda.vision import vision_shapes


@pytest.mark.parametrize("component", ["image_processor", "video_processor"])
def test_admission_accounts_for_largest_declared_processor_pool(component, tmp_path, monkeypatch):
    raw = tiny_config()
    raw.update(image_token_id=90, video_token_id=91, vision_config={
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1,
        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 64,
        "patch_size": 16, "pooling_kernel_size": 3, "position_embedding_size": 1024,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6,
    })
    _, where = inventory(raw)
    headers = {name: {"dtype": value[3], "shape": value[4]} for name, value in where.items()}
    headers.update({name: {"dtype": "BF16", "shape": shape} for name, shape in vision_shapes(raw).items()})
    headers.update({
        "embed_vision.embedding_projection.weight": {"dtype": "U32", "shape": (64, 8)},
        "embed_vision.embedding_projection.scales": {"dtype": "BF16", "shape": (64, 1)},
        "embed_vision.embedding_projection.biases": {"dtype": "BF16", "shape": (64, 1)},
    })
    options = {"image_processor": {"max_soft_tokens": 70}, "video_processor": {"max_soft_tokens": 70}}
    metadata = ModuleType("tensorfold.checkpoint_metadata")
    metadata.read_config = lambda directory: raw
    metadata.read_json = lambda directory, filename: options
    runtime = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "tensorfold.checkpoint_metadata", metadata)
    monkeypatch.setitem(sys.modules, "torch", runtime)
    monkeypatch.setattr(common, "headers", lambda directory: headers)

    def admitted(directory, context, explicit, torch, geometry, rule, *, weight_estimator):
        # The boundary stands in for physical admission, not the family formula.
        # Real config/header validation, processor policy and geometry all run.
        assert torch is runtime
        weights, _ = weight_estimator(directory)
        return weights, geometry

    monkeypatch.setattr(common, "admit", admitted)
    small_weights, small = capacity.admit(tmp_path, 128, True, vision=True)
    options[component]["max_soft_tokens"] = 1120
    large_weights, large = capacity.admit(tmp_path, 128, True, vision=True)

    assert small_weights == large_weights
    assert large.needed(128) > small.needed(128)
    # Independent minimum: the native pool's one_hot(int64) -> float32
    # conversion owns both dense matrices during conversion. A legal 1120-
    # token spatial grid has 1120 * 3**2 patches. The old fixed280 admission
    # was below this unavoidable allocation peak even for this tiny tower.
    minimum_pool_bytes = (1120 * 3**2) * 1120 * (8 + 4)
    assert large.needed(128) >= minimum_pool_bytes


def test_drafter_admission_counts_bf16_or_private_fp8_copies(tmp_path, monkeypatch):
    """--drafter-bits 16 admits the companion's BF16 bytes; 8 admits what DFlash(bits=8) keeps resident (checked
    against the loaded tensors in test_gemma_cuda_dflash): per fused projection ([k|v|q], o, [gate|up], down, fc) an
    E4M3 byte per weight and an FP32 scale per output row over rows padded to 128, the BF16 norms, and the private
    FP8 copy of the target's tied [vocab, hidden] head; its load staging holds every BF16 source. The companion's K/V
    geometry does not depend on its weight precision."""
    raw = tiny_config()
    text = raw["text_config"]
    _, where = inventory(raw)
    target_headers = {name: {"dtype": value[3], "shape": value[4]} for name, value in where.items()}
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    draft, _ = write_checkpoint(draft_dir, {**draft_config(), "num_target_layers": text["num_hidden_layers"],
                                            "vocab_size": text["vocab_size"],
                                            "dflash_config": {"target_layer_ids": [0, 1], "mask_token_id": 4}})
    metadata = ModuleType("tensorfold.checkpoint_metadata")
    metadata.read_config = lambda directory: raw
    monkeypatch.setitem(sys.modules, "tensorfold.checkpoint_metadata", metadata)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    read = common.headers                   # the companion's real header-only read; the target's inventory above
    monkeypatch.setattr(common, "headers", lambda directory: read(directory) if directory == draft_dir
                        else target_headers)
    monkeypatch.setattr(common, "admit", lambda directory, context, explicit, torch, geometry, rule, *,
                        weight_estimator: (weight_estimator(directory)[0], geometry))

    alone, plain = capacity.admit(tmp_path, 128, True)
    bf16, with_bf16 = capacity.admit(tmp_path, 128, True, drafter_dir=draft_dir)
    fp8, with_fp8 = capacity.admit(tmp_path, 128, True, drafter_dir=draft_dir, drafter_bits=8)
    shapes = draft.shapes().values()
    assert bf16.resident - alone.resident == sum(2 * math.prod(s) for s in shapes)

    def copy(n, k):                         # E4M3 codes and FP32 row scales over 128-padded output rows
        rows = -(-n // 128) * 128
        return rows * k + rows * 4

    hidden, width, heads = draft.hidden, draft.width, draft.heads * draft.head_dim
    kvq = (2 * draft.kv_heads * draft.head_dim + heads, hidden)
    fused = [(hidden, len(draft.tap_layers) * hidden)] + draft.layers * [kvq, (hidden, heads), (2 * width, hidden),
                                                                        (hidden, width)]
    assert fp8.resident - alone.resident == (sum(copy(n, k) for n, k in fused)
                                            + sum(2 * n for (n,) in (s for s in shapes if len(s) == 1))
                                            + copy(text["vocab_size"], text["hidden_size"]))
    assert fp8.staging >= sum(2 * math.prod(s) for s in shapes)
    assert with_fp8.needed(128) == with_bf16.needed(128) > plain.needed(128)
