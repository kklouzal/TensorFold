"""Processor metadata reaches Gemma's real header-only serving geometry."""

import sys
from types import ModuleType

import pytest
from gemma_cuda_target_metadata_test import inventory, tiny_config

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
