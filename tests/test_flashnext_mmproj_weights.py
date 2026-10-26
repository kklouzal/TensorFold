"""Actual pinned HF model parameters are an independent GGUF mapping control."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp",
                    reason="requires the pinned Qwen4Exp vision provider")

from gguf_fixture import write  # noqa: E402 - optional backend guard precedes CUDA imports
from tensorfold.families.qwen4_exp.cuda.mmproj_weights import (  # noqa: E402 - optional backend guard
    VisionLayout, load_cpu_vision_model, plan_mmproj, read_vision_weights,
)


class Reader:
    def __init__(self):
        self.lengths = []

    def read(self, path, offset, length, device):
        assert device == "cpu"
        self.lengths.append(length)
        with Path(path).open("rb") as stream:
            stream.seek(offset)
            data = stream.read(length)
        return torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())


@pytest.fixture
def checkpoint(tmp_path):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    torch.manual_seed(7139)
    layout = VisionLayout(depth=2, hidden=8, heads=2, intermediate=12, output=6,
                          patch=2, temporal=2, merge=2, channels=3, positions=4, theta=10000.)
    model = Qwen4ExpVisionModel(Qwen4ExpVisionConfig(**layout.config_fields())).eval()
    metadata = {"general.architecture": "clip", "general.type": "mmproj", "clip.has_vision_encoder": True,
                "clip.projector_type": "qwen3vl_merger", "clip.use_gelu": True,
                "clip.vision.projection_dim": 6, "clip.vision.patch_size": 2,
                "clip.vision.embedding_length": 8, "clip.vision.feed_forward_length": 12,
                "clip.vision.block_count": 2, "clip.vision.attention.head_count": 2,
                "clip.vision.spatial_merge_size": 2, "clip.vision.is_deepstack_layers": [False, False],
                "clip.vision.image_mean": [.5, .5, .5], "clip.vision.image_std": [.5, .5, .5],
                "clip.vision.attention.layer_norm_epsilon": 1e-6}
    tensors = []

    def add(name, values, bf16=False):
        if bf16:
            source = values.to(torch.bfloat16)
            data = source.view(torch.uint8).numpy()
            decoded = source.half()
            kind = 30
        else:
            source = values.float()
            data = source.view(torch.uint8).numpy()
            decoded = source.half()
            kind = 0
        tensors.append((name, tuple(values.shape), kind, data))
        return decoded

    # This reverse map follows the original converter; no production map is read.
    block_names = {"attn.qkv": "attn_qkv", "attn.proj": "attn_out", "mlp.linear_fc1": "ffn_up",
                   "mlp.linear_fc2": "ffn_down", "norm1": "ln1", "norm2": "ln2"}
    special = {"pos_embed": "v.position_embd", "merger.norm": "v.post_ln",
               "merger.linear_fc1": "mm.0", "merger.linear_fc2": "mm.2"}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name == "patch_embed.proj.weight":
                first = add("v.patch_embd.weight", parameter[:, :, 0].contiguous())
                second = add("v.patch_embd.weight.1", parameter[:, :, 1].contiguous())
                parameter.data = torch.stack((first, second), dim=2)
            elif name == "patch_embed.proj.bias":
                parameter.data = add("v.patch_embd.bias", parameter)
            else:
                base, suffix = name.rsplit(".", 1)
                if base.startswith("blocks."):
                    _, index, target = base.split(".", 2)
                    source_name = f"v.blk.{index}." + block_names[target]
                else:
                    source_name = special[base]
                parameter.data = add(source_name + "." + suffix, parameter,
                                     bf16=parameter.ndim == 2 and base != "pos_embed")
    path = write(tmp_path / "vision.gguf", tensors, metadata)
    return layout, model, path, tensors, metadata


def test_all_parameters_and_temporal_kernel_match_original_converter_layout(checkpoint):
    layout, reference, path, _, _ = checkpoint
    plan = plan_mmproj(path, layout)
    reader = Reader()
    loaded = read_vision_weights(plan, reader, staging_bytes=256)
    expected = reference.state_dict()
    assert set(loaded) == set(expected)
    for name, wanted in expected.items():
        assert loaded[name].dtype == torch.float16
        torch.testing.assert_close(loaded[name], wanted, rtol=0, atol=0)
    assert plan.resident_bytes == sum(t.numel() * t.element_size() for t in loaded.values())
    assert reader.lengths and max(reader.lengths) <= 256
    assert loaded["patch_embed.proj.weight"].shape == (8, 3, 2, 2, 2)


def test_imported_tiny_hf_encoder_preserves_positions_merger_and_complete_output(checkpoint):
    layout, reference, path, _, _ = checkpoint
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    loaded = Qwen4ExpVisionModel(Qwen4ExpVisionConfig(**layout.config_fields())).eval()
    result = loaded.load_state_dict(read_vision_weights(plan_mmproj(path, layout), Reader(), staging_bytes=256),
                                    strict=True, assign=True)
    assert not result.missing_keys and not result.unexpected_keys
    # Two independent temporal segments with different spatial grids, including interpolation.
    grids = torch.tensor([[1, 2, 2], [2, 4, 2]], dtype=torch.int64)
    patches = torch.arange(20 * 24).reshape(20, 24).half() / 256
    with torch.inference_mode():
        actual = loaded(patches, grids)
        expected = reference(patches, grids)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
    torch.testing.assert_close(actual.pooler_output, expected.pooler_output, rtol=0, atol=0)
    assert actual.pooler_output.shape == (5, 6)


@pytest.mark.parametrize("corruption", ["deepstack", "metadata", "pixels", "missing", "duplicate", "extra", "shape"])
def test_layout_and_complete_name_failures_precede_parameter_allocation(checkpoint, corruption, monkeypatch):
    layout, _, path, tensors, metadata = checkpoint
    tensors, metadata = list(tensors), dict(metadata)
    if corruption == "deepstack":
        metadata["clip.vision.is_deepstack_layers"] = [True, False]
    elif corruption == "metadata":
        metadata["clip.vision.projection_dim"] = 7
    elif corruption == "pixels":
        metadata["clip.vision.image_mean"] = [.5, .4, .5]
    elif corruption == "missing":
        tensors.pop()
    elif corruption == "duplicate":
        tensors.append(tensors[-1])
    elif corruption == "extra":
        tensors.append(("unknown", (8,), 0, np.zeros(8, np.float32).view(np.uint8)))
    else:
        name, shape, kind, raw = tensors[0]
        tensors[0] = (name, (1, *shape), kind, raw)
    write(path, tensors, metadata)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("parameter allocation before layout validation"))
    with pytest.raises(ValueError):
        plan_mmproj(path, layout)


def test_import_staging_is_a_real_bound_and_fp16_overflow_is_not_clamped(checkpoint):
    layout, _, path, tensors, metadata = checkpoint
    with pytest.raises(ValueError, match="source row"):
        read_vision_weights(plan_mmproj(path, layout), Reader(), staging_bytes=1)
    for i, (name, shape, kind, raw) in enumerate(tensors):
        if name == "v.patch_embd.bias":
            values = np.zeros(shape, np.float32)
            values[0] = 1e10
            tensors[i] = (name, shape, 0, values.view(np.uint8))
            break
    write(path, tensors, metadata)
    with pytest.raises(ValueError, match="overflows FP16"):
        read_vision_weights(plan_mmproj(path, layout), Reader(), staging_bytes=256)


def test_meta_model_load_materializes_one_parameter_set_and_original_fp32_axial_buffers(checkpoint):
    layout, reference, path, _, _ = checkpoint
    model = load_cpu_vision_model(plan_mmproj(path, layout), Reader(), staging_bytes=256)
    for name, wanted in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], wanted, rtol=0, atol=0)
    assert all(parameter.dtype == torch.float16 and not parameter.requires_grad for parameter in model.parameters())
    assert model.rotary_pos_emb.inv_freq.dtype == torch.float32
    assert model.rotary_pos_emb.original_inv_freq.dtype == torch.float32
    assert not model.training and not any(buffer.is_meta for buffer in model.buffers())
    pixels = torch.zeros(4, 24, dtype=torch.float16)
    grid = torch.tensor([[1, 2, 2]])
    with torch.inference_mode():
        actual, expected = model(pixels, grid), reference(pixels, grid)
    torch.testing.assert_close(actual.pooler_output, expected.pooler_output, rtol=0, atol=0)


def test_cpu_preparation_overrides_enclosing_meta_device_without_changing_library_math(checkpoint):
    layout, reference, path, _, _ = checkpoint
    plan = plan_mmproj(path, layout)
    with torch.device("meta"):
        loaded = load_cpu_vision_model(plan, Reader(), staging_bytes=256)
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    assert all(buffer.device.type == "cpu" for buffer in loaded.buffers())
    torch.testing.assert_close(loaded.rotary_pos_emb.inv_freq, reference.rotary_pos_emb.inv_freq, rtol=0, atol=0)


def test_pinned_sdpa_forward_uses_only_supplied_grid_metadata(checkpoint):
    from transformers.vision_utils import (
        get_vision_cu_seqlens, get_vision_interpolation_indices_and_weights, get_vision_position_ids,
    )

    layout, _, path, _, _ = checkpoint
    model = load_cpu_vision_model(plan_mmproj(path, layout), Reader(), staging_bytes=256)
    grid = torch.tensor([[1, 2, 2], [2, 4, 2]])
    pixels = torch.arange(20 * 24).reshape(20, 24).half() / 256
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, model.num_grid_per_side, mode="bilinear", align_corners=True, spatial_merge_size=layout.merge)
    positions, lengths = get_vision_position_ids(grid, layout.merge), get_vision_cu_seqlens(grid)
    assert lengths.tolist() == [0, 4, 12, 20]
    with torch.inference_mode():
        expected = model(pixels, grid)
        actual = model(pixels, grid.to("meta"), interp_indices=indices, interp_weights=weights,
                       position_ids=positions, cu_seqlens=lengths)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
    torch.testing.assert_close(actual.pooler_output, expected.pooler_output, rtol=0, atol=0)
