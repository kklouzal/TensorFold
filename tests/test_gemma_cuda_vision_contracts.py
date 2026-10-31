"""Gemma native vision schemas, processor geometry and request ownership."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.families.gemma4.cuda.vision import (
    GemmaVisionProcessor, PreparedGemmaVision, _frame, _video_batch, continued, media_blocks, processor_geometry, projector_spec,
    validate_vision_config, vision_shapes, vision_weight_bytes,
)


def configuration():
    return {"model_type": "gemma4", "image_token_id": 90, "video_token_id": 91,
            "text_config": {"hidden_size": 24}, "vision_config": {
                "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                "patch_size": 2, "pooling_kernel_size": 2, "position_embedding_size": 32,
                "rope_parameters": {"rope_type": "default", "rope_theta": 100.0},
                "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6,
                "standardize": True, "attention_bias": False, "use_clipped_linears": False}}


def patches(width=4, height=2, padded=16):
    xy = np.full((padded, 2), -1, dtype=np.int64)
    xy[:width * height, 0] = np.arange(width * height) % width
    xy[:width * height, 1] = np.arange(width * height) // width
    pixels = np.linspace(0, 1, padded * 12, dtype=np.float32).reshape(padded, 12)
    pixels[width * height:] = 0
    return pixels, xy


def test_float_tower_schema_and_byte_accounting_preserve_projector_separation():
    raw = configuration()
    shapes = vision_shapes(raw)
    headers = {key: {"shape": shape, "dtype": "BF16"} for key, shape in shapes.items()}
    headers["embed_vision.embedding_projection.weight"] = {"shape": (24, 2), "dtype": "U32"}
    assert vision_weight_bytes(raw, headers) == sum(np.prod(s) * 2 for s in shapes.values())
    assert shapes["vision_tower.patch_embedder.position_embedding_table"] == (2, 32, 16)
    for damage in ("missing", "extra", "packed", "shape", "mixed_dtype"):
        broken = deepcopy(headers)
        key = "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"
        if damage == "missing":
            broken.pop(key)
        elif damage == "extra":
            broken[key.removesuffix("weight") + "scales"] = {"shape": (16, 1), "dtype": "BF16"}
        elif damage == "packed":
            broken[key]["dtype"] = "U32"
        elif damage == "mixed_dtype":
            broken[key]["dtype"] = "F32"
        else:
            broken[key]["shape"] = (16, 8)
        with pytest.raises(ValueError):
            vision_weight_bytes(raw, broken)


@pytest.mark.parametrize("bits", [4, 6])
def test_projector_preserves_its_module_override_rather_than_global_quantization(bits):
    raw = configuration()
    raw["vision_config"].update(hidden_size=64, num_attention_heads=8, num_key_value_heads=8)
    raw["quantization"] = {"bits": 4, "group_size": 32,
                           "embed_vision.embedding_projection": {"bits": bits, "group_size": 64}}
    prefix = "embed_vision.embedding_projection."
    headers = {prefix + "weight": {"shape": (24, 2 * bits), "dtype": "U32"},
               prefix + "scales": {"shape": (24, 1), "dtype": "BF16"},
               prefix + "biases": {"shape": (24, 1), "dtype": "BF16"}}
    assert projector_spec(raw, headers) == (bits, 64)
    headers[prefix + "weight"]["shape"] = (24, 8 if bits == 6 else 12)
    with pytest.raises(ValueError, match="projector tensor"):
        projector_spec(raw, headers)


@pytest.mark.parametrize("field,value", [("head_dim", 7), ("num_key_value_heads", 1),
                                        ("pooling_kernel_size", 0), ("use_clipped_linears", True),
                                        ("hidden_activation", "silu"), ("rms_norm_eps", float("nan"))])
def test_unsupported_vision_contract_rejects_before_payload_allocation(field, value):
    raw = configuration()
    raw["vision_config"][field] = value
    with pytest.raises(ValueError):
        validate_vision_config(raw)


def test_mixed_media_blocks_keep_frames_separate_and_detect_missing_extra_or_wrong_runs():
    tokens = [1, 91, 91, 2, 90, 90, 90, 3, 91, 91, 4]
    assert media_blocks(tokens, 90, 91, [("video", 2), ("image", 3), ("video", 2)]) == (
        (1, 3), (4, 7), (8, 10))
    for frames in [[("video", 3)], [("image", 2)], [("video", 2), ("image", 3)],
                   [("video", 2), ("image", 3), ("video", 3)]]:
        with pytest.raises(ValueError):
            media_blocks(tokens, 90, 91, frames)


def test_standard_processor_arrays_are_owned_and_frozen():
    p, xy = patches()
    frame = _frame(p, xy, configuration()["vision_config"], "image")
    assert frame.soft_tokens == 2
    assert not np.shares_memory(p, frame.pixels) and not np.shares_memory(xy, frame.positions)
    assert not frame.pixels.flags.writeable and not frame.positions.flags.writeable
    p.fill(1)
    xy.fill(7)
    assert frame.positions[0].tolist() == [0, 0]
    assert frame.pixels[-1].tolist() == [0] * 12


@pytest.mark.parametrize("damage", ["partial_pad", "interior_pad", "reordered", "bounds", "nan", "shape"])
def test_processor_geometry_rejects_invalid_positions_and_pixel_boundaries(damage):
    p, xy = patches()
    if damage == "partial_pad":
        xy[-1] = [-1, 0]
    elif damage == "interior_pad":
        xy[1] = [-1, -1]
    elif damage == "reordered":
        xy[[1, 2]] = xy[[2, 1]]
    elif damage == "bounds":
        xy[0] = [32, 0]
    elif damage == "nan":
        p[0, 0] = np.nan
    else:
        p = p[:, :-1]
    with pytest.raises(ValueError):
        _frame(p, xy, configuration()["vision_config"], "image")


class Processor:
    image_token, video_token = "<image>", "<video>"

    def __init__(self):
        component = dict(patch_size=2, pooling_kernel_size=2, max_soft_tokens=4, do_resize=True,
                         do_rescale=True, do_normalize=False, rescale_factor=1 / 255)
        self.image_processor = SimpleNamespace(**component)
        self.video_processor = SimpleNamespace(**component)
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda text: {"<image>": 90, "<video>": 91}[text])
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        p, xy = patches()
        return {"input_ids": np.array([[1, 90, 90, 2]]),
                "pixel_values": p[None], "image_position_ids": xy[None]}


def test_budget_is_checked_before_processor_patch_allocation_and_tokens_match_exact_geometry():
    processor = Processor()
    front = GemmaVisionProcessor(configuration(), processor, lambda **kwargs: (4, 8))
    image = SimpleNamespace(width=8, height=4, content_hash="abc", to_pil=lambda: None)
    with pytest.raises(ValueError, match="visual-token budget"):
        front.prepare("<image>", [image], max_prompt_tokens=10, max_visual_tokens=1)
    assert processor.calls == 0
    prepared = front.prepare("<image>", [image], max_prompt_tokens=4)
    assert prepared.token_ids == (1, 90, 90, 2)
    assert prepared.blocks == ((1, 3),) and prepared.visual_tokens == 2
    assert prepared.image_hashes == ("abc",) and prepared.video_hashes == ()
    counted = front.prepare("<image>", [image], max_prompt_tokens=None)
    assert counted.token_ids == prepared.token_ids and counted.visual_tokens == prepared.visual_tokens
    calls = processor.calls
    with pytest.raises(ValueError, match="visual-token budget"):
        front.prepare("<image>", [image], max_prompt_tokens=None, max_visual_tokens=1)
    assert processor.calls == calls
    with pytest.raises(ValueError, match="context length"):
        front.prepare("<image>", [image], max_prompt_tokens=3)
    with pytest.raises(ValueError, match="one marker"):
        front.prepare("text", [image], max_prompt_tokens=10)


def test_video_frame_count_is_charged_to_budget_before_any_processor_work():
    processor = Processor()
    front = GemmaVisionProcessor(configuration(), processor, lambda **kwargs: (4, 8))
    video = SimpleNamespace(frames=np.zeros((16, 4, 8, 3), dtype=np.uint8), indices=tuple(range(16)))
    with pytest.raises(ValueError, match="visual-token budget"):
        front.prepare("<video>", [], videos=[video], max_prompt_tokens=100, max_visual_tokens=16)
    assert processor.calls == 0


def test_initial_video_region_rejects_unequal_sampled_counts_before_processor_work():
    processor = Processor()
    front = GemmaVisionProcessor(configuration(), processor, lambda **kwargs: (4, 8))
    videos = [SimpleNamespace(frames=np.zeros((n, 4, 8, 3), dtype=np.uint8), indices=tuple(range(n)))
              for n in (2, 3)]
    with pytest.raises(ValueError, match="same sampled frame count"):
        front.prepare("<video> <video>", [], videos=videos, max_prompt_tokens=100)
    assert processor.calls == 0


def test_pinned_video_identity_normalization_is_valid_but_nonidentity_is_rejected():
    processor = Processor()
    processor.video_processor.do_normalize = True
    processor.video_processor.image_mean = [0.0, 0.0, 0.0]
    processor.video_processor.image_std = [1.0, 1.0, 1.0]
    GemmaVisionProcessor(configuration(), processor, lambda **kwargs: (4, 8))
    processor.video_processor.image_mean = [0.1, 0.0, 0.0]
    with pytest.raises(ValueError, match="video_processor"):
        GemmaVisionProcessor(configuration(), processor, lambda **kwargs: (4, 8))


def test_pinned_video_batch_shape_is_proved_before_flattening_declared_frames():
    pixels = np.arange(2 * 3 * 16 * 12, dtype=np.float32).reshape(2, 3, 16, 12)
    positions = np.zeros((2, 3, 16, 2), dtype=np.int64)
    flat, xy = _video_batch(pixels, positions, [3, 3])
    assert flat.shape == (6, 16, 12) and xy.shape == (6, 16, 2)
    np.testing.assert_array_equal(flat[3], pixels[1, 0])
    for counts in ([3], [3, 2], [3, 4]):
        with pytest.raises(ValueError, match="declared video/frame"):
            _video_batch(pixels, positions, counts)
    with pytest.raises(ValueError, match="declared video/frame"):
        _video_batch(pixels.reshape(6, 16, 12), positions.reshape(6, 16, 2), [3, 3])


def test_processor_geometry_uses_checkpoint_fields_then_pinned_class_defaults_without_torch(tmp_path):
    pytest.importorskip("tensorfold._fd_owner", reason="requires the packaged descriptor ownership extension")
    raw = configuration()
    raw["vision_config"].update(patch_size=16, pooling_kernel_size=3)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    default = processor_geometry(tmp_path)
    assert (default.image_soft_tokens, default.video_soft_tokens) == (280, 70)
    assert (default.image_patches, default.video_patches) == (2520, 630)
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": {"max_soft_tokens": 560},
                                                               "video_processor": {"max_soft_tokens": 140}}))
    chosen = processor_geometry(tmp_path)
    assert (chosen.image_patches, chosen.video_patches) == (5040, 1260)
    (tmp_path / "processor_config.json").write_text('{"image_processor":{"max_soft_tokens":71}}')
    with pytest.raises(ValueError, match="geometry"):
        processor_geometry(tmp_path)
    (tmp_path / "processor_config.json").write_text('{"image_processor":{},"image_processor":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        processor_geometry(tmp_path)


def test_text_continuations_preserve_original_media_identity_and_reject_added_or_changed_media():
    frame = _frame(*patches(), configuration()["vision_config"], "image")
    prepared = PreparedGemmaVision((1, 90, 90, 2), (frame,), ((1, 3),), ("image",), ())
    extended = continued(prepared, [1, 90, 90, 2, 3, 4], configuration())
    assert extended.token_ids == (1, 90, 90, 2, 3, 4)
    assert extended.frames is prepared.frames and extended.blocks is prepared.blocks
    assert extended.image_hashes is prepared.image_hashes
    for tokens in ([1, 90, 2], [1, 90, 90, 3], [1, 90, 90, 2, 91]):
        with pytest.raises(ValueError):
            continued(prepared, tokens, configuration())
