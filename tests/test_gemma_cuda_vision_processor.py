"""Exercise the installed standard processor on owned image/video data, without model weights."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers.models.gemma4.processing_gemma4")

from transformers import GemmaTokenizer  # noqa: E402
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor  # noqa: E402
from transformers.models.gemma4.image_processing_gemma4 import (  # noqa: E402
    Gemma4ImageProcessor, get_aspect_ratio_preserving_size,
)
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor  # noqa: E402
from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor  # noqa: E402

from tensorfold.families.gemma4.cuda.vision import GemmaVisionProcessor  # noqa: E402
from tensorfold.vision.images import ImageInput  # noqa: E402
from tensorfold.vision.videos import VideoInput  # noqa: E402
from test_gemma_cuda_vision_contracts import configuration  # noqa: E402


def standard_processor():
    names = {"image_token": "<|image|>", "boi_token": "<|image>", "eoi_token": "<image|>",
             "audio_token": "<|audio|>", "boa_token": "<|audio>", "eoa_token": "<audio|>"}
    symbols = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", *names.values(), "<|video|>", "▁", ":",
               *"0123456789"]
    tokenizer = GemmaTokenizer(vocab={token: i for i, token in enumerate(symbols)},
                              model_specific_special_tokens=names, extra_special_tokens=["<|video|>"])
    image = Gemma4ImageProcessor(patch_size=2, pooling_kernel_size=2, max_soft_tokens=70)
    video = Gemma4VideoProcessor(patch_size=2, pooling_kernel_size=2, max_soft_tokens=70)
    processor = Gemma4Processor(Gemma4AudioFeatureExtractor(), image, tokenizer, video, image_seq_length=70)
    raw = configuration()
    raw["image_token_id"] = tokenizer.convert_tokens_to_ids(names["image_token"])
    raw["video_token_id"] = tokenizer.convert_tokens_to_ids("<|video|>")
    return raw, processor


def test_standard_processor_images_and_video_frames_preserve_native_geometry_and_timestamps():
    raw, processor = standard_processor()
    frontend = GemmaVisionProcessor(raw, processor, get_aspect_ratio_preserving_size)
    pixels = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    image = ImageInput(8, 4, pixels.tobytes())
    video = VideoInput(np.stack([pixels, 255 - pixels]), (3, 9), 2.0)
    prepared = frontend.prepare("<|video|> <|image|>", [image], videos=[video], max_prompt_tokens=512)
    assert [frame.kind for frame in prepared.frames] == ["video", "video", "image"]
    assert len(prepared.blocks) == 3
    for frame, (start, end) in zip(prepared.frames, prepared.blocks):
        assert end - start == frame.soft_tokens
        assert all(token == raw["video_token_id" if frame.kind == "video" else "image_token_id"]
                   for token in prepared.token_ids[start:end])
        assert frame.positions.shape == (280, 2)
        assert not frame.pixels.flags.writeable
    decoded = processor.tokenizer.decode(prepared.token_ids, skip_special_tokens=False)
    assert "00:01" in decoded and "00:04" in decoded
    assert prepared.image_hashes == (image.content_hash,)
    assert prepared.video_hashes == (video.content_hash,)


def test_local_processor_roundtrip_loads_checkpoint_data_without_model_code(tmp_path):
    raw, processor = standard_processor()
    processor.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    frontend = GemmaVisionProcessor.from_directory(tmp_path)
    image = ImageInput(8, 4, bytes(4 * 8 * 3))
    prepared = frontend.prepare("<|image|>", [image], max_prompt_tokens=128)
    assert len(prepared.frames) == 1 and len(prepared.blocks) == 1
    assert prepared.visual_tokens == prepared.frames[0].soft_tokens
    # Harness counts complete history with no generation context-window gate.
    # Its None sentinel must retain every visual safety limit.
    counted = frontend.prepare("<|image|>", [image], max_prompt_tokens=None)
    assert counted.token_ids == prepared.token_ids and counted.visual_tokens == prepared.visual_tokens
    with pytest.raises(ValueError, match="context length"):
        frontend.prepare("<|image|>", [image], max_prompt_tokens=len(prepared.token_ids) - 1)
    with pytest.raises(ValueError, match="visual-token budget"):
        frontend.prepare("<|image|>", [image], max_prompt_tokens=None,
                         max_visual_tokens=prepared.visual_tokens - 1)


def test_saved_tokenizer_batch_policies_cannot_pad_or_truncate_expanded_media(tmp_path):
    raw, processor = standard_processor()
    backend = processor.tokenizer.backend_tokenizer
    backend.enable_padding(direction="left", length=4096, pad_id=0, pad_token="<pad>")
    backend.enable_truncation(max_length=8)
    processor.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    tokenizer_path = tmp_path / "tokenizer.json"
    original = tokenizer_path.read_bytes()
    frontend = GemmaVisionProcessor.from_directory(tmp_path)
    image = ImageInput(8, 4, bytes(4 * 8 * 3))
    text = "<|image|>" + " 1234" * 20
    counted = frontend.prepare(text, [image], max_prompt_tokens=None)
    assert 8 < len(counted.token_ids) < 4096
    assert 0 not in counted.token_ids
    assert counted.visual_tokens == counted.frames[0].soft_tokens
    decoded = frontend.processor.tokenizer.decode(counted.token_ids, skip_special_tokens=False)
    assert decoded.replace(" ", "").endswith("1234" * 20)
    generated = frontend.prepare(text, [image], max_prompt_tokens=len(counted.token_ids))
    assert generated.token_ids == counted.token_ids
    with pytest.raises(ValueError, match="context length"):
        frontend.prepare(text, [image], max_prompt_tokens=len(counted.token_ids) - 1)
    assert tokenizer_path.read_bytes() == original


def test_combined_mlx_metadata_needs_no_standalone_video_file_and_preserves_native_defaults(tmp_path):
    """The real checkpoint packages image/audio settings together and omits video settings."""
    raw, processor = standard_processor()
    raw["vision_config"].update(patch_size=16, pooling_kernel_size=3, position_embedding_size=10240)
    processor.tokenizer.save_pretrained(tmp_path)
    metadata = {"processor_class": "Gemma4Processor", "image_processor": Gemma4ImageProcessor().to_dict(),
                "feature_extractor": {"feature_extractor_type": "Gemma4AudioFeatureExtractor",
                                      "sampling_rate": 16000, "num_mel_filters": 128, "fft_length": 512,
                                      "hop_length": 160, "chunk_duration": 8.0, "overlap_duration": 1.0},
                "image_seq_length": 280,
                "audio_seq_length": 750, "audio_ms_per_token": 40}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "processor_config.json").write_text(json.dumps(metadata))
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    assert not (tmp_path / "video_preprocessor_config.json").exists()
    frontend = GemmaVisionProcessor.from_directory(tmp_path)
    assert (frontend.geometry.image_soft_tokens, frontend.geometry.video_soft_tokens) == (280, 70)
    assert frontend.processor.image_seq_length == 280
    assert frontend.processor.audio_seq_length == 750 and frontend.processor.audio_ms_per_token == 40
    assert frontend.processor.feature_extractor.sampling_rate == 16000
    assert frontend.processor.chat_template == "{{ messages }}"
    prepared = frontend.prepare("<|image|>", [ImageInput(48, 48, bytes(48 * 48 * 3))], max_prompt_tokens=512)
    assert len(prepared.frames) == 1 and prepared.visual_tokens == 256
