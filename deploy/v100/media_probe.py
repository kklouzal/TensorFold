#!/usr/bin/env python3
"""Installed pinned CPU media/processor controls, never model-quality evidence."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile


def probe(configs):
    import av
    import numpy as np
    import PIL
    from PIL import features
    import torch
    import torchvision
    from transformers import AutoImageProcessor, AutoVideoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
    from transformers.video_utils import VideoMetadata

    expected = {"preprocessor_config.json": (390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
                "video_preprocessor_config.json": (385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13")}
    for name, (size, digest) in expected.items():
        raw = (configs / name).read_bytes()
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("processor configuration differs from the pinned official source")
    image = AutoImageProcessor.from_pretrained(configs, backend="torchvision", local_files_only=True)
    video = AutoVideoProcessor.from_pretrained(configs, local_files_only=True, cap_pixels_per_frame=False)
    if type(image) is not Qwen2VLImageProcessor or image.backend != "torchvision" or type(video) is not Qwen3VLVideoProcessor:
        raise ValueError("official image/video processor resolved a different class or backend")
    results = []
    for height, width, color, grid in ((256, 256, 255, [1, 16, 16]), (270, 330, 0, [1, 16, 20])):
        pixels = torch.full((3, height, width), color, dtype=torch.uint8)
        output = image(images=pixels, return_tensors="pt", device="cpu")
        if output["image_grid_thw"].tolist() != [grid]:
            raise ValueError("official image resize/grid contract changed")
        values = output["pixel_values"]
        wanted = 1.0 if color == 255 else -1.0
        if values.dtype != torch.float32 or values.shape != (grid[1] * grid[2], 1536) or not torch.equal(values, torch.full_like(values, wanted)):
            raise ValueError("CPU FP32 image rescale/normalization/patch contract changed")
        results.append({"input_hw": [height, width], "grid": grid, "constant_oracle": wanted})
    frames = torch.zeros((3, 3, 64, 64), dtype=torch.uint8)
    frames[1].fill_(255)
    metadata = VideoMetadata(total_num_frames=3, fps=2.0, frames_indices=[0, 1, 2])
    output = video(videos=frames, video_metadata=[metadata], do_sample_frames=False,
                   return_tensors="pt", cap_pixels_per_frame=False, device="cpu")
    if output["video_grid_thw"].tolist() != [[2, 4, 4]]:
        raise ValueError("official odd-frame temporal padding/grid contract changed")
    values = output["pixel_values_videos"]
    if values.dtype != torch.float32 or values.shape != (32, 1536) or not torch.equal(values[16:], torch.full_like(values[16:], -1)):
        raise ValueError("official final-frame repetition or video normalization changed")

    # A tiny independently specified uncompressed Y4M fixture needs no encoder
    # capability. CPU PyAV must return RGB pixels and reliable 2fps timestamps.
    with tempfile.TemporaryDirectory(prefix="tf-media-decoder-") as temporary:
        path = Path(temporary) / "constant.y4m"
        raw = bytearray(b"YUV4MPEG2 W4 H4 F2:1 Ip A1:1 C444\n")
        for index in range(4):
            raw += b"FRAME\n" + bytes([16 if index % 2 == 0 else 235]) * 16 + bytes([128]) * 32
        path.write_bytes(raw)
        with av.open(path) as container:
            stream = container.streams.video[0]
            if float(stream.average_rate) != 2.0:
                raise ValueError("PyAV lost explicit source FPS")
            decoded = []
            for index, frame in enumerate(container.decode(stream)):
                pixels = frame.to_ndarray(format="rgb24")
                expected_pixel = 0 if index % 2 == 0 else 255
                if not np.array_equal(pixels, np.full((4, 4, 3), expected_pixel, dtype=np.uint8)):
                    raise ValueError("CPU PyAV RGB conversion differs from the integer fixture")
                decoded.append(float(frame.pts * frame.time_base))
            if decoded != [0.0, 0.5, 1.0, 1.5]:
                raise ValueError("PyAV frame order/timestamps changed")

    # This external FATE bitstream has an independent libaom reference, rather
    # than expected pixels derived from the selected dav1d decoder itself.
    pins = json.loads((configs / "media-pins.json").read_text())
    fixture = next(row for row in pins["artifacts"] if row["name"] == "av1-decoder-fixture")
    path = configs / fixture["artifact"]
    raw = path.read_bytes()
    if len(raw) != fixture["bytes"] or hashlib.sha256(raw).hexdigest() != fixture["sha256"]:
        raise ValueError("AV1 decoder fixture differs from its independent pinned source")
    oracle = fixture["oracle"]
    digest, frame_count, pixel_bytes = hashlib.sha256(), 0, 0
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 4
        if stream.codec_context.name != "libdav1d":
            raise ValueError("CPU AV1 did not resolve the selected source-built dav1d provider")
        for frame in container.decode(stream):
            if (frame_count >= oracle["frames"] or frame.width != oracle["width"] or frame.height != oracle["height"]
                    or frame.pts * frame.time_base != Fraction(frame_count, oracle["fps"])):
                raise ValueError("AV1 decoded dimensions/count/timestamps differ from the independent fixture")
            pixels = frame.to_ndarray(format=oracle["format"]).tobytes()
            digest.update(pixels)
            pixel_bytes += len(pixels)
            frame_count += 1
    if frame_count != oracle["frames"] or pixel_bytes != oracle["bytes"] or digest.hexdigest() != oracle["sha256"]:
        raise ValueError("AV1 decoded pixels differ from the independent libaom reference")

    configuration = Qwen4ExpVisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                                       out_hidden_size=32, num_position_embeddings=16,
                                       patch_size=16, spatial_merge_size=2, temporal_patch_size=2)
    configuration._attn_implementation = "eager"
    torch.manual_seed(719)
    model = Qwen4ExpVisionModel(configuration).eval().half()
    with torch.inference_mode():
        result = model(torch.zeros((4, 1536), dtype=torch.float16), grid_thw=torch.tensor([[1, 2, 2]]))
    if (result.last_hidden_state.shape != (4, 32) or result.pooler_output.shape != (1, 32)
            or result.last_hidden_state.dtype != torch.float16 or result.pooler_output.dtype != torch.float16
            or not torch.isfinite(result.last_hidden_state).all() or not torch.isfinite(result.pooler_output).all()):
        raise ValueError("tiny actual HF FP16 vision model failed its shape/finite contract")
    return {"schema": "tensorfold.v100-media-software-probe", "version": 1,
            "scope": "CPU processor/decoder and tiny actual HF model; not SC117 quality or V100 qualification",
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "av", "pillow")},
            "image_class": type(image).__name__, "image_backend": image.backend, "video_class": type(video).__name__,
            "torchvision_cuda_version": torchvision.extension._check_cuda_version(),
            "ffmpeg_libraries": {name: list(value) for name, value in av.library_versions.items()},
            "pillow": {"version": PIL.__version__, "jpeg": features.version_codec("jpg"),
                       "zlib": features.version_codec("zlib"), "webp": features.version_module("webp")},
            "image_controls": results, "video_grid": output["video_grid_thw"].tolist(),
            "decoder_timestamps": decoded, "tiny_model_type": type(model).__name__,
            "av1_decoder": {"provider": "libdav1d", "frames": frame_count, "bytes": pixel_bytes, "pixel_sha256": digest.hexdigest()},
            "tiny_model_result_type": type(result).__name__,
            "tiny_model_shapes": {"last_hidden_state": list(result.last_hidden_state.shape),
                                  "pooler_output": list(result.pooler_output.shape)}, "release_qualified": False,
            "pending": ["alpha/grayscale/orientation and general codec parity", "bounded service integration",
                        "SC117/mmproj tensors and target equivalence", "V100 execution/resources/quality/performance",
                        "compiler-6111901"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.configs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
