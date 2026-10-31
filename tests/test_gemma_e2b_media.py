"""E2B transport boundaries and native CPU-processor compatibility."""

import base64
from dataclasses import replace
from fractions import Fraction
import io
import os
from pathlib import Path
import time
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from tensorfold.vision.audio import AudioInput, AudioLimits, audio_source, load_audios
from tensorfold.vision.images import ImageInput
from tensorfold.vision.images_http import ImageInputError
from tensorfold.vision.videos import VideoInput, VideoLimits, decode_video


def wav_source(samples, *, rate=16_000):
    samples = np.asarray(samples, dtype="<i2")
    channels = samples.shape[1] if samples.ndim == 2 else 1
    out = io.BytesIO()
    with wave.open(out, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())
    return audio_source({"data": base64.b64encode(out.getvalue()).decode(), "format": "wav"})


def encoded_video(format, presentation_indices):
    import av

    out = io.BytesIO()
    with av.open(out, mode="w", format=format) as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = stream.height = 48
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 0
        for i, pts in enumerate(presentation_indices):
            frame = av.VideoFrame.from_ndarray(np.full((48, 48, 3), i * 30, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out.getvalue()


@pytest.mark.parametrize("value", [
    None, {}, {"data": "", "format": "wav"}, {"data": "abcd", "format": "flac"},
    {"data": "é", "format": "wav"}, {"data": 4, "format": "wav"},
    {"data": "abcd", "format": []}, {"data": "abcd", "format": {}},
])
def test_audio_source_boundary(value):
    with pytest.raises(ImageInputError):
        audio_source(value)


def test_audio_encoded_limit_precedes_base64_allocation():
    with pytest.raises(ImageInputError, match="encoded byte limit"):
        audio_source({"data": "A" * 12, "format": "wav"}, AudioLimits(max_encoded_bytes=3))


def test_audio_invalid_base64_and_invalid_codec_data():
    with pytest.raises(ImageInputError, match="base64"):
        load_audios([audio_source({"data": "!!!!", "format": "wav"})], max_total_samples=480_000)
    pytest.importorskip("av")
    with pytest.raises(ImageInputError, match="invalid|unsupported"):
        load_audios([audio_source({"data": "YWJjZA==", "format": "mp3"})], max_total_samples=480_000)


def test_native_16k_pcm_decode_is_exact_and_request_owned():
    pytest.importorskip("av")
    pcm = np.arange(1600, dtype=np.int16) - 800
    result = load_audios([wav_source(pcm)], max_total_samples=480_000)[0]
    np.testing.assert_array_equal(result.samples, pcm.astype(np.float32) / 32768)
    assert not result.samples.flags.writeable
    samples = np.ones(100, dtype=np.float32)
    owned = AudioInput(samples)
    samples[:] = 0
    assert np.all(owned.samples == 1)


def test_stereo_mean_precedes_resampling_and_preserves_amplitude():
    pytest.importorskip("av")
    pcm = np.tile(np.array([16384, -8192], dtype=np.int16), (1600, 1))
    result = load_audios([wav_source(pcm)], max_total_samples=480_000)[0]
    np.testing.assert_array_equal(result.samples, np.full(1600, 0.125, dtype=np.float32))


def test_mp3_decode_matches_native_codec_waveform():
    av = pytest.importorskip("av")
    out = io.BytesIO()
    signal = (np.sin(np.arange(3200, dtype=np.float32) * (2 * np.pi * 440 / 16_000)) * 0.25)[None, :]
    with av.open(out, mode="w", format="mp3") as container:
        stream = container.add_stream("libmp3lame", rate=16_000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(signal, format="fltp", layout="mono")
        frame.sample_rate = 16_000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    data = out.getvalue()
    with av.open(io.BytesIO(data), mode="r", format="mp3") as container:
        expected = np.concatenate([frame.to_ndarray()[0] for frame in container.decode(container.streams.audio[0])])
    source = audio_source({"data": base64.b64encode(data).decode(), "format": "mp3"})
    result = load_audios([source], max_total_samples=480_000)[0]
    np.testing.assert_array_equal(result.samples, expected)


def test_audio_native_duration_and_cumulative_retention_limits():
    pytest.importorskip("av")
    with pytest.raises(ImageInputError, match="30 seconds"):
        load_audios([wav_source(np.zeros(480_001, dtype=np.int16))], max_total_samples=960_000)
    with pytest.raises(ImageInputError, match="sample budget"):
        load_audios([wav_source(np.zeros(1000, dtype=np.int16))] * 2, max_total_samples=1500)


def test_audio_finite_and_layout_contract():
    for samples in [np.array([np.nan], dtype=np.float32), np.ones((1, 10), dtype=np.float32),
                    np.ones(10, dtype=np.float64), np.array([], dtype=np.float32)]:
        with pytest.raises(ImageInputError):
            AudioInput(samples)
    with pytest.raises(ImageInputError):
        AudioInput(np.ones(2, dtype=np.float32), 2**40)
    with pytest.raises(ValueError, match="array range"):
        AudioLimits(max_seconds=1e308)


def test_cancelled_audio_never_starts_the_codec():
    from tensorfold.vision.replay import MediaCancelled
    with pytest.raises(MediaCancelled, match="cancelled"):
        load_audios([audio_source({"data": "YWJjZA==", "format": "wav"})],
                    max_total_samples=480_000, cancelled=lambda: True)


def test_video_optional_pts_does_not_change_legacy_hash():
    pixels = np.zeros((2, 48, 48, 3), dtype=np.uint8)
    legacy = VideoInput(pixels.copy(), (0, 1), 10.0)
    pts = VideoInput(pixels.copy(), (0, 1), 10.0, (0.0, 0.4))
    assert legacy.content_hash == pts.content_hash
    assert pts.timestamp_seconds == (0.0, 0.4)
    with pytest.raises(ImageInputError, match="presentation times"):
        VideoInput(pixels.copy(), (0, 1), 10.0, (0.4, 0.0))


@pytest.mark.parametrize("format", ["mp4", "matroska"])
def test_encoded_vfr_uses_actual_pts_and_unindexed_clips_have_exact_counts(format):
    av = pytest.importorskip("av")
    data = encoded_video(format, [0, 1, 2, 8, 10, 12])
    with av.open(io.BytesIO(data), mode="r") as source:
        actual = [float(frame.pts * frame.time_base) for frame in source.decode(source.streams.video[0])]
    result = decode_video(data, lambda _, h, w: (h, w), VideoLimits(), time.monotonic() + 20,
                          preserve_timestamps=True)
    expected = tuple(actual[i] - actual[0] for i in result.indices)
    assert result.timestamp_seconds == expected
    assert result.timestamp_seconds != tuple(i / result.fps for i in result.indices)
    assert len(result.indices) == 4


def test_video_without_nominal_rate_uses_measured_presentation_intervals(monkeypatch):
    av = pytest.importorskip("av")
    from tensorfold.vision import videos

    data = encoded_video("mp4", [0, 1, 2, 8, 10, 12])
    class NoRateContainer:
        def __init__(self, source):
            self.source = source
            stream = source.streams.video[0]
            proxy = SimpleNamespace(codec_context=stream.codec_context, frames=stream.frames,
                                    average_rate=None, guessed_rate=None, duration=stream.duration,
                                    time_base=stream.time_base)
            self.streams, self.duration = SimpleNamespace(video=[proxy]), source.duration
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.source.close()
        def decode(self, stream):
            return self.source.decode(self.source.streams.video[0])
    monkeypatch.setattr(videos, "_av", lambda: SimpleNamespace(
        open=lambda *args, **kwargs: NoRateContainer(av.open(*args, **kwargs))))
    result = decode_video(data, lambda _, h, w: (h, w), VideoLimits(), time.monotonic() + 20,
                          preserve_timestamps=True)
    assert result.fps == 5 / 0.48
    assert result.fps != 24
    assert result.timestamp_seconds[-1] == 0.48


def test_native_single_frame_video_is_opt_in_without_changing_legacy_minimum():
    pytest.importorskip("av")
    data = encoded_video("mp4", [0])
    result = decode_video(data, lambda _, h, w: (h, w), VideoLimits(), time.monotonic() + 20,
                          preserve_timestamps=True, minimum_frames=1)
    assert result.frames.shape[0] == 1 and result.timestamp_seconds == (0.0,)
    with pytest.raises(ImageInputError, match="fewer than two"):
        decode_video(data, lambda _, h, w: (h, w), VideoLimits(), time.monotonic() + 20)


@pytest.fixture(scope="module")
def native_frontend():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    root = os.environ.get("TF_GEMMA_E2B_TEST_MODEL")
    if not root or not (Path(root) / "processor_config.json").is_file():
        pytest.skip("set TF_GEMMA_E2B_TEST_MODEL to a local complete E2B processor snapshot")
    from tensorfold.vision.gemma4_e2b import GemmaE2BProcessor

    return GemmaE2BProcessor.from_directory(root)


def native_result(frontend, text, images=(), videos=(), audios=()):
    from transformers.video_utils import VideoMetadata

    kwargs = dict(text=text, return_tensors="pt",
                  text_kwargs={"add_special_tokens": False, "padding": False, "truncation": False})
    if images:
        kwargs["images"] = [[x.to_pil() for x in images]]
    if videos:
        kwargs.update(videos=[np.array(x.frames, copy=True) for x in videos], do_sample_frames=False,
                      video_metadata=[VideoMetadata(total_num_frames=max(x.indices) + 1, fps=x.fps,
                                                    frames_indices=list(x.indices)) for x in videos])
    if audios:
        kwargs["audio"] = [np.asarray(x.samples) for x in audios]
        kwargs["audio_kwargs"] = {"padding": "longest", "truncation": False, "max_length": 480_000,
                                 "pad_to_multiple_of": 128}
    return frontend.processor(**kwargs)


def assert_native_equal(prepared, reference):
    import torch

    assert prepared.token_ids == tuple(reference["input_ids"][0].tolist())
    owned_audio = {"input_features", "input_features_mask", "audio_feature_lengths"}
    for key, value in prepared.model_inputs.items():
        if key in owned_audio:
            continue
        assert key in reference, key
        assert value.dtype == reference[key].dtype, key
        assert torch.equal(value, reference[key]), key
    expected = {k for k, v in reference.items() if isinstance(v, torch.Tensor)} - {"input_ids"}
    assert set(prepared.model_inputs) - owned_audio == expected - owned_audio
    if "audio_feature_lengths" in prepared.model_inputs:
        lengths = prepared.model_inputs["audio_feature_lengths"]
        assert lengths.dtype == torch.int64 and lengths.ndim == 1
        assert len(lengths) == reference["input_features"].shape[0]
        cursor = 0
        for i, length in enumerate(lengths.tolist()):
            values = prepared.model_inputs["input_features"][0, cursor:cursor + length]
            mask = prepared.model_inputs["input_features_mask"][0, cursor:cursor + length]
            expected_mask = reference["input_features_mask"][i].bool()
            assert torch.equal(mask.bool().nonzero().flatten(), expected_mask.nonzero().flatten())
            assert torch.equal(values[mask.bool()], reference["input_features"][i][expected_mask])
            assert not torch.count_nonzero(values[~mask.bool()])
            assert not torch.count_nonzero(reference["input_features"][i][~expected_mask])
            cursor += length
        assert cursor == prepared.model_inputs["input_features"].shape[1]


def test_image_component_adapter_matches_whole_native_processor(native_frontend):
    image = ImageInput(48, 48, bytes(48 * 48 * 3))
    text = "Describe <|image|>."
    prepared = native_frontend.prepare(text, [image], max_prompt_tokens=131_072)
    assert_native_equal(prepared, native_result(native_frontend, text, images=[image]))
    assert prepared.visual_tokens > 0
    assert prepared.host_bytes > 0


def test_audio_component_adapter_matches_whole_native_processor(native_frontend):
    t = np.arange(3200, dtype=np.float32) / 16_000
    audio = AudioInput(np.sin(t * 2 * np.pi * 440).astype(np.float32))
    text = "Transcribe <|audio|>."
    prepared = native_frontend.prepare(text, [], audios=[audio], max_prompt_tokens=131_072)
    assert_native_equal(prepared, native_result(native_frontend, text, audios=[audio]))
    assert prepared.audio_tokens > 0


def test_variable_audio_batch_matches_native_padding_and_mixed_image(native_frontend):
    image = ImageInput(48, 48, bytes([100]) * (48 * 48 * 3))
    audios = [AudioInput(np.zeros(n, dtype=np.float32)) for n in [3200, 7200]]
    text = "<|audio|> Compare <|image|> with <|audio|>."
    prepared = native_frontend.prepare(text, [image], audios=audios, max_prompt_tokens=131_072)
    assert_native_equal(prepared, native_result(native_frontend, text, images=[image], audios=audios))


def test_video_component_adapter_matches_native_frame_axis(native_frontend):
    frames = np.zeros((3, 48, 48, 3), dtype=np.uint8)
    video = VideoInput(frames, (0, 1, 2), 2.0, (0.0, 0.5, 1.0))
    text = "Summarize <|video|>."
    prepared = native_frontend.prepare(text, [], videos=[video], max_prompt_tokens=131_072)
    reference = native_result(native_frontend, text, videos=[video])
    assert_native_equal(prepared, reference)


def test_unequal_vfr_clips_are_not_padded_dropped_or_given_fake_timestamps(native_frontend):
    first = VideoInput(np.zeros((2, 48, 48, 3), dtype=np.uint8), (0, 1), 30.0, (0.0, 1.7))
    second = VideoInput(np.full((3, 48, 48, 3), 128, dtype=np.uint8),
                        (0, 1, 2), 30.0, (0.0, 0.2, 2.1))
    prepared = native_frontend.prepare("<|video|> then <|video|>", [], videos=[first, second],
                                       max_prompt_tokens=131_072)
    assert prepared.model_inputs["pixel_values_videos"].shape[:2] == (1, 5)
    decoded = native_frontend.processor.tokenizer.decode(prepared.token_ids)
    assert "00:01" in decoded and "00:02" in decoded
    assert "00:00" in decoded
    assert len(prepared.video_hashes) == 2


def test_media_identity_and_plain_text_continuation(native_frontend):
    a = AudioInput(np.zeros(3200, dtype=np.float32))
    b = AudioInput(np.ones(3200, dtype=np.float32) * 0.25)
    one = native_frontend.prepare("<|audio|>", [], audios=[a], max_prompt_tokens=131_072)
    two = native_frontend.prepare("<|audio|>", [], audios=[b], max_prompt_tokens=131_072)
    assert one.media_key != two.media_key
    continued = one.continued((*one.token_ids, 42, 43))
    assert continued.media_key == one.media_key
    assert continued.model_inputs["attention_mask"].shape[-1] == len(one.token_ids) + 2
    with pytest.raises(ValueError, match="plain text"):
        one.continued((*one.token_ids, one.media_token_ids[2]))


def test_long_plus_many_tiny_clips_never_create_longest_batch_padding(native_frontend, monkeypatch):
    actual = native_frontend.processor.feature_extractor
    calls = []
    def one_clip_only(raw, **kwargs):
        assert len(raw) == 1
        calls.append((len(raw[0]), kwargs["max_length"]))
        return actual(raw, **kwargs)
    monkeypatch.setattr(native_frontend.processor, "feature_extractor", one_clip_only)
    # Keep the native analysis-window metadata on this callable wrapper.
    one_clip_only.frame_length = actual.frame_length
    one_clip_only.hop_length = actual.hop_length
    clips = [AudioInput(np.zeros(480_000, dtype=np.float32))] + [
        AudioInput(np.zeros(320, dtype=np.float32)) for _ in range(1000)]
    prepared = native_frontend.prepare("<|audio|>" * len(clips), [], audios=clips,
                                       max_prompt_tokens=131_072)
    assert len(calls) == len(clips)
    assert max(n for n, _ in calls[1:]) == 320
    assert max(pad for _, pad in calls[1:]) == actual.frame_length + 1
    assert prepared.model_inputs["input_features"].shape[1] < 10_000


def test_valid_short_and_mixed_duration_multiclip_contexts_are_preserved(native_frontend):
    short = [AudioInput(np.zeros(16_000, dtype=np.float32)) for _ in range(3)]
    text = "<|audio|>" * 3
    prepared = native_frontend.prepare(text, [], audios=short, max_prompt_tokens=2048)
    assert_native_equal(prepared, native_result(native_frontend, text, audios=short))
    mixed = [AudioInput(np.zeros(n, dtype=np.float32)) for n in [480_000, 320, 320]]
    prepared = native_frontend.prepare(text, [], audios=mixed, max_prompt_tokens=2048)
    assert_native_equal(prepared, native_result(native_frontend, text, audios=mixed))


@pytest.mark.parametrize("length", [1, 128, 129, 199, 200, 201])
def test_tiny_audio_native_frame_validity_and_mixed_clip_masks(native_frontend, length):
    audio = AudioInput(np.zeros(length, dtype=np.float32))
    native = native_result(native_frontend, "<|audio|>", audios=[audio])
    if native["input_features"].shape[1] == 0:
        with pytest.raises(ImageInputError, match="no native feature frames"):
            native_frontend.prepare("<|audio|>", [], audios=[audio], max_prompt_tokens=2048)
    else:
        prepared = native_frontend.prepare("<|audio|>", [], audios=[audio], max_prompt_tokens=2048)
        assert_native_equal(prepared, native)
    mixed = [audio, AudioInput(np.zeros(16_000, dtype=np.float32))]
    prepared = native_frontend.prepare("<|audio|><|audio|>", [], audios=mixed, max_prompt_tokens=2048)
    assert_native_equal(prepared, native_result(native_frontend, "<|audio|><|audio|>", audios=mixed))


def test_missing_media_marker_and_prompt_token_budget_are_explicit(native_frontend):
    image = ImageInput(48, 48, bytes(48 * 48 * 3))
    with pytest.raises(ImageInputError, match="exactly one marker"):
        native_frontend.prepare("no marker", [image], max_prompt_tokens=131_072)
    with pytest.raises(ImageInputError, match="context length"):
        native_frontend.prepare("<|image|>", [image], max_prompt_tokens=8)


def test_message_audio_transport_reaches_the_same_native_processor(native_frontend):
    pcm = np.arange(3200, dtype=np.int16) - 1600
    source = wav_source(pcm)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe "},
        {"type": "input_audio", "input_audio": {"data": source.data, "format": "wav"}},
    ]}]
    def render(template):
        assert template[0]["content"] == [{"type": "text", "text": "Transcribe "}, {"type": "audio"}]
        return "Transcribe <|audio|>"
    prepared = native_frontend.prepare_messages(messages, render, max_prompt_tokens=131_072)
    audio = AudioInput(pcm.astype(np.float32) / 32768)
    assert_native_equal(prepared, native_result(native_frontend, "Transcribe <|audio|>", audios=[audio]))


def test_app_image_count_override_does_not_mutate_frontend_limits(native_frontend):
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (48, 48)).save(out, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}} for _ in range(6)]}]
    prior = native_frontend.image_limits
    prepared = native_frontend.prepare_messages(messages, lambda _: "<|image|>" * 6,
        max_prompt_tokens=131_072, image_limits=replace(prior, max_images=6))
    assert len(prepared.image_hashes) == 6
    assert native_frontend.image_limits is prior
