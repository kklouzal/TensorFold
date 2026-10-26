"""Fresh exec-only CPU media worker. Internal descriptor protocol, never a server.

Hard process-global limits are installed before native decoders/processors are
imported. The supervisor owns the entire session until descendants retire.
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import warnings

from .images_http import ImageInputError
from .replay import MediaEnvelope, MediaProbe, ReplayRef, open_replay

PROCESSOR_ID = "qwen3vl-50e35cc0-cpu-fp32-global-t-actual-pts-v1"
_CONFIGS = Path("/opt/tensorfold-media-pins")
_HASHES = {"preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
           "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"}


def _processors(e):
    import torch
    import torchvision
    from transformers import AutoImageProcessor, AutoVideoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ImageInputError("CPU media worker must not own a CUDA context")
    if torchvision.extension._check_cuda_version() != -1:
        raise ImageInputError("CPU media requires the pinned CPU TorchVision backend")
    torch.set_num_threads(e.max_worker_threads)
    torch.set_num_interop_threads(1)
    for name, digest in _HASHES.items():
        if hashlib.sha256((_CONFIGS / name).read_bytes()).hexdigest() != digest:
            raise ImageInputError("pinned official media processor configuration changed")
    image = AutoImageProcessor.from_pretrained(_CONFIGS, backend="torchvision", local_files_only=True)
    video = AutoVideoProcessor.from_pretrained(_CONFIGS, local_files_only=True, cap_pixels_per_frame=False)
    if type(image) is not Qwen2VLImageProcessor or image.backend != "torchvision" or type(video) is not Qwen3VLVideoProcessor:
        raise ImageInputError("pinned processor class/backend is required")
    return image, video


def _allocation(size, e):
    if size > e.max_single_allocation_bytes:
        raise ImageInputError("media work unit exceeds the single allocation bound")


def _dimensions(width, height, e):
    if (min(width, height) <= 0 or max(width, height) > e.max_dimension
            or width * height > e.max_pixels_per_frame):
        raise ImageInputError("decoded media dimensions exceed the envelope")


@contextmanager
def _image(handle, e):
    from PIL import Image
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(handle, formats=("JPEG", "PNG", "WEBP")) as image:
            _dimensions(*image.size, e)
            if getattr(image, "n_frames", 1) != 1:
                raise ImageInputError("animated or multipage images require an explicit frame policy")
            # Four bytes per decoded Pillow pixel covers the accepted RGB,
            # RGBA, CMYK, grayscale/palette and 32-bit scalar image modes.
            size = image.width * image.height * 4
            _allocation(size, e)
            if image.width * image.height > e.max_total_decoded_pixels or size > e.max_decoded_bytes:
                raise ImageInputError("image decoded pixel/byte budget exceeded")
            yield image


def _av_limit(e):
    import av
    import ctypes
    # Bind the same owned libavutil already loaded by PyAV; never resolve an
    # ambient second copy by a soname. av_max_alloc is process-global, hence exec.
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if "/libavutil.so" in line and line.split()[-1].startswith("/")}
    if len(paths) != 1:
        raise ImageInputError("CPU decoder requires one exact owned libavutil")
    path = Path(next(iter(paths)))
    if not path.resolve().is_relative_to("/opt/tensorfold-media"):
        raise ImageInputError("CPU decoder libavutil is outside the owned media prefix")
    library = ctypes.CDLL(str(path))
    library.av_max_alloc.argtypes = (ctypes.c_size_t,)
    library.av_max_alloc.restype = None
    library.av_max_alloc(e.max_single_allocation_bytes)
    return av, library


@contextmanager
def _video(handle, e):
    av, library = _av_limit(e)
    options = {"max_pixels": str(e.max_pixels_per_frame), "probesize": str(e.max_probe_bytes),
               "analyzeduration": str(e.probe_microseconds),
               "protocol_whitelist": "file,pipe", "format_whitelist": "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm"}
    with av.open(handle, mode="r", options=options, timeout=(e.max_probe_seconds, e.max_worker_wall_seconds)) as container:
        if len(container.streams.video) != 1:
            raise ImageInputError("media requires exactly one video stream")
        stream = container.streams.video[0]
        _dimensions(stream.codec_context.width, stream.codec_context.height, e)
        stream.codec_context.thread_count = e.max_worker_threads
        stream.codec_context.options = {**stream.codec_context.options, "max_pixels": str(e.max_pixels_per_frame)}
        yield container, stream
    del library  # retains the exact process-global owner through decoder close


def stage(record, e, pixels_fd):
    from .images import _data_bytes
    from .images_http import MEDIA_TYPES, fetch_image
    from .videos import VIDEO_MEDIA_TYPES
    import time
    url, remaining = record["url"], record["remaining_bytes"]
    media_types = MEDIA_TYPES if record["kind"] == "image" else VIDEO_MEDIA_TYPES
    if type(remaining) is not int or not 0 < remaining <= e.max_upload_bytes:
        raise ImageInputError("invalid encoded staging byte allowance")
    if url.startswith("data:"):
        media = url[5:].partition(";")[0].partition(",")[0].lower()
        if media not in media_types:
            raise ImageInputError("media data URL type is outside the declared formats")
        raw = _data_bytes(url, remaining)
    else:
        raw, _ = fetch_image(url, max_bytes=remaining, deadline=time.monotonic() + e.max_worker_wall_seconds,
                             max_redirects=e.max_redirects, max_url_chars=e.max_url_chars, media_types=media_types)
    with os.fdopen(os.dup(pixels_fd), "wb") as target:
        target.write(raw)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "byte_length": len(raw)}


def _frames(container, stream, e):
    count = pixels = decoded_bytes = 0
    for packet_count, packet in enumerate(container.demux(stream), 1):
        if packet_count > e.max_scan_packets:
            raise ImageInputError("video packet scan budget exceeded")
        for frame in packet.decode():
            count += 1
            if count > e.max_scan_frames:
                raise ImageInputError("video frame scan budget exceeded")
            _dimensions(frame.width, frame.height, e)
            pixels += frame.width * frame.height
            size = sum(plane.buffer_size for plane in frame.planes)
            decoded_bytes += size
            if pixels > e.max_total_decoded_pixels or decoded_bytes > e.max_decoded_bytes:
                raise ImageInputError("video cumulative decoded pixel/byte budget exceeded")
            yield count - 1, frame, pixels, decoded_bytes


def _grid(kind, count, height, width):
    if kind == "image":
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        h, w = smart_resize(height, width, factor=32, min_pixels=65_536, max_pixels=16_777_216)
        return (h, w), (1, h // 16, w // 16)
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    h, w = smart_resize(count, height, width, temporal_factor=2, factor=32,
                        min_pixels=4_096, max_pixels=25_165_824)
    return (h, w), ((count + 1) // 2, h // 16, w // 16)


def probe(ref, e, processors):
    _, video = processors
    if ref.kind == "image":
        with open_replay(ref, e) as handle, _image(handle, e) as image:
            width, height = image.size
            image.verify()
        hw, grid = _grid("image", 1, height, width)
        result = MediaProbe(ref, e.identity, PROCESSOR_ID, width, height, 1, (), (), None,
                            "encoded-orientation-pil-rgb-v1", hw, grid, width * height, width * height * 4)
    else:
        with open_replay(ref, e) as handle, _video(handle, e) as (container, stream):
            width, height = stream.codec_context.width, stream.codec_context.height
            rate = stream.average_rate
            pts, pixels, decoded_bytes = [], 0, 0
            reliable = True
            for index, frame, pixels, decoded_bytes in _frames(container, stream, e):
                if (frame.width, frame.height) != (width, height):
                    raise ImageInputError("video resolution changes require an explicit processor policy")
                timestamp = Fraction(frame.pts) * frame.time_base if frame.pts is not None and frame.time_base else None
                reliable &= timestamp is not None and (not pts or pts[-1] is None or timestamp > pts[-1])
                pts.append(timestamp)
            total = len(pts)
            if total < 2:
                raise ImageInputError("one-frame video requires explicit still-image conversion")
            policy = "actual-pts-v1"
            if rate is None or rate <= 0 or not reliable:
                if e.assumed_fps is None:
                    raise ImageInputError("video requires reliable actual PTS and FPS or explicit assumed_fps")
                rate = Fraction(*e.assumed_fps)
                pts = [Fraction(i, 1) / rate for i in range(total)]
                policy = "explicit-assumed-fps-v1"
            rate = Fraction(rate)
            seconds = max(Fraction(total, 1) / rate, pts[-1] - pts[0])
            if stream.duration is not None and stream.time_base:
                seconds = max(seconds, stream.duration * stream.time_base)
            if seconds > e.max_video_seconds:
                raise ImageInputError("video duration exceeds the media envelope")
            from transformers.video_utils import VideoMetadata
            # Use the pinned sampling operation itself. FPS is never absent,
            # avoiding its upstream 24fps fallback; enforce service max after.
            indices = tuple(int(i) for i in video.sample_frames(VideoMetadata(total_num_frames=total, fps=float(rate))))
            if len(indices) > e.max_selected_frames:
                raise ImageInputError("selected video frames exceed the media envelope")
            selected_pts = tuple((pts[i].numerator, pts[i].denominator) for i in indices)
        hw, grid = _grid("video", len(indices), height, width)
        result = MediaProbe(ref, e.identity, PROCESSOR_ID, width, height, total, indices, selected_pts,
                            (rate.numerator, rate.denominator), policy, hw, grid, pixels, decoded_bytes)
    if result.visual_rows > e.max_expanded_rows:
        raise ImageInputError("media expanded visual rows exceed the envelope")
    _unit_allocation(result, e)
    return result


def _unit_allocation(probed, e):
    h, w = probed.resized_hw
    # Largest explicit tensor: source conversion, resized FP32 pair, or final
    # temporal patch rows. Native library allocator internals are bounded by AS.
    for size in (2 * 3 * probed.height * probed.width * 4, 2 * 3 * h * w * 4,
                 probed.grid_thw[1] * probed.grid_thw[2] * 1536 * 4):
        _allocation(size, e)


def pixel_unit(probed, index, e, processors):
    import torch
    image_processor, video_processor = processors
    if probed.envelope_id != e.identity or probed.processor_id != PROCESSOR_ID:
        raise ImageInputError("media probe processor or envelope identity changed")
    if type(index) is not int or not 0 <= index < probed.unit_count:
        raise ImageInputError("pixel unit index is outside the media grid")
    _unit_allocation(probed, e)
    if probed.ref.kind == "image":
        with open_replay(probed.ref, e) as handle, _image(handle, e) as image:
            if image.size != (probed.width, probed.height):
                raise ImageInputError("replay image dimensions changed")
            # Pinned supplied-PIL semantics, including alpha/grayscale and
            # encoded orientation; no unrelated eager decoder transformation.
            output = image_processor(images=image, return_tensors="pt", device="cpu")
            values, grid = output["pixel_values"], output["image_grid_thw"]
    else:
        first = 2 * index
        positions = probed.selected_indices[first:first + 2]
        expected = dict(zip(positions, probed.selected_pts[first:first + 2]))
        kept = []
        with open_replay(probed.ref, e) as handle, _video(handle, e) as (container, stream):
            for source_index, frame, _, _ in _frames(container, stream, e):
                if source_index not in expected:
                    continue
                if (frame.width, frame.height) != (probed.width, probed.height):
                    raise ImageInputError("replayed video dimensions changed")
                if probed.timestamp_policy == "actual-pts-v1":
                    if frame.pts is None or not frame.time_base or frame.pts * frame.time_base != Fraction(*expected[source_index]):
                        raise ImageInputError("replayed video actual timestamp changed")
                array = frame.to_ndarray(format="rgb24")
                kept.append(torch.from_numpy(array).permute(2, 0, 1))
                if len(kept) == len(positions):
                    break
        if len(kept) != len(positions):
            raise ImageInputError("video replay ended before the selected temporal unit")
        frames = torch.stack(kept)
        from torchvision.transforms.v2 import functional as tvf
        from torchvision.transforms import InterpolationMode
        # Global T resize was computed once at probe time. Pair-local resize
        # selection would change the pinned total temporal-spatial budget.
        frames = tvf.resize(frames, list(probed.resized_hw), interpolation=InterpolationMode.BICUBIC, antialias=True)
        output = video_processor(videos=frames, do_sample_frames=False, do_resize=False,
                                 cap_pixels_per_frame=False, return_tensors="pt", device="cpu")
        values, grid = output["pixel_values_videos"], output["video_grid_thw"]
    expected_grid = [1, probed.grid_thw[1], probed.grid_thw[2]]
    if (grid.tolist() != [expected_grid] or values.dtype != torch.float32 or values.device.type != "cpu"
            or tuple(values.shape) != (expected_grid[1] * expected_grid[2], 1536)):
        raise ImageInputError("pinned processor output disagrees with the admitted grid")
    return values.contiguous()


def main():
    request_fd, result_fd, pixels_fd, diagnostic_fd, request_bound, cpu_bound, address_bound, output_limit = map(int, sys.argv[1:9])
    # These small trusted owner arguments constrain parsing of the descriptor
    # manifest too; no untrusted URL JSON is parsed before the hard limits.
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_bound, cpu_bound))
    resource.setrlimit(resource.RLIMIT_AS, (address_bound, address_bound))
    resource.setrlimit(resource.RLIMIT_FSIZE, (output_limit, output_limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.dup2(diagnostic_fd, 1)
    os.dup2(diagnostic_fd, 2)
    if os.fstat(request_fd).st_size > request_bound:
        raise ImageInputError("media worker request manifest exceeds its bound")
    with os.fdopen(os.dup(request_fd), "rb") as request:
        record = json.load(request)
    e = MediaEnvelope.from_record(record["envelope"])
    if cpu_bound != e.max_worker_cpu_seconds or address_bound != e.max_worker_address_bytes or request_bound != e.max_manifest_bytes:
        raise ImageInputError("worker hard limits disagree with the admitted envelope")
    try:
        if record["operation"] == "stage":
            output = stage(record, e, pixels_fd)
        elif record["operation"] == "probe":
            processors = _processors(e)
            output = {"probe": probe(ReplayRef(**record["ref"]), e, processors).record()}
        elif record["operation"] == "pixel_unit":
            processors = _processors(e)
            values = pixel_unit(MediaProbe.from_record(record["probe"]), record["index"], e, processors)
            with os.fdopen(os.dup(pixels_fd), "wb") as target:
                values.numpy().tofile(target)
            output = {"shape": list(values.shape), "dtype": "float32", "bytes": values.numel() * 4}
        else:
            raise ImageInputError("unknown CPU media worker operation")
    except Exception as error:
        # Bounded context, no input URLs, paths or payloads in external errors.
        output = {"error": type(error).__name__, "message": str(error)[:512]}
    raw = json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > e.max_result_bytes:
        raise ImageInputError("media worker result exceeds its bound")
    with os.fdopen(os.dup(result_fd), "wb") as target:
        target.write(raw)


if __name__ == "__main__":
    main()
