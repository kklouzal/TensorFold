"""Actual pinned CPU software worker controls; no GPU/model/performance claim."""
from pathlib import Path
import threading

import pytest

from tensorfold.vision.images_http import ImageInputError
from tensorfold.vision.media_frontend import MediaExecutor, ReplayMediaFrontend
from tensorfold.vision.replay import MediaCancelled, ReplayStore
from tests.test_media_replay import envelope, image_source, video_source

pytestmark = pytest.mark.skipif(not Path("/opt/tensorfold-media-pins/media-install.json").is_file(),
                                reason="requires the pinned source CPU media image")


@pytest.mark.parametrize("mode,color,orientation", [("RGB", (11, 71, 201), None), ("RGBA", (11, 71, 201, 0), None),
                                                   ("L", 71, None), ("RGB", (11, 71, 201), 6)])
def test_exec_image_rows_match_independent_whole_official_processor(tmp_path, mode, color, orientation):
    import io
    import base64
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor
    source = image_source(mode, color, orientation=orientation)
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    lease = store.stage([source], allow_urls=False, cancelled=lambda: False)
    probed = executor.probe_replay(lease.refs[0], cancelled=lambda: False)
    assert probed.grid_thw == (1, 14, 20) and probed.timestamp_policy == "encoded-orientation-pil-rgb-v1"
    official = AutoImageProcessor.from_pretrained("/opt/tensorfold-media-pins", backend="torchvision", local_files_only=True)
    with Image.open(io.BytesIO(base64.b64decode(source.url.split(",", 1)[1]))) as image:
        expected = official(images=image, return_tensors="pt", device="cpu")["pixel_values"].numpy()
    pixels = executor.pixel_unit(probed, 0, cancelled=lambda: False)
    held = pixels.borrow()
    actual = held.values
    assert not actual.flags.writeable and np.array_equal(actual, expected)
    with pytest.raises(ValueError):
        actual[0, 0] = 0
    with pytest.raises(RuntimeError, match="borrowers"):
        pixels.close()
    with pytest.raises(RuntimeError, match="borrowers"):
        lease.close()
    del actual
    held.close()
    pixels.close()
    lease.close()
    assert store._staging == store._bytes == 0


def test_retired_borrow_with_live_export_refuses_mapping_release(tmp_path):
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    lease = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    probed = executor.probe_replay(lease.refs[0], cancelled=lambda: False)
    pixels = executor.pixel_unit(probed, 0, cancelled=lambda: False)
    borrowed = pixels.borrow()
    exported = borrowed.values
    borrowed.close()
    with pytest.raises(BufferError):
        pixels.close()
    assert exported.shape == pixels.shape and store._staging > 0 and lease._borrows == 1
    del exported
    pixels.close()
    lease.close()


def test_exec_temporal_pairs_equal_whole_pinned_video_and_repeat_odd_last(tmp_path):
    import numpy as np
    import torch
    from transformers import AutoVideoProcessor
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    lease = store.stage([video_source()], allow_urls=False, cancelled=lambda: False)
    probed = executor.probe_replay(lease.refs[0], cancelled=lambda: False)
    assert probed.selected_indices == (0, 1, 2, 3, 4)
    assert probed.selected_pts == ((0, 1), (1, 2), (1, 1), (3, 2), (2, 1))
    assert probed.timestamp_markers() == ("<0.2 seconds>", "<1.2 seconds>", "<2.0 seconds>")
    assert probed.grid_thw == (3, 4, 4)
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (23, 47, 89), (255, 255, 255))
    frames = torch.stack([torch.tensor(color, dtype=torch.uint8).view(3, 1, 1).expand(3, 48, 64) for color in colors])
    official = AutoVideoProcessor.from_pretrained("/opt/tensorfold-media-pins", local_files_only=True,
                                                  cap_pixels_per_frame=False)
    expected = official(videos=frames, do_sample_frames=False, cap_pixels_per_frame=False,
                        return_tensors="pt", device="cpu")["pixel_values_videos"].numpy()
    pieces = []
    for index in range(3):
        with executor.pixel_unit(probed, index, cancelled=lambda: False) as pixels:
            with pixels.borrow() as borrowed:
                pieces.append(borrowed.values.copy())
    assert np.array_equal(np.concatenate(pieces), expected)
    lease.close()


@pytest.mark.parametrize("changes,match", [({"max_pixels_per_frame": 100}, "dimensions"),
    ({"max_scan_frames": 2}, "scan budget"), ({"max_scan_packets": 2}, "packet scan"),
    ({"max_selected_frames": 2}, "selected video"), ({"max_video_seconds": 1.0}, "duration"),
    ({"max_expanded_rows": 1}, "visual rows"), ({"max_total_decoded_pixels": 100}, "decoded pixel"),
    ({"max_decoded_bytes": 100}, "decoded pixel"),
    ({"max_single_allocation_bytes": 4096}, "allocation|hard resource limit")])
def test_exec_worker_rejects_explicit_envelope_violations(tmp_path, changes, match):
    store = ReplayStore(tmp_path, envelope(**changes))
    executor = MediaExecutor(store)
    lease = store.stage([video_source()], allow_urls=False, cancelled=lambda: False)
    with pytest.raises(ImageInputError, match=match):
        executor.probe_replay(lease.refs[0], cancelled=lambda: False)
    assert lease._borrows == 0 and store._staging == 0
    lease.close()


def test_global_t_resize_is_not_a_pair_budget():
    from tensorfold.vision.media_worker import _grid
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    hw, grid = _grid("video", 61, 1024, 1024)
    assert hw == smart_resize(61, 1024, 1024, temporal_factor=2, factor=32, min_pixels=4096, max_pixels=25_165_824)
    assert hw != _grid("video", 2, 1024, 1024)[0] and grid[0] == 31


def test_actual_vfr_pts_are_preserved_instead_of_frame_index_over_fps(tmp_path):
    import base64
    from tensorfold.vision.videos import VideoSource
    raw = (Path(__file__).parent / "fixtures/media_front/colors-vfr.mkv").read_bytes()
    source = VideoSource("data:video/x-matroska;base64," + base64.b64encode(raw).decode())
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    lease = store.stage([source], allow_urls=False, cancelled=lambda: False)
    probed = executor.probe_replay(lease.refs[0], cancelled=lambda: False)
    assert probed.selected_indices == (0, 1, 2, 3, 4) and probed.fps == (2, 1)
    assert probed.selected_pts == ((0, 1), (1, 2), (2, 1), (5, 2), (3, 1))
    assert probed.timestamp_markers() == ("<0.2 seconds>", "<2.2 seconds>", "<3.0 seconds>")
    assert probed.timestamp_policy == "actual-pts-v1"
    lease.close()


@pytest.mark.parametrize("seconds,accepted", [(3.1, False), (3.5, True)])
def test_video_duration_includes_the_final_frame_presentation(tmp_path, seconds, accepted):
    import base64
    from tensorfold.vision.videos import VideoSource
    raw = (Path(__file__).parent / "fixtures/media_front/colors-vfr.mkv").read_bytes()
    source = VideoSource("data:video/x-matroska;base64," + base64.b64encode(raw).decode())
    store = ReplayStore(tmp_path, envelope(max_video_seconds=seconds))
    executor = MediaExecutor(store)
    lease = store.stage([source], allow_urls=False, cancelled=lambda: False)
    if accepted:
        probed = executor.probe_replay(lease.refs[0], cancelled=lambda: False)
        assert probed.fps == (2, 1) and probed.selected_indices == (0, 1, 2, 3, 4)
        assert probed.selected_pts[-1] == (3, 1)
    else:
        with pytest.raises(ImageInputError, match="duration exceeds"):
            executor.probe_replay(lease.refs[0], cancelled=lambda: False)
    lease.close()


def test_missing_pts_requires_explicit_assumption_and_changes_identity(tmp_path, monkeypatch):
    from tensorfold.vision import media_worker
    from transformers import AutoVideoProcessor
    processor = AutoVideoProcessor.from_pretrained("/opt/tensorfold-media-pins", local_files_only=True,
                                                   cap_pixels_per_frame=False)
    original = media_worker._frames
    def missing(*args):
        for index, frame, pixels, size in original(*args):
            frame.pts = None
            yield index, frame, pixels, size
    store = ReplayStore(tmp_path, envelope())
    lease = store.stage([video_source()], allow_urls=False, cancelled=lambda: False)
    actual = media_worker.probe(lease.refs[0], store.envelope, (None, processor))
    monkeypatch.setattr(media_worker, "_frames", missing)
    with pytest.raises(ImageInputError, match="reliable actual PTS"):
        media_worker.probe(lease.refs[0], store.envelope, (None, processor))
    assumed = media_worker.probe(lease.refs[0], envelope(assumed_fps=(2, 1)), (None, processor))
    assert assumed.selected_pts == actual.selected_pts
    assert assumed.timestamp_policy == "explicit-assumed-fps-v1" and assumed.identity != actual.identity
    lease.close()


def test_known_pts_without_a_video_endpoint_are_refused(tmp_path, monkeypatch):
    import base64
    from types import SimpleNamespace
    from tensorfold.vision import media_worker
    from tensorfold.vision.videos import VideoSource
    from transformers import AutoVideoProcessor
    raw = (Path(__file__).parent / "fixtures/media_front/colors-vfr.mkv").read_bytes()
    source = VideoSource("data:video/x-matroska;base64," + base64.b64encode(raw).decode())
    processor = AutoVideoProcessor.from_pretrained("/opt/tensorfold-media-pins", local_files_only=True,
                                                   cap_pixels_per_frame=False)
    original = media_worker._frames
    def without_duration(*args):
        for index, frame, pixels, size in original(*args):
            metadata = SimpleNamespace(width=frame.width, height=frame.height, pts=frame.pts,
                                       time_base=frame.time_base, duration=0)
            yield index, metadata, pixels, size
    store = ReplayStore(tmp_path, envelope())
    lease = store.stage([source], allow_urls=False, cancelled=lambda: False)
    monkeypatch.setattr(media_worker, "_frames", without_duration)
    with pytest.raises(ImageInputError, match="presentation endpoint is missing"):
        media_worker.probe(lease.refs[0], store.envelope, (None, processor))
    lease.close()


def test_exec_cancellation_retires_worker_and_releases_all_admission(tmp_path):
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    lease = store.stage([video_source()], allow_urls=False, cancelled=lambda: False)
    cancelled = threading.Event()
    timer = threading.Timer(0.2, cancelled.set)
    timer.start()
    try:
        with pytest.raises(MediaCancelled):
            executor.probe_replay(lease.refs[0], cancelled=cancelled.is_set)
    finally:
        timer.cancel()
        timer.join()
    assert lease._borrows == 0 and store._staging == 0
    lease.close()


@pytest.mark.parametrize("changes", [{"max_worker_wall_seconds": 0.1}, {"max_worker_address_bytes": 64 * 1024**2},
                                      {"max_worker_cpu_seconds": 1}])
def test_exec_hard_process_limits_fail_without_stranding_owner(tmp_path, changes):
    store = ReplayStore(tmp_path, envelope(**changes))
    executor = MediaExecutor(store)
    lease = None
    with pytest.raises(ImageInputError):
        lease = store.stage([video_source()], allow_urls=False, cancelled=lambda: False)
        executor.probe_replay(lease.refs[0], cancelled=lambda: False)
    if lease is not None:
        assert lease._borrows == 0
        lease.close()
    assert store._staging == 0


def test_frontend_planner_failure_rolls_back_replay_and_real_probe(tmp_path):
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    def fail(*args, **kwargs):
        raise ValueError("canonical expanded prefix does not fit")
    frontend = ReplayMediaFrontend(store, executor, fail, cancelled=lambda: False, allow_urls=False)
    with pytest.raises(ValueError, match="does not fit"):
        frontend.prepare_sources("template", [image_source()], allow_urls=False,
                                 max_prompt_tokens=1, max_visual_tokens=None)
    assert store._bytes == store._staging == store._leases == 0
    assert not list(tmp_path.iterdir())
