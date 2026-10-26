"""Explicit replay/resource/lifetime contracts, independent of CUDA and models."""
import base64
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from tensorfold.vision.images import ImageSource
from tensorfold.vision.images_http import ImageInputError
from tensorfold.vision.media_frontend import MediaExecutor, ReplayMediaFrontend
from tensorfold.vision.replay import MediaBusy, MediaCancelled, MediaEnvelope, MediaProbe, ReplayRef, ReplayStore, open_replay
from tensorfold.vision.videos import VideoSource


def envelope(**changes):
    """Test-only explicit limits; no shipping profile or resource qualification."""
    values = dict(max_images=4, max_videos=3, max_upload_bytes=1024**2, max_total_upload_bytes=4 * 1024**2,
                  max_dimension=4096, max_pixels_per_frame=8 * 1024**2, max_total_decoded_pixels=64 * 1024**2,
                  max_decoded_bytes=256 * 1024**2, max_selected_frames=64, max_video_seconds=30.0,
                  max_expanded_rows=65_536, max_workers=1, max_worker_threads=1,
                  max_worker_address_bytes=32 * 1024**3, max_worker_cpu_seconds=60, max_worker_wall_seconds=90.0,
                  max_staging_wall_seconds=90.0,
                  max_single_allocation_bytes=128 * 1024**2, max_probe_bytes=1024**2, max_probe_seconds=10.0,
                  max_scan_frames=128, max_scan_packets=256, max_staging_bytes=256 * 1024**2,
                  max_replay_bytes=8 * 1024**2, max_replay_items=16, max_replay_leases=4, max_replay_borrows=16,
                  max_manifest_bytes=65_536, max_result_bytes=65_536, max_redirects=0, max_url_chars=4096,
                  retirement_grace_seconds=0.2, assumed_fps=None)
    return MediaEnvelope(**{**values, **changes})


def image_source(mode="RGB", color=(11, 71, 201), size=(64, 48), *, orientation=None):
    output = io.BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    Image.new(mode, size, color).save(output, format="PNG", exif=exif)
    return ImageSource("data:image/png;base64," + base64.b64encode(output.getvalue()).decode())


def video_source():
    raw = (Path(__file__).parent / "fixtures/media_front/colors.mkv").read_bytes()
    return VideoSource("data:video/x-matroska;base64," + base64.b64encode(raw).decode())


@pytest.mark.parametrize("changes", [{"max_upload_bytes": 0}, {"max_workers": True},
                                     {"max_worker_wall_seconds": float("inf")}, {"assumed_fps": (24, 0)},
                                     {"max_images": 0, "max_videos": 0}, {"max_probe_bytes": 31},
                                     {"max_pixels_per_frame": 2**31}])
def test_envelope_rejects_invalid_explicit_bounds(changes):
    with pytest.raises(ValueError):
        envelope(**changes)


def test_envelope_has_no_shipping_defaults_and_roundtrips():
    with pytest.raises(TypeError):
        MediaEnvelope()
    configured = envelope(assumed_fps=(30000, 1001))
    restored = MediaEnvelope.from_record(json.loads(json.dumps(asdict(configured))))
    assert restored == configured and restored.identity == configured.identity
    assert envelope().identity != configured.identity


def test_ordered_immutable_refs_and_explicit_lifetime(tmp_path):
    store = ReplayStore(tmp_path, envelope())
    lease = store.stage([image_source(), video_source(), image_source(color=(7, 3, 9))],
                        allow_urls=False, cancelled=lambda: False)
    assert [r.kind for r in lease.refs] == ["image", "video", "image"]
    assert [r.ordinal for r in lease.refs] == [0, 1, 2]
    assert ReplayRef(**json.loads(json.dumps(lease.refs[0].record()))) == lease.refs[0]
    assert lease.refs[0].sha256 != lease.refs[2].sha256
    borrowed = lease.borrow()
    with pytest.raises(RuntimeError, match="outstanding"):
        lease.close()
    borrowed.close()
    lease.close()
    lease.close()
    assert not list(tmp_path.iterdir())
    assert (store._bytes, store._items, store._leases, store._staging) == (0, 0, 0, 0)


def test_capacity_and_cancellation_do_not_leak_partial_files(tmp_path):
    store = ReplayStore(tmp_path, envelope(max_replay_leases=1))
    lease = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    with pytest.raises(ImageInputError, match="capacity"):
        store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    lease.close()
    with pytest.raises(MediaCancelled):
        store.stage([image_source()], allow_urls=False, cancelled=lambda: True)
    with pytest.raises(ImageInputError, match="URL encoding"):
        store.stage([ImageSource("data:image/png;base64,!!!")], allow_urls=False, cancelled=lambda: False)
    assert not list(tmp_path.iterdir()) and store._leases == store._items == store._bytes == 0


@pytest.mark.parametrize("changes", [{"max_upload_bytes": 8}, {"max_total_upload_bytes": 8},
                                     {"max_replay_bytes": 8}, {"max_staging_bytes": 8},
                                     {"max_images": 0}, {"max_replay_items": 1}])
def test_stage_enforces_all_encoded_admission_counts_and_bytes(tmp_path, changes):
    store = ReplayStore(tmp_path, envelope(**changes))
    with pytest.raises(ImageInputError):
        store.stage([image_source(), image_source()], allow_urls=False, cancelled=lambda: False)
    assert store._bytes == store._staging == store._leases == store._items == 0


def test_ref_mutation_symlink_and_wrong_owner_are_rejected(tmp_path):
    store = ReplayStore(tmp_path, envelope())
    lease = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    ref = lease.refs[0]
    with open_replay(ref, store.envelope) as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
    other = ReplayStore(tmp_path, envelope())
    with pytest.raises(ImageInputError, match="not owned"):
        other.borrow_ref(ref)
    path = Path(ref.path)
    path.chmod(0o600)
    with pytest.raises(ImageInputError, match="mutable"):
        with open_replay(ref, store.envelope):
            pytest.fail("mutable replay opened")
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    path.chmod(0o400)
    with pytest.raises(ImageInputError, match="identity changed"):
        with open_replay(ref, store.envelope):
            pytest.fail("changed replay opened")
    path.unlink()
    path.symlink_to(tmp_path / "absent")
    with pytest.raises(OSError):
        with open_replay(ref, store.envelope):
            pytest.fail("symlink replay opened")
    lease.close()


def test_url_staging_reuses_bounded_security_fetch(tmp_path, monkeypatch):
    from tensorfold.vision import images_http, media_worker
    raw = b"not-decoded-at-staging"
    calls = []
    monkeypatch.setattr(images_http, "fetch_image", lambda url, **kwargs: (calls.append((url, kwargs)) or (raw, "image/png")))
    store = ReplayStore(tmp_path, envelope())
    with pytest.raises(ImageInputError, match="off"):
        store.stage([ImageSource("https://example.com/test.png")], allow_urls=False, cancelled=lambda: False)
    import tempfile
    with tempfile.TemporaryFile() as encoded:
        output = media_worker.stage({"url": "https://example.com/test.png", "remaining_bytes": store.envelope.max_upload_bytes,
                                     "kind": "image"}, store.envelope, encoded.fileno())
        assert output["byte_length"] == len(raw)
    assert len(calls) == 1 and calls[0][1]["max_bytes"] == store.envelope.max_upload_bytes
    assert calls[0][1]["max_redirects"] == 0 and calls[0][1]["max_url_chars"] == 4096


def test_bounded_source_hook_precedes_eager_decode_and_preserves_interleaving(tmp_path, monkeypatch):
    from tensorfold.server.prompts import prepare_images
    from tensorfold.vision import images
    from tensorfold.vision.media_frontend import PreparedReplayMedia
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    monkeypatch.setattr(images, "load_images", lambda *a, **kw: pytest.fail("eager decode called"))
    calls = []
    frontend = SimpleNamespace(media_envelope=store.envelope, videos=True, allow_urls=False,
                               check_media_health=lambda: None)
    def prepare(rendered, sources, **kwargs):
        calls.append((rendered, sources, kwargs))
        return SimpleNamespace(token_ids=(1, 2, 3))
    frontend.prepare_sources = prepare
    contents = [{"type": "image_url", "image_url": {"url": image_source().url}},
                {"type": "text", "text": "middle"},
                {"type": "video_url", "video_url": {"url": video_source().url}}]
    prepared = prepare_images(frontend, [{"role": "user", "content": contents}], lambda m: m,
                              context_limit=1024)
    assert prepared.tokens == [1, 2, 3]
    assert [p["type"] for p in calls[0][0][0]["content"]] == ["image", "text", "video"]
    assert [type(s) for s in calls[0][1]] == [ImageSource, VideoSource]
    assert calls[0][2]["max_prompt_tokens"] == 1024
    assert isinstance(ReplayMediaFrontend(store, executor, lambda *a, **k: None,
                                         cancelled=lambda: False, allow_urls=False).media_envelope, MediaEnvelope)
    assert PreparedReplayMedia  # public owner exists; actual lifetime is exercised below


def test_actual_pts_adapter_has_explicit_identity_and_pinned_format():
    ref = ReplayRef("/tmp/owned", "0" * 64, 1, 0, "video", "auto")
    probed = MediaProbe(ref, "env", "processor", 64, 48, 3, (0, 1, 2), ((0, 1), (3, 5), (9, 5)),
                        (2, 1), "actual-pts-v1", (64, 64), (2, 4, 4), 3 * 64 * 48, 3 * 64 * 48 * 3)
    assert probed.timestamp_markers() == ("<0.3 seconds>", "<1.8 seconds>")
    assert MediaProbe.from_record(json.loads(json.dumps(probed.record()))) == probed


def test_cancelled_executor_never_starts_a_worker(tmp_path, monkeypatch):
    from tensorfold.vision import media_frontend
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    lease = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    monkeypatch.setattr(media_frontend, "supervise", lambda *a, **kw: pytest.fail("worker started"))
    with pytest.raises(MediaCancelled):
        executor.probe_replay(lease.refs[0], cancelled=lambda: True)
    assert lease._borrows == 0 and store._staging == 0
    lease.close()


def test_unresolved_worker_poison_preserves_capacity_and_replay_owners(tmp_path, monkeypatch):
    from tensorfold.cuda.expert_launch import UnresolvedRankSessions
    from tensorfold.vision import media_frontend
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    lease = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    unresolved = UnresolvedRankSessions((object(),), RuntimeError("unreaped test owner"))
    def refuse(*args, **kwargs):
        raise unresolved
    monkeypatch.setattr(media_frontend, "supervise", refuse)
    with pytest.raises(RuntimeError, match="retirement unresolved"):
        executor.probe_replay(lease.refs[0], cancelled=lambda: False)
    assert store._failure is unresolved and lease._borrows == 1 and store._staging > 0
    with pytest.raises(RuntimeError, match="retirement unresolved"):
        lease.close()
    with pytest.raises(RuntimeError, match="retirement unresolved"):
        executor.probe_replay(lease.refs[0], cancelled=lambda: False)


def test_replay_root_requires_actual_service_ownership_and_no_symlink(tmp_path):
    tmp_path.chmod(0o777)
    with pytest.raises(ValueError, match="not writable"):
        ReplayStore(tmp_path, envelope())
    tmp_path.chmod(0o700)
    linked = tmp_path / "link"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical"):
        ReplayStore(linked, envelope())


def test_failed_partial_stage_keeps_encoded_charge_when_removal_fails(tmp_path, monkeypatch):
    from tensorfold.vision import replay
    store = ReplayStore(tmp_path, envelope())
    primary = ValueError("staging failed after writing")
    def failed(self, record, target, cancelled, **kwargs):
        target.write(b"partial")
        target.flush()
        raise primary
    monkeypatch.setattr(MediaExecutor, "_run", failed)
    monkeypatch.setattr(replay.shutil, "rmtree", lambda *a: (_ for _ in ()).throw(OSError("cleanup failed")))
    with pytest.raises(ValueError) as raised:
        store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    assert raised.value is primary and isinstance(primary.__cause__, OSError)
    assert store._bytes == store.envelope.max_upload_bytes and store._items == store._leases == 1
    assert store._staging == 0 and list(tmp_path.glob("media-*/*"))


def test_diagnostic_seal_failure_closes_exact_raw_descriptor(tmp_path, monkeypatch):
    import os
    from tensorfold import _fd_owner
    from tensorfold.vision import media_frontend
    store = ReplayStore(tmp_path, envelope())
    original, acquired, descriptors = _fd_owner.OwnedFD, [], []
    def create():
        owner = original()
        acquired.append(owner)
        return owner
    def failed(descriptor, *args):
        descriptors.append(descriptor)
        raise OSError("seal failed")
    monkeypatch.setattr(_fd_owner, "OwnedFD", create)
    monkeypatch.setattr(media_frontend.fcntl, "fcntl", failed)
    with pytest.raises(OSError, match="seal failed"):
        with media_frontend._diagnostic_fd(256, store, 0):
            pytest.fail("unsealed descriptor published")
    assert acquired[0].closed
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_supervisor_constructor_oom_fallback_retains_owner_and_capacity(tmp_path, monkeypatch):
    import tempfile
    from tensorfold.cuda import expert_launch
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    primary = RuntimeError("retirement proof failed")
    allocation = MemoryError("retained-session diagnostic allocation failed")
    child = SimpleNamespace(pid=246801, returncode=None)
    def failed_retirement(*args):
        raise primary
    def failed_diagnostic(*args):
        raise allocation
    monkeypatch.setattr(expert_launch.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(expert_launch, "_exited", lambda owner: 0)
    monkeypatch.setattr(expert_launch.os, "killpg", lambda *args: None)
    monkeypatch.setattr(expert_launch, "_retire", failed_retirement)
    monkeypatch.setattr(expert_launch, "UnresolvedRankSessions", failed_diagnostic)
    with tempfile.TemporaryFile() as pixels, pytest.raises(RuntimeError) as raised:
        executor._run({"operation": "probe", "ref": {}}, pixels, lambda: False)
    assert raised.value is primary and store._failure is primary
    assert store._retained_workers == [primary] and store._active_workers[0] is not None and store._staging > 0
    assert BaseException.__cause__.__get__(primary) is allocation
    traceback = BaseException.__traceback__.__get__(primary)
    frames = []
    while traceback is not None:
        if traceback.tb_frame.f_code is expert_launch.supervise.__code__:
            frames.append(traceback.tb_frame)
        traceback = traceback.tb_next
    assert frames and any(owner is child for owner in frames[0].f_locals["children"])


def test_supervisor_hostile_exception_needs_no_attribute_inspection(tmp_path, monkeypatch):
    import tempfile
    from tensorfold.vision import media_frontend
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    class OverriddenError(RuntimeError):
        @property
        def __cause__(self):
            pytest.fail("overrideable exception property read")
    error = OverriddenError("original")
    def failed(*args, **kwargs):
        raise error
    monkeypatch.setattr(media_frontend, "supervise", failed)
    with tempfile.TemporaryFile() as pixels, pytest.raises(OverriddenError) as raised:
        executor._run({"operation": "probe", "ref": {}}, pixels, lambda: False)
    assert raised.value is error and store._failure is error and store._staging > 0


def test_borrow_handle_concurrent_close_retires_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    store = ReplayStore(tmp_path, envelope())
    lease = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    borrowed = lease.borrow()
    with ThreadPoolExecutor(max_workers=8) as threads:
        list(threads.map(lambda i: borrowed.close(), range(64)))
    assert lease._borrows == 0
    lease.close()


def test_server_source_publication_failure_retires_acquired_replay(tmp_path, monkeypatch):
    from tensorfold.server import prompts
    from tensorfold.vision.media_frontend import PreparedReplayMedia
    store = ReplayStore(tmp_path, envelope())
    source = image_source()
    lease = store.stage([source], allow_urls=False, cancelled=lambda: False)
    acquired = PreparedReplayMedia(SimpleNamespace(token_ids=(1, 2)), (), lease, None)
    frontend = SimpleNamespace(media_envelope=store.envelope, videos=True, allow_urls=False,
                               prepare_sources=lambda *a, **k: acquired, check_media_health=lambda: None)
    primary = MemoryError("result publication failed")
    def failed(*args, **kwargs):
        raise primary
    monkeypatch.setattr(prompts, "RenderedPrompt", failed)
    with pytest.raises(MemoryError) as raised:
        prompts.prepare_images(frontend, [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": source.url}}]}], str)
    assert raised.value is primary and store._bytes == store._leases == store._items == 0


@pytest.mark.parametrize("changes", [{"max_single_allocation_bytes": 2**64}, {"max_probe_bytes": 2**63},
                                     {"max_probe_seconds": 1e20}])
def test_foreign_size_and_probe_option_ranges_reject_before_conversion(changes):
    with pytest.raises(ValueError, match="range"):
        envelope(**changes)


def test_replay_partial_close_refuses_borrows_and_requires_containment(tmp_path, monkeypatch):
    from tensorfold.vision import replay
    store = ReplayStore(tmp_path, envelope())
    lease = store.stage([image_source(), image_source()], allow_urls=False, cancelled=lambda: False)
    original, count = replay.shutil.rmtree, 0
    def partial(path):
        nonlocal count
        count += 1
        if count == 1:
            Path(lease.refs[0].path).unlink()
            raise OSError("partial retirement")
        return original(path)
    monkeypatch.setattr(replay.shutil, "rmtree", partial)
    before = store._bytes
    with pytest.raises(OSError, match="partial retirement"):
        lease.close()
    with pytest.raises(RuntimeError, match="containment"):
        lease.borrow()
    assert store._bytes == before
    with pytest.raises(RuntimeError, match="containment"):
        lease.close()
    assert store._bytes == before and store._leases == 1 and count == 1


@pytest.mark.parametrize("failure", ["preentry", "consumed", "unretired"])
def test_diagnostic_close_uses_observable_native_retirement(tmp_path, monkeypatch, failure):
    import os
    import tempfile
    from tensorfold import _fd_owner
    from tensorfold.vision import media_frontend
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    original, captured = _fd_owner.OwnedFD, []
    class Owner:
        def __init__(self):
            self.native, self.calls = original(), 0
            captured.append(self)
        @property
        def closed(self):
            return self.native.closed
        def memfd(self):
            return self.native.memfd()
        def fileno(self):
            return self.native.fileno()
        def close(self):
            self.calls += 1
            if failure == "unretired" or failure == "preentry" and self.calls == 1:
                raise OSError("before native close")
            self.native.close()
            if failure == "consumed":
                raise OSError("consumed native close")
    def completed(commands, environments, **kwargs):
        os.write(kwargs["pass_fds"][1], b"{}")
        return 0
    monkeypatch.setattr(_fd_owner, "OwnedFD", Owner)
    monkeypatch.setattr(media_frontend, "supervise", completed)
    try:
        with tempfile.TemporaryFile() as pixels, pytest.raises(OSError):
            executor._run({"operation": "probe", "ref": {}}, pixels, lambda: False)
        if failure == "unretired":
            assert not captured[0].closed and store._failure is not None and store._staging > 0
        else:
            assert captured[0].closed and store._failure is None and store._staging == 0
            assert captured[0].calls == (2 if failure == "preentry" else 1)
    finally:
        captured[0].native.close()


@pytest.mark.parametrize("failure", ["handle", "reservation"])
def test_pixel_partial_close_rejects_new_borrows_and_release_is_idempotent(tmp_path, monkeypatch, failure):
    import tempfile
    from tensorfold.vision.media_frontend import PixelUnitLease
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    replay = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    size = 4 * 1536 * 4
    reservation = object()
    executor._reserve_staging(reservation, size)
    handle = tempfile.TemporaryFile(dir=tmp_path)
    handle.write(bytes(size))
    handle.flush()
    class Handle:
        failed = False
        @property
        def closed(self):
            return handle.closed
        def fileno(self):
            return handle.fileno()
        def close(self):
            handle.close()
            if failure == "handle" and not self.failed:
                self.failed = True
                raise OSError("handle consumed close")
    pixels = PixelUnitLease(executor, Handle(), size, (4, 1536), replay.borrow(), reservation)
    release, failed = executor._release_staging, False
    def interrupted(token):
        nonlocal failed
        release(token)
        if failure == "reservation" and not failed:
            failed = True
            raise RuntimeError("release interrupted after removal")
    monkeypatch.setattr(executor, "_release_staging", interrupted)
    with pytest.raises((OSError, RuntimeError)):
        pixels.close()
    with pytest.raises(RuntimeError, match="retired"):
        pixels.borrow()
    pixels.close()
    pixels.close()
    replay.close()
    assert store._staging == store._bytes == store._leases == 0


@pytest.mark.parametrize("kind", ["replay", "pixels"])
def test_interrupted_borrow_retirement_cannot_expose_retired_storage(tmp_path, kind):
    import tempfile
    from tensorfold.vision.media_frontend import PixelUnitLease
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    replay = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    target = replay
    if kind == "pixels":
        size = 4 * 1536 * 4
        handle = tempfile.TemporaryFile(dir=tmp_path)
        handle.write(bytes(size))
        handle.flush()
        reservation = object()
        executor._reserve_staging(reservation, size)
        target = PixelUnitLease(executor, handle, size, (4, 1536), replay.borrow(), reservation)
    borrowed = target.borrow()
    class InterruptedSet(set):
        first = True
        def discard(self, value):
            super().discard(value)
            if self.first:
                self.first = False
                raise KeyboardInterrupt("after authoritative retirement")
    target._borrowers = InterruptedSet(target._borrowers)
    with pytest.raises(KeyboardInterrupt):
        borrowed.close()
    target.close()
    with pytest.raises(RuntimeError, match="retired"):
        getattr(borrowed, "refs" if kind == "replay" else "values")
    borrowed.close()
    replay.close()


def test_replay_hash_reads_only_the_admitted_extent_plus_one(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from tensorfold.cuda import tensor_file
    store = ReplayStore(tmp_path, envelope())
    lease = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    ref = lease.refs[0]
    original, requests = tensor_file._regular_stream, []
    class Growing:
        def __init__(self, handle):
            self.handle, self.first = handle, True
        def fileno(self):
            return self.handle.fileno()
        def read(self, count):
            requests.append(count)
            block = self.handle.read(count)
            if self.first:
                self.first = False
                path = Path(ref.path)
                path.chmod(0o600)
                with path.open("ab") as target:
                    target.write(b"x" * 4096)
                path.chmod(0o400)
            return block
    @contextmanager
    def growing(path, **kwargs):
        with original(path, **kwargs) as handle:
            yield Growing(handle)
    monkeypatch.setattr(tensor_file, "_regular_stream", growing)
    with pytest.raises(ImageInputError, match="grew"):
        with open_replay(ref, store.envelope):
            pytest.fail("growing input published")
    assert sum(requests) == ref.byte_length + 1
    lease.close()


@pytest.mark.parametrize("target", ["stage", "metadata", "pixels"])
def test_post_publication_staging_failure_releases_the_caller_known_token(tmp_path, monkeypatch, target):
    import tempfile
    from tensorfold.vision.media_worker import PROCESSOR_ID
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    replay = None
    if target != "stage":
        replay = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    original = store._reserve_staging
    def interrupted(token, size):
        original(token, size)
        raise MemoryError("after reservation publication")
    monkeypatch.setattr(store, "_reserve_staging", interrupted)
    with pytest.raises(MemoryError, match="after reservation"):
        if target == "stage":
            store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
        elif target == "metadata":
            with tempfile.TemporaryFile() as output:
                executor._run({"operation": "probe", "ref": replay.refs[0].record()}, output, lambda: False)
        else:
            probed = MediaProbe(replay.refs[0], store.envelope.identity, PROCESSOR_ID, 64, 48, 1, (), (), None,
                                "encoded-orientation-pil-rgb-v1", (224, 320), (1, 14, 20), 64 * 48, 64 * 48 * 4)
            executor.pixel_unit(probed, 0, cancelled=lambda: False)
    assert store._staging == 0 and store._active_workers == [None]
    if replay is not None:
        assert replay._borrows == 0
        replay.close()
    assert store._bytes == store._leases == store._items == 0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("target", ["metadata", "worker"])
def test_post_release_interruption_does_not_orphan_worker_capacity(tmp_path, monkeypatch, target):
    import os
    import tempfile
    from tensorfold.vision import media_frontend
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    original = executor._release_staging if target == "metadata" else executor._retire_worker
    interrupted = False
    def release(*args):
        nonlocal interrupted
        original(*args)
        if not interrupted:
            interrupted = True
            raise RuntimeError("after resource retirement")
    def completed(commands, environments, **kwargs):
        os.write(kwargs["pass_fds"][1], b"{}")
        return 0
    monkeypatch.setattr(executor, "_release_staging" if target == "metadata" else "_retire_worker", release)
    monkeypatch.setattr(media_frontend, "supervise", completed)
    with tempfile.TemporaryFile() as output:
        with pytest.raises(RuntimeError, match="after resource"):
            executor._run({"operation": "probe", "ref": {}}, output, lambda: False)
        assert store._staging == 0 and store._active_workers == [None]
        assert executor._run({"operation": "probe", "ref": {}}, output, lambda: False) == {}
        assert store._staging == 0 and store._active_workers == [None]


@pytest.mark.parametrize("target", ["metadata", "worker", "preparation"])
def test_preentry_cleanup_failure_retains_claims_and_requires_containment(tmp_path, monkeypatch, target):
    import os
    import tempfile
    from tensorfold.vision import media_frontend
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    primary = MemoryError("before resource retirement")
    def failed(*args):
        raise primary
    def completed(commands, environments, **kwargs):
        os.write(kwargs["pass_fds"][1], b"{}")
        return 0
    if target == "preparation":
        monkeypatch.setattr(media_frontend, "finish", failed)
    else:
        monkeypatch.setattr(executor, "_release_staging" if target == "metadata" else "_retire_worker", failed)
    monkeypatch.setattr(media_frontend, "supervise", completed)
    with tempfile.TemporaryFile() as output, pytest.raises(MemoryError) as raised:
        executor._run({"operation": "probe", "ref": {}}, output, lambda: False)
    assert raised.value is primary and store._failure is primary and store._retained_workers == [primary]
    assert store._staging > 0 or store._active_workers[0] is not None
    with pytest.raises(RuntimeError, match="containment"):
        executor._check_owner()


@pytest.mark.parametrize("target", ["stage", "stage_existing_fatal", "pixels"])
def test_preentry_unpublished_cleanup_failure_contains_the_store(tmp_path, monkeypatch, target):
    from tensorfold.vision.media_worker import PROCESSOR_ID
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    replay = None
    if target == "pixels":
        replay = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    cleanup = MemoryError("before unpublished reservation retirement")
    primary = ValueError("pixel preparation failed")
    first_fatal = RuntimeError("earlier concurrent worker fatal")
    release = store._release_staging
    def interrupted(token):
        if target == "pixels" or store._staging_reservations.get(token) == 2 * store.envelope.max_upload_bytes:
            if target == "stage_existing_fatal":
                with store._lock:
                    store._failure = first_fatal
                    store._retained_workers[0] = first_fatal
            raise cleanup
        release(token)
    monkeypatch.setattr(store, "_release_staging", interrupted)
    if target.startswith("stage"):
        with pytest.raises(MemoryError) as raised:
            store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
        assert raised.value is cleanup
        assert store._failure is (first_fatal if target == "stage_existing_fatal" else cleanup)
        if target == "stage_existing_fatal":
            assert store._retained_workers[0] is first_fatal
    else:
        def failed(*args, **kwargs):
            raise primary
        monkeypatch.setattr(executor, "_run", failed)
        probed = MediaProbe(replay.refs[0], store.envelope.identity, PROCESSOR_ID, 64, 48, 1, (), (), None,
                            "encoded-orientation-pil-rgb-v1", (224, 320), (1, 14, 20), 64 * 48, 64 * 48 * 4)
        with pytest.raises(ValueError) as raised:
            executor.pixel_unit(probed, 0, cancelled=lambda: False)
        assert raised.value is primary and store._failure is primary
    assert store._staging > 0
    with pytest.raises(RuntimeError, match="containment"):
        executor._check_owner()


def test_directory_wrapper_failure_removes_the_already_created_directory(tmp_path, monkeypatch):
    from tensorfold.vision import replay
    store = ReplayStore(tmp_path, envelope())
    def failed(*args):
        raise MemoryError("path wrapper failed")
    monkeypatch.setattr(replay, "Path", failed)
    with pytest.raises(MemoryError, match="path wrapper"):
        store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    assert not list(tmp_path.iterdir()) and store._bytes == store._items == store._leases == 0


def test_uncertain_removed_directory_retry_preserves_reused_path_and_new_claims(tmp_path, monkeypatch):
    from tensorfold.vision import replay
    store = ReplayStore(tmp_path, envelope())
    old = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    directory = Path(old.refs[0].path).parent
    original = replay.shutil.rmtree
    def removed_then_failed(path):
        original(path)
        raise OSError("removed before acknowledgement")
    with monkeypatch.context() as failure:
        failure.setattr(replay.shutil, "rmtree", removed_then_failed)
        with pytest.raises(OSError, match="before acknowledgement"):
            old.close()
    def reused(**kwargs):
        directory.mkdir()
        return str(directory)
    replacement = ReplayStore(tmp_path, envelope())
    with monkeypatch.context() as creation:
        creation.setattr(replay.tempfile, "mkdtemp", reused)
        new = replacement.stage([image_source(color=(91, 17, 13))], allow_urls=False, cancelled=lambda: False)
    new_bytes = Path(new.refs[0].path).read_bytes()
    with pytest.raises(RuntimeError, match="containment"):
        old.close()
    assert Path(new.refs[0].path).read_bytes() == new_bytes
    assert replacement._live[new.refs[0].path] is new and replacement._leases == 1
    assert store._live[old.refs[0].path] is old and store._leases == 1
    with replacement.borrow_ref(new.refs[0]):
        pass
    new.close()
    assert store._leases == 1  # uncertain old reservation remains contained


def test_staging_pressure_is_busy_but_an_oversize_unit_is_permanent(tmp_path):
    store = ReplayStore(tmp_path, envelope())
    first, second = object(), object()
    store._reserve_staging(first, store.envelope.max_staging_bytes)
    with pytest.raises(MediaBusy):
        store._reserve_staging(second, 1)
    store._release_staging(first)
    store._reserve_staging(second, 1)
    store._release_staging(second)
    with pytest.raises(ImageInputError) as raised:
        store._reserve_staging(second, store.envelope.max_staging_bytes + 1)
    assert type(raised.value) is ImageInputError and store._staging == 0


def test_worker_slot_pressure_is_busy_without_adding_or_releasing_claims(tmp_path):
    import tempfile
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    token = object()
    store._active_workers[0] = token
    with tempfile.TemporaryFile() as pixels, pytest.raises(MediaBusy):
        executor._run({"operation": "probe", "ref": {}}, pixels, lambda: False)
    assert store._active_workers == [token] and store._staging == 0


def test_replay_borrow_pressure_is_busy_and_recovers_after_retirement(tmp_path):
    store = ReplayStore(tmp_path, envelope(max_replay_borrows=1))
    lease = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    borrowed = lease.borrow()
    with pytest.raises(MediaBusy):
        lease.borrow()
    borrowed.close()
    with lease.borrow():
        pass
    lease.close()


@pytest.mark.parametrize("busy,status", [(True, 503), (False, 400)])
def test_source_preparation_preserves_typed_busy_and_bad_input(tmp_path, busy, status):
    from tensorfold.server.errors import RequestError, status_of
    from tensorfold.server.prompts import prepare_images
    error = MediaBusy("temporarily occupied") if busy else ImageInputError("invalid media")
    def failed(*args, **kwargs):
        raise error
    frontend = SimpleNamespace(media_envelope=envelope(), videos=True, allow_urls=False, prepare_sources=failed,
                               check_media_health=lambda: None)
    source = image_source()
    with pytest.raises(RequestError) as raised:
        prepare_images(frontend, [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": source.url}}]}], str)
    assert status_of(raised.value) == status


def test_fatal_probe_context_preserves_primary_and_retained_borrow(tmp_path, monkeypatch):
    from tensorfold.vision import media_frontend
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    lease = store.stage([image_source()], allow_urls=False, cancelled=lambda: False)
    primary = ValueError("supervisor operation failure")
    def failed(*args, **kwargs):
        raise primary
    monkeypatch.setattr(media_frontend, "supervise", failed)
    with pytest.raises(ValueError) as raised:
        executor.probe_replay(lease.refs[0], cancelled=lambda: False)
    assert raised.value is primary and store._failure is primary
    assert lease._borrows == 1 and store._staging > 0 and store._bytes > 0
    retained = BaseException.__dict__["__dict__"].__get__(primary)["_tensorfold_retained_owners"]
    assert any(owner in lease._borrowers for owner in retained)


def test_server_fatal_value_error_is_containment_not_input_refusal(tmp_path, monkeypatch):
    from tensorfold.server.prompts import prepare_images
    from tensorfold.server.errors import RequestError
    from tensorfold.vision import media_frontend
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    frontend = ReplayMediaFrontend(store, executor, lambda *a, **k: None, cancelled=lambda: False, allow_urls=False)
    primary = ValueError("supervisor failure is not bad media")
    def failed(*args, **kwargs):
        raise primary
    monkeypatch.setattr(media_frontend, "supervise", failed)
    source = image_source()
    with pytest.raises(RuntimeError, match="containment") as raised:
        prepare_images(frontend, [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": source.url}}]}], str)
    assert not isinstance(raised.value, RequestError)
    assert BaseException.__cause__.__get__(raised.value) is primary and store._failure is primary


def test_server_planner_error_with_terminal_directory_cleanup_is_containment(tmp_path, monkeypatch):
    from tensorfold.server.prompts import prepare_images
    from tensorfold.vision import replay
    from tensorfold.vision.media_worker import PROCESSOR_ID
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    primary, cleanup = ValueError("invalid expanded input"), OSError("terminal directory cleanup")
    def planner(*args, **kwargs):
        raise primary
    frontend = ReplayMediaFrontend(store, executor, planner, cancelled=lambda: False, allow_urls=False)
    def probed(ref, **kwargs):
        return MediaProbe(ref, store.envelope.identity, PROCESSOR_ID, 64, 48, 1, (), (), None,
                          "encoded-orientation-pil-rgb-v1", (224, 320), (1, 14, 20), 64 * 48, 64 * 48 * 4)
    def failed(*args):
        raise cleanup
    monkeypatch.setattr(executor, "probe_replay", probed)
    monkeypatch.setattr(replay.shutil, "rmtree", failed)
    source = image_source()
    with pytest.raises(RuntimeError, match="containment"):
        prepare_images(frontend, [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": source.url}}]}], str)
    assert store._failure is cleanup and store._leases == 1
    assert BaseException.__cause__.__get__(primary) is cleanup
    retained = BaseException.__dict__["__dict__"].__get__(primary)["_tensorfold_retained_owners"]
    assert retained and retained[0]._directory_failure is cleanup
