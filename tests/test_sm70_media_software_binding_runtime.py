"""Actual combined CPU media/source authority, not device or model qualification."""

import ctypes
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

import pytest

from tensorfold.cuda.sm70 import media_software_binding as media
from tensorfold.cuda.sm70.software_binding import SoftwareBindingError

pytestmark = pytest.mark.skipif(not Path("/opt/tensorfold-media-pins/media-install.json").is_file(),
                                reason="requires reviewed source media image")


def test_actual_combined_binding_and_explicit_warm_reissue(tmp_path):
    from tensorfold.cuda.expert_launch import supervise

    authority = media.issue_media_binding()
    record = media.verify_media_binding(authority)
    assert record["software_only"] is True and record["release_qualified"] is False
    assert record["media"]["image_backend"] == "torchvision" and record["media"]["decoder_backend"] == "pyav"
    assert record["media"]["torchvision_cuda"] == -1 and record["media"]["dav1d_version"] == "1.5.4"
    assert record["protected_media"]["preprocessor_config.json"]["sha256"] == (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516")
    for wrong in (record, authority._source):
        with pytest.raises(SoftwareBindingError, match="forged"):
            media.verify_media_binding(wrong)

    # A real new process DSO is a warm observation, never an immutable file
    # replacement. Its independently consumed result confirms actual loading.
    compiler = shutil.which("cc")
    assert compiler is not None
    code, binary = tmp_path / "media_warm.c", tmp_path / "media_warm_fixture.so"
    code.write_text("int media_warm_value(void) { return 37; }\n")
    deadline = time.monotonic() + 30
    status = supervise([[compiler, "-shared", "-fPIC", str(code), "-o", str(binary)]],
                       [dict(os.environ)], grace=0, failure_probe=lambda: time.monotonic() >= deadline)
    assert status == 0
    library = ctypes.CDLL(str(binary))
    assert library.media_warm_value() == 37
    with pytest.raises(SoftwareBindingError, match="runtime changed"):
        media.verify_media_binding(authority)
    replacement = media.reissue_media_binding(authority)
    updated = media.verify_media_binding(replacement)
    assert updated["identity_sha256"] != record["identity_sha256"]
    assert updated["protected_media"] == record["protected_media"]
    with pytest.raises(SoftwareBindingError, match="runtime changed"):
        media.verify_media_binding(authority)


def test_actual_protected_hierarchy_and_metadata_mutation_cannot_be_repaired():
    authority = media.issue_media_binding()
    root = Path("/opt/tensorfold-media-pins")
    mode = root.stat().st_mode & 0o7777
    try:
        root.chmod(0o777)
        for operation in (media.verify_media_binding, media.reissue_media_binding):
            with pytest.raises(SoftwareBindingError, match="directory"):
                operation(authority)
    finally:
        root.chmod(mode)
    pins = root / "media-pins.json"
    before = pins.stat()
    try:
        os.utime(pins, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        for operation in (media.verify_media_binding, media.reissue_media_binding):
            with pytest.raises(SoftwareBindingError, match="immutable.*changed"):
                operation(authority)
    finally:
        os.utime(pins, ns=(before.st_atime_ns, before.st_mtime_ns))


def _forked_media_child(authority):
    try:
        media.verify_media_binding(authority)
    except SoftwareBindingError:
        os._exit(0)
    except BaseException:
        os._exit(2)
    os._exit(1)


def _cold_media_fork_control():
    from test_sm70_software_binding_runtime import _exercise_fork_control

    authority = media.issue_media_binding()
    process = multiprocessing.get_context("fork").Process(target=_forked_media_child, args=(authority,))
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("combined real-fork control requires exactly one native thread before fork")
    _exercise_fork_control(process)


def test_actual_combined_authority_cannot_cross_a_safe_real_fork():
    from tensorfold.cuda.expert_launch import supervise

    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    deadline = time.monotonic() + 90
    status = supervise([[sys.executable, str(Path(__file__).resolve())]], [environment], grace=0,
                       failure_probe=lambda: time.monotonic() >= deadline)
    assert status == 0


def test_actual_provider_file_identity_cannot_be_repaired_by_reissue():
    authority = media.issue_media_binding()
    provider = Path(authority.record()["media"]["loaded_media_providers"]["libavcodec"])
    before = provider.stat()
    # Timestamp-only mutation leaves mapped executable bytes untouched. Even
    # restoring its mtime cannot restore the issued immutable inode/ctime scope.
    try:
        os.utime(provider, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        for operation in (media.verify_media_binding, media.reissue_media_binding):
            with pytest.raises(SoftwareBindingError, match="immutable.*changed"):
                operation(authority)
    finally:
        os.utime(provider, ns=(before.st_atime_ns, before.st_mtime_ns))


if __name__ == "__main__":
    _cold_media_fork_control()
