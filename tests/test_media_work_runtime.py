"""Real fresh installed CPU workers behind the owned readiness coordinator.

Requires the project's pinned CPU media image. This checks no GPU, language
model, model quality or throughput; controlled IO lifetime tests are separate.
"""

from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tensorfold.vision import media_frontend
from tensorfold.vision.media_frontend import MediaExecutor, PreparedReplayMedia
from tensorfold.vision.media_work import MediaWork
from tensorfold.vision.replay import MediaCancelled, ReplayStore
from tests.test_media_replay import envelope, image_source, video_source


pytestmark = pytest.mark.skipif(not Path("/opt/tensorfold-media-pins/media-install.json").is_file(),
                                reason="requires the pinned source CPU media image")
PROFILE = "a" * 64


def prepared_source(tmp_path, source):
    store = ReplayStore(tmp_path, envelope())
    executor = MediaExecutor(store)
    replay = store.stage([source], allow_urls=False, cancelled=lambda: False)
    try:
        probe = executor.probe_replay(replay.refs[0], cancelled=lambda: False)
        plan = NS(profile_identity=PROFILE, media_identity=probe.identity, token_ids=(1, 31),
            units=tuple(NS(probe_index=0, run=NS(temporal_group=i)) for i in range(probe.unit_count)))
        return PreparedReplayMedia(plan, (probe,), replay, executor)
    except BaseException:
        replay.close()
        raise


def worker_gate(monkeypatch):
    entered, release, calls = threading.Event(), threading.Event(), []
    supervise = media_frontend.supervise

    def gated(commands, environments, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        assert commands[0][:3] == [sys.executable, "-m", "tensorfold.vision.media_worker"]
        assert environments[0]["CUDA_VISIBLE_DEVICES"] == ""
        assert not environments[0].get("PYTHONPATH")
        calls.append(commands[0][:3])
        entered.set()
        assert release.wait(3)
        return supervise(commands, environments, **kwargs)

    monkeypatch.setattr(media_frontend, "supervise", gated)
    return entered, release, calls


@pytest.mark.parametrize("kind,index", [("image", 0), ("video", 1)])
def test_real_installed_worker_returns_exact_direct_unit_and_never_blocks_readiness(tmp_path, monkeypatch, kind, index):
    # The child must resolve its normal installed package, even when the parent
    # is a labelled source-composition control. No worker source-path fallback.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.chdir(tmp_path)
    prepared = prepared_source(tmp_path, image_source() if kind == "image" else video_source())
    direct = prepared.executor.pixel_unit(prepared.probes[0], index, cancelled=lambda: False)
    with direct.borrow() as borrowed:
        expected = borrowed.values.copy()  # One test-only unit, not a video archive.
    direct.close()
    context = dict(generation=3, profile_identity=PROFILE, media_identity=prepared.plan.media_identity)
    original_probe = prepared.probes[0]
    entered, release, calls = worker_gate(monkeypatch)
    work, pixels = MediaWork(prepared.executor), None
    try:
        assert work.request(prepared, index, **context)
        assert entered.wait(3)
        assert not work.ready(prepared, index, **context)
        assert work.take_ready(prepared, index, **context) is None
        assert prepared.replay._borrows >= 1
        release.set()
        deadline = time.monotonic() + prepared.executor.envelope.max_worker_wall_seconds + prepared.executor.envelope.retirement_grace_seconds
        while not work.ready(prepared, index, **context):
            assert time.monotonic() < deadline, "real CPU unit did not finish within its admitted worker wait"
            threading.Event().wait(.01)
        pixels = work.take_ready(prepared, index, **context)
        with pixels.borrow() as borrowed:
            assert np.array_equal(borrowed.values, expected)
        assert prepared.probes[0] is original_probe and len(calls) == 1
        work.check_health()
    finally:
        release.set()
        work.close()
        if pixels is not None:
            pixels.close()
        prepared.close()
    assert prepared.executor.store._staging == prepared.executor.store._bytes == 0


def test_real_installed_worker_cancellation_retires_process_before_encoded_release(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.chdir(tmp_path)
    prepared = prepared_source(tmp_path, image_source())
    context = dict(generation=3, profile_identity=PROFILE, media_identity=prepared.plan.media_identity)
    entered, release, _calls = worker_gate(monkeypatch)
    work = MediaWork(prepared.executor)
    try:
        work.request(prepared, 0, **context)
        assert entered.wait(3) and not work.cancel(prepared)
        assert prepared.replay._borrows >= 1
        release.set()
        job = work._jobs[prepared]
        assert job.work.retired.wait(prepared.executor.envelope.max_worker_wall_seconds +
                                     prepared.executor.envelope.retirement_grace_seconds)
        assert type(job.work.failure) is MediaCancelled
        prepared.executor.check_health()  # Includes actual supervised session/descriptor retirement.
        assert work.cancel(prepared) and prepared.replay._borrows == 0
    finally:
        release.set()
        work.close()
        prepared.close()
    assert prepared.executor.store._staging == prepared.executor.store._bytes == 0
