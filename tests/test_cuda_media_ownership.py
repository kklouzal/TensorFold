"""Actual CUDA App replay ownership with a synchronous fake engine, no GPU.

The fixture installs a real prepared replay/registry and canonical position
contract. Pixel preparation and engine math have independent coverage; these
controls exercise the actual server publication, subrun and caller boundaries.
"""
from __future__ import annotations

import hashlib
import io
import os
from types import SimpleNamespace as NS
import weakref

import pytest

pytest.importorskip("torch")

from tensorfold.cuda import server
from tensorfold.families.qwen4_exp.cuda.media_requests import MediaRequest, MediaRequests
from tensorfold.families.qwen4_exp.cuda.media_runtime import MediaRuntime
from tensorfold.families.qwen4_exp.media_plan import MediaPlanContract
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import ContextLengthError
from tensorfold.server.prompts import RenderedPrompt
from tensorfold.server.token_routes import count_chat
from tensorfold.vision.media_frontend import MediaExecutor, PreparedReplayMedia
from tensorfold.vision.media_worker import PROCESSOR_ID
from tensorfold.vision.replay import MediaProbe, ReplayLease, ReplayRef, ReplayStore
from tests.test_cuda_thinking_controls import ChainEngine, app_for
from tests.test_cuda_tool_choice import THINK
from tests.test_media_replay import envelope
from tests.test_token_routes import VocabTokens


MESSAGES = [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]


class MediaEngine(ChainEngine):
    """Synchronous fake rank retirement around the actual request registry."""

    def __init__(self, runtime, events):
        super().__init__()
        self.vision, self.events, self.entered = runtime, events, []
        self.fail_at = self.cancel_at = None
        self.failure = RuntimeError("engine failed")
        self.cancelled = False
        self.defer = False
        self.deferred_request = None

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, *, vision=None):
        number = len(self.entered) + 1
        assert vision.replay._closed is False, "logical media retired before a generation subrun"
        assert tuple(prompt) == vision.token_ids
        self.entered.append(vision)
        self.events.append(("engine-enter", number))
        request = MediaRequest(vision, number, self.vision)
        self.vision.requests.claim(request)
        try:
            if number == self.fail_at:
                raise self.failure
            if number == self.cancel_at:
                self.cancelled = True
            return super().generate(prompt, max_tokens, sampling, on_tokens, draft=draft)
        finally:
            self.events.append(("engine-retired", number))
            if self.defer:
                self.deferred_request = request
            else:
                request.retired = True
                self.vision.requests.release(request)
                assert not vision.replay._closed, "a backend subrun closed the logical replay owner"


@pytest.fixture
def media_case(tmp_path, monkeypatch):
    from PIL import Image
    from tensorfold.server import prompts

    bounds = envelope()
    root = tmp_path / "store"
    root.mkdir()
    store = ReplayStore(root, bounds)
    executor = MediaExecutor(store)
    directory = root / "encoded"
    directory.mkdir()
    image = io.BytesIO()
    Image.new("RGB", (96, 64), (23, 47, 89)).save(image, format="PNG")
    raw = image.getvalue()
    path = directory / "0"
    path.write_bytes(raw)
    path.chmod(0o400)
    ref = ReplayRef(str(path), hashlib.sha256(raw).hexdigest(), len(raw), 0, "image", "auto")
    # Explicit physical owner fixture, as in test_flashnext_media_requests.
    # This does not substitute a fake close or run an unrelated decoder job.
    reservation = object()
    lease = ReplayLease(store, directory, (ref,), reservation)
    store._replay_reservations[reservation] = (len(raw), 1)
    store._live[ref.path] = lease
    probe = MediaProbe(ref, bounds.identity, PROCESSOR_ID, 96, 64, 1, (), (), None,
                       "encoded-orientation-pil-rgb-v1", (224, 320), (1, 14, 20), 6144, 24576)
    contract = MediaPlanContract(("1" * 64, "2" * 64, "3" * 64, "4" * 64), PROCESSOR_ID,
                                 (2, 3, 4, 5), 1100, 512, 8, 128, 256)
    ids = (ord("p"), 4, *([2] * probe.visual_rows), 5, THINK)
    plan = contract.from_expanded(ids, (probe,))
    prepared = PreparedReplayMedia(plan, (probe,), lease, executor)
    runtime = MediaRuntime.__new__(MediaRuntime)
    runtime.pid, runtime.closed = os.getpid(), False
    runtime.producer = object()  # No device method is called by these App controls.
    runtime.work = None  # This synchronous App fixture issues no CPU decoder work.
    runtime.store, runtime.executor, runtime.profile = store, executor, NS(cpu=bounds)
    runtime.identity, runtime.contract = plan.profile_identity, contract
    runtime.tokenizer = VocabTokens()
    runtime.frontend = NS(continued=lambda *a: pytest.fail("legacy frontend continuation selected"))
    runtime.requests = MediaRequests(runtime)
    events = []
    engine = MediaEngine(runtime, events)
    app = app_for(tmp_path, engine)
    app.tok, app.vision = runtime.tokenizer, runtime
    original_close = runtime.requests.close_if_unclaimed

    def close(value):
        assert value.replay is lease
        events.append(("logical-close", len(engine.entered)))
        return original_close(value)

    monkeypatch.setattr(runtime.requests, "close_if_unclaimed", close)
    monkeypatch.setattr(prompts, "prepare_images", lambda *a, **k: RenderedPrompt(list(ids), vision=prepared))
    case = NS(app=app, engine=engine, runtime=runtime, prepared=prepared, lease=lease,
              store=store, directory=directory, events=events, close=original_close)
    yield case
    # The synchronous fake has no outstanding CPU/rank operation on fixture
    # retirement, even when a test intentionally deferred its bridge release.
    for request in tuple(runtime.requests.active.values()):
        request.retired, request.close_requested = True, True
        runtime.requests.release(request)
    if not lease._closed:
        lease.close()


def body(**fields):
    return {"messages": MESSAGES, "max_tokens": 24, **fields}


def test_app_adopts_the_leader_tokenizer_without_reloading_it(media_case, tmp_path, monkeypatch):
    from tensorfold import token_metadata

    case = media_case
    case.engine.chat_template = case.app.template
    monkeypatch.setattr(token_metadata, "load_tokenizer", lambda *a, **k: pytest.fail("leader tokenizer reloaded"))
    monkeypatch.setattr(server, "_segment_encoder", lambda *a: None)
    app = server.App(case.engine, tmp_path, "owned-media", context_window=0)
    assert app.tok is case.runtime.tokenizer


@pytest.mark.parametrize("route", ["tokenize", "count_chat"])
def test_token_only_routes_retire_unclaimed_replay_without_generation(media_case, route):
    case = media_case
    if route == "tokenize":
        result = case.app.tokenize(body())
        assert result["tokens"] == list(case.prepared.token_ids)
        count = result["count"]
    else:
        count = count_chat(case.app, body())
    assert count == len(case.prepared.token_ids)
    assert case.lease._closed and not case.directory.exists()
    assert case.runtime.requests.active == {} and case.engine.entered == []
    assert case.events == [("logical-close", 0)]


def test_prepare_validation_failure_retires_the_media_owner(media_case):
    case = media_case
    case.app.context_window = len(case.prepared.token_ids) + 1
    with pytest.raises(ContextLengthError):
        case.app.prepare(body(max_tokens=2), True)
    assert case.lease._closed and case.engine.entered == []


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_prepare_exception_cleanup_preserves_primary(media_case, monkeypatch, cleanup_failure):
    case = media_case
    primary, cleanup = ValueError("validation failed"), OSError("close failed")
    observed = []
    def check(*args, **kwargs):
        observed.append(kwargs["prepared"])
        assert not case.lease._closed
        raise primary
    monkeypatch.setattr(case.app, "check", check)
    if cleanup_failure:
        def failed(*args):
            raise cleanup
        monkeypatch.setattr(case.runtime.requests, "close_if_unclaimed", failed)
    with pytest.raises(ValueError) as raised:
        case.app.prepare(body(), True)
    assert raised.value is primary
    if cleanup_failure:
        assert not case.lease._closed and observed[0]._media_release is not None
        assert BaseException.__cause__.__get__(primary) is cleanup
    else:
        assert case.lease._closed and observed[0]._media_release is None


@pytest.mark.parametrize("failure_at", ["sampling", "constructor"])
def test_prepared_request_publication_failure_retires_real_replay(media_case, monkeypatch, failure_at):
    case = media_case
    primary = MemoryError("prepared publication failed")
    def failed(*args, **kwargs):
        assert not case.lease._closed
        raise primary
    if failure_at == "sampling":
        monkeypatch.setattr(case.app, "sampling_for", failed)
    else:
        monkeypatch.setattr(server, "PreparedRequest", failed)
    with pytest.raises(MemoryError) as raised:
        case.app.prepare(body(), True)
    assert raised.value is primary and case.lease._closed


def test_close_authority_failure_cannot_strand_transferred_media(media_case, monkeypatch):
    from tensorfold.server import prompts

    case = media_case
    primary = MemoryError("close authority lookup failed")
    transferred = []
    def prepared(*args, **kwargs):
        transferred.append(case.prepared)
        return RenderedPrompt(list(case.prepared.token_ids), vision=case.prepared)
    original = MediaRequests.__getattribute__
    def refusing(self, name):
        if self is case.runtime.requests and name == "close_if_unclaimed":
            raise primary
        return original(self, name)
    monkeypatch.setattr(prompts, "prepare_images", prepared)
    monkeypatch.setattr(MediaRequests, "__getattribute__", refusing)
    with pytest.raises(MemoryError) as raised:
        case.app._prepare(body(), True)
    assert raised.value is primary
    assert not transferred or case.lease._closed


def test_token_string_publication_runs_after_media_retirement(media_case, monkeypatch):
    case = media_case
    primary = MemoryError("token strings failed")
    def failed(token):
        assert case.lease._closed and not case.directory.exists()
        raise primary
    monkeypatch.setattr(case.app.tok, "id_to_token", failed)
    with pytest.raises(MemoryError) as raised:
        case.app.tokenize(body(return_token_strs=True))
    assert raised.value is primary and case.lease._closed


def test_logical_thinking_generation_keeps_replay_across_all_subruns(media_case, monkeypatch):
    from tensorfold.vision import qwen_processing

    case = media_case
    monkeypatch.setattr(qwen_processing, "continued", lambda *a: pytest.fail("legacy continuation selected"))
    prepared = case.app.prepare(body(thinking_budget=3), True)
    result = case.app.run(body(thinking_budget=3), True, lambda _: True, prepared=prepared)
    assert result["completion_tokens"] == 24 and len(case.engine.entered) == 2
    assert case.engine.entered[0] is case.prepared
    assert case.engine.entered[1].replay is case.prepared.replay
    assert case.engine.entered[1].plan.media_identity == case.prepared.plan.media_identity
    assert case.engine.entered[1].token_ids[:len(case.prepared.token_ids)] == case.prepared.token_ids
    assert case.events[-1] == ("logical-close", 2)
    assert [event for event in case.events if event[0] == "logical-close"] == [("logical-close", 2)]
    assert prepared._media_release is None and case.lease._closed


@pytest.mark.parametrize("failure_at", [1, 2])
def test_generation_failure_releases_only_after_backend_retirement(media_case, failure_at):
    case = media_case
    case.engine.fail_at = failure_at
    prepared = case.app.prepare(body(thinking_budget=3), True)
    with pytest.raises(RuntimeError) as raised:
        case.app.run(body(thinking_budget=3), True, lambda _: True, prepared=prepared)
    assert raised.value is case.engine.failure
    assert case.events[-2:] == [("engine-retired", failure_at), ("logical-close", failure_at)]
    assert case.lease._closed and prepared._media_release is None


def test_emitter_failure_waits_for_engine_retirement_and_preserves_primary(media_case):
    case = media_case
    primary = ValueError("output callback failed")
    prepared = case.app.prepare(body(max_tokens=2), True)
    def failed(_chunk):
        raise primary
    with pytest.raises(ValueError) as raised:
        case.app.run(body(max_tokens=2), True, failed, prepared=prepared)
    assert raised.value is primary
    assert case.events[-2:] == [("engine-retired", 1), ("logical-close", 1)]
    assert case.lease._closed and prepared._media_release is None


def test_generation_failure_with_failed_media_cleanup_retains_owner_and_primary(media_case, monkeypatch):
    case = media_case
    cleanup = OSError("media final close failed")
    def failed(_vision):
        raise cleanup
    monkeypatch.setattr(case.runtime.requests, "close_if_unclaimed", failed)
    case.engine.fail_at = 1
    prepared = case.app.prepare(body(), True)
    with pytest.raises(RuntimeError) as raised:
        case.app.run(body(), True, lambda _: True, prepared=prepared)
    assert raised.value is case.engine.failure
    assert BaseException.__cause__.__get__(case.engine.failure) is cleanup
    assert not case.lease._closed and prepared._media_release is not None


@pytest.mark.parametrize("cancel_at", [0, 1, 2])
def test_cancellation_retires_media_after_the_last_started_subrun(media_case, cancel_at):
    case = media_case
    case.engine.cancel_at = cancel_at
    case.engine.cancelled = cancel_at == 0
    prepared = case.app.prepare(body(thinking_budget=3), True)
    with pytest.raises(RequestCancelled):
        case.app.run(body(thinking_budget=3), True, lambda _: True, prepared=prepared,
                     cancelled=lambda: case.engine.cancelled)
    assert len(case.engine.entered) == cancel_at and case.events[-1] == ("logical-close", cancel_at)
    assert case.lease._closed and prepared._media_release is None


def test_deferred_bridge_close_keeps_callback_until_later_true_and_profile_release_independent(media_case):
    case = media_case
    case.app.default_thinking = False
    case.engine.defer = True
    prepared = case.app.prepare(body(max_tokens=2), True)
    releases = []
    class CloseOnce:
        closed = False
        def __init__(self, name):
            self.name = name
        def close(self):
            if not self.closed:
                self.closed = True
                releases.append(self.name)
    prepared.profile_binding, prepared._admission = CloseOnce("binding"), CloseOnce("admission")
    prepared._profile_finalizer = weakref.finalize(prepared, lambda: releases.append("unexpected-gc"))
    case.app.run(body(max_tokens=2), True, lambda _: True, prepared=prepared)
    request = case.engine.deferred_request
    assert request.close_requested and not case.lease._closed
    assert prepared._media_release is not None and releases == []
    prepared.close()
    assert prepared._media_release is not None and releases == ["binding", "admission"]
    request.retired = True
    case.runtime.requests.release(request)
    assert case.lease._closed and not case.directory.exists()
    prepared.close()
    assert prepared._media_release is None
    count = len(case.events)
    prepared.close()
    assert len(case.events) == count and releases == ["binding", "admission"]


@pytest.mark.parametrize("close_failure", [False, True])
def test_generic_count_failure_still_closes_the_actual_prepared_owner(media_case, monkeypatch, close_failure):
    case = media_case
    prepare = case.app.prepare
    primary = MemoryError("count result failed")
    cleanup = OSError("count cleanup failed")
    if close_failure:
        def failed_close(_prepared):
            raise cleanup
        monkeypatch.setattr(case.runtime.requests, "close_if_unclaimed", failed_close)
    class FailingCount(list):
        def __len__(self):
            raise primary
    def prepared(*args, **kwargs):
        value = prepare(*args, **kwargs)
        value.prompt = FailingCount(value.prompt)
        return value
    monkeypatch.setattr(case.app, "prepare", prepared)
    with pytest.raises(MemoryError) as raised:
        count_chat(case.app, body())
    assert raised.value is primary
    assert case.lease._closed is not close_failure
    if close_failure:
        assert raised.value.__cause__ is cleanup and case.directory.exists()


@pytest.mark.parametrize("close_failure", [False, True])
def test_tokenize_prompt_handoff_preserves_interruption_and_cleanup(media_case, monkeypatch, close_failure):
    case = media_case
    prepare = case.app._prepare
    primary = KeyboardInterrupt("tokenization handoff interrupted")
    cleanup = OSError("tokenization media close failed")

    class InterruptedRequest(server.PreparedRequest):
        __slots__ = ()
        __repr__ = object.__repr__

        def __getattribute__(self, name):
            if name == "prompt":
                raise primary
            return super().__getattribute__(name)

    def prepared(*args, **kwargs):
        value = prepare(*args, **kwargs)
        value.__class__ = InterruptedRequest
        return value

    if close_failure:
        def failed(_prepared):
            raise cleanup
        monkeypatch.setattr(case.runtime.requests, "close_if_unclaimed", failed)
    monkeypatch.setattr(case.app, "_prepare", prepared)
    with pytest.raises(KeyboardInterrupt) as caught:
        case.app.tokenize(body())
    assert caught.value is primary
    assert case.lease._closed is not close_failure
    if close_failure:
        assert caught.value.__cause__ is cleanup and case.directory.exists()
