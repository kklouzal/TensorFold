"""Harness token counting uses native CPU preparation and preserves admission limits."""

import base64
import importlib.util
from http.client import HTTPMessage
from io import BytesIO
import json
from pathlib import Path
import socket
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, models, pre_tokenizers

from tensorfold.cuda.server import App
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import CapacityError, RequestError
from tensorfold.vision.images import ImageLimits


@pytest.fixture(scope="module")
def bridge():
    path = Path(__file__).resolve().parents[1] / "deploy" / "gb10" / "tensorfold_harness.py"
    spec = importlib.util.spec_from_file_location("test_tensorfold_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoGenerationEngine:
    eos = (0,)
    context_window = 8
    vision = None

    def generate(self, *args, draft=True, constraint=None, **kwargs):
        raise AssertionError("token counting must not generate")

    def prefill(self, *args, **kwargs):
        raise AssertionError("token counting must not prefill")


@pytest.fixture
def app(tmp_path):
    vocabulary = {"[UNK]": 0, "header": 1, "thinking": 2, "tools": 3, "w": 4}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    template = ("{% for m in messages %}{{ m.content }} {% endfor %}header"
                "{% if enable_thinking %} thinking{% endif %}{% if tools %} tools{% endif %}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 8}))
    return App(NoGenerationEngine(), tmp_path, "served", aliases=("alias",),
               context_window=8, max_tokens=2, thinking_budget=0,
               sampling={"temperature": 0})


def request(count=3, **options):
    return {"model": "served", "messages": [{"role": "user", "content": " ".join(["w"] * count)}],
            **options}


def test_native_template_count_and_inspection_without_generation(bridge, app):
    body = request(model="alias", return_tokens=True, return_prompt=True,
                   chat_template_kwargs={"enable_thinking": True},
                   tools=[{"type": "function", "function": {"name": "lookup", "parameters": {
                       "type": "object", "properties": {}}}}])
    capabilities = {"structured_output": True, "vision": False, "video": False}
    result = bridge.count_request(app, body, RequestError, capabilities)
    assert result == {"count": 6, "model": "alias", "context_length": 8,
                      "harness_tokenizer_version": bridge.BRIDGE_VERSION,
                      "tokens": [4, 4, 4, 1, 2, 3], "prompt": "w w w header thinking tools",
                      "capabilities": capabilities}


def test_oversized_history_count_preserves_generation_admission(bridge, app):
    body = request(20, max_tokens=2)
    assert bridge.count_request(app, body, RequestError)["count"] == 21
    assert app.effective_context_window == 8
    with pytest.raises(RequestError, match="context"):
        app.prepare(body, True)


@pytest.mark.parametrize("body, problem", [
    ([], "JSON object"),
    ({}, "model"),
    (request(model="unknown"), "model"),
    (request(model=False), "model"),
    (request(return_tokens=1), "boolean"),
    (request(return_prompt=None), "boolean"),
    (request(messages={}), "messages must be a list"),
    (request(messages=[{"role": "unknown", "content": "w"}]), "role"),
    (request(chat_template_kwargs=[]), "chat_template_kwargs"),
    (request(top_p="invalid"), "top_p"),
    (request(max_tokens="invalid"), "max_tokens"),
    (request(reasoning_effort="unknown"), "reasoning_effort"),
    (request(response_format={"type": "unknown"}), "response_format"),
])
def test_native_validation_is_retained(bridge, app, body, problem):
    with pytest.raises(RequestError, match=problem):
        bridge.count_request(app, body, RequestError)


@pytest.mark.parametrize("inspection", ["return_tokens", "return_prompt"])
def test_inspection_limit_leaves_count_only_available(bridge, app, inspection):
    body = request(bridge.MAX_RETURN_TOKENS - 1, **{inspection: True})
    assert bridge.count_request(app, body, RequestError)["count"] == bridge.MAX_RETURN_TOKENS
    oversized = request(bridge.MAX_RETURN_TOKENS, **{inspection: True})
    with pytest.raises(RequestError, match="prompt inspection is limited"):
        bridge.count_request(app, oversized, RequestError)
    oversized.pop(inspection)
    result = bridge.count_request(app, oversized, RequestError)
    assert result["count"] == bridge.MAX_RETURN_TOKENS + 1
    assert "tokens" not in result and "prompt" not in result


class CpuVision:
    allow_urls = False
    videos = False

    def __init__(self):
        self.calls = []

    def prepare(self, text, images, *, max_prompt_tokens):
        self.calls.append((text, images, max_prompt_tokens))
        if max_prompt_tokens is not None and max_prompt_tokens < 24:
            raise ValueError("expanded image prompt exceeds context")
        return SimpleNamespace(token_ids=[4] * 23 + [1])


@pytest.fixture
def image_body():
    Image = pytest.importorskip("PIL.Image")

    data = BytesIO()
    Image.new("RGB", (2, 2), (11, 22, 33)).save(data, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode("ascii")
    return {"model": "served", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "w"}, {"type": "image_url", "image_url": {"url": url}}
    ]}]}


def test_image_expansion_uses_native_preparation_without_context_admission(bridge, app, image_body):
    app.vision = CpuVision()
    result = bridge.count_request(app, image_body, RequestError)
    assert result["count"] == 24 and result["context_length"] == 8
    assert app.vision.calls[0][2] is None
    image = app.vision.calls[0][1][0]
    assert (image.width, image.height, image.pixels) == (2, 2, bytes((11, 22, 33)) * 4)
    assert app.effective_context_window == 8
    with pytest.raises(RequestError, match="context"):
        app.prepare(image_body, True)
    assert app.vision.calls[-1][2] == 8


def test_counting_retains_image_count_and_pixel_safety_limits(bridge, app, image_body):
    app.vision = CpuVision()
    app.image_limits = ImageLimits(max_images=1)
    parts = image_body["messages"][0]["content"]
    parts.append(parts[-1])
    with pytest.raises(RequestError, match="at most 1 images"):
        bridge.count_request(app, image_body, RequestError)
    assert not app.vision.calls
    parts.pop()
    app.image_limits = ImageLimits(max_pixels=3)
    with pytest.raises(RequestError, match="pixel"):
        bridge.count_request(app, image_body, RequestError)
    assert not app.vision.calls


def assert_gate_resources_released(gate):
    held = []
    try:
        for _ in range(16):
            assert gate._waiters.acquire(blocking=False)
            held.append(True)
        assert not gate._waiters.acquire(blocking=False)
    finally:
        for _ in held:
            gate._waiters.release()
    assert gate._slot.acquire(blocking=False)
    gate._slot.release()


@pytest.mark.parametrize("phase", ["before_acquire", "after_acquire", "body"])
def test_preparation_gate_releases_resources_on_cancellation_and_failure(bridge, phase):
    gate = bridge.PreparationGate()
    answers = iter([phase == "before_acquire", phase == "after_acquire"])
    with pytest.raises(RequestCancelled if phase != "body" else RuntimeError):
        with gate.enter(lambda: next(answers), CapacityError, RequestCancelled):
            raise RuntimeError("preparation failed")
    assert_gate_resources_released(gate)


def test_preparation_gate_timeout_and_full_queue_release_resources(bridge, monkeypatch):
    gate = bridge.PreparationGate()
    gate._slot.acquire()
    clock = iter((0.0, bridge.PREPARATION_TIMEOUT_S + 1))
    with monkeypatch.context() as patch:
        patch.setattr(bridge.time, "monotonic", lambda: next(clock))
        with pytest.raises(CapacityError, match="busy"):
            with gate.enter(lambda: False, CapacityError, RequestCancelled):
                pytest.fail("busy preparation must not enter")
    gate._slot.release()
    assert_gate_resources_released(gate)
    for _ in range(16):
        gate._waiters.acquire()
    try:
        with pytest.raises(CapacityError, match="queue is full"):
            with gate.enter(lambda: False, CapacityError, RequestCancelled):
                pytest.fail("full queue must not enter")
    finally:
        for _ in range(16):
            gate._waiters.release()
    assert_gate_resources_released(gate)


def test_preparation_gate_serializes_concurrent_preparation(bridge):
    gate = bridge.PreparationGate()
    waiting, entered = threading.Event(), threading.Event()
    errors = []

    def cancelled():
        waiting.set()
        return False

    def worker():
        try:
            with gate.enter(cancelled, CapacityError, RequestCancelled):
                entered.set()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    with gate.enter(lambda: False, CapacityError, RequestCancelled):
        thread.start()
        assert waiting.wait(2)
        assert not entered.is_set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not errors and entered.is_set()
    assert_gate_resources_released(gate)


def handler_post(bridge, app, payload, headers, path="/v1/tokenize"):
    class NativeHandler:
        def do_POST(self):
            return "native route"

        def _json(self, status, result):
            return status, result

    handler_type = bridge.make_handler_factory(lambda ignored: NativeHandler)(app)
    handler = handler_type()
    parsed_headers = HTTPMessage()
    for name, value in headers.items():
        parsed_headers[name] = value
    handler.path, handler.headers, handler.rfile = path, parsed_headers, BytesIO(payload)
    handler.close_connection = False
    handler.server = SimpleNamespace(stopping=threading.Event())
    connected, peer = socket.socketpair()
    try:
        connected.settimeout(7)
        handler.connection = connected
        result = handler.do_POST()
        assert connected.gettimeout() == 7
        return result
    finally:
        connected.close()
        peer.close()


def test_tokenize_route_and_native_route_delegation(bridge, app):
    payload = json.dumps(request()).encode()
    status, result = handler_post(bridge, app, payload, {"Content-Length": str(len(payload))},
                                 "/v1/tokenize/?probe=1")
    assert status == 200 and result["count"] == 4
    assert result["capabilities"]["vision"] is False
    assert handler_post(bridge, app, payload, {}, "/v1/chat/completions") == "native route"


@pytest.mark.parametrize("payload, headers, problem", [
    (b"{}", {}, "body must be nonempty"),
    (b"", {"Content-Length": "0"}, "body must be nonempty"),
    (b"{}", {"Content-Length": "invalid"}, "Content-Length must contain a decimal byte count"),
    (b"{}", {"Content-Length": str(96 * 1024**2 + 1)}, "exceeds the 96 MiB limit"),
    (b"{}", {"Content-Length": "3"}, "ended before Content-Length"),
    (b"{}", {"Content-Length": "2", "Transfer-Encoding": "chunked"}, "Transfer-Encoding is unsupported; use Content-Length"),
    (b"[}", {"Content-Length": "2"}, "valid UTF-8 JSON"),
    (b"\xff", {"Content-Length": "1"}, "valid UTF-8 JSON"),
    (b"{\"x\":NaN}", {"Content-Length": "9"}, "valid UTF-8 JSON"),
    (b"[]", {"Content-Length": "2"}, "JSON object"),
])
def test_tokenize_http_boundary_refusals(bridge, app, payload, headers, problem):
    status, result = handler_post(bridge, app, payload, headers)
    assert status == 400
    assert problem in result["error"]["message"]


def test_harness_factory_requires_request_selected_thinking_budget(bridge, app):
    app.thinking_budget = 1
    with pytest.raises(ValueError, match="thinking-budget 0"):
        bridge.make_handler_factory(lambda ignored: object)(app)
