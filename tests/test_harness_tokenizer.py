"""Harness token counting uses native CPU preparation and preserves admission limits."""

import base64
import hashlib
import importlib.util
from http.client import HTTPMessage
from io import BytesIO
import json
import marshal
from pathlib import Path
import socket
import sys
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


TEMPLATE = ("{% for m in messages %}{{ m.content }} {% endfor %}header"
            "{% if enable_thinking %} thinking{% endif %}{% if tools %} tools{% endif %}")


def served_app(directory, template=TEMPLATE, *, separate_template=None, **options):
    """A real App over a small tokenizer; ``separate_template`` writes chat_template.jinja."""
    vocabulary = {"[UNK]": 0, "header": 1, "thinking": 2, "tools": 3, "w": 4}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    if separate_template is not None:
        (directory / "chat_template.jinja").write_text(separate_template)
    (directory / "config.json").write_text(json.dumps({"max_position_embeddings": 8}))
    return App(NoGenerationEngine(), directory, "served", aliases=("alias",),
               context_window=8, max_tokens=2, thinking_budget=0,
               sampling={"temperature": 0}, **options)


@pytest.fixture
def app(tmp_path):
    return served_app(tmp_path)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def test_metadata_identifies_the_active_template_and_canonical_efforts(bridge, tmp_path):
    template = TEMPLATE + ("{% if reasoning_effort in ['low', 'medium'] %} brief"
                           "{% elif reasoning_effort == 'xhigh' %} deep{% endif %}")
    app = served_app(tmp_path, template, reasoning_effort="high")
    capabilities, fingerprint, provenance = bridge.capability_metadata(app)
    # Exact: thinking-control, default-thinking and reasoning-replay facts stay absent.
    assert capabilities == {
        "structured_output": importlib.util.find_spec("xgrammar") is not None,
        "vision": False, "video": False,
        "reasoning_efforts": ["low", "medium", "xhigh"],
        "default_reasoning_effort": "xhigh",            # the template's nearest named level, ties upward
    }
    assert fingerprint == sha256(template)

    def code_sha256(function):
        return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()

    assert provenance == {
        "server_version": "0.6.6", "bridge_version": bridge.BRIDGE_VERSION,
        "server_type": "tensorfold.cuda.server.App",
        "engine_type": f"{NoGenerationEngine.__module__}.NoGenerationEngine",
        "tokenizer_type": f"{Tokenizer.__module__}.{Tokenizer.__qualname__}",
        "server_prepare_sha256": code_sha256(App._prepare),
        "server_check_fields_sha256": code_sha256(app._check_fields),
        "engine_generate_sha256": code_sha256(NoGenerationEngine.generate),
        "code_identity": "python_runtime_bytecode",
        "python_version": sys.version.split()[0], "marshal_version": marshal.version,
        "tokenizer_sha256": sha256(app.tok.to_str()), "tokenizer_identity": "runtime_serialization",
        "template_source": "tokenizer_config.chat_template",
        "template_identity": "equivalent_active_compilation",
    }


@pytest.mark.parametrize("template, effort, efforts", [
    (TEMPLATE, None, []),
    (TEMPLATE, "high", []),                                             # no named levels: high resolves to xhigh
    (TEMPLATE + "{% if reasoning_effort == 'low' %} brief{% endif %}", "xhigh", ["low"]),  # xhigh stays xhigh
    (TEMPLATE + "{% if reasoning_effort == 'low' %} brief{% endif %}", "none", ["low"]),   # none is never canonical
])
def test_default_effort_is_omitted_when_unset_or_not_named_by_the_template(bridge, tmp_path, template, effort, efforts):
    capabilities, _, _ = bridge.capability_metadata(served_app(tmp_path, template, reasoning_effort=effort))
    assert capabilities["reasoning_efforts"] == efforts
    assert "default_reasoning_effort" not in capabilities


def test_separate_template_file_takes_precedence_and_exact_source_is_hashed(bridge, tmp_path):
    source = "  café {{ messages }}\n"
    app = served_app(tmp_path, "different {{ messages }}", separate_template=source)
    _, fingerprint, provenance = bridge.capability_metadata(app)
    assert fingerprint == sha256(source)
    assert provenance["template_source"] == "chat_template.jinja"


@pytest.mark.parametrize("config", [
    json.dumps({"chat_template": "changed {{ messages }}{% if reasoning_effort == 'low' %} brief{% endif %}"}),
    json.dumps({"chat_template": "{% if"}),
    json.dumps({"chat_template": [{"name": "default", "template": TEMPLATE}]}),
    json.dumps({}), "[]", "{", b"\xff", None,
], ids=["changed", "invalid_jinja", "template_list", "no_template", "json_array", "malformed_json", "undecodable",
        "missing"])
def test_changed_or_unusable_source_is_not_claimed_as_active(bridge, tmp_path, config):
    app = served_app(tmp_path, TEMPLATE + "{% if reasoning_effort == 'medium' %} more{% endif %}")
    path = tmp_path / "tokenizer_config.json"
    if config is None:
        path.unlink()
    elif isinstance(config, bytes):
        path.write_bytes(config)
    else:
        path.write_text(config)
    capabilities, fingerprint, provenance = bridge.capability_metadata(app)
    assert fingerprint is None
    assert "template_identity" not in provenance and "template_source" not in provenance
    assert capabilities["reasoning_efforts"] == ["medium"]                  # the initialized template's, not the file's


def test_changed_block_is_rejected_even_when_root_code_is_unchanged(bridge, tmp_path):
    app = served_app(tmp_path, "{% block result %}active{% endblock %}")
    replacement = "{% block result %}changed{% endblock %}"
    candidate = app.template.template.environment.from_string(replacement)
    assert candidate.root_render_func.__code__ == app.template.template.root_render_func.__code__
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": replacement}))
    assert bridge.capability_metadata(app)[1] is None


def test_changed_generated_binding_or_default_is_not_equivalent(bridge, tmp_path):
    app = served_app(tmp_path)
    root = app.template.template.root_render_func
    name = next(name for name in root.__code__.co_names if name in root.__globals__)
    root.__globals__[name] = object()
    assert bridge.capability_metadata(app)[1] is None
    app = served_app(tmp_path)
    root = app.template.template.root_render_func
    root.__defaults__ = (*root.__defaults__[:-1], object())
    assert bridge.capability_metadata(app)[1] is None


def test_loaded_code_provenance_distinguishes_implementations_and_omits_unknown_code(bridge, app):
    initial = bridge.capability_metadata(app)[2]
    app.engine.generate = lambda *args, constraint=None, **kwargs: None
    changed = bridge.capability_metadata(app)[2]
    assert changed["engine_generate_sha256"] != initial["engine_generate_sha256"]
    assert changed["server_prepare_sha256"] == initial["server_prepare_sha256"]
    app.engine.generate = len
    assert "engine_generate_sha256" not in bridge.capability_metadata(app)[2]


class NativeHandler:
    def do_POST(self):
        return "native route"

    def _json(self, status, result):
        return status, result


def handler_post(bridge, app, payload, headers, path="/v1/tokenize", handler_type=None):
    handler_type = handler_type or bridge.make_handler_factory(lambda ignored: NativeHandler)(app)
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
    assert result["capabilities"]["reasoning_efforts"] == []
    assert result["chat_template_sha256"] == sha256(TEMPLATE)
    assert result["provenance"]["template_identity"] == "equivalent_active_compilation"
    assert handler_post(bridge, app, payload, {}, "/v1/chat/completions") == "native route"


def test_tokenize_metadata_keeps_the_initialized_template_identity_after_file_changes(bridge, app):
    handler_type = bridge.make_handler_factory(lambda ignored: NativeHandler)(app)
    (app.model_dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": "changed"}))
    payload = json.dumps(request()).encode()
    status, result = handler_post(bridge, app, payload, {"Content-Length": str(len(payload))},
                                  handler_type=handler_type)
    assert status == 200
    assert result["chat_template_sha256"] == sha256(TEMPLATE)


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


def test_launcher_pin_matches_owned_package_source(bridge):
    import tensorfold
    assert bridge.TENSORFOLD_VERSION == tensorfold.__version__ == '0.6.6'


@pytest.fixture
def matching_install(bridge):
    import tensorfold
    from importlib import metadata
    try:
        installed = metadata.version('tensorfold')
    except metadata.PackageNotFoundError:
        pytest.skip('launcher qualification needs an installed owned TensorFold distribution')
    if installed != bridge.TENSORFOLD_VERSION or tensorfold.__version__ != bridge.TENSORFOLD_VERSION:
        pytest.skip('launcher qualification needs matching installed/module TensorFold 0.6.6')
    return installed


@pytest.mark.parametrize('module_version,installed_version', [('0.6.1', '0.6.6'), ('0.6.6', '0.6.1'),
                                                              ('0.6.7', '0.6.7')])
def test_launcher_refuses_inexact_versions_before_cli_loading(bridge, monkeypatch, capsys,
                                                              module_version, installed_version):
    import tensorfold
    from importlib import metadata
    from tensorfold import cli
    calls = []
    monkeypatch.setattr(tensorfold, '__version__', module_version)
    monkeypatch.setattr(metadata, 'version', lambda name: installed_version)
    monkeypatch.setattr(cli, 'main', lambda args: calls.append(args))
    assert bridge.main(['serve', 'controlled-model', '--backend', 'cuda', '--thinking-budget', '0']) == 2
    assert calls == []
    assert 'requires owned TensorFold 0.6.6' in capsys.readouterr().err


def test_launcher_refuses_missing_installed_distribution(bridge, monkeypatch, capsys):
    from importlib import metadata
    def absent(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(metadata, 'version', absent)
    assert bridge.main(['serve', 'controlled-model']) == 2
    assert 'requires installed TensorFold 0.6.6' in capsys.readouterr().err


@pytest.mark.parametrize('fail', [False, True])
def test_launcher_binds_both_preimported_cli_factory_surfaces_and_restores_after_dispatch(
        bridge, matching_install, app, monkeypatch, fail):
    from tensorfold import cli
    from tensorfold.cuda import http, server
    native = http.make_handler
    assert server.make_handler is native
    primary = KeyboardInterrupt('controlled launcher dispatch interruption')
    called = []
    def dispatch(arguments):
        called.append(arguments)
        assert http.make_handler is server.make_handler
        assert http.make_handler is not native
        # Use the actual CLI import surface and patched handler, with generation forbidden.
        patched = server.make_handler(app)
        base = native(app)
        assert patched.do_POST is not base.do_POST
        if fail:
            raise primary
        return 17
    monkeypatch.setattr(cli, 'main', dispatch)
    arguments = ['serve', 'controlled-model', '--backend', 'cuda', '--thinking-budget', '0']
    if fail:
        with pytest.raises(KeyboardInterrupt) as caught:
            bridge.main(arguments)
        assert caught.value is primary
    else:
        assert bridge.main(arguments) == 17
    assert called == [arguments]
    assert http.make_handler is server.make_handler is native


@pytest.mark.parametrize('boundary', ['framing', 'cancellation', 'exports'])
def test_launcher_refuses_incoherent_owned_interfaces_before_dispatch(
        bridge, matching_install, monkeypatch, capsys, boundary):
    from tensorfold import cli
    from tensorfold.cuda import server
    from tensorfold.server import request_body, cancellation
    if boundary == 'framing':
        monkeypatch.setattr(request_body, 'content_length', lambda handler, limit: 0)
    elif boundary == 'cancellation':
        monkeypatch.setattr(cancellation, 'socket_cancellation', lambda connection: None)
    else:
        monkeypatch.setattr(server, 'make_handler', lambda app: None)
    calls = []
    monkeypatch.setattr(cli, 'main', lambda arguments: calls.append(arguments))
    assert bridge.main(['serve', 'controlled-model', '--backend', 'cuda']) == 2
    assert calls == []
    assert 'requires' in capsys.readouterr().err


@pytest.mark.parametrize('backend,arguments,message', [
    ('cuda', ['models'], 'usage:'),
    ('cuda', ['serve', 'controlled-model', '--thinking-budget', '1'], 'thinking-budget 0'),
    ('mlx', ['serve', 'controlled-model', '--backend', 'mlx'], 'CUDA only'),
])
def test_exact_installed_launcher_preserves_supported_command_guards(
        bridge, matching_install, monkeypatch, capsys, backend, arguments, message):
    from tensorfold import cli
    calls = []
    monkeypatch.setattr(cli, 'main', lambda arguments: calls.append(arguments))
    assert bridge.main(arguments) == 2
    assert calls == []
    assert message in capsys.readouterr().err


def test_real_harness_token_route_keeps_authentication_rotation_and_reply_counts(bridge, app, tmp_path):
    import http.client
    from tensorfold.cuda.http import make_handler
    from tensorfold.server.authentication import KeyStore
    from tensorfold.server.http import Server
    from tensorfold.server import metrics
    key_file = tmp_path / 'keys'
    key_file.write_text('first: controlled-first\n')
    key_file.chmod(0o600)
    app.auth = KeyStore(key_file=key_file)
    from tensorfold.server.auth_http import Authenticated
    selected = bridge.make_handler_factory(make_handler)(app)
    assert issubclass(selected, Authenticated)
    assert selected.auth is app.auth
    server = Server(('127.0.0.1', 0), selected)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    worker.start()
    body = json.dumps(request(20)).encode()
    def post(key=None):
        client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        try:
            headers = {'Content-Type': 'application/json'}
            if key:
                headers['Authorization'] = 'Bearer ' + key
            client.request('POST', '/v1/tokenize', body, headers)
            response = client.getresponse()
            return response.status, json.loads(response.read())
        finally:
            client.close()
    try:
        assert post()[0] == 401
        assert post('controlled-wrong')[0] == 401
        status, counted = post('controlled-first')
        assert status == 200 and counted['count'] == 21 and counted['context_length'] == 8
        assert metrics.of(app).http_requests == {('unauthenticated', 401): 2, ('first', 200): 1}
        key_file.write_text('second: controlled-second\n')
        app.auth.request_reload()
        assert post('controlled-first')[0] == 401
        assert post('controlled-second')[0] == 200
        assert metrics.of(app).http_requests == {('unauthenticated', 401): 3, ('second', 200): 1}
        assert metrics.of(app).http_replies == {401: 3, 200: 2}
        key_file.write_text('')
        app.auth.request_reload()
        assert post('controlled-second')[0] == 401
        assert app.auth.enabled
        assert metrics.of(app).http_requests == {('unauthenticated', 401): 4}
        assert metrics.of(app).http_replies == {401: 4, 200: 2}
        with pytest.raises(RequestError, match='context'):
            app.prepare(request(20), True)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(3)
    assert not worker.is_alive()
