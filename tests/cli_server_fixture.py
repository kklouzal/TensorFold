"""Bounded fake listener ownership for CLI option-forwarding tests.

The CLI constructs the listener, passes that exact owner to ``serve``, then
closes it. These fixtures acquire no sockets and preserve that lifetime.
"""

import json
import threading
from hashlib import sha256
from types import SimpleNamespace


class PreparedTokenizer:
    """MLX host protocol over real tokenizers/Jinja operations, no array math.

    EOS and the rendered template are mutable through the same public metadata
    properties the MLX wrapper exposes. Core dependencies suffice on Linux.
    """

    def __init__(self, tokenizer, template):
        self._tokenizer = tokenizer
        self.eos_token_ids = (2,)
        self.chat_template = template

    @property
    def chat_template(self):
        return self._source

    @chat_template.setter
    def chat_template(self, value):
        from jinja2 import Template

        self._source, self._compiled = value, Template(value)

    @property
    def added_tokens_decoder(self):
        return self._tokenizer.get_added_tokens_decoder()

    @property
    def all_special_ids(self):
        return [index for index, token in self.added_tokens_decoder.items() if token.special]

    def get_vocab(self):
        return self._tokenizer.get_vocab()

    def encode(self, text, *, add_special_tokens=True):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, *, skip_special_tokens=True):
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False, **kwargs):
        text = self._compiled.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)
        return self.encode(text, add_special_tokens=False) if tokenize else text


def prepared_chat_template(directory, source="fixture"):
    """Actual captured template protocol, with isolated regular-file metadata."""
    from tensorfold.cuda.chat_template import ChatTemplate

    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": source}))
    return ChatTemplate(directory)


def prepared_tokenizer(directory):
    """Real core tokenization/template files before CLI identity capture."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "user": 1, "stop": 2, "assistant": 3, "word": 4}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(["[UNK]", "stop"])
    tokenizer.save(str(directory / "tokenizer.json"))
    template = "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": template, "eos_token": "stop", "unk_token": "[UNK]"}))
    config = directory / "config.json"
    if not config.exists():
        config.write_text(json.dumps({"model_type": "fixture", "vocab_size": len(vocab), "eos_token_id": 2}))
    return PreparedTokenizer(tokenizer, template)


def cuda_cli_engine(**members):
    """A prepared engine publishes reply EOS independently of listener tests."""
    return SimpleNamespace(**{"eos": (2,), **members})


def cuda_cli_app(*, effective_context_window, **members):
    """Publish a concrete prepared template identity, without reading a model."""
    source = "{{ fixture }}"

    def provenance():
        return {"name": "default", "source": "cli_server_fixture", "sha256": sha256(source.encode()).hexdigest()}

    return SimpleNamespace(**{"effective_context_window": effective_context_window,
                             "template": SimpleNamespace(provenance=provenance), **members})


def install_cli_server(monkeypatch, request, module, *, on_serve=None):
    owners, served = [], []

    class Server:
        def __init__(self, address, handler, *, max_connections=None):
            self.address, self.handler = address, handler
            self.max_connections = max_connections
            self.stopping = threading.Event()
            self.close_calls = 0
            owners.append(self)

        def server_close(self):
            self.close_calls += 1
            self.stopping.set()

    def serve(app, host, port, *, server):
        assert any(server is owner for owner in owners)
        assert server.address == (host, port)
        assert server.close_calls == 0 and not server.stopping.is_set()
        assert server.max_connections is None or type(server.max_connections) is int
        served.append(server)
        if on_serve is not None:
            on_serve(app)

    def retired():
        assert len(owners) == len(served)
        assert all(owner is callback_owner for owner, callback_owner in zip(owners, served, strict=True))
        assert all(owner.close_calls == 1 and owner.stopping.is_set() for owner in owners)

    monkeypatch.setattr(module, "Server", Server)
    monkeypatch.setattr(module, "serve", serve)
    request.addfinalizer(retired)
