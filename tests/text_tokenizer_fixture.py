"""A character tokenizer stand-in for CUDA App tests (shared by the server, policy, disconnect and format tests)."""

from types import SimpleNamespace


class TextTokenizer:
    """The tokenizers.Tokenizer members the CUDA App uses: token t is chr(t). ``token_to_id`` maps a one-character
    token to its id and any other token (Gemma's "<channel|>", Qwen's "</think>", "<tool_call>") to None, as a real
    tokenizer answers for a token outside its vocabulary: the App then serves these tests as a model without
    reasoning or call markers."""

    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids, **kwargs):
        return "".join(chr(c) for c in ids)

    def token_to_id(self, token):
        return ord(token) if len(token) == 1 else None
