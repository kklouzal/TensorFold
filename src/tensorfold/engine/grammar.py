"""Structured output for every engine: a request's grammar as per-row token masks on each drafted verify window."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tensorfold.server.errors import RequestError

EXTRA = "tensorfold[grammar]"             # the optional dependency that brings xgrammar
KINDS = ("json", "json_schema", "regex", "choice", "grammar", "tools")   # tools: strict calls (server.strict_tools)
CACHE_BYTES = 256 << 20                   # compiled grammars kept for repeated schemas
OBJECT = '{"type": "object"}'             # json_object: any JSON object (OpenAI's contract), not an array
MAX_SCHEMA_NODES = 2000                    # native recursive schema compiler: quarter of measured stack-crash size
MAX_GRAMMAR_BYTES = 96 << 20               # the rank/HTTP source envelope
BLANKS = 32                               # the most blank characters between JSON tokens: pretty-printing, not endless


@dataclass(frozen=True)
class Spec:
    """A request's structured output: ``kind`` json, json_schema, regex, choice, grammar (EBNF) or tools (an xgrammar
    structural tag for strict function calls), and its text."""

    kind: str
    text: str = ""
    field: str = "response_format"        # the request field it came from, for error messages


class GrammarError(RuntimeError):
    """A reply's grammar failed: that request ends with this error, the server and other requests go on."""


def _schema_text(value: Any, where: str) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except RecursionError:
            raise RequestError(f"{where}: schema JSON nests too deeply") from None
        except json.JSONDecodeError as exc:
            raise RequestError(f"{where} is not valid JSON: {exc.msg}") from None
    if isinstance(value, bool):
        value = {} if value else None
    if not isinstance(value, dict):
        raise RequestError(f"{where} must be a JSON schema object")
    _schema_resources(value, where)
    try:
        return json.dumps(value, ensure_ascii=False)
    except (RecursionError, ValueError) as exc:
        raise RequestError(f"{where}: schema data cannot be encoded: {_message(exc)}") from None


def _schema_resources(schema: Any, where: str) -> None:
    """Bound native schema recursion before crossing into xgrammar, including chains hidden in definitions.

    The 2000-node envelope is shared with strict JSON parameters, based on the pinned native compiler's measured
    8000-node stack crash. Annotation/enum values are data, not schema graph edges. Local refs add no new nodes.
    """

    stack, seen = [schema], set()
    while stack:
        node = stack.pop()
        if not isinstance(node, dict) or id(node) in seen:
            continue
        seen.add(id(node))
        if len(seen) > MAX_SCHEMA_NODES:
            raise RequestError(f"{where}: schemas are limited to {MAX_SCHEMA_NODES} subschemas")
        pattern = node.get("pattern")
        if isinstance(pattern, str):
            _regex_resources(pattern, where)
        for keyword in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
            children = node.get(keyword)
            if isinstance(children, dict):
                stack.extend(children.values())
        for keyword in ("items", "additionalProperties", "additionalItems", "contains", "not", "if", "then", "else",
                        "propertyNames", "unevaluatedProperties", "unevaluatedItems"):
            child = node.get(keyword)
            stack.extend(child if isinstance(child, list) else [child])
        for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
            children = node.get(keyword)
            if isinstance(children, list):
                stack.extend(children)


def _regex_resources(pattern: str, where: str) -> None:
    """Reject recursive regex parser exhaustion before the pinned C++ parser can overflow its process stack.

    Use the non-caching parser already required by strict pattern validation on supported CPython 3.11+. Public
    re.compile would retain untrusted compiled patterns in its count-only process-global cache; re.purge would affect
    unrelated callers. Syntax differences are deliberately forwarded to the native parser unchanged.
    """

    from re import _parser

    try:
        _parser.parse(pattern)
    except RecursionError:
        raise RequestError(f"{where}: the pattern nests its groups too deeply") from None
    except (re.error, OverflowError):
        pass                                     # native parser owns syntax/feature diagnostics for admitted patterns


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RequestError(f"{where} must be a non-empty string")
    return value


def _choices(value: Any, where: str) -> str:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise RequestError(f"{where} must be a non-empty list of non-empty strings")
    return json.dumps(value, ensure_ascii=False)


def request_spec(body: dict[str, Any]) -> Spec | None:
    """The body's structured-output request, None for plain text; RequestError (HTTP 400) when it is malformed."""

    rf = body.get("response_format")
    if rf is not None:
        if not isinstance(rf, dict):
            raise RequestError('response_format must be an object such as {"type": "json_object"}')
        kind = rf.get("type")
        if kind == "json_object":
            return Spec("json")
        if kind == "json_schema":
            js = rf.get("json_schema")
            if not isinstance(js, dict) or js.get("schema") is None:
                raise RequestError("response_format json_schema needs json_schema.schema (a JSON schema object)")
            text = _schema_text(js["schema"], "response_format json_schema.schema")
            strict = js.get("strict")
            if strict is not None and not isinstance(strict, bool):
                raise RequestError("response_format json_schema.strict must be a boolean or null")
            if strict is True:
                from tensorfold.server.strict_tools import response_schema_spec

                return response_schema_spec(json.loads(text), "response_format json_schema.schema")
            return Spec("json_schema", text)
        if kind != "text":
            raise RequestError(f"response_format type must be text, json_object or json_schema, not {kind!r}")
    guided = {"guided_json": ("json_schema", _schema_text), "guided_regex": ("regex", _text),
              "guided_choice": ("choice", _choices), "guided_grammar": ("grammar", _text)}
    for name, (kind, read) in guided.items():
        if body.get(name) is not None:
            return Spec(kind, read(body[name], name), name)
    so = body.get("structured_outputs")
    if so is not None:
        if not isinstance(so, dict):
            raise RequestError('structured_outputs must be an object such as {"json": {...}}')
        given = sorted(k for k, v in so.items() if v is not None and v is not False)
        known = {"json": ("json_schema", _schema_text), "regex": ("regex", _text), "choice": ("choice", _choices),
                 "grammar": ("grammar", _text)}
        other = [k for k in given if k not in known and k != "json_object"]
        if other:
            raise RequestError(f"structured_outputs {', '.join(other)} is not supported: use json, json_object, "
                               "regex, choice or grammar")
        for name in given:
            if name in known:
                kind, read = known[name]
                return Spec(kind, read(so[name], f"structured_outputs.{name}"), "structured_outputs")
        if so.get("json_object"):
            return Spec("json", field="structured_outputs")
    return None


def _message(exc: Exception) -> str:
    """xgrammar's error without its timestamp and source location."""

    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return re.sub(r"^\[[^\]]*\]\s*\S+:\d+:\s*", "", text)


class Grammars:
    """One tokenizer's grammar compiler (xgrammar), with compiled grammars cached by their text."""

    def __init__(self, info, *, decode=None) -> None:
        import xgrammar as xgr

        self.xgr = xgr
        self.info = info
        self.decode = decode  # original tokenizer decode, retained by this prepared compiler
        self.vocab_size = int(info.vocab_size)
        self.compiler = xgr.GrammarCompiler(info, max_threads=8, cache_limit_bytes=CACHE_BYTES)
        self.lock = threading.Lock()              # requests compile on their own HTTP threads

    @classmethod
    def for_model(cls, model_dir: str | Path, vocab_size: int, stop_ids: Sequence[int]) -> "Grammars":
        """From the served ``tokenizer.json`` (``--tokenizer``'s, else the checkpoint's): ``vocab_size`` is the logits'
        width, ``stop_ids`` its eos ids."""

        try:
            import xgrammar as xgr
            from transformers import PreTrainedTokenizerFast       # xgrammar depends on transformers
        except ImportError:
            raise RequestError(f"structured output needs xgrammar on the server: pip install '{EXTRA}'") from None
        from tensorfold.token_metadata import tokenizer_override

        override = tokenizer_override()          # --tokenizer: the text admitted at startup, never the file again
        if override is not None:
            from tokenizers import Tokenizer

            tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(override.text))
        else:
            tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(model_dir) / "tokenizer.json"))
        def decode(ids, *, skip_special_tokens=False):
            # Rank-local structural context must use actual decoded bytes, not
            # HF's optional punctuation/whitespace cleanup. This prepared owner
            # retains its original tokenizer; no metadata is reread per token.
            return tok.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)

        return cls(xgr.TokenizerInfo.from_huggingface(tok, vocab_size=int(vocab_size), stop_token_ids=list(stop_ids)),
                   decode=decode)

    def compile(self, spec: Spec):
        """The compiled grammar, or RequestError naming what the request's grammar gets wrong."""

        c = self.compiler
        try:
            if not isinstance(spec.text, str) or len(spec.text.encode("utf-8")) > MAX_GRAMMAR_BYTES:
                raise RequestError(f"{spec.field}: grammar source exceeds the 96 MiB envelope")
            if spec.kind == "json_schema":
                _schema_resources(json.loads(spec.text), spec.field)
            elif spec.kind == "regex":
                _regex_resources(spec.text, spec.field)
            with self.lock:
                if spec.kind in ("json", "json_schema"):
                    schema = OBJECT if spec.kind == "json" else spec.text
                    return c.compile_json_schema(schema, max_whitespace_cnt=BLANKS)
                if spec.kind == "regex":
                    return c.compile_regex(spec.text)
                if spec.kind == "choice":
                    options = " | ".join(json.dumps(v, ensure_ascii=False) for v in json.loads(spec.text))
                    return c.compile_grammar("root ::= " + options)
                if spec.kind == "tools":
                    return c.compile_structural_tag(spec.text)
                return c.compile_grammar(spec.text)
        except RequestError:
            raise
        except (RuntimeError, ValueError, TypeError, RecursionError) as exc:
            raise RequestError(f"{spec.field}: the grammar cannot be enforced: {_message(exc)}") from None

    def accepts(self, compiled, text: str) -> bool:
        """Whether ``text`` is a complete reply of the compiled grammar (a fresh matcher reads it whole)."""

        m = self.xgr.GrammarMatcher(compiled)
        return bool(m.accept_string(text)) and bool(m.is_completed())

    def constraint(self, compiled, *, think_end: int | None = None, spec: Spec | None = None,
                   answer_start: int | None = None, answer_opener: str | None = None,
                   answer_headers=(), decode=None, think_starts=(), thought_open=False, stop_ids=()) -> "Constraint":
        """A fresh reply's grammar state; with ``think_end`` (thinking on) it starts at the token after that one."""

        c = Constraint(self.xgr, compiled, self.vocab_size, think_end=think_end, answer_start=answer_start,
                       answer_opener=answer_opener, answer_headers=answer_headers,
                       decode=self.decode if decode is None else decode, think_starts=think_starts, thought_open=thought_open,
                       stop_ids=stop_ids)
        c.spec = spec
        return c

    def follow(self, values: Sequence[int]) -> "Constraint | None":
        """The constraint another rank ``pack``ed, compiled here: both ranks then walk and mask the same rows."""

        if not values:
            return None
        if int(values[0]) == -2:
            kind, end, answer, meta, text = activation_wire(values, self.vocab_size)
            spec = Spec(KINDS[kind], text)
            return self.constraint(self.compile(spec), think_end=end-1 if end else None, spec=spec,
                                   answer_start=answer-1 if answer else None, answer_opener=meta["opener"],
                                   answer_headers=meta["headers"], think_starts=meta.get("starts", ()),
                                   thought_open=meta.get("opened", False), stop_ids=meta.get("stops", ()))
        if not 0 <= int(values[0]) < len(KINDS):
            raise GrammarError("unknown grammar wire version/kind")
        spec = Spec(KINDS[int(values[0])], bytes(int(v) for v in values[2:]).decode())
        return self.constraint(self.compile(spec), think_end=int(values[1]) - 1 if int(values[1]) else None, spec=spec)


def pack(constraint: "Constraint | None") -> list[int]:
    """A constraint's grammar for the other rank: [kind, think_end + 1 (0: none), the text's UTF-8 bytes], or []."""

    if constraint is None:
        return []
    spec, end = constraint.spec, constraint.think_end
    if constraint.answer_start is not None or constraint.think_starts or constraint.initial_thought_open:
        fields = {"opener": constraint.answer_opener, "headers": constraint.answer_headers}
        if constraint.think_starts or constraint.initial_thought_open:
            fields.update(starts=constraint.think_starts, opened=constraint.initial_thought_open,
                          stops=constraint.stop_ids)
        meta = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode()
        text = spec.text.encode()
        if len(meta) + len(text) > MAX_GRAMMAR_BYTES:
            raise GrammarError("grammar activation metadata/source exceeds its 96 MiB wire envelope")
        return [-2, KINDS.index(spec.kind), 0 if end is None else int(end)+1,
                0 if constraint.answer_start is None else constraint.answer_start+1, len(meta), *meta, *text]
    return [KINDS.index(spec.kind), 0 if end is None else int(end) + 1, *spec.text.encode()]


def activation_wire(values, vocab):
    """One authority for the bounded offered-header rank schema, before compilation."""
    if (len(values) < 5 or len(values) > MAX_GRAMMAR_BYTES+5
            or any(type(v) is not int for v in values) or values[0] != -2):
        raise GrammarError("invalid grammar activation wire")
    _, kind, end, answer, size = values[:5]
    if (kind not in range(len(KINDS)) or not 0 <= end <= vocab or not 0 <= answer <= vocab
            or not 0 < size <= MAX_GRAMMAR_BYTES or 5+size > len(values)
            or any(not 0 <= v <= 255 for v in values[5:])):
        raise GrammarError("invalid grammar activation fields")
    try:
        meta = json.loads(bytes(values[5:5+size]).decode())
        text = bytes(values[5+size:]).decode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise GrammarError("invalid grammar activation UTF8/JSON") from exc
    if not isinstance(meta, dict) or set(meta) not in ({"opener", "headers"}, {"opener", "headers", "starts", "opened", "stops"}):
        raise GrammarError("invalid offered-header activation metadata")
    if not isinstance(meta["headers"], list):
        raise GrammarError("invalid offered headers")
    if answer:
        if (not isinstance(meta["opener"], str) or not meta["opener"] or len(meta["opener"].encode()) > 256
                or not meta["headers"] or any(not isinstance(h, str) or not h.startswith(meta["opener"])
                or len(h) <= len(meta["opener"]) for h in meta["headers"])):
            raise GrammarError("invalid offered-header activation metadata")
    elif meta["opener"] is not None or meta["headers"]:
        raise GrammarError("native thought-only metadata cannot declare a tool header")
    if "starts" in meta and (not isinstance(meta["starts"], list) or type(meta["opened"]) is not bool
            or not isinstance(meta["stops"], list) or any(type(t) is not int or not 0 <= t < vocab
                                                       for t in [*meta["starts"], *meta["stops"]])):
        raise GrammarError("invalid native thought phase metadata")
    return kind, end, answer, meta, text


_MODELS: dict[tuple[str, int, tuple[int, ...]], Grammars] = {}
_BUILD = threading.Lock()


def for_model(model_dir: str | Path, vocab_size: int, stop_ids: Sequence[int]) -> Grammars:
    """``Grammars.for_model``, built once per checkpoint (the first structured request pays a second or two)."""

    key = (str(Path(model_dir).resolve()), int(vocab_size), tuple(int(t) for t in stop_ids))
    with _BUILD:
        found = _MODELS.get(key)
        if found is None:
            found = _MODELS[key] = Grammars.for_model(model_dir, vocab_size, stop_ids)
        return found


def vocab_size(model_dir: str | Path) -> int | None:
    """The checkpoint's logits width from ``config.json`` (``text_config`` first), or None."""

    path = Path(model_dir) / "config.json"
    if not path.is_file():
        return None
    config = json.loads(path.read_text())
    for part in (config.get("text_config") or {}, config):
        if isinstance(part.get("vocab_size"), int):
            return int(part["vocab_size"])
    return None


def compiler(owner, model_dir: str | Path, stop_ids: Sequence[int]) -> Grammars:
    """``owner``'s grammar compiler (kept on it as ``grammars``), built from its checkpoint on first use."""

    found = getattr(owner, "grammars", None)
    if found is None:
        vocab = vocab_size(model_dir) if model_dir is not None else None
        if vocab is None:
            raise RequestError("structured output needs the checkpoint's config.json vocab_size")
        found = owner.grammars = for_model(model_dir, vocab, tuple(stop_ids))
    return found


def check_stopping(fields: dict[str, Any], spec: Spec | None) -> None:
    """A grammar's EOS must end generation: suppressing it would release its mask after termination."""

    if spec is not None and fields.get("ignore_eos") is True:
        raise RequestError(f"ignore_eos true cannot be combined with {spec.field}: structured output must stop at "
                           "its grammar's end-of-sequence token")


def request_constraint(owner, fields: dict[str, Any], think_end: int | None, spec: Spec | None = None
                       ) -> "Constraint | None":
    """A fresh constraint for a request's grammar fields, or ``spec`` (strict tools' calls), or None (``owner`` has
    ``model_dir`` and ``stop_ids``)."""

    spec = spec if spec is not None else request_spec(fields)
    check_stopping(fields, spec)
    if spec is None:
        return None
    grammars = compiler(owner, owner.model_dir, owner.stop_ids)
    return grammars.constraint(grammars.compile(spec), think_end=think_end, spec=spec)


SPECS = ("response_format", "guided_json", "guided_regex", "guided_choice", "guided_grammar", "structured_outputs")
# every request field a reply's grammar reads (the HTTP layer hands these to the engine app): the specs above, and
# strict tools' parallel_tool_calls (one call or many; the tools themselves travel on their own)
FIELDS = (*SPECS, "parallel_tool_calls")


def refusal(body: dict[str, Any], owner=None) -> str | None:
    """Why the body's grammar cannot run (HTTP 400): malformed, beside a required call, or (``owner``) not compiling;
    an ``owner`` with ``strict_calls`` (the Mac's chat app) also checks the request's strict tools."""

    from tensorfold.server.tools import active_tool_specs, tool_choice_requires_call

    try:
        spec = request_spec(body)
        required = tool_choice_requires_call(body.get("tool_choice"))
        if spec is not None and body.get("tools") and required:
            return f'{spec.field} cannot be combined with tool_choice "required" or a named function: send one'
        if owner is not None:
            from tensorfold.server import strict_tools

            try:
                tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
            except ValueError:             # malformed tools: the chat route refuses them itself; others ignore them
                tools = []
            if hasattr(owner, "strict_calls"):
                plan = owner.strict_calls(tools, required=required,
                                          parallel=body.get("parallel_tool_calls") is not False, beside=spec,
                                          offered=body.get("tools"))
                spec = plan.spec if plan is not None else spec
            elif strict_tools.wanted(tools):
                return "this server does not enforce strict function tools: send strict false"
        check_stopping(body, spec)
        if spec is not None and getattr(owner, "model_dir", None) is not None:
            compiler(owner, owner.model_dir, owner.stop_ids).compile(spec)   # cached for the request's own compile
    except RequestError as exc:
        return str(exc)
    return None


@dataclass
class Window:
    """A verify window the grammar keeps: its rows, and the allowed-token bits of the rows it constrains."""

    tokens: list[int]
    parents: list[int]
    rows: list[int] = field(default_factory=list)     # constrained rows, in window order
    bits: Any = None                                  # [len(rows), words] int32 numpy (xgrammar's bitmask)


@dataclass(frozen=True)
class _BoundaryStep:
    active: bool
    seed: tuple[int, ...] = ()
    start: int | None = None
    pending: int | None = None
    opened: bool = False


class AnswerBoundary:
    """Thinking/answer intent shared by producer matching and published-ID observation.

    A bare opener is ambiguous reasoning. Only the selected offered-name header
    confirms intent, with its bytes withheld while tentative. Header prefixes
    are validated with the selected native grammar before any role promotion.
    """
    def __init__(self, vocab, *, think_end=None, answer_start=None, answer_opener=None,
                 answer_headers=(), decode=None, prefix_valid=None, think_starts=(), thought_open=False):
        self.vocab = None if vocab is None else int(vocab)
        if type(thought_open) is not bool or any(type(t) is not int or t < 0
                or self.vocab is not None and t >= self.vocab for t in think_starts):
            raise GrammarError("invalid native thought opener metadata")
        self.think_starts, self.initial_thought_open = tuple(think_starts), thought_open
        self.opened_thought = thought_open
        if answer_start is not None and (type(answer_start) is not int or not 0 <= answer_start or self.vocab is not None and answer_start >= self.vocab
                                         or answer_start == think_end):
            raise GrammarError("invalid grammar answer activation token")
        if answer_start is not None and (not isinstance(answer_opener, str) or not answer_opener
                or len(answer_opener.encode()) > 256 or not callable(decode)
                or not isinstance(answer_headers, (tuple, list)) or not answer_headers
                or any(not isinstance(h, str) or not h.startswith(answer_opener)
                       or len(h) <= len(answer_opener) for h in answer_headers)):
            raise GrammarError("structural activation requires actual offered headers and original tokenizer decoder")
        if decode is not None and not callable(decode):
            raise GrammarError("grammar boundary decoder must be callable")
        self.answer_opener, self.decode = answer_opener, decode
        self.answer_headers = tuple(answer_headers)
        self._prefix_valid = prefix_valid
        self.think_closer = (decode([think_end], skip_special_tokens=False)
                             if think_end is not None and decode is not None else None)
        if think_end is not None and decode is not None and (not isinstance(self.think_closer, str)
                                                             or not self.think_closer):
            raise GrammarError("grammar boundary decoder must preserve its native close marker")
        self._thinking_tokens = []  # references bounded by the request's reply allowance
        self.think_end, self.answer_start = think_end, answer_start
        self.active = think_end is None
        self.observed = 0
        self.answer_offset = 0 if self.active else None
        self.pending_answer_offset = self.thinking_close_offset = None

    def _native_start(self, tokens, char_start):
        # Intent begins at the selected native opener, not a quoted spelling or
        # a byte slice synthesized into a token. Header continuation may use any
        # valid token segmentation; all actual seed tokens remain unchanged.
        for index in range(len(tokens)-1, -1, -1):
            if tokens[index] == self.answer_start and len(self.decode(tokens[:index], skip_special_tokens=False)) == char_start:
                return index
        return None

    def _boundary_step(self, token, active, path=(), opened=None):
        opened = self.opened_thought if opened is None else opened
        if active:
            return _BoundaryStep(True, (token,))
        if token == self.think_end:
            impossible_direct = (self.think_starts and self._prefix_valid is not None
                and not self._prefix_valid([*self._thinking_tokens, *path]))
            if opened or impossible_direct:
                # Real native close controls reasoning, not Markdown/prose
                # quoting. Only a confirmed native tool payload stays opaque.
                from tensorfold.markup_lex import markup_regions
                text = self.decode([*self._thinking_tokens, *path, token], skip_special_tokens=False)
                point = len(text) - len(self.think_closer)
                if self.think_closer == "<channel|>":
                    # Gemma channel controls are literal only inside actual
                    # quoted tool values, independent of permitted call names.
                    # An unquoted close in malformed call markup is structural.
                    for region in markup_regions(text):
                        if region.kind != "call" or not region.start <= point < region.end:
                            continue
                        body = region.payload_start
                        if any(value.kind == "string" and body + value.start <= point < body + value.end
                               for value in markup_regions(text[body:region.payload_end])):
                            return _BoundaryStep(False)
                    return _BoundaryStep(True)
                if not any(r.kind == "call" and r.start <= point < r.end
                           for r in markup_regions(text, call_headers=self.answer_headers)):
                    return _BoundaryStep(True)
            if self.decode is None:
                return _BoundaryStep(True)
            from tensorfold.markup_lex import structural_marker
            text = self.decode([*self._thinking_tokens, *path, token], skip_special_tokens=False)
            if structural_marker(text, self.think_closer, call_headers=self.answer_headers):
                return _BoundaryStep(True)
        if token in self.think_starts and self.decode is not None:
            from tensorfold.markup_lex import structural_marker
            text = self.decode([*self._thinking_tokens, *path, token], skip_special_tokens=False)
            marker = self.decode([token], skip_special_tokens=False)
            if structural_marker(text, marker, call_headers=self.answer_headers):
                return _BoundaryStep(False, opened=True)
        if self.answer_start is not None and (not self.think_starts or opened) and (token == self.answer_start or self.pending_answer_offset is not None
                                              or self.answer_start in path):
            from tensorfold.markup_lex import structural_call_header, pending_call_header
            tokens = [*self._thinking_tokens, *path, token]
            text = self.decode(tokens, skip_special_tokens=False)
            confirmed = structural_call_header(text, self.answer_opener, self.answer_headers)
            if confirmed is not None:
                start = self._native_start(tokens, confirmed[0])
                if start is not None:
                    return _BoundaryStep(True, tuple(tokens[start:]), start)
            pending = pending_call_header(text, self.answer_opener, self.answer_headers)
            if pending is not None:
                return _BoundaryStep(False, pending=self._native_start(tokens, pending))
        return _BoundaryStep(False)

    def _commit_boundary(self, token, step):
        if step.opened:
            self.opened_thought = True
        if step.active and not self.active:
            self.thinking_close_offset = self.observed if token == self.think_end and step.start is None else None
            self.answer_offset = (step.start if step.start is not None else
                                  self.observed + int(token == self.think_end))
        if step.active:
            self._thinking_tokens.clear()
        elif self.decode is not None:
            self._thinking_tokens.append(token)
        self.pending_answer_offset = None if step.active else step.pending
        self.active = step.active
        self.observed += 1

    def advance(self, tokens):
        for value in tokens:
            token = int(value)
            step = self._boundary_step(token, self.active)
            if step.start is not None:
                if self._prefix_valid is None:
                    raise GrammarError("header observer requires the selected native prefix validator")
                if not self._prefix_valid(step.seed):
                    step = _BoundaryStep(False)  # intra-token invalid payload is still uncommitted intent
            self._commit_boundary(token, step)

    def confirmation_index(self, tokens):
        if self.active:
            return 0
        path = []
        opened = self.opened_thought
        for index, value in enumerate(tokens):
            token = int(value)
            step = self._boundary_step(token, False, path, opened)
            if step.active and (step.start is None or self._prefix_valid is not None and self._prefix_valid(step.seed)):
                return index
            opened = opened or step.opened
            path.append(token)
        return None

    def budget_relevant(self, tokens):
        if not self.think_starts or self.opened_thought:
            return True
        path = []
        for value in tokens:
            token = int(value)
            step = self._boundary_step(token, False, path)
            if step.opened:
                return True
            path.append(token)
        # A viable direct answer is not invented thinking. An impossible direct
        # prefix can only complete through the real-close alternative.
        return self._prefix_valid is not None and not self._prefix_valid([*self._thinking_tokens, *path])

    def force_think_close(self, token):
        if self.active or int(token) != self.think_end:
            raise GrammarError("forced thinking close does not match its request state")
        self._commit_boundary(int(token), _BoundaryStep(True))



class Constraint(AnswerBoundary):
    """One reply's grammar at its chosen tokens: keeps and masks verify rows, follows the chosen tokens.

    The generation owner stops at the grammar's EOS; request admission rejects ignore_eos for active constraints.
    """

    def __init__(self, xgr, compiled, vocab_size: int, *, think_end: int | None = None,
                 answer_start: int | None = None, answer_opener: str | None = None,
                 answer_headers=(), decode=None, think_starts=(), thought_open=False, stop_ids=()) -> None:
        self.xgr, self.compiled = xgr, compiled
        self.m = xgr.GrammarMatcher(compiled)
        super().__init__(vocab_size, think_end=think_end, answer_start=answer_start,
                         answer_opener=answer_opener, answer_headers=answer_headers, decode=decode,
                         think_starts=think_starts, thought_open=thought_open)
        self.stop_ids = tuple(stop_ids)
        self.direct_m = (xgr.GrammarMatcher(compiled) if think_starts and not thought_open and think_end is not None else None)
        self.direct_valid = self.direct_m is not None
        if self.direct_m is not None and (not self.stop_ids or any(type(t) is not int or not 0 <= t < self.vocab for t in self.stop_ids)):
            raise GrammarError("native composite grammar requires the actual engine stop IDs")
        self.words = (self.vocab + 31) // 32
        self.spec: Spec | None = None                 # what ``pack`` sends another rank
        self._shifts: dict[Any, Any] = {}
        self._pinned: Any = None                      # host staging for the bits a GPU mask reads
        self._landed: Any = None                      # the event after the staging buffer's last copy

    def fresh(self) -> "Constraint":
        """This grammar from its start again, for a reply decoded again from its prompt (a yielded stream's replay)."""

        again = Constraint(self.xgr, self.compiled, self.vocab, think_end=self.think_end,
                           answer_start=self.answer_start, answer_opener=self.answer_opener,
                           answer_headers=self.answer_headers, decode=self.decode, think_starts=self.think_starts,
                           thought_open=self.initial_thought_open, stop_ids=self.stop_ids)
        again.spec = self.spec
        return again

    def fresh_boundary(self):
        xgr, compiled = self.xgr, self.compiled
        def prefix_valid(tokens):
            matcher = xgr.GrammarMatcher(compiled)
            return all(matcher.accept_token(int(token)) for token in tokens)
        return AnswerBoundary(self.vocab, think_end=self.think_end, answer_start=self.answer_start,
                              answer_opener=self.answer_opener, answer_headers=self.answer_headers,
                              decode=self.decode, prefix_valid=prefix_valid, think_starts=self.think_starts,
                              thought_open=self.initial_thought_open)

    def retire_device_staging(self) -> None:
        """Retire disposable mask/DMA storage on its serialized model coordinator.

        The native CPU matcher and its accepted-token state remain owned by
        this exact constraint. A retained request must call at a quiescent
        boundary before its CPU continuation can outlive the GPU slot.
        """
        import torch

        if self._landed is not None:
            self._landed.synchronize()
        for device in self._shifts:
            if torch.device(device).type == "cuda":
                completed = torch.cuda.Event()
                completed.record(torch.cuda.current_stream(device))
                completed.synchronize()
        self._shifts.clear()
        self._pinned = self._landed = None

    @property
    def finished(self) -> bool:
        """The grammar has taken its stop token: the reply is complete."""

        return (self.active and self.m.is_terminated() or
                self.direct_valid and self.direct_m is not None and self.direct_m.is_terminated())

    def window(self, tokens: Sequence[int], parents: Sequence[int]) -> Window:
        """The rows an accepted path can use (row 0 the pending token, parents first), each constrained row's bits."""

        try:
            return self._window(list(tokens), list(parents))
        except GrammarError:
            raise
        except Exception as exc:                      # noqa: BLE001  (xgrammar's failure ends this reply only)
            raise GrammarError(f"the reply's grammar failed: {_message(exc)}") from None

    def _window(self, tokens: list[int], parents: list[int]) -> Window:
        import numpy as np                            # xgrammar fills any DLPack array: one bitmask for torch and MLX

        if self.finished or (not self.active and self.direct_m is None
                             and self.pending_answer_offset is None
                             and self.think_end not in tokens[1:] and self.answer_start not in tokens[1:]):
            return Window(tokens, parents)            # the reply has ended, or no row reaches the grammar
        n = len(tokens)
        children: list[list[int]] = [[] for _ in range(n)]
        for r in range(1, n):
            children[parents[r]].append(r)
        bits = np.full((n, self.words), -1, dtype=np.int32)
        kept = [True] + [False] * (n - 1)
        filled: list[int] = []
        m = self.m

        path = []

        def visit(r: int, active: bool, direct_valid: bool, opened: bool) -> None:
            if active:
                m.fill_next_token_bitmask(bits, r)
                filled.append(r)
            elif self.direct_m is not None and not opened and not (direct_valid and self.direct_m.is_completed()):
                # Union language: free prefix can still reach REAL close+schema,
                # but EOS cannot certify invalid direct output as a success.
                unsigned = bits.view(np.uint32)
                for eos in self.stop_ids:
                    unsigned[r, eos // 32] &= np.uint32(0xffffffff ^ (1 << (eos % 32)))
                filled.append(r)
            for c in children[r]:
                step, consumed, valid, next_direct, direct_count = self._transition(tokens[c], active, path, direct_valid, opened)
                if valid:
                    ended = (step.active and m.is_terminated()) or (self.direct_m is not None and tokens[c] in self.stop_ids)
                    if not ended:
                        kept[c] = True
                        if not step.active:
                            path.append(tokens[c])
                        visit(c, step.active, next_direct, opened or step.opened)
                        if not step.active:
                            path.pop()
                    if consumed:
                        m.rollback(consumed)
                    if direct_count:
                        self.direct_m.rollback(direct_count)

        visit(0, self.active, self.direct_valid, self.opened_thought)
        index = [r for r in range(n) if kept[r]]
        new = {r: i for i, r in enumerate(index)}
        window = Window([tokens[r] for r in index], [-1] + [new[parents[r]] for r in index[1:]])
        if filled:
            filled.sort()
            window.rows = [new[r] for r in filled]
            window.bits = bits if len(filled) == n else bits[filled]
        return window

    def mask(self, logits, window: Window | None = None, offset: int = 0):
        """``logits`` (columns from token ``offset``), constrained rows' other tokens -inf: torch in place, MLX new."""

        if window is None:                            # one row: the token after the chosen ones
            window = self.window([0], [-1])
        if not window.rows:
            return logits
        if type(logits).__module__.startswith("mlx"):
            return _mask_mlx(logits, window, offset)
        return self._mask_torch(logits, window, offset)

    def allowed(self, window: Window, width: int, device, offset: int = 0):
        """The constrained rows' allowed columns [len(rows), width] (bool, on ``device``) from token ``offset`` on."""

        import torch

        shifts = self._shifts.get(device)
        if shifts is None:
            shifts = self._shifts[device] = torch.arange(8, dtype=torch.uint8, device=device)
        # token t is bit t % 8 of byte t // 8 (xgrammar's int32 words, little-endian)
        packed = self._staged(window.bits, device).view(torch.uint8)
        allowed = ((packed.unsqueeze(-1) >> shifts) & 1).view(len(window.rows), -1)[:, offset:offset + width].bool()
        if allowed.shape[1] < width:                  # logits past the grammar's vocabulary: never allowed
            allowed = torch.nn.functional.pad(allowed, (0, width - allowed.shape[1]), value=False)
        return allowed

    def _staged(self, bits, device):
        """``bits`` on ``device``; to a GPU through a pinned buffer, copied in stream order without the host waiting for
        the work queued before it (the forward), and rewritten only once its previous copy has landed."""

        import torch

        if torch.device(device).type != "cuda":
            return torch.from_numpy(bits).to(device)
        rows, words = bits.shape
        if self._pinned is None or self._pinned.shape[0] < rows or self._pinned.shape[1] != words:
            self._pinned = torch.empty((max(rows, 8), words), dtype=torch.int32, pin_memory=True)
        elif self._landed is not None:
            self._landed.synchronize()
        staged = self._pinned[:rows]
        staged.numpy()[:] = bits
        out = staged.to(device, non_blocking=True)
        self._landed = torch.cuda.Event()
        self._landed.record()
        return out

    def _mask_torch(self, logits, window: Window, offset: int = 0):
        import torch

        dev, width = logits.device, logits.shape[-1]
        allowed = self.allowed(window, width, dev, offset)
        rows = logits.view(-1, width)
        if window.rows == list(range(rows.shape[0])):
            rows.masked_fill_(~allowed, float("-inf"))
        else:
            index = torch.tensor(window.rows, device=dev)
            rows[index] = rows[index].masked_fill(~allowed, float("-inf"))
        return logits

    def _transition(self, token: int, active: bool, path=(), direct_valid=None, opened=None):
        direct_valid = self.direct_valid if direct_valid is None else direct_valid
        opened = self.opened_thought if opened is None else opened
        step = self._boundary_step(token, active, path, opened)
        if not active and self.direct_m is not None and token == self.think_end and not direct_valid:
            step = _BoundaryStep(True)
        consumed = 0
        for value in step.seed:
            if not self.m.accept_token(value):
                if consumed:
                    self.m.rollback(consumed)
                if step.start is not None:
                    return _BoundaryStep(False), 0, True, False, 0
                return step, 0, False, direct_valid, 0
            consumed += 1
        direct_count = 0
        next_direct = direct_valid and not step.active and not step.opened
        if next_direct and self.direct_m is not None:
            next_direct = bool(self.direct_m.accept_token(token))
            direct_count = int(next_direct)
        return step, consumed, True, next_direct, direct_count

    def advance(self, tokens: Sequence[int]) -> None:
        """Follow actual chosen tokens; free thinking ends only by close or the declared tool opener."""
        try:
            for token in tokens:
                if self.finished:
                    return
                step, _, valid, next_direct, _ = self._transition(int(token), self.active)
                if not valid:
                    raise GrammarError(f"the reply's grammar rejected chosen token {int(token)}")
                self.direct_valid = next_direct
                self._commit_boundary(int(token), step)
        except GrammarError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GrammarError(f"the reply's grammar failed: {_message(exc)}") from None


def _mask_mlx(logits, window: Window, offset: int = 0):
    """The MLX form of ``Constraint.mask``: rows [R, V] (or [1, R, V]), the bits unpacked on the GPU."""

    import mlx.core as mx

    shape, width = logits.shape, logits.shape[-1]
    rows = logits.reshape(-1, width)
    words = mx.array(window.bits.view("uint32"))                          # [k, W]
    allowed = ((words[:, :, None] >> mx.arange(32, dtype=mx.uint32)) & 1).reshape(len(window.rows), -1)
    allowed = allowed[:, offset:offset + width]
    if allowed.shape[1] < width:                      # logits past the grammar's vocabulary: never allowed
        allowed = mx.pad(allowed, [(0, 0), (0, width - allowed.shape[1])])
    floor = mx.array(float("-inf"), dtype=rows.dtype)
    if window.rows == list(range(rows.shape[0])):
        rows = mx.where(allowed.astype(mx.bool_), rows, floor)
    else:
        index = mx.array(window.rows)
        rows[index] = mx.where(allowed.astype(mx.bool_), rows[index], floor)
    return rows.reshape(shape)


__all__ = ["Constraint", "EXTRA", "FIELDS", "GrammarError", "Grammars", "KINDS", "SPECS", "Spec", "Window", "compiler",
           "check_stopping", "for_model", "pack", "refusal", "request_constraint", "request_spec", "vocab_size"]
