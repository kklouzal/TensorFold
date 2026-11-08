"""OpenAI server for the CUDA engines: a family's ``cuda_engine`` gives ``eos``, ``generate`` and ``follow``."""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from tensorfold.engine import grammar
from tensorfold.server import strict_tools
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import CONTEXT_LIMIT, RequestError, refusal
from tensorfold.server.messages import validate_modalities
from tensorfold.server.probabilities import TokenBytes, probability_options
from tensorfold.server.request_ids import new_request_id
from tensorfold.server.request_options import (
    PENALTY_FIELDS, drop_penalties, fixed_thinking, heard_effort, parse_numbers, thinking_fields, thinking_budget,
)
from tensorfold.server.request_limits import RequestLimit, optional_limit
from tensorfold.server.segments import SegmentEncoder
from tensorfold.server.stopping import matched_stop, stop_options
from tensorfold.server import failure_capture
from tensorfold.server.thinking_notes import ended_thinking, ended_thinking_note, unanswered
from tensorfold.server.token_routes import flag, token_ids
from tensorfold.server.tool_policy import ToolCallPolicy, visible_tool_prose
from tensorfold.engine.call_gate import CallGate, ThinkBudget, call_format, generate_gated
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server.tools import (
    active_tool_specs, normalize_qwen_tool_markup, parse_tool_calls_from_content, tool_choice_requires_call, tool_calls_complete, tool_spec_name,
)

from tensorfold.cuda import health
from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.cuda.reply_text import StopStrings, StreamDecoder, hide_tool_calls, parse_tool_calls
from tensorfold.cuda.turns import Turns, Yield
from tensorfold.server.text import CHANNEL_MARKERS, THINK_MARKERS, is_title_request, reasoning_count, split_thinking
from tensorfold.vision.images import DEFAULT_LIMITS, ImageLimits


# -- requests --------------------------------------------------------------------------------

_MADE = threading.Lock()                # guards the lazily made per-app ``Turns``

_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seed")


@dataclass(slots=True, weakref_slot=True)
class PreparedRequest:
    prompt: list[int]
    max_tokens: int
    tools: list[dict[str, Any]]
    thinking: bool
    sampling: Any          # the engine's ``Sampling``, or None for greedy decoding
    ignore_eos: bool = False
    stop: tuple[str, ...] = ()
    vision: Any = None
    grammar: Any = None     # (spec, compiled grammar) of the request's response_format or strict tools, or None
    think_budget: int = 0   # reply tokens before the server closes a think block the reply leaves open (0: no limit)
    calls: Any = None       # strict tools' call grammar and reader (server.strict_tools.Plan), or None
    thought_open: bool = False   # the Gemma prompt ends inside the thought channel it opened: the reply begins in it
    penalties: Any = None   # active presence/frequency/repetition penalties (engine.penalties.Penalties), or None

    profile_binding: Any = None
    _profile_finalizer: Any = None
    _profile_use: Any = None
    _admission: Any = None
    _media_release: Any = None
    count_only: bool = False  # Validation/count preparation never authorizes model execution.

    def release_media(self):
        """Whole logical request owns replay across every generation subrun."""
        if self._media_release is not None and self._media_release(self.vision):
            self._media_release = None

    def close(self):
        """Retire an unused/consumed prepared model binding; safe to repeat."""
        self.release_media()
        if self._profile_finalizer is not None:
            from tensorfold.cleanup import finish

            operations = (self.profile_binding.close,) + (() if self._admission is None else (self._admission.close,))
            finish(operations)
            self._profile_finalizer.detach()


def applies_penalties(engine: Any) -> bool:
    """Whether ``engine`` applies presence/frequency/repetition penalties: a ``penalties`` keyword, not refused."""

    return ("penalties" in inspect.signature(engine.generate).parameters
            and bool(getattr(engine, "penalties_supported", True)))


def unapplied_penalties(engine: Any, sampling: dict[str, Any]) -> dict[str, Any]:
    """Remove the penalty defaults (flags, generation_config.json) of ``sampling`` that ``engine`` cannot apply and
    return the active ones, which the server names at startup as the Metal backend does: requests sending no
    penalties are then served, and those asking for an active penalty still get HTTP 400."""

    if not any(k in sampling for k in PENALTY_FIELDS) or applies_penalties(engine):
        return {}
    return drop_penalties(sampling)


def _native_context(model_dir: Path) -> int:
    path = model_dir / "config.json"
    if not path.exists():
        return 0
    config = json.loads(path.read_text())
    text = config.get("text_config") or config
    limit = text.get("max_position_embeddings") or config.get("max_position_embeddings")
    return int(limit) if isinstance(limit, int) and limit > 0 else 0


def _segment_encoder(tok, template) -> SegmentEncoder | None:
    """A message-at-a-time encoder when the chat template opens every message with one token the tokenizer splits its
    input on exactly (Qwen's ``<|im_start|>``), else None."""

    from tensorfold.cuda.markers import TemplateTokens
    from tensorfold.engine.prefill_plan import message_markers

    openers, _ = message_markers(TemplateTokens(tok, template))
    opener = tok.id_to_token(int(openers[0])) if len(openers) == 1 else None
    return SegmentEncoder(tok, opener) if opener is not None and SegmentEncoder.qualifies(tok, opener) else None


class App:
    """Serve one engine with sampling and reply-length defaults for requests that omit them."""

    reads_ignore_eos = False            # True where the engine reads ``ignore_eos`` itself; a ``stop_eos`` engine is given it
    segments: SegmentEncoder | None = None   # chat prompts a message at a time (None: encoded whole)

    def __init__(self, engine, model_dir: Path, served: str, *, default_thinking: bool = False,
                 sampling: dict[str, Any] | None = None, max_tokens: int = 4096,
                 context_window: int | None = None, reasoning_effort: str | None = None, thinking_budget: int = 0,
                 aliases: tuple[str, ...] | list[str] = (), vision_max_images: int | None = None,
                 vision_image_tokens: int | None = None,
                 background_ids: tuple[str, ...] | frozenset[str] = (),
                 max_pending_requests: int | None = None, max_pending_requests_per_principal: int | None = None,
                 max_http_connections: int | None = None, max_http_body_bytes: int | None = None,
                 logical_api_limits=None, startup_owner=None,
                 chat_template_file=None, chat_template_name=None):
        self.engine = engine
        self._http_payload_failure = None
        self.max_http_connections = optional_limit(max_http_connections, "max_http_connections")
        from tensorfold.cuda.http import GENERATION_BODY_LIMIT

        body_limit = optional_limit(max_http_body_bytes, "max_http_body_bytes")
        self.max_http_body_bytes = (GENERATION_BODY_LIMIT if body_limit is None
                                    else max(GENERATION_BODY_LIMIT, body_limit))
        if logical_api_limits is not None:
            from tensorfold.server.api_records import ApiRecordLimits

            if type(logical_api_limits) is not ApiRecordLimits:
                raise TypeError("logical_api_limits must be the actual parsed ApiRecordLimits")
            self.max_http_body_bytes = max(self.max_http_body_bytes, logical_api_limits.max_operation_bytes)
        self.logical_api_limits = logical_api_limits
        if startup_owner is not None:
            if not callable(startup_owner):
                raise TypeError("startup_owner must register the partial App")
            startup_owner(self)
        self.max_pending_requests = optional_limit(max_pending_requests, "max_pending_requests")
        self.max_pending_requests_per_principal = optional_limit(
            max_pending_requests_per_principal, "max_pending_requests_per_principal")
        if self.max_pending_requests_per_principal is not None and self.max_pending_requests is None:
            raise ValueError("max_pending_requests_per_principal requires max_pending_requests")
        self._request_limit = (RequestLimit(max_pending_requests, per_principal=max_pending_requests_per_principal)
                               if max_pending_requests is not None else None)
        from tensorfold.token_metadata import load_tokenizer

        self.vision = getattr(engine, "vision", None)
        self.vision_image_tokens = vision_image_tokens
        self.image_limits = DEFAULT_LIMITS if vision_max_images is None and vision_image_tokens is None else \
            ImageLimits(**({} if vision_max_images is None else {"max_images": vision_max_images}),
                        **({} if vision_image_tokens is None else {"max_visual_tokens": vision_image_tokens}))
        self.served = served
        self.aliases = tuple(str(alias).strip() for alias in aliases if str(alias).strip())
        self.background_ids = frozenset(background_ids)   # --name-priority ID=background: a default for this id
        self.model_dir = Path(model_dir)
        if getattr(self.vision, "is_media_runtime", False):
            self.tok = self.vision.tokenizer
            if self.tok is None:
                raise ValueError("the media server requires its adopted leader tokenizer")
        else:
            self.tok = load_tokenizer(model_dir)
        self.template = getattr(engine, "chat_template", None)
        options = (None if chat_template_file is None else str(Path(chat_template_file).expanduser().resolve()), chat_template_name)
        if self.template is None:
            self.template = ChatTemplate(model_dir, template_file=chat_template_file, template_name=chat_template_name)
        elif options != (None, None) and options != self.template.options:
            raise ValueError("engine and server must use the same captured chat template selection")
        self.segments = _segment_encoder(self.tok, self.template)
        self.default_thinking = default_thinking
        fixed = self._fixed_thinking(False)
        if fixed is not None:
            self.default_thinking = fixed
        self.reasoning_effort, self.thinking_budget = reasoning_effort, int(thinking_budget)   # the Mac's defaults
        # what generation_config.json and the flags leave unset falls back as on the Mac (README): greedy, filters off
        self.sampling = {"temperature": 0.0, "top_k": 0, "top_p": 1.0, **(sampling or {})}
        unapplied = unapplied_penalties(engine, self.sampling)
        if unapplied:
            print(f"[tensorfold] this engine does not apply penalties; not applying defaults {unapplied}",
                  file=sys.stderr, flush=True)
        self.max_tokens = int(max_tokens)
        self.native_context_window = _native_context(model_dir)
        self.context_window = self.native_context_window if context_window is None else int(context_window)
        if self.context_window < 0:
            raise ValueError("context_window must be 0 or a positive token count")
        self.turns = Turns()                # one request at a time where the engine decodes one
        self._finalize_retention_resources()
        profiles = getattr(engine, "serving_profiles", None)
        # The endpoint's actual tokenizer, context and host commitments must
        # exist before the scheduler can accept this logical request owner.
        if self._request_limit is not None:
            scheduler = getattr(engine, "scheduler", None)
            if scheduler is not None:
                scheduler.configure_requests(self._request_limit)
            if profiles is None:
                self.run = self._request_limit.wrap(self.run)
            self.decisions = self._request_limit.wrap(self.decisions)
        if profiles is not None:
            from tensorfold.cuda.profile_app import AppProfiles

            self._profiles = AppProfiles(self, profiles)

    def _fatal_http_payload(self, primary):
        """Irrecoverable native encoder allocation failure retires the cohort."""
        if self._http_payload_failure is not None:
            return
        self._http_payload_failure = primary
        self.engine._abort_comm = True
        from tensorfold.cuda.cohort import notify_failure

        notify_failure()  # Borrowed launcher capability; no communicator destruction here.

    def _finalize_retention_resources(self):
        catalog = getattr(self.engine, "retention_catalog", None)
        if catalog is None:
            return None
        from tensorfold.families.qwen4_exp.cuda.retention_resources import finalize_retention_resources

        scheduler = getattr(self.engine, "scheduler", None)
        # The HTTP class outlives profile publication. Keep managed endpoints
        # conservative until their encoder hook can follow that lifetime.
        fixed_profile = getattr(self.engine, "serving_profiles", None) is None
        serialized_chars = 0
        static_metadata = None
        if (fixed_profile and catalog.world == 8 and catalog.policy is None
                and callable(getattr(self.tok, "to_str", None))):
            serialized_chars = len(self.tok.to_str())
            from tensorfold.server import stats
            static_metadata = stats._fixed(self)
        result = finalize_retention_resources(catalog, tokenizer=self.tok,
            max_pending_requests=None if self._request_limit is None else self._request_limit.maximum,
            max_waiting_input_bytes=getattr(scheduler, "max_waiting_input_bytes", None),
            output_queue_tokens=getattr(scheduler, "output_queue_tokens", None),
            max_http_connections=self.max_http_connections, max_http_body_bytes=self.max_http_body_bytes,
            logical_api_limits=self.logical_api_limits, scheduler=scheduler if fixed_profile else None,
            request_limit=self._request_limit if fixed_profile else None,
            segment_encoder=getattr(self, "segments", None), tokenizer_serialized_chars=serialized_chars,
            static_http_metadata=static_metadata)
        self.host_payload_pool = getattr(catalog, "_http_payload_pool", None)
        if self.host_payload_pool is not None:
            self.engine.host_payload_pool = self.host_payload_pool
            scheduler.host_payload_pool = self.host_payload_pool
        return result

    def _host_credits(self):
        from tensorfold.server.host_payload import current

        owner = current()
        if owner is not None and owner.pool is not self.host_payload_pool:
            raise ValueError("HTTP tokenizer belongs to a different host payload pool")
        return owner

    def _retained_constraint_refusal(self):
        catalog = getattr(self.engine, "retention_catalog", None)
        if catalog is None:
            return None
        from tensorfold.families.qwen4_exp.cuda.retention_resources import retained_constraint_reason
        from tensorfold.server.api_records import current_operation

        operation = current_operation()
        return retained_constraint_reason(catalog,
            logical_retained=operation is not None and operation.logical_retained)

    @property
    def thinking_markers(self):
        """Checkpoint channel tokens select Gemma's reasoning boundary once."""
        if not hasattr(self, "_thinking_markers"):
            self._thinking_markers = CHANNEL_MARKERS if self.tok.token_to_id(CHANNEL_MARKERS[1]) is not None else THINK_MARKERS
        return self._thinking_markers

    def _fixed_thinking(self, tools):
        """The selected immutable template's unconditional switch, cached by route."""
        if self.thinking_markers != CHANNEL_MARKERS:
            return None
        cache = self.__dict__.setdefault("_thinking_fixed", {})
        route = bool(tools)
        if route not in cache:
            source = self.template.selected(tools=tools).source.source
            if (getattr(self.template, "options", (None, None)) != (None, None)
                    and ("<think>" in source or "</think>" in source)
                    and CHANNEL_MARKERS[0] not in source and CHANNEL_MARKERS[1] not in source):
                raise RequestError("the selected chat template uses Qwen thinking markers with a Gemma tokenizer; "
                                   "select a template with Gemma's native thought channels")
            cache[route] = fixed_thinking(source)
        return cache[route]

    @property
    def pending_requests(self) -> int | None:
        return self._request_limit.used if self._request_limit is not None else None

    @property
    def model_ids(self) -> list[str]:
        """The ids this endpoint answers to, as the MLX server lists them: ``--name`` first, then each ``--alias``."""

        ids: list[str] = []
        for model_id in (self.served, *getattr(self, "aliases", ())):
            if model_id and model_id not in ids:
                ids.append(model_id)
        return ids

    def reply_model(self, body: Any) -> str:
        """The id ``--name-priority`` keys a request's default priority by: the one it asked for when this endpoint
        answers to it, else ``--name``. (The id a reply names is ``server.http.requested_model``'s.)"""

        asked = body.get("model") if isinstance(body, dict) else None
        if isinstance(asked, str) and getattr(self, "model_policies", False):     # GLM's ID@POLICY names ID
            asked = asked.split("@", 1)[0]
        return asked if isinstance(asked, str) and asked in self.model_ids else self.served

    def _check_fields(self, body: dict[str, Any]) -> str | None:
        import inspect

        if not isinstance(body, dict):
            return "the request body must be a JSON object"
        try:
            probability_options(body, supported=bool(getattr(self.engine, "supports_logprobs", False)))
        except RequestError as exc:
            return str(exc)
        if body.get("draft", True) is False and "draft" not in inspect.signature(self.engine.generate).parameters:
            return "this model's CUDA engine has no serial switch (\"draft\": false)"
        if not isinstance(body.get("messages", []), list):
            return "messages must be a list"
        problem = grammar.refusal(body)                 # a malformed grammar field, or one beside a required call
        if problem is None and grammar.request_spec(body) and "constraint" not in inspect.signature(
                self.engine.generate).parameters:
            problem = "this model's engine does not enforce structured output"
        if problem is None and grammar.request_spec(body) and getattr(self.engine, "refuses_structured_output", None):
            problem = self.engine.refuses_structured_output       # e.g. Flash Next on two ranks with --parallel
        return problem

    def _grammars(self) -> grammar.Grammars:
        return grammar.compiler(self, getattr(self, "model_dir", None), self.engine.eos)

    def _engine_capacity(self) -> int | None:
        capacities = []
        for name in ("context_window", "limit"):
            limit = getattr(self.engine, name, None)
            if isinstance(limit, int):
                capacities.append(max(0, limit))
        return min(capacities) if capacities else None

    def _restart(self, need: int, ranks: str = "") -> str:
        """A larger ``--context`` to restart with, only where the startup admission would accept it."""

        largest = (getattr(self.engine, "capacity_plan", None) or {}).get("largest_window")
        if largest is None or need > largest:
            return ""
        if getattr(self.engine, "serving_profiles", None) is not None:
            return f", or select a prepared serving profile with context {need} or more through /v1/runtime"
        return f", or restart{ranks} with --context {need} or more (this memory admits up to {largest})"

    def _context_limit(self) -> int | None:
        limits = [self.context_window] if self.context_window > 0 else []
        capacity = self._engine_capacity()
        if capacity is not None:
            limits.append(capacity)
        return min(limits) if limits else None

    @property
    def effective_context_window(self) -> int | None:
        """Safe prompt-plus-reply capacity; None is unlimited, while zero refuses every prompt."""

        return self._context_limit()

    @property
    def max_model_len(self) -> int | None:
        """The context length /v1/models and /stats report: the effective window, else the checkpoint's own (None
        when neither is known)."""

        limit = self._context_limit()
        return limit if limit is not None else (self.native_context_window or None)

    def decisions(self, body: dict[str, Any]) -> dict[str, Any]:
        """Answer typed questions from next-token label logits. No text is generated."""

        from jinja2.exceptions import TemplateError

        from tensorfold.server.decisions import DecisionError, build_response, prompts_for
        from tensorfold.server.prompt_literals import PromptEncodingError

        if not hasattr(self.engine, "score_labels"):
            raise RequestError("this model's CUDA engine does not score decision labels")
        credit = self._host_credits() if getattr(self, "host_payload_pool", None) is not None else None
        held_ids = 0
        if credit is not None:
            credit.render_input(body)

        def render(content: str) -> str:
            try:
                return self.template.render([{"role": "user", "content": content}], tools=None, enable_thinking=False,
                    **({"render_credit": credit.render} if credit is not None else {}))
            except TemplateError as exc:
                raise DecisionError(f"the chat template failed: {exc}") from exc

        def encode(text: str) -> list[int]:
            nonlocal held_ids
            try:
                if credit is None:
                    return [int(token) for token in self.template.encode(self.tok, text, segments=self.segments)]
                def reserve(count, **kwargs):
                    credit.ids(held_ids + count, **kwargs)
                values = self.template.encode(self.tok, text, segments=self.segments,
                    ids_credit=reserve, cache_credit=credit.segment_cache)
                held_ids += len(values)
                credit.ids(held_ids + len(values))  # Source IDs and the caller's typed copy coexist.
                return [int(token) for token in values]
            except PromptEncodingError as exc:
                raise DecisionError(f"the chat template could not encode literal content: {exc}") from exc

        try:
            prepared = prompts_for(body, render, encode, context_len=self.effective_context_window)
        except DecisionError as exc:
            raise RequestError(str(exc)) from exc
        turns = self._turns()
        turns.take(False)
        try:
            scored = []
            for item in prepared:
                try:
                    scored.append(self.engine.score_labels(item.prompt_ids, item.label_ids))
                except ValueError as exc:
                    raise RequestError(f"question {item.id!r}: {exc}") from exc
            return build_response(body, prepared, scored)
        finally:
            turns.give()

    def _requested_tokens(self, body: dict[str, Any]) -> int:
        for name in ("max_tokens", "max_completion_tokens"):
            value = body.get(name)
            if value is not None:
                try:
                    int(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RequestError(f"{name} must be an integer token count") from exc
        return max(1, int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens))

    def _prepare(self, body: dict[str, Any], chat: bool, *, for_generation=True,
                 add_generation_prompt: bool = True) -> PreparedRequest:
        from jinja2.exceptions import TemplateError

        credit = self._host_credits() if getattr(self, "host_payload_pool", None) is not None else None
        if credit is not None:
            credit.render_input(body)
        if type(add_generation_prompt) is not bool:
            raise RequestError("add_generation_prompt must be a boolean")
        validate_modalities(body)
        policy = ToolCallPolicy(body)
        ignore_eos, stop = stop_options(body)
        max_tokens = self._requested_tokens(body)
        try:
            tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
        except ValueError as exc:
            raise RequestError(str(exc)) from None
        kwargs = body.get("chat_template_kwargs")
        if kwargs is None:                   # absent or null: the server's defaults
            kwargs = {}
        elif not isinstance(kwargs, dict):   # [], "", false and 0 included
            raise RequestError("chat_template_kwargs must be a JSON object or null")
        kwargs = dict(kwargs)
        # reasoning_effort and enable_thinking as the Mac server reads them: the template hears an effort when thinking
        levels = self.template.selected(tools=tools).efforts
        fields = thinking_fields(body, levels)
        fixed = self._fixed_thinking(tools)
        if fixed is not None:
            if "enable_thinking" in fields and fields["enable_thinking"] is not fixed:
                raise RequestError(f"the selected chat template fixes enable_thinking={str(fixed).lower()}; "
                                   "select a chat template that supports the requested thinking switch")
            fields["enable_thinking"] = fixed
        kwargs.pop("enable_thinking", None)
        kwargs.pop("reasoning_effort", None)
        thinking = bool(fields.get("enable_thinking", self.default_thinking))
        effort = heard_effort(fields.get("reasoning_effort"), getattr(self, "reasoning_effort", None), levels)
        if thinking and effort:
            kwargs["reasoning_effort"] = effort
        budget = parse_numbers({"thinking_budget": body.get("thinking_budget")})["thinking_budget"]
        penalties = self.penalties_for(body)
        budget = thinking_budget(budget, getattr(self, "thinking_budget", 0)) if chat and thinking else 0
        spec = grammar.request_spec(body)
        if for_generation and (spec is not None or tools and strict_tools.wanted(tools)):
            reason = self._retained_constraint_refusal()
            if reason is not None:
                raise RequestError(reason)
        calls = self._strict_calls(body, tools, spec, policy, for_generation=for_generation)
        spec = calls.spec if calls is not None else spec
        grammar.check_stopping(body, spec)
        top = probability_options(body, supported=bool(getattr(self.engine, "supports_logprobs", False)))
        if top is not None:
            if not chat or body.get("stream") or thinking or tools or stop or spec is not None or budget:
                raise RequestError("logprobs support nonstreamed text chat with thinking off, without tools, "
                                   "stop strings or structured output")
            if not hasattr(self, "_probability_decoder"):
                self._probability_decoder = TokenBytes(self.tok)
        compiled = (spec, self._grammars().compile(spec)) if spec is not None and for_generation else None
        thought_open = False
        if chat:
            if not isinstance(body.get("messages"), list):
                raise RequestError("messages must be a list")
            from tensorfold.server.prompts import has_images, prepare_images

            def render(messages: list[dict[str, Any]], **images: bool) -> str:   # text renders as it always has
                nonlocal thought_open
                try:
                    ceiling = getattr(self, "render_budget_bytes", None)
                    text = self.template.render(messages, tools=tools, enable_thinking=thinking, extra=kwargs,
                                                **({"render_credit": credit.render} if credit is not None else {}),
                                                **({"max_render_bytes": ceiling} if ceiling is not None else {}),
                                                **({"add_generation_prompt": False} if not add_generation_prompt else {}),
                                                **images)
                except TemplateError as exc:     # the checkpoint's template refuses the request (``raise_exception``)
                    raise RequestError(f"the chat template rejected the request: {exc}") from exc
                if self.thinking_markers == CHANNEL_MARKERS:
                    from tensorfold.server.gemma_values import prompt_opens_thought

                    from tensorfold.server.prompt_literals import trusted_prompt_text
                    thought_open = prompt_opens_thought(trusted_prompt_text(text))
                return text

            if has_images(body["messages"]):
                retire = (self.vision.requests.close_if_unclaimed
                          if getattr(self.vision, "is_media_runtime", False) else None)
                rendered = prepare_images(self.vision, body["messages"],
                                          lambda messages: render(messages, allow_images=True,
                                                                  allow_audio=bool(getattr(self.vision, 'audio', False))),
                                          context_limit=self._context_limit(),
                                          limits=getattr(self, "image_limits", DEFAULT_LIMITS),
                                          max_visual_tokens=getattr(self, "vision_image_tokens", None))
                try:
                    return PreparedRequest(rendered.tokens, max_tokens, tools, thinking,
                                           self.sampling_for(body, rendered.tokens), ignore_eos=ignore_eos, stop=stop,
                                           vision=rendered.vision, grammar=compiled, think_budget=budget, calls=calls,
                                           thought_open=thought_open, penalties=penalties, _media_release=retire,
                                           count_only=not for_generation)
                except BaseException as primary:
                    if retire is not None:
                        from tensorfold.cleanup import rollback

                        rollback(rendered.vision, primary, lambda: retire(rendered.vision))
                    raise
            text = render(body["messages"])
            from tensorfold.server.prompt_literals import PromptEncodingError
            try:
                if credit is None:
                    prompt = self.template.encode(self.tok, text, segments=self.segments)
                else:
                    prompt = self.template.encode(self.tok, text, segments=self.segments,
                                                  ids_credit=credit.ids, cache_credit=credit.segment_cache)
            except PromptEncodingError as exc:
                raise RequestError(f"the chat template could not encode literal content: {exc}") from exc
        elif isinstance(body.get("prompt"), list):       # token ids (vLLM's and OpenAI's form): served as given
            prompt = self.token_ids(body["prompt"])
        else:
            text = body.get("prompt")
            if not isinstance(text, str):
                raise RequestError("prompt must be a string or a list of token ids")
            if credit is None:
                prompt = self.tok.encode(text, add_special_tokens=flag(body, "add_special_tokens", False)).ids
            else:
                encoded = self.tok.encode(text, add_special_tokens=flag(body, "add_special_tokens", False))
                credit.ids(len(encoded))
                prompt = encoded.ids
                del encoded
        if not prompt:
            raise RequestError("rendered prompt is empty")
        # sampling is resolved here, so a malformed control is refused before a stream opens
        return PreparedRequest(prompt, max_tokens, tools, thinking, self.sampling_for(body, prompt),
                               ignore_eos=ignore_eos, stop=stop, grammar=compiled, think_budget=budget, calls=calls,
                               thought_open=thought_open, penalties=penalties, count_only=not for_generation)

    def _strict_calls(self, body: dict[str, Any], tools: list[dict[str, Any]], spec: Any,
                      policy: ToolCallPolicy, *, for_generation: bool = True) -> "strict_tools.Plan | None":
        """The call grammar of a request whose active tools ask for strict arguments, else None (RequestError: 400)."""

        if not tools or not strict_tools.wanted(tools):
            return None
        if spec is not None:
            raise RequestError(f"{spec.field} cannot be combined with strict function tools: send one")
        if for_generation and "constraint" not in inspect.signature(self.engine.generate).parameters:
            raise RequestError("this model's engine does not enforce strict function tools: send strict false")
        if for_generation and getattr(self.engine, "refuses_structured_output", None):
            raise RequestError(self.engine.refuses_structured_output)
        if getattr(self, "_strict_markup", None) is None:          # (plan factory, why not), read once
            try:
                rendered = self.template.render(strict_tools.PROBE, tools=strict_tools.PROBE_TOOLS, enable_thinking=False)
            except Exception as exc:  # noqa: BLE001 - a template that renders no calls cannot be held to them
                self._strict_markup = (False, f"could not render a call ({exc})")
            else:
                why = "writes calls otherwise"
                if strict_tools.template_supported(rendered):
                    plan = strict_tools.Plan
                else:
                    from tensorfold.server import strict_scalar_tools

                    messages = strict_scalar_tools.false_probe()
                    plan = None
                    try:
                        rendered_false = self.template.render(messages, tools=strict_tools.PROBE_TOOLS,
                                                              enable_thinking=False)
                    except Exception as exc:  # noqa: BLE001 - the actual false-value template route must render too
                        why = f"could not render a call ({exc})"
                    else:
                        if strict_scalar_tools.template_supported(rendered, rendered_false):
                            plan = strict_scalar_tools.Plan
                        else:
                            from tensorfold.server import strict_gemma_tools

                            try:
                                rendered = self.template.render(strict_gemma_tools.PROBE,
                                                                tools=strict_gemma_tools.PROBE_TOOLS, enable_thinking=False)
                                rendered_false = self.template.render(strict_gemma_tools.false_probe(),
                                                                      tools=strict_gemma_tools.PROBE_TOOLS,
                                                                      enable_thinking=False)
                            except Exception as exc:  # noqa: BLE001 - discovery failure is an explicit refusal
                                why = f"could not render a call ({exc})"
                            else:
                                null = strict_gemma_tools.template_null(rendered, rendered_false)
                                if null is not None:
                                    plan = functools.partial(strict_gemma_tools.Plan, null=null)
                self._strict_markup = (plan, why)
        if not self._strict_markup[0]:
            raise RequestError("strict function tools require exact supported Qwen XML or Gemma call markup, "
                               f"and this model's chat template {self._strict_markup[1]}: send strict false")
        return self._strict_markup[0](tools, required=tool_choice_requires_call(body.get("tool_choice")),
                                      parallel=not policy.single, offered=body.get("tools"))

    def token_ids(self, value: Any, field: str = "prompt") -> list[int]:
        """Token ids as a request gives them (a list, or a list holding one list); RequestError outside the
        vocabulary."""

        from tensorfold.token_metadata import tokenizer_vocab_size

        if getattr(self, "host_payload_pool", None) is not None:
            credit = self._host_credits()
            if credit is not None:
                items = value[0] if type(value) is list and len(value) == 1 and type(value[0]) is list else value
                if type(items) is list:
                    credit.ids(len(items))
        size = tokenizer_vocab_size(self.tok) if hasattr(self.tok, "get_vocab_size") else None
        return token_ids(value, size, field)

    def tokenize(self, body: dict[str, Any]) -> dict[str, Any]:
        """vLLM's ``/tokenize`` (``server.token_routes``): a prompt's ids (``add_special_tokens``, default true, as
        vLLM's), or ``messages``' as the chat route renders them (``add_generation_prompt``, default true)."""

        from tensorfold.cleanup import rollback

        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        strings = flag(body, "return_token_strs", False)
        credit = self._host_credits() if getattr(self, "host_payload_pool", None) is not None else None
        if credit is not None:
            credit.tokenization_reply, credit.lexical_output = True, strings
        if "messages" in body:
            fields = dict(body)
            def release_prepared_media():
                prepared.release_media()

            prepared = self._prepare(fields, True, for_generation=False,
                                     add_generation_prompt=flag(body, "add_generation_prompt", True))
            try:
                ids = prepared.prompt
            except BaseException as primary:
                rollback(prepared, primary, release_prepared_media)
            release_prepared_media()
        else:
            text = body.get("prompt")
            if not isinstance(text, str):
                raise RequestError("prompt must be a string (or send messages)")
            if credit is None:
                ids = self.tok.encode(text, add_special_tokens=flag(body, "add_special_tokens", True)).ids
            else:
                encoded = self.tok.encode(text, add_special_tokens=flag(body, "add_special_tokens", True))
                credit.ids(len(encoded))
                ids = encoded.ids
                del encoded
        if credit is not None:
            credit.ids(len(ids))  # Reply copy/lexical strings are funded before either is built.
        limit = self._context_limit()
        reply: dict[str, Any] = {"count": len(ids), "tokens": [int(t) for t in ids],
                                 "max_model_len": limit if limit is not None else self.native_context_window}
        if strings:
            unused = getattr(self.tok, "unused_token_ids", ())
            reply["token_strs"] = ["" if t in unused else self.tok.id_to_token(t) for t in reply["tokens"]]
        return reply

    def detokenize(self, body: dict[str, Any]) -> dict[str, Any]:
        """vLLM's ``/detokenize``: the text of ``tokens``, special tokens included."""

        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        return {"prompt": self.tok.decode(self.token_ids(body.get("tokens"), "tokens"), skip_special_tokens=False)}

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Why the request cannot run, or None; rendered before a stream's headers are sent."""

        problem = self._check_fields(body)
        if problem:
            return problem
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        limit = self._context_limit()
        if limit is not None and len(prepared.prompt) >= limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            native = f" (model window: {self.native_context_window} tokens)" if self.native_context_window else ""
            return (f"{CONTEXT_LIMIT} {limit} tokens: the rendered prompt has {len(prepared.prompt)} tokens and leaves "
                    f"no room for a reply in the server's {limit}-token {kind}{native}, which exceeds the context "
                    f"window; shorten the prompt"
                    f"{self._restart(len(prepared.prompt) + 1)}")
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        if limit is not None and asked and len(prepared.prompt) + prepared.max_tokens > limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            return (f"{CONTEXT_LIMIT} {limit} tokens: the rendered prompt has {len(prepared.prompt)} tokens and "
                    f"requests {prepared.max_tokens} reply tokens, which exceeds the context window (the server's "
                    f"{limit}-token {kind}); reduce the prompt or reply length"
                    f"{self._restart(len(prepared.prompt) + prepared.max_tokens)}")
        return None

    def prepare_for_count(self, body: dict[str, Any], chat: bool) -> PreparedRequest:
        """Existing prompt/constraint validation without a generation-tier promise."""
        return self.prepare(body, chat, _for_generation=False)

    def prepare(self, body: dict[str, Any], chat: bool, *, _for_generation=True) -> PreparedRequest:
        from tensorfold.cleanup import rollback

        def release_prepared_media():
            prepared.release_media()

        problem = self._check_fields(body)
        if problem:
            raise refusal(problem)
        prepared = self._prepare(body, chat, for_generation=_for_generation)
        try:
            problem = self.check(body, prepared=prepared)
            if problem:
                raise refusal(problem)
            limit = self._context_limit()
            if limit is not None:
                prepared.max_tokens = min(prepared.max_tokens, limit - len(prepared.prompt))
            return prepared
        except BaseException as primary:
            rollback(prepared, primary, release_prepared_media)

    def sampling_for(self, body: dict[str, Any], prompt: list[int]):
        """Keyed sampling (the seed, else one drawn from the prompt), or None for greedy; RequestError if malformed."""

        from tensorfold.engine.exact_sampling import Sampling, seed_for

        fields = parse_numbers({k: body[k] for k in _SAMPLING_FIELDS if body.get(k) is not None})
        temp = float(fields.get("temperature", self.sampling["temperature"]))
        if temp <= 0:
            return None
        top_p = fields.get("top_p", self.sampling["top_p"])
        if top_p == 0:                                  # top_p 0 keeps only the likeliest token (HF): greedy
            return None
        seed = fields.get("seed")
        top_k = fields.get("top_k", self.sampling["top_k"])
        min_p = fields.get("min_p", self.sampling.get("min_p", 0.0))
        return Sampling(int(seed) if seed is not None else seed_for(prompt), temp, int(top_k), float(top_p),
                        float(min_p))

    def penalties_for(self, body: dict[str, Any]):
        """The request's active penalties over the server's defaults, None when off; RequestError if malformed or
        the engine has no penalty path (penalties apply under greedy decoding too, as in vLLM)."""

        from tensorfold.engine.penalties import Penalties
        from tensorfold.server.request_options import parse_logit_bias
        from tensorfold.server.token_routes import _vocab

        fields = parse_numbers({k: body[k] for k in PENALTY_FIELDS if body.get(k) is not None})
        value = {k: float(fields.get(k, self.sampling.get(k, off))) for k, off in PENALTY_FIELDS.items()}
        raw_bias = body.get("logit_bias")
        vocab = None
        if isinstance(raw_bias, dict) and raw_bias:
            config = getattr(getattr(self.engine, "target", None), "config", None)
            if config is None:
                config = getattr(getattr(self.engine, "w", None), "config", None)
            vocab = getattr(config, "vocab", None)
            if type(vocab) is not int or vocab <= 0:
                vocab = _vocab(self.tok)
            if type(vocab) is not int or vocab <= 0:
                raise RequestError("logit_bias requires the model's output vocabulary metadata")
        bias = parse_logit_bias(raw_bias, vocab)
        penalties = Penalties(value["presence_penalty"], value["frequency_penalty"], value["repetition_penalty"], bias=bias)
        if not penalties.active:
            return None
        if not applies_penalties(self.engine):
            if bias:
                raise RequestError("this model's engine does not apply logit_bias; omit active bias")
            raise RequestError("this model's engine does not apply presence, frequency or repetition penalties: "
                               "send them as 0, 0 and 1 (or omit them)")
        return penalties

    def _reserve_http_generation(self, body, prepared, cancelled):
        pool = getattr(self, "host_payload_pool", None)
        if pool is None:
            return
        from tensorfold.server.host_payload import current

        owner = current()
        if owner is None:
            # A direct, non-HTTP caller cannot borrow socket/phase authority.
            # Preserve its original conservative commitment check unchanged.
            from tensorfold.cuda.capacity import admit_host_commitment
            record = self.engine.retention_catalog.host_resource_envelope["static_maximum_projection"]
            admit_host_commitment(record["global_host_commitment_bytes"],
                                  materialized_owned_bytes=record["materialized_owned_bytes"])
            return
        if owner.pool is not pool:
            raise ValueError("HTTP generation belongs to a different host payload pool")
        if cancelled is not None:
            owner.cancelled = cancelled
        context = pool.context_tokens
        output = context
        # Thinking/tool/grammar/gating/replay regions keep the established full
        # context envelope. Only this plain no-thinking operation has no legal
        # forced/gated continuation beyond its effective requested token cap.
        if prepared is not None:
            plain = (not prepared.thinking and not prepared.tools and prepared.grammar is None
                     and prepared.calls is None)
            if plain:
                output = min(context, max(1, prepared.max_tokens))
        elif (not self.default_thinking and body.get("enable_thinking", False) is False
              and all(key in (
                "model", "messages", "prompt", "temperature", "top_k", "top_p", "min_p", "seed",
                "max_tokens", "max_completion_tokens", "return_token_ids", "stream", "stream_options",
                "draft", "ignore_eos", "stop", "priority", "service_tier", "user", "enable_thinking")
                for key in body)):
            output = min(context, max(1, self._requested_tokens(body)))
        owner.generation(body_bytes=owner.body_bytes, output_tokens=output)

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, cancelled: Callable[[], bool] | None = None,
            request_id: str | None = None, arrived: float | None = None,
            progress: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
        """One reply; once ``cancelled()`` holds, a waiting request raises ``RequestCancelled`` unstarted, a running one stops at its next round and raises it after ``generate``.

        ``request_id`` names the request in its ``tensorfold`` block and ``done`` line (one is made if None);
        ``arrived`` is the handler's clock when the request came (perf_counter): its timings start there.
        ``progress`` receives llama.cpp's ``prompt_progress`` (total, cache, processed, time_ms) while the prompt
        prefills, where the engine reports it, and returns False once the client has gone."""

        if prepared is not None and getattr(prepared, "count_only", False):
            raise RequestError("count-only preparation cannot be used for generation; prepare the request for generation")
        self._reserve_http_generation(body, prepared, cancelled)
        entered = time.perf_counter()
        request_id = request_id or new_request_id()
        arrived = entered if arrived is None else float(arrived)
        if prepared is None:                # admitted (--max-pending-requests) before preparing: that wait is queued
            prepared = self.prepare(body, chat)
            prepare_s, admitted_s = time.perf_counter() - entered, entered - arrived
        else:
            prepare_s, admitted_s = entered - arrived, 0.0
        _media_primary = None
        try:
            if prepared.grammar is not None or prepared.calls is not None:
                reason = self._retained_constraint_refusal()
                if reason is not None:
                    raise RequestError(reason)
            prompt, max_tokens = prepared.prompt, prepared.max_tokens
            tools, thinking = prepared.tools, prepared.thinking
            markers = self.thinking_markers
            split_reasoning = chat and (thinking or markers == CHANNEL_MARKERS)
            gemma = markers == CHANNEL_MARKERS
            policy = ToolCallPolicy(body)
            sampling = prepared.sampling
            # end tokens end the reply and stay out of its text, unless it asks ignore_eos of an engine that reads it
            takes_stop_eos = "stop_eos" in inspect.signature(self.engine.generate).parameters
            ends = () if prepared.ignore_eos and (self.reads_ignore_eos or takes_stop_eos) else tuple(self.engine.eos)
            stops = StopStrings(prepared.stop, self.tok, ends)
            out: list[int] = []
            sent = {"reasoning": 0, "content": 0}
            stopped = {"client": False, "stop": False}
            failed: list[Exception | None] = [None]  # preallocated callback failure cell; no append on exhaustion
            stream = StreamDecoder(self.tok, ends)
            # streamed calls go out as argument deltas while written (as on the Mac); one-call requests keep the end parser
            strict = prepared.calls
            calls_stream = ((strict.reader() if strict is not None else ToolCallStreamer(tools))
                            if tools and not policy.single and body.get("stream")
                            and (prepared.grammar is None or strict is not None)
                            and (strict is not None or not gemma) else None)
            answer_raw = [""]
            native_starts = tuple(t for t in (self.tok.token_to_id("<|channel>"), self.tok.token_to_id("<|think|>"))
                                  if t is not None) if gemma else ()
            grammar_state = [None]  # request-owned boundary, independent of producer/rank matcher copies
            if gemma and prepared.grammar is None:
                grammar_state[0] = grammar.AnswerBoundary(None, think_end=self.tok.token_to_id(markers[1]),
                    think_starts=native_starts, thought_open=prepared.thought_open, decode=self.tok.decode,
                    answer_headers=tuple("<|tool_call>call:" + tool_spec_name(t) + "{" for t in tools))
            forced_close = [None]  # at most one explicit budget event in this logical reply
            answer_stream = (StreamDecoder(self.tok, ends, literal=prepared.grammar is not None)
                             if prepared.grammar is not None or gemma else None)
            answer_fed = [0]

            def normal_tool_end() -> bool:
                return not stopped["stop"] and bool(out and out[-1] in ends)

            def visible(finished: bool) -> tuple[str, str]:
                raw = stream.final() if finished else stream.text
                # stop strings match the generated text, reasoning included, before it is split (as on the Mac)
                raw = stops.visible(raw, partial=not finished) if stops.strings else raw
                if grammar_state[0] is not None:
                    from tensorfold.token_metadata import decode_presentation
                    offset = grammar_state[0].answer_offset
                    if offset is None:
                        pending = grammar_state[0].pending_answer_offset
                        prefix = (decode_presentation(self.tok, [t for t in out[:pending] if t not in ends])
                                  if pending is not None and not finished else raw)
                        answer = ""
                        if gemma and not grammar_state[0].opened_thought:
                            # An ambiguous prefix stays buffered until a real
                            # close or normal finality proves direct content.
                            prefix = ""
                            if finished:
                                answer_stream.add(out[answer_fed[0]:])
                                answer_fed[0] = len(out)
                                answer = answer_stream.final()
                    else:
                        close_at = grammar_state[0].thinking_close_offset
                        prefix_end = close_at if close_at is not None else offset
                        prefix = decode_presentation(self.tok, [t for t in out[:prefix_end] if t not in ends])
                        answer_stream.add(out[offset + answer_fed[0]:])
                        answer_fed[0] = max(0, len(out) - offset)
                        answer = answer_stream.final() if finished else answer_stream.text
                    if gemma and prepared.grammar is None and answer:
                        from tensorfold.server.gemma_values import _clean_controls
                        answer = _clean_controls(answer, finished=finished)
                    if split_reasoning:
                        if gemma:
                            from tensorfold.server.gemma_values import split_gemma_thinking
                            reasoning = split_gemma_thinking(prefix, finished=finished,
                                opened=prepared.thought_open, expect_thinking=thinking, reasoning_only=True)[0]
                        else:
                            reasoning = split_thinking(prefix, finished=finished, markers=markers)[0]
                        # Gemma's known-phase cleaner already retains unframed
                        # reasoning. Empty can be framing-only: never reinsert
                        # raw headers or poison cumulative streamed-field lengths.
                        if not gemma and offset is None and not reasoning:
                            reasoning = prefix
                    else:
                        reasoning = ""
                elif split_reasoning:
                    if gemma:
                        from tensorfold.server.gemma_values import split_gemma_thinking

                        reasoning, answer = split_gemma_thinking(raw, finished=finished, opened=prepared.thought_open,
                                                                 expect_thinking=thinking)
                    else:
                        reasoning, answer = split_thinking(raw, finished=finished, markers=markers)
                    if finished and tools and not gemma and not answer:
                        # the end parser's split (below): a call missing only its closer, in a block that never closed,
                        # ends the reasoning where it starts, so its markup is the call and never reasoning_content
                        fixed = split_thinking(normalize_qwen_tool_markup(raw, finished=normal_tool_end()), finished=True, markers=markers)
                        if fixed[1]:
                            reasoning, answer = fixed
                else:
                    reasoning, answer = "", raw
                answer_raw[0] = answer
                if strict is not None:
                    answer = strict.content(answer, finished=finished)
                elif tools and gemma and prepared.grammar is None:
                    from tensorfold.server.gemma_values import hide_gemma_calls

                    answer = hide_gemma_calls(answer, finished=finished)
                elif tools and prepared.grammar is None:
                    answer = (policy.content(answer, finished=finished) if policy.single
                              else hide_tool_calls(answer, finished=finished))
                if tools and (prepared.grammar is None or strict is not None):
                    answer = visible_tool_prose(answer)
                return reasoning, answer

            serving: list[Any] = [None]

            def on_tokens(new: list[int]) -> bool:
                # True stops the engine after this round; engines that finish on both ranks keep calling and get True
                if stopped["client"] or stopped["stop"] or failed[0] is not None:
                    return True
                try:
                    if not new:                # a tick while the request waits or its prompt fills: keepalive, client check
                        if not emit(None) or (cancelled is not None and cancelled()):
                            stopped["client"] = True
                        return stopped["client"]
                    if stops.strings:
                        kept = []
                        for token in new:             # token by token: the round's width cannot move the cut
                            kept.append(token)
                            out.append(token)
                            if token not in stops.skip and stops.hit(out):
                                stopped["stop"] = True
                                break
                        new = kept
                    else:
                        out.extend(new)
                    if grammar_state[0] is not None:
                        for token in new:
                            if grammar_state[0].observed == forced_close[0]:
                                grammar_state[0].force_think_close(token)
                            else:
                                grammar_state[0].advance([token])
                    stream.add(new)
                    reasoning, answer = visible(False)
                    delta: dict[str, Any] = {}
                    if len(reasoning) > sent["reasoning"]:
                        delta["reasoning_content"] = reasoning[sent["reasoning"]:]
                    if len(answer) > sent["content"]:
                        delta["content"] = answer[sent["content"]:]
                    if delta:
                        if not emit(delta):
                            stopped["client"] = True
                        else:
                            # Callback acceptance follows canonical token commit.
                            # HTTP's callback includes write+flush, not peer receipt.
                            sent["reasoning"], sent["content"] = len(reasoning), len(answer)
                    if (not delta and body.get("stream")
                            and time.perf_counter() - last_ping[0] >= 1.0):
                        last_ping[0] = time.perf_counter()
                        if not emit(None):
                            stopped["client"] = True
                    if calls_stream is not None and not stopped["client"]:
                        for call_delta in calls_stream.feed(answer_raw[0]):   # never the reasoning
                            if not emit(call_delta):
                                stopped["client"] = True
                                break
                    if not stopped["client"] and cancelled is not None and cancelled():   # every round, text or not
                        stopped["client"] = True
                    if serving[0] is not None:
                        serving[0].saw()
                except Exception as exc:        # noqa: BLE001  raised after generate returns, never into the engine
                    if failed[0] is None:
                        failed[0] = exc
                    return True
                return stopped["client"] or stopped["stop"]

            last_ping = [time.perf_counter()]

            def pulse() -> bool:
                """Existing SSE comments only; no new bytes/protocol for nonstream replies."""
                now = time.perf_counter()
                if body.get("stream") and now - last_ping[0] >= 1.0:
                    last_ping[0] = now
                    stopped_here = on_tokens([])
                    if failed[0] is not None:
                        raise failed[0]  # queue callback has no active engine/native frame
                    return stopped_here
                return stopped["client"]

            draft = body.get("draft", True) is not False
            required = tools and strict is None and tool_choice_requires_call(body.get("tool_choice"))
            gate = self._call_gate(prompt, tools, thinking=thinking) if required else None      # strict calls: the grammar requires them

            options: dict[str, Any] = {} if draft else {"draft": False}
            probabilities = None
            if body.get("logprobs"):
                from tensorfold.engine.probabilities import Probabilities

                probabilities = Probabilities(body.get("top_logprobs") or 0, len(prompt), max_tokens)
                options["probabilities"] = probabilities
            if takes_stop_eos:
                options["stop_eos"] = not prepared.ignore_eos
            shaped = prepared.grammar is not None or prepared.think_budget > 0
            think_end = self.tok.token_to_id(markers[1]) if chat and thinking and shaped else None
            budget = self._think_budget(prepared, think_end)
            if gemma and grammar_state[0] is not None:
                if budget is not None:
                    budget.intent = grammar_state[0]
                if gate is not None:
                    gate.intent = grammar_state[0]
            # background: the request's own priority (OpenAI's service_tier "flex" when it sends none), else
            # --name-priority's default for its id; title requests always
            if "priority" in body:
                priority = body["priority"]
            elif body.get("service_tier") == "flex":
                priority = "background"
            else:
                priority = "background" if self.reply_model(body) in getattr(self, "background_ids", ()) else None
            background = priority == "background" or (chat and is_title_request(body.get("messages"), tools))
            concurrent = getattr(self.engine, "concurrent", False)
            turns = None if concurrent else self._turns()
            if concurrent and background and "background" in inspect.signature(self.engine.generate).parameters:
                options["background"] = True            # the engine's scheduler orders its lanes and prompts
            # one engine at a time: a background reply yields between rounds (not on two ranks, which decode to the end)
            yielding = background and turns is not None and getattr(self.engine, "tp", 1) == 1
            gates = [g for g in (gate, budget, Yield(turns) if yielding else None) if g is not None]

            cached: list[int] = []              # the prompt tokens the first run found cached (usage's cached_tokens)
            queued = [admitted_s]               # seconds waiting for the engine's turn (and for admission)

            def filled(total: int, cache: int, processed: int, elapsed_ms: float) -> bool:
                """The first run's prompt progress, until its first token; True: the client has gone."""
                if out or stopped["client"]:
                    return stopped["client"]
                if cancelled is not None and cancelled():
                    stopped["client"] = True
                    return True
                # Empty callback is already the SSE keepalive contract. Poll it
                # independently of opt-in progress frames, and never throw IO
                # into a running engine: on_tokens retains that failure.
                if body.get("stream") and time.perf_counter() - last_ping[0] >= 1.0:
                    last_ping[0] = time.perf_counter()
                    if on_tokens([]):
                        return True
                if progress is not None and not progress({"total": total, "cache": cache, "processed": processed,
                                                          "time_ms": int(elapsed_ms)}):
                    stopped["client"] = True
                return stopped["client"]

            # Nonstream clients have no progress writer, but still need the
            # supported engine's chunk-boundary disconnect poll during prefill.
            reports = ((progress is not None or cancelled is not None or body.get("stream"))
                       and "progress" in inspect.signature(self.engine.generate).parameters)

            def generate(ids: list[int], count: int, feed: Callable[[list[int]], bool]) -> Any:
                nonlocal released
                if yielding and cached:                 # a run after a cut: foreground requests waiting go first
                    release()
                    waited = time.perf_counter()
                    try:
                        turns.take(True, cancelled, **({"tick": pulse} if body.get("stream") else {}))
                        released = False                # ownership returns only after take succeeds
                    finally:
                        queued[0] += time.perf_counter() - waited
                extra = dict(options)
                if cached and "cache_prompt" in inspect.signature(self.engine.generate).parameters:
                    # Budget/call continuations are the SAME logical request.
                    # Do not replace its reusable original prompt checkpoints
                    # with a prompt containing generated/forced reply tokens.
                    extra["cache_prompt"] = False
                if stops.strings and "token_limit" in inspect.signature(self.engine.generate).parameters:
                    # A verified block can contain a stop before its final token.
                    # Resolve that boundary before any recurrent/KV commit. This
                    # pure worker callback owns neither output IO nor caller state;
                    # continuations include their already generated target prefix.
                    from tensorfold.cuda.reply_text import stop_prefix_limit

                    baseline = tuple(stops.tail_ids(ids, start=len(prompt)))
                    extra["token_limit"] = lambda new, prefix: stop_prefix_limit(stops, baseline, prefix, new)
                if prepared.grammar is not None:    # response_format: a fresh grammar state, after </think> when thinking
                    spec, compiled = prepared.grammar
                    # Reconstruct exactly this subrun's already committed reply,
                    # not merely whether a close occurred. Partial structured
                    # answers and forced tool prefixes must keep matcher state.
                    opener = (self.tok.token_to_id(strict.opener) if strict is not None and think_end is not None else None)
                    constraint = self._grammars().constraint(compiled, think_end=think_end, spec=spec,
                        decode=self.tok.decode,
                        **({"think_starts": native_starts, "thought_open": prepared.thought_open,
                            "stop_ids": self.engine.eos} if gemma and think_end is not None else {}),
                        **({"answer_start": opener, "answer_opener": strict.opener,
                            "answer_headers": strict.thinking_headers} if opener is not None else {}))
                    if grammar_state[0] is None:
                        grammar_state[0] = constraint.fresh_boundary()
                    if budget is not None:
                        budget.intent = grammar_state[0]
                    prior = ids[len(prompt):]
                    for index, token in enumerate(prior):
                        if index == forced_close[0]:
                            constraint.force_think_close(token)
                        else:
                            constraint.advance([token])
                    if prior and constraint.answer_offset == len(prior):
                        # A budget continuation starts at the empty answer root.
                        # Preserve that established active-root rank wire contract.
                        constraint = self._grammars().constraint(compiled, spec=spec)
                    extra["constraint"] = constraint
                if prepared.vision is not None:          # a gate's continuation keeps the images, positions extended
                    same = list(ids) == list(prepared.vision.token_ids)
                    if same:
                        extra["vision"] = prepared.vision
                    elif getattr(self.vision, "is_media_runtime", False):
                        extra["vision"] = self.vision.continued(prepared.vision, ids)
                    elif callable(getattr(self.vision, "continued", None)):
                        extra["vision"] = self.vision.continued(prepared.vision, ids)
                    elif hasattr(self.vision.frontend, "continued"):
                        extra["vision"] = self.vision.frontend.continued(prepared.vision, ids)
                    else:
                        from tensorfold.vision.qwen_processing import continued

                        extra["vision"] = continued(prepared.vision, ids, self.vision.frontend.config)
                if reports and not cached:              # the first run's prompt only: a later run's prompt is the reply's
                    extra["progress"] = filled
                if prepared.penalties is not None:      # a gate's continuation re-reads reply tokens as prompt: count them
                    extra["penalties"] = replace(prepared.penalties, prompt_length=len(prompt))
                stats = self.engine.generate(ids, count, sampling, feed, **extra)
                if not cached:
                    cached.append(int((stats or {}).get("cached") or 0))
                return stats

            # an engine that decodes concurrent requests together (``concurrent``) takes them as they come
            if turns is not None:
                waited = time.perf_counter()
                try:
                    turns.take(background, cancelled, **({"tick": pulse} if body.get("stream") else {}))
                except RequestCancelled:
                    # No engine turn was acquired, but this admitted request still has an outcome and queue time.
                    with health.of(self).running(len(prompt), out, arrived) as request:
                        request.stats = {"queue_s": round(queued[0] + time.perf_counter() - waited, 4)}
                        raise
                queued[0] += time.perf_counter() - waited
            released = turns is None

            def release() -> None:              # the engine's turn goes back once, as soon as generation ends
                nonlocal released
                if not released:
                    released = True
                    turns.give()

            try:
                # running until its reply is made, then counted with how it ended; /health reads ``out``; rounds never call in
                with health.of(self).running(len(prompt), out, arrived) as request:
                    generation_primary = None
                    try:
                        if cancelled is not None and cancelled():            # the client left while this request waited
                            raise RequestCancelled("the client left before the request started")
                        serving[0] = request
                        try:
                            def forced(index):
                                if forced_close[0] is not None:
                                    raise RuntimeError("a logical reply cannot force its thinking close twice")
                                forced_close[0] = index
                            stats = request.stats = generate_gated(generate, prompt, max_tokens, gates, on_tokens,
                                                                    on_think_close=forced)
                        finally:
                            serving[0] = None
                    except BaseException as generation_error:
                        generation_primary = generation_error
                        raise
                    finally:
                        callback_primary = failed[0] if failed[0] is not None else generation_primary
                        _callback_native_cause = (BaseException.__cause__.__get__(callback_primary)
                                                  if callback_primary is not None else None)
                        _callback_native_context = (BaseException.__context__.__get__(callback_primary)
                                                    if callback_primary is not None else None)
                        try:
                            release()
                            request.stats = {**(request.stats or {}),
                                             **timings(request.stats or {}, len(out), queued[0], prepare_s, arrived,
                                                       request.first)}
                        except BaseException as bookkeeping:
                            if callback_primary is not None and bookkeeping is not callback_primary:
                                raise callback_primary from bookkeeping
                            raise
                    if failed[0] is not None:
                        raise failed[0]
                    if stopped["client"]:                                        # as the Mac server: nothing more is written
                        raise RequestCancelled("the client left during the reply")
                    reasoning, answer = visible(True)
                    if calls_stream is not None and strict is None and normal_tool_end():
                        # Only a normal actual terminal resolves malformed-JSON
                        # raw fallback. Never finalize on cancel/error/reply cap
                        # or a user stop that can cut a still-valid call short.
                        for call_delta in calls_stream.feed(answer_raw[0], finished=True):
                            if not emit(call_delta):
                                stopped["client"] = True
                                raise RequestCancelled("the client left during the terminal tool reply")
                    final: dict[str, Any] = {}
                    if len(reasoning) > sent["reasoning"]:
                        final["reasoning_content"] = reasoning[sent["reasoning"]:]
                    from tensorfold.token_metadata import decode_presentation

                    raw_text = decode_presentation(self.tok, [t for t in out if t not in ends])
                    text = stops.visible(raw_text)
                    calls_complete = True
                    if strict is not None:          # a strict reply's calls follow its grammar exactly: read as written
                        raw_answer = answer_raw[0] if prepared.grammar is not None else text
                        content, calls, calls_complete = strict.parse_finished(raw_answer, max_calls=policy.max_calls)
                    elif tools and prepared.grammar is None:
                        if not gemma:
                            text = normalize_qwen_tool_markup(text, finished=normal_tool_end())
                        if split_reasoning and gemma:
                            from tensorfold.server.gemma_values import split_gemma_thinking

                            raw_answer = split_gemma_thinking(text, finished=True, opened=prepared.thought_open,
                                                              expect_thinking=thinking)[1]
                        else:
                            raw_answer = split_thinking(text, finished=True, markers=markers)[1] if split_reasoning else text
                        parse = parse_tool_calls_from_content if gemma else parse_tool_calls
                        content, calls = parse(raw_answer, tools, max_calls=policy.max_calls, finished=normal_tool_end())
                        calls_complete = tool_calls_complete(raw_answer, tools, finished=normal_tool_end())
                        shown = policy.content(content)
                        # a one-call request whose only call could not be parsed shows its markup rather than an empty reply
                        content = shown if calls or shown.strip() else content
                    else:
                        content, calls = answer, None
                    if tools and (prepared.grammar is None or strict is not None):
                        if not calls and not content and answer_raw[0].isspace():
                            content = answer_raw[0]  # restore held model-only whitespace after no-call proof
                        content = visible_tool_prose(content, calls=bool(calls))
                    # the calls already sent as deltas; the handler sends the rest (a call the streamer could not follow)
                    streamed = calls_stream.index + 1 if calls_stream is not None and calls_stream.streamed else 0
                    if streamed > len(calls or ()):
                        content = hide_tool_calls(content, finished=True).strip()   # those calls are the streamed deltas
                    sent_text = answer[:sent["content"]]
                    lead = sent_text if content.startswith(sent_text) else sent_text.lstrip()
                    tail = content[len(lead):] if content.startswith(lead) else ""
                    if tail:
                        final["content"] = tail
                    ended = stopped["stop"] or bool(out and out[-1] in ends)
                    # An emitted head is not a completed strict call. Retain all
                    # prior calls/partial deltas, but report the actual stop/budget
                    # outcome when any opened strict call remained unfinished.
                    stream_complete = calls_stream is None or strict is not None or calls_stream.complete
                    finished_calls = bool(calls or (streamed and ended)) and calls_complete and stream_complete
                    finish = request.outcome = "tool_calls" if finished_calls else ("stop" if ended else "length")
            finally:
                release()
            stats = {"stats_version": STATS_VERSION, "request_id": request_id, **(request.stats or {}),
                     "token_sha": token_sha(out)}
            warning = unanswered(finish, chat and thinking, content, calls,
                                 can_disable=self._fixed_thinking(tools) is not True)
            if warning:
                print(warning, flush=True)
            self._ended_thinking(request_id, body, prompt, out, finish, chat and thinking, content, calls, reasoning,
                                 raw_text, stopped["stop"], ends, stats)
            print_done(request_id, len(prompt), (cached or [0])[0], thinking, out, finish, stats, request)
            if body.get("return_token_ids"):              # the reply's ids in the "tensorfold" block, for exactness checks
                stats = {**(stats or {}), "token_ids": [int(t) for t in out]}
            logprobs = (self._probability_decoder.format(probabilities.emitted(out), ends)
                        if probabilities is not None else None)
            return {"final": final, "calls": calls, "finish": finish, "content": content, "reasoning": reasoning,
                    "stop_sequence": matched_stop(raw_text, stops.strings),
                    **({"logprobs": logprobs} if logprobs is not None else {}),
                    "prompt_tokens": len(prompt), "completion_tokens": len(out), "cached_tokens": (cached or [0])[0],
                    "reasoning_tokens": (min(len(out), grammar_state[0].answer_offset)
                        if chat and thinking and grammar_state[0] is not None and grammar_state[0].answer_offset is not None
                        else 0 if gemma and grammar_state[0] is not None and not grammar_state[0].opened_thought
                        else reasoning_count(out, self.tok.token_to_id(markers[1]) if chat and thinking else None)),
                    "stats": stats, "calls_streamed": streamed}
        except BaseException as error:
            _media_primary = error
            raise
        finally:
            if getattr(prepared, "_media_release", None) is not None:
                if _media_primary is not None:
                    from tensorfold.cleanup import rollback

                    rollback(prepared, _media_primary, prepared.release_media)
                else:
                    prepared.release_media()

    def _ended_thinking(self, request_id: str, body: dict[str, Any], prompt: list[int], out: list[int], finish: str,
                        thinking: bool, content: str, calls: Any, reasoning: str, raw_text: str, stop_string: bool,
                        ends: tuple[int, ...], stats: dict[str, Any]) -> None:
        """A reply that ended at an end token inside its think block (no answer, no call): one warning line, and with
        TENSORFOLD_FAILURE_CAPTURE_DIR set the request, its token ids and stats captured (``failure_capture``). The
        reply is not changed."""

        end = out[-1] if out and out[-1] in ends and not stop_string else None
        if not thinking or end is None or finish != "stop":         # the common replies: nothing to look up
            return
        marker = self.thinking_markers[1]
        close = self.tok.token_to_id(marker)
        closed = close in out if close is not None else marker in raw_text
        if not ended_thinking(finish, thinking, content, calls, reasoning, True, closed):
            return
        text = self.tok.id_to_token(end)
        print(ended_thinking_note(request_id, len(prompt), len(out), end, text), flush=True)
        if failure_capture.directory() is None:                    # capture off: no record is built
            return
        failure_capture.capture("ended_in_think", request_id, body, {
            "finish": finish, "prompt_tokens": len(prompt), "end_token": {"id": int(end), "text": text},
            "token_ids": [int(t) for t in out], "stats": stats})

    def _turns(self) -> Turns:
        """The engine's turns (one request at a time, background ones last), made on first use."""

        with _MADE:
            return self.__dict__.setdefault("turns", Turns())

    def _think_budget(self, prepared: PreparedRequest, think_end: int | None) -> ThinkBudget | None:
        """The Mac's thinking budget: the cut becomes a newline, </think> and a blank line (</think> alone under a grammar)."""

        if prepared.think_budget <= 0 or think_end is None:
            return None
        close = [*self.tok.encode("\n", add_special_tokens=False).ids, think_end]
        if prepared.grammar is None:
            close += self.tok.encode("\n\n", add_special_tokens=False).ids
        return ThinkBudget(prepared.think_budget, close, think_end, stop_ids=self.engine.eos,
                           decode=self.tok.decode)

    def _call_gate(self, prompt: list[int], tools: list[dict[str, Any]], *, thinking: bool = False) -> CallGate:
        """The gate a required tool call needs, from this template's call markup and the rendered prompt."""

        if not hasattr(self, "_form"):
            probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "tfprobe_fn", "arguments": {}}}]}]
            try:
                text = self.template.render(probe, tools=[{"type": "function", "function": {
                    "name": "tfprobe_fn", "parameters": {"type": "object", "properties": {}}}}],
                    enable_thinking=False)
            except Exception:  # noqa: BLE001 - a template that renders no calls: the opener alone
                text = ""
            openers = [o for o in ("<tool_call>", "<|tool_call>") if self.tok.token_to_id(o) is not None]
            self._form = call_format(text, "tfprobe_fn", openers) or ((openers[0], None, None) if openers else None)
        if self._form is None:
            raise RequestError('tool_choice "required" or a named function needs a chat template that marks tool calls '
                               '(<tool_call> or <|tool_call>), and this one does not: send "auto"')
        opener, lead, tail = self._form
        eos = set(self.engine.eos)

        def text(token: int) -> str:
            return self.tok.decode([token], skip_special_tokens=False)

        def blank(token: int) -> bool:
            return token not in eos and not text(token).strip()

        opens, closes = self.thinking_markers
        opening = self.tok.encode(opens, add_special_tokens=False).ids if opens else ()
        think = [opening[0] if opening else self.tok.token_to_id("<think>"), self.tok.token_to_id(closes)]
        think = [-1 if token is None else token for token in think]
        names = [str((t.get("function") or t).get("name") or "") for t in tools] if lead is not None else []
        return CallGate.after_prompt(prompt, self.tok.token_to_id(opener), blank, think_open=think[0],
                                     thinking=thinking, decode=self.tok.decode,
                                     think_end=think[1], text=text, lead=lead or "", names=names, tail=tail or "",
                                     encode=lambda t: list(self.tok.encode(t, add_special_tokens=False).ids))


def token_sha(tokens: list[int]) -> str:
    """A reply's token ids, hashed as the Mac server does: drafted and ``"draft": false`` replies must match."""

    return hashlib.sha256(",".join(str(int(t)) for t in tokens).encode()).hexdigest()[:12]


STATS_VERSION = 2          # the ``tensorfold`` block's keys: 2 adds request_id, queue_s, prepare_s, ttft_s and decode_tps


def timings(stats: dict[str, Any], tokens: int, queued: float, prepare_s: float, arrived: float,
            first: float | None) -> dict[str, float]:
    """A reply's ``tensorfold`` timings past its engine's own: ``queue_s`` (waiting for a lane or the engine's turn,
    and for admission), ``prepare_s`` (arrival to a prepared prompt: template, tokens, images), ``ttft_s`` (arrival to
    the first generated token) and ``decode_tps`` (tokens after the first over the engine's ``decode_s``). A time the
    request did not have (no token, no decode time) is left out."""

    engine_queue = stats.get("queue_s")
    out = {"queue_s": round(queued + (engine_queue if isinstance(engine_queue, (int, float)) else 0.0), 4),
           "prepare_s": round(max(0.0, prepare_s), 4)}
    if first is not None:
        out["ttft_s"] = round(max(0.0, first - arrived), 4)
    decode = stats.get("decode_s")
    if isinstance(decode, (int, float)) and not isinstance(decode, bool) and decode > 0 and tokens > 1:
        out["decode_tps"] = round((tokens - 1) / decode, 2)
    return out


def print_done(request_id: str, prompt: int, cached: int, thinking: bool, out: list[int], finish: str,
               stats: dict[str, Any], request: Any) -> None:
    """The Mac server's ``done`` line for a finished reply, named by its request id (its ``x-request-id``); tok/s runs
    from the first token to the last, ttft from the request's arrival."""

    ended = time.perf_counter()
    first, started = getattr(request, "first", None), getattr(request, "started", ended)
    decode = ended - first if first is not None else 0.0
    rate = (len(out) - 1) / decode if decode > 0 and len(out) > 1 else 0.0
    print(f"[tensorfold] done {request_id} prompt={prompt} cached={cached} thinking={thinking} "
          f"tokens={len(out)} sha={token_sha(out)} finish={finish} tok/s={rate:.1f} "
          f"ttft={(first - started) if first is not None else -1:.2f}s prefill={stats.get('prefill_s', -1):.2f}s "
          f"rounds={stats.get('rounds', 0)} accepted={stats.get('accepted', 0)}/{stats.get('drafted', 0)}",
          flush=True)


from tensorfold.cuda.http import Server, make_handler, serve, usage_of  # noqa: E402,F401  (the HTTP side)
