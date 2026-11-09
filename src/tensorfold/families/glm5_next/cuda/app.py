"""GLM request routes validate context before streaming and render an empty think block without a reasoning-effort line when thinking is off."""

from __future__ import annotations

from typing import Any, Callable

from tensorfold.cuda.server import App, PreparedRequest, RequestError
from tensorfold.families.glm5_next.prompts import clear_thinking, thinking_off_prompt
from tensorfold.server.errors import CONTEXT_LIMIT
from tensorfold.server.prompt_literals import MAX_RENDER_BYTES, PromptEncodingError


class ThinkingOffTemplate:
    """The checkpoint template as GLM-5.3's thinking-off template renders it (``prompts.thinking_off``), with earlier
    turns' reasoning kept unless the request's ``chat_template_kwargs.clear_thinking`` says otherwise
    (``prompts.clear_thinking``)."""

    def __init__(self, inner, clear: bool | None = None) -> None:
        self.inner = inner
        self.efforts = getattr(inner, "efforts", frozenset())
        self.clear = clear_thinking() if clear is None else clear

    def render(self, messages, *, tools, enable_thinking, extra=None, allow_images=False, allow_audio=False,
               max_render_bytes=MAX_RENDER_BYTES, add_generation_prompt=True, render_credit=None) -> str:
        if extra is None or isinstance(extra, dict):
            extra = {"clear_thinking": self.clear, **(extra or {})}   # a request's own value wins
        rendered = self.inner.render(messages, tools=tools, enable_thinking=enable_thinking, extra=extra,
                                     allow_images=allow_images, allow_audio=allow_audio,
                                     max_render_bytes=max_render_bytes, add_generation_prompt=add_generation_prompt,
                                     render_credit=render_credit)
        if enable_thinking:
            return rendered
        try:
            return thinking_off_prompt(rendered, max_render_bytes=max_render_bytes, render_credit=render_credit)
        except (PromptEncodingError, UnicodeError) as exc:
            from jinja2.exceptions import TemplateError

            raise TemplateError(str(exc)) from exc

    def encode(self, tokenizer, text, *, segments=None, ids_credit=None, cache_credit=None):
        return self.inner.encode(tokenizer, text, segments=segments,
                                 ids_credit=ids_credit, cache_credit=cache_credit)

    def selected(self, *, tools=False):
        return self.inner.selected(tools=tools)

    def provenance(self, *, tools=False):
        return self.inner.provenance(tools=tools)


class GlmApp(App):
    reads_ignore_eos = True             # ``run`` hands it to the engine's request
    model_policies = True               # a served id may carry ``@POLICY`` (the recipe's draft policies)

    def __init__(self, engine, model_dir, served: str, **kwargs: Any) -> None:
        super().__init__(engine, model_dir, served, **kwargs)
        self.template = ThinkingOffTemplate(self.template)

    def check(self, body: dict[str, Any], *, prepared: PreparedRequest | None = None) -> str | None:
        """Validate the rendered prompt plus max_tokens against the engine context limit before streaming."""

        problem = self._check_fields(body)
        limit = getattr(self.engine, "limit", None)
        if problem or limit is None:
            return problem or super().check(body, prepared=prepared)
        if prepared is None:
            try:
                prepared = self._prepare(body, "messages" in body)
            except RequestError as exc:
                return str(exc)
        prompt = len(prepared.prompt)
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        need = prompt + (int(asked) if asked else 1)
        if need <= limit:
            return super().check(body, prepared=prepared)
        detail = f"{prompt} prompt tokens plus max_tokens {int(asked)}" if asked else f"a {prompt}-token prompt"
        # OpenAI's wording, so prepare refuses it as context_length_exceeded (clients compact on it)
        return (f"{CONTEXT_LIMIT} {limit} tokens: this request needs a {need}-token context ({detail}), which exceeds "
                f"the context window this server was started for; shorten the prompt or reply"
                f"{self._restart(need, ' both ranks')}")

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            prepared: PreparedRequest | None = None, **request: Any) -> dict[str, Any]:
        model = str(body.get("model") or "")
        self.engine.request.policy = body.get("tf_policy") or (model.split("@", 1)[1] if "@" in model else None)
        self.engine.request.stop_eos = not bool(body.get("ignore_eos", False))
        return super().run(body, chat, emit, prepared=prepared, **request)
