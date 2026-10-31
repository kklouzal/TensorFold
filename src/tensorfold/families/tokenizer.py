"""mlx_lm's tokenizer for checkpoints whose per-layer lists also count MTP layers (retry trims those lists)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tensorfold.chat_metadata import MissingChatTemplate, resolve_chat_templates
from tensorfold.checkpoint_metadata import read_config
from tensorfold.token_metadata import resolve_eos, token_ids


def trimmed_config(model_dir: Path) -> Any | None:
    """A transformers config with per-layer lists cut to the decoder's layers, or None when none is longer."""

    raw = read_config(model_dir)
    text = raw.get("text_config") or raw
    layers = int(text.get("num_hidden_layers") or 0)
    long = [k for k, v in text.items() if k.endswith("layer_types") and isinstance(v, list) and len(v) > layers > 0]
    if not long:
        return None
    for key in long:
        text[key] = text[key][:layers]
    from transformers import AutoConfig

    return AutoConfig.for_model(raw["model_type"], **{k: v for k, v in raw.items() if k != "model_type"})


def load_tokenizer(model_dir: Path, eos_token_ids: Any = None) -> Any:
    """``mlx_lm.utils.load_tokenizer``, retried with ``trimmed_config`` when transformers refuses the layer lists."""

    from mlx_lm.utils import load_tokenizer as load

    try:
        return load(Path(model_dir), tokenizer_config_extra={"trust_remote_code": False}, eos_token_ids=eos_token_ids)
    except Exception as error:
        if "layer_types" not in str(error):
            raise
        config = trimmed_config(Path(model_dir))
        if config is None:
            raise
        print("[tensorfold] config.json lists a layer type for each MTP layer too; the tokenizer reads it without "
              "them", flush=True)
        return load(Path(model_dir), tokenizer_config_extra={"config": config, "trust_remote_code": False},
                    eos_token_ids=eos_token_ids)


def apply_model_metadata(tokenizer, model_dir, *, eos_token_ids=None, chat_template_file=None, chat_template_name=None):
    """Apply captured defaults/overrides before any MLX planner, decoder or App.

    mlx-lm's supported public EOS setter replaces its stop set; its public
    chat_template setter forwards to HF. Owned GLM wrappers delegate rendering
    through _inner, so update that actual receiver, never a shadow template.
    Custom callable templates remain authoritative unless an override is
    requested, which is refused when the HF setter cannot affect rendering.
    """
    wrappers, receiver = [], tokenizer
    while True:
        if any(receiver is previous for previous in wrappers):
            raise ValueError("tokenizer wrapper ownership cycle")
        wrappers.append(receiver)
        inner = vars(receiver).get("_inner")
        if inner is None:
            break
        receiver = inner
    explicit_template = chat_template_file is not None or chat_template_name is not None
    custom = vars(receiver).get("_chat_template")
    if custom is not None and explicit_template:
        raise ValueError("this tokenizer's custom callable template does not support chat template overrides")
    sources = None
    if custom is None:
        try:
            sources = resolve_chat_templates(model_dir, template_file=chat_template_file,
                                              template_name=chat_template_name)
        except MissingChatTemplate:
            if explicit_template:
                raise
    raw = read_config(model_dir)
    text = raw.get("text_config") or raw
    vocab_size = text.get("vocab_size", raw.get("vocab_size"))
    current = getattr(tokenizer, "eos_token_ids", None)
    if current is None:
        current = getattr(tokenizer, "eos_token_id", None)
    values = () if current is None else current if isinstance(current, (list, tuple, set, frozenset)) else (current,)
    # mlx-lm's unset HF role may yield {None}; that is absence, not token zero.
    values = [value for value in values if value is not None]
    values = token_ids(values, field="configured tokenizer EOS", vocab_size=vocab_size)
    if isinstance(current, (set, frozenset)):
        values = tuple(sorted(values))
    current = values
    stops = resolve_eos(model_dir, raw.get("eos_token_id", text.get("eos_token_id")), vocab_size=vocab_size,
                        override=eos_token_ids, fallback=current, tokenizer=receiver)
    if eos_token_ids is None:
        stops = token_ids((*stops, *current), vocab_size=vocab_size)
    for wrapper in wrappers:
        wrapper.eos_token_ids = stops
    if sources is not None:
        default, tools = sources
        receiver.chat_template = (default.source if default is tools
                                  else {"default": default.source, "tool_use": tools.source})
        receiver.has_chat_template = True
