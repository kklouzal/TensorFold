"""Strict Qwen XML whose scalar booleans/null use Python's template spelling.

The model template writes containers as JSON and strings raw, but scalar values
as True/False/None. Only this exact captured profile is admitted. The standard
XML planner/reader remains unchanged; no lenient parameter parser participates.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from tensorfold.engine.grammar import Spec
from tensorfold.server import strict_tools as xml

_PROBED = xml._PROBED.replace(
    f"{xml.PARAMETER}tfprobe_bool{xml.NAME_END}true{xml.VALUE_CLOSE}",
    f"{xml.PARAMETER}tfprobe_bool{xml.NAME_END}True{xml.VALUE_CLOSE}").replace(
    f"{xml.PARAMETER}tfprobe_null{xml.NAME_END}null{xml.VALUE_CLOSE}",
    f"{xml.PARAMETER}tfprobe_null{xml.NAME_END}None{xml.VALUE_CLOSE}")
_WORDS = ((True, "True"), (False, "False"), (None, "None"))
_PROBED_FALSE = _PROBED.replace(
    f"{xml.PARAMETER}tfprobe_bool{xml.NAME_END}True{xml.VALUE_CLOSE}",
    f"{xml.PARAMETER}tfprobe_bool{xml.NAME_END}False{xml.VALUE_CLOSE}")


def false_probe():
    """The owned fixed probe with False, leaving legacy probe data unchanged."""
    messages = deepcopy(xml.PROBE)
    messages[1]["tool_calls"][0]["function"]["arguments"]["tfprobe_bool"] = False
    return messages


def template_supported(rendered: str, rendered_false: str) -> bool:
    """Recognize both scalar booleans/null and unchanged nested canonical JSON."""
    return _PROBED in rendered and _PROBED_FALSE in rendered_false


def _may_spell_none(format):
    """Conservative raw-language collision check; unknown regex cases refuse.

    The existing regex analysis proves absence only when at least one required
    character cannot occur. It never guesses that a nullable raw value is safe.
    """
    kind = format["type"]
    if kind == "const_string":
        return format["value"] == "None"
    if kind == "or":
        return any(_may_spell_none(part) for part in format["elements"])
    if kind == "any_text":
        return not any(word in "None" for word in format.get("excludes", ()))
    if kind == "regex":
        tree = xml._regex(format["pattern"], "tools")
        return all(xml._may_match(tree, char) for char in "None")
    raise RuntimeError("unexpected checked raw parameter format")


def _scalar_format(schema, node, where):
    """Split only root alternatives; container/ref contents keep canonical JSON."""
    formats, copied, done = [], 0, set()
    pending = [(node, where)]
    while pending:
        current, at = pending.pop()
        if id(current) in done:
            continue
        done.add(id(current))
        if "$ref" in current:
            pending.append((schema.resolve(current["$ref"], at), at))
            continue
        if "anyOf" in current:
            pending.extend((part, f"{at}.anyOf[{index}]")
                           for index, part in reversed(list(enumerate(current["anyOf"]))))
            continue
        kinds = schema.types(current)
        for value, word in _WORDS:
            if xml._value_type(value) not in kinds:
                continue
            literal = xml._canonical(value)
            if "const" in current and xml._canonical(current["const"]) != literal:
                continue
            if "enum" in current and literal not in {xml._canonical(item) for item in current["enum"]}:
                continue
            formats.append(xml._const(word))
        remaining = kinds - {"boolean", "null"}
        if not remaining:
            continue
        if "string" in remaining:
            raise RuntimeError("ambiguous string/non-string schema escaped strict admission")
        own = dict(current)
        if "enum" in own:
            own["enum"] = [value for value in own["enum"] if value is not None and type(value) is not bool]
            if not own["enum"]:
                continue
        if "const" in own and (own["const"] is None or type(own["const"]) is bool):
            continue
        # number already includes integer; avoid introducing a redundant union
        # beside validated single-type numeric constraints.
        remaining = remaining - {"integer"} if "number" in remaining else remaining
        own["type"] = next(iter(remaining)) if len(remaining) == 1 else sorted(remaining)
        checked, copies = schema.json(own, at)
        copied += copies
        if copied > xml.MAX_GRAMMAR_BYTES:
            raise xml._too_large()
        formats.append(schema.json_format(checked, at))
    if not formats:
        raise xml._refuse(where, "no finite value satisfies this scalar parameter schema")
    return formats[0] if len(formats) == 1 else {"type": "or", "elements": formats}, copied


@dataclass(frozen=True)
class ScalarParam(xml.Param):
    """A prepared parameter whose top-level scalar spelling needs conversion."""


class Plan(xml.Plan):
    """The exact Python-scalar XML profile; the normal Plan remains its default."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tag = json.loads(self.spec.text)
        # A supported empty grammar prefix names this distinct cache language
        # even for a string-only request; it adds no visible or token content.
        tag["format"] = xml._seq(xml._const(""), tag["format"])
        text = json.dumps(tag, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode()) > xml.MAX_GRAMMAR_BYTES:
            raise xml._too_large()
        self.spec = Spec("tools", text, "tools")

    @staticmethod
    def _strict(name, parameters, where):
        tool, content, copied = xml.Plan._strict(name, parameters, where)
        top = {"type": "object", **{key: value for key, value in parameters.items() if key != "$schema"}}
        schema = xml._Schema(top, where)
        schema.check()
        params = list(tool.params)
        for index, param in enumerate(params):
            sub = top["properties"][param.name]
            at = f"{where}.properties.{param.name}"
            part = content["elements"][index]
            part = part["content"] if part["type"] == "optional" else part
            if param.mode == xml.NULLABLE:
                raw = schema.raw(sub, at)
                if _may_spell_none(raw):
                    raise xml._refuse(at, "nullable raw strings may spell None, which this template also writes for "
                                      "null: use an unambiguous enum/pattern or a non-nullable string")
                part["elements"][1] = {"type": "or", "elements": [raw, xml._const("None")]}
            elif param.mode == xml.JSON and {"boolean", "null"} & schema.types(sub):
                _old, old_copies = schema.json(sub, at)
                value, new_copies = _scalar_format(schema, sub, at)
                part["elements"][1] = value
                copied += new_copies - old_copies
            else:
                continue
            params[index] = ScalarParam(param.name, param.required, param.mode)
        return xml.Tool(tool.name, tool.strict, tuple(params)), content, copied

    def reader(self):
        return Reader(self)

    def parse(self, answer, *, max_calls=None):
        reader = self.reader()
        reader.read(answer, finished=True)
        calls = reader.calls if max_calls is None else reader.calls[:max_calls]
        return "".join(reader.text).strip(), calls or None

    def content(self, answer, *, finished):
        reader = self.reader()
        reader.read(answer, finished=finished)
        return "".join(reader.text)


class Reader(xml.Reader):
    """Canonicalize only grammar-admitted top-level scalar value fragments."""
    def _emit(self, fragment):
        if self._state == "value" and isinstance(self._param, ScalarParam):
            if self._param.mode == xml.JSON:
                if fragment.strip() in ("true", "false", "null"):
                    raise xml.CallMarkupError("the reply used JSON scalar spelling outside its template profile")
                if fragment == "True":
                    fragment = "true"
                elif fragment == "False":
                    fragment = "false"
                elif fragment == "None":
                    fragment = "null"
            elif self._param.mode == xml.NULLABLE:
                # The raw string None was refused above. In this profile raw
                # null is a string, while the scalar None denotes JSON null.
                if fragment == '"None"':
                    fragment = "null"
                elif fragment == "null":
                    fragment = '"null"'
        super()._emit(fragment)
