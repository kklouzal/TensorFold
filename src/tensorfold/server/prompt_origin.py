"""Native Jinja values with output-origin metadata, never in-band input sentinels.

String operations use Python/Jinja's actual values. One immutable byte per
Unicode character records whether it came from request data. This is linear in
the bounded render size; no textual diff or guessed substring alignment decides
whether a token is trusted template syntax.
"""
from __future__ import annotations

import json
import re
from string import Formatter

MAX_ORIGIN_CHARS = 96 << 20


class PromptEncodingError(ValueError):
    """Native values cannot retain origin within the bounded render contract."""


def _plain(value):
    return str.__str__(value) if isinstance(value, str) else str(value)


class OriginText(str):
    __slots__ = ("_mask", "_data")

    def __new__(cls, value, unsafe=None, *, data=None):
        value = _plain(value)
        if len(value) > MAX_ORIGIN_CHARS:
            raise PromptEncodingError("chat string exceeds the render-work character budget")
        if unsafe is None:
            unsafe = b"\1" * len(value)
            if data is None:
                data = True
        if len(unsafe) != len(value):
            raise PromptEncodingError("prompt origin length disagrees with its text")
        obj = str.__new__(cls, value)
        object.__setattr__(obj, "_mask", bytes(unsafe))
        object.__setattr__(obj, "_data", any(unsafe) if data is None else bool(data))
        return obj

    def __setattr__(self, name, value):
        raise AttributeError("prompt origin is immutable")

    def __delattr__(self, name):
        raise AttributeError("prompt origin is immutable")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # Text and its immutable bytes have no mutable children or ownership
        # transfer. Native message-copying adapters can safely share this value.
        return self

    def __str__(self):
        # Jinja's normal str()/soft_str() conversion must not erase origin.
        return self

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        value = str.__getitem__(self, index)
        mask = self._mask[index]
        return OriginText(value, bytes((mask,)) if isinstance(mask, int) else mask,
                          data=self._data if not value else None)

    def __add__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return origin_concat((self, other))

    def __radd__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return origin_concat((other, self))

    def __mul__(self, count):
        if not isinstance(count, int):
            return NotImplemented
        if len(self) * max(0, count) > MAX_ORIGIN_CHARS:
            raise PromptEncodingError("chat string repetition exceeds the render-work character budget")
        return OriginText(str.__mul__(self, count), self._mask * count, data=self._data)

    __rmul__ = __mul__

    def _case(self, operation, unit):
        result = operation(_plain(self))
        # Python's language-independent full casing is codepoint-local except
        # final sigma, whose contextual substitution is still one codepoint.
        # Obtain the actual result ONCE; per-codepoint lengths only transport
        # origin through expansions such as sharp-s and dotted capital I.
        if len(result) > MAX_ORIGIN_CHARS:
            raise PromptEncodingError("chat string exceeds the render-work character budget")
        mask, at = bytearray(len(result)), 0
        for char, flag in zip(_plain(self), self._mask):
            size = len(unit(char))
            if at + size > len(mask):
                raise PromptEncodingError("unsupported contextual Unicode origin expansion")
            mask[at:at + size] = bytes((flag,)) * size
            at += size
        if at != len(result):
            raise PromptEncodingError("unsupported contextual Unicode origin expansion")
        return OriginText(result, mask, data=self._data)

    def upper(self):
        return self._case(str.upper, str.upper)

    def lower(self):
        return self._case(str.lower, str.lower)

    def capitalize(self):
        # Native capitalize titlecases only the first codepoint, then lowercases
        # the rest. The complete native value is computed once by _case; these
        # per-codepoint operations transport masks through Unicode expansions.
        first = True

        def unit(char):
            nonlocal first
            operation = str.title if first else str.lower
            first = False
            return operation(char)

        return self._case(str.capitalize, unit)

    def casefold(self):
        return self._case(str.casefold, str.casefold)

    def swapcase(self):
        return self._case(str.swapcase, str.swapcase)

    def replace(self, old, new, count=-1):
        # Delegate validation, including count's __index__ contract, to str.
        result = str.replace(self, old, new, count)
        if not result:
            return OriginText(result, b"", data=self._data or as_origin(new)._data)
        pieces, at, left = [], 0, count
        replacement = as_origin(new)
        if self._data or isinstance(old, OriginText) and old._data:
            # A replacement into request-derived text remains request data,
            # even if its spelling is supplied by a trusted template constant.
            replacement = OriginText(new)
        if old:
            while left != 0:
                end = str.find(self, old, at)
                if end < 0:
                    break
                pieces.extend((self[at:end], replacement))
                at = end + len(old)
                left -= 1
            pieces.append(self[at:])
        else:
            # Python inserts at character boundaries, including both ends.
            for i in range(len(self) + 1):
                if left != 0:
                    pieces.append(replacement)
                    left -= 1
                if i < len(self):
                    pieces.append(self[i])
        tracked = origin_concat(pieces)
        if tracked != result:
            raise PromptEncodingError("string replacement origin disagrees with native result")
        return tracked

    def strip(self, chars=None):
        left = len(self) - len(str.lstrip(self, chars))
        end = len(str.rstrip(self, chars))
        return self[left:max(left, end)]

    def lstrip(self, chars=None):
        return self[len(self) - len(str.lstrip(self, chars)):]

    def rstrip(self, chars=None):
        return self[:len(str.rstrip(self, chars))]

    def removeprefix(self, prefix):
        result = str.removeprefix(self, prefix)
        return self[len(self) - len(result):]

    def removesuffix(self, suffix):
        return self[:len(str.removesuffix(self, suffix))]

    def join(self, iterable):
        values = list(iterable)
        # str.join owns validation; metadata never converts a non-string item.
        result = str.join(self, values)
        pieces = []
        for i, value in enumerate(values):
            if i:
                pieces.append(self)
            pieces.append(as_origin(value))
        tracked = origin_concat(pieces)
        if tracked != result:
            raise PromptEncodingError("string join origin disagrees with native result")
        return tracked

    def split(self, sep=None, maxsplit=-1):
        values = str.split(self, sep, maxsplit)
        at, result = 0, []
        for value in values:
            start = str.find(self, value, at)
            result.append(self[start:start + len(value)])
            at = start + len(value) + (len(sep) if sep is not None else 0)
        return result

    def rsplit(self, sep=None, maxsplit=-1):
        values = str.rsplit(self, sep, maxsplit)
        end, result = len(self), []
        for value in reversed(values):
            start = str.rfind(self, value, 0, end)
            result.append(self[start:start + len(value)])
            end = start - (len(sep) if sep is not None else 0)
        return list(reversed(result))

    def __format__(self, spec):
        if not spec:
            return self
        # Native string formatting only truncates a prefix and pads it. The
        # library validates the mini-language and computes the actual result.
        shape = re.fullmatch(r"(?:(.)([<^>])|([<^>]))?(0)?(\d+)?(?:\.(\d+))?s?", spec, re.DOTALL)
        if shape is not None and any(int(v) > MAX_ORIGIN_CHARS for v in shape.groups()[-2:] if v):
            raise PromptEncodingError("format field exceeds the render-work character budget")
        result = str.__format__(self, spec)
        if shape is None:
            raise PromptEncodingError("native string format has no bounded origin mapping")
        fill, align, plain_align, zero, width, precision = shape.groups()
        align = align or plain_align or "<"
        fill = fill if fill is not None else "0" if zero else " "
        value = self if precision is None else self[:int(precision)]
        padding = max(0, int(width or 0) - len(value))
        left = padding if align == ">" else padding // 2 if align == "^" else 0
        padding_origin = OriginText(fill) if as_origin(spec)._data else trusted(fill)
        tracked = origin_concat((padding_origin * left, value, padding_origin * (padding - left)))
        if tracked != result:
            raise PromptEncodingError("string format origin disagrees with native result")
        return tracked

    def format(self, *args, **kwargs):
        return _OriginFormatter().vformat(self, args, kwargs)

    def format_map(self, mapping):
        return _OriginFormatter().vformat(self, (), mapping)

    def __mod__(self, values):
        return _percent_format(self, values)


def as_origin(value):
    return value if isinstance(value, OriginText) else OriginText(value)


def trusted(value):
    text = _plain(value)
    return OriginText(text, b"\0" * len(text), data=False)


def origin_concat(values):
    parts, masks, size, data = [], [], 0, False
    for value in values:
        value = as_origin(value)
        size += len(value)
        if size > MAX_ORIGIN_CHARS:
            raise PromptEncodingError("chat string exceeds the render-work character budget")
        parts.append(_plain(value))
        masks.append(value._mask)
        data |= value._data
    return OriginText("".join(parts), b"".join(masks), data=data)


class _OriginFormatter(Formatter):
    """Delegate field lookup/formatting/recursion to Python; retain its outputs.

    Each recursive format call owns its field records. Literal brace decoding
    only maps origin; it does not implement a second format-field evaluator.
    """
    def __init__(self):
        self._frames = []

    def format_field(self, value, spec):
        rendered = super().format_field(value, spec)
        if not isinstance(rendered, OriginText):
            rendered = as_origin(rendered)
        self._frames[-1].append(rendered)
        return rendered

    def parse(self, format_string):
        # Formatter.parse supplies the actual field grammar. Reconstruct its
        # exact source spelling only to locate each nested format-spec origin.
        source, at = as_origin(format_string), 0
        for literal, name, spec, conversion in super().parse(format_string):
            for char in literal:
                at += 2 if char in "{}" and source[at:at + 2] == char * 2 else 1
            if name is not None:
                head = "{" + name + ("!" + conversion if conversion else "")
                start = at + len(head)
                colon = source[start:start + 1] == ":"
                if colon:
                    start += 1
                spec_origin = source[start:start + len(spec)]
                at = start + len(spec) + 1
                yield literal, name, spec_origin, conversion
            else:
                yield literal, name, spec, conversion

    def _vformat(self, format_string, args, kwargs, used_args, recursion_depth, auto_arg_index=0):
        fields = []
        self._frames.append(fields)
        try:
            result, auto_arg_index = super()._vformat(format_string, args, kwargs, used_args,
                                                      recursion_depth, auto_arg_index)
        finally:
            self._frames.pop()
        source, parts, at, index = as_origin(format_string), [], 0, 0
        for literal, name, spec, conversion in self.parse(format_string):
            chars, masks = [], []
            for char in literal:
                length = 2 if char in "{}" and source[at:at + 2] == char * 2 else 1
                if source[at:at + length] != char * length:
                    raise PromptEncodingError("format literal origin disagrees with native parse")
                chars.append(char)
                masks.append(any(source._mask[at:at + length]))
                at += length
            parts.append(OriginText("".join(chars), bytes(masks), data=source._data if not chars else None))
            if name is not None:
                head = "{" + name + ("!" + conversion if conversion else "")
                colon = source[at + len(head):at + len(head) + 1] == ":"
                field = head + (":" + spec if colon else "") + "}"
                if source[at:at + len(field)] != field:
                    raise PromptEncodingError("format field origin disagrees with native parse")
                at += len(field)
                parts.append(OriginText(fields[index]) if source._data else fields[index])
                index += 1
        tracked = origin_concat(parts)
        if tracked != result or index != len(fields):
            raise PromptEncodingError("format output origin disagrees with native result")
        return OriginText(tracked, tracked._mask, data=tracked._data or source._data), auto_arg_index


def _percent_parts(source):
    """Locate old-style format fields; Python still evaluates every field."""
    at = 0
    while at < len(source):
        start = str.find(source, "%", at)
        if start < 0:
            yield source[at:], None, None, ((), 0, None), ""
            return
        literal, end, key = source[at:start], start + 1, None
        if end < len(source) and source[end] == "%":
            yield origin_concat((literal, source[start:start + 1])), None, None, ((), 0, None), ""
            at = end + 1
            continue
        if end < len(source) and source[end] == "(":
            depth, begin = 1, end + 1
            end += 1
            while end < len(source) and depth:
                depth += (source[end] == "(") - (source[end] == ")")
                end += 1
            key = str.__str__(source[begin:end - 1])
        flags_start = end
        while end < len(source) and source[end] in "#0- +":
            end += 1
        flags = str.__str__(source[flags_start:end])
        stars, width, precision = [], 0, None
        for position in ("width", "precision"):
            if position == "precision":
                if end >= len(source) or source[end] != ".":
                    break
                end += 1
            begin = end
            if end < len(source) and source[end] == "*":
                stars.append(position)
                end += 1
            else:
                while end < len(source) and source[end].isdecimal():
                    end += 1
                if end > begin and int(source[begin:end]) > MAX_ORIGIN_CHARS:
                    raise PromptEncodingError("format field exceeds the render-work character budget")
                amount = int(source[begin:end] or 0)
                if position == "width":
                    width = amount
                else:
                    precision = amount
        if end < len(source) and source[end] in "hlL":
            end += 1
        end += 1  # Native '%' owns invalid/missing conversion diagnostics.
        yield literal, source[start:end], key, (tuple(stars), width, precision), flags
        at = end


def _percent_format(source, values):
    parsed = list(_percent_parts(source))
    arguments = values if isinstance(values, tuple) else (values,)
    probe = 0
    for _, field, key, shape, _ in parsed:
        if field is None or key is not None:
            continue
        stars = shape[0]
        for position, amount in zip(stars, arguments[probe:probe + len(stars)]):
            if isinstance(amount, int):
                bound = abs(amount) if position == "width" else max(0, amount)
                if bound > MAX_ORIGIN_CHARS:
                    raise PromptEncodingError("format field exceeds the render-work character budget")
        probe += len(stars) + 1
    # Validation/evaluation/error behavior remains the native operation.
    result = str.__mod__(source, values)
    index, parts = 0, []
    for literal, field, key, shape, flags in parsed:
        parts.append(literal)
        if field is None:
            continue
        stars, width, precision = shape
        if key is not None:
            value, field_args = values[key], values
            star_values = ()
        else:
            star_values = arguments[index:index + len(stars)]
            index += len(stars)
            value = arguments[index]
            index += 1
            field_args = (*star_values, value)
        rendered = str.__mod__(field, field_args)
        conversion = field[-1:]
        if conversion == "s" and isinstance(value, OriginText):
            # '%s' pads/truncates a string exactly as its corresponding native
            # string format. Numeric conversions and repr/ascii are data.
            for position, amount in zip(stars, star_values):
                if position == "width":
                    width = abs(amount)
                    if amount < 0:
                        flags += "-"
                else:
                    precision = max(0, amount)
            spec = ("<" if "-" in flags else ">") + str(width)
            if precision is not None:
                spec += "." + str(precision)
            tracked = value.__format__(OriginText(spec) if field._data else trusted(spec))
            if field._data:
                tracked = OriginText(tracked)
            if tracked != rendered:
                raise PromptEncodingError("percent string origin disagrees with native result")
            parts.append(tracked)
        else:
            parts.append(OriginText(rendered))
    tracked = origin_concat(parts)
    if tracked != result:
        raise PromptEncodingError("percent output origin disagrees with native result")
    return OriginText(tracked, tracked._mask, data=tracked._data or source._data)


def request_values(value, depth=0):
    """Keep native collection/value semantics while attaching string origin."""
    if depth > 128:
        raise PromptEncodingError("chat values exceed the rendering nesting limit")
    if isinstance(value, str):
        return OriginText(value)
    if isinstance(value, dict):
        return {request_values(k, depth + 1): request_values(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [request_values(v, depth + 1) for v in value]
    return value


def trusted_values(value):
    """Model-owned token-role containers, before request kwargs override them."""
    if isinstance(value, str):
        return trusted(value)
    if isinstance(value, list):
        return [trusted_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(trusted_values(item) for item in value)
    if isinstance(value, dict):
        return {key: trusted_values(item) for key, item in value.items()}
    return value


def trust_message_roles(messages):
    """Only the normalized top-level role enum can authorize message framing."""
    from tensorfold.server.messages import MESSAGE_ROLES

    for message in messages:
        role = message.get("role") if isinstance(message, dict) else None
        if not isinstance(role, str) or role not in MESSAGE_ROLES:
            raise PromptEncodingError("message role is outside the validated role enum")
        message["role"] = trusted(role)
    return messages


def trust_policy_strings(values):
    """Closed resolved policy fields, never arbitrary extra strings/arguments."""
    from tensorfold.server.request_options import EFFORTS

    effort = values.get("reasoning_effort")
    if isinstance(effort, str) and effort in EFFORTS:
        values["reasoning_effort"] = trusted(effort)
    return values


def origin_environment(environment):
    """An owned overlay of the actual native Jinja environment, not a patch.

    Source constants are trusted; dynamic values with lost provenance are data.
    The normal compiler still owns expressions, branching, sandboxing and errors.
    """
    from jinja2 import nodes
    from jinja2.compiler import CodeGenerator
    from jinja2.idtracking import VAR_LOAD_ALIAS, VAR_LOAD_RESOLVE

    policy_names = frozenset(("enable_thinking", "add_generation_prompt"))
    string_filters = frozenset(("string", "upper", "lower", "trim", "replace", "capitalize", "title"))

    def scalar_piece(value, owned):
        if isinstance(value, OriginText):
            return value
        if owned and type(value) in (str, bool, int, float, type(None)):
            return trusted(str(value))
        return as_origin(value)

    class OriginCodeGenerator(CodeGenerator):
        @staticmethod
        def _binding(frame, reference):
            while frame is not None:
                flags = getattr(frame, "_prompt_flags", {})
                if reference in flags:
                    return flags[reference]
                frame = frame.parent
            return "False"

        def enter_frame(self, frame):
            super().enter_frame(frame)
            frame._prompt_flags = {}
            for reference, (action, name) in frame.symbols.loads.items():
                flag = self.temporary_identifier()
                if action == VAR_LOAD_RESOLVE and name in policy_names:
                    value = f"(type({reference}) is bool)"
                elif action == VAR_LOAD_ALIAS:
                    value = self._binding(frame.parent, name)
                else:
                    value = "False"
                self.writeline(f"{flag} = {value}")
                frame._prompt_flags[reference] = flag

        def _expression_origin(self, node, frame):
            if isinstance(node, nodes.Const):
                return "True"
            if isinstance(node, nodes.Name):
                return self._binding(frame, frame.symbols.ref(node.name))
            if isinstance(node, nodes.Not):
                return self._expression_origin(node.node, frame)
            if isinstance(node, (nodes.And, nodes.Or)):
                return (f"({self._expression_origin(node.left, frame)} and "
                        f"{self._expression_origin(node.right, frame)})")
            if isinstance(node, nodes.Filter) and node.name in string_filters and node.dyn_args is None and node.dyn_kwargs is None:
                values = [node.node, *node.args, *(keyword.value for keyword in node.kwargs)]
                return "(" + " and ".join(self._expression_origin(value, frame) for value in values) + ")"
            return "False"

        def visit_Assign(self, node, frame):
            # Metadata follows the actual lexical binding, not bool singleton
            # equality or a whitelisted variable spelling after shadowing.
            def updates(target, value):
                if isinstance(target, nodes.Name):
                    return [(target, self._expression_origin(value, frame) if value is not None else "False")]
                if isinstance(target, nodes.Tuple):
                    values = value.items if isinstance(value, (nodes.Tuple, nodes.List)) else [None] * len(target.items)
                    return [item for child, expr in zip(target.items, values) for item in updates(child, expr)]
                return []
            bindings = []
            for name, expression in updates(node.target, node.node):
                before = self.temporary_identifier()
                self.writeline(f"{before} = {expression}")
                bindings.append((name, before))
            super().visit_Assign(node, frame)
            for name, before in bindings:
                flag = self._binding(frame, frame.symbols.ref(name.name))
                if flag != "False":
                    self.writeline(f"{flag} = {before}")

        def visit_With(self, node, frame):
            # The native With generator assigns its bindings directly rather
            # than through visit_Assign. Carry the same sidecar at that boundary.
            inner = frame.inner()
            inner.symbols.analyze_node(node)
            self.enter_frame(inner)
            for target, expression in zip(node.targets, node.values):
                self.newline()
                self.visit(target, inner)
                self.write(" = ")
                self.visit(expression, frame)
                names = [target] if isinstance(target, nodes.Name) else list(target.find_all(nodes.Name))
                for name in names:
                    flag = self._binding(inner, inner.symbols.ref(name.name))
                    value = self._expression_origin(expression, frame) if isinstance(target, nodes.Name) else "False"
                    if flag != "False":
                        self.writeline(f"{flag} = {value}")
            self.blockvisit(node.body, inner)
            self.leave_frame(inner)

        def _output_const_repr(self, group):
            return f"environment._prompt_trusted({''.join(group)!r})"

        def visit_Const(self, node, frame):
            value = node.as_const(frame.eval_ctx)
            if isinstance(value, str):
                self.write(f"environment._prompt_trusted({value!r})")
            else:
                super().visit_Const(node, frame)

        def visit_Concat(self, node, frame):
            self.write("environment.concat((")
            for arg in node.nodes:
                self.write("environment._prompt_piece(")
                self.visit(arg, frame)
                self.write(f", {self._expression_origin(arg, frame)})")
                self.write(", ")
            self.write("))")

        def _output_child_pre(self, node, frame, finalize):
            self.write("environment._prompt_piece(")
            if finalize.src is not None:
                self.write(finalize.src)

        def _output_child_post(self, node, frame, finalize):
            if finalize.src is not None:
                self.write(")")
            self.write(f", {self._expression_origin(node, frame)})")

    env = environment.overlay()
    env.code_generator_class = OriginCodeGenerator
    env.concat = origin_concat
    env._prompt_trusted = trusted
    env._prompt_piece = scalar_piece
    env.filters = dict(environment.filters)
    native_join = env.filters["join"]
    from jinja2 import pass_eval_context
    default_delimiter = object()

    @pass_eval_context
    def join(context, value, delimiter=default_delimiter, attribute=None):
        separator = trusted("") if delimiter is default_delimiter else as_origin(delimiter)
        return native_join(context, value, separator, attribute)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # JSON data is literal even when it contains a trusted spelling. The
        # standard encoder owns all escaping/ordering/value semantics.
        return OriginText(json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                                     separators=separators, sort_keys=sort_keys))

    env.filters["join"] = join
    env.filters["tojson"] = tojson
    if hasattr(env, "wrap_str_format"):
        from functools import update_wrapper
        from jinja2.sandbox import SandboxedFormatter

        native_wrap = env.wrap_str_format

        def wrap_format(value):
            receiver = getattr(value, "__self__", None)
            name = getattr(value, "__name__", "")
            if not isinstance(receiver, OriginText) or name not in ("format", "format_map"):
                return native_wrap(value)

            def formatted(*args, **kwargs):
                if name == "format_map":
                    if kwargs or len(args) != 1:
                        raise TypeError("format_map requires exactly one mapping")
                    args, kwargs = (), args[0]
                formatter = _OriginFormatter()
                # Preserve the native sandbox's attribute/item authorization;
                # only its output-origin collection is an owned extension.
                formatter.get_field = SandboxedFormatter(env).get_field
                return formatter.vformat(receiver, args, kwargs)

            return update_wrapper(formatted, value)

        env.wrap_str_format = wrap_format
    return env
