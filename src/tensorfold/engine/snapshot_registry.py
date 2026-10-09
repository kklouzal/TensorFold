"""Explicit current-model cache-restore authority; standard library only.

The owner supplies initialized current-model cache prototypes and exact tensor
schemas derived from the family/runtime contract. No stored string imports or
constructs code. Metadata validation finishes before a tensor loader is called.
Families bind declared geometry before restoration. A caller must supply
this explicit current-model authority; disk class names never select code.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
from typing import Any

from tensorfold.engine.snapshot_integrity import VerificationMemo


@dataclass(frozen=True)
class TensorSchema:
    """Fixed axes are exact; None axes vary within the explicit owner bound."""

    dtypes: tuple[str, ...]
    shape: tuple[int | None, ...]
    max_elements: int

    def validate(self, tensor):
        if (
            type(tensor) is not dict
            or tensor.get("dtype") not in self.dtypes
            or type(tensor.get("shape")) is not list
            or len(tensor["shape"]) != len(self.shape)
        ):
            raise ValueError("cache tensor dtype/rank differs from trusted schema")
        for actual, expected in zip(tensor["shape"], self.shape):
            if type(actual) is not int or actual < 0 or (expected is not None and actual != expected):
                raise ValueError("cache tensor axis differs from trusted schema")
        if math.prod(tensor["shape"]) > self.max_elements:
            raise ValueError("cache tensor exceeds trusted allocation bound")


@dataclass(frozen=True)
class LayerSchema:
    prototype: Any
    tensors: dict[str, TensorSchema]
    mutable_integers: dict[str, tuple[int, int]]
    # Exact source-declared list capacities also allow a None→owned-list field,
    # e.g. the Qwen MTP head's three side buffers. No metadata derives a size.
    list_lengths: dict[str, int] = field(default_factory=dict)
    required_list_slots: dict[str, frozenset[int]] = field(default_factory=dict)
    numpy_fields: frozenset[str] = frozenset()
    mutable_flags: frozenset[str] = frozenset()
    required_tensors: frozenset[str] = frozenset()
    invariants: tuple = ()
    token_invariants: tuple = ()

    @property
    def class_id(self):
        cls = type(self.prototype)
        return f"{cls.__module__}:{cls.__qualname__}"

    def fields(self):
        transient = set(getattr(type(self.prototype), "transient", ()))
        return {name: value for name, value in vars(self.prototype).items() if name not in transient}


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate cache metadata key")
        result[name] = value
    return result


def _bad_constant(value):
    raise ValueError("nonfinite cache metadata number")


def _parse(text):
    if type(text) is not str:
        raise ValueError("cache metadata must contain JSON strings")
    try:
        value = json.loads(text, object_pairs_hook=_object, parse_constant=_bad_constant)
    except RecursionError as error:
        raise ValueError("cache metadata nesting exceeds the JSON parser limit") from error
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) in (list, dict):
            pending.extend(item.values() if type(item) is dict else item)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("nonfinite cache metadata number")
        elif type(item) is str:
            item.encode("utf-8")
    return value


def _capture(prototype):
    fields = vars(prototype)
    if type(fields) is not dict:
        raise ValueError("trusted cache needs an owned attribute mapping")
    return {
        name: (list, tuple(value)) if type(value) is list else (type(value), value) for name, value in fields.items()
    }


def _unchanged(fields, before):
    if type(fields) is not dict or set(fields) != set(before):
        return False
    for name, (kind, value) in before.items():
        got = fields[name]
        if type(got) is not kind:
            return False
        if kind is list:
            if len(got) != len(value) or any(a is not b for a, b in zip(got, value)):
                return False
        elif kind in (type(None), bool, int, float, str):
            if got.hex() != value.hex() if kind is float else got != value:
                return False
        elif got is not value:
            return False
    return True


def _clone(prototype):
    """Capture authority before callbacks; detect mutation, never undo code."""
    original = vars(prototype)
    before = _capture(prototype)
    clone = copy.copy(prototype)
    if vars(prototype) is not original or not _unchanged(vars(prototype), before):
        raise ValueError("cache copy callback mutated authority; prototype must be retired")
    if clone is prototype or type(clone) is not type(prototype):
        raise ValueError("cache clone must be a distinct object of the trusted class")
    copied = vars(clone)
    if copied is original or not _unchanged(copied, before):
        raise ValueError("cache clone must own complete unchanged constructor authority")
    for name, (kind, value) in before.items():
        if kind is list:
            copied[name] = list(value)
    return clone


class Registry:
    """Operation-owned trusted layer/schema tuple, captured before disk input."""

    def __init__(
        self,
        layers: tuple[LayerSchema, ...],
        *,
        token_limit: int,
        token_id_limit: int,
        tensor_byte_limit: int,
        sizes: dict[str, int],
    ):
        if (
            type(layers) is not tuple
            or not layers
            or any(type(s) is not LayerSchema for s in layers)
            or any(type(n) is not int or n < 0 for n in (token_limit, token_id_limit, tensor_byte_limit))
        ):
            raise ValueError("explicit bounded current-model restore authority required")
        self._unusable = True
        captured = []
        for schema in layers:
            prototype = _clone(schema.prototype)
            # Constructor-owned empty lists must not remain borrowed from the
            # caller across an independently versioned loader callback.
            for name, value in vars(prototype).items():
                if type(value) is list:
                    vars(prototype)[name] = list(value)
            captured.append(
                LayerSchema(
                    prototype,
                    dict(schema.tensors),
                    dict(schema.mutable_integers),
                    dict(schema.list_lengths),
                    {name: frozenset(slots) for name, slots in schema.required_list_slots.items()},
                    frozenset(schema.numpy_fields),
                    frozenset(schema.mutable_flags),
                    frozenset(schema.required_tensors),
                    tuple(schema.invariants),
                    tuple(schema.token_invariants),
                )
            )
        self.layers = tuple(captured)
        self.token_limit, self.token_id_limit, self.tensor_byte_limit = token_limit, token_id_limit, tensor_byte_limit
        self.sizes = dict(sizes)
        if not self.sizes or any(
            type(name) is not str or type(size) is not int or size <= 0 for name, size in self.sizes.items()
        ):
            raise ValueError("explicit positive tensor byte widths required")
        for schema in self.layers:
            fields = schema.fields()
            if (
                type(schema.invariants) is not tuple
                or type(schema.token_invariants) is not tuple
                or any(not callable(fn) for fn in (*schema.invariants, *schema.token_invariants))
            ):
                raise ValueError("trusted layer invariants require fixed callable authority")
            if not set(schema.mutable_integers) <= set(fields):
                raise ValueError("mutable integer is not an initialized trusted cache field")
            for name, bounds in schema.mutable_integers.items():
                if (
                    type(fields[name]) is not int
                    or type(bounds) is not tuple
                    or len(bounds) != 2
                    or any(type(n) is not int for n in bounds)
                    or bounds[0] > bounds[1]
                ):
                    raise ValueError("trusted mutable cache integer needs exact bounds")
            for name, length in schema.list_lengths.items():
                if (
                    name not in fields
                    or type(length) is not int
                    or length < 0
                    or (fields[name] is not None and (type(fields[name]) is not list or len(fields[name]) != length))
                ):
                    raise ValueError("declared cache list capacity differs from constructor authority")
            for name, slots in schema.required_list_slots.items():
                prototype_list = fields.get(name)
                length = schema.list_lengths.get(name, len(prototype_list) if type(prototype_list) is list else None)
                if length is None or any(type(slot) is not int or not 0 <= slot < length for slot in slots):
                    raise ValueError("required cache list slot lacks constructor capacity")
            for name in schema.mutable_flags:
                if name not in fields or type(fields[name]) is not bool:
                    raise ValueError("declared mutable cache flag must be initialized boolean")
            if (
                not schema.numpy_fields <= set(schema.tensors)
                or not schema.required_tensors <= set(schema.tensors)
                or any("." in name for name in schema.numpy_fields)
            ):
                raise ValueError("array representation/required fields must have tensor authority")
            for name, tensor in schema.tensors.items():
                if (
                    type(tensor) is not TensorSchema
                    or type(tensor.dtypes) is not tuple
                    or not tensor.dtypes
                    or type(tensor.shape) is not tuple
                    or type(tensor.max_elements) is not int
                    or tensor.max_elements < 0
                    or name.split(".")[0] not in fields
                    or any(n is not None and (type(n) is not int or n < 0) for n in tensor.shape)
                    or any(dtype not in self.sizes for dtype in tensor.dtypes)
                ):
                    raise ValueError("invalid trusted tensor field schema")

        self._unusable = False
        # A bounded optimization of immutable-file write suppression only;
        # native restoration always recomputes integrity while copying.
        self.integrity_memo = VerificationMemo()

    def validate(self, metadata, header, *, model_id: str):
        """Reject missing/foreign/oversized state before loader/import/restore."""
        if self._unusable:
            raise RuntimeError("cache registry was retired after a failed copy")
        if type(metadata) is not dict or metadata.get("format") != "2" or metadata.get("model") != model_id:
            raise ValueError("snapshot schema/model identity mismatch")
        tokens, layers = _parse(metadata.get("tokens")), _parse(metadata.get("layers"))
        if (
            type(tokens) is not list
            or len(tokens) > self.token_limit
            or any(type(t) is not int or not 0 <= t < self.token_id_limit for t in tokens)
            or type(layers) is not list
            or len(layers) != len(self.layers)
        ):
            raise ValueError("snapshot tokens/layer count differs from current model")
        wanted = set()
        total = 0
        for index, (entry, schema) in enumerate(zip(layers, self.layers)):
            if (
                type(entry) is not dict
                or set(entry) != {"class", "plain", "arrays", "numpy", "lists"}
                or entry["class"] != schema.class_id
                or type(entry["plain"]) is not dict
                or type(entry["arrays"]) is not list
                or type(entry["numpy"]) is not list
                or type(entry["lists"]) is not dict
            ):
                raise ValueError("snapshot layer differs from trusted class/schema")
            fields = schema.fields()
            seen = set()

            def field(name):
                if type(name) is not str or name not in fields or name in seen:
                    raise ValueError("unknown/duplicate cache field")
                seen.add(name)

            for name, value in entry["plain"].items():
                field(name)
                expected = fields[name]
                if name in schema.required_tensors:
                    raise ValueError("required cache tensor cannot be replaced by plain state")
                if name in schema.mutable_flags:
                    if type(value) is not bool:
                        raise ValueError("cache flag must be an exact boolean")
                elif name in schema.mutable_integers:
                    lower, upper = schema.mutable_integers[name]
                    if type(value) is not int or not lower <= value <= upper:
                        raise ValueError("cache integer exceeds trusted bounds")
                elif (
                    type(expected) not in (type(None), bool, int, float, str)
                    or type(value) is not type(expected)
                    or (value.hex() != expected.hex() if type(expected) is float else value != expected)
                ):
                    raise ValueError("cache constructor/configuration field differs")
            references = []
            for mode in ("arrays", "numpy"):
                for name in entry[mode]:
                    field(name)
                    if (name in schema.numpy_fields) != (mode == "numpy"):
                        raise ValueError("cache array storage representation differs from trusted schema")
                    references.append(name)
            for name, spec in entry["lists"].items():
                field(name)
                prototype = fields[name]
                length = schema.list_lengths.get(name, len(prototype) if type(prototype) is list else None)
                if (
                    length is None
                    or type(spec) is not dict
                    or set(spec) != {"length", "slots"}
                    or type(spec["length"]) is not int
                    or spec["length"] != length
                    or type(spec["slots"]) is not list
                    or len(spec["slots"]) > length
                    or any(type(n) is not int or not 0 <= n < length for n in spec["slots"])
                    or len(set(spec["slots"])) != len(spec["slots"])
                ):
                    raise ValueError("cache list exceeds trusted constructor capacity")
                if not schema.required_list_slots.get(name, frozenset()) <= set(spec["slots"]):
                    raise ValueError("cache list omitted required initialized state")
                references.extend(f"{name}.{slot}" for slot in spec["slots"])
            if not schema.required_tensors <= set(references):
                raise ValueError("snapshot omitted required initialized tensor state")
            if seen != set(fields):
                raise ValueError("snapshot omitted an initialized trusted field")
            for name in references:
                shape = schema.tensors.get(name)
                key = f"{index}.{name}"
                if shape is None or key not in header or key in wanted:
                    raise ValueError("unregistered/duplicate cache tensor reference")
                shape.validate(header[key])
                wanted.add(key)
                total += math.prod(header[key]["shape"]) * self.sizes[header[key]["dtype"]]
                if total > self.tensor_byte_limit:
                    raise ValueError("snapshot exceeds current memory admission")
            for check in schema.invariants:
                check(entry, header, index)
            for check in schema.token_invariants:
                check(entry, header, index, tokens)
        if wanted != set(header) - {"__metadata__"}:
            raise ValueError("snapshot payload differs from validated field references")
        return tokens, layers

    def restore(self, layers, tensors, *, convert_numpy):
        """Only validated state may reach this operation; initialized clones."""
        if self._unusable:
            raise RuntimeError("cache registry was retired after a failed copy")
        try:
            cache = []
            for index, (entry, schema) in enumerate(zip(layers, self.layers)):
                item = _clone(schema.prototype)
                values = dict(entry["plain"])
                for name in entry["arrays"]:
                    values[name] = tensors[f"{index}.{name}"]
                for name in entry["numpy"]:
                    values[name] = convert_numpy(tensors[f"{index}.{name}"])
                for name, spec in entry["lists"].items():
                    elements = [None] * spec["length"]
                    for slot in spec["slots"]:
                        elements[slot] = tensors[f"{index}.{name}.{slot}"]
                    values[name] = elements
                vars(item).update(values)
                cache.append(item)
            return cache
        except BaseException:
            self._unusable = True
            raise

    def load(self, metadata, header, *, model_id, tensor_loader, describe_tensor, convert_numpy, expected_tokens=None):
        """Whole restore boundary; foreign loader sees only validated metadata.

        The owner borrows an immutable regular file through this operation and
        provides the validated safetensors header before entry. ``tensor_loader``
        owns/evaluates all native reads and returns (arrays, identical metadata).
        Runtime tensor geometry is verified again before initialized restoration.
        """
        tokens, layers = self.validate(metadata, header, model_id=model_id)
        if expected_tokens is not None and tokens != list(expected_tokens):
            raise ValueError("snapshot tokens changed from the selected prefix")
        # Capture data before entering independently versioned callbacks.
        bound_metadata = copy.deepcopy(metadata)
        bound_layers = copy.deepcopy(layers)
        bound_tokens = tuple(tokens)
        bound_header = copy.deepcopy(header)
        tensors, loaded_metadata = tensor_loader()
        if (
            type(tensors) is not dict
            or loaded_metadata != bound_metadata
            or set(tensors) != set(bound_header) - {"__metadata__"}
        ):
            raise ValueError("native cache load differs from validated metadata")
        for name, tensor in tensors.items():
            if describe_tensor(tensor) != {"dtype": bound_header[name]["dtype"], "shape": bound_header[name]["shape"]}:
                raise ValueError("native cache tensor differs from validated geometry")
        cache = self.restore(bound_layers, tensors, convert_numpy=convert_numpy)
        return list(bound_tokens), cache
