"""Explicit builtin type authority plus evaluated cache metadata, no SDK imports.

Only classes already installed by the current model's own loader are admitted.
Fixed module/class declarations refer to trusted loaded modules, never disk
metadata. Unknown classes require an explicit caller Registry. The profile is
startup-owned; it retains descriptors and fresh initialized prototypes, never
observed tensor storage. Complete family/native compatibility remains a gate.
"""

from __future__ import annotations

import math
import sys

from tensorfold.engine.snapshot_registry import LayerSchema, Registry, TensorSchema, _clone

# Exact supported runtime classes; this is source authority, not serialization.
# Values are (tensor path -> growing axis or None, initialized class defaults).
KV = {"keys": 2, "values": 2}
ATTENTION = {**KV, "index_keys": 1, "pooled": 1}
LAYER = {"keys": None, "pool": 0, "ipool": 0, "proj": None}
DECLARATIONS = {
    "mlx_lm.models.cache": {
        "KVCache": (KV, {}),
        "RotatingKVCache": (KV, {}),
        "ArraysCache": ({}, {}),
    },
    "tensorfold.engine.alternating_kv": {"AlternatingKVCache": (KV, {})},
    "tensorfold.families.gemma4.cache": {
        "LinearKVCache": (KV, {}),
        "RingKVCache": ({"ring_keys": None, "ring_values": None}, {}),
    },
    "tensorfold.families.nemotron_h.state_cache": {"RowStateCache": ({}, {})},
    "tensorfold.families.nemotron_h.mtp": {"MTPCache": (KV, {"drafted": 0})},
    "tensorfold.families.qwen4_exp.model_layers": {
        "LinearCache": ({"conv": None, "ssm": None, "ple_conv": None, "history": None}, {}),
        "AttentionCache": (ATTENTION, {}),
    },
    "tensorfold.families.qwen4_exp.mtp_cache": {
        "MTPCache": (
            {**ATTENTION, "side.0": 2, "side.1": 2, "side.2": 1},
            {"drafted": 0, "chaining": False, "side": None, "side_base": 0},
        ),
    },
    "tensorfold.families.glm5_next.caches": {
        "KDACache": ({"conv": None, "ssm": None}, {}),
        "MLACache": ({"keys": 0, "ik": 0, "ig": 0, "pool": 0}, {}),
    },
    "tensorfold.families.glm5_next.runtime": {
        "MTPCache": ({"keys": 0, "ik": 0, "ig": 0, "pool": 0}, {"drafted": 0}),
    },
    "tensorfold.families.deepseek_v4.caches": {"LayerCache": (LAYER, {})},
    "tensorfold.families.deepseek_v4.mtp": {"MTPCache": (LAYER, {"drafted": 0})},
}


def builtin_authorities():
    """Capture already loaded exact class objects; do not import any module."""
    result = {}
    for module_name, declarations in DECLARATIONS.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name, layout in declarations.items():
            cls = vars(module).get(name)
            if isinstance(cls, type) and cls.__module__ == module_name and cls.__qualname__ == name:
                result[cls] = layout
    return result


class UnregisteredCache(ValueError):
    """No explicit current-model persistence authority for this class."""


def _token_timeline(main_layer):
    """Main rows equal prefix tokens; auxiliary paired context may be shorter."""

    def check(entry, header, index, tokens):
        plain = entry["plain"]
        if "offset" not in plain:
            return
        offset = plain["offset"]
        if main_layer:
            if offset != len(tokens):
                raise ValueError("main cache offset differs from stored prefix tokens")
        elif not 0 <= offset - plain.get("drafted", 0) <= len(tokens):
            raise ValueError("auxiliary cache retained context exceeds stored prefix tokens")

    return check


def _invariants(prototype, paths, conditional_required=frozenset()):
    """Trusted source geometry relations, checked before tensor materialization."""
    rotating = type(prototype).__name__ == "RotatingKVCache"
    ratio = vars(prototype).get("ratio", 0)

    def check(entry, header, index):
        def tensor(name):
            return header.get(f"{index}.{name}")

        lengths = []
        for name in ("keys", "values", "index_keys", "ik", "ig"):
            if name in paths and paths[name] is not None and tensor(name) is not None:
                lengths.append(tensor(name)["shape"][paths[name]])
        if lengths and len(set(lengths)) != 1:
            raise ValueError("cache attention buffers disagree on source capacity")
        plain = entry["plain"]
        offset = plain.get("offset", 0)
        side = entry["lists"].get("side")
        if side is not None:
            side_lengths = [tensor(f"side.{i}")["shape"][paths[f"side.{i}"]] for i in range(3)]
            if len(set(side_lengths)) != 1 or offset - plain["side_base"] != side_lengths[0]:
                raise ValueError("MTP side buffers disagree on absolute timeline")
        required_offset = plain["side_base"] if side is not None else offset
        if conditional_required:
            present = {name for name in conditional_required if tensor(name) is not None}
            # Paired MTP context can be empty for a one-token prefix. A chain
            # can also live entirely in side buffers from absolute base zero.
            # Once retained main rows or any main allocation exist, preserve
            # the complete observed coallocation contract.
            if (required_offset > 0 or present) and present != conditional_required:
                raise ValueError("MTP main cache omits initialized coallocated tensors")
            if plain.get("drafted", 0) > offset:
                raise ValueError("MTP draft count exceeds its retained timeline")
        if lengths and not rotating and required_offset > lengths[0]:
            raise ValueError("cache offset exceeds retained attention capacity")
        if rotating and lengths:
            if offset >= plain["max_size"] and lengths[0] < plain["max_size"]:
                raise ValueError("rotated cache must retain its complete window capacity")
            if plain["_idx"] > lengths[0]:
                raise ValueError("rotating cache cursor exceeds retained capacity")
            if offset < plain["max_size"] and plain["_idx"] != offset:
                raise ValueError("unrotated cache cursor must match its absolute offset")
        if ratio:
            for name in ("pool", "ipool") if ratio == 4 else ("pool",):
                if offset // ratio and (tensor(name) is None or tensor(name)["shape"][0] < offset // ratio):
                    raise ValueError("compressed cache omits complete retained pools")
        elif "ik" in paths and lengths:
            pool = tensor("pool")
            if pool is None or pool["shape"][0] != lengths[0] // 4:
                raise ValueError("GLM index pool capacity must track its key allocation")

    return check


class Profile:
    """Capture native-free descriptors once from a complete evaluated prefix.

    The caller materializes RowStateCache before observation and describes only
    actual runtime arrays/NumPy arrays as {dtype,shape}; other values return None.
    Token, draft and byte budgets are explicit operation authority. A longer
    prefix may be needed to populate sparse pooled state; missing metadata is
    not guessed and required-array restoration remains refused until described.
    """

    def __init__(
        self,
        prototypes,
        *,
        describe_tensor,
        token_limit,
        token_id_limit,
        tensor_byte_limit,
        max_draft,
        sizes,
        authorities=None,
        main_layers=None,
    ):
        if (
            any(type(n) is not int or n < 0 for n in (token_limit, token_id_limit, tensor_byte_limit, max_draft))
            or token_id_limit == 0
        ):
            raise ValueError("current-model nonnegative persistence budgets required")
        self.describe = describe_tensor
        self.token_limit, self.token_id_limit = token_limit, token_id_limit
        self.tensor_byte_limit, self.max_draft = tensor_byte_limit, max_draft
        self.sizes = dict(sizes)
        bindings = builtin_authorities() if authorities is None else dict(authorities)
        self.prototypes, self.layouts = [], []
        for prototype in prototypes:
            if not getattr(prototype, "stored", True):
                continue
            cls = type(prototype)
            if cls not in bindings:
                raise UnregisteredCache("explicit initialized cache schema required for " + cls.__name__)
            paths, defaults = bindings[cls]
            owned = _clone(prototype)
            for name, value in defaults.items():
                if name not in vars(owned):
                    if type(getattr(owned, name)) is not type(value) or getattr(owned, name) != value:
                        raise ValueError("builtin initialized class default differs from source declaration")
                    vars(owned)[name] = value
            if cls.__name__ in ("ArraysCache", "RowStateCache"):
                name = "_cache" if cls.__name__ == "RowStateCache" else "cache"
                initial = vars(owned).get(name)
                if type(initial) is not list:
                    raise ValueError("builtin arrays cache needs constructor-owned slots")
                paths = {f"{name}.{slot}": None for slot in range(len(initial))}
            for name, value in vars(owned).items():
                if self.describe(value) is not None or (
                    type(value) is list and any(self.describe(v) is not None for v in value)
                ):
                    raise ValueError("fresh initialized prototypes must not retain evaluated tensor storage")
            if any(vars(owned).get(name, 0) != 0 for name in ("offset", "_idx", "drafted", "side_base")):
                raise ValueError("fresh initialized cache cursors required before startup observation")
            self.prototypes.append(owned)
            self.layouts.append(dict(paths))
        if not self.prototypes:
            raise UnregisteredCache("at least one registered stored cache layer required")
        if main_layers is not None and (type(main_layers) is not int or not 0 <= main_layers <= len(self.prototypes)):
            raise ValueError("explicit current-model main layer count required")
        self.main_layers = main_layers
        self.observed = None

    def observe(self, cache):
        rows = [item for item in cache if getattr(item, "stored", True)]
        if len(rows) != len(self.prototypes):
            raise ValueError("startup cache layer count differs from initialized authority")
        observed = []
        for item, prototype, paths in zip(rows, self.prototypes, self.layouts):
            if type(item) is not type(prototype):
                raise ValueError("startup cache exact class differs from authority")
            record = {}
            for path in paths:
                name, dot, slot = path.partition(".")
                value = vars(item).get(name)
                if dot:
                    value = value[int(slot)] if type(value) is list and int(slot) < len(value) else None
                shape = self.describe(value)
                if shape is not None:
                    if (
                        type(shape) is not dict
                        or set(shape) != {"dtype", "shape"}
                        or shape["dtype"] not in self.sizes
                        or type(shape["shape"]) not in (list, tuple)
                        or any(type(n) is not int or n < 0 for n in shape["shape"])
                    ):
                        raise ValueError("startup tensor descriptor differs from runtime schema")
                    record[path] = {"dtype": shape["dtype"], "shape": tuple(shape["shape"])}
            # Side buffers use exactly their main attention tensor geometry and
            # dtype when first created by MTPCache.update; no draft evaluation.
            for side, main in (("side.0", "keys"), ("side.1", "values"), ("side.2", "index_keys")):
                if side in paths and side not in record and main in record:
                    record[side] = dict(record[main])
            # Qwen's pooled index keys share index width/dtype; length alone
            # varies. A prefix shorter than one complete block may have None.
            if "pooled" in paths and "pooled" not in record and "index_keys" in record:
                record["pooled"] = dict(record["index_keys"])
            if "ik" in paths and "keys" in record and "pool" not in record:
                raise ValueError("GLM index pool must be described with its key allocation")
            observed.append(record)
        # All mutable shape containers become owner-only tuples before return.
        self.observed = tuple(observed)

    def registry(self):
        if self.observed is None:
            raise ValueError("evaluated current-model cache profile required before persistence")
        layers = []
        for index, (prototype, paths, observed) in enumerate(zip(self.prototypes, self.layouts, self.observed)):
            tensors = {}
            step = getattr(prototype, "step", 0)
            if type(step) is not int or step < 0:
                raise ValueError("builtin allocation step must be a nonnegative integer")
            capacity = self.token_limit + self.max_draft + step
            for path, info in observed.items():
                shape = list(info["shape"])
                axis = paths[path]
                if axis is not None:
                    if not 0 <= axis < len(shape):
                        raise ValueError("source-declared growing axis differs from native rank")
                    bound = max(capacity, int(getattr(prototype, "max_size", 0)))
                    shape[axis] = bound
                    max_elements = math.prod(shape)
                    shape[axis] = None
                else:
                    max_elements = math.prod(shape)
                tensors[path] = TensorSchema((info["dtype"],), tuple(shape), max_elements)
            fields = vars(prototype)
            integers = {
                name: (0, self.token_limit + self.max_draft) for name in ("offset", "side_base") if name in fields
            }
            if "drafted" in fields:
                integers["drafted"] = (0, self.max_draft)
            if "_idx" in fields:
                integers["_idx"] = (0, max(capacity, int(fields["max_size"])))
            lists = {name: len(value) for name, value in fields.items() if type(value) is list}
            if "side" in fields:
                lists["side"] = 3
            required_slots = {
                name: frozenset(int(path.split(".")[1]) for path in observed if path.startswith(name + "."))
                for name in lists
            }
            required = frozenset(
                path
                for path in observed
                if "." not in path
                and path != "pooled"
                and (path not in ("pool", "ipool") or (path == "pool" and "ik" in paths))
            )
            auxiliary = "drafted" in fields and (self.main_layers is None or index >= self.main_layers)
            conditional = required if auxiliary else frozenset()
            layers.append(
                LayerSchema(
                    prototype,
                    tensors,
                    integers,
                    list_lengths=lists,
                    required_list_slots=required_slots,
                    numpy_fields=frozenset({"history"} & set(tensors)),
                    mutable_flags=frozenset({"chaining"} & set(fields)),
                    required_tensors=frozenset() if auxiliary else required,
                    invariants=(_invariants(prototype, paths, conditional),),
                    token_invariants=() if self.main_layers is None else (_token_timeline(index < self.main_layers),),
                )
            )
        return Registry(
            tuple(layers),
            token_limit=self.token_limit,
            token_id_limit=self.token_id_limit,
            tensor_byte_limit=self.tensor_byte_limit,
            sizes=self.sizes,
        )
