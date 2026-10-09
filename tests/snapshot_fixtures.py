"""Explicit trusted cache authority for existing native round-trip fixtures.

This helper runs only in tensor-runtime tests. It derives no authority from disk
metadata and is not a production fallback or an independent geometry oracle.
Builtin fixtures use the maintained source-declared layout. The simple custom
fixture contracts declare growing ``keys``/``rows`` axes; other geometry stays
fixed at its owned evaluated fixture shape.
"""

from __future__ import annotations

import copy
import math


def fixture_registry(cache):
    import mlx.core as mx
    import numpy as np

    from tensorfold.engine.snapshot_builtin import Profile, UnregisteredCache
    from tensorfold.engine.snapshot_codec import Codec, SIZES
    from tensorfold.engine.snapshot_registry import LayerSchema, Registry, TensorSchema

    codec = Codec(mx, np)
    prototypes = []
    rows = [item for item in cache if getattr(item, "stored", True)]
    for item in rows:
        materialize = getattr(item, "materialize", None)
        if materialize is not None:
            materialize()
        prototype = copy.copy(item)  # initialized trusted fixture; never __new__
        for name, value in vars(item).items():
            if codec.kind(value) is not None:
                vars(prototype)[name] = None
            elif type(value) is list:
                vars(prototype)[name] = [None] * len(value)
            elif name in ("offset", "_idx", "drafted", "side_base"):
                vars(prototype)[name] = 0
        if "side" in vars(prototype):
            vars(prototype)["side"] = None
        if "chaining" in vars(prototype):
            vars(prototype)["chaining"] = False
        drop = getattr(prototype, "drop_spare", None)
        if drop is not None:
            drop()
        prototypes.append(prototype)
    arguments = dict(
        token_limit=2**63 - 1,
        token_id_limit=2**63 - 1,
        tensor_byte_limit=4 << 30,
        max_draft=64,
        sizes=SIZES,
        describe_tensor=codec.describe,
    )
    try:
        profile = Profile(prototypes, **arguments)
    except UnregisteredCache:
        schemas = []
        for item, prototype in zip(rows, prototypes):
            tensors, host, mutable = {}, set(), {}
            for name, value in vars(item).items():
                if name in getattr(type(item), "transient", ()):
                    continue
                if name in ("offset", "_idx", "drafted", "side_base"):
                    mutable[name] = (0, 2**63 - 1)
                values = (
                    [(f"{name}.{slot}", element) for slot, element in enumerate(value)]
                    if type(value) is list
                    else [(name, value)]
                )
                for field, element in values:
                    description = codec.describe(element)
                    if description is None:
                        continue
                    shape = list(description["shape"])
                    axis = {"keys": 2, "rows": 1}.get(name) if "." not in field else None
                    if axis is not None and axis < len(shape):
                        shape[axis] = None
                        bound = (4 << 30) // SIZES[description["dtype"]]
                    else:
                        bound = math.prod(shape)
                    tensors[field] = TensorSchema((description["dtype"],), tuple(shape), bound)
                    if codec.kind(element) == "numpy":
                        host.add(field)
            schemas.append(LayerSchema(prototype, tensors, mutable, numpy_fields=frozenset(host)))
        return Registry(
            tuple(schemas),
            token_limit=arguments["token_limit"],
            token_id_limit=arguments["token_id_limit"],
            tensor_byte_limit=arguments["tensor_byte_limit"],
            sizes=SIZES,
        )
    profile.observe(cache)
    return profile.registry()
