"""Stack current projections with equal K splits and consume strided outputs without changing arithmetic."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

enabled = False         # the lane decoder uses the stacked groups
auto_build = True       # a layer's groups are stacked on first use when build() has not run
kinds = {"zba", "kv", "gu"}     # the groups used (a subset switches the others back to separate calls)
separate_rows: dict[str, range] = {"gu": range(17, 33)}

# group kind -> the parent module's projections, stacked in this order
GROUPS: dict[str, tuple[str, ...]] = {
    "zba": ("in_proj_z", "in_proj_b", "in_proj_a"),
    "kv": ("k_proj", "v_proj"),
    "gu": ("gate_proj", "up_proj"),
}
_ATTR = "_lane_fuse_groups"      # parent.__dict__[_ATTR]: {kind: _Group}


def _schema(module: Any) -> tuple[Any, ...]:
    return (module.bits, int(module.group_size), getattr(module, "mode", "affine"),
            tuple(module["weight"].shape), module["weight"].dtype,
            tuple(module["scales"].shape), module["scales"].dtype,
            tuple(module["biases"].shape), module["biases"].dtype, "bias" in module,
            bool(getattr(module, "_lane_tile", False)), int(getattr(module, "_lane_nt", 32)))


def _shared_transform(outer: tuple[Any, ...]) -> bool:
    """Admit equal current transforms for both initial and cached geometry.

    Rotation owners remain mutable between synchronous calls. Shared signs
    prove equal transforms only for the declared wrapper and rotation policy.
    """

    if (len({id(getattr(m, "signs", None)) for m in outer}) != 1
            or len({hasattr(m, "rotate") for m in outer}) != 1):
        return False
    if not hasattr(outer[0], "rotate"):
        return True
    from tensorfold.families.bonsai.modules import RotatedLinear, RotationCache

    return all(type(m) is RotatedLinear and type(m.rotation) is RotationCache
               and getattr(m.rotate, "__func__", None) is RotatedLinear.rotate for m in outer)


class _Group:
    """A geometry plan; current member arrays are materialized for each projection.

    Public arrays retain standard MLX encoding and remain mutable between calls.
    A synchronous call borrows the members and transform without concurrent mutation.
    The plan retains no derived weight/scales arrays and never replaces member arrays.
    """

    __slots__ = ("tiled", "sk", "k", "sizes", "added", "members", "nt", "group", "parent", "kind", "outer",
                 "schema", "signs", "rotators")

    def __init__(self, sk: int, k: int, sizes: tuple[int, ...], members: tuple[Any, ...], parent: Any,
                 kind: str, outer: tuple[Any, ...], tiled: bool, nt: int, group: int) -> None:
        self.tiled, self.sk, self.k, self.sizes = tiled, sk, k, sizes
        self.members, self.parent, self.kind, self.outer = members, parent, kind, outer
        self.nt, self.group, self.added = nt, group, 0
        self.schema = tuple(_schema(m) for m in members)
        self.signs = tuple(getattr(m, "signs", None) for m in outer)
        self.rotators = tuple(getattr(getattr(m, "rotate", None), "__func__", getattr(m, "rotate", None))
                              for m in outer)

    def valid(self) -> bool:
        return (all(getattr(self.parent, name, None) is m for name, m in zip(GROUPS[self.kind], self.outer))
                and all((getattr(o, "inner", o) if hasattr(o, "rotate") else o) is m
                        for o, m in zip(self.outer, self.members))
                and tuple(_schema(m) for m in self.members) == self.schema
                and all(getattr(o, "signs", None) is signs for o, signs in zip(self.outer, self.signs))
                and tuple(getattr(getattr(o, "rotate", None), "__func__", getattr(o, "rotate", None))
                          for o in self.outer) == self.rotators
                and _shared_transform(self.outer))

    @property
    def rotate(self) -> Any:
        return getattr(self.outer[0], "rotate", None)

    @property
    def weight(self) -> mx.array:
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm

        weight = mx.concatenate([m["weight"] for m in self.members], axis=0)
        return lane_qmm.tile_weight(weight, self.nt, self.group, bits=self.members[0].bits) if self.tiled else weight

    @property
    def sbt(self) -> mx.array:
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm

        return mx.concatenate([lane_qmm.pack_scales(m["scales"], m["biases"]) for m in self.members], axis=1)


class _Unfusable:
    """A failed admission is reevaluated on use because parameters remain mutable."""

    def __init__(self, members: tuple[Any, ...]) -> None:
        pass

    def valid(self) -> bool:
        return False


def _build(parent: Any, kind: str) -> _Group | _Unfusable:
    """Prepare current geometry without retaining or replacing parameter arrays."""

    import mlx.nn as nn

    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    outer = tuple(getattr(parent, name, None) for name in GROUPS[kind])
    members = tuple(getattr(m, "inner", m) if hasattr(m, "rotate") else m for m in outer)
    no = _Unfusable(members)
    if not all(isinstance(m, nn.QuantizedLinear) for m in members):
        return no
    if not _shared_transform(outer):
        return no
    for m in members:
        w = m["weight"]
        if not lane_qmm.takes(m) or m.group_size not in (32, 64) or "bias" in m or w.dtype != mx.uint32 or w.ndim != 2:
            return no
    bits, group = members[0].bits, int(members[0].group_size)
    if any(m.bits != bits or int(m.group_size) != group for m in members):
        return no
    kw = int(members[0]["weight"].shape[1])
    k = kw * 32 // bits
    sizes = tuple(int(m["weight"].shape[0]) for m in members)
    if (k % 64 or k * bits != kw * 32 or any(int(m["weight"].shape[1]) != kw for m in members)
            or any(n % 4 for n in sizes)
            or any(tuple(m["scales"].shape) != (n, k // group)
                   or tuple(m["biases"].shape) != (n, k // group)
                   or m["biases"].dtype != mx.bfloat16 for m, n in zip(members, sizes))):
        return no
    splits = {lane_qmm.split_k(n, k) for n in sizes}
    if len(splits) != 1:
        return no
    sk = splits.pop()
    widths = {int(getattr(m, "_lane_nt", lane_qmm.NT)) for m in members}
    tiled = all(bool(getattr(m, "_lane_tile", False)) for m in members)
    if tiled and len(widths) != 1:
        return no
    nt = widths.pop() if tiled and bits == 4 else lane_qmm.NT
    tiled = tiled and sum(sizes) % nt == 0
    return _Group(sk, k, sizes, members, parent, kind, outer, tiled, nt, group)


def _group(parent: Any, kind: str, *, build: bool | None = None) -> _Group | None:
    groups = parent.__dict__.get(_ATTR)
    if groups is None:
        groups = {}
        object.__setattr__(parent, _ATTR, groups)
    group = groups.get(kind)
    if group is not None:
        if group.valid():
            return group if isinstance(group, _Group) else None
        del groups[kind]                                   # stale: its members changed since
    if not (auto_build if build is None else build):
        return None
    group = _build(parent, kind)
    groups[kind] = group
    return group if isinstance(group, _Group) else None


def _project(parent: Any, kind: str, x: mx.array) -> mx.array | None:
    """Return the stacked projection (..., sum N) only when every member would use lane matmul, otherwise None."""

    if not enabled or kind not in kinds:
        return None
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    if not lane_qmm.enabled or x.dtype != mx.bfloat16:
        return None
    k = int(x.shape[-1])
    rows = x.size // max(k, 1)
    if rows < 1 or rows > lane_qmm.max_rows or rows in separate_rows.get(kind, ()):
        return None
    group = _group(parent, kind)
    if group is None or group.k != k:
        return None
    if group.rotate is not None:
        x = group.rotate(x)
    return lane_qmm.lane_matmul(x, group.weight, group.sbt, tiled=group.tiled, sk=group.sk, nt=group.nt,
                                group=group.group)


def gdn_in(gdn: Any, x: mx.array) -> mx.array | None:
    """[z | b | a] of a Gated DeltaNet layer (..., nv*dv + 2 nv) in one lane matmul, or None."""

    return _project(gdn, "zba", x)


def attn_kv(attn: Any, x: mx.array) -> mx.array | None:
    """[k | v] of an attention layer (..., 2 kv_heads * head_dim) in one lane matmul, or None."""

    return _project(attn, "kv", x)


def mlp_gate_up(mlp: Any, x: mx.array) -> mx.array | None:
    """[gate | up] of an MLP block (..., 2 N) in one lane matmul, or None."""

    return _project(mlp, "gu", x)


def build(model: Any) -> dict[str, int]:
    """Prepare group geometry after lane installation: {kind: groups prepared}."""

    counts = {kind: 0 for kind in GROUPS}
    for _, module in model.named_modules():
        for kind, names in GROUPS.items():
            if all(hasattr(module, name) for name in names) and _group(module, kind, build=True) is not None:
                counts[kind] += 1
    return counts


def clear(model: Any) -> None:
    """Drop geometry plans; public parameters remain unchanged."""

    for _, module in model.named_modules():
        if _ATTR in module.__dict__:
            del module.__dict__[_ATTR]


def stats(model: Any) -> dict[str, Any]:
    """Prepared groups per kind and retained derived-array bytes (zero)."""

    counts = {kind: 0 for kind in GROUPS}
    added = 0
    seen: set[int] = set()
    for _, module in model.named_modules():
        for kind, group in module.__dict__.get(_ATTR, {}).items():
            if isinstance(group, _Group) and group.valid() and id(group) not in seen:
                seen.add(id(group))                        # a module listed twice is one stack
                counts[kind] += 1
                added += group.added
    return {"groups": counts, "added_bytes": added}


# -- the consumers, reading the stacked outputs in place ----------------------------------------

_variants: dict[str, tuple[str, Any]] = {}


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"lane_fuse: lane_glue's kernel changed ({old!r} found {source.count(old)} times)")
    return source.replace(old, new)


def _variant_sources() -> dict[str, tuple[str, list[str], list[str]]]:
    from tensorfold.kernels.qwen.dense.v1 import lane_glue

    post = _replace_once(lane_glue._GDN_POST, "float(Z[m * NV * DV + hv * DV + d])", "float(Z[m * ZS + hv * DV + d])")
    act = _replace_once(lane_glue._MLP_ACT, "float(GATE[e])", "float(GATE[e + int(m) * N])")    # rows of 2N
    act = _replace_once(act, "float(UP[e])", "float(UP[e + int(m) * N + N])")
    return {
        "gdn_post": (post, ["Y", "Z", "NW", "eps", "dims"], ["OUT", "XS"]),
        "mlp_act": (act, ["GATE", "UP", "dims"], ["HOUT", "XS"]),
    }


def sources() -> dict[str, str]:
    """The consumer kernels' sources (for a decoder-version hash: a kernel edit can change bits)."""

    return {name: spec[0] for name, spec in _variant_sources().items()}


def _kernel(name: str) -> Any:
    hit = _variants.get(name)
    if hit is None:
        source, inputs, outputs = _variant_sources()[name]
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        kernel = mx.fast.metal_kernel(name=f"lane_fuse_{name}_{digest}", input_names=inputs, output_names=outputs,
                                      source=source)
        hit = _variants[name] = (source, kernel)
    return hit[1]


_consts: dict[Any, mx.array] = {}


def _dims(m: int) -> mx.array:
    key = ("dims", m)
    if key not in _consts:
        _consts[key] = mx.array([m, 16 * ((m + 15) // 16)], dtype=mx.int32)
    return _consts[key]


def _eps(eps: float) -> mx.array:
    key = ("eps", float(eps))
    if key not in _consts:
        _consts[key] = mx.array([float(eps)], dtype=mx.float32)
    return _consts[key]


def gdn_post(y: mx.array, zba: mx.array, weight: mx.array, eps: float) -> mx.array:
    """``lane_glue.gdn_post`` with z read in place from ``gdn_in``'s [z | b | a] rows."""

    from tensorfold.kernels.qwen.dense.v1 import lane_glue

    _, W, nv, dv = (int(s) for s in y.shape)
    zs = int(zba.shape[-1])
    MP = 16 * ((W + 15) // 16)
    out, xs = _kernel("gdn_post")(
        inputs=[y, zba.reshape(W, zs), weight, _eps(eps), _dims(W)],
        template=[("NV", nv), ("DV", dv), ("ZS", zs)],
        grid=(32, nv, MP), threadgroup=(32, 1, 1),
        output_shapes=[(1, W, nv * dv), (nv * dv // 64, MP)], output_dtypes=[y.dtype, mx.float32])
    return lane_glue.remember(out, xs)


def mlp_act(gu: mx.array) -> mx.array:
    """``lane_glue.mlp_act`` on ``mlp_gate_up``'s [gate | up] rows: SiLU(gate) * up, (..., N)."""

    from tensorfold.kernels.qwen.dense.v1 import lane_glue

    N2 = int(gu.shape[-1])
    N = N2 // 2
    W = int(gu.size // N2)
    MP = 16 * ((W + 15) // 16)
    gu2 = gu.reshape(W, N2)
    h, xs = _kernel("mlp_act")(
        inputs=[gu2, gu2, _dims(W)], template=[("N", N)],          # GATE and UP: the same rows, halves
        grid=(64 * (N // 64), MP, 1), threadgroup=(64, 1, 1),
        output_shapes=[(W, N), (N // 64, MP)], output_dtypes=[gu.dtype, mx.float32])
    return lane_glue.remember(h.reshape(*gu.shape[:-1], N), xs)


def warm(model: Any, *, rows: tuple[int, ...] = (1, 17, 33)) -> int:
    """Compile the stacked shapes' lane matmul variants (per row tile) and the consumer kernels."""

    from tensorfold.kernels.qwen.dense.v1 import lane_qmm, stream_gdn

    seen: set[tuple[int, ...]] = set()
    outs: list[mx.array] = []
    for _, module in model.named_modules():
        for kind in list(module.__dict__.get(_ATTR, {})):
            group = _group(module, kind, build=True)
            if not isinstance(group, _Group):
                continue
            weight, sbt = group.weight, group.sbt
            key = (int(weight.shape[0]), group.k, lane_qmm.weight_bits(weight, group.k), group.sk,
                   group.tiled, group.nt, group.group)
            if key in seen:
                continue
            seen.add(key)
            for m in rows:
                y = lane_qmm.lane_matmul(mx.zeros((m, group.k), dtype=mx.bfloat16), weight, sbt,
                                         tiled=group.tiled, sk=group.sk, nt=group.nt, group=group.group)
                outs.append(y)
                if kind == "gu":
                    outs.append(mlp_act(y[None]))
                elif kind == "zba":
                    nv = int(module.num_v_heads)
                    dv = int(module.head_v_dim)
                    outs.append(gdn_post(mx.zeros((1, m, nv, dv), dtype=mx.bfloat16), y[None],
                                         module.norm.weight, module.norm.eps))
                    nk, dk = int(module.num_k_heads), int(module.head_k_dim)
                    taps = int(module.conv_kernel_size)
                    C = 2 * nk * dk + nv * dv
                    plan = stream_gdn.ConvPlan([[-1] + list(range(m - 1))], taps - 1)
                    outs.extend(stream_gdn.gdn_pre(mx.zeros((1, m, C), dtype=mx.bfloat16),
                                                   [mx.zeros((1, taps - 1, C), dtype=mx.bfloat16)],
                                                   module.conv1d.weight, plan, y[None], module.A_log, module.dt_bias,
                                                   nk=nk, nv=nv, dk=dk, dv=dv))
    mx.eval(outs)
    return len(seen)


__all__ = ["GROUPS", "attn_kv", "auto_build", "build", "clear", "enabled", "gdn_in", "gdn_post", "kinds",
           "mlp_act", "mlp_gate_up", "separate_rows", "sources", "stats", "warm"]
