"""Decode without tensor units using row-independent arithmetic, with serial steps as the reference for windows and streams."""

from __future__ import annotations

from itertools import islice
from operator import index as integer_index
import os
import math
from typing import Any, Callable, Sequence

import mlx.core as mx

from tensorfold.engine.family_common import cache_contents
from tensorfold.kernels.inputs import ints
from tensorfold.kernels.qwen.dense.v1 import projection_operation, row_matmul
from tensorfold.kernels.qwen.dense.v1.row_glue import _chain, add_norm, gated_delta, gdn_post, gdn_pre, mlp_act
from tensorfold.kernels.qwen.dense.v1.row_matmul import WINDOW_ROWS, logits, project, project_stack, stack_of


# rows a prompt chain takes
PROMPT_ROWS = 128


# TF_ROW_ATTENTION selects row-exact tree attention; otherwise use exact_attention query by query for chains.
ROW_ATTENTION = os.environ.get("TF_ROW_ATTENTION", "0") == "1"


class Record(list):
    """Record one stream's layer commits and its window start, width and parent indices."""

    start: int = 0
    width: int = 0
    parents: tuple[int, ...] = ()
    borrowed_cache: tuple[Any, ...] | None = None
    failed: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed = False


def _stream_inputs(values: Sequence[Any], count: int) -> tuple[Any, ...]:
    """Snapshot one operation's bounded stream list without over-consuming a producer."""

    snapshot = tuple(islice(iter(values), count + 1))
    if len(snapshot) != count:
        raise ValueError("row_forward: stream input changed during its borrow")
    return snapshot


class _Rows:
    """Group rows by stream; each row's absolute position is its stream start plus its tree depth."""

    __slots__ = ("offsets", "widths", "parents", "chains", "starts", "records", "positions", "host_positions", "windows", "single")

    def __init__(self, parents: Sequence[Sequence[int]], starts: Sequence[int]) -> None:
        from tensorfold.kernels.qwen.dense.v1 import lane_tree

        self.offsets: list[int] = []
        self.widths: list[int] = []
        self.parents: list[tuple[int, ...]] = []
        self.chains: list[bool] = []
        count = len(parents)
        if count == 0 or len(starts) != count:
            raise ValueError("row_forward: parents and starts must cover nonempty streams")
        widths = [len(p) for p in parents]
        if any(width < 1 for width in widths) or sum(widths) > row_matmul.BACKEND.max_rows:
            raise ValueError("row_forward: nonempty stream rows must fit the selected backend")
        start_values = tuple(islice(iter(starts), count + 1))
        if len(start_values) != count:
            raise ValueError("row_forward: starts changed during their borrow")
        self.starts = []
        for start in start_values:
            try:
                if isinstance(start, bool):
                    raise TypeError
                start = integer_index(start)
            except TypeError as error:
                raise ValueError("row_forward: starts require nonnegative integers") from error
            if not 0 <= start < (1 << 31):
                raise ValueError("row_forward: starts must fit signed32 positions")
            self.starts.append(start)
        self.records: list[Record] = []
        positions: list[int] = []
        total = 0
        for rows_parents, start, width in zip(parents, self.starts, widths):
            snapshot = tuple(islice(iter(rows_parents), width + 1))
            if len(snapshot) != width:
                raise ValueError("row_forward: parents changed during their borrow")
            normalized = []
            for row, parent in enumerate(snapshot):
                try:
                    if isinstance(parent, bool):
                        raise TypeError
                    parent = integer_index(parent)
                except TypeError as error:
                    raise ValueError("row_forward: parents require ordered integer references") from error
                if not -(1 << 31) <= parent < row:
                    raise ValueError("row_forward: parents require signed32 references before their children")
                normalized.append(parent)
            rows_parents = tuple(normalized)
            chain = rows_parents == _CHAINS.get(len(rows_parents)) or _chain(rows_parents)
            depths = range(len(rows_parents)) if chain else lane_tree.tree_paths(rows_parents)[0]
            stream_positions = [start + d for d in depths]
            if any(position >= (1 << 31) for position in stream_positions):
                raise ValueError("row_forward: absolute positions must fit signed32")
            positions.extend(stream_positions)
            record = Record()
            record.start, record.width, record.parents = start, len(rows_parents), rows_parents
            self.offsets.append(total)
            self.widths.append(len(rows_parents))
            self.parents.append(rows_parents)
            self.chains.append(chain)
            self.records.append(record)
            total += len(rows_parents)
        self.single = len(self.widths) == 1
        self.host_positions = tuple(positions)
        self.positions = mx.array(positions, dtype=mx.int32)
        self.windows: list[mx.array] | None = None


# chain parents by width (-1, 0, 1, ...), for a cheap comparison
_CHAINS: dict[int, tuple[int, ...]] = {w: tuple(range(-1, w - 1)) for w in range(1, 257)}
_conv_cache: dict[tuple[tuple[int, ...], int], mx.array] = {}


def _conv_windows(parents: tuple[int, ...], n_keep: int) -> mx.array:
    """``lane_tree._conv_windows``, built once per window shape."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    key = (parents, n_keep)
    hit = _conv_cache.get(key)
    if hit is None:
        if len(_conv_cache) > 4096:
            _conv_cache.clear()
        hit = _conv_cache[key] = lane_tree._conv_windows(parents, n_keep)
    return hit


def _in(module: Any, x: mx.array) -> mx.array:
    return project(module, x)



def _attention_metadata(cache, heads, kv_heads, width, dim, start, parents, scale):
    """Prove raw geometry before update, retaining exact SDK storage delegation.

    Raw cache callbacks append full BF16 buffers using their declared step
    growth or concatenate policy. Custom callbacks are trusted code and must
    obey that contract; their effects cannot be proved from metadata.
    """
    from tensorfold.kernels.qwen.dense.v1 import row_attention

    limit = (1 << 31) - 1
    end_limit = limit if ROW_ATTENTION else limit + 1
    start = _row_integer(start, 0, end_limit - width, "attention start")
    if _row_integer(cache.offset, 0, end_limit - width, "attention offset") != start:
        raise ValueError("row_forward: attention offset differs from its window start")
    if not callable(getattr(cache, "update_and_fetch", None)) or not callable(getattr(cache, "trim", None)):
        raise ValueError("row_forward: attention cache requires update and trim")
    if not ROW_ATTENTION:
        return
    if hasattr(cache, "bits") or hasattr(cache, "max_size"):
        raise ValueError("row_forward: raw attention requires full append-only array buffers")
    stored_k, stored_v = cache.keys, cache.values
    if (stored_k is None) != (stored_v is None):
        raise ValueError("row_forward: attention cache has incomplete KV storage")
    capacity = 0
    if stored_k is not None:
        shape = _array_shape(stored_k, 4, "stored keys", max_elements=(1 << 64) - 1)
        if (shape != _array_shape(stored_v, 4, "stored values", max_elements=(1 << 64) - 1)
                or shape[:2] != (1, kv_heads) or shape[3] != dim
                or stored_k.dtype != mx.bfloat16 or stored_v.dtype != mx.bfloat16 or start > shape[2]):
            raise ValueError("row_forward: raw cache geometry/dtype differs from its layer")
        capacity = shape[2]
    elif start:
        raise ValueError("row_forward: nonempty attention offset has no KV storage")
    step = getattr(cache, "step", None)
    if step is None:
        # ConcatenateKVCache/SimpleKVCache exposes only its complete live rows.
        if capacity != start:
            raise ValueError("row_forward: concatenate cache capacity differs from its live offset")
        capacity += width
    elif start + width > capacity:
        step = _row_integer(step, 1, limit, "attention growth step")
        capacity = (start if start % step else capacity) + ((width + step - 1) // step) * step
    group = heads // kv_heads
    if (dim % 32 or group * row_attention.SPLIT * 32 > 1024
            or group * row_attention.SPLIT * (dim + 2) * 4 > 32768
            or max(heads, kv_heads, width, dim, capacity, start + width,
                   kv_heads * width, heads * width * dim) > limit):
        raise ValueError("row_forward: raw attention exceeds native dimension/thread/shared bounds")
    scale = float(scale)
    if not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("row_forward: raw attention scale requires finite FP32")
    depths = []
    for parent in parents:
        depths.append(0 if parent < 0 else depths[parent] + 1)
    max_depth = max(depths) + 1
    chunks = (start + max_depth + row_attention.CK - 1) // row_attention.CK
    if (max(width * max_depth, heads * width * chunks,
            chunks * row_attention.CK + row_attention.SPLIT - 1) > limit
            or kv_heads * capacity * dim * 2 > (1 << 64) - 1):
        raise ValueError("row_forward: raw attention exceeds padded path/partial/pointer bounds")


def _attention_inputs(queries, keys, values, cache, parents, scale):
    from tensorfold.kernels.qwen.dense.v1 import row_attention

    query = _array_shape(queries, 4, "attention queries", max_elements=(1 << 64) - 1)
    incoming = _array_shape(keys, 4, "incoming keys", max_elements=(1 << 64) - 1)
    value_shape = _array_shape(values, 4, "incoming values", max_elements=(1 << 64) - 1)
    if (incoming[:3] != value_shape[:3] or query[0] != 1 or incoming[0] != 1
            or incoming[2:] != query[2:] or query[1] % incoming[1]
            or query[2] > row_matmul.BACKEND.max_rows):
        raise ValueError("row_forward: attention query/KV head/window geometry differs")
    parents = row_attention._parents(parents, query[2])
    if not ROW_ATTENTION and not _chain(parents):
        raise NotImplementedError("row_forward: draft trees need row_attention")
    if ROW_ATTENTION and (incoming != value_shape or keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16):
        raise ValueError("row_forward: raw attention requires BF16 incoming KV")
    end_limit = (1 << 31) - 1 if ROW_ATTENTION else 1 << 31
    start = _row_integer(cache.offset, 0, end_limit - query[2], "attention offset")
    _attention_metadata(cache, query[1], incoming[1], query[2], query[3], start, parents, scale)
    return parents, start, query[2]


def _forward_cache_inputs(layers, caches, rows):
    """Admit every stream/layer before the first forward cache write."""
    from tensorfold.kernels.qwen.dense.v1 import lane_tree, row_streams
    limit = (1 << 31) - 1
    normalized = tuple(_stream_inputs(cache, len(layers)) for cache in caches)
    seen = set()
    for layer_at, layer in enumerate(layers):
        inner = getattr(layer, "_layer", layer)
        for stream, cache in enumerate(normalized):
            item = cache[layer_at]
            if id(item) in seen:
                raise ValueError("row_forward: streams/layers require independent mutable caches")
            seen.add(id(item))
            if getattr(inner, "is_linear", False):
                gdn = inner.linear_attn
                if (len(item) != 2 or not callable(getattr(item, "advance", None))
                        or getattr(item, "left_padding", None) is not None
                        or getattr(item, "lengths", None) is not None):
                    raise ValueError("row_forward: raw recurrent cache needs two unmasked state entries")
                nk = _row_integer(gdn.num_k_heads, 1, limit, "key heads")
                nv = _row_integer(gdn.num_v_heads, 1, limit, "value heads")
                dk = _row_integer(gdn.head_k_dim, 1, limit, "key width")
                dv = _row_integer(gdn.head_v_dim, 1, limit, "value width")
                history = _row_integer(gdn.conv_kernel_size, 1, limit, "convolution taps") - 1
                conv_dim = _row_integer(gdn.conv_dim, 1, limit, "convolution width")
                if nv % nk or dk != dv or dk % 32 or conv_dim != 2 * nk * dk + nv * dv:
                    raise ValueError("row_forward: recurrent head/convolution geometry differs")
                group = _row_integer(row_streams.GROUP, 1, row_matmul.BACKEND.max_rows, "recurrent stream group")
                begin = (stream // group) * group
                end = min(begin + group, len(normalized))
                chain = rows.chains[stream] if rows.single else all(rows.chains[begin:end])
                max_width = lane_tree.MAX_DEPTH if chain else lane_tree.MAX_TREE
                if rows.widths[stream] > max_width:
                    raise ValueError("row_forward: recurrent tree/chain window exceeds its selected kernel")
                weight = gdn.conv1d.weight
                if (not isinstance(weight, mx.array) or weight.ndim < 2
                        or weight.shape[1] != history + 1 or weight.size != conv_dim * (history + 1)):
                    raise ValueError("row_forward: convolution parameters differ from declared history/channels")
                for value in (gdn.A_log, gdn.dt_bias):
                    if not isinstance(value, mx.array) or value.size < nv:
                        raise ValueError("row_forward: recurrent parameter cannot cover its value heads")
                width = rows.widths[stream] if rows.single else sum(rows.widths[begin:end])
                streams = 1 if rows.single else end - begin
                stacked = conv_dim + nv * dv + 2 * nv
                # QKV's selected prefix, CW/CS/CO and tree row*head
                # intermediates are signed. A/B, post and final state/output
                # offsets use uint w/m/hv/lane; check their exact extrema.
                signed = ((width - 1) * stacked + conv_dim - 1,
                          (history + 1) * conv_dim - 1,
                          width * history * conv_dim - 1,
                          (width - 1) * nk, (width - 1) * nv,
                          nv * streams - 1, stacked)
                unsigned = (nv * dv * dk - 1, width * stacked - 1,
                            width * nk * dk - 1, width * nv * dv - 1,
                            width * (history + 1) - 1)
                if max(signed) > limit or max(unsigned) > (1 << 32) - 1:
                    raise ValueError("row_forward: recurrent input/state/window spans exceed native representation")
                for previous, expected in ((item[0], (1, history, conv_dim)),
                                           (item[1], (1, nv, dv, dk))):
                    if previous is not None and _array_shape(previous, len(expected), "recurrent cache",
                            zero_axes=(1,) if len(expected) == 3 else (),
                            max_elements=(1 << 64) - 1 if len(expected) == 3 else 1 << 32) != expected:
                        raise ValueError("row_forward: recurrent cache differs from its layer")
            else:
                attn = inner.self_attn
                heads = _row_integer(attn.num_attention_heads, 1, limit, "query heads")
                kv_heads = _row_integer(attn.num_key_value_heads, 1, limit, "KV heads")
                dim = getattr(attn, "head_dim", None)
                if dim is None:
                    # Match the original q-projection-derived fallback. Known
                    # SDK Linear/QuantizedLinear retains its output rows here.
                    weight = getattr(attn.q_proj, "weight", None)
                    if not isinstance(weight, mx.array) or weight.ndim < 1 or weight.shape[0] % (2 * heads):
                        raise ValueError("row_forward: attention needs provable query projection geometry")
                    dim = weight.shape[0] // (2 * heads)
                dim = _row_integer(dim, 1, limit, "attention head width")
                if heads % kv_heads:
                    raise ValueError("row_forward: query heads must cover whole KV groups")
                _attention_metadata(item, heads, kv_heads, rows.widths[stream], dim,
                                    rows.starts[stream], rows.parents[stream], attn.scale)
                shift = _row_integer(getattr(item, "vision_rope_delta", 0), -(1 << 31), limit,
                                     "vision position delta")
                at, width = rows.offsets[stream], rows.widths[stream]
                if any(not -(1 << 31) <= position + shift <= limit
                       for position in rows.host_positions[at:at + width]):
                    raise ValueError("row_forward: shifted vision positions exceed signed32")
    return normalized


def _forward_windows(core, windows, widths):
    """Validate host IDs before uint32 conversion; lazy windows are never read."""
    weight = getattr(core.embed_tokens, "weight", None)
    vocab = weight.shape[0] if isinstance(weight, mx.array) and weight.ndim >= 1 else 1 << 32
    upper = min((1 << 32) - 1, vocab - 1)
    admitted = []
    for window, width in zip(windows, widths):
        if isinstance(window, mx.array):
            admitted.append(window)
        else:
            values = _stream_inputs(window, width)
            admitted.append(tuple(_row_integer(token, 0, upper, "token ID") for token in values))
    return tuple(admitted)


def _implicit_stream_inputs(core, windows, caches):
    """Bound facade producers before deriving default parents or starts."""
    count = _row_integer(len(windows), 1, row_matmul.BACKEND.max_rows, "streams")
    if len(caches) != count:
        raise ValueError("row_forward: implicit streams need one cache per window")
    windows, caches = _stream_inputs(windows, count), _stream_inputs(caches, count)
    layer_count = len(core.layers)
    if any(len(cache) != layer_count for cache in caches):
        raise ValueError("row_forward: implicit stream caches must cover every layer")
    caches = tuple(_stream_inputs(cache, layer_count) for cache in caches)
    widths = tuple(_row_integer(window.size if isinstance(window, mx.array) else len(window),
                              1, row_matmul.BACKEND.max_rows, "window rows") for window in windows)
    if sum(widths) > row_matmul.BACKEND.max_rows:
        raise ValueError("row_forward: implicit stream rows exceed selected backend")
    return windows, caches, widths

def _attend(attn: Any, queries: mx.array, keys: mx.array, values: mx.array, cache: Any, parents: tuple[int, ...],
            chain: bool, record: list[Any]) -> mx.array:
    """One stream's rows [1, H, w, D] through attention over its own cache (which takes the rows' keys)."""

    from tensorfold.kernels.qwen.dense.v1 import exact_attention, row_attention

    if not ROW_ATTENTION and not chain:
        raise NotImplementedError("row_forward: draft trees need row_attention")
    parents, expected_start, L = _attention_inputs(queries, keys, values, cache, parents, attn.scale)
    record.append(("kv", keys, values))
    keys, values = cache.update_and_fetch(keys, values)
    if ROW_ATTENTION:
        _row_integer(cache.offset, expected_start + L, expected_start + L, "updated cache offset")
        # the cache's whole buffers: the window's rows sit at start + row
        return row_attention.row_sdpa(queries, cache.keys, cache.values, attn.scale, int(cache.offset) - L, parents)
    return exact_attention.exact_sdpa(queries, keys, values, cache, attn.scale, "causal" if L > 1 else None)


def _attention(attn: Any, x: mx.array, items: Sequence[Any], rows: _Rows) -> mx.array:
    """Return Qwen3-Next attention inputs to o_proj using per-row RoPE positions and each stream's own keys."""

    B, L, _ = x.shape
    H, nkv = attn.num_attention_heads, attn.num_key_value_heads
    stack = stack_of(attn, "qkv")
    if stack is not None:
        qkv = project_stack(stack, x)
        nq, nk = stack.sizes[0], stack.sizes[1]
        q_proj_output, k_out, v_out = qkv[..., :nq], qkv[..., nq:nq + nk], qkv[..., nq + nk:]
    else:
        q_proj_output, k_out, v_out = _in(attn.q_proj, x), _in(attn.k_proj, x), _in(attn.v_proj, x)
    D = int(attn.head_dim) if hasattr(attn, "head_dim") else int(q_proj_output.shape[-1]) // (2 * H)
    gate = q_proj_output.reshape(B, L, H, 2 * D)[..., D:].reshape(B, L, -1)
    # RMSNorm per head row over [q_h | gate_h] halves, then the q halves (rows are their own: no copy of a slice)
    queries = attn.q_norm(q_proj_output.reshape(B, L, 2 * H, D))[:, :, 0::2]
    keys = attn.k_norm(k_out.reshape(B, L, nkv, -1))
    values = v_out.reshape(B, L, nkv, -1)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)
    pos = rows.positions
    if any(getattr(c, "vision_rope_delta", 0) for c in items):
        shifts = [int(getattr(c, "vision_rope_delta", 0)) for c, width in zip(items, rows.widths)
                  for _ in range(width)]
        pos = pos + mx.array(shifts, dtype=mx.int32)
    queries = attn.rope(queries.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    keys = attn.rope(keys.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    if rows.single:
        output = _attend(attn, queries, keys, values, items[0], rows.parents[0], rows.chains[0], rows.records[0])
    else:
        outs = []
        for s, item in enumerate(items):
            a, w = rows.offsets[s], rows.widths[s]
            outs.append(_attend(attn, queries[:, :, a:a + w], keys[:, :, a:a + w], values[:, :, a:a + w], item,
                                rows.parents[s], rows.chains[s], rows.records[s]))
        output = mx.concatenate(outs, axis=2)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return output * mx.sigmoid(gate)


def _recur(gdn: Any, y: mx.array, cache: Any, parents: tuple[int, ...], chain: bool, windows: mx.array,
           record: list[Any], n_keep: int) -> mx.array:
    """Return recurrence output [1, w, Hv, Dv] from stacked [qkv | z | b | a] rows and one stream's conv tail and state."""

    conv_state = cache[0] if cache[0] is not None else mx.zeros((1, n_keep, gdn.conv_dim), dtype=y.dtype)
    state = cache[1]
    if state is None:
        state = mx.zeros((1, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32)
    q, k, v, g, beta, conv_out = gdn_pre(y, conv_state, gdn.conv1d.weight, windows, gdn.A_log, gdn.dt_bias,
                                         nk=gdn.num_k_heads, nv=gdn.num_v_heads, dk=gdn.head_k_dim, dv=gdn.head_v_dim)
    rec, state_out = gated_delta(q, k, v, g, beta, state, parents, chain=chain)
    record.append(("gdn", q, k, v, g, beta, state, (conv_state, y, gdn.conv_dim, state_out, conv_out, chain),
                   n_keep))
    return rec


def _gdn(gdn: Any, x: mx.array, items: Sequence[Any], rows: _Rows) -> mx.array:
    """The recurrent layer for every stream's rows (each from its own conv tail and state): the input of out_proj."""

    from tensorfold.kernels.qwen.dense.v1 import row_streams

    stack = stack_of(gdn, "in")
    y = (project_stack(stack, x) if stack is not None else mx.concatenate(
        [project(getattr(gdn, name), x) for name in row_matmul.GROUPS["in"]], axis=-1))
    n_keep = gdn.conv_kernel_size - 1
    if rows.single:
        if rows.windows is None:
            rows.windows = [_conv_windows(p, n_keep) for p in rows.parents]
        rec = _recur(gdn, y, items[0], rows.parents[0], rows.chains[0], rows.windows[0], rows.records[0], n_keep)
        return gdn_post(rec, y, gdn.norm.weight, gdn.norm.eps, zo=gdn.conv_dim)
    recs = []
    for g0 in range(0, len(items), row_streams.GROUP):
        g1 = min(g0 + row_streams.GROUP, len(items))
        a, b = rows.offsets[g0], rows.offsets[g1 - 1] + rows.widths[g1 - 1]
        chain = all(rows.chains[g0:g1])          # a tree in the launch: no state after the last row, commits replay
        rec, per = row_streams.recur(gdn, y[:, a:b], items[g0:g1], rows.parents[g0:g1], chain, n_keep)
        for record, (q, k, v, g, beta, state, conv_state, ys, state_out, tails) in zip(rows.records[g0:g1], per):
            record.append(("gdn", q, k, v, g, beta, state, (conv_state, ys, gdn.conv_dim, state_out, tails, chain),
                           n_keep))
        recs.append(rec)
    rec = recs[0] if len(recs) == 1 else mx.concatenate(recs, axis=1)
    return gdn_post(rec, y, gdn.norm.weight, gdn.norm.eps, zo=gdn.conv_dim)


def _row_integer(value: Any, lower: int, upper: int, name: str) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        value = integer_index(value)
    except TypeError as error:
        raise ValueError(f"row_forward: {name} requires an integer") from error
    if not lower <= value <= upper:
        raise ValueError(f"row_forward: {name} exceeds its admitted range")
    return value


def _array_shape(value: Any, rank: int, name: str, *, zero_axes: tuple[int, ...] = (),
                 max_elements: int = (1 << 31) - 1) -> tuple[int, ...]:
    if not isinstance(value, mx.array) or value.ndim != rank:
        raise ValueError(f"row_forward: {name} requires a rank-{rank} array")
    shape = tuple(value.shape)
    if any(d < 0 or (d == 0 and at not in zero_axes) for at, d in enumerate(shape)):
        raise ValueError(f"row_forward: {name} requires nonempty dimensions")
    size = 1
    for dimension in shape:
        size *= dimension
    if size > max_elements:
        raise ValueError(f"row_forward: {name} exceeds its native indexing range")
    return shape


def _commit_inputs(cache: Sequence[Any], record: Record, path: Sequence[int], window: int, start: int):
    """Prove the complete commit before its first cache/record/device mutation."""
    limit = (1 << 31) - 1
    window = _row_integer(window, 1, row_matmul.BACKEND.max_rows, "window")
    start = _row_integer(start, 0, limit + 1 - window, "start")
    if (type(record) is not Record or record.failed is not False
            or _row_integer(record.width, 1, row_matmul.BACKEND.max_rows, "record window") != window
            or _row_integer(record.start, 0, limit + 1 - window, "record start") != start):
        raise ValueError("row_forward: commit requires its usable matching forward record")
    count = len(cache)
    if count == 0 or len(record) != count:
        raise ValueError("row_forward: commit records must cover every cache layer exactly")
    caches, entries = _stream_inputs(cache, count), _stream_inputs(record, count)
    if record.borrowed_cache is not None and (type(record.borrowed_cache) is not tuple
            or len(record.borrowed_cache) != count
            or any(a is not b for a, b in zip(caches, record.borrowed_cache))):
        raise ValueError("row_forward: commit cache differs from its forward borrow")
    keep = len(path)
    if not 1 <= keep <= window:
        raise ValueError("row_forward: commit path must be nonempty and fit its window")
    selected = _stream_inputs(path, keep)
    selected = tuple(_row_integer(row, 0, window - 1, "path row") for row in selected)
    in_place = selected == tuple(range(keep))
    parents = tuple(_row_integer(parent, -(1 << 31), row - 1, "record parent")
                    for row, parent in enumerate(_stream_inputs(record.parents, window)))
    if parents[selected[0]] >= 0 or any(parents[row] != previous for previous, row in zip(selected, selected[1:])):
        raise ValueError("row_forward: commit rows must follow one complete root path")
    for item, entry in zip(caches, entries):
        if type(entry) is not tuple or not entry:
            raise ValueError("row_forward: commit entries require complete immutable tuples")
        if hasattr(item, "keys") and hasattr(item, "values"):
            if len(entry) != 3 or entry[0] != "kv" or not callable(getattr(item, "trim", None)):
                raise ValueError("row_forward: attention cache needs one complete KV record")
            keys, values = entry[1:]
            shape = _array_shape(keys, 4, "record keys", max_elements=(1 << 64) - 1)
            if shape != _array_shape(values, 4, "record values", max_elements=(1 << 64) - 1) or shape[0] != 1 or shape[2] != window:
                raise ValueError("row_forward: recorded KV rows differ from the window")
            if not in_place:
                # Only relocation addresses resident buffers. Prefix trim
                # preserves STOCK-delegated packed/rotating cache handling.
                stored = _array_shape(item.keys, 4, "cache keys", max_elements=(1 << 64) - 1)
                if (stored != _array_shape(item.values, 4, "cache values", max_elements=(1 << 64) - 1)
                        or stored[:2] != shape[:2]
                        or stored[3] != shape[3] or stored[2] < start + window
                        or item.keys.dtype != keys.dtype or item.values.dtype != values.dtype):
                    raise ValueError("row_forward: KV cache cannot hold its recorded window")
            _row_integer(item.offset, start + window, start + window, "cache offset")
            continue
        if len(entry) != 9 or entry[0] != "gdn" or not callable(getattr(item, "advance", None)):
            raise ValueError("row_forward: recurrent cache needs one complete GDN record")
        q, k, v, g, beta, state0, extra, n_keep = entry[1:]
        if type(extra) is not tuple or len(extra) != 6 or type(extra[-1]) is not bool:
            raise ValueError("row_forward: recurrent commit metadata is incomplete")
        qshape = _array_shape(q, 4, "queries", max_elements=1 << 32)
        vshape = _array_shape(v, 4, "values", max_elements=1 << 32)
        if (qshape != _array_shape(k, 4, "keys", max_elements=1 << 32) or qshape[:2] != (1, window) or qshape[3] % 32
                or vshape[:2] != (1, window) or vshape[2] % qshape[2]
                or _array_shape(g, 3, "gates", max_elements=1 << 32) != vshape[:3]
                or _array_shape(beta, 3, "beta", max_elements=1 << 32) != vshape[:3]
                or (window - 1) * max(qshape[2], vshape[2]) > limit):
            raise ValueError("row_forward: recurrent record head/window geometry differs")
        expected_state = (1, vshape[2], vshape[3], qshape[3])
        if (_array_shape(state0, 4, "initial state", max_elements=1 << 32) != expected_state
                or _array_shape(extra[3], 4, "final state", max_elements=1 << 32) != expected_state):
            raise ValueError("row_forward: recurrent state geometry differs")
        n_keep = _row_integer(n_keep, 0, limit, "convolution history")
        conv_dim = _row_integer(extra[2], 1, limit, "convolution width")
        if _array_shape(extra[4], 3, "convolution tails", zero_axes=(1,), max_elements=(1 << 64) - 1) != (window, n_keep, conv_dim):
            raise ValueError("row_forward: recurrent convolution tails differ")
        for previous, expected in ((item[0], (1, n_keep, conv_dim)), (item[1], expected_state)):
            if previous is not None and _array_shape(previous, len(expected), "recurrent cache",
                                                    zero_axes=(1,) if len(expected) == 3 else (),
                                                    max_elements=(1 << 64) - 1 if len(expected) == 3 else 1 << 32) != expected:
                raise ValueError("row_forward: recurrent cache geometry differs")
    return caches, entries, selected, window, start


def commit(cache: list[Any], record: Record, path: Sequence[int], window: int, start: int) -> None:
    """Keep a validated root path using the original cache/replay arithmetic.

    Borrow record tensors and caches unchanged through this synchronous call.
    Failed mutation poisons the record; discard the affected cache rather than
    retry. Lazy evaluation failure after return also requires cache retirement.
    """

    cache, entries, path, window, start = _commit_inputs(cache, record, path, window, start)
    try:
        _commit_validated(cache, entries, path, window, start)
    except BaseException:
        record.failed = True
        raise


def _commit_validated(cache: Sequence[Any], record: Sequence[Any], path: Sequence[int], window: int,
                      start: int) -> None:
    """Original per-layer mutation/replay, after complete metadata admission."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    keep = len(path)
    in_place = list(path) == list(range(keep))
    whole = keep == window and in_place
    last = int(path[-1])
    rows = count = None
    j = 0
    for item in cache:
        kind, *entry = record[j]
        j += 1
        if hasattr(item, "keys") and hasattr(item, "values"):
            if kind != "kv":
                raise RuntimeError(f"record entry {j - 1} is {kind!r}, the cache has an attention layer")
            if not in_place:
                # from the window's own rows, not the cache buffer: the slice update then writes in place
                taken = mx.array(list(path), dtype=mx.int32)
                item.keys[..., start:start + keep, :] = mx.take(entry[0], taken, axis=2)
                item.values[..., start:start + keep, :] = mx.take(entry[1], taken, axis=2)
            item.trim(window - keep)
            continue
        if kind != "gdn":
            raise RuntimeError(f"record entry {j - 1} is {kind!r}, the cache has a recurrent layer")
        q, k, v, g, beta, state0, extra, _ = entry
        _, _, _, state_out, conv_tails, chain = extra
        if whole and chain:
            item[1] = state_out
        else:
            if rows is None:
                rows, count = ints(path), mx.array([keep], dtype=mx.int32)
            item[1] = lane_tree.replay_path(q, k, v, g, beta, state0, rows, count)
        item[0] = conv_tails[last:last + 1]
        item.advance(keep)
    if j != len(record):
        raise RuntimeError(f"recorded {len(record)} layers, cache has {j}")


def keep_rows(cache: list[Any], record: Record, keep: int) -> None:
    """Keep a chain window's prefix with caches and recurrent state exactly as after row keep - 1."""

    commit(cache, record, list(range(int(keep))), record.width, record.start)


def _token_ids(windows: Sequence[Any]) -> mx.array:
    """[1, R] uint32 ids of every stream's window, in order (lists, or lazy GPU arrays, which are not read)."""

    if len(windows) == 1 and isinstance(windows[0], mx.array):
        return windows[0].reshape(1, -1).astype(mx.uint32)
    if not any(isinstance(w, mx.array) for w in windows):
        return mx.array([[int(t) for w in windows for t in w]], dtype=mx.uint32)
    parts = [w.reshape(-1).astype(mx.uint32) if isinstance(w, mx.array) else mx.array([int(t) for t in w],
                                                                                        dtype=mx.uint32)
             for w in windows]
    return mx.concatenate(parts).reshape(1, -1)


def _gate_up(mlp: Any, x: mx.array) -> mx.array:
    stack = stack_of(mlp, "gu")
    return project_stack(stack, x) if stack is not None else mx.concatenate(
        [project(mlp.gate_proj, x), project(mlp.up_proj, x)], axis=-1)


# Qwen3.6 MoE rows: "batched" (one gather_qmm for all rows) or "rows" (each alone); one-row steps take the same path
MOE_ROWS = os.environ.get("TF_MOE_ROWS", "batched")


def _per_row(fn: Callable[[mx.array], mx.array], x: mx.array) -> mx.array:
    W = int(x.shape[1])
    return fn(x) if W == 1 else mx.concatenate([fn(x[:, r:r + 1]) for r in range(W)], axis=1)


def moe(mlp: Any, x: mx.array) -> mx.array:
    """The MoE block's output for the normed rows ``x`` (1, W, K)."""

    if MOE_ROWS != "batched":
        return _per_row(mlp, x)
    gates = mx.softmax(_per_row(mlp.gate, x), axis=-1, precise=True)
    k = mlp.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if mlp.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    sw = mlp.switch_mlp
    xe = mx.expand_dims(x, (-2, -3))
    act = sw.activation(sw.up_proj(xe, inds), sw.gate_proj(xe, inds))
    y = (sw.down_proj(act, inds).squeeze(-2) * scores[..., None]).sum(axis=-2)
    shared = mlp.shared_expert
    gate = mx.sigmoid(_per_row(mlp.shared_expert_gate, x))
    return y + gate * project(shared.down_proj, mlp_act(_gate_up(shared, x)))


@projection_operation.operation()
def _rows_forward(core: Any, windows: Sequence[Any], parents: Sequence[Sequence[int]], caches: Sequence[list[Any]],
                  starts: Sequence[int], *, pipeline_layers: int = 4, first_alone: bool = True
                  ) -> tuple[mx.array, _Rows]:
    """Return final-normed rows [1, R, D] and each stream's commit record.

    Borrow model parameters, stream lists and caches unchanged until this synchronous
    operation returns. Host metadata uses ordered signed32 parent references and
    nonnegative signed32 absolute positions. Caches cover every model layer exactly.
    Lazy device token windows retain their caller-proved vocabulary range; this path
    does not read them back. Attention caches receive rows during the forward.
    """

    count = len(parents)
    if not 1 <= count <= row_matmul.BACKEND.max_rows:
        raise ValueError("row_forward: nonempty stream count must fit the selected backend")
    if len(windows) != count or len(caches) != count or len(starts) != count:
        raise ValueError(f"row_forward: {len(windows)} windows, {count} parent lists, {len(caches)} caches, "
                         f"{len(starts)} starts")
    windows, parents, caches, starts = (_stream_inputs(values, count) for values in
                                      (windows, parents, caches, starts))
    widths = [len(p) for p in parents]
    for window, width in zip(windows, widths):
        if (int(window.size) if isinstance(window, mx.array) else len(window)) != width:
            raise ValueError("row_forward: a window's tokens and parents differ in length")
    if sum(widths) > row_matmul.BACKEND.max_rows:
        raise ValueError(f"row_forward: {sum(widths)} rows, the {row_matmul.BACKEND.name} matmul takes up to {row_matmul.BACKEND.max_rows}")
    layers = list(core.layers)
    if any(len(cache) != len(layers) for cache in caches):
        raise ValueError("row_forward: each stream cache must cover every model layer exactly")
    rows = _Rows(parents, starts)
    for record, cache in zip(rows.records, caches):
        record.borrowed_cache = tuple(cache)
    if not ROW_ATTENTION and not all(rows.chains):
        raise NotImplementedError("row_forward: draft trees need row_attention")
    caches = _forward_cache_inputs(layers, caches, rows)
    windows = _forward_windows(core, windows, widths)
    hidden = core.embed_tokens(_token_ids(windows))
    pending: mx.array | None = None                   # the last output projection's rows, added in the next norm
    tapped: Any = None
    for index, (layer, *items) in enumerate(zip(layers, *caches)):
        inner = getattr(layer, "_layer", layer)
        hidden, x = add_norm(hidden, pending, inner.input_layernorm.weight, inner.input_layernorm.eps)
        if tapped is not None:
            tapped[0][tapped[1]] = hidden
        if getattr(inner, "is_linear", False):
            pending = project(inner.linear_attn.out_proj, _gdn(inner.linear_attn, x, items, rows))
        else:
            pending = project(inner.self_attn.o_proj, _attention(inner.self_attn, x, items, rows))
        norm = inner.post_attention_layernorm
        hidden, x = add_norm(hidden, pending, norm.weight, norm.eps)
        mlp = inner.mlp
        if hasattr(mlp, "switch_mlp"):
            pending = moe(mlp, x)
        else:
            pending = project(mlp.down_proj, mlp_act(_gate_up(mlp, x)))
        storage = getattr(layer, "_storage", None)
        tapped = (storage, layer._idx) if storage is not None else None
        if pipeline_layers and ((index + 1) % pipeline_layers == 0 or (index == 0 and first_alone)) \
                and index + 1 < len(layers):
            mx.async_eval(hidden, pending)
    hidden, x = add_norm(hidden, pending, core.norm.weight, core.norm.eps)
    if tapped is not None:
        tapped[0][tapped[1]] = hidden
    return x, rows


def forward(core: Any, head: Any, tokens: Sequence[int], parents: Sequence[int], cache: list[Any], start: int, *,
            pipeline_layers: int = 4, last_only: bool = False, first_alone: bool = True) -> tuple[mx.array, Record]:
    """Return logits [1, W, V] and a commit record; attention caches take rows immediately, while recurrent states wait for commit."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    x, rows = _rows_forward(core, [tokens], [parents], [cache], [start], pipeline_layers=pipeline_layers,
                            first_alone=first_alone)
    sink = lane_tree.HIDDEN_SINK          # a proposer that drafts from the rows' post-norm hidden states
    if sink is not None:
        sink.append(x)
        if len(sink) > 1024:
            del sink[0]
    return logits(head, x[:, -1:] if last_only else x), rows.records[0]


def multi_forward(core: Any, head: Any, windows: Sequence[Any], parents: Sequence[Sequence[int]],
                  caches: Sequence[list[Any]], starts: Sequence[int], *, pipeline_layers: int = 4,
                  first_alone: bool = True) -> tuple[mx.array, list[Record], list[int]]:
    """Return logits [1, R, V], commit records and row offsets for grouped streams, preserving each stream's standalone bits."""

    x, rows = _rows_forward(core, windows, parents, caches, starts, pipeline_layers=pipeline_layers,
                            first_alone=first_alone)
    return logits(head, x), rows.records, rows.offsets


def hidden_rows(core: Any, windows: Sequence[Any], caches: Sequence[list[Any]], *,
                starts: Sequence[int] | None = None, parents: Sequence[Sequence[int]] | None = None,
                pipeline_layers: int = 4, first_alone: bool = True) -> tuple[mx.array, list[Record]]:
    """Return final-normed chain rows [1, R, D] and keep_rows records; starts default to each stream's attention cache length."""

    if starts is None or parents is None:
        windows, caches, widths = _implicit_stream_inputs(core, windows, caches)
    if starts is None:
        starts = [next((c.offset for c in cache if hasattr(c, "keys")), 0) for cache in caches]
    if parents is None:
        parents = [_CHAINS.get(n) or tuple(range(-1, n - 1))
                   for n in widths]
    x, rows = _rows_forward(core, windows, parents, caches, starts, pipeline_layers=pipeline_layers,
                            first_alone=first_alone)
    return x, rows.records


def check_streams(core: Any, head: Any, make_cache: Callable[[], list[Any]], copy: Callable[[list[Any]], list[Any]],
                  mixes: Sequence[Sequence[int]] = ((1, 1), (1, 4), (8, 8), (3, 1, 8, 5), (16, 2)),
                  prefixes: Sequence[int] = (32, 21, 40, 9)) -> tuple[bool, list[str]]:
    """Return equality and failures for batched versus standalone logits and partial-window cache commits across prompt lengths and window widths."""

    def arrays(cache: list[Any]) -> list[mx.array]:
        return [a for item in cache for a in cache_contents(item)]

    vocab = int(core.embed_tokens["weight"].shape[0])   # MLX's gather reads past the table for larger ids, unchecked
    bases = []
    for s, length in enumerate(prefixes):
        cache = make_cache()
        prompt = [(1000 + 37 * s + 11 * i) % vocab for i in range(length)]
        step = min(WINDOW_ROWS, row_matmul.BACKEND.max_rows)
        for begin in range(0, length, step):
            chunk = prompt[begin:begin + step]
            _, record = forward(core, head, chunk, _CHAINS[len(chunk)], cache, begin)
            commit(cache, record, list(range(len(chunk))), len(chunk), begin)
        mx.eval(arrays(cache))
        bases.append((cache, length))
    failures: list[str] = []
    for mix in mixes:
        streams = [bases[s % len(bases)] for s in range(len(mix))]
        windows = [[(1500 + 13 * s + 7 * i) % vocab for i in range(w)] for s, w in enumerate(mix)]
        keeps = [max(1, (w + 1) // 2) for w in mix]
        alone = []
        for (cache, start), window, keep in zip(streams, windows, keeps):
            own = copy(cache)
            lg, record = forward(core, head, window, _CHAINS[len(window)], own, start)
            commit(own, record, list(range(keep)), len(window), start)
            mx.eval(lg, arrays(own))
            alone.append((lg, own))
        caches = [copy(cache) for cache, _ in streams]
        lg, records, offsets = multi_forward(core, head, windows, [_CHAINS[len(w)] for w in windows], caches,
                                             [start for _, start in streams])
        for cache, record, keep in zip(caches, records, keeps):
            keep_rows(cache, record, keep)
        mx.eval(lg, *[arrays(c) for c in caches])
        for s, ((ref, own), a, w) in enumerate(zip(alone, offsets, mix)):
            if not bool(mx.array_equal(lg[0, a:a + w], ref[0]).item()):
                failures.append(f"mix {tuple(mix)} stream {s}: logits differ")
            mine, theirs = arrays(caches[s]), arrays(own)
            if len(mine) != len(theirs) or not all(bool(mx.array_equal(x, y).item()) for x, y in zip(mine, theirs)):
                failures.append(f"mix {tuple(mix)} stream {s}: caches differ after keeping {keeps[s]} of {w} rows")
    return not failures, failures


__all__ = ["PROMPT_ROWS", "ROW_ATTENTION", "Record", "check_streams", "commit", "forward", "hidden_rows", "keep_rows",
           "multi_forward"]
